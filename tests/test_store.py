import csv
import sqlite3

import pytest

from tagger import CATEGORIES, store
from tagger.frames import DONE, MANIFEST, FrameRecord


@pytest.fixture
def conn():
    c = store.connect(":memory:")
    yield c
    c.close()


def scores_with_top(top: str, p: float = 0.7) -> dict[str, float]:
    rest = (1 - p) / (len(CATEGORIES) - 1)
    return {c: (p if c == top else rest) for c in CATEGORIES}


def make_image(conn, name="a.jpg") -> int:
    media_id = store.add_media(conn, "/mnt/d/GH01.MP4", "video", fingerprint="fp1", trip="azores")
    return store.add_image(conn, media_id, name, t_s=2.0, sharpness=100.0)


def test_connect_is_idempotent(tmp_path):
    db = tmp_path / "t.db"
    store.connect(db).close()
    conn = store.connect(db)  # schema creation must not fail the second time
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"media", "images", "labels", "predictions"} <= tables


def test_same_content_and_path_are_not_duplicated(conn):
    a = store.add_media(conn, "/x/GH01.MP4", "video", fingerprint="fp1")
    b = store.add_media(conn, "/y/copy_of_GH01.MP4", "video", fingerprint="fp1")
    assert a == b
    assert store.add_image(conn, a, "f.jpg") == store.add_image(conn, a, "f.jpg")


def test_prediction_keeps_all_scores_and_top_label(conn):
    image_id = make_image(conn)
    store.add_prediction(conn, image_id, "clip-zeroshot", scores_with_top("rays"))
    store.set_label(conn, image_id, "rays", "test_labeling")
    [row] = store.evaluation_rows(conn, "clip-zeroshot")
    assert row["predicted"] == "rays"
    assert row["confidence"] == pytest.approx(0.7)
    assert set(row["scores"]) == set(CATEGORIES)


def test_prediction_must_cover_every_category(conn):
    with pytest.raises(ValueError):
        store.add_prediction(conn, make_image(conn), "m", {"rays": 1.0})


def test_rejects_unknown_label_and_source(conn):
    image_id = make_image(conn)
    with pytest.raises(ValueError):
        store.set_label(conn, image_id, "dolphin", "test_labeling")
    with pytest.raises(ValueError):
        store.set_label(conn, image_id, "rays", "agent")


def test_foreign_keys_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_image(conn, media_id=999, file_path="orphan.jpg")


def test_correction_does_not_overwrite_test_label(conn):
    image_id = make_image(conn)
    store.add_prediction(conn, image_id, "m", scores_with_top("fish"))
    store.set_label(conn, image_id, "rays", "test_labeling")
    store.set_label(conn, image_id, "fish", "manual_correction")
    [row] = store.evaluation_rows(conn, "m")
    assert row["true_label"] == "rays"


def test_evaluation_skips_unusable_unlabeled_and_other_models(conn):
    media_id = store.add_media(conn, "/x/GH01.MP4", "video", fingerprint="fp1")
    labeled, unusable, unlabeled = (store.add_image(conn, media_id, f"{n}.jpg") for n in "abc")
    for image_id in (labeled, unusable, unlabeled):
        store.add_prediction(conn, image_id, "m1", scores_with_top("fish"))
    store.add_prediction(conn, labeled, "m2", scores_with_top("rays"))
    store.set_label(conn, labeled, "fish", "test_labeling")
    store.set_label(conn, unusable, "unusable", "test_labeling")

    rows = store.evaluation_rows(conn, "m1")
    assert [r["image_id"] for r in rows] == [labeled]
    assert store.evaluation_rows(conn, "m2")[0]["predicted"] == "rays"


def test_gallery_filters_orders_and_pages(conn):
    azores = store.add_media(conn, "/a/GH01.MP4", "video", fingerprint="a", trip="azores")
    mexico = store.add_media(conn, "/m/GH02.MP4", "video", fingerprint="m", trip="mexico")
    for n, (media_id, label, p) in enumerate(
        [(azores, "rays", 0.9), (azores, "rays", 0.5), (azores, "fish", 0.8), (mexico, "rays", 0.7)]
    ):
        image_id = store.add_image(conn, media_id, f"{n}.jpg")
        store.add_prediction(conn, image_id, "m", scores_with_top(label, p))

    rows, total = store.gallery(conn, "m", labels=["rays"], trips=["azores"])
    assert total == 2
    assert [r["confidence"] for r in rows] == pytest.approx([0.9, 0.5])

    rows, total = store.gallery(conn, "m", min_confidence=0.75)
    assert total == 2 and {r["label"] for r in rows} == {"rays", "fish"}

    rows, total = store.gallery(conn, "m", order="least confident", limit=1, offset=1)
    assert total == 4 and rows[0]["confidence"] == pytest.approx(0.7)
    assert store.trips(conn) == ["azores", "mexico"]

    rows, total = store.gallery(conn, "m", image_ids=[2, 4])
    assert total == 2 and {r["image_id"] for r in rows} == {2, 4}


def test_import_manifest_loads_kept_frames_once(conn, tmp_path):
    fields = list(FrameRecord.__dataclass_fields__)
    rows = [
        FrameRecord(
            "/d/GH01.MP4", "azores", "GH01", "video", 0.0, 90.0, -1, "kept", "azores/a.jpg"
        ),
        FrameRecord("/d/GH01.MP4", "azores", "GH01", "video", 2.0, 5.0, 0.2, "blurry", ""),
        FrameRecord(
            "/d/GOPR1.JPG", "azores", "GOPR1", "photo", 0.0, 80.0, -1, "kept", "azores/p.jpg"
        ),
    ]
    with open(tmp_path / MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(r.__dict__ for r in rows)
    (tmp_path / DONE).write_text("fpv\t/d/GH01.MP4\nfpp\t/d/GOPR1.JPG\n")

    assert store.import_manifest(conn, tmp_path) == 2
    assert store.import_manifest(conn, tmp_path) == 0  # re-run adds nothing
    photo = conn.execute("SELECT t_s FROM images WHERE file_path LIKE '%p.jpg'").fetchone()
    assert photo["t_s"] is None
    assert conn.execute("SELECT COUNT(*) FROM media").fetchone()[0] == 2


def test_manual_correction_overrides_the_shown_label_but_not_test_labels():
    conn = store.connect(":memory:")
    media = store.add_media(conn, "/d/a.MP4", "video", "fp", "azores")
    wrong = store.add_image(conn, media, "/f/1.jpg", t_s=0.0)
    right = store.add_image(conn, media, "/f/2.jpg", t_s=30.0)
    for image_id in (wrong, right):
        store.add_prediction(
            conn, image_id, "m", {c: (0.9 if c == "scenery" else 0.02) for c in CATEGORIES}
        )
    store.set_label(conn, wrong, "fish", "test_labeling", "random")  # my test label: untouched
    store.set_label(conn, wrong, "turtles", store.CORRECTION)

    rows, _ = store.gallery(conn, "m", order="as filmed")
    first = rows[0]
    assert (first["label"], first["model_label"], first["corrected"]) == (
        "turtles",
        "scenery",
        True,
    )
    assert rows[1]["corrected"] is False
    assert [r["image_id"] for r in store.gallery(conn, "m", labels=["turtles"])[0]] == [wrong]
    assert store.correction_for(conn, wrong) == "turtles"
    assert store.evaluation_rows(conn, "m")[0]["true_label"] == "fish"

    store.delete_label(conn, wrong, store.CORRECTION)
    assert store.gallery(conn, "m", labels=["turtles"])[1] == 0


def test_a_frame_can_be_corrected_to_unusable():
    conn = store.connect(":memory:")
    media = store.add_media(conn, "/d/a.MP4", "video", "fp", "mexico")
    image = store.add_image(conn, media, "/f/1.jpg", t_s=34.0)
    store.add_prediction(conn, image, "m", {c: (0.9 if c == "diver" else 0.02) for c in CATEGORIES})
    store.set_label(conn, image, store.UNUSABLE, store.CORRECTION)
    [row], total = store.gallery(conn, "m", labels=[store.UNUSABLE])
    assert total == 1 and (row["label"], row["model_label"]) == ("unusable", "diver")


def test_only_my_unusable_test_labels_show_in_the_gallery():
    conn = store.connect(":memory:")
    media = store.add_media(conn, "/d/a.MP4", "video", "fp", "mexico")
    ids = [store.add_image(conn, media, f"/f/{i}.jpg", t_s=i * 30.0) for i in range(3)]
    for image_id in ids:
        store.add_prediction(
            conn, image_id, "m", {c: (0.9 if c == "diver" else 0.02) for c in CATEGORIES}
        )
    store.set_label(conn, ids[0], store.UNUSABLE, "test_labeling", "random")
    store.set_label(conn, ids[1], "fish", "test_labeling", "random")  # stays the model's guess
    store.set_label(conn, ids[2], store.UNUSABLE, "test_labeling", "random")
    store.set_label(conn, ids[2], "turtles", store.CORRECTION)  # my correction wins

    rows = {r["image_id"]: r for r in store.gallery(conn, "m", order="as filmed")[0]}
    assert (rows[ids[0]]["label"], rows[ids[0]]["labeled_unusable"]) == ("unusable", True)
    assert (rows[ids[1]]["label"], rows[ids[1]]["labeled_unusable"]) == ("diver", False)
    assert (rows[ids[2]]["label"], rows[ids[2]]["corrected"]) == ("turtles", True)
    assert rows[ids[2]]["labeled_unusable"] is False
    assert store.gallery(conn, "m", labels=[store.UNUSABLE])[1] == 1
