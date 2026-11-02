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
