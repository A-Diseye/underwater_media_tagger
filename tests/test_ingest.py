import cv2
import pytest

from tagger import CATEGORIES, ingest, store
from tagger.baseline import MODEL_ID as CLIP
from tests.test_frames import textured, write_video


class FakeTagger:
    """Stands in for CLIP so these tests run in CI: everything is a ray."""

    def __init__(self):
        self.calls = 0

    def predict_paths(self, paths):
        self.calls += 1
        return [{c: (0.9 if c == "rays" else 0.1 / 6) for c in CATEGORIES} for _ in paths]


@pytest.fixture
def conn():
    c = store.connect(":memory:")
    yield c
    c.close()


def test_photo_is_stored_and_tagged(conn, tmp_path):
    photo = tmp_path / "IMG_1.png"
    cv2.imwrite(str(photo), cv2.GaussianBlur(textured(0), (51, 51), 20))  # blurry on purpose
    fake = FakeTagger()

    result = ingest.ingest_file(conn, photo, {CLIP: fake}, frames_dir=tmp_path / "frames")

    assert len(result.image_ids) == 1  # photos skip the blur filter
    rows, total = store.gallery(conn, CLIP)
    assert total == 1
    assert rows[0]["label"] == "rays" and rows[0]["trip"] == ingest.UPLOAD_TRIP


def test_video_frames_are_extracted_and_tagged(conn, tmp_path):
    video = tmp_path / "dive.mp4"
    write_video(video, seconds=5)
    result = ingest.ingest_file(conn, video, {CLIP: FakeTagger()}, frames_dir=tmp_path / "frames")
    assert len(result.image_ids) == 3  # t = 0, 2, 4
    t = [r["t_s"] for r in conn.execute("SELECT t_s FROM images ORDER BY id")]
    assert [round(x) for x in t] == [0, 2, 4]


def test_same_file_twice_is_not_reprocessed(conn, tmp_path):
    photo = tmp_path / "IMG_1.jpg"
    cv2.imwrite(str(photo), textured(0))
    fake = FakeTagger()
    first = ingest.ingest_file(conn, photo, {CLIP: fake}, frames_dir=tmp_path / "frames")
    second = ingest.ingest_file(conn, photo, {CLIP: fake}, frames_dir=tmp_path / "frames")
    assert first.image_ids == second.image_ids
    assert first.new and not second.new
    assert fake.calls == 1


def test_rejects_unsupported_files(conn, tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("not media")
    with pytest.raises(ValueError):
        ingest.ingest_file(conn, doc, {CLIP: FakeTagger()}, frames_dir=tmp_path / "frames")


def test_save_upload_keeps_same_named_files_apart(tmp_path):
    a = ingest.save_upload("IMG_1.jpg", b"first", upload_dir=tmp_path)
    b = ingest.save_upload("IMG_1.jpg", b"second", upload_dir=tmp_path)
    assert a != b and a.read_bytes() == b"first" and b.read_bytes() == b"second"
    assert ingest.save_upload("IMG_1.jpg", b"first", upload_dir=tmp_path) == a


def upload_photo(conn, tmp_path, name, seed) -> ingest.IngestResult:
    path = tmp_path / name
    cv2.imwrite(str(path), textured(seed))
    return ingest.ingest_file(conn, path, {CLIP: FakeTagger()}, frames_dir=tmp_path / "frames")


def test_remove_upload_deletes_rows_and_files(conn, tmp_path):
    keep = upload_photo(conn, tmp_path, "keep.jpg", 0)
    gone = upload_photo(conn, tmp_path, "gone.jpg", 1)
    [frame] = [
        r["file_path"]
        for r in conn.execute("SELECT file_path FROM images WHERE media_id = ?", (gone.media_id,))
    ]

    ingest.remove_upload(conn, gone.media_id)

    assert not (tmp_path / "gone.jpg").exists() and not tmp_path.joinpath(frame).exists()
    assert (tmp_path / "keep.jpg").exists()
    assert [r["image_id"] for r in store.gallery(conn, CLIP)[0]] == keep.image_ids
    # Re-uploading the removed file processes it again instead of finding stale rows.
    assert upload_photo(conn, tmp_path, "gone.jpg", 1).image_ids


def test_library_media_cannot_be_removed(conn, tmp_path):
    media_id = store.add_media(conn, "/mnt/d/GH01.MP4", "video", fingerprint="fp", trip="azores")
    with pytest.raises(ValueError):
        ingest.remove_upload(conn, media_id)
    assert conn.execute("SELECT COUNT(*) FROM media").fetchone()[0] == 1


def test_clear_uploads_leaves_library_alone(conn, tmp_path):
    store.add_media(conn, "/mnt/d/GH01.MP4", "video", fingerprint="fp", trip="azores")
    upload_photo(conn, tmp_path, "a.jpg", 0)
    upload_photo(conn, tmp_path, "b.jpg", 1)
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "stray.jpg").write_bytes(b"left behind")
    assert ingest.clear_uploads(conn, upload_dir=tmp_path / "uploads") == 2
    assert [r["trip"] for r in conn.execute("SELECT trip FROM media")] == ["azores"]
    assert not any((tmp_path / "uploads").iterdir())


def test_duplicate_upload_under_new_name_leaves_no_extra_file(conn, tmp_path):
    uploads, frames_dir = tmp_path / "uploads", tmp_path / "frames"
    ok, data = cv2.imencode(".jpg", textured(0))
    kwargs = dict(upload_dir=uploads, frames_dir=frames_dir)
    first = ingest.ingest_upload(conn, "a.jpg", data.tobytes(), {CLIP: FakeTagger()}, **kwargs)
    second = ingest.ingest_upload(
        conn, "a-copy.jpg", data.tobytes(), {CLIP: FakeTagger()}, **kwargs
    )
    assert not second.new and second.image_ids == first.image_ids
    assert [p.name.split("_", 1)[1] for p in uploads.iterdir()] == ["a.jpg"]


def test_uploads_are_tagged_by_every_model(conn, tmp_path):
    class FishTagger:
        def predict_paths(self, paths):
            return [{c: (0.8 if c == "fish" else 0.04) for c in CATEGORIES} for _ in paths]

    photo = tmp_path / "IMG_0001.jpg"
    cv2.imwrite(str(photo), textured(0))
    taggers = {CLIP: FakeTagger(), "v2_both_s0": FishTagger()}
    [image_id] = ingest.ingest_file(conn, photo, taggers, frames_dir=tmp_path / "frames").image_ids
    answers = store.predictions_for(conn, image_id)
    assert answers[CLIP]["label"] == "rays" and answers["v2_both_s0"]["label"] == "fish"
    assert set(answers["v2_both_s0"]["scores"]) == set(CATEGORIES)
