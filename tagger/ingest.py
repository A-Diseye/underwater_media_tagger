"""Add a new photo or video to the library: extract frames, store them, and tag them.

This is the core of the app's upload flow. It has no UI code, so the Streamlit page (and later
a FastAPI endpoint or React frontend) can all call it the same way.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tagger import frames, store

UPLOAD_TRIP = "uploads"
UPLOAD_DIR = Path("data/uploads")


class Tagger(Protocol):
    """Anything that scores image files, e.g. ClipBaseline or the fine-tuned Classifier.
    Tests pass a fake."""

    def predict_paths(self, paths: list[str]) -> list[dict[str, float]]: ...


@dataclass
class IngestResult:
    media_id: int
    image_ids: list[int]  # empty if every frame of a video was dropped
    new: bool = True  # False if this content was already stored


def save_upload(name: str, data: bytes, upload_dir: Path = UPLOAD_DIR) -> Path:
    """Write uploaded bytes to disk. A content-hash prefix keeps same-named files apart."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{hashlib.blake2b(data, digest_size=4).hexdigest()}_{Path(name).name}"
    if not dest.exists():
        dest.write_bytes(data)
    return dest


def ingest_file(
    conn,
    path: Path,
    taggers: Mapping[str, Tagger],
    frames_dir: Path = Path("data/frames"),
    trip: str = UPLOAD_TRIP,
) -> IngestResult:
    """Store and tag one photo or video, with every model in `taggers` ({model id: tagger}).

    Tagging with more than one model (the fine-tuned model and CLIP) keeps both answers for
    every image, so the app can compare them.

    A file whose content is already stored is not processed again; its existing rows are
    returned. Videos go through the same blur and duplicate filters as the dive library.
    Photos skip the blur filter: someone who uploads a single photo wants it tagged.
    """
    fingerprint = frames.file_fingerprint(path)
    existing = conn.execute("SELECT id FROM media WHERE fingerprint = ?", (fingerprint,)).fetchone()
    if existing:
        rows = conn.execute(
            "SELECT id FROM images WHERE media_id = ? ORDER BY id", (existing["id"],)
        ).fetchall()
        return IngestResult(existing["id"], [r["id"] for r in rows], new=False)

    ext = path.suffix.lower()
    if ext in frames.VIDEO_EXTS:
        kind, blur_threshold = "video", frames.BLUR_THRESHOLD
    elif ext in frames.PHOTO_EXTS:
        kind, blur_threshold = "photo", 0.0
    else:
        raise ValueError(f"unsupported file type: {path.name}")

    records = frames.process_source(path, trip, frames_dir, blur_threshold=blur_threshold)
    kept = [r for r in records if r.decision == "kept"]
    with conn:
        media_id = store.add_media(conn, str(path), kind, fingerprint, trip)
        image_ids = [
            store.add_image(
                conn,
                media_id,
                str(frames_dir / r.frame_path),
                t_s=r.t_s if kind == "video" else None,
                sharpness=r.sharpness,
            )
            for r in kept
        ]

    if image_ids:
        paths = [str(frames_dir / r.frame_path) for r in kept]
        for model_id, tagger in taggers.items():
            scores = tagger.predict_paths(paths)
            with conn:
                for image_id, s in zip(image_ids, scores, strict=True):
                    store.add_prediction(conn, image_id, model_id, s)
    return IngestResult(media_id, image_ids)


def ingest_upload(
    conn,
    name: str,
    data: bytes,
    taggers: Mapping[str, Tagger],
    upload_dir: Path = UPLOAD_DIR,
    **kwargs,
) -> IngestResult:
    """Save an uploaded file and ingest it. Entry point for the upload UI and future API.

    If the same content was already stored (e.g. the same photo under another name), the
    new copy on disk is redundant and gets deleted.
    """
    path = save_upload(name, data, upload_dir)
    result = ingest_file(conn, path, taggers, **kwargs)
    if not result.new:
        stored = conn.execute("SELECT path FROM media WHERE id = ?", (result.media_id,)).fetchone()
        if Path(stored["path"]).resolve() != path.resolve():
            path.unlink(missing_ok=True)
    return result


def remove_upload(conn, media_id: int) -> None:
    """Delete an uploaded file: its database rows, extracted frames, and saved copy.

    Only uploads can be removed. Library media (my dive footage) raises ValueError, so a UI
    button can never delete library frames.
    """
    row = conn.execute("SELECT path, trip FROM media WHERE id = ?", (media_id,)).fetchone()
    if row is None:
        return
    if row["trip"] != UPLOAD_TRIP:
        raise ValueError(f"media {media_id} is not an upload (trip={row['trip']!r})")
    for file_path in [*store.delete_media(conn, media_id), row["path"]]:
        Path(file_path).unlink(missing_ok=True)


def clear_uploads(conn, upload_dir: Path = UPLOAD_DIR) -> int:
    """Remove every upload, plus any stray files left in the upload folder.

    Returns how many uploads were removed.
    """
    ids = [r["id"] for r in conn.execute("SELECT id FROM media WHERE trip = ?", (UPLOAD_TRIP,))]
    for media_id in ids:
        remove_upload(conn, media_id)
    if upload_dir.is_dir():
        for stray in upload_dir.iterdir():
            if stray.is_file():
                stray.unlink()
    return len(ids)
