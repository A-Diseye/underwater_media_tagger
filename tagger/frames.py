"""Turn dive videos and photos into a pool of sharp, non-repetitive frames.

Pipeline per video:
1. Decode keyframes only. GoPro writes a full keyframe every ~1s; skipping the in-between
   frames avoids ~97% of the decoding work.
2. Keep one keyframe every `every_s` seconds.
3. Drop clearly ruined frames: no tile of the image has crisp detail (Laplacian variance).
4. Drop near-duplicates: thumbnail nearly identical to the last kept frame (correlation).

Raw footage is only ever read. Kept frames are written as JPEGs under `out_dir`, and every
sampled frame (kept or dropped) is recorded in `manifest.csv` so thresholds can be revisited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import cv2
import numpy as np

VIDEO_EXTS = {".mp4"}
PHOTO_EXTS = {".jpg", ".jpeg"}

# Calibrated on 283 sampled frames from all four trips (Oct 2026):
# - Fogged-lens frames scored 10-23 on tile sharpness; the lowest real subject (a distant
#   mobula ray in open blue) scored 26. 15 drops only the clearly ruined frames; borderline
#   ones are left for the "unusable" button during labeling.
# - Frame pairs with thumbnail correlation >= 0.90 were true near-duplicates; a diver
#   leaving the shot scored 0.87.
EVERY_S = 2.0
BLUR_THRESHOLD = 15.0
DUP_SIMILARITY = 0.90
MAX_SIDE = 1920
SHARPNESS_WIDTH = 512
SHARPNESS_GRID = 4
THUMB_SIZE = (64, 36)

MANIFEST = "manifest.csv"
DONE = "done.txt"


@dataclass
class FrameRecord:
    source: str
    trip: str
    clip: str
    kind: str  # "video" or "photo"
    t_s: float
    sharpness: float
    similarity: float  # thumbnail correlation with last kept frame; -1 if none to compare
    decision: str  # "kept", "blurry", or "duplicate"
    frame_path: str


def is_gopro_file(path: Path) -> bool:
    """GoPro names start with G (GH, GX, GOPR, ...). Phone files (PXL_, dates) do not.

    GoPro's .LRV previews and .THM thumbnails are excluded by extension.
    """
    ext = path.suffix.lower()
    return path.name[:1].upper() == "G" and ext in VIDEO_EXTS | PHOTO_EXTS


def iter_media(folder: Path) -> Iterator[Path]:
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            path = Path(root) / name
            if is_gopro_file(path):
                yield path


def file_fingerprint(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    """Cheap content ID: size + hash of the first and last chunk.

    Catches byte-identical copies (e.g. the same clip exported twice) without reading whole
    multi-GB files over USB.
    """
    size = path.stat().st_size
    h = hashlib.blake2b(str(size).encode(), digest_size=16)
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(size - chunk, chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


def fit_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, max_side / max(width, height))
    # Even dimensions keep video scalers happy.
    return int(width * scale) // 2 * 2, int(height * scale) // 2 * 2


def sharpness(img_bgr: np.ndarray, grid: int = SHARPNESS_GRID) -> float:
    """Highest Laplacian variance over a grid of tiles.

    Laplacian variance measures crisp edges: high = sharp, low = blurry or featureless.
    Scoring the sharpest tile instead of the whole image keeps a small sharp subject (a ray
    in open blue water) from being mistaken for blur. The image is resized to a fixed width
    first so 1080p and 4K sources score on the same scale.
    """
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (SHARPNESS_WIDTH, round(h * SHARPNESS_WIDTH / w)))
    lap = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    th, tw = lap.shape[0] // grid, lap.shape[1] // grid
    return float(
        max(
            lap[i * th : (i + 1) * th, j * tw : (j + 1) * tw].var()
            for i in range(grid)
            for j in range(grid)
        )
    )


def thumbnail(img_bgr: np.ndarray) -> np.ndarray:
    """Tiny grayscale thumbnail, normalized to mean 0 / std 1 so lighting shifts don't matter."""
    small = cv2.resize(img_bgr, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return (gray - gray.mean()) / (gray.std() + 1e-6)


def similarity(thumb_a: np.ndarray, thumb_b: np.ndarray) -> float:
    """Correlation of two normalized thumbnails: 1.0 = same layout, ~0 = unrelated."""
    return float((thumb_a * thumb_b).mean())


def sample_video(path: Path, every_s: float = EVERY_S, max_side: int = MAX_SIDE):
    """Yield (timestamp_s, BGR image) for one keyframe every `every_s` seconds."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        stream.thread_type = "AUTO"
        size = fit_size(stream.codec_context.width, stream.codec_context.height, max_side)
        last_t = None
        for frame in container.decode(stream):
            t = frame.time
            if t is None:
                continue
            # Small tolerance: GoPro keyframes land at 0, 1.001, 2.002, ... seconds.
            if last_t is not None and t - last_t < every_s - 0.1:
                continue
            last_t = t
            img = frame.reformat(width=size[0], height=size[1], format="bgr24").to_ndarray()
            yield t, img


def select_frames(
    frames: Iterable[tuple[float, np.ndarray]],
    blur_threshold: float = BLUR_THRESHOLD,
    dup_similarity: float = DUP_SIMILARITY,
) -> Iterator[tuple[float, np.ndarray, float, float, str]]:
    """Decide keep/drop for a sequence of frames from one video.

    Yields (t, img, sharpness, similarity_to_last_kept, decision). Blurry frames are
    checked first and never become the comparison point for duplicates.
    """
    last_kept = None
    for t, img in frames:
        score = sharpness(img)
        thumb = thumbnail(img)
        sim = similarity(thumb, last_kept) if last_kept is not None else -1.0
        if score < blur_threshold:
            decision = "blurry"
        elif sim >= dup_similarity:
            decision = "duplicate"
        else:
            decision = "kept"
            last_kept = thumb
        yield t, img, score, sim, decision


def load_photo(path: Path, max_side: int = MAX_SIDE) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read image: {path}")
    h, w = img.shape[:2]
    size = fit_size(w, h, max_side)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA) if size != (w, h) else img


def _load_done(out_dir: Path) -> dict[str, str]:
    done_path = out_dir / DONE
    if not done_path.exists():
        return {}
    with open(done_path) as f:
        return dict(line.rstrip("\n").split("\t", 1) for line in f if line.strip())


def process_source(
    path: Path,
    trip: str,
    out_dir: Path,
    every_s: float = EVERY_S,
    blur_threshold: float = BLUR_THRESHOLD,
    dup_similarity: float = DUP_SIMILARITY,
    max_side: int = MAX_SIDE,
) -> list[FrameRecord]:
    """Sample one video or photo, write kept frames as JPEGs, and return all records."""
    kind = "video" if path.suffix.lower() in VIDEO_EXTS else "photo"
    if kind == "video":
        frames = sample_video(path, every_s, max_side)
    else:
        frames = [(0.0, load_photo(path, max_side))]

    trip_dir = out_dir / trip
    trip_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for t, img, score, sim, decision in select_frames(frames, blur_threshold, dup_similarity):
        frame_path = ""
        if decision == "kept":
            name = f"{path.stem}_{t:07.1f}s.jpg" if kind == "video" else f"{path.stem}.jpg"
            frame_path = str((trip_dir / name).relative_to(out_dir))
            cv2.imwrite(str(trip_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append(
            FrameRecord(
                source=str(path),
                trip=trip,
                clip=path.stem,
                kind=kind,
                t_s=round(t, 3),
                sharpness=round(score, 2),
                similarity=round(sim, 3),
                decision=decision,
                frame_path=frame_path,
            )
        )
    return records


def extract(sources: list[tuple[str, Path]], out_dir: Path, **params) -> None:
    """Process every GoPro file under each (trip, folder), skipping work already done.

    Resumable: a source's records are appended to the manifest only once it finishes, and its
    fingerprint goes into done.txt. Byte-identical copies of a processed file are skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_dir)
    manifest_path = out_dir / MANIFEST
    fields = list(FrameRecord.__dataclass_fields__)
    if not manifest_path.exists():
        with open(manifest_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    paths = [(trip, p) for trip, folder in sources for p in iter_media(folder)]
    for i, (trip, path) in enumerate(paths, 1):
        fp = file_fingerprint(path)
        if fp in done:
            print(f"[{i}/{len(paths)}] skip {path} (already processed as {done[fp]})", flush=True)
            continue
        records = process_source(path, trip, out_dir, **params)
        with open(manifest_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerows(asdict(r) for r in records)
        with open(out_dir / DONE, "a") as f:
            f.write(f"{fp}\t{path}\n")
        done[fp] = str(path)
        kept = sum(r.decision == "kept" for r in records)
        print(f"[{i}/{len(paths)}] {path.name}: kept {kept}/{len(records)}", flush=True)


def _parse_source(value: str) -> tuple[str, Path]:
    trip, sep, folder = value.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("expected TRIP=FOLDER")
    return trip, Path(folder)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=_parse_source,
        action="append",
        required=True,
        help="TRIP=FOLDER; repeat for each folder. Several folders may share a trip name.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/frames"))
    parser.add_argument("--every-s", type=float, default=EVERY_S)
    parser.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD)
    parser.add_argument("--dup-similarity", type=float, default=DUP_SIMILARITY)
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    args = parser.parse_args(argv)
    extract(
        args.source,
        args.out,
        every_s=args.every_s,
        blur_threshold=args.blur_threshold,
        dup_similarity=args.dup_similarity,
        max_side=args.max_side,
    )


if __name__ == "__main__":
    main()
