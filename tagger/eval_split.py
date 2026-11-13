"""Split my test footage into a dev half and a locked holdout half, by clip.

v1 was scored on all labeled frames, and then I studied its mistakes, so those frames can no
longer give v2 an honest score. From here on:
- dev: study errors and choose between v2 fixes, as often as needed
- holdout: never looked at while building v2; scored once, when v2 is final

Whole clips go to one half, because frames of one encounter are near-duplicates: a turtle in a
dev frame must not reappear in a holdout frame. Labeled clips are balanced per category, rarest
categories first, so rare animals (turtles: 4 clips) land in both halves. Unlabeled clips are
assigned too, so new labels have a fixed home. The split is saved to training/test_split.json
(by file path, which survives rebuilding the database) and never regenerated.

Run once with:  uv run python -m tagger.eval_split
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from tagger import CATEGORIES, store
from tagger.ingest import UPLOAD_TRIP
from tagger.labeling import SOURCE

SPLIT_PATH = Path("training/test_split.json")
HALVES = ("dev", "holdout")


def make_split(conn: sqlite3.Connection, seed: int = 0) -> dict[str, list[str]]:
    """{"dev": [clip paths], "holdout": [clip paths]} covering every library clip."""
    rng = random.Random(seed)
    clips = [
        r["path"]
        for r in conn.execute(
            "SELECT path FROM media WHERE trip IS NOT NULL AND trip != ? ORDER BY path",
            (UPLOAD_TRIP,),
        )
    ]
    labels: dict[str, Counter] = defaultdict(Counter)  # clip -> category -> labeled frames
    for r in conn.execute(
        """
        SELECT m.path, l.label FROM labels l
        JOIN images i ON i.id = l.image_id JOIN media m ON m.id = i.media_id
        WHERE l.source = ? AND l.label != ?
        """,
        (SOURCE, store.UNUSABLE),
    ):
        labels[r["path"]][r["label"]] += 1
    clips_per_category = Counter(c for counts in labels.values() for c in counts)

    def rarity(clip: str) -> tuple[int, float]:
        return min(clips_per_category[c] for c in labels[clip]), rng.random()

    halves = {h: [] for h in HALVES}
    clip_counts = {h: Counter() for h in HALVES}  # half -> category -> clips
    frame_totals = Counter()  # half -> labeled frames
    for clip in sorted((c for c in clips if c in labels), key=rarity):
        rarest = min(labels[clip], key=lambda c: clips_per_category[c])
        half = min(HALVES, key=lambda h: (clip_counts[h][rarest], frame_totals[h], rng.random()))
        halves[half].append(clip)
        clip_counts[half].update(labels[clip].keys())
        frame_totals[half] += sum(labels[clip].values())
    for clip in clips:
        if clip not in labels:
            halves[rng.choice(HALVES)].append(clip)
    return {h: sorted(paths) for h, paths in halves.items()}


def save_split(split: dict[str, list[str]], path: Path = SPLIT_PATH) -> None:
    """Write the split once. Refuses to overwrite: a regenerated split would leak."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; the split is frozen")
    path.write_text(json.dumps(split, indent=1) + "\n")


def load_split(path: Path = SPLIT_PATH) -> dict[str, set[str]] | None:
    if not path.exists():
        return None
    return {h: set(paths) for h, paths in json.loads(path.read_text()).items()}


def media_ids(conn: sqlite3.Connection, half: str, path: Path = SPLIT_PATH) -> frozenset[int]:
    """Database ids of the clips in one half. Clips added after the split belong to neither."""
    split = load_split(path)
    if split is None:
        raise FileNotFoundError(f"no test split at {path}")
    paths = split[half]
    return frozenset(
        r["id"] for r in conn.execute("SELECT id, path FROM media") if r["path"] in paths
    )


def summary(conn: sqlite3.Connection, split: dict[str, set[str]]) -> dict[str, Counter]:
    """Labeled frames per category in each half."""
    out = {h: Counter() for h in HALVES}
    for r in conn.execute(
        """
        SELECT m.path, l.label FROM labels l
        JOIN images i ON i.id = l.image_id JOIN media m ON m.id = i.media_id
        WHERE l.source = ? AND l.label != ?
        """,
        (SOURCE, store.UNUSABLE),
    ):
        for h in HALVES:
            if r["path"] in split[h]:
                out[h][r["label"]] += 1
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Freeze the dev/holdout split of the test set.")
    parser.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=SPLIT_PATH)
    args = parser.parse_args(argv)
    conn = store.connect(args.db)
    split = make_split(conn)
    save_split(split, args.out)
    counts = summary(conn, {h: set(p) for h, p in split.items()})
    for h in HALVES:
        frames = {c: counts[h][c] for c in CATEGORIES}
        print(f"{h}: {len(split[h])} clips, {sum(frames.values())} labeled frames {frames}")


if __name__ == "__main__":
    main()
