"""Frame sets for developing and judging color correction (docs/color_plan.md, Stage 4).

- A: tune the Grad-CAM mask and the correction defaults by eye
- B: score Grad-CAM's hit rate once, by hand (and Haiku, to check Haiku against me)
- C: the blind A/B test of subject-aware vs. whole-frame correction
- pool: everything else, for Haiku's larger hit-rate count

Sets are split by clip, not by frame: neighboring frames of one clip look nearly the same, so a
frame-level split would let tuning on A leak into scoring on B. Rare subjects sit in a handful of
clips (confident turtles appear in about ten), so clips are dealt out per stratum: each clip is
grouped by the rarest subject it confidently contains, and each group is dealt in turn to A, B,
C, pool, pool. Strata come from the model's tags only, never from labels.

Only unlabeled library frames are used, so nothing here touches the test set: no test frame is
shown with a heatmap or correction for judging. Draws are random but fixed (seeded), so a set
is the same every time.
"""

from __future__ import annotations

import hashlib
import random

from tagger import CATEGORIES
from tagger.labeling import SOURCE as TEST_LABEL_SOURCE

SETS = ("A", "B", "C", "pool")
SUBJECTS = tuple(c for c in CATEGORIES if c != "scenery")  # where subject mode can apply
UPLOADS = "uploads"


DEAL = ("A", "B", "C", "pool", "pool")
RARITY = ("turtles", "invertebrates", "rays", "fish", "diver")  # rarest first


def _hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8])


def assign_sets(conn, model: str, min_confidence: float = 0.5) -> dict[str, str]:
    """{media path: set} for every library clip or photo, dealt out per stratum (see above)."""
    rows = conn.execute(
        """
        SELECT m.path, p.label, MAX(p.confidence >= ?) AS confident
        FROM media m JOIN images i ON i.media_id = m.id
        JOIN predictions p ON p.image_id = i.id AND p.model = ?
        WHERE m.trip != ?
        GROUP BY m.path, p.label
        """,
        (min_confidence, model, UPLOADS),
    ).fetchall()
    present: dict[str, set[str]] = {}
    for r in rows:
        present.setdefault(r["path"], set())
        if r["confident"]:
            present[r["path"]].add(r["label"])
    strata: dict[str, list[str]] = {}
    for path, labels in present.items():
        stratum = next((c for c in RARITY if c in labels), "none")
        strata.setdefault(stratum, []).append(path)
    out = {}
    for paths in strata.values():
        for k, path in enumerate(sorted(paths, key=_hash)):
            out[path] = DEAL[k % len(DEAL)]
    return out


def candidates(conn, model: str, which: str) -> list[dict]:
    """Unlabeled library frames in set `which`, with the model's tag and confidence. Frames I
    corrected use my tag at confidence 1.0, like the gallery."""
    rows = conn.execute(
        """
        SELECT i.id AS image_id, i.file_path, i.t_s, m.path AS source, m.trip,
               COALESCE(c.label, p.label) AS label,
               CASE WHEN c.label IS NULL THEN p.confidence ELSE 1.0 END AS confidence
        FROM images i
        JOIN media m ON m.id = i.media_id
        JOIN predictions p ON p.image_id = i.id AND p.model = ?
        LEFT JOIN labels c ON c.image_id = i.id AND c.source = 'manual_correction'
        WHERE m.trip != ?
          AND i.id NOT IN (SELECT image_id FROM labels WHERE source = ?)
        ORDER BY i.id
        """,
        (model, UPLOADS, TEST_LABEL_SOURCE),
    ).fetchall()
    sets = assign_sets(conn, model)
    return [dict(r) for r in rows if sets.get(r["source"]) == which]


def draw(
    rows: list[dict],
    per_category: int,
    categories: tuple[str, ...] = SUBJECTS,
    min_confidence: float = 0.0,
    seed: int = 0,
) -> list[dict]:
    """A fixed random sample per category, spread over trips and clips: frames are taken in
    turn from each trip, at most one per clip until every clip has given one."""
    rng = random.Random(seed)
    picked = []
    for category in categories:
        pool = [r for r in rows if r["label"] == category and r["confidence"] >= min_confidence]
        rng.shuffle(pool)
        by_trip: dict[str, list[dict]] = {}
        for r in pool:
            by_trip.setdefault(r["trip"], []).append(r)
        chosen, used_clips = [], set()
        for one_per_clip in (True, False):
            queues = [list(q) for q in by_trip.values()]
            while len(chosen) < per_category and any(queues):
                for q in queues:
                    while q and len(chosen) < per_category:
                        r = q.pop(0)
                        if r in chosen or (one_per_clip and r["source"] in used_clips):
                            continue
                        chosen.append(r)
                        used_clips.add(r["source"])
                        break
        picked.extend(chosen)
    return picked
