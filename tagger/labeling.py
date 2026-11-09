"""Choose which frames to hand-label for the test set, without biasing it toward CLIP.

The trap: picking frames from CLIP's own "turtle" predictions would only test CLIP on turtles
it already recognizes (selection bias). So frames come from a mix of sources, and each label
records which one picked it:

- "random": any unlabeled library frame. The unbiased core of the test set.
- "clip_top3:<category>": a frame where <category> is anywhere in CLIP's top 3, used to find
  rare subjects. Wider than CLIP's top pick, so frames CLIP got wrong can still get in.
- "browse": frames I pick myself by stepping through a clip I remember.

The page targets the category furthest below quota (based on my labels, not CLIP's guesses),
mixes in random frames so I can't tell which pool a frame came from, and never shows CLIP's
guess. Frames near an already-labeled frame in the same clip are skipped so one encounter
doesn't fill a category.
"""

from __future__ import annotations

import random
import sqlite3
from collections import Counter
from dataclasses import dataclass

from tagger import CATEGORIES
from tagger.baseline import MODEL_ID
from tagger.ingest import UPLOAD_TRIP
from tagger.store import UNUSABLE

QUOTA = 40
RANDOM_SHARE = 0.30
SPACING_S = 10.0
GIVE_UP_AFTER = 80  # CLIP-candidate frames shown for a category before it stops being targeted
SOURCE = "test_labeling"


@dataclass(frozen=True)
class Pick:
    image_id: int
    file_path: str
    selection: str


def label_counts(conn: sqlite3.Connection) -> Counter:
    """My test labels per category (and 'unusable')."""
    rows = conn.execute(
        "SELECT label, COUNT(*) n FROM labels WHERE source = ? GROUP BY label", (SOURCE,)
    )
    return Counter({r["label"]: r["n"] for r in rows})


def candidate_attempts(conn: sqlite3.Connection) -> Counter:
    """How many frames from each category's CLIP top-3 pool I've labeled (whatever they were)."""
    rows = conn.execute(
        """
        SELECT substr(selection, length('clip_top3:') + 1) AS category, COUNT(*) n
        FROM labels WHERE source = ? AND selection LIKE 'clip_top3:%' GROUP BY category
        """,
        (SOURCE,),
    )
    return Counter({r["category"]: r["n"] for r in rows})


def category_status(
    conn: sqlite3.Connection, quota: int = QUOTA, give_up_after: int = GIVE_UP_AFTER
) -> dict[str, str]:
    """'done' (quota met), 'gave up' (pool kept turning up other things), or 'active'."""
    counts, attempts = label_counts(conn), candidate_attempts(conn)
    status = {}
    for c in CATEGORIES:
        if counts[c] >= quota:
            status[c] = "done"
        elif attempts[c] >= give_up_after:
            status[c] = "gave up"
        else:
            status[c] = "active"
    return status


def _top3_condition(category: str) -> str:
    """SQL that is true when `category` ranks in the top 3 of a prediction's scores.

    Rank = how many categories beat it. Ties go to the category listed first in CATEGORIES,
    so "top 3" is always exactly 3 categories.
    """
    score = f"json_extract(p.scores, '$.{category}')"
    beats = []
    for i, c in enumerate(CATEGORIES):
        if c == category:
            continue
        other = f"json_extract(p.scores, '$.{c}')"
        tie_wins = i < CATEGORIES.index(category)
        beats.append(f"({other} {'>=' if tie_wins else '>'} {score})")
    return f"({' + '.join(beats)}) < 3"


def _eligible_frames(category: str | None, exclude_ids: tuple[int, ...]) -> tuple[str, list]:
    """FROM/WHERE for unlabeled library frames, spaced away from labeled ones."""
    sql = f"""
        FROM images i
        JOIN media m ON m.id = i.media_id
        JOIN predictions p ON p.image_id = i.id AND p.model = ?
        WHERE m.trip IS NOT NULL AND m.trip != ?
          AND NOT EXISTS (
              SELECT 1 FROM labels l JOIN images j ON j.id = l.image_id
              WHERE l.source = ? AND j.media_id = i.media_id
                AND (j.id = i.id
                     OR (l.label != ? AND i.t_s IS NOT NULL AND j.t_s IS NOT NULL
                         AND ABS(j.t_s - i.t_s) <= ?)))
          {f"AND i.id NOT IN ({', '.join('?' * len(exclude_ids))})" if exclude_ids else ""}
          {f"AND {_top3_condition(category)}" if category else ""}
    """
    return sql, [MODEL_ID, UPLOAD_TRIP, SOURCE, UNUSABLE, SPACING_S, *exclude_ids]


def _random_from(
    conn: sqlite3.Connection, rng: random.Random, category: str | None, exclude_ids
) -> tuple[int, str] | None:
    where, params = _eligible_frames(category, tuple(exclude_ids))
    n = conn.execute(f"SELECT COUNT(*) {where}", params).fetchone()[0]
    if n == 0:
        return None
    # Pick by offset with our own RNG (instead of ORDER BY RANDOM()) so tests are repeatable.
    row = conn.execute(
        f"SELECT i.id, i.file_path {where} ORDER BY i.id LIMIT 1 OFFSET ?",
        [*params, rng.randrange(n)],
    ).fetchone()
    return row["id"], row["file_path"]


def next_frame(
    conn: sqlite3.Connection,
    rng: random.Random | None = None,
    exclude_ids: set[int] | frozenset[int] = frozenset(),
    random_share: float = RANDOM_SHARE,
    quota: int = QUOTA,
    give_up_after: int = GIVE_UP_AFTER,
) -> Pick | None:
    """The next frame to label, or None when nothing eligible is left.

    `exclude_ids` holds frames skipped this session, so they don't come straight back.
    """
    rng = rng or random.Random()
    status = category_status(conn, quota, give_up_after)
    counts = label_counts(conn)
    # Furthest below quota first.
    targets = sorted((c for c in CATEGORIES if status[c] == "active"), key=lambda c: counts[c])

    if targets and rng.random() >= random_share:
        for category in targets:
            found = _random_from(conn, rng, category, exclude_ids)
            if found:
                return Pick(*found, selection=f"clip_top3:{category}")
    found = _random_from(conn, rng, None, exclude_ids)
    return Pick(*found, selection="random") if found else None


def library_clips(conn: sqlite3.Connection, trip: str) -> list[dict]:
    """Videos and photos in one library trip, with frame and label counts, for browse mode."""
    rows = conn.execute(
        """
        SELECT m.id, m.path, COUNT(i.id) AS frames, COUNT(l.image_id) AS labeled
        FROM media m
        JOIN images i ON i.media_id = m.id
        LEFT JOIN labels l ON l.image_id = i.id AND l.source = ?
        WHERE m.trip = ?
        GROUP BY m.id ORDER BY m.path
        """,
        (SOURCE, trip),
    )
    return [dict(r) for r in rows]  # plain dicts: UI widgets need copyable values


def clip_frames(conn: sqlite3.Connection, media_id: int) -> list[dict]:
    """One clip's frames in filmed order, with my existing label (if any)."""
    rows = conn.execute(
        """
        SELECT i.id, i.file_path, i.t_s, l.label
        FROM images i LEFT JOIN labels l ON l.image_id = i.id AND l.source = ?
        WHERE i.media_id = ? ORDER BY i.t_s, i.id
        """,
        (SOURCE, media_id),
    )
    return [dict(r) for r in rows]
