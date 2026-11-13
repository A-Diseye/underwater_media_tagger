"""SQLite storage for media, images, hand labels, and model predictions.

Tables:
- media: one row per source file (dive clip, GoPro photo, or app upload)
- images: one row per frame or photo the model sees
- labels: hand labels only; keyed by (image, source) so app corrections never overwrite
  ground-truth test labels
- predictions: one row per (image, model), with all category scores kept as JSON
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from tagger import CATEGORIES

DEFAULT_DB = Path("data/tagger.db")

UNUSABLE = "unusable"
LABEL_SOURCES = ("test_labeling", "manual_correction")
# A tag I fixed in the gallery. It overrides the model's label for display and search, but never
# touches test labels (separate source) and is never used for training.
CORRECTION = "manual_correction"
# How a test frame was chosen for labeling, so results can be split by it later:
# "random", "browse", or "clip_top3:<category>" (CLIP had <category> in its top 3).
SELECTIONS = ("random", "browse", *(f"clip_top3:{c}" for c in CATEGORIES))

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS media (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('video', 'photo')),
    trip        TEXT,
    fingerprint TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS images (
    id         INTEGER PRIMARY KEY,
    media_id   INTEGER NOT NULL REFERENCES media(id),
    file_path  TEXT NOT NULL UNIQUE,
    t_s        REAL,  -- seconds into the clip; NULL for photos
    sharpness  REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS labels (
    image_id   INTEGER NOT NULL REFERENCES images(id),
    source     TEXT NOT NULL CHECK (source IN {LABEL_SOURCES!r}),
    label      TEXT NOT NULL,
    selection  TEXT,  -- how the frame was chosen (see SELECTIONS); NULL for corrections
    labeled_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (image_id, source)
);

CREATE TABLE IF NOT EXISTS predictions (
    image_id   INTEGER NOT NULL REFERENCES images(id),
    model      TEXT NOT NULL,
    label      TEXT NOT NULL,
    confidence REAL NOT NULL,
    scores     TEXT NOT NULL,  -- JSON object: category -> probability, for all categories
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (image_id, model)
);

CREATE INDEX IF NOT EXISTS images_media ON images(media_id);
"""


def connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    """Open the database, creating tables if needed. Rows behave like dicts."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    # Databases created before labels.selection existed: add the column.
    if "selection" not in {r["name"] for r in conn.execute("PRAGMA table_info(labels)")}:
        conn.execute("ALTER TABLE labels ADD COLUMN selection TEXT")
    return conn


def add_media(
    conn: sqlite3.Connection, path: str, kind: str, fingerprint: str, trip: str | None = None
) -> int:
    """Insert a source file, or return the existing id if this content is already stored."""
    conn.execute(
        "INSERT OR IGNORE INTO media (path, kind, trip, fingerprint) VALUES (?, ?, ?, ?)",
        (path, kind, trip, fingerprint),
    )
    row = conn.execute("SELECT id FROM media WHERE fingerprint = ?", (fingerprint,)).fetchone()
    return row["id"]


def add_image(
    conn: sqlite3.Connection,
    media_id: int,
    file_path: str,
    t_s: float | None = None,
    sharpness: float | None = None,
) -> int:
    """Insert a frame or photo, or return the existing id for this file path."""
    conn.execute(
        "INSERT OR IGNORE INTO images (media_id, file_path, t_s, sharpness) VALUES (?, ?, ?, ?)",
        (media_id, file_path, t_s, sharpness),
    )
    row = conn.execute("SELECT id FROM images WHERE file_path = ?", (file_path,)).fetchone()
    return row["id"]


def set_label(
    conn: sqlite3.Connection,
    image_id: int,
    label: str,
    source: str,
    selection: str | None = None,
) -> None:
    """Record a hand label (one of the categories, or 'unusable'). Re-labeling replaces it."""
    if label not in (*CATEGORIES, UNUSABLE):
        raise ValueError(f"unknown label: {label!r}")
    if source not in LABEL_SOURCES:
        raise ValueError(f"unknown label source: {source!r}")
    if selection is not None and selection not in SELECTIONS:
        raise ValueError(f"unknown selection method: {selection!r}")
    conn.execute(
        """
        INSERT INTO labels (image_id, source, label, selection) VALUES (?, ?, ?, ?)
        ON CONFLICT (image_id, source) DO UPDATE SET
            label = excluded.label, selection = excluded.selection,
            labeled_at = CURRENT_TIMESTAMP
        """,
        (image_id, source, label, selection),
    )


def delete_label(conn: sqlite3.Connection, image_id: int, source: str) -> None:
    conn.execute("DELETE FROM labels WHERE image_id = ? AND source = ?", (image_id, source))


def add_prediction(
    conn: sqlite3.Connection, image_id: int, model: str, scores: dict[str, float]
) -> None:
    """Store a model's scores for every category; the top score becomes label + confidence."""
    if set(scores) != set(CATEGORIES):
        raise ValueError(f"scores must cover exactly {CATEGORIES}, got {sorted(scores)}")
    label = max(scores, key=scores.get)
    conn.execute(
        """
        INSERT INTO predictions (image_id, model, label, confidence, scores)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (image_id, model) DO UPDATE SET
            label = excluded.label, confidence = excluded.confidence,
            scores = excluded.scores, created_at = CURRENT_TIMESTAMP
        """,
        (image_id, model, label, float(scores[label]), json.dumps(scores)),
    )


def evaluation_rows(conn: sqlite3.Connection, model: str) -> list[sqlite3.Row]:
    """Test-set images with both a usable hand label and a prediction from `model`.

    Each row has image_id, file_path, trip, source (original file path), t_s, true_label,
    selection, predicted, confidence, and scores (dict).
    """
    rows = conn.execute(
        """
        SELECT i.id AS image_id, i.file_path, m.trip, m.path AS source, i.t_s,
               l.label AS true_label, l.selection,
               p.label AS predicted, p.confidence, p.scores
        FROM images i
        JOIN media m ON m.id = i.media_id
        JOIN labels l ON l.image_id = i.id AND l.source = 'test_labeling'
        JOIN predictions p ON p.image_id = i.id AND p.model = ?
        WHERE l.label != ?
        ORDER BY i.id
        """,
        (model, UNUSABLE),
    ).fetchall()
    return [{**dict(r), "scores": json.loads(r["scores"])} for r in rows]


def delete_media(conn: sqlite3.Connection, media_id: int) -> list[str]:
    """Delete a source file's rows: predictions, labels, images, then the media row.

    Returns the deleted images' file paths so the caller can remove the files.
    """
    image_rows = conn.execute("SELECT id, file_path FROM images WHERE media_id = ?", (media_id,))
    images = image_rows.fetchall()
    ids = [r["id"] for r in images]
    marks = ", ".join("?" * len(ids))
    with conn:
        if ids:
            conn.execute(f"DELETE FROM predictions WHERE image_id IN ({marks})", ids)
            conn.execute(f"DELETE FROM labels WHERE image_id IN ({marks})", ids)
            conn.execute(f"DELETE FROM images WHERE id IN ({marks})", ids)
        conn.execute("DELETE FROM media WHERE id = ?", (media_id,))
    return [r["file_path"] for r in images]


GALLERY_ORDERS = {
    "most confident": "p.confidence DESC, i.id",
    "least confident": "p.confidence ASC, i.id",
    "as filmed": "m.path, i.t_s, i.id",  # clip by clip, in time order within each clip
}


def gallery(
    conn: sqlite3.Connection,
    model: str,
    labels: list[str] | None = None,
    trips: list[str] | None = None,
    image_ids: list[int] | None = None,
    min_confidence: float = 0.0,
    order: str = "most confident",
    limit: int = 48,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """One page of images with `model`'s predictions, filtered. Returns (rows, total matches).

    A manual correction replaces the model's label (for display and the `labels` filter);
    `model_label` keeps what the model said and `corrected` flags the row. Empty or None
    `labels`/`trips`/`image_ids` means no filter. Each row has image_id, media_id, file_path,
    trip, source (original file path), t_s, label, model_label, corrected, confidence, and
    scores (dict).
    """
    where = ["p.model = ?", "p.confidence >= ?"]
    params: list = [model, min_confidence]
    shown_label = "COALESCE(c.label, p.label)"
    for column, values in ((shown_label, labels), ("m.trip", trips), ("i.id", image_ids)):
        if values:
            where.append(f"{column} IN ({', '.join('?' * len(values))})")
            params.extend(values)
    base = f"""
        FROM images i
        JOIN media m ON m.id = i.media_id
        JOIN predictions p ON p.image_id = i.id
        LEFT JOIN labels c ON c.image_id = i.id AND c.source = '{CORRECTION}'
        WHERE {" AND ".join(where)}
    """
    total = conn.execute(f"SELECT COUNT(*) {base}", params).fetchone()[0]
    rows = conn.execute(
        f"""
        SELECT i.id AS image_id, i.media_id, i.file_path, m.trip, m.path AS source, i.t_s,
               {shown_label} AS label, p.label AS model_label, c.label IS NOT NULL AS corrected,
               p.confidence, p.scores
        {base}
        ORDER BY {GALLERY_ORDERS[order]}
        LIMIT ? OFFSET ?
        """,
        [*params, limit, offset],
    ).fetchall()
    return [
        {**dict(r), "corrected": bool(r["corrected"]), "scores": json.loads(r["scores"])}
        for r in rows
    ], total


def correction_for(conn: sqlite3.Connection, image_id: int) -> str | None:
    row = conn.execute(
        "SELECT label FROM labels WHERE image_id = ? AND source = ?", (image_id, CORRECTION)
    ).fetchone()
    return row["label"] if row else None


def predictions_for(conn: sqlite3.Connection, image_id: int) -> dict[str, dict]:
    """Every model's prediction for one image: {model: {label, confidence, scores}}."""
    rows = conn.execute(
        "SELECT model, label, confidence, scores FROM predictions WHERE image_id = ?",
        (image_id,),
    )
    return {
        r["model"]: {
            "label": r["label"],
            "confidence": r["confidence"],
            "scores": json.loads(r["scores"]),
        }
        for r in rows
    }


def trips(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT trip FROM media WHERE trip IS NOT NULL ORDER BY trip")
    return [r["trip"] for r in rows]


def import_manifest(conn: sqlite3.Connection, frames_dir: Path) -> int:
    """Load kept frames from frames.py output (manifest.csv + done.txt). Safe to re-run.

    Returns the number of newly added images.
    """
    from tagger.frames import DONE, MANIFEST

    with open(frames_dir / DONE) as f:
        fingerprints = {
            path: fp
            for fp, path in (line.rstrip("\n").split("\t", 1) for line in f if line.strip())
        }
    before = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    with open(frames_dir / MANIFEST, newline="") as f, conn:
        for row in csv.DictReader(f):
            if row["decision"] != "kept":
                continue
            media_id = add_media(
                conn, row["source"], row["kind"], fingerprints[row["source"]], row["trip"]
            )
            add_image(
                conn,
                media_id,
                str(frames_dir / row["frame_path"]),
                t_s=float(row["t_s"]) if row["kind"] == "video" else None,
                sharpness=float(row["sharpness"]),
            )
    return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] - before
