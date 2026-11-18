"""Find frames that look alike, using CLIP image embeddings.

An embedding is a list of 512 numbers that summarizes what an image looks like; CLIP was trained
so that images that look alike get vectors pointing in similar directions. With unit-length
vectors, "how alike" is just a dot product, so comparing one frame against the whole library is
a single matrix multiplication, fast enough without any special index.

CLIP rather than the fine-tuned classifier: the classifier only learned to separate 6 categories,
so to it two very different fish photos look nearly the same.

The embeddings are computed once (a few minutes on the GPU) and saved to EMBEDDINGS; uploads are
added as they're tagged.

Build or update with:  uv run python -m tagger.similar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tagger import store

EMBEDDINGS = Path("data/embeddings/clip.npz")


def _load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    with np.load(path) as data:
        return data["ids"], data["vectors"]


def add(conn, clip, image_ids: list[int] | None = None, path: Path = EMBEDDINGS) -> int:
    """Embed images that have no embedding yet: all of them, or just `image_ids`. Returns how
    many were added."""
    ids, vectors = _load_arrays(path)
    have = set(ids.tolist())
    rows = conn.execute("SELECT id, file_path FROM images ORDER BY id").fetchall()
    wanted = set(image_ids) if image_ids is not None else None
    todo = [r for r in rows if r["id"] not in have and (wanted is None or r["id"] in wanted)]
    if not todo:
        return 0
    new = clip.embed_paths([r["file_path"] for r in todo]).astype(np.float32)
    ids = np.concatenate([ids, np.array([r["id"] for r in todo], dtype=np.int64)])
    vectors = new if vectors.size == 0 else np.concatenate([vectors, new])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, ids=ids, vectors=vectors)
    tmp.replace(path)  # an interrupted save never leaves a half-written file behind
    return len(todo)


class Index:
    """Embeddings in memory, for nearest-neighbor lookups."""

    def __init__(self, ids: np.ndarray, vectors: np.ndarray):
        self.ids, self.vectors = ids, vectors
        self.rows = {int(image_id): row for row, image_id in enumerate(ids.tolist())}

    @classmethod
    def load(cls, path: Path = EMBEDDINGS) -> Index | None:
        ids, vectors = _load_arrays(path)
        return cls(ids, vectors) if len(ids) else None

    def __contains__(self, image_id: int) -> bool:
        return image_id in self.rows

    def nearest(
        self, image_id: int, k: int, exclude: set[int] = frozenset()
    ) -> list[tuple[int, float]]:
        """The `k` most similar images as (image id, similarity from -1 to 1), most similar
        first, leaving out the image itself and anything in `exclude`."""
        if image_id not in self.rows:
            raise KeyError(f"no embedding for image {image_id}")
        similarity = self.vectors @ self.vectors[self.rows[image_id]]
        found = []
        for row in np.argsort(-similarity):
            other = int(self.ids[row])
            if other != image_id and other not in exclude:
                found.append((other, float(similarity[row])))
                if len(found) == k:
                    break
        return found


def main(argv: list[str] | None = None) -> None:
    from tagger.baseline import ClipBaseline

    parser = argparse.ArgumentParser(description="Embed library images with CLIP.")
    parser.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=EMBEDDINGS)
    args = parser.parse_args(argv)
    added = add(store.connect(args.db), ClipBaseline(), path=args.out)
    print(f"embedded {added} new images into {args.out}")


if __name__ == "__main__":
    main()
