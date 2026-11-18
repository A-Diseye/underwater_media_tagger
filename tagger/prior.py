"""Correct a model's tags for how common each category really is in my library.

Why: a classifier's probabilities carry the category mix it was trained on. v2 trained on ~127
turtles to 68 scenery photos, so it treats "turtle" as a common answer. In my footage turtles
are rare (a few dozen frames out of 12,000), so even a 1% false-alarm rate on scenery frames
buries the real turtles. That's the Azores problem: about 100 turtle tags where there are none.

The fix (Saerens et al., 2002, "label shift" correction) uses no labels at all:
1. Bayes' rule says a probability can be re-weighted for a new category mix:
   p_new(c | image) is proportional to p_train(c | image) * prior_new(c) / prior_train(c).
2. prior_new is unknown, so estimate it from the model's own predictions on the unlabeled
   library: start at the training mix, re-weight every frame, average the re-weighted
   probabilities into a new mix, and repeat until it stops changing (an EM algorithm).

Groups: one mix for the whole library, or one per trip (the Azores have no turtles, the Bahamas
do). Neither looks at my hand labels, so this isn't tuning on the test set; the dev half only
judges whether it helped.

Writes the corrected scores as a new model in the predictions table, without re-running the
network:  uv run python -m tagger.prior MODEL_ID [--by trip|all] [--train-prior 127,135,...]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tagger import CATEGORIES, store


def reweight(probs: np.ndarray, train_prior: np.ndarray, new_prior: np.ndarray) -> np.ndarray:
    """Bayes' rule: move each row of probabilities from the training mix to `new_prior`."""
    adjusted = probs * (new_prior / train_prior)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def estimate_prior(
    probs: np.ndarray, train_prior: np.ndarray, iterations: int = 200, tol: float = 1e-6
) -> np.ndarray:
    """The category mix that best explains these predictions (EM, Saerens et al. 2002)."""
    train_prior = train_prior / train_prior.sum()
    prior = train_prior.copy()
    for _ in range(iterations):
        new = reweight(probs, train_prior, prior).mean(axis=0)
        if np.abs(new - prior).max() < tol:
            return new
        prior = new
    return prior


def correct_store(
    conn, model_id: str, train_prior: np.ndarray, by: str = "trip", out_id: str | None = None
) -> dict[str, dict[str, float]]:
    """Store corrected predictions for every image `model_id` tagged, under `out_id`.
    Returns the estimated mix per group."""
    out_id = out_id or f"{model_id}_prior_{by}"
    rows = conn.execute(
        """
        SELECT p.image_id, p.scores, m.trip FROM predictions p
        JOIN images i ON i.id = p.image_id JOIN media m ON m.id = i.media_id
        WHERE p.model = ?
        """,
        (model_id,),
    ).fetchall()
    groups = defaultdict(list)
    for r in rows:
        groups[r["trip"] if by == "trip" else "all"].append(r)
    train_prior = np.asarray(train_prior, dtype=float)
    train_prior = train_prior / train_prior.sum()
    mixes = {}
    with conn:
        for group, members in sorted(groups.items()):
            probs = np.array([[json.loads(r["scores"])[c] for c in CATEGORIES] for r in members])
            prior = estimate_prior(probs, train_prior)
            mixes[group] = dict(zip(CATEGORIES, prior.round(4).tolist(), strict=True))
            for r, p in zip(members, reweight(probs, train_prior, prior), strict=True):
                store.add_prediction(
                    conn, r["image_id"], out_id, dict(zip(CATEGORIES, p.tolist(), strict=True))
                )
    return mixes


def checkpoint_prior(checkpoint: Path) -> np.ndarray | None:
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    prior = saved.get("train_prior")
    return None if prior is None else np.array([prior[c] for c in CATEGORIES])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Correct stored tags for the library's mix.")
    parser.add_argument("model_id")
    parser.add_argument("--by", choices=("trip", "all"), default="trip")
    parser.add_argument(
        "--train-prior",
        help="Training mix as counts in category order, e.g. 127,135,115,133,74,68 "
        "(default: saved in models/<model_id>.pt).",
    )
    parser.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    args = parser.parse_args(argv)
    if args.train_prior:
        prior = np.array([float(x) for x in args.train_prior.split(",")])
    else:
        prior = checkpoint_prior(Path("models") / f"{args.model_id}.pt")
    if prior is None or len(prior) != len(CATEGORIES):
        parser.error("no training mix: pass --train-prior with one count per category")
    mixes = correct_store(store.connect(args.db), args.model_id, prior, args.by)
    for group, mix in mixes.items():
        print(f"{group:18}", {c: f"{v:.1%}" for c, v in mix.items()})


if __name__ == "__main__":
    main()
