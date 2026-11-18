"""v3 sweep: which training tweaks actually help, alone and together?

A factorial experiment: every on/off combination of three training tweaks, 3 seeds each
(2 x 2 x 2 x 3 = 24 runs), so each tweak's effect is an average over 12 runs with it vs. 12
without, and pairs that only help (or hurt) together show up as interactions. Testing tweaks
one at a time on top of each other can't tell those apart.

Training tweaks (each run starts from the v3 recipe: balanced sampling, depth cast, wide photos,
macro-F1 epoch selection):
- image_size 224 vs 320: small fish and distant animals keep more pixels.
- zoom_out 0.3 vs 0.5: more practice finding a small animal in a wide scene.
- mixup off vs on: blended photos and labels, against one-cue shortcuts.

After-the-fact tweaks, applied to every run's predictions (no retraining):
- flip: average each frame's prediction with its mirror image's.
- smoothing window 0, 2.5, or 4.5 s (1 or 2 neighbors each side): average with neighboring
  frames in the clip (tagger/smooth.py).

Scored on the dev half (my labels; the holdout is never read here), plus two checks from what I
know about the footage, with no labels involved: turtle tags in the Azores and ray tags in
Mexico, where there are none. Validation macro-F1 comes from each checkpoint.

Only the frames needed are tagged: whole dev clips (smoothing needs neighbors) and the Azores
and Mexico trips. They're cached once at a small size, since decoding 1080p JPEGs, not the GPU,
is what makes tagging slow. Outputs go to data/sweep/.

Run with:  uv run python -m training.sweep [--steps train predict report]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from tagger import CATEGORIES, eval_split, store
from tagger.smooth import smooth

SWEEP_DIR = Path("data/sweep")
GRID = {"image_size": (224, 320), "zoom_out": (0.3, 0.5), "mixup": (False, True)}
SEEDS = (0, 1, 2)
# Frames are ~2 s apart but snap to keyframes, so windows are set between frame gaps:
# ±2.5 s is one neighbor on each side, ±4.5 s two.
WINDOWS = (0, 2.5, 4.5)
CACHE_SIDE = 366  # short side; the 320 px model resizes to 365 before its center crop
# Known-absent checks: (trip, category) pairs where I know the count should be about zero.
ABSENT = (("azores", "turtles"), ("mexico", "rays"))
METRICS = ("val_f1", "dev_f1", "dev_ba", "azores_turtles", "mexico_rays")


def recipes() -> list[dict]:
    return [dict(zip(GRID, values, strict=True)) for values in itertools.product(*GRID.values())]


def run_name(recipe: dict, seed: int) -> str:
    mix = "mx" if recipe["mixup"] else "nomx"
    return f"sw_{recipe['image_size']}_z{round(recipe['zoom_out'] * 100)}_{mix}_s{seed}"


def subset(conn, split: dict[str, set[str]]) -> list[dict]:
    """Frames the sweep scores: whole dev clips plus the known-absent trips."""
    trips = {trip for trip, _ in ABSENT}
    rows = conn.execute(
        """
        SELECT i.id, i.file_path, i.media_id, i.t_s, m.trip, m.path FROM images i
        JOIN media m ON m.id = i.media_id ORDER BY i.id
        """
    ).fetchall()
    return [dict(r) for r in rows if r["path"] in split["dev"] or r["trip"] in trips]


def dev_labels(conn, split: dict[str, set[str]]) -> dict[int, str]:
    rows = store.evaluation_rows(conn, "clip-vitb32-zeroshot-6cat")  # any model: same labels
    return {r["image_id"]: r["true_label"] for r in rows if r["source"] in split["dev"]}


def _cache_one(job: tuple[str, str]) -> None:
    src, dst = job
    with Image.open(src) as img:
        img = img.convert("RGB")
        scale = CACHE_SIDE / min(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
        img.save(dst, quality=95)


def build_cache(rows: list[dict], cache_dir: Path) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [str(cache_dir / f"{r['id']}.jpg") for r in rows]
    todo = [(r["file_path"], p) for r, p in zip(rows, paths, strict=True) if not Path(p).exists()]
    if todo:
        with Pool(16) as pool:
            pool.map(_cache_one, todo, chunksize=32)
    return paths


def predict(checkpoint: Path, paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(probabilities, probabilities averaged with the mirror image's), one row per path."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    from tagger.classify import Classifier

    clf = Classifier(checkpoint)

    class Frames(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            with Image.open(paths[i]) as img:
                return clf.preprocess(img.convert("RGB"))

    plain, flipped = [], []
    loader = DataLoader(Frames(), batch_size=128, num_workers=8)
    with torch.no_grad(), torch.autocast(clf.device, enabled=clf.device == "cuda"):
        for batch in loader:
            batch = batch.to(clf.device)
            p = clf.model(batch).float().softmax(-1)
            q = clf.model(torch.flip(batch, dims=[3])).float().softmax(-1)
            plain.append(p.cpu().numpy())
            flipped.append(((p + q) / 2).cpu().numpy())
    return np.concatenate(plain), np.concatenate(flipped)


def score(rows: list[dict], probs: np.ndarray, labels: dict[int, str]) -> dict:
    from training.evaluate import metrics

    guesses = [CATEGORIES[k] for k in probs.argmax(1)]
    dev = [
        {"true_label": labels[r["id"]], "predicted": g}
        for r, g in zip(rows, guesses, strict=True)
        if r["id"] in labels
    ]
    m = metrics(dev)
    out = {"dev_f1": m["macro_f1"], "dev_ba": m["balanced_accuracy"], "dev_n": m["n"]}
    for trip, category in ABSENT:
        out[f"{trip}_{category}"] = sum(
            r["trip"] == trip and g == category for r, g in zip(rows, guesses, strict=True)
        )
    return out


def effect(records: list[dict], factor: str, metric: str) -> float:
    """Average `metric` with the factor's second level minus with its first."""
    levels = sorted({r[factor] for r in records})
    on = [r[metric] for r in records if r[factor] == levels[1]]
    off = [r[metric] for r in records if r[factor] == levels[0]]
    return statistics.mean(on) - statistics.mean(off)


def interaction(records: list[dict], a: str, b: str, metric: str) -> float:
    """How much A's effect changes when B is switched on: effect(A | B on) - effect(A | B off).
    Near zero means the two tweaks act independently."""
    levels = sorted({r[b] for r in records})
    on = [r for r in records if r[b] == levels[1]]
    off = [r for r in records if r[b] == levels[0]]
    return effect(on, a, metric) - effect(off, a, metric)


def seed_noise(records: list[dict], metric: str) -> float:
    """Typical seed-to-seed spread: the standard deviation across seeds, averaged over recipes."""
    groups = {}
    for r in records:
        groups.setdefault(tuple(r[f] for f in GRID), []).append(r[metric])
    return statistics.mean(statistics.stdev(v) for v in groups.values() if len(v) > 1)


def report(records: list[dict], reference: dict[str, dict]) -> str:
    fmt = {"val_f1": "{:+.1%}", "dev_f1": "{:+.1%}", "dev_ba": "{:+.1%}"}
    show = lambda m, v: fmt.get(m, "{:+.0f}").format(v)  # noqa: E731
    base = [r for r in records if not r["flip"] and r["window"] == 0]
    lines = [
        "# v3 sweep (dev half; the holdout is not used)",
        "",
        f"{len(base)} trained runs. Dev macro-F1 and balanced accuracy rest on "
        f"{base[0]['dev_n']} labeled dev frames. The last two columns count tags where the "
        "category is absent (lower is better).",
        "",
        "## Effect of each tweak (with minus without, averaged over everything else)",
        "",
        "| Tweak | " + " | ".join(METRICS) + " |",
        "|" + " --- |" * (len(METRICS) + 1),
    ]
    for factor in GRID:
        cells = [show(m, effect(base, factor, m)) for m in METRICS]
        lines.append(f"| {factor} | " + " | ".join(cells) + " |")
    post = [r for r in records if r["window"] == 0]
    cells = [show(m, effect(post, "flip", m)) if m != "val_f1" else "–" for m in METRICS]
    lines.append("| flip at test time | " + " | ".join(cells) + " |")
    for window in WINDOWS[1:]:
        pair = [r for r in records if not r["flip"] and r["window"] in (0, window)]
        cells = [show(m, effect(pair, "window", m)) if m != "val_f1" else "–" for m in METRICS]
        lines.append(f"| smoothing ±{window} s | " + " | ".join(cells) + " |")
    noise = [show(m, seed_noise(base, m)).lstrip("+") for m in METRICS]
    lines += [
        "| *seed-to-seed spread (std)* | " + " | ".join(f"*±{n}*" for n in noise) + " |",
        "",
        "An effect smaller than about the seed spread is noise: each side averages 12 runs, so "
        "the uncertainty of a difference is roughly 0.4 x the spread, but a few dev frames in "
        "rare categories still move these numbers.",
        "",
        "## Interactions (does A's effect change when B is on?)",
        "",
        "| A × B | " + " | ".join(METRICS) + " |",
        "|" + " --- |" * (len(METRICS) + 1),
    ]
    for a, b in itertools.combinations(GRID, 2):
        cells = [show(m, interaction(base, a, b, m)) for m in METRICS]
        lines.append(f"| {a} × {b} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Every recipe (mean of 3 seeds; no flip, no smoothing)",
        "",
        "| " + " | ".join(GRID) + " | " + " | ".join(METRICS) + " |",
        "|" + " --- |" * (len(GRID) + len(METRICS)),
    ]
    for recipe in recipes():
        rs = [r for r in base if all(r[f] == v for f, v in recipe.items())]
        means = [statistics.mean(r[m] for r in rs) for m in METRICS]
        cells = [
            f"{v:.1%}" if m in fmt else f"{v:.0f}" for m, v in zip(METRICS, means, strict=True)
        ]
        lines.append(
            "| " + " | ".join(str(v) for v in recipe.values()) + " | " + " | ".join(cells) + " |"
        )
    lines += ["", "Reference models on the same frames:", ""]
    for name, s in reference.items():
        lines.append(
            f"- `{name}`: dev macro-F1 {s['dev_f1']:.1%}, balanced accuracy {s['dev_ba']:.1%}, "
            f"Azores turtles {s['azores_turtles']}, Mexico rays {s['mexico_rays']}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", nargs="+", default=["train", "predict", "report"])
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args(argv)
    runs = [(recipe, seed, run_name(recipe, seed)) for recipe in recipes() for seed in SEEDS]

    if "train" in args.steps:
        from training import train as tr

        for recipe, seed, name in runs:
            if (args.models_dir / f"{name}.pt").exists():
                continue
            cfg = tr.Config(
                run_name=name,
                models_dir=args.models_dir,
                seed=seed,
                balanced=True,
                select="macro_f1",
                **recipe,
            )
            best = tr.train(cfg)["best"]
            print(f"{name}: val macro-F1 {best['macro_f1']:.3f}", flush=True)

    conn = store.connect()
    split = eval_split.load_split()
    rows = subset(conn, split)
    preds_dir = SWEEP_DIR / "preds"
    if "predict" in args.steps:
        paths = build_cache(rows, SWEEP_DIR / "cache")
        preds_dir.mkdir(parents=True, exist_ok=True)
        ids = np.array([r["id"] for r in rows])
        for _, _, name in runs:
            out = preds_dir / f"{name}.npz"
            if not out.exists():
                plain, flipped = predict(args.models_dir / f"{name}.pt", paths)
                np.savez(out, ids=ids, plain=plain, flipped=flipped)
                print(f"predicted {name}", flush=True)

    if "report" in args.steps:
        import torch

        labels = dev_labels(conn, split)
        media = np.array([r["media_id"] for r in rows])
        times = np.array([np.nan if r["t_s"] is None else r["t_s"] for r in rows])
        records = []
        for recipe, seed, name in runs:
            saved = torch.load(
                args.models_dir / f"{name}.pt", map_location="cpu", weights_only=False
            )
            data = np.load(preds_dir / f"{name}.npz")
            assert data["ids"].tolist() == [r["id"] for r in rows], "subset changed: re-predict"
            for flip, window in itertools.product((False, True), WINDOWS):
                probs = smooth(media, times, data["flipped" if flip else "plain"], window)
                records.append(
                    {
                        **recipe,
                        "seed": seed,
                        "flip": flip,
                        "window": window,
                        "val_f1": saved["best"]["selection_score"],
                        **score(rows, probs, labels),
                    }
                )
        reference = {}
        for model in ("clip-vitb32-zeroshot-6cat", "v2_both_s0", "v3_bal_ens"):
            scores = dict(
                conn.execute("SELECT image_id, scores FROM predictions WHERE model = ?", (model,))
            )
            probs = np.array([[json.loads(scores[r["id"]])[c] for c in CATEGORIES] for r in rows])
            reference[model] = score(rows, probs, labels)
        with open(SWEEP_DIR / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (SWEEP_DIR / "report.md").write_text(report(records, reference))
        print(f"report: {SWEEP_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
