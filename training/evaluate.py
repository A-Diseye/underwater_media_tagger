"""Compare models on my hand-labeled test set.

Read-only, and run once after the model is chosen: these numbers never feed back into any
training choice (hard rule: tuning uses validation only). Every model is scored on exactly the
same frames: those with a usable hand label and a prediction from every model.

Headline: balanced accuracy (the average of per-category recall), so the hundreds of scenery
frames can't hide a weak turtle score. Every number shows how many frames it rests on.

Slices:
- random-only: frames picked at random for labeling, the check that targeted picks didn't bias it
- camera: my GoPro vs. trip-mates' phones (bahamas_shared)
- trip

Outputs go to docs/results/<version>/: results.md, a confusion matrix per model, and a
contact sheet of frames the primary model got wrong.

Run with:  uv run python -m training.evaluate --primary ID [--ablation ID ...] [--split dev]
           (the holdout half also needs --final, once)
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from tagger import CATEGORIES, eval_split, store
from tagger.baseline import MODEL_ID as CLIP_MODEL_ID

OUT_DIR = Path("docs/results/v1")
PHONE_FOLDER = "bahamas_shared"


def camera(source: str) -> str:
    return "phone" if PHONE_FOLDER in source else "gopro"


def shared_rows(conn, models: list[str]) -> dict[str, list[dict]]:
    """Each model's evaluation rows, restricted to frames every model has predicted."""
    rows = {m: store.evaluation_rows(conn, m) for m in models}
    common = set.intersection(*({r["image_id"] for r in rs} for rs in rows.values()))
    return {m: [r for r in rs if r["image_id"] in common] for m, rs in rows.items()}


def metrics(rows: list[dict]) -> dict:
    """Accuracy, balanced accuracy, per-category precision/recall with counts, confusion."""
    confusion = Counter((r["true_label"], r["predicted"]) for r in rows)
    per_class = {}
    for c in CATEGORIES:
        n_true = sum(n for (t, _), n in confusion.items() if t == c)
        n_pred = sum(n for (_, p), n in confusion.items() if p == c)
        hits = confusion[c, c]
        per_class[c] = {
            "n": n_true,
            "recall": hits / n_true if n_true else None,
            "precision": hits / n_pred if n_pred else None,
        }
    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    return {
        "n": len(rows),
        "accuracy": sum(r["true_label"] == r["predicted"] for r in rows) / len(rows) if rows else 0,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0,
        "per_class": per_class,
        "confusion": confusion,
    }


def slices(rows: list[dict]) -> dict[str, list[dict]]:
    out = {"all": rows, "random-only": [r for r in rows if r["selection"] == "random"]}
    for cam in ("gopro", "phone"):
        out[cam] = [r for r in rows if camera(r["source"]) == cam]
    for trip in sorted({r["trip"] for r in rows}):
        out[f"trip: {trip}"] = [r for r in rows if r["trip"] == trip]
    return out


def pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"


def confusion_image(m: dict, title: str, path: Path) -> None:
    """Rows = true category, columns = predicted. Cell shade = share of that row."""
    cell, left, top = 64, 120, 120
    size = len(CATEGORIES)
    img = Image.new("RGB", (left + size * cell + 10, top + size * cell + 10), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), title, fill="black")
    draw.text((left, top - 40), "predicted ->", fill="gray")
    for j, c in enumerate(CATEGORIES):
        draw.text((left + j * cell + 4, top - 18), c[:9], fill="black")
    for i, t in enumerate(CATEGORIES):
        draw.text((10, top + i * cell + cell // 2 - 6), t, fill="black")
        row_total = sum(m["confusion"][t, p] for p in CATEGORIES) or 1
        for j, p in enumerate(CATEGORIES):
            n = m["confusion"][t, p]
            share = n / row_total
            shade = int(255 * (1 - share))
            color = (shade, shade, 255) if i == j else (255, shade, shade)
            box = (left + j * cell, top + i * cell, left + (j + 1) * cell, top + (i + 1) * cell)
            draw.rectangle(box, fill=color, outline="lightgray")
            if n:
                fill = "white" if share > 0.6 else "black"
                draw.text((box[0] + cell // 2 - 8, box[1] + cell // 2 - 6), str(n), fill=fill)
    img.save(path)


def failure_sheet(rows: dict[str, list[dict]], primary: str, path: Path, limit: int = 48) -> int:
    """Frames the primary model got wrong, with every model's guess under each."""
    by_model = {m: {r["image_id"]: r for r in rs} for m, rs in rows.items()}
    wrong = [r for r in rows[primary] if r["true_label"] != r["predicted"]][:limit]
    w, h, text, cols = 240, 135, 14 * (len(rows) + 1), 6
    sheet = Image.new("RGB", (cols * w, max(1, -(-len(wrong) // cols)) * (h + text)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(wrong):
        x, y = (i % cols) * w, (i // cols) * (h + text)
        with Image.open(r["file_path"]) as img:
            thumb = img.convert("RGB")
            thumb.thumbnail((w - 4, h - 2))
            sheet.paste(thumb, (x, y))
        draw.text((x + 2, y + h), f"true: {r['true_label']}", fill="darkgreen")
        for k, (m, rs) in enumerate(by_model.items(), start=1):
            guess = rs[r["image_id"]]["predicted"]
            draw.text((x + 2, y + h + 14 * k), f"{m[:18]}: {guess}", fill="red")
    sheet.save(path)
    return len(wrong)


def report(
    rows: dict[str, list[dict]], names: dict[str, str], primary: str, half: str | None = None
) -> str:
    models = list(rows)
    scope = {
        None: "all labeled frames",
        "dev": "the **dev** half (used for error analysis and choosing fixes)",
        "holdout": "the **holdout** half (locked until now, scored once)",
    }[half]
    lines = [
        f"# Test-set results: {scope}",
        "",
        f"Hand-labeled frames from my own dives, `unusable` excluded: **{len(rows[primary])}** "
        "frames, the same frames for every model. Balanced accuracy is the average of "
        "per-category recall, so every category counts equally.",
        "",
        "| Model | Role | Balanced accuracy | Accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for m in models:
        m_all = metrics(rows[m])
        lines.append(
            f"| `{m}` | {names[m]} | **{pct(m_all['balanced_accuracy'])}** | "
            f"{pct(m_all['accuracy'])} |"
        )
    lines += ["", "## Recall per category (frames it rests on)", ""]
    header = "| Category | n | " + " | ".join(f"`{m}`" for m in models) + " |"
    lines += [header, "|" + " --- |" * (len(models) + 2)]
    per_model = {m: metrics(rows[m])["per_class"] for m in models}
    for c in CATEGORIES:
        n = per_model[primary][c]["n"]
        lines.append(
            f"| {c} | {n} | " + " | ".join(pct(per_model[m][c]["recall"]) for m in models) + " |"
        )
    lines += ["", "## Slices: balanced accuracy (frames)", ""]
    lines += [header.replace("Category | n", "Slice | n"), "|" + " --- |" * (len(models) + 2)]
    for name, subset in slices(rows[primary]).items():
        ids = {r["image_id"] for r in subset}
        cells = [
            pct(metrics([r for r in rows[m] if r["image_id"] in ids])["balanced_accuracy"])
            for m in models
        ]
        lines.append(f"| {name} | {len(subset)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Confusion matrices: " + ", ".join(f"[`{m}`](confusion_{m}.png)" for m in models) + ".",
        "Frames the primary model got wrong: [failures.jpg](failures.jpg).",
        "",
    ]
    return "\n".join(lines)


def evaluate(
    conn,
    primary: str,
    ablations: list[str],
    out_dir: Path = OUT_DIR,
    half: str | None = None,
    split_path: Path = eval_split.SPLIT_PATH,
) -> dict:
    """Score the models. `half` limits scoring to one half of the dev/holdout split."""
    names = {CLIP_MODEL_ID: "zero-shot CLIP baseline", primary: "fine-tuned (chosen)"}
    names.update({a: "fine-tuned (comparison, didn't affect the choice)" for a in ablations})
    if half == "dev":  # nothing is chosen yet: dev is where candidates are compared
        names.update({m: "fine-tuned candidate" for m in (primary, *ablations)})
    rows = shared_rows(conn, list(names))
    if half is not None:
        split = eval_split.load_split(split_path)
        if split is None:
            raise FileNotFoundError(f"no test split at {split_path}")
        rows = {m: [r for r in rs if r["source"] in split[half]] for m, rs in rows.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in rows:
        confusion_image(metrics(rows[m]), f"{m} ({names[m]})", out_dir / f"confusion_{m}.png")
    n_wrong = failure_sheet(rows, primary, out_dir / "failures.jpg")
    (out_dir / "results.md").write_text(report(rows, names, primary, half))
    return {"frames": len(rows[primary]), "primary_wrong_shown": n_wrong}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare models on the hand-labeled test set.")
    parser.add_argument("--primary", required=True, help="Model ID of the chosen model.")
    parser.add_argument("--ablation", action="append", default=[], help="Other model IDs.")
    parser.add_argument("--split", choices=eval_split.HALVES, default="dev")
    parser.add_argument(
        "--final",
        action="store_true",
        help="Required to score the holdout half. Use once, when the model is final.",
    )
    parser.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    parser.add_argument("--out", type=Path, help="Default: docs/results/<split>.")
    args = parser.parse_args(argv)
    if args.split == "holdout" and not args.final:
        parser.error("the holdout half is locked: pass --final, once, when the model is final")
    out = args.out or Path("docs/results") / args.split
    result = evaluate(store.connect(args.db), args.primary, args.ablation, out, half=args.split)
    print(f"{result['frames']} test frames ({args.split}); results in {out / 'results.md'}")


if __name__ == "__main__":
    main()
