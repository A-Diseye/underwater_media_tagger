"""Contact sheets for checking Grad-CAM and color correction by eye (docs/color_plan.md).

Checkpoint 1 (default): each frame appears twice, as it is and with the heatmap (red = most
evidence for the tag) and the subject mask's outline (white) on top. The caption gives the tag,
confidence, and whether the subject or the whole frame would steer the color correction.

Checkpoint 2 (--correct): each frame appears three times: original | whole-frame correction |
subject-aware correction. Set A only, since B and C are for scoring.

Run with:  uv run python -m training.color_sheets --set A [--per-category 5] [--correct]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tagger import classify, color, color_sets, locate, store
from tagger.locate import MIN_CONFIDENCE, Located, Locator, overlay

OUT_DIR = Path("data/color")
PANEL_W = 400


def panels(images: list[np.ndarray], caption: str) -> np.ndarray:
    """Images side by side at PANEL_W each, under a caption bar."""
    h = round(images[0].shape[0] * PANEL_W / images[0].shape[1])
    small = [cv2.resize(im, (PANEL_W, h), interpolation=cv2.INTER_AREA) for im in images]
    gap = np.full((h, 6, 3), 255, np.uint8)
    row = [small[0]]
    for im in small[1:]:
        row += [gap, im]
    body = np.hstack(row)
    bar = np.full((28, body.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, caption, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, body])


def cell(rgb: np.ndarray, found: Located, caption: str) -> np.ndarray:
    return panels([rgb, overlay(rgb, found)], caption)


def outline(rgb: np.ndarray, found: Located) -> np.ndarray:
    """A thin outline of the subject mask, so the sheet shows what steered the correction."""
    out = rgb.copy()
    if found.has_subject:
        edge = (found.mask >= 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), max(1, rgb.shape[1] // 600))
    return out


def corrections(rgb: np.ndarray, found: Located) -> dict[str, np.ndarray]:
    """The frame as it is and both corrections, at full resolution."""
    mask = found.mask if found.has_subject else None
    return {
        "original": rgb,
        "whole": color.correct(rgb, mode=color.WHOLE),
        "subject": color.correct(rgb, mask, color.SUBJECT),
    }


def correction_cell(views: dict[str, np.ndarray], found: Located, caption: str) -> np.ndarray:
    return panels([views["original"], views["whole"], outline(views["subject"], found)], caption)


def sheet(cells: list[np.ndarray], columns: int = 2) -> np.ndarray:
    width = max(c.shape[1] for c in cells)
    rows = []
    for i in range(0, len(cells), columns):
        group = cells[i : i + columns]
        height = max(c.shape[0] for c in group)
        padded = [
            np.pad(
                c,
                ((0, height - c.shape[0]), (0, width - c.shape[1] + 10), (0, 0)),
                constant_values=255,
            )
            for c in group
        ]
        padded += [np.full_like(padded[0], 255)] * (columns - len(group))
        rows.append(np.hstack(padded))
    return np.vstack(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM contact sheet for one frame set.")
    parser.add_argument("--set", choices=color_sets.SETS[:3], required=True)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--min-peak", type=float, help="Override locate.MIN_PEAK (tuning on A).")
    parser.add_argument("--threshold", type=float, help="Override locate.CAM_THRESHOLD.")
    parser.add_argument("--diver-threshold", type=float, help="Override DIVER_THRESHOLD.")
    parser.add_argument("--grow", type=float, help="Override locate.GROW.")
    parser.add_argument("--bridge", type=float, help="Override locate.BRIDGE (0 = off).")
    parser.add_argument("--correct", action="store_true", help="Before/after sheet (set A).")
    parser.add_argument("--red", type=float, help="Override color.RED_AMOUNT (set A).")
    parser.add_argument(
        "--finish", choices=tuple(color.FINISHES), help="Contrast and sharpening (set A)."
    )
    parser.add_argument(
        "--no-shallow", action="store_true", help="Full correction near the surface too."
    )
    parser.add_argument(
        "--files", action="store_true", help="With --correct: also save each view full size."
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    tuning = (args.min_peak, args.threshold, args.diver_threshold, args.grow, args.bridge)
    if args.set != "A" and any(v is not None for v in tuning):
        parser.error("mask settings are tuned on set A only")
    if args.set != "A" and (args.correct or args.red is not None or args.finish or args.no_shallow):
        parser.error("correction defaults are tuned on set A only")
    if args.red is not None:
        color.RED_AMOUNT = args.red
    if args.finish:
        color.CONTRAST, color.SHARPEN = color.FINISHES[args.finish]
    look = next((k for k, v in color.FINISHES.items() if v == (color.CONTRAST, color.SHARPEN)), "")
    look = "" if look in ("", "none") else f"_{look}"  # sharpened files say so in their name
    if args.no_shallow:
        color.SHALLOW_SCALING = False
    if color.SHALLOW_SCALING:
        look += "_shallow"
    if args.min_peak is not None:
        locate.MIN_PEAK = args.min_peak
    if args.threshold is not None:
        locate.CAM_THRESHOLD = args.threshold
    if args.diver_threshold is not None:
        locate.DIVER_THRESHOLD = args.diver_threshold
    if args.grow is not None:
        locate.GROW = args.grow
    if args.bridge is not None:
        locate.BRIDGE = args.bridge
    conn = store.connect()
    rows = color_sets.draw(
        color_sets.candidates(conn, classify.ACTIVE_MODEL_ID, args.set),
        args.per_category,
        min_confidence=MIN_CONFIDENCE,
    )
    locator = Locator(classify.Ensemble(classify.ACTIVE_CHECKPOINTS).members)
    cells, start = [], time.perf_counter()
    for r in rows:
        with Image.open(r["file_path"]) as img:
            img = img.convert("RGB")
        found = locator.locate(img, r["label"], r["confidence"])
        what = "subject" if found.has_subject else f"whole frame ({found.reason})"
        caption = f"#{r['image_id']} {r['trip']} | {r['label']} {r['confidence']:.2f} | {what}"
        if args.correct:
            caption += "   [original | whole | subject]"
        if not args.correct:
            cells.append(cell(np.asarray(img), found, caption))
            continue
        views = corrections(np.asarray(img), found)
        cells.append(correction_cell(views, found, caption))
        if args.files:
            folder = args.out / f"color_set_A_red{color.RED_AMOUNT}{look}"
            folder.mkdir(parents=True, exist_ok=True)
            for view, rgb in views.items():
                red = f"{color.RED_AMOUNT:g}".removeprefix("0")  # 0.7 -> .7, 1.0 -> 1
                name = f"{r['image_id']}_{r['label']}_{view}_{red}{look}.jpg"
                Image.fromarray(rgb).save(folder / name, quality=92)
    per_frame = (time.perf_counter() - start) / max(len(rows), 1)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = (
        f"_t{locate.CAM_THRESHOLD}_d{locate.DIVER_THRESHOLD}_p{locate.MIN_PEAK}"
        f"_g{locate.GROW}_b{locate.BRIDGE}"
        if args.set == "A"
        else ""
    )
    name = (
        f"color_set_A_red{color.RED_AMOUNT}{look}"
        if args.correct
        else f"gradcam_set_{args.set}{tag}"
    )
    out = args.out / f"{name}.jpg"
    Image.fromarray(sheet(cells, columns=1 if args.correct else 2)).save(out, quality=88)
    print(f"{len(rows)} frames, {per_frame:.2f} s per frame -> {out}")


if __name__ == "__main__":
    main()
