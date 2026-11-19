"""Contact sheets for checking Grad-CAM by eye (docs/color_plan.md, checkpoint 1).

Each frame appears twice: as it is, and with the heatmap (red = most evidence for the tag) and
the subject mask's outline (white) on top. The caption gives the tag, confidence, and whether
the subject or the whole frame would steer the color correction.

Run with:  uv run python -m training.color_sheets --set A [--per-category 5]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tagger import classify, color_sets, locate, store
from tagger.locate import MIN_CONFIDENCE, Located, Locator, overlay

OUT_DIR = Path("data/color")
PANEL_W = 400


def cell(rgb: np.ndarray, found: Located, caption: str) -> np.ndarray:
    h = round(rgb.shape[0] * PANEL_W / rgb.shape[1])
    left = cv2.resize(rgb, (PANEL_W, h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(overlay(rgb, found), (PANEL_W, h), interpolation=cv2.INTER_AREA)
    bar = np.full((28, 2 * PANEL_W + 6, 3), 255, np.uint8)
    cv2.putText(bar, caption, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    gap = np.full((h, 6, 3), 255, np.uint8)
    return np.vstack([bar, np.hstack([left, gap, right])])


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
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    tuning = (args.min_peak, args.threshold, args.diver_threshold, args.grow, args.bridge)
    if args.set != "A" and any(v is not None for v in tuning):
        parser.error("mask settings are tuned on set A only")
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
        cells.append(cell(np.asarray(img), found, caption))
    per_frame = (time.perf_counter() - start) / max(len(rows), 1)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = (
        f"_t{locate.CAM_THRESHOLD}_d{locate.DIVER_THRESHOLD}_p{locate.MIN_PEAK}"
        f"_g{locate.GROW}_b{locate.BRIDGE}"
        if args.set == "A"
        else ""
    )
    out = args.out / f"gradcam_set_{args.set}{tag}.jpg"
    Image.fromarray(sheet(cells)).save(out, quality=88)
    print(f"{len(rows)} frames, {per_frame:.2f} s per frame -> {out}")


if __name__ == "__main__":
    main()
