"""Grad-CAM hit check: does the subject mask land on the subject? (docs/color_plan.md, Stage 1)

`export` draws a set's frames, locates each subject with the final settings, and saves the
original and the heatmap overlay as JPEGs plus a manifest. The Color check page shows them one
at a time, and I mark each: hit, partial, miss, or "not there" (the tag is wrong, so there was
nothing to hit). Haiku later marks the same files (agent A), and its agreement with me decides
whether its larger count can be trusted.

Sets B and final are scored once: exporting again after marking has started is refused, so the
frames can't be redrawn until the numbers look better. Set A can be exported any time (tuning).
B scored the tight masks; final scores the grown ones on fresh frames from the pool.

Agent A (Haiku, a vision check, not an agent loop) sees the same two images and the tag, and
picks the same marks I do, with the same definitions. Round 1 (kept in check_B/round1/) used a
looser "hit" (the outline mostly on the subject); its reasons showed I'd been lenient, so the
definition was tightened to "covers the whole subject" for both of us, and set B re-marked.
Its larger count on the pool is used only if it lands in the same
group as me (hit / off: partial or miss / wrong tag) on at least AGREEMENT_BAR of set B. That
bar was set before Haiku's first run.

Run with:  uv run python -m training.color_check --set final         (export frames to mark)
           uv run python -m training.color_check --set B --haiku     (Haiku marks them too)
"""

from __future__ import annotations

import argparse
import base64
import json
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

OUT_DIR = Path("data/color")
PER_CATEGORY = 6
DISPLAY_WIDTH = 1280
MARKS = ("hit", "partial", "miss", "not_there", "wrong_tag_on_subject")
MARK_HELP = {
    "hit": "the outline covers the whole subject (a little beyond its edges is fine); with "
    "several of them (three divers), it covers all of them",
    "partial": "it cuts off part of the subject (fins, tail, head, the edge of the body), or "
    "it covers the subject but is mostly background",
    "miss": "the subject is there, but the outline is mostly somewhere else",
    "not_there": "there's no such subject in the frame: the tag is wrong",
    "wrong_tag_on_subject": "the tag is wrong (an eel called an invertebrate), but the outline "
    "covers the real main subject, which is what the color correction uses",
}


# The final score of the grown masks: fresh frames from the pool, since set B was spent on the
# tight masks (and on checking Haiku). Up to 25, marked once with the strict definitions.
FINAL_PER_CATEGORY = 5
SOURCE_SET = {"A": "A", "B": "B", "final": "pool"}  # which frame set each check draws from
AGREEMENT_BAR = 0.8
GROUPS = {
    "hit": "hit",
    "partial": "off",
    "miss": "off",
    "not_there": "wrong tag",
    "wrong_tag_on_subject": "wrong tag",
}

HIT_PROMPT = """You are checking where an image classifier looked. The two images are the same \
underwater frame: first as it is, then with a heatmap (red = where the classifier found the most \
evidence) and a white outline around the region it chose as the main subject.

The classifier tagged this frame: {tag}.
Its categories: turtles (sea turtles); rays (stingrays, eagle rays, mantas); fish (all other fish, \
including sharks and eels); invertebrates (nudibranchs, crabs, shrimp, lobsters, sea stars, \
octopus, squid, cuttlefish, clams or scallops); diver (people); scenery (coral, sponges, reef, \
sand, open water: no main animal or diver).

Pick one mark, judging only from what you can see. The outline will steer a photo editor's color \
correction, so it must take in the whole subject; going a little beyond its edges is fine.
- hit: the tag is right and the white outline covers the whole subject, including fins, tail, \
head and limbs (a little beyond its edges is fine). If there are several of the subject, such as \
three divers, it covers all of them
- partial: the tag is right, but the outline cuts off part of the subject, or it covers the \
subject but is mostly background
- miss: the tag is right and the subject is in the frame, but the outline is mostly somewhere else
- not_there: there is no {tag} in the frame at all, so the tag is wrong
- wrong_tag_on_subject: the tag is wrong (for example an eel tagged as invertebrates), but the \
outline covers the real main subject

Give a one-sentence reason."""


class HitCheck(BaseModel):
    """Haiku's answer for one frame. The API enforces this exact shape."""

    mark: Literal["hit", "partial", "miss", "not_there", "wrong_tag_on_subject"]
    reason: str


def set_dir(which: str, root: Path = OUT_DIR) -> Path:
    return root / f"check_{which}"


def load_manifest(which: str, root: Path = OUT_DIR) -> list[dict]:
    path = set_dir(which, root) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else []


def load_marks(which: str, root: Path = OUT_DIR, who: str = "me") -> dict[str, str]:
    """{image id (str): mark} for one marker: "me" or "haiku"."""
    path = set_dir(which, root) / f"marks_{who}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_mark(which: str, image_id: int, mark: str, root: Path = OUT_DIR, who: str = "me"):
    if mark not in MARKS:
        raise ValueError(f"unknown mark: {mark!r}")
    marks = load_marks(which, root, who)
    marks[str(image_id)] = mark
    (set_dir(which, root) / f"marks_{who}.json").write_text(json.dumps(marks, indent=1))


def exported_sets(root: Path = OUT_DIR) -> list[str]:
    return sorted(p.name.removeprefix("check_") for p in root.glob("check_*") if p.is_dir())


def summary(manifest: list[dict], marks: dict[str, str]) -> dict:
    """Hit rate over frames where a mask was drawn and the subject was really there.

    Frames where Grad-CAM found no clear subject have no outline to judge (they're corrected on
    the whole frame); they're counted as fallbacks, not marked.

    Frames whose tag was wrong are counted on their own, not in the hit rate: "not_there" (no
    such subject, so nothing to hit) shows how often a false tag still passes the confidence
    gate; "wrong_tag_on_subject" (wrong category, but the outline is on the real subject) shows
    how often the color correction is still steered by the right thing.
    """
    to_mark = [f for f in manifest if f["reason"] == "subject"]
    marked = [f for f in to_mark if str(f["image_id"]) in marks]
    counts = Counter(marks[str(f["image_id"])] for f in marked)
    judged = counts["hit"] + counts["partial"] + counts["miss"]
    by_category: dict[str, Counter] = {}
    for f in marked:
        by_category.setdefault(f["label"], Counter())[marks[str(f["image_id"])]] += 1
    return {
        "frames": len(manifest),
        "to_mark": len(to_mark),
        "fallbacks": len(manifest) - len(to_mark),  # no clear subject: whole-frame correction
        "marked": len(marked),
        "counts": dict(counts),
        "hit_rate": counts["hit"] / judged if judged else None,
        "hit_or_partial_rate": (counts["hit"] + counts["partial"]) / judged if judged else None,
        "by_category": {c: dict(v) for c, v in sorted(by_category.items())},
    }


def agreement(manifest: list[dict], mine: dict[str, str], haiku: dict[str, str]) -> dict:
    """How often Haiku's mark lands in the same group as mine, on frames we both marked."""
    both = [
        str(f["image_id"])
        for f in manifest
        if f["reason"] == "subject" and str(f["image_id"]) in mine and str(f["image_id"]) in haiku
    ]
    same_group = sum(GROUPS[mine[i]] == GROUPS[haiku[i]] for i in both)
    pairs = Counter((mine[i], haiku[i]) for i in both)
    rate = same_group / len(both) if both else None
    return {
        "frames": len(both),
        "same_mark": sum(mine[i] == haiku[i] for i in both) / len(both) if both else None,
        "same_group": rate,
        "trusted": rate is not None and rate >= AGREEMENT_BAR,
        "pairs (me, haiku)": {f"{a} / {b}": n for (a, b), n in sorted(pairs.items())},
    }


def _image(path: str) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def haiku_mark(client, frame: dict) -> tuple[HitCheck | None, float]:
    """Ask Haiku to mark one frame. Returns (answer or None if it couldn't answer, cost)."""
    import anthropic

    from tagger.pricing import HAIKU, cost_usd

    try:
        response = client.messages.parse(
            model=HAIKU,
            max_tokens=1024,
            output_config={"effort": "low"},
            output_format=HitCheck,
            messages=[
                {
                    "role": "user",
                    "content": [
                        _image(frame["original"]),
                        _image(frame["overlay"]),
                        {"type": "text", "text": HIT_PROMPT.format(tag=frame["label"])},
                    ],
                }
            ],
        )
    except anthropic.BadRequestError:  # one frame the API won't accept shouldn't end the run
        return None, 0.0
    cost = cost_usd(HAIKU, response.usage)
    if response.stop_reason in ("refusal", "max_tokens"):
        return None, cost
    return response.parsed_output, cost


def run_haiku(client, which: str, root: Path = OUT_DIR) -> dict:
    """Haiku marks every outlined frame of a set it hasn't marked yet. Reasons are logged."""
    frames = [f for f in load_manifest(which, root) if f["reason"] == "subject"]
    done = load_marks(which, root, who="haiku")
    log = set_dir(which, root) / "haiku_log.jsonl"
    cost, failed = 0.0, 0
    for f in frames:
        if str(f["image_id"]) in done:
            continue
        answer, c = haiku_mark(client, f)
        cost += c
        if answer is None:
            failed += 1
            continue
        save_mark(which, f["image_id"], answer.mark, root, who="haiku")
        with log.open("a") as out:
            out.write(json.dumps({"image_id": f["image_id"], **answer.model_dump()}) + "\n")
    return {"cost_usd": cost, "failed": failed}


def export(which: str, root: Path = OUT_DIR, per_category: int = PER_CATEGORY) -> Path:
    from tagger import classify, color_sets, locate, store

    out = set_dir(which, root)
    marked_before = load_marks(which, root) or (out / "round1").exists()
    if which != "A" and marked_before:
        raise FileExistsError(f"set {which} is already being marked; it's scored once")
    conn = store.connect()
    rows = color_sets.draw(
        color_sets.candidates(conn, classify.ACTIVE_MODEL_ID, SOURCE_SET[which]),
        per_category,
        min_confidence=locate.MIN_CONFIDENCE,
    )
    locator = locate.Locator(classify.Ensemble(classify.ACTIVE_CHECKPOINTS).members)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for r in rows:
        with Image.open(r["file_path"]) as img:
            img = img.convert("RGB")
        found = locator.locate(img, r["label"], r["confidence"])
        rgb = np.asarray(img)
        size = (DISPLAY_WIDTH, round(rgb.shape[0] * DISPLAY_WIDTH / rgb.shape[1]))
        names = {"original": f"{r['image_id']}.jpg", "overlay": f"{r['image_id']}_cam.jpg"}
        Image.fromarray(rgb).resize(size, Image.LANCZOS).save(out / names["original"], quality=90)
        Image.fromarray(locate.overlay(rgb, found)).resize(size, Image.LANCZOS).save(
            out / names["overlay"], quality=90
        )
        manifest.append(
            {
                "image_id": r["image_id"],
                "trip": r["trip"],
                "source": r["source"],
                "label": r["label"],
                "confidence": round(r["confidence"], 3),
                "reason": found.reason,
                **{k: str(out / v) for k, v in names.items()},
            }
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a frame set for the Grad-CAM hit check.")
    parser.add_argument("--set", choices=tuple(SOURCE_SET), required=True)
    parser.add_argument("--haiku", action="store_true", help="Haiku marks an exported set.")
    args = parser.parse_args(argv)
    if args.haiku:
        import anthropic
        from dotenv import load_dotenv

        load_dotenv()  # the API key comes from .env, never the shell
        result = run_haiku(anthropic.Anthropic(), args.set)
        print(f"Haiku: ${result['cost_usd']:.4f}, {result['failed']} frames unanswered")
        if args.set == "B":
            print(
                json.dumps(
                    agreement(load_manifest("B"), load_marks("B"), load_marks("B", who="haiku")),
                    indent=1,
                )
            )
        return
    per = FINAL_PER_CATEGORY if args.set == "final" else PER_CATEGORY
    out = export(args.set, per_category=per)
    frames = load_manifest(args.set)
    reasons = Counter(f["reason"] for f in frames)
    print(f"{len(frames)} frames -> {out} ({dict(reasons)})")


if __name__ == "__main__":
    main()
