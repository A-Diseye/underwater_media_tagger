"""Blind A/B test: does subject-aware color correction look better than whole-frame correction?
(docs/color_plan.md, Stage 4)

There's no ground truth for "correct" underwater colors, so I judge, blind. `export` draws frames
from set C (untouched by tuning) where subject mode applies, corrects each both ways with the
locked defaults, and saves the two versions as left and right in a random order. The Color test
page shows the pairs; I pick left, right, or no difference, and can flag a frame whose tag is
wrong ("no turtle here"). Nothing is revealed until every frame has a vote.

The rule, set before the test: subject-aware counts as better if it wins at least 60% of the
votes that aren't ties. The README reports the win rate with its uncertainty, per category, and
says that it covers stills only (in video, subject mode would flicker more).

Agent B (Sonnet) is a second judge: it sees the same pairs, in its own random order, only after
all my votes are in, and its agreement with me is reported. It never replaces my votes.

These are unlabeled library frames, not test-set frames, and nobody labels a category here.

Round 2 ("pool"): round 1 failed the rule, and the cause was found after unblinding (water
detection mistook sand for water). The fix was tuned on set A, then tested once more on fresh
pool frames, leaving out the 25 I marked for Grad-CAM. Round 1 stays the main result.

Run with:  uv run python -m training.color_test --export     (draw and correct the frames)
           uv run python -m training.color_test --sonnet     (after my votes: Sonnet judges)
           uv run python -m training.color_test              (results so far)
           add --round pool for round 2
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

OUT_DIR = Path("data/color")
PER_CATEGORY = 6
SEED = 23
WIN_BAR = 0.6  # set before the test: subject-aware must win this share of non-tie votes
CHOICES = ("left", "right", "same")
JUDGE_WIDTH = 1280  # Sonnet sees the pairs at this width (cost; detail is still clear)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Each round: the frame set it draws from, and frames to leave out (seen in an earlier check).
ROUNDS = {"C": ("C", ()), "pool": ("pool", ("final",))}

JUDGE_PROMPT = """You are judging color correction of underwater photos. The two images are \
the same photo, color-corrected in two different ways. The main subject is a {tag}.

Which version would a skilled underwater photographer prefer? Judge the whole image: natural \
colors on the subject, water that still looks like water rather than gray or purple, no odd \
color casts, glowing patches, or halos. Neither version is the original.

Answer "first" or "second" for the image you prefer, or "same" if they look essentially the \
same or neither is clearly better. Give a one-sentence reason."""


class Pick(BaseModel):
    choice: Literal["first", "second", "same"]
    reason: str


def test_dir(root: Path = OUT_DIR, which: str = "C") -> Path:
    return root / f"test_{which}"


def exported_rounds(root: Path = OUT_DIR) -> list[str]:
    return [w for w in ROUNDS if (test_dir(root, w) / "manifest.json").exists()]


def load_manifest(root: Path = OUT_DIR, which: str = "C") -> list[dict]:
    path = test_dir(root, which) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else []


def load_votes(root: Path = OUT_DIR, who: str = "me", which: str = "C") -> dict[str, dict]:
    """{image id (str): {"winner": "subject" | "whole" | "same", "no_subject": bool}}."""
    path = test_dir(root, which) / f"votes_{who}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def winner(frame: dict, choice: str) -> str:
    """Map a left/right/same choice back to the method, using the frame's hidden order."""
    if choice not in CHOICES:
        raise ValueError(f"unknown choice: {choice!r}")
    if choice == "same":
        return "same"
    other = "whole" if frame["left"] == "subject" else "subject"
    return frame["left"] if choice == "left" else other


def save_vote(
    frame: dict,
    choice: str,
    no_subject: bool = False,
    root: Path = OUT_DIR,
    who: str = "me",
    which: str = "C",
) -> None:
    votes = load_votes(root, who, which)
    votes[str(frame["image_id"])] = {"winner": winner(frame, choice), "no_subject": no_subject}
    (test_dir(root, which) / f"votes_{who}.json").write_text(json.dumps(votes, indent=1))


def undo_vote(image_id: int, root: Path = OUT_DIR, who: str = "me", which: str = "C") -> None:
    votes = load_votes(root, who, which)
    votes.pop(str(image_id), None)
    (test_dir(root, which) / f"votes_{who}.json").write_text(json.dumps(votes, indent=1))


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a win rate; behaves sensibly for small n."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def p_value(wins: int, n: int) -> float:
    """One-sided: the chance of at least `wins` out of `n` if both methods were equally good."""
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n if n else 1.0


def _tally(frames: list[dict], votes: dict) -> dict:
    counts = {"subject": 0, "whole": 0, "same": 0}
    for f in frames:
        counts[votes[str(f["image_id"])]["winner"]] += 1
    decided = counts["subject"] + counts["whole"]
    return {
        **counts,
        "win_rate": counts["subject"] / decided if decided else None,
        "interval": wilson(counts["subject"], decided),
        "p_value": p_value(counts["subject"], decided),
    }


def results(manifest: list[dict], votes: dict) -> dict:
    """Win rates over frames with a real subject; frames flagged "no subject here" (a wrong tag
    that passed the gate) are counted separately."""
    voted = [f for f in manifest if str(f["image_id"]) in votes]
    real = [f for f in voted if not votes[str(f["image_id"])]["no_subject"]]
    wrong = [f for f in voted if votes[str(f["image_id"])]["no_subject"]]
    overall = _tally(real, votes)
    by_category = {}
    for f in real:
        by_category.setdefault(f["label"], []).append(f)
    return {
        "frames": len(manifest),
        "voted": len(voted),
        "complete": bool(manifest) and len(voted) == len(manifest),
        "overall": overall,
        "passed": overall["win_rate"] is not None and overall["win_rate"] >= WIN_BAR,
        "by_category": {k: _tally(v, votes) for k, v in sorted(by_category.items())},
        "wrong_tag": _tally(wrong, votes),
    }


def agreement(manifest: list[dict], mine: dict, other: dict) -> dict:
    both = [str(f["image_id"]) for f in manifest if str(f["image_id"]) in mine]
    both = [k for k in both if k in other]
    same = sum(mine[k]["winner"] == other[k]["winner"] for k in both)
    return {"frames": len(both), "same_winner": same / len(both) if both else None}


def export(root: Path = OUT_DIR, per_category: int = PER_CATEGORY, which: str = "C") -> Path:
    """Draw frames where subject mode applies and save both corrections, left and right in a
    random (seeded) order. Refused once voting has started: each round is run once."""
    from tagger import classify, color, color_sets, locate, store
    from training import color_check

    out = test_dir(root, which)
    if load_votes(root, which=which):
        raise FileExistsError("voting has started; the blind test is run once")
    source, skip_checks = ROUNDS[which]
    seen = {f["image_id"] for c in skip_checks for f in color_check.load_manifest(c, root)}
    conn = store.connect()
    candidates = color_sets.candidates(conn, classify.ACTIVE_MODEL_ID, source)
    # Draw extra, since frames without a clear subject (whole-frame fallback) are skipped.
    rows = color_sets.draw(
        [r for r in candidates if r["image_id"] not in seen],
        per_category * 3,
        min_confidence=locate.MIN_CONFIDENCE,
    )
    locator = locate.Locator(classify.Ensemble(classify.ACTIVE_CHECKPOINTS).members)
    out.mkdir(parents=True, exist_ok=True)
    manifest, taken = [], {}
    for r in rows:
        if taken.get(r["label"], 0) >= per_category:
            continue
        with Image.open(r["file_path"]) as img:
            img = img.convert("RGB")
        found = locator.locate(img, r["label"], r["confidence"])
        if not found.has_subject:
            continue
        rgb = np.asarray(img)
        settings = color.estimate(rgb, found.mask, color.SUBJECT)
        views = {
            "subject": color.apply(rgb, found.mask, settings),
            "whole": color.correct(rgb, mode=color.WHOLE),
        }
        coin = random.Random(f"{SEED}-{which}-{r['image_id']}").random()
        left = "subject" if coin < 0.5 else "whole"
        right = "whole" if left == "subject" else "subject"
        paths = {}
        for side, method in (("left", left), ("right", right)):
            paths[side] = str(out / f"{r['image_id']}_{side}.jpg")
            Image.fromarray(views[method]).save(paths[side], quality=92)
        difference = np.abs(views["subject"].astype(int) - views["whole"]).mean()
        manifest.append(
            {
                "image_id": r["image_id"],
                "trip": r["trip"],
                "source": r["source"],
                "label": r["label"],
                "confidence": round(r["confidence"], 3),
                "left": left,  # hidden from the page until every vote is in
                "shallow_amount": round(settings.amount, 2),
                "mean_difference": round(float(difference), 1),  # 0-255 scale
                "left_path": paths["left"],
                "right_path": paths["right"],
            }
        )
        taken[r["label"]] = taken.get(r["label"], 0) + 1
    random.Random(SEED).shuffle(manifest)  # categories interleaved, same order every time
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def _judge_image(path: str) -> dict:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((JUDGE_WIDTH, round(img.height * JUDGE_WIDTH / img.width)), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
    data = base64.standard_b64encode(buffer.getvalue()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def sonnet_pick(client, frame: dict) -> tuple[dict | None, float]:
    """Sonnet judges one pair, in its own random order. Returns ({"winner", "reason"} or None if
    it couldn't answer, cost in USD)."""
    import anthropic

    from tagger.pricing import SONNET, cost_usd

    flip = random.Random(f"sonnet-{SEED}-{frame['image_id']}").random() < 0.5
    first, second = ("right", "left") if flip else ("left", "right")
    try:
        response = client.beta.messages.parse(
            model=SONNET,
            max_tokens=8000,
            betas=[FALLBACK_BETA],
            fallbacks="default",  # a refused request is retried on a fallback model
            output_config={"effort": "medium"},
            output_format=Pick,
            messages=[
                {
                    "role": "user",
                    "content": [
                        _judge_image(frame[f"{first}_path"]),
                        _judge_image(frame[f"{second}_path"]),
                        {"type": "text", "text": JUDGE_PROMPT.format(tag=frame["label"])},
                    ],
                }
            ],
        )
    except anthropic.BadRequestError:  # one pair the API won't accept shouldn't end the run
        return None, 0.0
    cost = cost_usd(SONNET, response.usage)
    if response.stop_reason in ("refusal", "max_tokens") or response.parsed_output is None:
        return None, cost
    pick = response.parsed_output
    side = {"first": first, "second": second, "same": "same"}[pick.choice]
    return {"winner": winner(frame, side), "reason": pick.reason}, cost


def run_sonnet(client, root: Path = OUT_DIR, which: str = "C") -> dict:
    """Agent B judges every pair it hasn't judged yet; only after all my votes are in."""
    manifest = load_manifest(root, which)
    if not results(manifest, load_votes(root, which=which))["complete"]:
        raise RuntimeError("Sonnet judges only after every frame has my vote")
    done = load_votes(root, who="sonnet", which=which)
    log = test_dir(root, which) / "sonnet_log.jsonl"
    cost, failed = 0.0, 0
    for f in manifest:
        key = str(f["image_id"])
        if key in done:
            continue
        answer, c = sonnet_pick(client, f)
        cost += c
        if answer is None:
            failed += 1
            continue
        done[key] = {"winner": answer["winner"], "no_subject": False}
        (test_dir(root, which) / "votes_sonnet.json").write_text(json.dumps(done, indent=1))
        with log.open("a") as out:
            out.write(json.dumps({"image_id": f["image_id"], **answer}) + "\n")
    return {"cost_usd": cost, "failed": failed}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Blind A/B test of the two color corrections.")
    parser.add_argument("--export", action="store_true", help="Draw and correct set C frames.")
    parser.add_argument("--sonnet", action="store_true", help="Sonnet judges (after my votes).")
    parser.add_argument("--round", choices=tuple(ROUNDS), default="C", help="C or pool (round 2).")
    args = parser.parse_args(argv)
    which = args.round
    if args.export:
        out = export(which=which)
        frames = load_manifest(which=which)
        by = {}
        for f in frames:
            by[f["label"]] = by.get(f["label"], 0) + 1
        print(f"{len(frames)} pairs -> {out} {by}")
        return
    manifest, mine = load_manifest(which=which), load_votes(which=which)
    if args.sonnet:
        import anthropic
        from dotenv import load_dotenv

        load_dotenv()  # the API key comes from .env, never the shell
        result = run_sonnet(anthropic.Anthropic(), which=which)
        print(f"Sonnet: ${result['cost_usd']:.4f}, {result['failed']} pairs unanswered")
    r = results(manifest, mine)
    if not r["complete"]:
        print(f"{r['voted']}/{r['frames']} voted; results stay hidden until all are in")
        return
    print(json.dumps(r, indent=1))
    sonnet = load_votes(who="sonnet", which=which)
    if sonnet:
        print("Sonnet agreement:", agreement(manifest, mine, sonnet))


if __name__ == "__main__":
    main()
