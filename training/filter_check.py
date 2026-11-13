"""Hand-check of the training-photo filter: was Haiku right to keep or reject each photo?

The README reports the filter's accuracy from this check. A fixed random sample of decisions
(half keeps, half rejects) comes from all three curation logs. Only decisions Haiku actually made
count: photos that failed to download or check are left out. My verdicts are appended to
data/curation/hand_check.jsonl, so the check can be done in several sittings.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

LOGS = ("log.jsonl", "diver_log.jsonl", "scenery_log.jsonl")
RESULTS = "hand_check.jsonl"
PER_SIDE = 50  # keeps and rejects each
SEED = 0


def decisions(curation_dir: Path) -> list[dict]:
    """Every keep/reject Haiku made, from all curation logs, in one shape."""
    found = []
    for name in LOGS:
        path = curation_dir / name
        if not path.exists():
            continue
        with open(path) as f:
            for event in map(json.loads, f):
                if "key" in event:
                    key = event["key"]
                elif event.get("event") == "photo":
                    key = f"inat:{event['observation_id']}"
                else:
                    continue  # agent turns and tool errors
                if event.get("check") is None:
                    continue  # never reached Haiku
                found.append(
                    {
                        "id": key,
                        "category": event["category"],
                        "kept": event["reason"] is None,
                        "reason": event["reason"],
                        "haiku_reason": event["check"]["reason"],
                        "image": event.get("path") or event["photo_url"],
                    }
                )
    return found


def sample(all_decisions: list[dict], per_side: int = PER_SIDE, seed: int = SEED) -> list[dict]:
    """A fixed sample: the same photos every time the page loads, keeps and rejects mixed."""
    rng = random.Random(seed)
    kept = [d for d in all_decisions if d["kept"]]
    rejected = [d for d in all_decisions if not d["kept"]]
    chosen = rng.sample(kept, min(per_side, len(kept)))
    chosen += rng.sample(rejected, min(per_side, len(rejected)))
    rng.shuffle(chosen)
    return chosen


def load_verdicts(curation_dir: Path) -> dict[str, str]:
    """Photo id -> "right" or "wrong". A later verdict on the same photo replaces an earlier one."""
    path = curation_dir / RESULTS
    if not path.exists():
        return {}
    with open(path) as f:
        return {v["id"]: v["verdict"] for v in map(json.loads, f)}


def save_verdict(curation_dir: Path, decision: dict, verdict: str) -> None:
    if verdict not in ("right", "wrong"):
        raise ValueError(f"unknown verdict: {verdict!r}")
    record = {
        "id": decision["id"],
        "category": decision["category"],
        "kept": decision["kept"],
        "verdict": verdict,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with open(curation_dir / RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")


def summary(checked: list[dict], verdicts: dict[str, str]) -> dict:
    """Accuracy of keeps and of rejects separately: they mean different things.

    A wrong keep puts a bad photo in the training set. A wrong reject only throws away a
    good photo (but many wrong rejects of one kind, like camouflaged fish, bias the data).
    """
    result = {}
    for side, is_kept in (("keeps", True), ("rejects", False)):
        done = [d for d in checked if d["kept"] == is_kept and d["id"] in verdicts]
        right = sum(verdicts[d["id"]] == "right" for d in done)
        result[side] = {"checked": len(done), "right": right}
    result["wrong_by_category"] = dict(
        Counter(d["category"] for d in checked if verdicts.get(d["id"]) == "wrong")
    )
    return result
