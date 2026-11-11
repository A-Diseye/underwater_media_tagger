"""The fixed collection loop shared by the diver and scenery collectors.

A workflow, not an agent: where the photos come from is decided in code (a list of sources taken
in turn), so no model chooses what to do next. Claude Haiku only judges each photo, with the same
check the iNaturalist agent uses.

Rules on top of the check:
- At most UPLOADER_SHARE of a category from one photographer or uploader, so one photo shoot
  can't fill it.
- Optional caps per source group (e.g. coral close-ups), so one kind of photo can't dominate.
- Train/val split by photographer, so one person's photos never land on both sides.
- Its own log and budget per category, so collectors can run at the same time and each budget
  still holds.
"""

from __future__ import annotations

import io
import json
import math
import urllib.error
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from training.collect_inat import PhotoCheck, check_photo, rejection, split_for

UPLOADER_SHARE = 0.15
# Skips in a row that mean nothing checkable is left (e.g. every source but a capped one has run
# out), so the loop doesn't page through thousands of photos it will never use.
MAX_SKIPS_IN_A_ROW = 500


@dataclass
class Candidate:
    key: str  # unique across sources, e.g. "commons:123" or "inat:456"
    source: str
    group: str  # for group caps and the summary
    uploader: int | str  # photographer, or uploader when that's all we know
    url: str
    fetch: Callable[[], bytes]
    check: PhotoCheck | None = None  # already known from an earlier check: no need to pay again


def interleave(iterables: Iterable[Iterable]) -> Iterator:
    """One item from each source in turn, lazily, until all run out."""
    iterators = [iter(i) for i in iterables]
    while iterators:
        for it in list(iterators):
            try:
                yield next(it)
            except StopIteration:
                iterators.remove(it)


def load_log(log_path: Path) -> tuple[set[str], Counter, Counter, float]:
    """(keys already decided, kept per uploader, kept per group, dollars spent)."""
    seen, uploaders, groups, spent = set(), Counter(), Counter(), 0.0
    if log_path.exists():
        with open(log_path) as f:
            for event in map(json.loads, f):
                seen.add(event["key"])
                spent += event["cost_usd"]
                if event["reason"] is None:
                    uploaders[event["uploader_key"]] += 1
                    groups[event["group"]] += 1
    return seen, uploaders, groups, spent


def collect(
    client,
    candidates: Iterable[Candidate],
    category: str,
    data_dir: Path,
    target: int,
    budget: float,
    group_shares: dict[str, float] | None = None,
) -> dict:
    """Check candidates in order until `target` are kept or `budget` is spent."""
    log_path = data_dir / "curation" / f"{category}_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    seen, uploaders, groups, spent = load_log(log_path)
    uploader_cap = math.ceil(target * UPLOADER_SHARE)
    group_caps = {g: math.ceil(target * share) for g, share in (group_shares or {}).items()}
    rejected, skipped = Counter(), Counter()
    skips_in_a_row = 0

    def kept() -> int:
        return sum(groups.values())

    for candidate in candidates:
        if kept() >= target or spent >= budget or skips_in_a_row >= MAX_SKIPS_IN_A_ROW:
            break
        uploader_key = f"{candidate.source}:{candidate.uploader}"
        if candidate.key in seen:
            skip = "already_seen"
        elif uploaders[uploader_key] >= uploader_cap:
            skip = "uploader_cap"
        elif groups[candidate.group] >= group_caps.get(candidate.group, target):
            skip = "group_cap"
        else:
            skip = None
        if skip:
            skipped[skip] += 1
            skips_in_a_row += 1
            continue
        skips_in_a_row = 0

        event = {
            "key": candidate.key,
            "source": candidate.source,
            "group": candidate.group,
            "uploader_key": uploader_key,
            "photo_url": candidate.url,
            "category": category,
            "reason": None,
            "check": None,
            "path": None,
            "cost_usd": 0.0,
        }
        try:
            data = candidate.fetch()
            is_jpeg = Image.open(io.BytesIO(data)).format == "JPEG"
        except (urllib.error.URLError, TimeoutError, UnidentifiedImageError):
            data, is_jpeg = None, False
        if not is_jpeg:
            event["reason"] = "unreadable"
        else:
            check = candidate.check
            if check is None:
                check, event["cost_usd"] = check_photo(client, data)
                spent += event["cost_usd"]
            if check is None:
                event["reason"] = "check_failed"
            else:
                event["check"] = check.model_dump()
                event["reason"] = rejection(check, category)
        if event["reason"] is None:
            split = split_for(candidate.uploader)
            name = candidate.key.replace(":", "_")
            path = data_dir / split / category / f"{name}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            event["split"], event["path"] = split, str(path)
            uploaders[uploader_key] += 1
            groups[candidate.group] += 1
        else:
            rejected[event["reason"]] += 1
        seen.add(candidate.key)
        with open(log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        if len(seen) % 10 == 0:
            print(f"  {category}: kept {kept()}/{target}, spent ${spent:.3f}", flush=True)

    return {
        "category": category,
        "kept": kept(),
        "target": target,
        "kept_by_group": dict(groups),
        "uploaders": len(uploaders),
        "rejected": dict(rejected),
        "skipped": dict(skipped),
        "spent_usd": round(spent, 3),
    }
