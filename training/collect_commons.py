"""Collect diver training photos from Wikimedia Commons.

A workflow, not an agent: the steps are fixed in code. There's one category to fill from a known
list of Commons categories, so there's nothing for a model to decide about what to do next. Code
takes photos round-robin across the categories (so no one country dominates), Claude Haiku checks
each one with the same prompt and rules as the iNaturalist agent, and good ones are saved.

Rules on top of the check:
- At most UPLOADER_SHARE of the photos from one uploader, so one photo shoot (many Commons diver
  photos are from a few US Navy photographers) can't fill the category.
- Train/val split by uploader, so one shoot never lands on both sides.
- Its own log and budget, separate from the iNaturalist agent's, so the two can run at the same
  time and each budget still holds.

Run with:  uv run python -m training.collect_commons   (needs ANTHROPIC_API_KEY in .env)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from training.collect_inat import check_photo, rejection, split_for

CATEGORY = "diver"
TARGET = 100
BUDGET_USD = 1.00
UPLOADER_SHARE = 0.15
BATCH = 50  # files per metadata request

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks automated clients to name themselves and give a contact in the User-Agent.
USER_AGENT = "underwater-media-tagger/0.1 (https://github.com/A-Diseye/underwater_media_tagger)"
PAUSE_S = 1.0
# This category's subcategories are per-country underwater diver photos.
ROOT = "Underwater photographs of scuba divers"
EXTRA_CATEGORIES = (
    "Scuba diving",
    "Night diving",
    "Technical diving",
    "Sidemount diving",
    "Rebreather diving",
    "Women scuba diving",
)
SKIP_CATEGORIES = {"Water clarity"}


class Commons:
    """Read-only Wikimedia Commons client that paces itself and backs off when rate limited."""

    def _fetch(self, url: str) -> bytes:
        for attempt in range(6):
            time.sleep(PAUSE_S)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=30) as response:
                    return response.read()
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
                wait = int(e.headers.get("Retry-After") or 10 * (attempt + 1))
                print(f"  rate limited, waiting {wait}s", flush=True)
                time.sleep(wait)
        raise TimeoutError("Wikimedia kept rate limiting")

    def _query(self, **params) -> Iterator[dict]:
        """Run an API query, following continuation, and yield each response."""
        params = {"action": "query", "format": "json", **params}
        while True:
            data = json.loads(self._fetch(f"{API}?{urllib.parse.urlencode(params)}"))
            yield data
            if "continue" not in data:
                return
            params.update(data["continue"])

    def members(self, category: str, kind: str) -> list[str]:
        """Titles of the files ("file") or subcategories ("subcat") in a category."""
        titles = []
        for data in self._query(
            list="categorymembers", cmtitle=f"Category:{category}", cmtype=kind, cmlimit=500
        ):
            titles += [m["title"] for m in data["query"]["categorymembers"]]
        return titles

    def image_info(self, titles: list[str]) -> dict[str, dict]:
        """Title -> {pageid, mime, user, url} with a 500 px thumbnail URL."""
        info = {}
        for data in self._query(
            titles="|".join(titles), prop="imageinfo", iiprop="url|mime|user", iiurlwidth=500
        ):
            for page in data["query"]["pages"].values():
                if page.get("imageinfo"):
                    ii = page["imageinfo"][0]
                    info[page["title"]] = {
                        "pageid": page["pageid"],
                        "mime": ii["mime"],
                        "user": ii.get("user", "unknown"),
                        "url": ii.get("thumburl") or ii["url"],
                    }
        return info

    def download(self, url: str) -> bytes:
        return self._fetch(url)


def source_categories(commons) -> list[str]:
    subcats = [s.removeprefix("Category:") for s in commons.members(ROOT, "subcat")]
    return [c for c in [ROOT, *subcats, *EXTRA_CATEGORIES] if c not in SKIP_CATEGORIES]


def round_robin(queues: Iterable[list[str]]) -> Iterator[str]:
    """One title from each queue in turn, skipping titles already given (files can sit in
    several categories)."""
    queues, given = [list(q) for q in queues], set()
    while any(queues):
        for queue in queues:
            while queue:
                title = queue.pop(0)
                if title not in given:
                    given.add(title)
                    yield title
                    break


def batched(items: Iterable[str], size: int) -> Iterator[list[str]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_log(log_path: Path) -> tuple[set[str], Counter, float]:
    """(titles already decided, photos kept per uploader, dollars spent) from earlier runs."""
    seen, uploaders, spent = set(), Counter(), 0.0
    if log_path.exists():
        with open(log_path) as f:
            for event in map(json.loads, f):
                seen.add(event["title"])
                spent += event["cost_usd"]
                if event["reason"] is None:
                    uploaders[event["uploader"]] += 1
    return seen, uploaders, spent


def collect(
    client, commons, data_dir: Path, target: int = TARGET, budget: float = BUDGET_USD, seed=0
) -> dict:
    """Fill the diver category. Returns a summary of what happened."""
    log_path = data_dir / "curation" / "commons_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    seen, uploaders, spent = load_log(log_path)
    uploader_cap = math.ceil(target * UPLOADER_SHARE)
    rejected, skipped = Counter(), Counter()

    rng = random.Random(seed)
    queues = []
    for category in source_categories(commons):
        titles = [t for t in commons.members(category, "file") if t not in seen]
        rng.shuffle(titles)
        queues.append(titles)

    def finished() -> bool:
        return sum(uploaders.values()) >= target or spent >= budget

    for titles in batched(round_robin(queues), BATCH):
        if finished():
            break
        info = commons.image_info(titles)
        for title in titles:
            if finished():
                break
            meta = info.get(title)
            if meta is None or meta["mime"] != "image/jpeg":  # diagrams, logos, icons, videos
                skipped["not_a_photo"] += 1
                continue
            if uploaders[meta["user"]] >= uploader_cap:
                skipped["uploader_cap"] += 1
                continue
            event = {
                "title": title,
                "uploader": meta["user"],
                "photo_url": meta["url"],
                "category": CATEGORY,
                "reason": None,
                "check": None,
                "path": None,
                "cost_usd": 0.0,
            }
            try:
                data = commons.download(meta["url"])
                is_jpeg = Image.open(io.BytesIO(data)).format == "JPEG"
            except (urllib.error.URLError, TimeoutError, UnidentifiedImageError):
                data, is_jpeg = None, False
            if not is_jpeg:
                event["reason"] = "unreadable"
            else:
                check, event["cost_usd"] = check_photo(client, data)
                spent += event["cost_usd"]
                if check is None:
                    event["reason"] = "check_failed"
                else:
                    event["check"] = check.model_dump()
                    event["reason"] = rejection(check, CATEGORY)
            if event["reason"] is None:
                split = split_for(meta["user"])
                path = data_dir / split / CATEGORY / f"commons_{meta['pageid']}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                event["split"], event["path"] = split, str(path)
                uploaders[meta["user"]] += 1
            else:
                rejected[event["reason"]] += 1
            with open(log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        print(f"  kept {sum(uploaders.values())}/{target}, spent ${spent:.3f}", flush=True)

    return {
        "kept": sum(uploaders.values()),
        "target": target,
        "uploaders": len(uploaders),
        "rejected": dict(rejected),
        "skipped": dict(skipped),
        "spent_usd": round(spent, 3),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--budget", type=float, default=BUDGET_USD, help="Hard stop, in dollars.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    load_dotenv()  # loaded here only, so the key never reaches the shell that runs Claude Code
    summary = collect(anthropic.Anthropic(), Commons(), args.data_dir, args.target, args.budget)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
