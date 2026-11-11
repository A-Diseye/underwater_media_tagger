"""Collect diver training photos from Wikimedia Commons.

A workflow, not an agent (see training/workflow.py): there's one category to fill from a known
list of Commons categories, so there's nothing for a model to decide about where to look next.
Photos come round-robin from the categories, so no one country dominates. Many Commons diver
photos come from a few US Navy photographers, which the per-uploader cap keeps in check.

Run with:  uv run python -m training.collect_commons   (needs ANTHROPIC_API_KEY in .env)
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from functools import partial
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from training.workflow import Candidate, collect, interleave

TARGET = 100
BUDGET_USD = 1.00
BATCH = 50  # files per metadata request

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks automated clients to name themselves and give a contact in the User-Agent.
USER_AGENT = "underwater-media-tagger/0.1 (https://github.com/A-Diseye/underwater_media_tagger)"
PAUSE_S = 1.0
# This category's subcategories are per-country underwater diver photos.
DIVER_ROOT = "Underwater photographs of scuba divers"
DIVER_EXTRAS = (
    "Scuba diving",
    "Night diving",
    "Technical diving",
    "Sidemount diving",
    "Rebreather diving",
    "Women scuba diving",
)
DIVER_SKIP = {"Water clarity"}


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

    def subcategories(self, category: str) -> list[str]:
        return [s.removeprefix("Category:") for s in self.members(category, "subcat")]


def diver_categories(commons) -> list[str]:
    subcats = commons.subcategories(DIVER_ROOT)
    return [c for c in [DIVER_ROOT, *subcats, *DIVER_EXTRAS] if c not in DIVER_SKIP]


def batched(items: Iterable[str], size: int) -> Iterator[list[str]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def commons_candidates(commons, categories: list[str], group: str, seed=0) -> Iterator[Candidate]:
    """Photos from the categories in turn. Logos, diagrams, and videos are dropped here,
    from their file type, before anything is downloaded or checked."""
    rng = random.Random(seed)
    queues = []
    for category in categories:
        titles = commons.members(category, "file")
        rng.shuffle(titles)
        queues.append(titles)
    given = set()
    titles = (t for t in interleave(queues) if not (t in given or given.add(t)))
    for batch in batched(titles, BATCH):
        info = commons.image_info(batch)
        for title in batch:
            meta = info.get(title)
            if meta and meta["mime"] == "image/jpeg":
                yield Candidate(
                    key=f"commons:{meta['pageid']}",
                    source="commons",
                    group=group,
                    uploader=meta["user"],
                    url=meta["url"],
                    fetch=partial(commons.download, meta["url"]),
                )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--budget", type=float, default=BUDGET_USD, help="Hard stop, in dollars.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    load_dotenv()  # loaded here only, so the key never reaches the shell that runs Claude Code
    commons = Commons()
    candidates = commons_candidates(commons, diver_categories(commons), "Commons divers")
    summary = collect(
        anthropic.Anthropic(), candidates, "diver", args.data_dir, args.target, args.budget
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
