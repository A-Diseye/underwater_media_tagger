"""Collect scenery training photos: underwater scenes with no main animal or diver.

A workflow, not an agent (see training/workflow.py). There's no taxonomy for "scenery", so the
label comes from the source plus Claude Haiku's check: underwater, category scenery, and no
animal as the main subject. Three sources, taken in turn:
1. Wikimedia Commons underwater photo categories: wide reef, seagrass, and wreck scenes, closest
   to how scenery looks in my footage.
2. iNaturalist corals, sponges, and octocorals: plentiful but mostly close-ups, so capped at
   CLOSEUP_SHARE of the category.
3. Photos the iNaturalist agent already rejected because no animal was the main subject (a reef
   with a tiny fish). Labeling rule 3 calls those scenery. Haiku already judged them, so they
   cost nothing to check again.

Run with:  uv run python -m training.collect_scenery   (needs ANTHROPIC_API_KEY in .env)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from training.collect_commons import Commons, commons_candidates
from training.collect_inat import INat, PhotoCheck
from training.workflow import Candidate, collect, interleave

TARGET = 100
BUDGET_USD = 1.00
CLOSEUP_SHARE = 0.30
REUSED_SHARE = 0.40

COMMONS_GROUP = "Commons underwater scenes"
CLOSEUP_GROUP = "iNat corals and sponges"
REUSED_GROUP = "iNat photos with no main animal"

COMMONS_ROOT = "Underwater photographs"
# Wide scenes. Skipped: deep-sea robot footage, half-above-water shots, and optics demos.
COMMONS_SUBCATS = (
    "Underwater photographs of nature",
    "Underwater photos of Adriatic sea",
    "Underwater photographs of the Red Sea",
    "Underwater photographs of man-made objects",
    "Quality images taken underwater",
)
COMMONS_BY_COUNTRY = "Underwater photographs by country"
COMMONS_EXTRAS = ("Coral reefs",)
# iNaturalist taxon ids: sponges, stony corals, octocorals.
CLOSEUP_TAXA = (48824, 47532, 340493)


def scenery_categories(commons) -> list[str]:
    countries = commons.subcategories(COMMONS_BY_COUNTRY)
    return [COMMONS_ROOT, *COMMONS_SUBCATS, *countries, *COMMONS_EXTRAS]


def inat_taxon(inat, taxon_id: int) -> Iterator[Candidate]:
    id_below = None
    while True:
        observations = inat.observations(taxon_id, id_below)
        if not observations:
            return
        id_below = observations[-1]["id"]
        for o in observations:
            if o.get("photos"):
                url = o["photos"][0]["url"]
                yield Candidate(
                    key=f"inat:{o['id']}",
                    source="inaturalist",
                    group=CLOSEUP_GROUP,
                    uploader=o["user"]["id"],
                    url=url.replace("/square.", "/medium."),
                    fetch=partial(inat.photo, url),
                )


def reused_rejects(agent_log: Path, inat) -> list[Candidate]:
    """Photos the animal agent rejected that Haiku saw as underwater scenery."""
    if not agent_log.exists():
        return []
    candidates = []
    with open(agent_log) as f:
        for event in map(json.loads, f):
            check = event.get("check")
            if (
                event["event"] == "photo"
                and check
                and check["underwater"]
                and check["category"] == "scenery"
                and not check["animal_is_main_subject"]
            ):
                candidates.append(
                    Candidate(
                        key=f"inat:{event['observation_id']}",
                        source="inaturalist",
                        group=REUSED_GROUP,
                        uploader=event["observer_id"],
                        url=event["photo_url"],
                        fetch=partial(inat.photo, event["photo_url"]),
                        check=PhotoCheck(**check),
                    )
                )
    return candidates


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--budget", type=float, default=BUDGET_USD, help="Hard stop, in dollars.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    load_dotenv()  # loaded here only, so the key never reaches the shell that runs Claude Code
    commons, inat = Commons(), INat()
    reused = reused_rejects(args.data_dir / "curation" / "log.jsonl", inat)
    print(f"{len(reused)} reusable photos from the animal agent's log", flush=True)
    candidates = interleave(
        [
            commons_candidates(commons, scenery_categories(commons), COMMONS_GROUP),
            interleave(inat_taxon(inat, t) for t in CLOSEUP_TAXA),
            reused,
        ]
    )
    summary = collect(
        anthropic.Anthropic(),
        candidates,
        "scenery",
        args.data_dir,
        args.target,
        args.budget,
        group_shares={CLOSEUP_GROUP: CLOSEUP_SHARE, REUSED_GROUP: REUSED_SHARE},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
