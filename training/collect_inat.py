"""Curation agent: build the animal training set from iNaturalist photos.

Three layers, each doing only what it's good at:
- Plain code owns the rules: taxonomy -> category, one photo per observation, the per-species
  cap, the train/val split, the log, and the budget.
- Claude Haiku checks each photo: underwater? animal the main subject? which category? It never
  sees iNaturalist's label, so a match means two independent sources agree. It can only reject
  a photo. The training label always comes from the taxonomy.
- Claude Sonnet is the agent: it reads the progress and rejection reasons and decides which
  species to search next.

Only internet photos pass through here, and they only go to data/train and data/val. The test
set is never read or written. Every decision is appended to data/curation/log.jsonl, which also
makes re-runs resume where they stopped, without paying for the same photo twice.

Run with:  uv run python -m training.collect_inat   (needs ANTHROPIC_API_KEY in .env)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

CHECK_MODEL = "claude-haiku-5-5"
AGENT_MODEL = "claude-sonnet-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Dollars per million tokens: (input, output, cache write, cache read).
PRICES = {
    CHECK_MODEL: (0.10, 0.50, 0.125, 0.01),
    AGENT_MODEL: (2.00, 10.00, 2.50, 0.20),
}

ANIMAL_CATEGORIES = ("turtles", "rays", "fish", "invertebrates")
TARGET = 100
BUDGET_USD = 5.00
# Wide runs (--wide): extra medium/wide training photos, plus a wide validation set.
WIDE_TARGET = 50
WIDE_VAL_TARGET = 30
WIDE_BUDGET_USD = 2.00
MAX_TURNS = 40
MAX_BATCH = 40  # photos checked per collect_from_taxon call
MAX_PAGES = 5  # observation pages read per call, so capped species can't page forever
PAGE_SIZE = 100
VAL_EVERY = 5  # 1 in 5 photographers goes to validation
# No species may dominate a category. Turtles get more room: there are only 7 species.
SPECIES_SHARE = {"turtles": 0.35, "rays": 0.15, "fish": 0.15, "invertebrates": 0.15}
# No taxon the agent collects from may supply more than this share of a category either, so a
# category can't fill up from one group (all nudibranchs, all butterflyfish).
GROUP_SHARE = 0.25

INAT_API = "https://api.inaturalist.org/v1"
USER_AGENT = "underwater-media-tagger/0.1"
INAT_PAUSE_S = 1.0  # iNaturalist asks API clients to stay near one request per second

# iNaturalist taxon ids. First match wins, so rays must come before the shark-and-ray group.
CATEGORY_TAXA = (
    (372234, "turtles"),  # Chelonioidea: sea turtles
    (448201, "rays"),  # Batoidea
    (47273, "fish"),  # Elasmobranchii: what's left once rays are taken is sharks
    (47178, "fish"),  # Actinopterygii: ray-finned fish, including eels
    (47115, "invertebrates"),  # Mollusca: nudibranchs, octopus, squid, clams
    (47187, "invertebrates"),  # Malacostraca: crabs, shrimp, lobsters
    (47549, "invertebrates"),  # Echinodermata: sea stars, urchins
)

CHECK_PROMPT = """You are screening photos for a training set of underwater images.
Answer only from what you can see in the photo.

underwater: true only if the photo was taken under water in a natural setting. False for land, \
the water's surface, boats, an animal held in a hand or on a dock, aquariums, and dead or \
preserved specimens.
animal_is_main_subject: true if an animal, or a school or group of animals, is clearly what the \
photo is about. False if it is tiny or incidental in the scene. Coral, sponges, sea fans, and \
anemones on their own are part of the reef, not animals, for this question.
category: the most prominent subject. If an animal is clearly a subject, choose the animal's \
category even when a person is also in the photo.
- turtles: sea turtles
- rays: stingrays, eagle rays, manta rays
- fish: all other fish, including sharks and eels
- invertebrates: crabs, shrimp, lobsters, nudibranchs, sea stars, octopus, squid, cuttlefish, \
clams and other mollusks
- diver: a person, when no animal is clearly a subject
- scenery: coral, sponges, sea fans, a lone anemone, reef, sand, wrecks, or open water, with \
no main animal or diver
- other: anything else
reason: one short sentence explaining your answers."""

AGENT_SYSTEM = """You are collecting training photos for an image classifier that tags underwater \
photos taken by scuba divers. The goal is to reach the target number of kept photos in each \
category (turtles, rays, fish, invertebrates), with as many different species as possible, \
within the budget.

Each collected photo is checked automatically, and the result comes back as counts of kept and \
rejected photos with reasons. Each species has a cap, and each taxon you collect from can \
supply at most a quarter of a category; past that, its photos are skipped as species_cap or \
group_cap. Use these results to decide what to do next:
- Favor marine animals that divers actually photograph on reefs and wrecks. Search specific \
groups (a family or a species) rather than whole classes, which are full of freshwater fish and \
land animals.
- If a taxon's photos are mostly rejected as not underwater, move on to a different one.

Cover the whole range of each category, roughly evenly, so the model learns what each one looks \
like in general rather than one kind of animal:
- turtles: green, hawksbill, loggerhead, and other sea turtles.
- rays: manta and devil rays, eagle rays, and stingrays.
- fish: colorful reef families (butterflyfish, angelfish, parrotfish, triggerfish) and just as \
much the plainer and temperate fish divers see all the time (sea bream, wrasses, groupers, \
damselfish, snappers, jacks, barracuda, mullet), eels, sharks, and schools of fish. Camouflaged \
bottom fish like scorpionfish, flounder, and lizardfish are useful hard cases, but keep them a \
small share.
- invertebrates: nudibranchs; crabs, shrimp, and lobsters; sea stars and urchins; octopus, squid, \
and cuttlefish; clams and other shelled mollusks.
get_progress shows how many kept photos came from each taxon you collected from. Use it to find \
and fill the gaps.

Stop once every category has reached its target, or when the budget is nearly gone or nothing \
productive is left. Finish with a short summary of what you collected."""

# Added to the system prompt for wide runs (--wide).
WIDE_NOTE = """

This run collects a different kind of photo: the animal is clearly the subject but is NOT a \
close-up. Medium shots, and wide shots where the animal is small within a reef, wreck, sand, or \
open-water scene, the way divers often film them. Close-up portraits are rejected automatically \
(reason close_up). Groups usually photographed as macro portraits (nudibranchs, small shrimp) \
will be rejected often; favor animals that are usually seen in wider scenes. Each category needs \
both training photos and separate validation photos; get_progress shows both counts."""

TOOLS = [
    {
        "name": "get_progress",
        "description": "Kept vs. target per category, top rejection reasons, and budget left.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_taxa",
        "description": (
            "Find iNaturalist taxa by common or scientific name, e.g. 'butterflyfish' or "
            "'Chaetodontidae'. Returns ids, ranks, observation counts, and the category each "
            "maps to. Taxa that map to no category are left out."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "collect_from_taxon",
        "description": (
            "Check up to n new photos from a taxon (any rank) and keep the good ones. Returns "
            "kept and rejected counts with reasons, and whether the taxon ran out of photos."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "taxon_id": {"type": "integer"},
                "n": {"type": "integer", "description": f"Photos to check, at most {MAX_BATCH}."},
            },
            "required": ["taxon_id", "n"],
            "additionalProperties": False,
        },
    },
]


class PhotoCheck(BaseModel):
    """What Haiku must answer for each photo. The API enforces this exact shape."""

    underwater: bool
    animal_is_main_subject: bool
    category: Literal["turtles", "rays", "fish", "invertebrates", "diver", "scenery", "other"]
    reason: str


FRAMING_PROMPT = """
framing: how much of the frame the main subject fills.
- close-up: it fills most of the frame, like a portrait
- medium: it is clear, but its surroundings take up a good part of the frame
- wide: it is small within a larger reef, wreck, sand, or open-water scene"""


class FramedPhotoCheck(PhotoCheck):
    """The check for wide runs: the same answers plus how the subject is framed."""

    framing: Literal["close-up", "medium", "wide"]


def category_for(taxon: dict) -> str | None:
    """The category a taxon belongs to, or None if it's outside the animals we collect."""
    lineage = {taxon["id"], *taxon.get("ancestor_ids", [])}
    return next((category for taxon_id, category in CATEGORY_TAXA if taxon_id in lineage), None)


def rejection(check: PhotoCheck, category: str) -> str | None:
    """Why a photo should be rejected, or None to keep it.

    Animal categories need an animal as the main subject. Diver photos don't (the person is the
    subject), and scenery photos must not have one.
    """
    if not check.underwater:
        return "not_underwater"
    if category == "scenery":
        if check.animal_is_main_subject:
            return "has_main_animal"
    elif category != "diver" and not check.animal_is_main_subject:
        return "not_main_subject"
    if check.category != category:
        return "category_mismatch"
    return None


def split_for(observer_id: int | str) -> str:
    """Train or val, fixed per photographer.

    If one person's photos landed on both sides, validation would partly measure recognizing
    their camera and style, and the score would look better than it is.
    """
    digest = hashlib.blake2b(str(observer_id).encode(), digest_size=8).digest()
    return "val" if int.from_bytes(digest) % VAL_EVERY == 0 else "train"


def cost_usd(model: str, usage) -> float:
    input_price, output_price, write_price, read_price = PRICES[model]
    tokens = (
        usage.input_tokens * input_price
        + usage.output_tokens * output_price
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * write_price
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * read_price
    )
    return tokens / 1_000_000


def check_photo(
    client, data: bytes, prompt: str = CHECK_PROMPT, answer: type[PhotoCheck] = PhotoCheck
) -> tuple[PhotoCheck | None, float]:
    """Ask Haiku about one photo. Returns (answer or None if it couldn't answer, cost)."""
    image = {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": base64.standard_b64encode(data).decode(),
    }
    try:
        response = client.messages.parse(
            model=CHECK_MODEL,
            max_tokens=1024,
            output_config={"effort": "low"},
            output_format=answer,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
    except anthropic.BadRequestError:  # one photo the API won't accept shouldn't end the run
        return None, 0.0
    cost = cost_usd(CHECK_MODEL, response.usage)
    if response.stop_reason in ("refusal", "max_tokens"):
        return None, cost
    return response.parsed_output, cost


class INat:
    """Read-only iNaturalist client that paces its own requests."""

    def _get(self, url: str) -> bytes:
        time.sleep(INAT_PAUSE_S)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()

    def _api(self, path: str, **params) -> list[dict]:
        url = f"{INAT_API}/{path}?{urllib.parse.urlencode(params)}"
        return json.loads(self._get(url))["results"]

    def search_taxa(self, query: str) -> list[dict]:
        # Autocomplete, because /taxa?q= returned mushrooms for "manta".
        return self._api("taxa/autocomplete", q=query, per_page=20, is_active="true")

    def observations(self, taxon_id: int, id_below: int | None = None) -> list[dict]:
        """Research-grade, wild observations with photos, newest first."""
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "captive": "false",
            "photos": "true",
            "per_page": PAGE_SIZE,
            "order_by": "id",
            "order": "desc",
        }
        if id_below is not None:
            params["id_below"] = id_below
        return self._api("observations", **params)

    def taxon(self, taxon_id: int) -> dict | None:
        results = self._api(f"taxa/{taxon_id}")
        return results[0] if results else None

    def photo(self, square_url: str) -> bytes:
        # The API lists the thumbnail URL. 500 px is plenty for a 224 px model input.
        return self._get(square_url.replace("/square.", "/medium."))


@dataclass
class State:
    """Everything the run knows, rebuilt from the log so a re-run picks up where it stopped."""

    kept: Counter = field(default_factory=Counter)  # category -> photos kept
    species: Counter = field(default_factory=Counter)  # (category, taxon id) -> photos kept
    groups: Counter = field(default_factory=Counter)  # (category, taxon collected from) -> kept
    reasons: Counter = field(default_factory=Counter)  # (category, reason) -> photos rejected
    split_kept: Counter = field(default_factory=Counter)  # (category, split) -> photos kept
    seen: set[int] = field(default_factory=set)  # observation ids already decided
    spent: float = 0.0

    @classmethod
    def load(cls, log_path: Path) -> State:
        state = cls()
        if log_path.exists():
            with open(log_path) as f:
                for line in f:
                    state.apply(json.loads(line))
        return state

    def apply(self, event: dict) -> None:
        self.spent += event.get("cost_usd", 0.0)
        if event["event"] != "photo":
            return
        self.seen.add(event["observation_id"])
        if event["reason"] is None:
            self.kept[event["category"]] += 1
            self.species[event["category"], event["taxon_id"]] += 1
            # Photos from before groups were logged count under their own species.
            self.groups[event["category"], event.get("group") or event["species"]] += 1
            self.split_kept[event["category"], event.get("split", "train")] += 1
        else:
            self.reasons[event["category"], event["reason"]] += 1


class Curator:
    """The agent's tools, and the bookkeeping behind them."""

    def __init__(
        self,
        client,
        inat,
        data_dir: Path,
        target: int,
        budget: float,
        wide: bool = False,
        val_target: int = 0,
    ) -> None:
        """`wide`: collect medium and wide shots, `target` for training and `val_target` for a
        separate wide-framing validation set (data/val_wide), with their own log and budget."""
        self.client, self.inat = client, inat
        self.data_dir, self.target, self.budget = data_dir, target, budget
        self.wide, self.val_target = wide, val_target
        main_log = data_dir / "curation" / "log.jsonl"
        self.log_path = data_dir / "curation" / "wide_log.jsonl" if wide else main_log
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = State.load(self.log_path)
        if wide:  # never re-check, or duplicate, a photo the main run already decided on
            self.state.seen |= State.load(main_log).seen
        self.cursor: dict[int, int] = {}  # taxon id -> lowest observation id read so far
        self.taxon_names: dict[int, str] = {}  # taxon id -> readable name, from searches

    def targets(self) -> dict[str, int]:
        """Photos to keep per split: training only, or training plus wide validation."""
        return {"train": self.target, "val_wide": self.val_target} if self.wide else {}

    def split_full(self, category: str, split: str) -> bool:
        return self.state.split_kept[category, split] >= self.targets()[split]

    def category_full(self, category: str) -> bool:
        if self.wide:
            return all(self.split_full(category, s) for s in self.targets())
        return self.state.kept[category] >= self.target

    @property
    def done(self) -> bool:
        return all(self.category_full(c) for c in ANIMAL_CATEGORIES)

    @property
    def over_budget(self) -> bool:
        return self.state.spent >= self.budget

    def total_target(self) -> int:
        return self.target + (self.val_target if self.wide else 0)

    def species_cap(self, category: str) -> int:
        return math.ceil(self.total_target() * SPECIES_SHARE[category])

    def group_cap(self) -> int:
        return math.ceil(self.total_target() * GROUP_SHARE)

    def split_of(self, observer_id: int) -> str:
        """Train or validation, by photographer. Wide validation photos go to their own set."""
        split = split_for(observer_id)
        return "val_wide" if self.wide and split == "val" else split

    def group_name(self, taxon_id: int) -> str:
        """A stable, readable name for a taxon, so its group cap holds across runs."""
        if taxon_id not in self.taxon_names:
            taxon = self.inat.taxon(taxon_id)
            if taxon is None:
                raise ValueError(f"no iNaturalist taxon with id {taxon_id}")
            self.remember_name(taxon)
        return self.taxon_names[taxon_id]

    def remember_name(self, taxon: dict) -> None:
        common = taxon.get("preferred_common_name")
        self.taxon_names[taxon["id"]] = f"{common} ({taxon['name']})" if common else taxon["name"]

    def log(self, event: dict) -> None:
        self.state.apply(event)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def record_cost(self, model: str, usage) -> None:
        self.log({"event": "agent_turn", "model": model, "cost_usd": cost_usd(model, usage)})

    def get_progress(self) -> dict:
        top_reasons = Counter()
        for (category, reason), n in self.state.reasons.items():
            top_reasons[f"{category}: {reason}"] += n
        return {
            "categories": {
                c: {
                    "kept": self.state.kept[c],
                    "target": self.target,
                    **(
                        {f"kept_{s}": self.state.split_kept[c, s] for s in self.targets()}
                        | {f"target_{s}": t for s, t in self.targets().items()}
                    ),
                    "species_used": sum(1 for cat, _ in self.state.species if cat == c),
                    "kept_by_taxon_collected": dict(
                        Counter(
                            {g: n for (cat, g), n in self.state.groups.items() if cat == c}
                        ).most_common(12)
                    ),
                }
                for c in ANIMAL_CATEGORIES
            },
            "top_rejections": dict(top_reasons.most_common(8)),
            "budget_left_usd": round(self.budget - self.state.spent, 3),
        }

    def search_taxa(self, query: str) -> list[dict]:
        results = []
        for taxon in self.inat.search_taxa(query):
            category = category_for(taxon)
            if category:
                self.remember_name(taxon)
                results.append(
                    {
                        "id": taxon["id"],
                        "name": taxon["name"],
                        "common_name": taxon.get("preferred_common_name"),
                        "rank": taxon["rank"],
                        "observations": taxon.get("observations_count"),
                        "category": category,
                    }
                )
        return results[:8]

    def review(self, observation: dict, category: str, group: str) -> str | None:
        """Download and check one observation's first photo. Returns the rejection reason.

        `group` is the taxon the agent asked for, so progress can show which groups are covered.
        """
        taxon, photo_url = observation["taxon"], observation["photos"][0]["url"]
        event = {
            "event": "photo",
            "observation_id": observation["id"],
            "taxon_id": taxon["id"],
            "species": taxon["name"],
            "category": category,
            "group": group,
            "observer_id": observation["user"]["id"],
            "photo_url": photo_url.replace("/square.", "/medium."),
            "reason": None,
            "check": None,
            "path": None,
            "cost_usd": 0.0,
        }
        try:
            data = self.inat.photo(photo_url)
            is_jpeg = Image.open(io.BytesIO(data)).format == "JPEG"
        except (urllib.error.URLError, TimeoutError, UnidentifiedImageError):
            data, is_jpeg = None, False

        if not is_jpeg:
            event["reason"] = "unreadable"
        else:
            if self.wide:
                check, event["cost_usd"] = check_photo(
                    self.client, data, CHECK_PROMPT + FRAMING_PROMPT, FramedPhotoCheck
                )
            else:
                check, event["cost_usd"] = check_photo(self.client, data)
            if check is None:
                event["reason"] = "check_failed"
            else:
                event["check"] = check.model_dump()
                event["reason"] = rejection(check, category)
                if event["reason"] is None and self.wide and check.framing == "close-up":
                    event["reason"] = "close_up"

        if event["reason"] is None:
            split = self.split_of(event["observer_id"])
            path = self.data_dir / split / category / f"inat_{observation['id']}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            event["split"], event["path"] = split, str(path)
        self.log(event)
        return event["reason"]

    def collect_from_taxon(self, taxon_id: int, n: int) -> dict:
        n = min(max(n, 1), MAX_BATCH)
        kept, checked = 0, 0
        rejected, skipped = Counter(), Counter()
        exhausted = False
        group = self.group_name(taxon_id)
        for _ in range(MAX_PAGES):
            observations = self.inat.observations(taxon_id, self.cursor.get(taxon_id))
            if not observations:
                exhausted = True
                break
            self.cursor[taxon_id] = observations[-1]["id"]
            for observation in observations:
                if checked >= n or self.over_budget:
                    break
                category = category_for(observation["taxon"])
                if category is None or not observation.get("photos"):
                    skipped["outside_categories"] += 1
                elif observation["id"] in self.state.seen:
                    skipped["already_seen"] += 1
                elif self.category_full(category):
                    skipped["category_full"] += 1
                elif self.wide and self.split_full(
                    category, self.split_of(observation["user"]["id"])
                ):
                    skipped["split_full"] += 1
                elif self.state.species[category, observation["taxon"]["id"]] >= self.species_cap(
                    category
                ):
                    skipped["species_cap"] += 1
                elif self.state.groups[category, group] >= self.group_cap():
                    skipped["group_cap"] += 1
                else:
                    reason = self.review(observation, category, group)
                    checked += 1
                    if reason is None:
                        kept += 1
                    else:
                        rejected[reason] += 1
            if checked >= n or self.over_budget:
                break
        summary = {
            "checked": checked,
            "kept": kept,
            "rejected": dict(rejected),
            "skipped": dict(skipped),
            "taxon_exhausted": exhausted,
            "budget_left_usd": round(self.budget - self.state.spent, 3),
        }
        print(f"  taxon {taxon_id}: {summary}", flush=True)
        return summary

    def run_tool(self, name: str, tool_input: dict) -> tuple[str, bool]:
        """Run one tool. Returns (JSON result, is_error), so the agent can react to failures."""
        try:
            if name == "get_progress":
                result = self.get_progress()
            elif name == "search_taxa":
                result = self.search_taxa(tool_input["query"])
            elif name == "collect_from_taxon":
                result = self.collect_from_taxon(tool_input["taxon_id"], tool_input["n"])
            else:
                return self.tool_error(name, tool_input, f"unknown tool: {name}")
        except ValueError as e:  # e.g. a taxon id that doesn't exist
            return self.tool_error(name, tool_input, str(e))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            return self.tool_error(name, tool_input, f"iNaturalist request failed: {e}")
        return json.dumps(result), False

    def tool_error(self, name: str, tool_input: dict, message: str) -> tuple[str, bool]:
        """Log a failed tool call (like a taxon id that doesn't exist) and tell the agent."""
        print(f"  {name} failed: {message}", flush=True)
        self.log({"event": "tool_error", "tool": name, "input": tool_input, "error": message})
        return message, True


def run_agent(curator: Curator, client, max_turns: int = MAX_TURNS) -> str:
    """The agent loop: ask Claude what to do, run the tools it picks, feed back the results.

    Written by hand rather than with the SDK's tool runner so the budget check on each turn,
    and the loop itself, are plain to see. Returns why the run stopped.
    """
    if curator.wide:
        goal = (
            f"Collect {curator.target} training and {curator.val_target} validation photos per "
            f"category, medium or wide shots only. Budget: ${curator.budget:.2f}."
        )
    else:
        goal = f"Collect {curator.target} photos per category. Budget: ${curator.budget:.2f}."
    system = AGENT_SYSTEM + WIDE_NOTE if curator.wide else AGENT_SYSTEM
    messages: list[dict] = [{"role": "user", "content": goal}]
    for turn in range(1, max_turns + 1):
        if curator.done:
            return "every category reached its target"
        if curator.over_budget:
            return "budget used up"
        response = client.beta.messages.create(
            model=AGENT_MODEL,
            max_tokens=16000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            cache_control={"type": "ephemeral"},
            output_config={"effort": "medium"},
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        curator.record_cost(AGENT_MODEL, response.usage)
        if response.stop_reason != "tool_use":
            text = " ".join(b.text for b in response.content if b.type == "text")
            return f"agent stopped ({response.stop_reason}): {text}"

        # Send back the whole response: thinking blocks must stay with their tool calls.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"[turn {turn}] {block.name} {block.input}", flush=True)
            content, is_error = curator.run_tool(block.name, block.input)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})
    return f"reached the {max_turns}-turn limit"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=int, help="Training photos to keep per category.")
    parser.add_argument("--budget", type=float, help="Hard stop, in dollars.")
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Collect medium and wide shots (not close-ups), plus a wide validation set.",
    )
    parser.add_argument("--val-target", type=int, default=WIDE_VAL_TARGET)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    # Loaded here only, so the key never reaches the shell that runs Claude Code.
    load_dotenv()
    client = anthropic.Anthropic()
    if args.wide:
        target, budget = args.target or WIDE_TARGET, args.budget or WIDE_BUDGET_USD
        curator = Curator(
            client, INat(), args.data_dir, target, budget, wide=True, val_target=args.val_target
        )
    else:
        target, budget = args.target or TARGET, args.budget or BUDGET_USD
        curator = Curator(client, INat(), args.data_dir, target, budget)
    print(f"Stopped: {run_agent(curator, client, args.max_turns)}")
    print(json.dumps(curator.get_progress(), indent=2))
    print(f"Spent ${curator.state.spent:.2f}")


if __name__ == "__main__":
    main()
