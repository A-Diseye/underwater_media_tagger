"""The curation agent's rules and loop, with iNaturalist and the Claude API faked."""

import io
import json
import urllib.error
from types import SimpleNamespace
from typing import get_args

import numpy as np
import pytest
from PIL import Image

from tagger import CATEGORIES
from training import collect_inat as ci

RAYS, SHARKS, TURTLES, CLAMS = 448201, 47273, 372234, 47115
USAGE = SimpleNamespace(input_tokens=1000, output_tokens=100)


def jpeg() -> bytes:
    pixels = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "JPEG")
    return buffer.getvalue()


def check(category="rays", underwater=True, main=True) -> ci.PhotoCheck:
    return ci.PhotoCheck(
        underwater=underwater, animal_is_main_subject=main, category=category, reason="test"
    )


def observation(obs_id, group=RAYS, species=1000, observer=1) -> dict:
    return {
        "id": obs_id,
        "taxon": {"id": species, "name": f"species {species}", "ancestor_ids": [1, group, species]},
        "user": {"id": observer},
        "photos": [{"url": f"https://example.org/photos/{obs_id}/square.jpg"}],
    }


class FakeINat:
    def __init__(self, pages):
        self.pages, self.photos_fetched = list(pages), []

    def search_taxa(self, query):
        return [
            {"id": 5, "name": "Mobula", "rank": "genus", "ancestor_ids": [1, RAYS, 5]},
            {"id": 6, "name": "A virus", "rank": "species", "ancestor_ids": [1, 6]},
        ]

    def observations(self, taxon_id, id_below=None):
        return self.pages.pop(0) if self.pages else []

    def taxon(self, taxon_id):
        return None if taxon_id == 64 else {"id": taxon_id, "name": f"Taxon {taxon_id}"}

    def photo(self, url):
        self.photos_fetched.append(url)
        return jpeg()


class FakeChecks:
    """Stands in for client.messages: answers every photo with `answer`."""

    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def parse(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(parsed_output=self.answer, usage=USAGE, stop_reason="end_turn")


def curator_for(tmp_path, pages, answer=None, target=10, budget=5.0):
    client = SimpleNamespace(messages=FakeChecks(answer or check()))
    return ci.Curator(client, FakeINat(pages), tmp_path, target, budget)


def test_haiku_categories_match_the_project():
    assert set(get_args(ci.PhotoCheck.model_fields["category"].annotation)) == {
        *CATEGORIES,
        "other",
    }
    assert set(ci.ANIMAL_CATEGORIES) <= set(CATEGORIES)


@pytest.mark.parametrize(
    ("lineage", "expected"),
    [
        ([TURTLES], "turtles"),
        ([SHARKS, RAYS], "rays"),  # rays sit inside the shark-and-ray group
        ([SHARKS], "fish"),  # sharks are fish
        ([47178], "fish"),
        ([CLAMS], "invertebrates"),
        ([47534], None),  # coral is scenery, not a collected animal
    ],
)
def test_category_comes_from_taxonomy(lineage, expected):
    assert ci.category_for({"id": 9, "ancestor_ids": [1, *lineage]}) == expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (check(), None),
        (check(underwater=False), "not_underwater"),
        (check(main=False), "not_main_subject"),
        (check(category="fish"), "category_mismatch"),
    ],
)
def test_rejection_rules(answer, expected):
    assert ci.rejection(answer, "rays") == expected


def test_split_is_fixed_per_photographer():
    assert ci.split_for(42) == ci.split_for(42)
    share = sum(ci.split_for(i) == "val" for i in range(2000)) / 2000
    assert 0.15 < share < 0.25


def test_collect_keeps_good_photos_and_only_writes_train_or_val(tmp_path):
    curator = curator_for(tmp_path, [[observation(1), observation(2, species=1001)]])
    summary = curator.collect_from_taxon(RAYS, 10)
    assert (summary["checked"], summary["kept"]) == (2, 2)
    written = [p for p in tmp_path.rglob("*.jpg")]
    assert len(written) == 2
    assert {p.relative_to(tmp_path).parts[0] for p in written} <= {"train", "val"}
    assert all(p.parent.name == "rays" for p in written)


def test_rejected_photos_are_logged_with_a_reason_but_not_saved(tmp_path):
    curator = curator_for(tmp_path, [[observation(1)]], answer=check(underwater=False))
    summary = curator.collect_from_taxon(RAYS, 10)
    assert summary["rejected"] == {"not_underwater": 1}
    assert not list(tmp_path.rglob("*.jpg"))
    event = json.loads(curator.log_path.read_text())
    assert event["reason"] == "not_underwater" and event["photo_url"].endswith("medium.jpg")


def test_species_cap_skips_without_paying_for_a_check(tmp_path):
    photos = [observation(i, species=1000) for i in range(1, 6)]
    curator = curator_for(tmp_path, [photos], target=10)  # rays cap: ceil(10 * 0.15) = 2
    summary = curator.collect_from_taxon(RAYS, 10)
    assert summary["kept"] == 2 and summary["skipped"] == {"species_cap": 3}
    assert curator.client.messages.calls == 2


def test_rerun_resumes_from_the_log_without_rechecking(tmp_path):
    first = curator_for(tmp_path, [[observation(1)]])
    first.collect_from_taxon(RAYS, 10)
    second = curator_for(tmp_path, [[observation(1), observation(2, species=1001)]])
    summary = second.collect_from_taxon(RAYS, 10)
    assert (summary["checked"], summary["skipped"]) == (1, {"already_seen": 1})
    assert second.state.kept["rays"] == 2
    assert second.state.spent == pytest.approx(ci.cost_usd(ci.CHECK_MODEL, USAGE) * 2)


def test_budget_stops_collection(tmp_path):
    per_photo = ci.cost_usd(ci.CHECK_MODEL, USAGE)
    photos = [observation(i, species=1000 + i) for i in range(1, 6)]
    curator = curator_for(tmp_path, [photos], budget=per_photo * 2)
    assert curator.collect_from_taxon(RAYS, 10)["checked"] == 2
    assert curator.over_budget


def test_search_hides_taxa_outside_the_categories(tmp_path):
    results = curator_for(tmp_path, []).search_taxa("manta")
    assert [(r["id"], r["category"]) for r in results] == [(5, "rays")]


class FakeAgent:
    """Stands in for client.beta.messages: replies with the scripted responses in order."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def agent_client(responses):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeAgent(responses)))


def tool_call(name, tool_input, tool_id="t1"):
    block = SimpleNamespace(type="tool_use", name=name, input=tool_input, id=tool_id)
    return SimpleNamespace(content=[block], stop_reason="tool_use", usage=USAGE)


def finished(text="All done.", stop_reason="end_turn"):
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block], stop_reason=stop_reason, usage=USAGE)


def test_agent_runs_the_tools_it_picks_and_feeds_back_results(tmp_path):
    curator = curator_for(tmp_path, [])
    client = agent_client([tool_call("search_taxa", {"query": "manta"}), finished()])
    assert "All done." in ci.run_agent(curator, client)
    second_request = client.beta.messages.requests[1]
    result = second_request["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "t1" and result["is_error"] is False
    assert json.loads(result["content"])[0]["name"] == "Mobula"


def test_agent_stops_at_the_turn_limit(tmp_path):
    curator = curator_for(tmp_path, [])
    client = agent_client([tool_call("get_progress", {}) for _ in range(3)])
    assert "2-turn limit" in ci.run_agent(curator, client, max_turns=2)


def test_agent_stops_when_the_budget_is_gone(tmp_path):
    curator = curator_for(tmp_path, [], budget=ci.cost_usd(ci.AGENT_MODEL, USAGE))
    client = agent_client([tool_call("get_progress", {}), finished()])
    assert ci.run_agent(curator, client) == "budget used up"
    assert len(client.beta.messages.requests) == 1


def test_agent_handles_a_refusal_without_crashing(tmp_path):
    curator = curator_for(tmp_path, [])
    client = agent_client([finished(text="", stop_reason="refusal")])
    assert "refusal" in ci.run_agent(curator, client)


def test_failed_tools_are_reported_to_the_agent_and_logged(tmp_path):
    class BrokenINat(FakeINat):
        def observations(self, taxon_id, id_below=None):
            raise urllib.error.HTTPError("https://example.org", 422, "Unprocessable", {}, None)

    curator = ci.Curator(
        SimpleNamespace(messages=FakeChecks(check())), BrokenINat([]), tmp_path, 10, 5.0
    )
    for name, tool_input in (("collect_from_taxon", {"taxon_id": 70, "n": 1}), ("nope", {})):
        message, is_error = curator.run_tool(name, tool_input)
        assert is_error
    events = [json.loads(line) for line in curator.log_path.read_text().splitlines()]
    assert [e["event"] for e in events] == ["tool_error", "tool_error"]
    assert events[0]["input"] == {"taxon_id": 70, "n": 1} and "422" in events[0]["error"]


def test_progress_shows_which_searched_groups_photos_came_from(tmp_path):
    curator = curator_for(tmp_path, [[observation(1), observation(2, species=1001)]])
    curator.search_taxa("manta")  # names taxon 5 "Mobula"
    curator.collect_from_taxon(5, 10)
    progress = curator.get_progress()["categories"]["rays"]
    assert progress["kept_by_taxon_collected"] == {"Mobula": 2}
    # Reloading from the log keeps the groups.
    reloaded = curator_for(tmp_path, []).get_progress()["categories"]["rays"]
    assert reloaded["kept_by_taxon_collected"] == {"Mobula": 2}


def test_group_cap_stops_one_taxon_filling_a_category(tmp_path):
    photos = [observation(i, species=1000 + i) for i in range(1, 6)]
    curator = curator_for(tmp_path, [photos], target=10)  # group cap: ceil(10 * 0.25) = 3
    summary = curator.collect_from_taxon(RAYS, 10)
    assert summary["kept"] == 3 and summary["skipped"] == {"group_cap": 2}
    assert curator.get_progress()["categories"]["rays"]["kept_by_taxon_collected"] == {
        f"Taxon {RAYS}": 3
    }


def test_made_up_taxon_id_is_a_tool_error(tmp_path):
    curator = curator_for(tmp_path, [[observation(1)]])
    message, is_error = curator.run_tool("collect_from_taxon", {"taxon_id": 64, "n": 5})
    assert is_error and "no iNaturalist taxon" in message
    assert curator.client.messages.calls == 0


def framed(framing="wide", category="rays") -> ci.FramedPhotoCheck:
    return ci.FramedPhotoCheck(
        underwater=True,
        animal_is_main_subject=True,
        category=category,
        reason="test",
        framing=framing,
    )


def wide_curator(tmp_path, pages, answer, target=4, val_target=2):
    client = SimpleNamespace(messages=FakeChecks(answer))
    return ci.Curator(
        client, FakeINat(pages), tmp_path, target, 5.0, wide=True, val_target=val_target
    )


# Photographers the train/val rule sends to each side.
TRAIN_OBSERVERS = [o for o in range(1, 200) if ci.split_for(o) == "train"]
VAL_OBSERVERS = [o for o in range(1, 200) if ci.split_for(o) == "val"]


def test_wide_run_rejects_close_ups(tmp_path):
    curator = wide_curator(
        tmp_path, [[observation(1, observer=TRAIN_OBSERVERS[0])]], framed("close-up")
    )
    summary = curator.collect_from_taxon(RAYS, 5)
    assert summary["rejected"] == {"close_up": 1}
    assert not list(tmp_path.rglob("*.jpg"))


def test_wide_validation_photos_go_to_their_own_folder(tmp_path):
    photos = [
        observation(1, species=1001, observer=TRAIN_OBSERVERS[0]),
        observation(2, species=1002, observer=VAL_OBSERVERS[0]),
    ]
    curator = wide_curator(tmp_path, [photos], framed("medium"))
    assert curator.collect_from_taxon(RAYS, 5)["kept"] == 2
    saved = {p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*.jpg")}
    assert saved == {"train", "val_wide"}  # never data/val, the close-up validation set
    progress = curator.get_progress()["categories"]["rays"]
    assert (progress["kept_train"], progress["kept_val_wide"]) == (1, 1)


def test_wide_run_skips_photos_the_main_run_decided(tmp_path):
    main = curator_for(tmp_path, [[observation(1)]])
    main.collect_from_taxon(RAYS, 5)
    wide = wide_curator(tmp_path, [[observation(1)]], framed())
    assert wide.collect_from_taxon(RAYS, 5)["skipped"] == {"already_seen": 1}
    assert wide.client.messages.calls == 0


def test_wide_run_fills_each_side_separately(tmp_path):
    photos = [observation(i, species=1000 + i, observer=TRAIN_OBSERVERS[i]) for i in range(1, 4)]
    curator = wide_curator(tmp_path, [photos], framed(), target=1, val_target=1)
    summary = curator.collect_from_taxon(RAYS, 5)
    assert summary["kept"] == 1 and summary["skipped"] == {"split_full": 2}
    assert not curator.category_full("rays")  # validation still needs a photo
