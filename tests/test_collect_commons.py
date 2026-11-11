"""The diver and scenery collection workflows, with Wikimedia, iNaturalist, and Claude faked."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from training import collect_commons as cc
from training import collect_inat as ci
from training import collect_scenery as cs
from training import workflow as wf

USAGE = SimpleNamespace(input_tokens=1000, output_tokens=100)


def jpeg() -> bytes:
    pixels = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "JPEG")
    return buffer.getvalue()


def answer(category="diver", underwater=True, main_animal=False) -> ci.PhotoCheck:
    return ci.PhotoCheck(
        underwater=underwater,
        animal_is_main_subject=main_animal,
        category=category,
        reason="test",
    )


class FakeChecks:
    def __init__(self, answers):
        self.answers, self.calls = iter(answers), 0

    def parse(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            parsed_output=next(self.answers), usage=USAGE, stop_reason="end_turn"
        )


def client_for(*answers):
    return SimpleNamespace(messages=FakeChecks(answers))


def always(a, n=100):
    return client_for(*[a] * n)


def candidate(key, uploader="u", group="g", check=None):
    return wf.Candidate(key, "test", group, uploader, f"https://example.org/{key}", jpeg, check)


def logged(tmp_path, category):
    return [json.loads(line) for line in (tmp_path / "curation" / f"{category}_log.jsonl").open()]


class FakeCommons:
    """category -> file titles; title -> (uploader, mime)."""

    def __init__(self, categories, files, subcats=()):
        self.categories, self.files, self.subcats = categories, files, list(subcats)

    def members(self, category, kind):
        return self.categories.get(category, [])

    def subcategories(self, category):
        return self.subcats

    def image_info(self, titles):
        return {
            t: {"pageid": i, "mime": self.files[t][1], "user": self.files[t][0], "url": t}
            for i, t in enumerate(titles)
            if t in self.files
        }

    def download(self, url):
        return jpeg()


# --- shared workflow ---


def test_interleave_takes_sources_in_turn_until_all_run_out():
    assert list(wf.interleave([[1, 2, 3], iter([10]), (x for x in [20, 21])])) == [
        1,
        10,
        20,
        2,
        21,
        3,
    ]


def test_keeps_good_photos_and_only_writes_train_or_val(tmp_path):
    client = client_for(answer(), answer("rays"))  # second: a diver with a manta -> rays
    summary = wf.collect(
        client, [candidate("a", "x"), candidate("b", "y")], "diver", tmp_path, 10, 1.0
    )
    assert summary["kept"] == 1 and summary["rejected"] == {"category_mismatch": 1}
    [saved] = tmp_path.rglob("*.jpg")
    assert saved.parent.name == "diver" and saved.relative_to(tmp_path).parts[0] in {"train", "val"}
    assert [e["reason"] for e in logged(tmp_path, "diver")] == [None, "category_mismatch"]


def test_uploader_and_group_caps(tmp_path):
    shoot = [candidate(f"navy{i}", "navy") for i in range(5)]
    closeups = [candidate(f"coral{i}", f"u{i}", "closeups") for i in range(5)]
    summary = wf.collect(
        always(answer("scenery")),
        shoot + closeups,
        "scenery",
        tmp_path,
        10,
        1.0,
        group_shares={"closeups": 0.3},
    )
    # Uploader cap ceil(10 * 0.15) = 2; close-up cap ceil(10 * 0.3) = 3.
    assert summary["kept_by_group"] == {"g": 2, "closeups": 3}
    assert summary["skipped"] == {"uploader_cap": 3, "group_cap": 2}


def test_rerun_resumes_within_budget(tmp_path):
    photos = [candidate(str(i), f"u{i}") for i in range(6)]
    per_photo = ci.cost_usd(ci.CHECK_MODEL, USAGE)
    first = wf.collect(always(answer()), photos, "diver", tmp_path, 10, budget=per_photo * 2)
    assert first["kept"] == 2
    client = always(answer())
    second = wf.collect(client, photos, "diver", tmp_path, 10, budget=1.0)
    assert second["kept"] == 6 and client.messages.calls == 4  # the first 2 aren't re-checked
    assert len(logged(tmp_path, "diver")) == 6


def test_known_checks_are_not_paid_for_again(tmp_path):
    client = always(answer("scenery"))
    known = candidate("old", check=answer("scenery"))
    assert wf.collect(client, [known], "scenery", tmp_path, 10, 1.0)["kept"] == 1
    assert client.messages.calls == 0


def test_stops_when_only_skips_are_left(tmp_path, monkeypatch):
    monkeypatch.setattr(wf, "MAX_SKIPS_IN_A_ROW", 3)

    def endless():
        i = 0
        while True:
            i += 1
            yield candidate(f"c{i}", "same uploader")

    summary = wf.collect(always(answer()), endless(), "diver", tmp_path, 10, 1.0)
    assert summary["kept"] == 2 and summary["skipped"] == {"uploader_cap": 3}


@pytest.mark.parametrize(
    ("check", "expected"),
    [
        (answer("scenery"), None),
        (answer("scenery", main_animal=True), "has_main_animal"),
        (answer("fish", main_animal=True), "has_main_animal"),
        (answer("scenery", underwater=False), "not_underwater"),
    ],
)
def test_scenery_must_have_no_main_animal(check, expected):
    assert ci.rejection(check, "scenery") == expected


# --- sources ---


def test_commons_candidates_skip_non_photos_and_duplicates():
    files = {
        "File:a.jpg": ("x", "image/jpeg"),
        "File:logo.png": ("x", "image/png"),
        "File:b.jpg": ("y", "image/jpeg"),
    }
    commons = FakeCommons(
        {"A": ["File:a.jpg", "File:logo.png"], "B": ["File:a.jpg", "File:b.jpg"]}, files
    )
    found = list(cc.commons_candidates(commons, ["A", "B"], "divers"))
    assert sorted(c.uploader for c in found) == ["x", "y"]


def test_diver_categories_include_countries_but_skip_non_photo_ones():
    commons = FakeCommons(
        {}, {}, subcats=["Underwater photographs of scuba divers in Egypt", "Water clarity"]
    )
    cats = cc.diver_categories(commons)
    assert "Underwater photographs of scuba divers in Egypt" in cats
    assert "Water clarity" not in cats and cats[0] == cc.DIVER_ROOT


def test_reused_rejects_are_underwater_scenes_without_a_main_animal(tmp_path):
    def event(obs, **check):
        base = {
            "underwater": True,
            "animal_is_main_subject": False,
            "category": "scenery",
            "reason": "r",
        }
        return {
            "event": "photo",
            "observation_id": obs,
            "observer_id": 7,
            "photo_url": f"https://example.org/{obs}/medium.jpg",
            "check": {**base, **check},
        }

    agent_log = tmp_path / "log.jsonl"
    events = [
        event(1),
        event(2, underwater=False),
        event(3, animal_is_main_subject=True),
        event(4, category="fish"),
        {"event": "agent_turn", "cost_usd": 0.01},
    ]
    agent_log.write_text("".join(json.dumps(e) + "\n" for e in events))
    [found] = cs.reused_rejects(agent_log, inat=SimpleNamespace(photo=lambda url: jpeg()))
    assert found.key == "inat:1" and found.check.category == "scenery"
