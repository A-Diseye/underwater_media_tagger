"""The Commons diver workflow, with Wikimedia and the Claude API faked."""

import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from training import collect_commons as cc
from training import collect_inat as ci

USAGE = SimpleNamespace(input_tokens=1000, output_tokens=100)


def jpeg() -> bytes:
    pixels = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "JPEG")
    return buffer.getvalue()


def answer(category="diver", underwater=True) -> ci.PhotoCheck:
    return ci.PhotoCheck(
        underwater=underwater, animal_is_main_subject=False, category=category, reason="test"
    )


class FakeChecks:
    def __init__(self, answers):
        self.answers, self.calls = answers, 0

    def parse(self, **kwargs):
        self.calls += 1
        result = self.answers(kwargs) if callable(self.answers) else self.answers
        return SimpleNamespace(parsed_output=result, usage=USAGE, stop_reason="end_turn")


class FakeCommons:
    """Categories -> files. Each file: (uploader, mime)."""

    def __init__(self, categories, files):
        self.categories, self.files = categories, files

    def members(self, category, kind):
        if kind == "subcat":
            return [f"Category:{c}" for c in self.categories.get("_subcats", [])]
        return self.categories.get(category, [])

    def image_info(self, titles):
        return {
            t: {"pageid": i, "mime": self.files[t][1], "user": self.files[t][0], "url": t}
            for i, t in enumerate(titles)
            if t in self.files
        }

    def download(self, url):
        return jpeg()


def client_for(answers):
    return SimpleNamespace(messages=FakeChecks(answers))


def test_round_robin_alternates_and_drops_duplicates():
    assert list(cc.round_robin([["a", "b", "c"], ["x", "a"], ["y"]])) == ["a", "x", "y", "b", "c"]


def test_source_categories_include_country_subcategories_but_skip_non_photo_ones():
    commons = FakeCommons(
        {"_subcats": ["Underwater photographs of scuba divers in Egypt", "Water clarity"]}, {}
    )
    cats = cc.source_categories(commons)
    assert "Underwater photographs of scuba divers in Egypt" in cats
    assert "Water clarity" not in cats and cats[0] == cc.ROOT


def test_keeps_divers_and_rejects_animals_with_divers(tmp_path):
    files = {"File:diver.jpg": ("alice", "image/jpeg"), "File:manta.jpg": ("bob", "image/jpeg")}
    commons = FakeCommons({cc.ROOT: list(files)}, files)
    # A diver with a manta: the animal wins (labeling rule 1), so it isn't a diver photo.
    calls = iter([answer(), answer("rays")])
    client = client_for(lambda kwargs: next(calls))
    summary = cc.collect(client, commons, tmp_path, target=10)
    assert summary["kept"] == 1 and summary["rejected"] == {"category_mismatch": 1}
    saved = list(tmp_path.rglob("*.jpg"))
    assert len(saved) == 1 and saved[0].parent.name == "diver"
    assert saved[0].relative_to(tmp_path).parts[0] in {"train", "val"}


def test_diagrams_and_logos_are_skipped_without_paying_for_a_check(tmp_path):
    files = {"File:logo.png": ("a", "image/png"), "File:regulator.svg": ("a", "image/svg+xml")}
    client = client_for(answer())
    summary = cc.collect(client, FakeCommons({cc.ROOT: list(files)}, files), tmp_path)
    assert summary["skipped"] == {"not_a_photo": 2} and client.messages.calls == 0


def test_uploader_cap_limits_one_photo_shoot(tmp_path):
    files = {f"File:navy{i}.jpg": ("navy", "image/jpeg") for i in range(5)}
    files["File:other.jpg"] = ("someone", "image/jpeg")
    client = client_for(answer())
    summary = cc.collect(client, FakeCommons({cc.ROOT: list(files)}, files), tmp_path, target=10)
    # Cap: ceil(10 * 0.15) = 2 from "navy".
    assert summary["kept"] == 3 and summary["skipped"] == {"uploader_cap": 3}


def test_rerun_resumes_and_respects_budget(tmp_path):
    files = {f"File:{i}.jpg": (f"user{i}", "image/jpeg") for i in range(6)}
    commons = FakeCommons({cc.ROOT: list(files)}, files)
    per_photo = ci.cost_usd(ci.CHECK_MODEL, USAGE)
    first = cc.collect(client_for(answer()), commons, tmp_path, target=10, budget=per_photo * 2)
    assert first["kept"] == 2
    second_client = client_for(answer())
    second = cc.collect(second_client, commons, tmp_path, target=10, budget=1.0)
    assert second["kept"] == 6 and second_client.messages.calls == 4  # the first 2 not re-checked
    events = [json.loads(line) for line in (tmp_path / "curation" / "commons_log.jsonl").open()]
    assert len(events) == 6
