"""Grad-CAM hit check bookkeeping: marks, the hit rate, and set B scored once."""

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from training import color_check


def _manifest(root, which, frames):
    d = color_check.set_dir(which, root)
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(frames))


FRAMES = [
    {"image_id": 1, "label": "turtles", "reason": "subject"},
    {"image_id": 2, "label": "rays", "reason": "subject"},
    {"image_id": 3, "label": "rays", "reason": "subject"},
    {"image_id": 4, "label": "fish", "reason": "subject"},
    {"image_id": 5, "label": "fish", "reason": "no_clear_subject"},
    {"image_id": 6, "label": "invertebrates", "reason": "subject"},
]


def test_hit_rate_leaves_out_wrong_tags_and_fallbacks(tmp_path):
    _manifest(tmp_path, "B", FRAMES)
    marks = ((1, "hit"), (2, "partial"), (3, "not_there"), (4, "miss"), (6, "wrong_tag_on_subject"))
    for image_id, mark in marks:
        color_check.save_mark("B", image_id, mark, root=tmp_path)
    s = color_check.summary(FRAMES, color_check.load_marks("B", tmp_path))
    assert s["to_mark"] == 5 and s["fallbacks"] == 1 and s["marked"] == 5
    assert s["hit_rate"] == pytest.approx(1 / 3)  # hit, partial, miss; wrong tags left out
    assert s["counts"]["wrong_tag_on_subject"] == 1
    assert s["hit_or_partial_rate"] == pytest.approx(2 / 3)
    assert s["by_category"]["rays"] == {"partial": 1, "not_there": 1}


def test_marks_are_kept_per_marker_and_validated(tmp_path):
    _manifest(tmp_path, "B", FRAMES)
    color_check.save_mark("B", 1, "hit", root=tmp_path)
    color_check.save_mark("B", 1, "miss", root=tmp_path, who="haiku")
    assert color_check.load_marks("B", tmp_path) == {"1": "hit"}
    assert color_check.load_marks("B", tmp_path, who="haiku") == {"1": "miss"}
    with pytest.raises(ValueError):
        color_check.save_mark("B", 1, "maybe", root=tmp_path)


@pytest.mark.parametrize("which", ["B", "final"])
def test_scored_sets_cannot_be_redrawn_once_marking_started(tmp_path, which):
    _manifest(tmp_path, which, FRAMES)
    color_check.save_mark(which, 1, "hit", root=tmp_path)
    with pytest.raises(FileExistsError):
        color_check.export(which, root=tmp_path)
    assert color_check.exported_sets(tmp_path) == [which]


def test_final_set_draws_fresh_frames_from_the_pool():
    assert color_check.SOURCE_SET["final"] == "pool"
    assert set(color_check.SOURCE_SET.values()).isdisjoint({"C"})  # C is for the blind test


def test_agreement_groups_marks():
    frames = [{"image_id": i, "reason": "subject"} for i in range(1, 6)]
    mine = {"1": "hit", "2": "partial", "3": "not_there", "4": "hit", "5": "miss"}
    haiku = {"1": "hit", "2": "miss", "3": "wrong_tag_on_subject", "4": "partial"}
    a = color_check.agreement(frames, mine, haiku)
    assert a["frames"] == 4  # frame 5 has no Haiku mark
    assert a["same_group"] == pytest.approx(3 / 4)  # partial~miss, the two wrong tags agree
    assert a["same_mark"] == pytest.approx(1 / 4)
    assert not a["trusted"]  # 75% is under the 80% bar


class FakeMessages:
    def __init__(self, mark):
        self.mark, self.prompts = mark, []

    def parse(self, **kwargs):
        content = kwargs["messages"][0]["content"]
        assert [c["type"] for c in content] == ["image", "image", "text"]
        self.prompts.append(content[2]["text"])
        answer = color_check.HitCheck(mark=self.mark, reason="outline on the turtle")
        usage = SimpleNamespace(input_tokens=2500, output_tokens=40)
        return SimpleNamespace(parsed_output=answer, usage=usage, stop_reason="end_turn")


def test_haiku_marks_outlined_frames_once_with_the_tag(tmp_path):
    frames = []
    for f in FRAMES:
        paths = {}
        for kind in ("original", "overlay"):
            path = tmp_path / f"{f['image_id']}_{kind}.jpg"
            Image.new("RGB", (8, 8)).save(path)
            paths[kind] = str(path)
        frames.append({**f, **paths})
    _manifest(tmp_path, "B", frames)
    messages = FakeMessages("hit")
    client = SimpleNamespace(messages=messages)
    result = color_check.run_haiku(client, "B", root=tmp_path)
    assert len(messages.prompts) == 5  # the no-clear-subject frame isn't asked
    assert "tagged this frame: turtles" in messages.prompts[0]
    assert result["cost_usd"] > 0 and result["failed"] == 0
    assert set(color_check.load_marks("B", tmp_path, who="haiku").values()) == {"hit"}
    color_check.run_haiku(client, "B", root=tmp_path)  # already marked: no new calls
    assert len(messages.prompts) == 5


def test_set_b_stays_locked_after_its_marks_move_to_round1(tmp_path):
    _manifest(tmp_path, "B", FRAMES)
    (color_check.set_dir("B", tmp_path) / "round1").mkdir()
    with pytest.raises(FileExistsError):
        color_check.export("B", root=tmp_path)
