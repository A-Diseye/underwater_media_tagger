"""Blind A/B test bookkeeping: votes map back to the right method, results stay honest."""

import json

import pytest

from training import color_test

FRAMES = [
    {"image_id": 1, "label": "turtles", "left": "subject"},
    {"image_id": 2, "label": "turtles", "left": "whole"},
    {"image_id": 3, "label": "fish", "left": "subject"},
    {"image_id": 4, "label": "fish", "left": "whole"},
]


def _manifest(root, frames=FRAMES):
    out = color_test.test_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(frames))


def test_votes_map_back_to_the_method_behind_each_side():
    subject_left, whole_left = FRAMES[0], FRAMES[1]
    assert color_test.winner(subject_left, "left") == "subject"
    assert color_test.winner(subject_left, "right") == "whole"
    assert color_test.winner(whole_left, "left") == "whole"
    assert color_test.winner(whole_left, "right") == "subject"
    assert color_test.winner(whole_left, "same") == "same"
    with pytest.raises(ValueError):
        color_test.winner(whole_left, "both")


def test_results_count_wins_ties_and_wrong_tags_separately(tmp_path):
    _manifest(tmp_path)
    color_test.save_vote(FRAMES[0], "left", root=tmp_path)  # subject
    color_test.save_vote(FRAMES[1], "right", root=tmp_path)  # subject
    color_test.save_vote(FRAMES[2], "same", root=tmp_path)
    half = color_test.results(FRAMES, color_test.load_votes(tmp_path))
    assert not half["complete"] and half["voted"] == 3
    color_test.save_vote(FRAMES[3], "left", no_subject=True, root=tmp_path)  # whole, wrong tag
    r = color_test.results(FRAMES, color_test.load_votes(tmp_path))
    assert r["complete"]
    assert r["overall"]["subject"] == 2 and r["overall"]["whole"] == 0 and r["overall"]["same"] == 1
    assert r["overall"]["win_rate"] == 1.0 and r["passed"]
    assert r["wrong_tag"]["whole"] == 1  # not in the win rate
    assert r["by_category"]["fish"]["same"] == 1


def test_the_win_bar_and_the_uncertainty():
    votes = {
        str(i): {"winner": "subject" if i < 11 else "whole", "no_subject": False} for i in range(20)
    }
    frames = [{"image_id": i, "label": "fish", "left": "subject"} for i in range(20)]
    r = color_test.results(frames, votes)
    assert r["overall"]["win_rate"] == pytest.approx(0.55) and not r["passed"]  # under 60%
    lo, hi = r["overall"]["interval"]
    assert lo < 0.55 < hi and hi - lo > 0.3  # 20 votes: wide
    assert color_test.p_value(15, 20) == pytest.approx(0.0207, abs=1e-3)


def test_undo_and_export_refusal(tmp_path):
    _manifest(tmp_path)
    color_test.save_vote(FRAMES[0], "left", root=tmp_path)
    with pytest.raises(FileExistsError):
        color_test.export(root=tmp_path)  # the test is run once
    color_test.undo_vote(1, root=tmp_path)
    assert color_test.load_votes(tmp_path) == {}


def test_sonnet_waits_for_my_votes(tmp_path):
    _manifest(tmp_path)
    with pytest.raises(RuntimeError):
        color_test.run_sonnet(client=None, root=tmp_path)


class _Response:
    def __init__(self, choice):
        from types import SimpleNamespace

        self.stop_reason = "end_turn"
        self.parsed_output = color_test.Pick(choice=choice, reason="water looks natural")
        self.usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=100,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )


class _Client:
    """Stands in for anthropic.Anthropic: always prefers the first image it's shown."""

    def __init__(self):
        from types import SimpleNamespace

        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return _Response("first")


def test_sonnet_judges_blind_in_its_own_order(tmp_path, monkeypatch):
    frames = [
        dict(f, left_path=f"{f['image_id']}_l", right_path=f"{f['image_id']}_r") for f in FRAMES
    ]
    _manifest(tmp_path, frames)
    for f in frames:
        color_test.save_vote(f, "same", root=tmp_path)
    shown = []
    monkeypatch.setattr(color_test, "_judge_image", lambda path: shown.append(path) or {})
    client = _Client()
    result = color_test.run_sonnet(client, root=tmp_path)
    assert result["failed"] == 0 and len(client.calls) == 4
    assert client.calls[0]["fallbacks"] == "default"
    sonnet = color_test.load_votes(tmp_path, who="sonnet")
    for f in frames:  # "first" means whichever side Sonnet was shown first
        first = shown[2 * frames.index(f)]
        side = "left" if first.endswith("_l") else "right"
        assert sonnet[str(f["image_id"])]["winner"] == color_test.winner(f, side)


def test_rounds_keep_their_votes_apart(tmp_path):
    _manifest(tmp_path)
    pool = color_test.test_dir(tmp_path, "pool")
    pool.mkdir(parents=True)
    (pool / "manifest.json").write_text(json.dumps(FRAMES))
    color_test.save_vote(FRAMES[0], "left", root=tmp_path, which="pool")
    assert color_test.load_votes(tmp_path) == {}
    assert color_test.load_votes(tmp_path, which="pool")["1"]["winner"] == "subject"
    assert color_test.exported_rounds(tmp_path) == ["C", "pool"]
    assert color_test.ROUNDS["pool"] == ("pool", ("final",))  # skips frames I marked for Grad-CAM
