"""The v3 sweep's math: neighbor-frame smoothing, and tweak effects and interactions."""

import numpy as np
import pytest

from tagger import CATEGORIES
from tagger.smooth import smooth
from training import sweep


def test_smoothing_averages_neighbors_in_the_same_clip_only():
    media = np.array([1, 1, 1, 1, 2, 3])
    times = np.array([0.0, 2.0, 4.0, 10.0, 2.0, np.nan])  # clip 3 is a photo
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.3, 0.7]])
    out = smooth(media, times, probs, window_s=2)
    assert out[1] == pytest.approx([2 / 3, 1 / 3])  # frames at 0, 2, and 4 s
    assert out[3] == pytest.approx([0.0, 1.0])  # 10 s: no neighbor within 2 s
    assert out[4] == pytest.approx([1.0, 0.0])  # another clip's frame at 2 s isn't mixed in
    assert out[5] == pytest.approx([0.3, 0.7])  # photos untouched
    assert smooth(media, times, probs, 0) == pytest.approx(probs)


def test_a_one_frame_blip_is_outvoted():
    turtle, scenery = CATEGORIES.index("turtles"), CATEGORIES.index("scenery")
    probs = np.full((5, len(CATEGORIES)), 0.02)
    probs[:, scenery] = 0.9
    probs[2, turtle], probs[2, scenery] = 0.6, 0.32  # one frame looks like a shell
    out = smooth(np.ones(5), np.arange(5) * 2.0, probs, window_s=2)
    assert (out.argmax(1) == scenery).all()


def records(effect_a=0.0, effect_b=0.0, together=0.0):
    rows = []
    for a, b in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        value = 10 + effect_a * a + effect_b * b + together * a * b
        rows += [{"a": a, "b": b, "m": value + noise} for noise in (-1, 0, 1)]
    return rows


def test_effects_average_over_the_other_tweak():
    rows = records(effect_a=2, effect_b=-4)
    assert sweep.effect(rows, "a", "m") == pytest.approx(2)
    assert sweep.effect(rows, "b", "m") == pytest.approx(-4)
    assert sweep.interaction(rows, "a", "b", "m") == pytest.approx(0)


def test_interaction_catches_tweaks_that_only_help_together():
    rows = records(together=6)  # neither helps alone; both together add 6
    assert sweep.effect(rows, "a", "m") == pytest.approx(3)  # the average hides how
    assert sweep.interaction(rows, "a", "b", "m") == pytest.approx(6)  # this shows it


def test_grid_and_run_names():
    assert len(sweep.recipes()) == 8
    names = {sweep.run_name(r, s) for r in sweep.recipes() for s in sweep.SEEDS}
    assert len(names) == 24 and "sw_320_z50_mx_s2" in names
