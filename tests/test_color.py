"""Color correction on synthetic frames: known colors under a known water cast."""

import numpy as np
import pytest

from tagger import color

CAST = np.array([0.45, 0.85, 1.0], np.float32)  # blue water: red mostly gone, green dimmed


def _frame(h=120, w=160):
    """Water (blue-green) on top, gray sand and rock below with a few gray levels, all seen
    through the cast. Returns the frame as uint8."""
    img = np.zeros((h, w, 3), np.float32)
    img[: h // 2] = np.array([0.15, 0.45, 0.65])  # open water, already blue
    for k, level in enumerate((0.35, 0.55, 0.75, 0.9)):
        img[h // 2 :, k * w // 4 : (k + 1) * w // 4] = level * CAST
    return (img * 255).round().astype(np.uint8)


def _with_patch(rgb_true, frame=None):
    frame = _frame() if frame is None else frame.copy()
    frame[70:100, 60:100] = (np.array(rgb_true) * CAST * 255).round().astype(np.uint8)
    mask = np.zeros(frame.shape[:2], np.float32)
    mask[66:104, 56:104] = 1.0
    return frame, mask


def _mean(img, ys, xs):
    return img[ys, xs].reshape(-1, 3).astype(float).mean(axis=0)


def test_a_gray_chart_under_a_cast_comes_back_near_neutral():
    levels = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
    chart = np.zeros((40, 40 * len(levels), 3), np.float32)
    for k, level in enumerate(levels):
        chart[:, k * 40 : (k + 1) * 40] = level * CAST
    out = color.correct((chart * 255).round().astype(np.uint8), mode=color.WHOLE, strength=1.0)
    for k in range(1, len(levels) - 1):  # the ends clip to black and white
        r, g, b = _mean(out, slice(5, 35), slice(k * 40 + 5, (k + 1) * 40 - 5))
        assert max(r, g, b) - min(r, g, b) < 0.12 * 255, (levels[k], r, g, b)


def test_strength_zero_is_the_original_and_shapes_hold():
    frame = _frame()
    out = color.correct(frame, strength=0.0)
    assert out.dtype == np.uint8 and out.shape == frame.shape
    assert np.array_equal(out, frame)


def test_gains_stay_in_their_limits_on_an_all_blue_frame():
    blue = np.zeros((60, 80, 3), np.uint8)
    blue[..., 2] = 200
    blue[..., 1] = 40
    s = color.estimate(blue, mode=color.WHOLE)
    r, g, b = s.subject_gains
    assert color.RED_GAIN[0] <= r <= color.RED_GAIN[1] and g == 1.0
    assert color.BLUE_GAIN[0] <= b <= color.BLUE_GAIN[1]
    out = color.correct(blue, mode=color.WHOLE, strength=1.0)
    assert out.min() >= 0 and out.max() <= 255


def test_an_orange_subject_stays_orange():
    frame, mask = _with_patch([0.9, 0.5, 0.2])  # a nudibranch, say
    s = color.estimate(frame, mask, color.SUBJECT)
    assert s.subject_gains[0] >= 1.0  # water never adds red, so red is never cut
    r, g, b = _mean(color.correct(frame, mask, strength=1.0), slice(75, 95), slice(65, 95))
    assert r > g > b and r - b > 0.3 * 255


def test_a_subject_mask_changes_the_balance():
    frame, mask = _with_patch([0.6, 0.6, 0.6])
    frame[: frame.shape[0] // 2] = [10, 160, 120]  # greener water above
    whole = color.estimate(frame, mask, color.WHOLE)
    subject = color.estimate(frame, mask, color.SUBJECT)
    assert whole.subject_gains != subject.subject_gains


def test_the_water_stays_blue_in_subject_mode():
    frame, mask = _with_patch([0.6, 0.6, 0.6])
    water = slice(5, 50), slice(5, 155)
    subject = _mean(color.correct(frame, mask, color.SUBJECT, strength=1.0), *water)
    whole = _mean(color.correct(frame, mask, color.WHOLE, strength=1.0), *water)
    assert subject[2] > subject[0] + 0.2 * 255  # clearly blue
    assert subject[2] - subject[0] > whole[2] - whole[0]  # bluer than the one-tap version


def test_estimate_then_apply_is_correct():
    frame, mask = _with_patch([0.9, 0.5, 0.2])
    s = color.estimate(frame, mask, color.SUBJECT)
    once = color.correct(frame, mask, strength=0.7)
    assert np.array_equal(color.apply(frame, mask, s, 0.7), once)


def test_no_mask_means_whole_frame():
    assert color.estimate(_frame(), None, color.SUBJECT).mode == color.WHOLE
    with pytest.raises(ValueError):
        color.estimate(_frame(), None, "local")


def test_water_inside_the_mask_matches_water_outside_it():
    """Grad-CAM's mask includes water around a small animal; that water must not show up as a
    lighter halo (set A, #4153)."""
    frame = _frame()
    frame[20:40, 70:90] = (np.array([0.7, 0.55, 0.4]) * CAST * 255).round().astype(np.uint8)
    mask = np.zeros(frame.shape[:2], np.float32)
    mask[5:55, 50:110] = 1.0  # the animal plus a lot of water around it
    out = color.correct(frame, mask, color.SUBJECT, strength=1.0)
    inside = _mean(out, slice(8, 15), slice(55, 105))
    outside = _mean(out, slice(8, 15), slice(5, 40))
    assert np.abs(inside - outside).max() < 4


def test_sharpening_touches_edges_but_not_flat_water():
    img = np.full((80, 80, 3), 0.4, np.float32)
    img[:, 40:] = 0.7  # one hard edge; the rest is flat
    sharp = color.finish(img, contrast=0.0, sharpen=0.8)
    assert np.allclose(sharp[:, :20], img[:, :20], atol=1e-3)  # flat area untouched
    assert sharp[40, 40].mean() > img[40, 40].mean()  # the bright side of the edge brightens
    assert sharp[40, 39].mean() < img[40, 39].mean()  # and the dark side darkens


def test_contrast_spreads_tones_without_shifting_color():
    img = np.zeros((10, 30, 3), np.float32)
    img[:, :10], img[:, 10:20], img[:, 20:] = 0.25, 0.5, 0.75
    out = color.finish(img, contrast=0.2, sharpen=0.0)
    assert out[0, 0].mean() < 0.25 and out[0, 25].mean() > 0.75
    assert np.ptp(out[0, 0]) < 0.01  # gray stays gray


def test_zero_finish_changes_nothing():
    img = np.random.default_rng(0).random((20, 20, 3)).astype(np.float32)
    assert color.finish(img, 0.0, 0.0) is img


def test_shallow_frames_get_less_correction(monkeypatch):
    deep = np.zeros((40, 40, 3), np.float32)
    deep[:] = [0.1, 0.5, 0.6]  # red long gone
    shallow = deep.copy()
    shallow[:8] = [0.8, 0.5, 0.3]  # something still clearly red: near the surface
    assert color.red_left(deep) < color.SHALLOW[0] and color.red_left(shallow) > color.SHALLOW[1]
    assert color.shallow_amount(0.5) == 1.0
    assert color.shallow_amount(1.5) == pytest.approx(color.SHALLOW_MIN)
    assert color.SHALLOW_MIN < color.shallow_amount(0.85) < 1.0
    monkeypatch.setattr(color, "SHALLOW_SCALING", True)
    frame = (shallow * 255).round().astype(np.uint8)
    assert color.estimate(frame, mode=color.WHOLE).amount == pytest.approx(color.SHALLOW_MIN)
    assert color.estimate((deep * 255).astype(np.uint8), mode=color.WHOLE).amount == 1.0


def test_finish_level_picks_a_preset():
    frame, mask = _with_patch([0.6, 0.6, 0.6])
    s = color.estimate(frame, mask, color.SUBJECT)
    plain = color.apply(frame, mask, s, 1.0, "none")
    medium = color.apply(frame, mask, s, 1.0, "medium")
    assert not np.array_equal(plain, medium)
    assert np.array_equal(color.apply(frame, mask, s, 1.0), medium)  # medium is the default
    with pytest.raises(KeyError):
        color.apply(frame, mask, s, 1.0, "extreme")
