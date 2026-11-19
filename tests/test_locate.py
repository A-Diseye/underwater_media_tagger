"""Grad-CAM subject location: the mask math runs everywhere; Grad-CAM itself needs torch."""

import numpy as np
from PIL import Image

from tagger import locate


def test_crops_cover_the_whole_length_evenly():
    assert locate.crop_origins(320, 320) == [0]
    assert locate.crop_origins(200, 320) == [0]
    assert locate.crop_origins(649, 320) == [0, 164, 329]  # 16:9 at tagging scale
    starts = locate.crop_origins(1000, 320)
    assert starts[0] == 0 and starts[-1] + 320 == 1000
    covered = np.zeros(1000, bool)
    for s in starts:
        covered[s : s + 320] = True
    assert covered.all()


def test_mask_keeps_the_region_holding_the_peak_only():
    heat = np.zeros((40, 60), np.float32)
    heat[5:15, 5:15] = 1.0  # the subject
    heat[25:35, 40:55] = 0.8  # a second hot spot (reef), also above threshold
    region, box = locate.subject_mask(heat, threshold=0.5, min_peak=1.0)
    assert region[10, 10] and not region[30, 45]
    assert box == (5, 5, 15, 15)
    both, box = locate.subject_mask(heat, threshold=0.5, min_peak=0.7)  # a second diver, say
    assert both[10, 10] and both[30, 45] and box == (5, 5, 55, 35)


def test_spread_out_evidence_is_no_subject():
    heat = np.ones((40, 60), np.float32)
    heat[0, 0] = 1.5
    assert locate.subject_mask(heat, max_area=0.6) == (None, None)
    assert locate.subject_mask(np.zeros((10, 10), np.float32)) == (None, None)


def test_feathered_mask_is_soft_and_covers_the_region():
    region = np.zeros((100, 100), bool)
    region[40:60, 40:60] = True
    soft = locate.feather(region)
    assert soft.dtype == np.float32 and soft.min() >= 0 and soft.max() <= 1
    assert soft[50, 50] > 0.9 and soft[0, 0] < 0.01
    assert 0 < soft[40, 35] < 1  # the edge fades instead of stopping


class _FakeLocator(locate.Locator):
    """Skips the model: a fixed heatmap, so the decision rules can be tested without torch."""

    def __init__(self, heat):
        self.heat = heat

    def heatmap(self, image, category):
        return self.heat


def _hot_spot():
    heat = np.zeros((36, 64), np.float32)
    heat[10:20, 20:30] = 1.0
    return heat


def test_whole_frame_for_scenery_unusable_and_low_confidence():
    img = Image.new("RGB", (128, 72))
    loc = _FakeLocator(_hot_spot())
    scenery = loc.locate(img, "scenery", 0.9)
    assert scenery.reason == locate.SCENERY and scenery.heatmap is None
    assert loc.locate(img, "unusable", 1.0).reason == locate.UNUSABLE
    low = loc.locate(img, "turtles", locate.MIN_CONFIDENCE - 0.01)
    assert low.reason == locate.LOW_CONFIDENCE and low.heatmap is not None
    for found in (scenery, low):
        assert not found.has_subject and found.mask.shape == (72, 128) and found.mask.min() == 1


def test_confident_tag_gets_a_subject_mask_at_full_resolution():
    found = _FakeLocator(_hot_spot()).locate(Image.new("RGB", (128, 72)), "turtles", 0.8)
    assert found.has_subject
    assert found.mask.shape == found.heatmap.shape == (72, 128)
    assert found.mask[30, 50] > 0.9 and found.mask[70, 5] < 0.05
    assert found.box == (40, 20, 60, 40)  # the 64x36 heatmap box, scaled by 2


def test_divers_get_the_lower_cut_off():
    heat = np.zeros((36, 64), np.float32)
    heat[10:20, 20:30] = 1.0  # fins: the hot spot
    heat[20:30, 20:30] = 0.3  # torso: weak evidence, below 0.5 but above 0.25
    img = Image.new("RGB", (64, 36))
    animal = _FakeLocator(heat).locate(img, "turtles", 0.9)
    diver = _FakeLocator(heat).locate(img, "diver", 0.9)
    assert animal.box[3] == 20 and diver.box[3] == 30  # the diver's mask reaches the torso


def test_growing_the_mask_covers_more_around_the_region():
    region = np.zeros((100, 100), bool)
    region[40:60, 40:60] = True
    tight = locate.feather(region, grow=0.04, bridge=0)
    wide = locate.feather(region, grow=0.2, bridge=0)
    assert (wide >= 0.5).sum() > (tight >= 0.5).sum()
    assert wide[50, 30] >= 0.5 > tight[50, 30]  # 10 px left of the region's edge


def test_bridging_joins_nearby_regions_but_not_distant_ones():
    region = np.zeros((100, 200), bool)
    region[40:60, 40:60] = True
    region[40:60, 66:86] = True  # 6 px gap: same subject
    region[40:60, 170:190] = True  # far away: a separate region
    joined = locate.feather(region, grow=0.04, bridge=0.1)
    apart = locate.feather(region, grow=0.04, bridge=0)
    assert joined[50, 63] > 0.9 and apart[50, 63] < joined[50, 63]  # the gap is filled
    assert joined[50, 130] < 0.01  # the distant region stays separate
    edge = np.zeros((100, 100), bool)
    edge[0:20, 0:20] = True  # touching the frame's corner
    assert locate.feather(edge, grow=0.04, bridge=0.2)[2, 2] > 0.9
