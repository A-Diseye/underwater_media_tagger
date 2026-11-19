"""Grad-CAM on a real (tiny, CPU-trained) model. Needs torch, so CI skips it."""

import numpy as np
import pytest
from PIL import Image

from tagger import locate

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
from tests.test_classify import checkpoint  # noqa: E402, F401  (the tiny CPU-trained model)


def test_gradcam_on_a_real_model_covers_wide_and_tall_frames(checkpoint):  # noqa: F811
    from tagger.classify import Classifier

    loc = locate.Locator([Classifier(checkpoint, device="cpu")] * 2)
    rng = np.random.default_rng(0)
    for size in ((96, 40), (40, 96)):  # wide and tall: several crops each
        img = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
        heat = loc.heatmap(img, "turtles")
        assert heat.min() >= 0 and heat.max() <= 1 + 1e-6
        small_short = int(loc.crop / 0.875)
        assert min(heat.shape) in (small_short, small_short - 1, small_short + 1)
        found = loc.locate(img, "turtles", 0.9)
        assert found.mask.shape == (size[1], size[0])
