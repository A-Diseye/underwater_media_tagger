"""Smoke test for fine-tuning: a tiny CPU run on generated images, no weight download."""

import csv

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from tagger import CATEGORIES  # noqa: E402
from training import train as tr  # noqa: E402


def make_split(root, per_class):
    for i, category in enumerate(CATEGORIES):
        folder = root / category
        folder.mkdir(parents=True)
        for j in range(per_class):
            pixels = np.full((40, 40, 3), (i * 40, j * 30, 100), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{j}.jpg")


def test_tiny_run_saves_a_checkpoint_and_history(tmp_path):
    make_split(tmp_path / "data" / "train", 2)
    make_split(tmp_path / "data" / "val", 1)
    cfg = tr.Config(
        run_name="smoke",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        epochs_head=1,
        epochs_full=1,
        batch_size=4,
        pretrained=False,
        image_size=32,
        device="cpu",
        workers=0,
    )
    result = tr.train(cfg)
    checkpoint = torch.load(result["checkpoint"], weights_only=False)
    assert checkpoint["classes"] == list(CATEGORIES)
    assert checkpoint["model_name"] == tr.MODEL_NAME
    with open(tmp_path / "models" / "smoke_history.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["stage"] for r in rows] == ["head", "full"]


def test_classes_follow_project_order_not_alphabetical(tmp_path):
    make_split(tmp_path / "train", 1)
    assert tr.CategoryFolder(tmp_path / "train").classes == list(CATEGORIES)
    (tmp_path / "train" / "turtles" / "0.jpg").unlink()
    (tmp_path / "train" / "turtles").rmdir()
    with pytest.raises(FileNotFoundError, match="turtles"):
        tr.CategoryFolder(tmp_path / "train")


def test_depth_cast_only_fades_and_red_fades_most():
    img = torch.rand(3, 16, 16)
    for _ in range(50):
        out = tr.DepthCast(p=1.0)(img)
        assert (out <= img + 1e-6).all()  # water and exposure never make a channel brighter
        fade = out.mean((1, 2)) / img.mean((1, 2))
        assert fade[0] <= fade[1] + 1e-6 and fade[0] <= fade[2] + 1e-6  # red fades fastest
        assert torch.isclose(fade.max(), torch.tensor(1.0), atol=1e-5)  # exposure restores one


def test_depth_cast_at_10m_leaves_almost_no_red(monkeypatch):
    monkeypatch.setattr(tr.random, "uniform", lambda lo, hi: 10.0 if hi == 15.0 else lo)
    out = tr.DepthCast(p=1.0, exposure=False)(torch.ones(3, 2, 2))
    assert out[0].mean() < 0.05  # e^(-0.35 * 10) is about 3%


def test_casts_can_be_switched_off():
    img = torch.full((3, 8, 8), 0.9)
    assert torch.equal(tr.DepthCast(p=0.0)(img), img)
    boosted = tr.BoostCast(p=1.0)(img)
    assert boosted[0].mean() <= img[0].mean() and boosted.max() <= 1
    assert tr.CASTS["none"] is None
