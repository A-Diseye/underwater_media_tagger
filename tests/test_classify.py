"""The fine-tuned classifier, on a tiny checkpoint trained on CPU (no weight download)."""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from tagger import CATEGORIES  # noqa: E402
from tagger.classify import Classifier, model_id_for  # noqa: E402
from training import train as tr  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    for split, per_class in (("train", 2), ("val", 1)):
        for i, category in enumerate(CATEGORIES):
            folder = root / "data" / split / category
            folder.mkdir(parents=True)
            for j in range(per_class):
                pixels = np.full((40, 40, 3), (i * 40, j * 30, 100), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{j}.jpg")
    cfg = tr.Config(
        run_name="tiny",
        data_dir=root / "data",
        models_dir=root / "models",
        epochs_head=1,
        epochs_full=0,
        batch_size=4,
        pretrained=False,
        image_size=32,
        device="cpu",
        workers=0,
    )
    return tr.train(cfg)["checkpoint"]


def test_predicts_a_probability_per_category(checkpoint, tmp_path):
    path = tmp_path / "frame.jpg"
    Image.fromarray(np.zeros((60, 80, 3), dtype=np.uint8)).save(path)
    [scores] = Classifier(checkpoint, device="cpu").predict_paths([path])
    assert list(scores) == list(CATEGORIES)
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)


def test_preprocessing_matches_training_validation(checkpoint):
    """Tagging must see images exactly as validation did, or the validation score lies."""
    saved = torch.load(checkpoint, weights_only=False)
    _, val_tf = tr.build_transforms(saved["data_config"], saved["image_size"], "none")
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (50, 70, 3), dtype=np.uint8))
    classifier = Classifier(checkpoint, device="cpu")
    assert torch.allclose(classifier.preprocess(img), val_tf(img))


def test_model_id_comes_from_the_file_name():
    assert model_id_for("models/convnext_tiny_depth_s0.pt") == "convnext_tiny_depth_s0"
