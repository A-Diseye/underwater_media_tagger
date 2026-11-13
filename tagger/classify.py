"""The fine-tuned classifier: load a trained checkpoint and tag images.

Same interface as the CLIP baseline (predict / predict_paths returning {category: probability}),
so the same tagging loop, gallery, and evaluation work with either model. The checkpoint carries
its class list and preprocessing settings, so predictions use exactly what training validated.

torch and timm are imported lazily so this module can be imported in CI, which does not install
the GPU packages.

Run with:  uv run python -m tagger.classify models/<checkpoint>.pt [--model-id ID]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from tagger import CATEGORIES


def model_id_for(checkpoint: Path | str) -> str:
    """Prediction-table name for a checkpoint, e.g. models/convnext_tiny_depth_s0.pt ->
    "convnext_tiny_depth_s0"."""
    return Path(checkpoint).stem


class Classifier:
    """Loads a fine-tuned checkpoint once and scores images."""

    def __init__(self, checkpoint: Path | str, device: str | None = None):
        import timm
        import torch
        from torchvision import transforms

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["classes"] != list(CATEGORIES):
            raise ValueError(f"checkpoint classes {saved['classes']} != {list(CATEGORIES)}")
        self.model = timm.create_model(
            saved["model_name"], pretrained=False, num_classes=len(CATEGORIES)
        )
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(self.device).eval()  # inference mode: no dropout, fixed normalization
        cfg, size = saved["data_config"], saved["image_size"]
        # The same steps as validation in training/train.py: resize, center crop, normalize.
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(int(size / cfg["crop_pct"])),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(cfg["mean"], cfg["std"]),
            ]
        )

    def predict(self, images: Sequence[Image.Image]) -> list[dict[str, float]]:
        """Return {category: probability} for each image."""
        torch = self.torch
        batch = torch.stack([self.preprocess(img.convert("RGB")) for img in images])
        with torch.no_grad():
            probs = self.model(batch.to(self.device)).softmax(dim=-1).cpu().tolist()
        return [dict(zip(CATEGORIES, p, strict=True)) for p in probs]

    def predict_paths(
        self, paths: Sequence[Path | str], batch_size: int = 64
    ) -> list[dict[str, float]]:
        results = []
        for i in range(0, len(paths), batch_size):
            images = []
            for p in paths[i : i + batch_size]:
                with Image.open(p) as img:
                    images.append(img.convert("RGB"))
            results.extend(self.predict(images))
        return results


def main(argv: list[str] | None = None) -> None:
    from tagger import store
    from tagger.baseline import tag_store

    parser = argparse.ArgumentParser(description="Tag all stored images with a fine-tuned model.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model-id", help="Name in the predictions table (default: file name).")
    parser.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    model_id = args.model_id or model_id_for(args.checkpoint)
    tagged = tag_store(
        store.connect(args.db), Classifier(args.checkpoint), args.batch_size, model_id=model_id
    )
    print(f"tagged {tagged} images as {model_id}")


if __name__ == "__main__":
    main()
