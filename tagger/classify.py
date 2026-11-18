"""The fine-tuned classifier: load a trained checkpoint and tag images.

Same interface as the CLIP baseline (predict / predict_paths returning {category: probability}),
so the same tagging loop, gallery, and evaluation work with either model. The checkpoint carries
its class list and preprocessing settings, so predictions use exactly what training validated.

The app's model, v3, is an Ensemble: three seeds of one recipe, each also shown the mirror image,
with video frames then smoothed over their neighbors (tagger/smooth.py). Each of those three
steps was chosen by the factorial sweep in training/sweep.py, on validation and the dev half.

torch and timm are imported lazily so this module can be imported in CI, which does not install
the GPU packages.

Run with:  uv run python -m tagger.classify [models/<checkpoint>.pt] [--model-id ID]
           (no checkpoint: tag the library with the v3 ensemble)
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from tagger import CATEGORIES
from tagger.smooth import smooth

# The model the app uses: v3 = 320 px input, 30% zoom-out, Mixup, three seeds averaged, flip at
# test time, ±2.5 s smoothing (README, "v3"). Weights aren't committed; train them with
# training/sweep.py --steps train.
ACTIVE_MODEL_ID = "v3"
ACTIVE_CHECKPOINTS = tuple(Path("models") / f"sw_320_z30_mx_s{seed}.pt" for seed in (0, 1, 2))
SMOOTH_WINDOW_S = 2.5  # 1 neighbor each side; ±4.5 s started erasing short real sightings


def active_checkpoints(paths: Sequence[Path] = ACTIVE_CHECKPOINTS) -> tuple[Path, ...] | None:
    """The app model's weights, or None if any are missing on this machine (the app then falls
    back to the CLIP baseline)."""
    return tuple(paths) if all(Path(p).exists() for p in paths) else None


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
        self.image_size = size
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


class Ensemble:
    """Several checkpoints of one recipe (different seeds), each also shown the mirror image.

    Averaging seeds cancels each run's own quirks (seed-to-seed spread was ±1.7 points of dev
    macro-F1), and the mirror image is a second look at the same frame for free: a turtle is
    still a turtle flipped left to right. predict_clip additionally smooths a video's frames.
    """

    def __init__(
        self,
        checkpoints: Sequence[Path | str],
        device: str | None = None,
        flip: bool = True,
        window_s: float = SMOOTH_WINDOW_S,
        workers: int = 0,
    ):
        self.members = [Classifier(c, device) for c in checkpoints]
        if len({m.image_size for m in self.members}) != 1:
            raise ValueError("ensemble members must share an input size")
        self.flip, self.window_s, self.workers = flip, window_s, workers

    def _probs(self, paths: Sequence[Path | str], batch_size: int) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader

        first = self.members[0]  # same input size, so one preprocessing serves every member
        loader = DataLoader(
            _Frames(paths, first.preprocess), batch_size=batch_size, num_workers=self.workers
        )
        out = []
        on_gpu = first.device == "cuda"
        with torch.no_grad(), torch.autocast(first.device, enabled=on_gpu):
            for batch in loader:
                batch = batch.to(first.device)
                views = [batch, torch.flip(batch, dims=[3])] if self.flip else [batch]
                probs = [m.model(v).float().softmax(-1) for m in self.members for v in views]
                out.append(torch.stack(probs).mean(0).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(CATEGORIES)))

    def predict_paths(
        self, paths: Sequence[Path | str], batch_size: int = 64
    ) -> list[dict[str, float]]:
        """Scores for independent images (photos): no smoothing."""
        return _as_dicts(self._probs(paths, batch_size))

    def predict_clip(
        self, paths: Sequence[Path | str], times: Sequence[float | None], batch_size: int = 64
    ) -> list[dict[str, float]]:
        """Scores for frames of one video, smoothed over neighbors within window_s seconds."""
        probs = self._probs(paths, batch_size)
        t = np.array([np.nan if x is None else x for x in times], dtype=float)
        return _as_dicts(smooth(np.zeros(len(t), dtype=int), t, probs, self.window_s))


class _Frames:
    """Image files as preprocessed tensors, for a DataLoader (worker processes decode JPEGs)."""

    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = list(paths), preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            return self.preprocess(img.convert("RGB"))


def _as_dicts(probs: np.ndarray) -> list[dict[str, float]]:
    return [dict(zip(CATEGORIES, map(float, p), strict=True)) for p in probs]


def tag_library(conn, ensemble: Ensemble, model_id: str = ACTIVE_MODEL_ID) -> int:
    """Tag every media item that has no `model_id` prediction yet, one whole clip at a time
    (smoothing needs a frame's neighbors). Commits per media item. Returns images tagged."""
    from tagger import store

    media = conn.execute(
        """
        SELECT DISTINCT i.media_id FROM images i
        WHERE i.id NOT IN (SELECT image_id FROM predictions WHERE model = ?)
        ORDER BY i.media_id
        """,
        (model_id,),
    ).fetchall()
    tagged = 0
    for n, row in enumerate(media, 1):
        images = conn.execute(
            "SELECT id, file_path, t_s FROM images WHERE media_id = ? ORDER BY id",
            (row["media_id"],),
        ).fetchall()
        paths = [r["file_path"] for r in images]
        scores = ensemble.predict_clip(paths, [r["t_s"] for r in images])
        with conn:
            for r, s in zip(images, scores, strict=True):
                store.add_prediction(conn, r["id"], model_id, s)
        tagged += len(images)
        print(f"media {n}/{len(media)}: {tagged} images", flush=True)
    return tagged


def main(argv: list[str] | None = None) -> None:
    from tagger import store
    from tagger.baseline import tag_store

    parser = argparse.ArgumentParser(description="Tag all stored images with a fine-tuned model.")
    parser.add_argument(
        "checkpoint", type=Path, nargs="?", help="One checkpoint (default: the v3 ensemble)."
    )
    parser.add_argument("--model-id", help="Name in the predictions table (default: file name).")
    parser.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        ensemble = Ensemble(ACTIVE_CHECKPOINTS, workers=8)
        model_id = args.model_id or ACTIVE_MODEL_ID
        tagged = tag_library(store.connect(args.db), ensemble, model_id)
        print(f"tagged {tagged} images as {model_id}")
        return
    model_id = args.model_id or model_id_for(args.checkpoint)
    tagged = tag_store(
        store.connect(args.db), Classifier(args.checkpoint), args.batch_size, model_id=model_id
    )
    print(f"tagged {tagged} images as {model_id}")


if __name__ == "__main__":
    main()
