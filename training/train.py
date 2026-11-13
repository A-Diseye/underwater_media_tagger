"""Fine-tune ConvNeXt-Tiny on the curated training photos (6 categories).

Fine-tuning instead of training from scratch: the model starts from features learned on ~14M
ImageNet-22k photos (edges, textures, shapes), so 100 photos per category is enough to teach it
our categories. Two stages:
1. Head only: the pretrained layers are frozen while the new 6-class layer learns. A freshly
   initialized layer would otherwise send large, noisy updates back into the pretrained ones.
2. Everything, at a low learning rate that decays smoothly (cosine), so the network adapts gently.

Augmentation fights the domain gap between internet photos (color-corrected, sharp) and my
GoPro footage (blue-green, murky, soft). Three color-cast options, compared on validation:
- "depth": water fades every color at its own rate (red fastest, blue slowest), then the
  camera's auto-exposure brightens the image back up. No channel ever gets brighter.
- "boost": the first version, a shortcut that cuts red and boosts blue and green.
- "none".

Zoom-out (v2): my footage usually shows animals small inside a wide scene, while internet
photos are close-ups. With --zoom-out P, an animal photo is sometimes shrunk and pasted, with a
soft edge, into a random training scenery photo, keeping its label.

The best epoch is picked by balanced accuracy (average of per-category accuracy) on validation
only: data/val, averaged with data/val_wide (wide-framing internet photos) when it exists. The
test set is never read here.

Run with:  uv run python -m training.train [--cast depth|boost|none] [--seed N] [--run-name NAME]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tagger import CATEGORIES

MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"


@dataclass
class Config:
    run_name: str = "convnext_tiny_v1"
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    cast: str = "depth"  # "depth", "boost", or "none"
    epochs_head: int = 3
    epochs_full: int = 15
    lr_head: float = 1e-3
    lr_full: float = 1e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    batch_size: int = 32
    seed: int = 0
    pretrained: bool = True
    image_size: int | None = None  # None: the model's own size (224); tests use small images
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 4
    zoom_out: float = 0.0  # chance an animal training photo is pasted small into a scenery photo
    wide_photos: bool = True  # include the wide-framing training photos (data/curation/wide_log)


class DepthCast:
    """Tint an image the way water and the camera do.

    Water absorbs each color exponentially with the distance light travels: strength = e^(-k * d).
    Red has the highest k and is mostly gone by ~10 m; blue has the lowest in clear ocean water.
    In greener coastal water blue fades faster, so the green and blue rates vary a little.
    The camera's auto-exposure then scales the image so the strongest channel is back to full,
    which is why deep footage looks blue-green rather than just dark. Every channel only fades.

    Works on a tensor in [0, 1], before normalization.
    """

    RED_RATE = 0.35  # per meter, approximate, clear ocean water
    GREEN_RATES = (0.05, 0.09)
    BLUE_RATES = (0.02, 0.10)  # the high end: greener coastal water

    def __init__(self, p: float = 0.7, max_depth: float = 15.0, exposure: bool = True):
        self.p, self.max_depth, self.exposure = p, max_depth, exposure

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return img
        depth = random.uniform(0, self.max_depth)
        rates = torch.tensor(
            [self.RED_RATE, random.uniform(*self.GREEN_RATES), random.uniform(*self.BLUE_RATES)]
        )
        fade = torch.exp(-rates * depth)
        if self.exposure:
            fade = fade / fade.max()
        return img * fade.view(3, 1, 1)


class BoostCast:
    """The first, cruder cast: cut red and boost blue and green.

    Kept for comparison. Water only fades colors; the "boost" stood in for the camera's gain.
    Red keeps 20-100% of its strength, and blue and green are boosted up to 30%.
    """

    def __init__(self, p: float = 0.7, red=(0.2, 1.0), blue_green=(1.0, 1.3)):
        self.p, self.red, self.blue_green = p, red, blue_green

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return img
        scale = torch.tensor(
            [
                random.uniform(*self.red),
                random.uniform(*self.blue_green),
                random.uniform(*self.blue_green),
            ]
        ).view(3, 1, 1)
        return (img * scale).clamp(0, 1)


CASTS = {"depth": DepthCast, "boost": BoostCast, "none": None}


def build_transforms(data_cfg: dict, size: int, cast: str):
    """(train, val) transforms. Validation images are only resized and normalized."""
    normalize = transforms.Normalize(data_cfg["mean"], data_cfg["std"])
    train = [
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
    ]
    if CASTS[cast]:
        train.append(CASTS[cast]())
    train += [
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
        normalize,
    ]
    val = [
        transforms.Resize(int(size / data_cfg["crop_pct"])),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ]
    return transforms.Compose(train), transforms.Compose(val)


class CategoryFolder(datasets.ImageFolder):
    """ImageFolder with class indexes in CATEGORIES order (not alphabetical), so index 0 is
    always turtles everywhere in the project. Every category folder must exist, unless
    `allow_missing` (the wide validation set has animal categories only)."""

    def __init__(self, root, transform=None, allow_missing: bool = False):
        self.allow_missing = allow_missing
        super().__init__(root, transform, allow_empty=allow_missing)

    def find_classes(self, directory):
        missing = [c for c in CATEGORIES if not (Path(directory) / c).is_dir()]
        if missing and not self.allow_missing:
            raise FileNotFoundError(f"missing category folders in {directory}: {missing}")
        return list(CATEGORIES), {c: i for i, c in enumerate(CATEGORIES)}


ANIMALS = ("turtles", "rays", "fish", "invertebrates")


def zoom_out(animal: Image.Image, background: Image.Image, scale=(0.3, 0.6)) -> Image.Image:
    """Paste `animal` small into `background`, with a feathered edge so there's no hard seam.

    The animal's center stays in the middle half of the frame, so the random crop that follows
    (which keeps at least 60% of the image) can't cut it out.
    """
    bg = background.convert("RGB").copy()
    width = max(8, int(bg.width * random.uniform(*scale)))
    height = max(8, int(animal.height * width / animal.width))
    if height > bg.height * 0.8:
        height = int(bg.height * 0.8)
        width = max(8, int(animal.width * height / animal.height))
    small = animal.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    cx = random.uniform(0.25, 0.75) * bg.width
    cy = random.uniform(0.25, 0.75) * bg.height
    x = int(min(max(cx - width / 2, 0), bg.width - width))
    y = int(min(max(cy - height / 2, 0), bg.height - height))
    feather = max(1, min(width, height) // 10)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle((feather, feather, width - feather, height - feather), fill=255)
    bg.paste(small, (x, y), mask.filter(ImageFilter.GaussianBlur(feather / 2)))
    return bg


class ZoomOutFolder(CategoryFolder):
    """The training set, where animal photos are sometimes zoomed out into a scenery photo."""

    def __init__(self, root, transform, p: float, exclude: set[str] = frozenset()):
        super().__init__(root, transform)
        if exclude:  # e.g. leave out the wide-framing photos, to train the v1 recipe
            self.samples = [(path, t) for path, t in self.samples if path not in exclude]
            self.targets = [t for _, t in self.samples]
            self.imgs = self.samples
        self.p = p
        scenery = CATEGORIES.index("scenery")
        self.backgrounds = [path for path, target in self.samples if target == scenery]

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if CATEGORIES[target] in ANIMALS and self.backgrounds and random.random() < self.p:
            img = zoom_out(img, self.loader(random.choice(self.backgrounds)))
        return self.transform(img), target


def wide_photo_paths(data_dir: Path) -> set[str]:
    """Training files the wide-framing run kept, as the paths the training folder lists."""
    log = data_dir / "curation" / "wide_log.jsonl"
    if not log.exists():
        return set()
    paths = set()
    with open(log) as f:
        for event in map(json.loads, f):
            if event.get("event") == "photo" and event.get("split") == "train":
                paths.add(str(data_dir / "train" / event["category"] / Path(event["path"]).name))
    return paths


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device, criterion) -> dict:
    """Loss, accuracy, balanced accuracy, and accuracy per category."""
    model.eval()
    correct, total = torch.zeros(len(CATEGORIES)), torch.zeros(len(CATEGORIES))
    loss_sum, n = 0.0, 0
    with torch.no_grad(), torch.autocast(device, enabled=device == "cuda"):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss_sum += criterion(logits.float(), targets).item() * len(targets)
            n += len(targets)
            hits = (logits.argmax(1) == targets).cpu()
            for c in range(len(CATEGORIES)):
                mask = targets.cpu() == c
                correct[c] += hits[mask].sum()
                total[c] += mask.sum()
    per_class = (correct / total.clamp(min=1)).tolist()
    present = [acc for acc, t in zip(per_class, total.tolist(), strict=True) if t > 0]
    return {
        "loss": loss_sum / max(n, 1),
        "accuracy": (correct.sum() / total.sum().clamp(min=1)).item(),
        "balanced_accuracy": sum(present) / max(len(present), 1),
        "per_class": dict(zip(CATEGORIES, per_class, strict=True)),
    }


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train(cfg: Config) -> dict:
    seed_everything(cfg.seed)
    device = cfg.device
    model = timm.create_model(MODEL_NAME, pretrained=cfg.pretrained, num_classes=len(CATEGORIES))
    data_cfg = timm.data.resolve_data_config({}, model=model)
    size = cfg.image_size or data_cfg["input_size"][-1]
    train_tf, val_tf = build_transforms(data_cfg, size, cfg.cast)
    exclude = set() if cfg.wide_photos else wide_photo_paths(cfg.data_dir)
    train_set = ZoomOutFolder(cfg.data_dir / "train", train_tf, cfg.zoom_out, exclude)
    val_sets = {"val": CategoryFolder(cfg.data_dir / "val", val_tf)}
    if (cfg.data_dir / "val_wide").is_dir():
        val_sets["val_wide"] = CategoryFolder(cfg.data_dir / "val_wide", val_tf, allow_missing=True)
    loader_args = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.workers,
        "pin_memory": device == "cuda",
    }
    train_loader = DataLoader(
        train_set, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed), **loader_args
    )
    val_loaders = {
        name: DataLoader(v, shuffle=False, **loader_args) for name, v in val_sets.items()
    }
    sizes = ", ".join(f"{len(v)} {name}" for name, v in val_sets.items())
    print(f"{len(train_set)} training images; validation: {sizes}; device {device}")

    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.models_dir / f"{cfg.run_name}.pt"
    history, best = [], {"selection_score": -1.0}

    stages = [("head", cfg.epochs_head, cfg.lr_head), ("full", cfg.epochs_full, cfg.lr_full)]
    epoch = 0
    for stage, epochs, lr in stages:
        if epochs == 0:
            continue
        for p in model.parameters():  # head stage: only the new layer learns
            p.requires_grad = stage == "full" or id(p) in head_ids
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(train_loader)
        )
        for _ in range(epochs):
            epoch += 1
            started = time.time()
            model.train()
            loss_sum, n = 0.0, 0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device, enabled=device == "cuda"):
                    loss = criterion(model(images).float(), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                loss_sum += loss.item() * len(targets)
                n += len(targets)
            scores = {
                name: evaluate(model, loader, device, criterion)
                for name, loader in val_loaders.items()
            }
            val = scores["val"]
            # Pick epochs on every validation set equally: close-ups and wide shots.
            selection = sum(v["balanced_accuracy"] for v in scores.values()) / len(scores)
            row = {
                "epoch": epoch,
                "stage": stage,
                "train_loss": round(loss_sum / n, 4),
                "val_loss": round(val["loss"], 4),
                "val_accuracy": round(val["accuracy"], 4),
                "val_balanced_accuracy": round(val["balanced_accuracy"], 4),
                **{
                    f"{name}_balanced_accuracy": round(v["balanced_accuracy"], 4)
                    for name, v in scores.items()
                    if name != "val"
                },
                "selection_score": round(selection, 4),
                "seconds": round(time.time() - started, 1),
            }
            history.append(row)
            print(row, flush=True)
            if selection > best["selection_score"]:
                best = {
                    **val,
                    "epoch": epoch,
                    "selection_score": selection,
                    "val_wide": scores.get("val_wide"),
                }
                torch.save(
                    {
                        "model_name": MODEL_NAME,
                        "state_dict": model.state_dict(),
                        "classes": list(CATEGORIES),
                        "data_config": data_cfg,
                        "image_size": size,
                        "best": best,
                        "config": {k: str(v) for k, v in asdict(cfg).items()},
                        "git_commit": git_commit(),
                    },
                    checkpoint_path,
                )

    with open(cfg.models_dir / f"{cfg.run_name}_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return {"run_name": cfg.run_name, "checkpoint": str(checkpoint_path), "best": best}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default=Config.run_name)
    parser.add_argument("--cast", choices=list(CASTS), default=Config.cast)
    parser.add_argument("--epochs-head", type=int, default=Config.epochs_head)
    parser.add_argument("--epochs-full", type=int, default=Config.epochs_full)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--zoom-out", type=float, default=Config.zoom_out)
    parser.add_argument(
        "--no-wide-photos", action="store_true", help="Train without the wide-framing photos."
    )
    args = parser.parse_args(argv)
    cfg = Config(
        run_name=args.run_name,
        cast=args.cast,
        epochs_head=args.epochs_head,
        epochs_full=args.epochs_full,
        seed=args.seed,
        zoom_out=args.zoom_out,
        wide_photos=not args.no_wide_photos,
    )
    result = train(cfg)
    best = result["best"]
    print(
        f"best epoch {best['epoch']}: balanced accuracy {best['balanced_accuracy']:.3f}, "
        f"accuracy {best['accuracy']:.3f}"
    )
    print("per category:", {c: round(a, 3) for c, a in best["per_class"].items()})
    if best.get("val_wide"):
        wide = best["val_wide"]
        print(
            f"wide validation: balanced accuracy {wide['balanced_accuracy']:.3f}, per category",
            {c: round(a, 3) for c, a in wide["per_class"].items() if c in ANIMALS},
        )
    print("saved", result["checkpoint"])


if __name__ == "__main__":
    main()
