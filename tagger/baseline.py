"""Zero-shot CLIP baseline: classify images by comparing them to text descriptions.

CLIP has an image encoder and a text encoder trained so that matching image/caption pairs land
close together in the same 512-number embedding space. To classify an image we:
1. Embed a few hand-written descriptions per category and average them into one vector each.
2. Embed the image.
3. Score how close the image is to each category vector (cosine similarity), then softmax the
   scores into probabilities that sum to 1.

No training on our data: this is the "free" score the fine-tuned model has to beat.

torch and open_clip are imported lazily so this module (and PROMPTS) can be imported in CI,
which does not install the GPU packages.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from tagger import CATEGORIES

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"  # trained on LAION-2B, a public set of ~2B image-text pairs
MODEL_ID = "clip-vitb32-zeroshot"  # name stored in the predictions table

# Several wordings per category, averaged: smooths out the quirks of any single phrasing.
PROMPTS: dict[str, list[str]] = {
    "turtles": [
        "an underwater photo of a sea turtle",
        "a sea turtle swimming over a coral reef",
        "a close-up photo of a sea turtle underwater",
        "a green sea turtle or hawksbill turtle swimming",
        "a sea turtle resting on the sea floor",
    ],
    "sharks": [
        "an underwater photo of a shark",
        "a shark swimming in the ocean",
        "a reef shark swimming over a coral reef",
        # Unusual-looking sharks CLIP might not link to "shark" on its own.
        "a hammerhead shark or whale shark underwater",
    ],
    "rays": [
        "an underwater photo of a manta ray",
        "an eagle ray or mobula ray swimming in blue water",
        "a stingray lying on the sandy sea floor",
        "a ray gliding through the water",
    ],
    "fish": [
        "an underwater photo of a fish",
        "a school of fish swimming over a reef",
        "a close-up photo of a colorful reef fish",
        "a moray eel peeking out of the reef",
        "a camouflaged flounder, scorpionfish, or lizardfish lying on the sandy sea floor",
    ],
    "invertebrates": [
        "an underwater photo of a nudibranch",
        "a crab, shrimp, or lobster on the sea floor",
        "a sea star or starfish on a reef",
        "an octopus, squid, or cuttlefish underwater",
        "a close-up macro photo of a small sea creature",
    ],
    "diver": [
        "an underwater photo of a scuba diver",
        "a scuba diver swimming with fins and an air tank",
        "a group of scuba divers underwater",
        "a close-up of a diver wearing a mask underwater",
    ],
    "scenery": [
        "an underwater photo of a coral reef with no animals",
        "empty open blue water in the ocean",
        "a sandy sea floor underwater",
        "a wide underwater view of a reef wall",
        "a blurry underwater photo",
    ],
}


class ClipBaseline:
    """Loads CLIP once and scores images against the category prompts."""

    def __init__(self, device: str | None = None, prompts: dict[str, list[str]] = PROMPTS):
        import open_clip
        import torch

        if set(prompts) != set(CATEGORIES):
            raise ValueError(f"prompts must cover exactly {CATEGORIES}")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, device=self.device
        )
        self.model.eval()  # inference mode: disables training-only behavior like dropout
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        with torch.no_grad():  # no gradients needed: we're not training
            category_vectors = []
            for category in CATEGORIES:
                text = self.model.encode_text(tokenizer(prompts[category]).to(self.device))
                text = text / text.norm(dim=-1, keepdim=True)  # unit length -> dot = cosine
                mean = text.mean(dim=0)
                category_vectors.append(mean / mean.norm())
            self.text_features = torch.stack(category_vectors)  # (7, 512)

    def predict(self, images: Sequence[Image.Image]) -> list[dict[str, float]]:
        """Return {category: probability} for each image."""
        torch = self.torch
        batch = torch.stack([self.preprocess(img.convert("RGB")) for img in images])
        with torch.no_grad():
            feats = self.model.encode_image(batch.to(self.device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            # CLIP's learned temperature (~100) sharpens cosine similarities before softmax.
            logits = self.model.logit_scale.exp() * feats @ self.text_features.T
            probs = logits.softmax(dim=-1).cpu().tolist()
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


def tag_store(conn, baseline: ClipBaseline, batch_size: int = 64) -> int:
    """Predict every image in the database that has no baseline prediction yet.

    Commits after each batch so an interrupted run keeps its progress. Returns the count tagged.
    """
    from tagger import store

    rows = conn.execute(
        """
        SELECT id, file_path FROM images
        WHERE id NOT IN (SELECT image_id FROM predictions WHERE model = ?)
        ORDER BY id
        """,
        (MODEL_ID,),
    ).fetchall()
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        scores = baseline.predict_paths([r["file_path"] for r in chunk], batch_size)
        with conn:
            for row, s in zip(chunk, scores, strict=True):
                store.add_prediction(conn, row["id"], MODEL_ID, s)
        print(f"tagged {min(i + batch_size, len(rows))}/{len(rows)}", flush=True)
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    from tagger import store

    parser = argparse.ArgumentParser(description="Tag all stored images with zero-shot CLIP.")
    parser.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    conn = store.connect(args.db)
    tag_store(conn, ClipBaseline(), args.batch_size)


if __name__ == "__main__":
    main()
