"""Find the subject of a frame with Grad-CAM, for subject-aware color correction.

Grad-CAM asks the classifier where its evidence was. Take the last convolutional feature map
(768 channels on a coarse grid), and the gradient of one category's score with respect to it.
Averaging the gradient over the grid gives one weight per channel: how much that channel pushed
the answer toward, say, "turtle". The weighted sum of the channels, positive part only, is a
heatmap of where the turtle evidence was. It works because a CNN's last layer still has a
spatial layout; a ViT's doesn't in the same way, which is one reason v3 is ConvNeXt.

Choices (reasoning in docs/color_plan.md):
- Same scale as tagging (short side resized to 365 px for 320 px crops), but a grid of crops
  that covers the whole frame, since tagging's single center crop misses the sides of a wide
  frame. The model is less reliable on the whole wide frame at once (dev balanced accuracy
  82% -> 69%), because it learned on square crops.
- The heatmap is averaged over the three v3 seeds; each seed's map is scaled by its own peak
  first, since raw Grad-CAM values aren't comparable between models.
- Whole frame instead of a subject when the tag is scenery, when confidence is low (false tags
  mostly are: false Azores turtles have a median 0.37, correct dev tags 0.74), or when the
  evidence is spread over most of the frame.

torch is imported lazily, like tagger/classify.py, so CI can import this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from tagger import CATEGORIES

MIN_CONFIDENCE = 0.5  # below this, the tag is too often wrong to steer the correction
CAM_THRESHOLD = 0.5  # subject = heatmap above this fraction of its peak (tuned on set A)
# Divers: Grad-CAM lights up what's distinctive (fins, legs, tank), not faces or torsos, so a
# lower cut-off lets the mask spread over more of the person. Animals keep the tight 0.5: lower
# spilled small fish masks into the sand (set A).
DIVER_THRESHOLD = 0.25
MAX_SUBJECT_AREA = 0.6  # a "subject" covering more of the frame than this isn't one
MIN_PEAK = 0.6  # also keep other hot regions this hot (a second diver, a school); set A
FEATHER = 0.04  # mask edge blur, as a fraction of the frame's short side
# How far the mask grows past the heatmap region, as a fraction of the short side. Grad-CAM's
# region sits on the subject but stops short of its edges (set B: on the subject 95% of the
# time, covering all of it 37%), and color correction needs the whole subject.
GROW = 0.12  # set A: 4/8/12/20% compared; 20% spilled too far around small subjects
# Gaps narrower than this (fraction of the short side) between kept regions are filled before
# growing, so two hot spots on one subject (an anemone's tentacles, #12967) become one mask
# instead of two outlines with the subject between them uncovered.
BRIDGE = 0.2  # set A: #12967 has an 18% gap; 3990 (21%) and farther stay apart

# Why a frame got the whole-frame mask, or "subject" if it didn't.
SUBJECT, SCENERY, UNUSABLE, LOW_CONFIDENCE, NO_CLEAR_SUBJECT = (
    "subject",
    "scenery",
    "unusable",  # my correction: fogged, surface, boat, or black frame
    "low_confidence",
    "no_clear_subject",
)


@dataclass
class Located:
    """Where the subject is, at the image's own resolution."""

    category: str
    confidence: float
    reason: str  # SUBJECT, or why the whole frame is used instead
    mask: np.ndarray  # H x W float32 in 0-1, feathered; all ones without a subject
    heatmap: np.ndarray | None  # H x W float32 in 0-1; None for scenery
    box: tuple[int, int, int, int] | None  # x0, y0, x1, y1 of the subject region

    @property
    def has_subject(self) -> bool:
        return self.reason == SUBJECT


def crop_origins(length: int, crop: int) -> list[int]:
    """Start positions of the fewest crops of size `crop` that cover `length`, spread evenly."""
    if length <= crop:
        return [0]
    n = int(np.ceil(length / crop))
    return [round(i * (length - crop) / (n - 1)) for i in range(n)]


def subject_mask(
    heat: np.ndarray,
    threshold: float = CAM_THRESHOLD,
    max_area: float = MAX_SUBJECT_AREA,
    min_peak: float = MIN_PEAK,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Hard subject mask from a 0-1 heatmap: the connected regions above `threshold` x peak
    whose own peak reaches `min_peak` x the overall peak (1.0 keeps only the region holding the
    peak; lower also keeps a second diver or the rest of a school). (None, None) if the heatmap
    is empty or the regions are too big to be a subject."""
    top = heat.max()
    if top <= 0:
        return None, None
    above = (heat >= threshold * top).astype(np.uint8)
    n, components = cv2.connectedComponents(above)
    region = np.zeros(heat.shape, bool)
    for k in range(1, n):
        part = components == k
        if heat[part].max() >= min_peak * top - 1e-6:
            region |= part
    if region.mean() > max_area:
        return None, None
    ys, xs = np.nonzero(region)
    return region, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def feather(
    region: np.ndarray, grow: float = GROW, blur: float = FEATHER, bridge: float = BRIDGE
) -> np.ndarray:
    """Soft mask: fill gaps narrower than `bridge` x the short side (a morphological closing:
    grow, then shrink back, so only gaps stay filled), grow the region by `grow` x the short
    side (round kernels, so masks stay round rather than boxy), then blur its edge so a
    correction weighted by it shows no seam."""
    short = min(region.shape)
    g = max(3, int(grow * short) | 1)  # odd kernel sizes
    b = max(3, int(blur * short) | 1)
    mask = region.astype(np.uint8)
    if bridge > 0:
        k = max(3, int(bridge * short) | 1)
        close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # Pad first so regions touching the frame edge aren't eaten by the shrink step.
        mask = np.pad(mask, k, mode="edge")
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)[k:-k, k:-k]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g))
    grown = cv2.dilate(mask, kernel).astype(np.float32)
    soft = cv2.GaussianBlur(grown, (0, 0), sigmaX=b / 2)
    return np.clip(soft / max(soft.max(), 1e-6), 0, 1)


class Locator:
    """Grad-CAM over a set of checkpoints (the v3 ensemble's members)."""

    def __init__(self, members: Sequence):
        """`members`: loaded tagger.classify.Classifier objects sharing an input size."""
        from torchvision import transforms

        self.members = list(members)
        first = self.members[0]
        self.crop = first.image_size
        cfg = first.data_config
        self.resize = transforms.Resize(int(self.crop / cfg["crop_pct"]))  # as tagging does
        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])]
        )

    def heatmap(self, image: Image.Image, category: str) -> np.ndarray:
        """0-1 heatmap of the evidence for `category`, at the resized scale (short side 365 px
        for v3), averaged over members."""
        import torch

        small = self.resize(image.convert("RGB"))
        w, h = small.size
        boxes = [(x, y) for y in crop_origins(h, self.crop) for x in crop_origins(w, self.crop)]
        batch = torch.stack(
            [self.to_tensor(small.crop((x, y, x + self.crop, y + self.crop))) for x, y in boxes]
        )
        k = CATEGORIES.index(category)
        # Overlapping crops cross-fade: each crop counts most at its center and little at its
        # edges, so a crop boundary leaves no seam. The floor keeps frame edges (covered by one
        # crop only) well defined.
        window = np.hanning(self.crop).astype(np.float32) + 0.05
        weight = np.outer(window, window)
        total = np.zeros((h, w), np.float32)
        for member in self.members:
            canvas, counts = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
            for (x, y), cam in zip(boxes, self._cams(member, batch, k), strict=True):
                cam = cv2.resize(cam, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)
                ch, cw = min(self.crop, h - y), min(self.crop, w - x)
                canvas[y : y + ch, x : x + cw] += (cam * weight)[:ch, :cw]
                counts[y : y + ch, x : x + cw] += weight[:ch, :cw]
            canvas /= np.maximum(counts, 1e-6)
            if canvas.max() > 0:
                total += canvas / canvas.max()  # seeds' raw scales differ: compare shapes
        return total / max(total.max(), 1e-6)

    @staticmethod
    def _cams(member, batch, k: int) -> np.ndarray:
        """Raw Grad-CAM per crop (n x grid_h x grid_w), from the last stage's feature map."""
        import torch

        model = member.model
        acts = {}
        hook = model.stages[-1].register_forward_hook(lambda m, i, o: acts.update(out=o))
        try:
            with torch.enable_grad():
                logits = model(batch.to(member.device))
                feats = acts["out"]
                (grads,) = torch.autograd.grad(logits[:, k].sum(), feats)
        finally:
            hook.remove()
        weights = grads.mean(dim=(2, 3), keepdim=True)  # one weight per channel
        cams = torch.relu((weights * feats).sum(dim=1))
        return cams.detach().float().cpu().numpy()

    def locate(self, image: Image.Image, category: str, confidence: float) -> Located:
        """The subject of `image` for its shown tag. `confidence` is the model's (1.0 for my
        own corrections)."""
        w, h = image.size
        whole = np.ones((h, w), np.float32)
        if category == "scenery":
            return Located(category, confidence, SCENERY, whole, None, None)
        if category not in CATEGORIES:
            return Located(category, confidence, UNUSABLE, whole, None, None)
        heat_small = self.heatmap(image, category)
        heat = cv2.resize(heat_small, (w, h), interpolation=cv2.INTER_LINEAR)
        if confidence < MIN_CONFIDENCE:
            return Located(category, confidence, LOW_CONFIDENCE, whole, heat, None)
        threshold = DIVER_THRESHOLD if category == "diver" else CAM_THRESHOLD
        region, box = subject_mask(heat_small, threshold, MAX_SUBJECT_AREA, MIN_PEAK)
        if region is None:
            return Located(category, confidence, NO_CLEAR_SUBJECT, whole, heat, None)
        soft = feather(region, GROW, FEATHER, BRIDGE)
        mask = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)
        sx, sy = w / heat_small.shape[1], h / heat_small.shape[0]
        x0, y0, x1, y1 = box
        full_box = (int(x0 * sx), int(y0 * sy), int(np.ceil(x1 * sx)), int(np.ceil(y1 * sy)))
        return Located(category, confidence, SUBJECT, np.clip(mask, 0, 1), heat, full_box)


def overlay(rgb: np.ndarray, found: Located) -> np.ndarray:
    """The frame (RGB uint8) with the heatmap blended in (red = most evidence) and the
    subject mask outlined in white."""
    out = rgb.copy()
    if found.heatmap is not None:
        heat = cv2.applyColorMap((found.heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out = cv2.addWeighted(out, 0.6, cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), 0.4, 0)
    if found.has_subject:
        edge = (found.mask >= 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thick = max(2, rgb.shape[1] // 300)
        cv2.drawContours(out, contours, -1, (255, 255, 255), thick)
    return out


def shown_tag(conn, image_id: int, model: str) -> tuple[str, float]:
    """The tag the gallery shows and how much to trust it: my correction (confidence 1.0), else
    the model's prediction."""
    from tagger import store

    corrected = store.correction_for(conn, image_id)
    if corrected:
        return corrected, 1.0
    p = store.predictions_for(conn, image_id)[model]
    return p["label"], p["confidence"]
