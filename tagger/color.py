"""Underwater color correction, modeled on how underwater photographers edit (docs/color_plan.md,
Stage 2).

Photographers white-balance on something that should be gray, then mask the subject (extra
correction) and the water (kept blue, a little darker). One-tap apps correct the whole frame
instead. The two modes here follow the two:

- `whole`: recover red, white balance, levels, clarity, all measured on the full frame.
- `subject`: the same, but the white balance is measured mostly on the subject, and the water
  gets the correction at half strength so it stays blue.

The mask is used for measuring only. Grad-CAM's mask is a soft blob that includes water around
a small animal, so painting with it (full correction inside, half outside) showed the blob as a
milky halo (set A, #4153). Lightroom avoids this because its masks follow real edges. Here the
water is found by color instead, like Lightroom's color-range and "Select Water" masks: the
water's color is read from the least red pixels outside the subject mask, and every pixel near
that color, inside the mask or not, counts as water.

The mask comes from tagger/locate.py, but this module doesn't care where it came from, and it
doesn't import torch.

Work is split into `estimate` (a few numbers, measured on a small copy) and `apply` (full
resolution). Stills use `correct`, which does both; video could later smooth the estimated
settings over a clip before applying them, as video editors grade a clip at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

WHOLE, SUBJECT = "whole", "subject"
MODES = (WHOLE, SUBJECT)

ALPHA = 1.0  # red (and blue) recovery from green, Ancuti et al. 2018
SUBJECT_WEIGHT = 0.8  # white balance weight: mask x 0.8 + 0.2, so the frame still counts a bit
GRAY_SHARE = 0.6  # white balance uses the least colorful 60% of pixels: the "eyedropper" ones
DARK, CLIPPED = 0.05, 0.98  # ...but not near-black or clipped pixels, whose color isn't real
# Water removes red and never adds it, so the red gain can't go below 1: an orange subject
# that "asks" for less red is refused. Gains are relative to green.
RED_GAIN = (1.0, 3.0)
# How much of the measured red correction is applied (both the recovery and the gain, in log
# terms for the gain). Below 1 leaves a little of the cool cast, as photographers do so a photo
# still reads as underwater; 1.0 looked too warm on set A.
RED_AMOUNT = 0.8  # set A: 0.7 / 0.75 / 0.8 / 0.85 / 1.0 compared; 0.75 and 0.8 looked alike
BLUE_GAIN = (0.5, 3.0)
WATER_STRENGTH = 0.5  # on water the gains apply as g ** 0.5, so the water stays blue
WATER_EXPOSURE = 0.95  # and the water is a little darker, which also hides backscatter
WATER_SAMPLE = 0.3  # the water's color: median of the least red 30% of pixels outside the mask
WATER_SIGMA = 0.06  # how close (in r, g, b shares of brightness) a pixel must be to count
LEVELS = (0.5, 99.5)  # black and white point percentiles
CLAHE_CLIP = 1.5  # light clarity; heavy clarity or dehaze makes water look dirty
# Finish, after the color is right: water scatters light, so dive photos are hazy and soft.
# Both work on lightness only, so colors don't shift, and both are kept small: too much looks
# crunchy. Sharpening is masked to real edges (Lightroom's "Masking" slider), so flat water,
# video compression blocks and backscatter aren't sharpened.
CONTRAST = 0.2  # S-curve blend: 0 = none, 1 = full smoothstep ("medium" on set A)
SHARPEN = 0.8  # unsharp mask amount ("medium" on set A)
SHARPEN_RADIUS = 1.0  # blur sigma in px at 1080 px short side, scaled with the frame
EDGE_LOW, EDGE_HIGH = 2.0, 6.0  # lightness gradient (0-100 scale): no sharpening below, full above
# Shallow frames need little correction: near the surface the water hasn't taken the red yet,
# and correcting as if deep makes red things glare (set A, #12608). "Red left" is the 95th
# percentile of red / green over usable pixels: at depth even the reddest pixels stay well
# under green (set A: 0.23-0.67), near the surface red survives somewhere (0.93-1.63). The whole
# correction is scaled from full at SHALLOW[0] down to SHALLOW_MIN at SHALLOW[1].
SHALLOW_SCALING = True
SHALLOW = (0.7, 1.0)
SHALLOW_MIN = 0.3
FINISHES = {"none": (0.0, 0.0), "light": (0.1, 0.4), "medium": (0.2, 0.8)}  # (contrast, sharpen)
CLAHE_MIN_LIGHTNESS = 0.15  # skipped on darker frames, where it mostly amplifies noise
STRENGTH = 0.8  # the app slider's default
ESTIMATE_SIDE = 360  # settings are measured on a copy with this short side (fast, and enough)


@dataclass
class Settings:
    """What a correction does, measured once per frame. Plain numbers, so a clip's settings
    could be smoothed over time."""

    mode: str
    red_boost: float  # amount of green mixed into red (0: red wasn't weak)
    blue_boost: float  # the same for blue, in green water
    subject_gains: tuple[float, float, float]  # RGB, inside the mask (everywhere in `whole`)
    water_gains: tuple[float, float, float]  # RGB, on water pixels (`subject` only)
    water_color: tuple[float, float, float] | None  # r, g, b shares of the water's brightness
    black: float  # levels: this maps to 0...
    white: float  # ...and this to 1
    clahe: bool
    amount: float = 1.0  # share of the correction applied: less for shallow frames


def _as_float(image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an H x W x 3 RGB image")
    return rgb.astype(np.float32) / 255.0


def _mask_for(mask, shape, mode: str) -> np.ndarray:
    if mode == WHOLE or mask is None:
        return np.ones(shape[:2], np.float32)
    mask = np.asarray(mask, np.float32)
    if mask.shape != shape[:2]:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask, 0, 1)


def _mean(img: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return (img * weight[..., None]).sum(axis=(0, 1)) / max(float(weight.sum()), 1e-6)


def compensate(img: np.ndarray, red_boost: float, blue_boost: float) -> np.ndarray:
    """Rebuild weak channels from green (Ancuti 2018): most where the channel is dark and green
    is bright, which is where water took the most."""
    out = img.copy()
    g = img[..., 1]
    out[..., 0] += red_boost * (1 - img[..., 0]) * g
    out[..., 2] += blue_boost * (1 - img[..., 2]) * g
    return np.clip(out, 0, 1)


def gray_gains(img: np.ndarray, weight: np.ndarray) -> tuple[float, float, float]:
    """White balance gains (RGB, green = 1) from the pixels most likely to be gray. A quick
    gray-world correction is applied first, so that "least colorful" means least colorful
    after the cast is gone: an orange animal stays out, sand and rock come in."""
    first = _mean(img, weight)
    rough = np.clip(img * (first[1] / np.maximum(first, 1e-6)), 0, 1)
    chroma = rough.max(axis=2) - rough.min(axis=2)
    usable = (img.max(axis=2) < CLIPPED) & (img.mean(axis=2) > DARK) & (weight > 0)
    if usable.sum() < 50:
        usable = weight > 0
    cut = np.quantile(chroma[usable], GRAY_SHARE)
    gray = usable & (chroma <= cut)
    m = _mean(img, weight * gray)
    r = float(np.clip(m[1] / max(m[0], 1e-6), *RED_GAIN))
    b = float(np.clip(m[1] / max(m[2], 1e-6), *BLUE_GAIN))
    return r, 1.0, b


def _shares(img: np.ndarray) -> np.ndarray:
    return img / np.maximum(img.sum(axis=2, keepdims=True), 1e-6)


def water_color(img: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """The water's color as r, g, b shares: water is the least red thing in a dive photo."""
    shares = _shares(img)
    outside = (mask < 0.5) & (img.mean(axis=2) > DARK)
    if outside.sum() < 50:
        outside = mask < 0.5 if (mask < 0.5).sum() >= 50 else np.ones(mask.shape, bool)
    candidates = shares[outside]
    redness = candidates[:, 0]
    least_red = candidates[redness <= np.quantile(redness, WATER_SAMPLE)]
    return tuple(float(v) for v in np.median(least_red, axis=0))


def water_map(img: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    """0-1 per pixel: how much it looks like the water, by color alone (edges stay sharp)."""
    distance = np.linalg.norm(_shares(img) - np.array(color, np.float32), axis=2)
    water = np.exp(-((distance / WATER_SIGMA) ** 2)).astype(np.float32)
    return cv2.GaussianBlur(water, (0, 0), sigmaX=1.0)  # no single-pixel speckle


def _balance(img: np.ndarray, s: Settings, original: np.ndarray) -> np.ndarray:
    if s.mode == WHOLE or s.water_color is None:
        return np.clip(img * np.array(s.subject_gains, np.float32), 0, 1)
    w = water_map(original, s.water_color)[..., None]
    log_gains = (1 - w) * np.log(s.subject_gains) + w * np.log(s.water_gains)
    exposure = (1 - w) + w * WATER_EXPOSURE
    return np.clip(img * np.exp(log_gains).astype(np.float32) * exposure, 0, 1)


def _levels(img: np.ndarray, black: float, white: float) -> np.ndarray:
    return np.clip((img - black) / max(white - black, 1e-3), 0, 1)


def _small(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = ESTIMATE_SIDE / min(img.shape[:2])
    if scale >= 1:
        return img, mask
    size = (round(img.shape[1] * scale), round(img.shape[0] * scale))
    return (
        cv2.resize(img, size, interpolation=cv2.INTER_AREA),
        cv2.resize(mask, size, interpolation=cv2.INTER_AREA),
    )


def red_left(img: np.ndarray) -> float:
    """How much red survived the water: the 95th percentile of red / green (see SHALLOW)."""
    usable = (img.mean(axis=2) > DARK) & (img.max(axis=2) < CLIPPED)
    if usable.sum() < 50:
        return 0.0
    ratio = img[..., 0][usable] / np.maximum(img[..., 1][usable], 1e-3)
    return float(np.percentile(ratio, 95))


def shallow_amount(red: float) -> float:
    """Share of the correction to apply, given how much red is left."""
    lo, hi = SHALLOW
    t = float(np.clip((red - lo) / (hi - lo), 0, 1))
    return 1.0 - t * (1.0 - SHALLOW_MIN)


def estimate(image, mask=None, mode: str = SUBJECT) -> Settings:
    """Measure the correction for one frame. `mask`: H x W in 0-1 (tagger.locate's), or None
    for `whole`."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    img = _as_float(image)
    if mask is None:
        mode = WHOLE
    img, mask = _small(img, _mask_for(mask, img.shape, mode))
    weight = mask * SUBJECT_WEIGHT + (1 - SUBJECT_WEIGHT) if mode == SUBJECT else mask
    means = _mean(img, weight)
    red_boost = ALPHA * RED_AMOUNT * max(float(means[1] - means[0]), 0.0)
    blue_boost = ALPHA * max(float(means[1] - means[2]), 0.0)
    recovered = compensate(img, red_boost, blue_boost)
    r, g, b = gray_gains(recovered, weight)
    gains = (r**RED_AMOUNT, g, b)
    water = tuple(g**WATER_STRENGTH for g in gains)
    color = water_color(img, mask) if mode == SUBJECT else None
    s = Settings(mode, red_boost, blue_boost, gains, water, color, 0.0, 1.0, False)
    balanced = _balance(recovered, s, img)
    s.black = float(np.percentile(balanced.min(axis=2), LEVELS[0]))
    s.white = float(np.percentile(balanced.max(axis=2), LEVELS[1]))
    leveled = _levels(balanced, s.black, s.white)
    s.clahe = bool(leveled.mean() > CLAHE_MIN_LIGHTNESS)
    if SHALLOW_SCALING:
        s.amount = shallow_amount(red_left(img))
    return s


def _clarity(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor((img * 255).round().astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def finish(img: np.ndarray, contrast: float, sharpen: float) -> np.ndarray:
    """Contrast S-curve and edge-masked sharpening on lightness (Lab L), floats in 0-1."""
    if contrast <= 0 and sharpen <= 0:
        return img
    before = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)[..., 0] / 100.0
    lightness = before
    if contrast > 0:
        curve = lightness * lightness * (3 - 2 * lightness)  # smoothstep: an S around mid-gray
        lightness = (1 - contrast) * lightness + contrast * curve
    if sharpen > 0:
        sigma = SHARPEN_RADIUS * min(img.shape[:2]) / 1080
        blurred = cv2.GaussianBlur(lightness, (0, 0), sigmaX=max(sigma, 0.5))
        gx = cv2.Sobel(blurred * 100, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(blurred * 100, cv2.CV_32F, 0, 1, ksize=3) / 8
        edges = np.clip((np.hypot(gx, gy) - EDGE_LOW) / (EDGE_HIGH - EDGE_LOW), 0, 1)
        lightness = lightness + sharpen * edges * (lightness - blurred)
    # Scale RGB by the lightness change instead of converting back from Lab: hue is kept, and
    # untouched pixels stay exactly as they were.
    ratio = np.clip(lightness, 0, 1) / np.maximum(before, 1e-4)
    return np.clip(img * ratio[..., None], 0, 1)


def apply(image, mask, settings: Settings, strength: float = STRENGTH) -> np.ndarray:
    """The corrected frame (H x W x 3 uint8). `strength` 0 gives back the original. `mask` is
    unused here (it was used to measure the settings), but kept so callers pass the same
    arguments to `estimate` and `apply`."""
    original = _as_float(image)
    out = compensate(original, settings.red_boost, settings.blue_boost)
    out = _levels(_balance(out, settings, original), settings.black, settings.white)
    if settings.clahe:
        out = _clarity(out)
    out = finish(out, CONTRAST, SHARPEN)
    share = float(np.clip(strength, 0, 1)) * settings.amount
    out = original + share * (out - original)
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


def correct(image, mask=None, mode: str = SUBJECT, strength: float = STRENGTH) -> np.ndarray:
    """Estimate and apply in one go, for a still."""
    settings = estimate(image, mask, mode)
    return apply(image, mask, settings, strength)
