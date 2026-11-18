"""Smooth video tags across neighboring frames.

Frames are kept every 2 s, and a real animal usually stays in view for several of them. A
boulder that happens to look like a turtle shell for one frame is a blip. Averaging each frame's
probabilities with the frames within `window_s` seconds of it in the same clip keeps the
encounters and drops most of the blips.

Photos (no timestamp) are left as they are. This uses no labels: only the model's own scores
and where each frame sits in its clip.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def smooth(
    media_ids: np.ndarray, times: np.ndarray, probs: np.ndarray, window_s: float
) -> np.ndarray:
    """Average each video frame's probabilities with its clip's frames within `window_s` seconds.

    `times` holds seconds into the clip, NaN for photos. Rows keep their order.
    """
    out = probs.copy()
    if window_s <= 0:
        return out
    clips = defaultdict(list)
    for row, (media, t) in enumerate(zip(media_ids.tolist(), times.tolist(), strict=True)):
        if not np.isnan(t):
            clips[media].append(row)
    for rows in clips.values():
        rows = np.array(rows)
        t = times[rows]
        near = np.abs(t[:, None] - t[None, :]) <= window_s  # clips are at most a few hundred frames
        out[rows] = (near.astype(probs.dtype) @ probs[rows]) / near.sum(axis=1, keepdims=True)
    return out
