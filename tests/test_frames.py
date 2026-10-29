import csv
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from tagger import frames


def textured(seed: int, size: tuple[int, int] = (480, 640)) -> np.ndarray:
    """A sharp, detailed synthetic image (random blocks upscaled with hard edges)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (size[0] // 16, size[1] // 16, 3), dtype=np.uint8)
    return cv2.resize(small, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


def write_video(path: Path, seconds: int, fps: int = 10, gop: int = 10) -> None:
    """Small test video with a keyframe every `gop` frames (1s), like GoPro footage."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        stream.codec_context.gop_size = gop
        for i in range(seconds * fps):
            img = textured(seed=i // fps, size=(240, 320))  # new scene each second
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("GH010419.MP4", True),
        ("GX060391.MP4", True),
        ("GOPR0123.JPG", True),
        ("GL010419.LRV", False),
        ("GH010419.THM", False),
        ("PXL_20240101_123456.jpg", False),
        ("20190312_101500.mp4", False),
    ],
)
def test_is_gopro_file(name, expected):
    assert frames.is_gopro_file(Path(name)) is expected


def test_blur_lowers_sharpness():
    img = textured(0)
    blurred = cv2.GaussianBlur(img, (21, 21), 8)
    assert frames.sharpness(img) > 10 * frames.sharpness(blurred)


def test_small_sharp_subject_is_not_blurry():
    """A small sharp subject on flat open water (like a distant ray) must still score as sharp."""
    water = np.full((540, 960, 3), (180, 110, 30), dtype=np.uint8)
    water[200:260, 400:480] = textured(3, size=(60, 80))
    assert frames.sharpness(water) > frames.BLUR_THRESHOLD * 10
    assert frames.sharpness(np.full_like(water, 128)) < frames.BLUR_THRESHOLD


def test_sharpness_roughly_resolution_independent():
    img = textured(0, size=(1080, 1920))
    small = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
    a, b = frames.sharpness(img), frames.sharpness(small)
    assert 0.5 < a / b < 2


def test_select_frames_drops_blurry_and_duplicates():
    a = textured(0)
    a_brighter = cv2.convertScaleAbs(a, alpha=1.0, beta=10)  # same scene, slight change
    blurry = cv2.GaussianBlur(textured(1), (31, 31), 12)
    b = textured(2)
    seq = [(0.0, a), (2.0, a_brighter), (4.0, blurry), (6.0, b)]
    decisions = [d for *_, d in frames.select_frames(seq, blur_threshold=50, dup_similarity=0.9)]
    assert decisions == ["kept", "duplicate", "blurry", "kept"]


def test_similarity_ignores_brightness_but_not_content():
    a = frames.thumbnail(textured(0))
    assert frames.similarity(a, frames.thumbnail(cv2.convertScaleAbs(textured(0), beta=30))) > 0.95
    assert frames.similarity(a, frames.thumbnail(textured(1))) < 0.5


def test_fit_size():
    assert frames.fit_size(3840, 2160, 1920) == (1920, 1080)
    assert frames.fit_size(1280, 720, 1920) == (1280, 720)


def test_sample_video_takes_keyframes_every_n_seconds(tmp_path):
    video = tmp_path / "GH000001.MP4"
    write_video(video, seconds=7)
    sampled = list(frames.sample_video(video, every_s=2.0))
    assert [round(t) for t, _ in sampled] == [0, 2, 4, 6]
    assert sampled[0][1].shape == (240, 320, 3)


def test_extract_is_resumable_and_skips_copies(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_video(src / "GH000001.MP4", seconds=5)
    (src / "GH000002.MP4").write_bytes((src / "GH000001.MP4").read_bytes())  # exact copy
    (src / "PXL_0001.mp4").write_bytes(b"phone video, ignored")
    out = tmp_path / "out"

    frames.extract([("trip", src)], out, blur_threshold=0)
    frames.extract([("trip", src)], out, blur_threshold=0)  # second run does nothing new

    with open(out / frames.MANIFEST) as f:
        rows = list(csv.DictReader(f))
    assert {r["clip"] for r in rows} == {"GH000001"}
    assert len(rows) == 3  # t = 0, 2, 4
    kept = [r for r in rows if r["decision"] == "kept"]
    assert kept and all((out / r["frame_path"]).exists() for r in kept)
