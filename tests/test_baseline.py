import numpy as np
import pytest
from PIL import Image

from tagger import CATEGORIES, baseline, store


def test_prompts_cover_every_category():
    assert set(baseline.PROMPTS) == set(CATEGORIES)
    for category, prompts in baseline.PROMPTS.items():
        assert len(prompts) >= 3, category
        assert len(set(prompts)) == len(prompts), category


@pytest.fixture(scope="module")
def clip():
    # Needs torch + open_clip (the "ml" group), which CI does not install: skip there.
    pytest.importorskip("open_clip")
    return baseline.ClipBaseline()


def solid(rgb) -> Image.Image:
    return Image.fromarray(np.full((224, 224, 3), rgb, dtype=np.uint8))


def test_predict_returns_probabilities_for_every_category(clip):
    [scores] = clip.predict([solid((20, 90, 180))])
    assert set(scores) == set(CATEGORIES)
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)


def test_empty_blue_water_is_scenery(clip):
    [scores] = clip.predict([solid((20, 90, 180))])
    assert max(scores, key=scores.get) == "scenery"


def test_tag_store_predicts_each_image_once(clip, tmp_path):
    conn = store.connect(":memory:")
    media_id = store.add_media(conn, "/x/GH01.MP4", "video", fingerprint="fp")
    for n in range(3):
        path = tmp_path / f"{n}.jpg"
        solid((20, 90, 180)).save(path)
        store.add_image(conn, media_id, str(path))

    assert baseline.tag_store(conn, clip, batch_size=2) == 3
    assert baseline.tag_store(conn, clip) == 0  # already tagged
    count = conn.execute("SELECT COUNT(*) FROM predictions WHERE model = ?", (baseline.MODEL_ID,))
    assert count.fetchone()[0] == 3
