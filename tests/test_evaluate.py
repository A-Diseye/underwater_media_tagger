import numpy as np
import pytest
from PIL import Image

from tagger import CATEGORIES, store
from tagger.baseline import MODEL_ID as CLIP
from training import evaluate as ev

FT = "convnext_tiny_depth_s0"


def scores_for(label):
    return {c: (0.9 if c == label else 0.02) for c in CATEGORIES}


@pytest.fixture
def conn(tmp_path):
    """Six labeled frames: CLIP gets the rays wrong, the fine-tuned model gets them right."""
    c = store.connect(":memory:")
    gopro = store.add_media(c, "/d/Videos/Azores/GX01.MP4", "video", "fp1", "azores")
    phone = store.add_media(c, "/d/x/bahamas_shared/Day 1/IMG.jpg", "photo", "fp2", "bahamas")
    frames = [  # (media, true label, CLIP's guess, fine-tuned guess, selection)
        (gopro, "rays", "fish", "rays", "random"),
        (gopro, "rays", "scenery", "rays", "clip_top3:rays"),
        (gopro, "scenery", "scenery", "scenery", "random"),
        (phone, "turtles", "turtles", "turtles", "browse"),
        (phone, "fish", "fish", "invertebrates", "random"),
        (gopro, "unusable", "scenery", "scenery", "random"),  # never scored
    ]
    for i, (media, true, clip, ft, selection) in enumerate(frames):
        path = tmp_path / f"{i}.jpg"
        Image.fromarray(np.zeros((36, 64, 3), dtype=np.uint8)).save(path)
        image_id = store.add_image(c, media, str(path), t_s=i * 30.0)
        store.add_prediction(c, image_id, CLIP, scores_for(clip))
        store.add_prediction(c, image_id, FT, scores_for(ft))
        store.set_label(c, image_id, true, "test_labeling", selection)
    yield c
    c.close()


def test_metrics_balanced_accuracy_weighs_categories_equally():
    rows = [{"true_label": "scenery", "predicted": "scenery"}] * 9 + [
        {"true_label": "turtles", "predicted": "scenery"}
    ]
    m = ev.metrics(rows)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["balanced_accuracy"] == pytest.approx(0.5)  # scenery 100%, turtles 0%
    assert m["per_class"]["turtles"] == {"n": 1, "recall": 0.0, "precision": None}
    assert m["per_class"]["fish"]["recall"] is None  # no fish: not part of the average


def test_unusable_is_excluded_and_models_share_frames(conn):
    rows = ev.shared_rows(conn, [CLIP, FT])
    assert len(rows[CLIP]) == len(rows[FT]) == 5
    assert ev.metrics(rows[CLIP])["per_class"]["rays"]["recall"] == 0.0
    assert ev.metrics(rows[FT])["per_class"]["rays"]["recall"] == 1.0


def test_slices_split_camera_and_random_only(conn):
    s = ev.slices(ev.shared_rows(conn, [FT])[FT])
    assert len(s["random-only"]) == 3 and len(s["phone"]) == 2 and len(s["gopro"]) == 3
    assert ev.camera("/d/x/bahamas_shared/a.jpg") == "phone"


def test_evaluate_writes_report_matrices_and_failures(conn, tmp_path):
    out = tmp_path / "results"
    result = ev.evaluate(conn, FT, [], out)
    assert result == {"frames": 5, "primary_wrong_shown": 1}  # the fish called invertebrates
    report = (out / "results.md").read_text()
    assert "fine-tuned (chosen)" in report and "zero-shot CLIP baseline" in report
    assert "| rays | 2 |" in report
    for model in (CLIP, FT):
        assert (out / f"confusion_{model}.png").exists()
    assert (out / "failures.jpg").exists()


def test_scores_one_half_and_locks_the_holdout(conn, tmp_path):
    from tagger import eval_split

    split_path = tmp_path / "split.json"
    eval_split.save_split(
        {"dev": ["/d/Videos/Azores/GX01.MP4"], "holdout": ["/d/x/bahamas_shared/Day 1/IMG.jpg"]},
        split_path,
    )
    dev = ev.evaluate(conn, FT, [], tmp_path / "dev", half="dev", split_path=split_path)
    assert dev["frames"] == 3  # the GoPro clip's usable frames only
    assert "**dev** half" in (tmp_path / "dev" / "results.md").read_text()
    with pytest.raises(SystemExit):
        ev.main(["--primary", FT, "--split", "holdout"])  # no --final
