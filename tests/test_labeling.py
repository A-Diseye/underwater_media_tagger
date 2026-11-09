import random

import pytest

from tagger import CATEGORIES, labeling, store
from tagger.baseline import MODEL_ID


def ranked_scores(*top: str) -> dict[str, float]:
    """Scores where the given categories rank 1st, 2nd, ... in that order.

    The remaining categories get small, distinct scores (like real CLIP output, no ties),
    ranked from the end of CATEGORIES: scenery, then diver, ... so turtles is always last.
    """
    weights = [0.5, 0.25, 0.12, 0.06]
    scores = dict(zip(top, weights, strict=False))
    rest = [c for c in reversed(CATEGORIES) if c not in top]
    for i, c in enumerate(rest):
        scores[c] = 0.01 / (i + 1)  # all below every explicitly ranked score
    return scores


@pytest.fixture
def conn():
    c = store.connect(":memory:")
    yield c
    c.close()


def add_clip(conn, trip, frames, name="GH01.MP4"):
    """frames: list of (t_s, scores). Returns image ids in order."""
    media_id = store.add_media(conn, f"/d/{trip}/{name}", "video", f"{trip}-{name}", trip)
    ids = []
    for t, scores in frames:
        image_id = store.add_image(conn, media_id, f"{trip}/{name}_{t}.jpg", t_s=t)
        store.add_prediction(conn, image_id, MODEL_ID, scores)
        ids.append(image_id)
    return ids


def label(conn, image_id, category, selection="random"):
    store.set_label(conn, image_id, category, labeling.SOURCE, selection)


def test_random_picks_come_from_the_whole_library(conn):
    ids = add_clip(conn, "azores", [(t * 20, ranked_scores("scenery")) for t in range(5)])
    pick = labeling.next_frame(conn, random.Random(0), random_share=1.0)
    assert pick.selection == "random" and pick.image_id in ids


def test_targets_category_furthest_below_quota(conn):
    [turtle] = add_clip(conn, "azores", [(0, ranked_scores("turtles"))], "A.MP4")
    add_clip(conn, "azores", [(t * 20, ranked_scores("rays")) for t in range(3)], "B.MP4")
    already = add_clip(conn, "azores", [(t * 20, ranked_scores("rays")) for t in range(2)], "C.MP4")
    for image_id in already:
        label(conn, image_id, "rays")
    # Every category except rays has 0 labels; turtles is the only one with a candidate pool.
    pick = labeling.next_frame(conn, random.Random(0), random_share=0.0)
    assert pick.image_id == turtle and pick.selection == "clip_top3:turtles"


def test_top3_pool_includes_runner_ups_but_not_4th(conn):
    [third] = add_clip(conn, "a", [(0, ranked_scores("rays", "scenery", "sharks"))], "A.MP4")
    add_clip(conn, "a", [(0, ranked_scores("rays", "scenery", "fish", "sharks"))], "B.MP4")
    picks = {labeling.next_frame(conn, random.Random(s), random_share=0.0) for s in range(20)}
    shark_picks = {p.image_id for p in picks if p and p.selection == "clip_top3:sharks"}
    assert shark_picks == {third}


def test_skips_frames_near_a_labeled_frame_in_the_same_clip(conn):
    ids = add_clip(conn, "azores", [(t, ranked_scores("turtles")) for t in (0, 4, 8, 12, 30)])
    label(conn, ids[1], "turtles")  # labeled at t=4: blocks 0, 8, 12 (within ±10s), not 30
    seen = {
        labeling.next_frame(conn, random.Random(s), random_share=1.0).image_id for s in range(20)
    }
    assert seen == {ids[4]}


def test_unusable_labels_do_not_block_neighbors(conn):
    ids = add_clip(conn, "azores", [(0, ranked_scores("scenery")), (2, ranked_scores("scenery"))])
    label(conn, ids[0], "unusable")  # boat frame next to a real dive frame
    assert labeling.next_frame(conn, random.Random(0), random_share=1.0).image_id == ids[1]


def test_uploads_and_skipped_frames_are_never_picked(conn):
    [upload] = add_clip(conn, "uploads", [(0, ranked_scores("turtles"))])
    [skipped, keep] = add_clip(
        conn, "azores", [(0, ranked_scores("turtles")), (60, ranked_scores("turtles"))]
    )
    for seed in range(10):
        pick = labeling.next_frame(conn, random.Random(seed), exclude_ids={skipped})
        assert pick.image_id == keep
    assert upload != keep


def test_category_stops_being_targeted_at_quota_or_give_up(conn):
    ids = add_clip(conn, "azores", [(t * 20, ranked_scores("sharks", "rays")) for t in range(6)])
    # Shark candidates keep turning out to be rays.
    for image_id in ids[:3]:
        label(conn, image_id, "rays", selection="clip_top3:sharks")
    status = labeling.category_status(conn, quota=2, give_up_after=3)
    assert status["sharks"] == "gave up"
    assert status["rays"] == "done"
    assert status["turtles"] == "active"
    pick = labeling.next_frame(conn, random.Random(0), random_share=0.0, quota=2, give_up_after=3)
    # Finished and given-up categories are no longer targeted.
    assert pick.selection not in ("clip_top3:sharks", "clip_top3:rays")


def test_returns_none_when_everything_is_labeled(conn):
    [only] = add_clip(conn, "azores", [(0, ranked_scores("fish"))])
    label(conn, only, "fish")
    assert labeling.next_frame(conn, random.Random(0)) is None


def test_browse_helpers(conn):
    ids = add_clip(conn, "azores", [(4, ranked_scores("fish")), (0, ranked_scores("fish"))])
    label(conn, ids[0], "fish", selection="browse")
    [clip] = labeling.library_clips(conn, "azores")
    assert (clip["frames"], clip["labeled"]) == (2, 1)
    frames = labeling.clip_frames(conn, clip["id"])
    assert [f["t_s"] for f in frames] == [0, 4] and frames[1]["label"] == "fish"


def test_undo_and_selection_validation(conn):
    [image_id] = add_clip(conn, "azores", [(0, ranked_scores("fish"))])
    with pytest.raises(ValueError):
        store.set_label(conn, image_id, "fish", labeling.SOURCE, selection="clip_top3:dolphins")
    label(conn, image_id, "fish")
    store.delete_label(conn, image_id, labeling.SOURCE)
    assert labeling.label_counts(conn)["fish"] == 0
