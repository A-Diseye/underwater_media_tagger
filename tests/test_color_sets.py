"""Frame sets for color correction: split by clip, test frames and uploads left out."""

from tagger import color_sets, store


def _library(conn):
    """Two trips, 30 clips, 2 frames each, every frame tagged turtles by model m."""
    for n in range(30):
        trip = "azores" if n % 2 else "bahamas"
        media = store.add_media(conn, f"/clips/{trip}/GX{n:04d}.MP4", "video", f"fp{n}", trip)
        for t in (0.0, 2.0):
            image = store.add_image(conn, media, f"frames/{n}_{t}.jpg", t_s=t)
            scores = {c: (0.9 if c == "turtles" else 0.02) for c in store.CATEGORIES}
            store.add_prediction(conn, image, "m", scores)
    up = store.add_media(conn, "/uploads/x.jpg", "photo", "fp-up", color_sets.UPLOADS)
    store.add_prediction(conn, store.add_image(conn, up, "frames/up.jpg"), "m", scores)


def test_rare_subjects_are_dealt_to_every_set():
    conn = store.connect(":memory:")
    for n in range(20):  # 5 clips with a confident turtle among 20
        media = store.add_media(conn, f"/clips/GX{n:04d}.MP4", "video", f"fp{n}", "azores")
        label = "turtles" if n < 5 else "diver"
        scores = {c: (0.9 if c == label else 0.02) for c in store.CATEGORIES}
        store.add_prediction(conn, store.add_image(conn, media, f"f{n}.jpg", t_s=0.0), "m", scores)
    sets = color_sets.assign_sets(conn, "m")
    turtle_sets = [sets[f"/clips/GX{n:04d}.MP4"] for n in range(5)]
    assert sorted(turtle_sets) == ["A", "B", "C", "pool", "pool"]
    assert sets == color_sets.assign_sets(conn, "m")  # fixed


def test_candidates_skip_test_frames_and_uploads():
    conn = store.connect(":memory:")
    _library(conn)
    everything = [r for s in color_sets.SETS for r in color_sets.candidates(conn, "m", s)]
    assert len(everything) == 60  # the upload is left out
    labeled = everything[0]["image_id"]
    store.set_label(conn, labeled, "turtles", source="test_labeling", selection="random")
    after = [r for s in color_sets.SETS for r in color_sets.candidates(conn, "m", s)]
    assert labeled not in {r["image_id"] for r in after} and len(after) == 59
    for s in color_sets.SETS:  # a clip's frames all land in one set
        clips = {r["source"] for r in color_sets.candidates(conn, "m", s)}
        for other in set(color_sets.SETS) - {s}:
            assert not clips & {r["source"] for r in color_sets.candidates(conn, "m", other)}


def test_draw_spreads_over_trips_and_clips_and_is_fixed():
    rows = [
        {"image_id": i, "label": "turtles", "confidence": 0.9, "trip": trip, "source": clip}
        for i, (trip, clip) in enumerate(
            [("azores", "a1")] * 5 + [("azores", "a2")] * 5 + [("bahamas", "b1")] * 5
        )
    ]
    picked = color_sets.draw(rows, 3, categories=("turtles",))
    assert sorted(r["source"] for r in picked) == ["a1", "a2", "b1"]  # one per clip first
    assert picked == color_sets.draw(rows, 3, categories=("turtles",))
    assert len(color_sets.draw(rows, 5, categories=("turtles",))) == 5  # then repeats clips
    assert color_sets.draw(rows, 3, categories=("turtles",), min_confidence=0.95) == []
