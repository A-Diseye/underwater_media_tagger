import json

import pytest

from tagger import eval_split as ts
from tagger import labeling, store
from tagger.baseline import MODEL_ID


@pytest.fixture
def conn():
    """Turtles in 4 clips, fish in 6, plus 4 clips with no labels yet."""
    c = store.connect(":memory:")
    clips = [("turtles", f"T{i}") for i in range(4)] + [("fish", f"F{i}") for i in range(6)]
    clips += [(None, f"U{i}") for i in range(4)]
    for k, (label, name) in enumerate(clips):
        media = store.add_media(c, f"/d/trip/{name}.MP4", "video", f"fp{k}", "trip")
        image = store.add_image(c, media, f"/f/{name}.jpg", t_s=0.0)
        store.add_prediction(c, image, MODEL_ID, {cat: 1 / 6 for cat in labeling.CATEGORIES})
        if label:
            store.set_label(c, image, label, labeling.SOURCE, "random")
    yield c
    c.close()


def test_every_clip_lands_in_exactly_one_half(conn):
    split = ts.make_split(conn)
    dev, holdout = set(split["dev"]), set(split["holdout"])
    assert not dev & holdout and len(dev | holdout) == 14


def test_rare_categories_are_shared_between_halves(conn):
    split = ts.make_split(conn)
    counts = ts.summary(conn, {h: set(p) for h, p in split.items()})
    assert counts["dev"]["turtles"] == counts["holdout"]["turtles"] == 2
    assert counts["dev"]["fish"] == counts["holdout"]["fish"] == 3


def test_split_is_repeatable_and_frozen(conn, tmp_path):
    assert ts.make_split(conn, seed=0) == ts.make_split(conn, seed=0)
    path = tmp_path / "split.json"
    ts.save_split(ts.make_split(conn), path)
    with pytest.raises(FileExistsError):
        ts.save_split(ts.make_split(conn), path)
    assert set(json.loads(path.read_text())) == {"dev", "holdout"}


def test_media_ids_maps_a_half_to_database_ids(conn, tmp_path):
    path = tmp_path / "split.json"
    ts.save_split(ts.make_split(conn), path)
    ids = ts.media_ids(conn, "holdout", path)
    paths = {r["path"] for r in conn.execute("select id, path from media") if r["id"] in ids}
    assert paths == ts.load_split(path)["holdout"]
