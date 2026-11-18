"""The library search agent: its read-only tools against a small library, and the tool runner
loop against a fake Claude API (real SDK client, scripted HTTP responses, no key needed)."""

import json
import sqlite3

import anthropic
import httpx2
import numpy as np
import pytest

from tagger import CATEGORIES, search, similar, store

FT, CLIP = "v2_both_s0", "clip-vitb32-zeroshot-6cat"


def scores(label):
    return {c: (0.9 if c == label else 0.02) for c in CATEGORIES}


@pytest.fixture
def library(tmp_path):
    """Two Azores clips and a Bahamas photo, opened read-only like the app does.

    GX1: rays at 0 s and 10 s (CLIP said fish for the first), scenery at 60 s.
    GX2: a ray at 0 s. IMG: a turtle photo, corrected by me to fish.
    """
    path = tmp_path / "tagger.db"
    conn = store.connect(path)
    gx1 = store.add_media(conn, "/d/Azores/GX1.MP4", "video", "a", "azores")
    gx2 = store.add_media(conn, "/d/Azores/GX2.MP4", "video", "b", "azores")
    img = store.add_media(conn, "/d/Bahamas/IMG.jpg", "photo", "c", "bahamas")
    frames = [
        (gx1, 0.0, "rays", "fish"),
        (gx1, 10.0, "rays", "rays"),
        (gx1, 60.0, "scenery", "scenery"),
        (gx2, 0.0, "rays", "rays"),
        (img, None, "turtles", "turtles"),
    ]
    ids = []
    for k, (media, t, ft, clip) in enumerate(frames):
        image_id = store.add_image(conn, media, f"/f/{k}.jpg", t_s=t)
        store.add_prediction(conn, image_id, FT, scores(ft))
        store.add_prediction(conn, image_id, CLIP, scores(clip))
        ids.append(image_id)
    store.set_label(conn, ids[4], "fish", store.CORRECTION)
    conn.commit()
    conn.close()
    # Image 0 looks like image 3 (another clip) more than like image 1 (same clip).
    vectors = np.array(
        [[1, 0, 0], [0.8, 0.6, 0], [0, 1, 0], [0.99, 0.14, 0], [0, 0, 1]], dtype=np.float32
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = similar.Index(np.array(ids), vectors)
    ro = store.connect_readonly(path)
    yield search.LibraryTools(ro, index, model_id=FT, clip_id=CLIP), ids
    ro.close()


def test_tools_cannot_write_to_the_library(library):
    lib, ids = library
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        lib.conn.execute("DELETE FROM labels")


def test_find_and_count_use_the_gallery_tags(library):
    lib, ids = library
    found = lib.find_images(categories=["rays"], trips=["azores"])
    assert found["total_matching"] == 3
    assert {i["clip"] for i in found["images"]} == {"GX1.MP4", "GX2.MP4"}
    fish = lib.find_images(categories=["fish"])["images"]
    assert [i["image_id"] for i in fish] == [ids[4]]  # my correction, not the model's turtle
    assert "corrected by user" in fish[0]["tag_source"]
    by_trip = lib.count_by("trip", categories=["rays"])
    assert by_trip == [{"group": "azores", "images": 3}]
    assert lib.find_images(max_confidence=0.5)["total_matching"] == 0


def test_list_trips_and_disagreements(library):
    lib, ids = library
    assert {t["trip"]: t["clips"] for t in lib.list_trips()} == {"azores": 2, "bahamas": 1}
    disagreements = lib.model_disagreements()
    assert disagreements["total_disagreements"] == 1
    [image] = disagreements["images"]
    assert (image["fine_tuned_said"], image["clip_said"]) == ("rays", "fish")


def test_clip_frames_and_details(library):
    lib, ids = library
    near = lib.clip_frames(ids[0], window_s=20)["images"]
    assert [i["image_id"] for i in near] == [ids[0], ids[1]]  # 60 s is outside the window
    assert near[1]["at"] == "0:10"
    details = lib.image_details(ids[0])
    assert details["fine_tuned_top3"]["rays"] == 0.9 and details["clip_top3"]["fish"] == 0.9
    with pytest.raises(ValueError, match="no image"):
        lib.image_details(999)


def test_similar_images_skip_the_same_clip_by_default(library):
    lib, ids = library
    other = lib.similar_images(ids[0], limit=2)["images"]
    assert other[0]["image_id"] == ids[3] and ids[1] not in [i["image_id"] for i in other]
    anywhere = lib.similar_images(ids[0], limit=1, other_clips_only=False)["images"]
    assert anywhere[0]["image_id"] == ids[3]
    assert anywhere[0]["similarity"] > 0.9


def test_every_returned_image_is_recorded(library):
    lib, ids = library
    lib.find_images(categories=["scenery"])
    lib.clip_frames(ids[0])
    assert lib.shown == [ids[2], ids[0], ids[1]]


def test_similar_index_add_and_nearest(tmp_path):
    conn = store.connect(":memory:")
    media = store.add_media(conn, "/d/a.MP4", "video", "fp", "azores")
    ids = [store.add_image(conn, media, f"/f/{i}.jpg", t_s=i * 10.0) for i in range(3)]

    class FakeClip:
        def embed_paths(self, paths):
            rows = {"/f/0.jpg": [1, 0], "/f/1.jpg": [0.9, 0.1], "/f/2.jpg": [0, 1]}
            return np.array([rows[p] for p in paths], dtype=np.float32)

    path = tmp_path / "clip.npz"
    assert similar.add(conn, FakeClip(), path=path) == 3
    assert similar.add(conn, FakeClip(), path=path) == 0  # nothing new
    index = similar.Index.load(path)
    assert [i for i, _ in index.nearest(ids[0], 2)] == [ids[1], ids[2]]
    assert [i for i, _ in index.nearest(ids[0], 1, exclude={ids[1]})] == [ids[2]]


def api_message(content, stop_reason, msg_id):
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": search.MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 100},
    }


def fake_claude(replies):
    """A real SDK client whose HTTP requests get scripted replies, in order."""
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=replies[len(requests) - 1])

    client = anthropic.Anthropic(
        api_key="test",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return client, requests


def test_ask_runs_the_tools_claude_picks_and_keeps_the_history(library):
    lib, ids = library
    client, requests = fake_claude(
        [
            api_message(
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "count_by",
                        "input": {"group": "trip", "categories": ["rays"]},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "find_images",
                        "input": {"categories": ["rays"], "limit": 2},
                    },
                ],
                "tool_use",
                "msg_1",
            ),
            api_message(
                [{"type": "text", "text": "The Azores had all 3 rays."}], "end_turn", "msg_2"
            ),
        ]
    )
    answer, history = search.ask(client, lib, [], "Which trip had the most rays?")

    assert answer.text == "The Azores had all 3 rays."
    assert [c["name"] for c in answer.tool_calls] == ["count_by", "find_images"]
    assert len(answer.image_ids) == 2 and set(answer.image_ids) <= {ids[0], ids[1], ids[3]}
    assert answer.cost_usd > 0 and answer.stop_reason == "end_turn"
    # Both tool results went back to Claude in one message.
    results = requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_1", "toolu_2"]
    assert json.loads(results[0]["content"]) == [{"group": "azores", "images": 3}]
    # The history for the next question: question, tool calls, results, answer.
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert requests[0]["model"] == search.MODEL and requests[0]["fallbacks"] == "default"


def test_ask_reports_a_tool_error_to_claude(library):
    lib, ids = library
    client, requests = fake_claude(
        [
            api_message(
                [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "image_details",
                        "input": {"image_id": 999},
                    }
                ],
                "tool_use",
                "msg_1",
            ),
            api_message(
                [{"type": "text", "text": "That image doesn't exist."}], "end_turn", "msg_2"
            ),
        ]
    )
    answer, _ = search.ask(client, lib, [], "Tell me about image 999")
    result = requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and answer.image_ids == []
