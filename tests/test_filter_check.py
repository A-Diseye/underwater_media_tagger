import json

import pytest

from training import filter_check as fc


def write_log(path, events):
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def check(reason="r"):
    return {
        "underwater": True,
        "animal_is_main_subject": True,
        "category": "rays",
        "reason": reason,
    }


@pytest.fixture
def curation(tmp_path):
    write_log(
        tmp_path / "log.jsonl",
        [
            {
                "event": "photo",
                "observation_id": 1,
                "category": "rays",
                "reason": None,
                "check": check(),
                "path": "data/train/rays/inat_1.jpg",
                "photo_url": "u1",
            },
            {
                "event": "photo",
                "observation_id": 2,
                "category": "rays",
                "reason": "not_underwater",
                "check": check(),
                "path": None,
                "photo_url": "u2",
            },
            {
                "event": "photo",
                "observation_id": 3,
                "category": "rays",
                "reason": "unreadable",
                "check": None,
                "path": None,
                "photo_url": "u3",
            },
            {"event": "agent_turn", "cost_usd": 0.01},
        ],
    )
    write_log(
        tmp_path / "diver_log.jsonl",
        [
            {
                "key": "commons:9",
                "category": "diver",
                "reason": None,
                "check": check(),
                "path": "data/val/diver/commons_9.jpg",
                "photo_url": "u9",
            }
        ],
    )
    return tmp_path


def test_collects_only_decisions_haiku_made_from_all_logs(curation):
    found = {d["id"]: d for d in fc.decisions(curation)}
    assert set(found) == {"inat:1", "inat:2", "commons:9"}
    assert found["inat:1"]["image"].endswith("inat_1.jpg")  # kept: the saved file
    assert found["inat:2"]["image"] == "u2"  # rejected: the original URL


def test_sample_is_fixed_and_balanced(curation):
    all_decisions = fc.decisions(curation)
    first, again = fc.sample(all_decisions, per_side=1), fc.sample(all_decisions, per_side=1)
    assert [d["id"] for d in first] == [d["id"] for d in again]
    assert sorted(d["kept"] for d in first) == [False, True]


def test_verdicts_resume_and_summarize(curation):
    checked = fc.decisions(curation)
    by_id = {d["id"]: d for d in checked}
    fc.save_verdict(curation, by_id["inat:1"], "wrong")
    fc.save_verdict(curation, by_id["inat:1"], "right")  # changed my mind: later one wins
    fc.save_verdict(curation, by_id["inat:2"], "wrong")
    verdicts = fc.load_verdicts(curation)
    assert verdicts == {"inat:1": "right", "inat:2": "wrong"}
    totals = fc.summary(checked, verdicts)
    assert totals["keeps"] == {"checked": 1, "right": 1}
    assert totals["rejects"] == {"checked": 1, "right": 0}
    assert totals["wrong_by_category"] == {"rays": 1}
    with pytest.raises(ValueError):
        fc.save_verdict(curation, by_id["inat:1"], "maybe")
