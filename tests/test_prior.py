"""Base-rate (label shift) correction: estimate a library's category mix without labels."""

import json

import numpy as np
import pytest

from tagger import CATEGORIES, prior, store


def test_reweight_moves_probabilities_to_the_new_mix():
    probs = np.array([[0.5, 0.5]])
    out = prior.reweight(probs, np.array([0.5, 0.5]), np.array([0.9, 0.1]))
    assert out[0] == pytest.approx([0.9, 0.1])


def test_em_recovers_the_true_mix_from_unlabeled_predictions():
    # Two classes, each a bell curve of some image feature x. A model trained on a 50/50 mix
    # gives exact posteriors; the library is 90% class 0. EM finds the 90/10 from x alone.
    rng = np.random.default_rng(0)
    truth = rng.random(5000) < 0.1
    x = rng.normal(np.where(truth, 1.5, -1.5), 1.0)
    logit = 3.0 * x  # log p(x|1) - log p(x|0) for unit-variance curves at -1.5 and 1.5
    probs = np.column_stack([1 / (1 + np.exp(logit)), 1 / (1 + np.exp(-logit))])
    mix = prior.estimate_prior(probs, np.array([0.5, 0.5]))
    assert mix == pytest.approx([0.9, 0.1], abs=0.02)
    # An uncertain frame now leans toward the common class.
    assert prior.reweight(np.array([[0.4, 0.6]]), np.array([0.5, 0.5]), mix)[0, 0] > 0.5


def test_a_category_absent_from_a_trip_fades_out():
    # Weak 40% "turtle" guesses on frames that are all scenery: the trip's turtle share shrinks
    # toward zero, and the tags flip to scenery.
    t, s = CATEGORIES.index("turtles"), CATEGORIES.index("scenery")
    probs = np.full((50, len(CATEGORIES)), 0.04)
    probs[:, t], probs[:, s] = 0.40, 0.44
    probs[:10, t], probs[:10, s] = 0.44, 0.40  # the model tags 10 of them turtles
    mix = prior.estimate_prior(probs, np.ones(len(CATEGORIES)))
    assert mix[t] < mix[s] and mix[s] > 0.5
    assert (prior.reweight(probs, np.ones(len(CATEGORIES)) / 6, mix).argmax(1) == s).all()


def test_correct_store_writes_a_new_model_per_trip():
    conn = store.connect(":memory:")

    def scores(label, p):
        rest = (1 - p) / (len(CATEGORIES) - 1)
        return {c: (p if c == label else rest) for c in CATEGORIES}

    for trip, labels in {
        "azores": ["scenery"] * 8 + ["turtles"],
        "bahamas": ["turtles"] * 5,
    }.items():
        media = store.add_media(conn, f"/d/{trip}.MP4", "video", trip, trip)
        for k, label in enumerate(labels):
            image = store.add_image(conn, media, f"/f/{trip}{k}.jpg", t_s=float(k))
            store.add_prediction(conn, image, "m", scores(label, 0.5))
    mixes = prior.correct_store(conn, "m", np.ones(len(CATEGORIES)))
    assert set(mixes) == {"azores", "bahamas"}
    assert mixes["azores"]["scenery"] > mixes["bahamas"]["scenery"]
    rows = conn.execute("SELECT scores FROM predictions WHERE model = 'm_prior_trip'").fetchall()
    assert len(rows) == 14
    assert all(abs(sum(json.loads(r["scores"]).values()) - 1) < 1e-9 for r in rows)
