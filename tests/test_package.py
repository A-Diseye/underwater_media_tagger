from tagger import CATEGORIES


def test_six_unique_categories():
    assert len(CATEGORIES) == 6
    assert len(set(CATEGORIES)) == 6
