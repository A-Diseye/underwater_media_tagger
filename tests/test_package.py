from tagger import CATEGORIES


def test_seven_unique_categories():
    assert len(CATEGORIES) == 7
    assert len(set(CATEGORIES)) == 7
