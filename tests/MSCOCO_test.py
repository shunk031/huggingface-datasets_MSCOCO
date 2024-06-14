from MSCOCO import CATEGORIES, SUPER_CATEGORIES


def test_consts():
    assert len(CATEGORIES) == 80
    assert len(SUPER_CATEGORIES) == 12
