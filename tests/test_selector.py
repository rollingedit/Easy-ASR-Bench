from app.model_selector import parse_selection


def test_parse_selection_range():
    assert parse_selection("1-3", 5) == [1, 2, 3]


def test_parse_selection_compact_single_digits():
    assert parse_selection("123", 5) == [1, 2, 3]
