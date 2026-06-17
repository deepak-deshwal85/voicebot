from utils.search_query import is_valid_search_query


def test_rejects_short_and_numeric_queries() -> None:
    assert is_valid_search_query("1") is False
    assert is_valid_search_query("at") is False
    assert is_valid_search_query("com") is False


def test_accepts_meaningful_queries() -> None:
    assert is_valid_search_query("Deepak") is True
    assert is_valid_search_query("company information") is True
    assert is_valid_search_query("ETF mutual fund differences") is True
