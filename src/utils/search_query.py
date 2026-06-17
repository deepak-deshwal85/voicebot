from __future__ import annotations

import re

_INVALID_QUERY_MESSAGE = (
    "Please ask a complete question with a few words so I can search accurately."
)


def is_valid_search_query(query: str) -> bool:
    """Reject STT fragments and numeric-only queries before expensive retrieval."""
    cleaned = query.strip()
    if len(cleaned) < 4:
        return False
    if cleaned.isdigit():
        return False

    words = re.findall(r"[a-z0-9]+", cleaned.lower())
    if not words:
        return False
    return not (len(words) == 1 and len(words[0]) < 5)


def invalid_search_query_message() -> str:
    return _INVALID_QUERY_MESSAGE
