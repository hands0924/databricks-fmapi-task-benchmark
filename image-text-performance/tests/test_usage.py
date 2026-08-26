"""usage.build_usage_query 검증 — request_id는 외부 입력이므로 SQL 리터럴 안전성이 핵심."""

import pytest

from src.cost.usage import build_usage_query


def test_empty_request_ids_returns_no_row_query():
    assert build_usage_query([]) == "SELECT * FROM system.ai_gateway.usage WHERE 1=0"


def test_valid_request_ids_are_quoted():
    q = build_usage_query(["abc-123", "req_9:x.y"])
    assert "'abc-123', 'req_9:x.y'" in q
    assert "FROM system.ai_gateway.usage" in q


@pytest.mark.parametrize(
    "bad",
    [
        "abc' OR '1'='1",
        "x'); DROP TABLE users; --",
        "abc\\",
        "abc\nid",
        "a" * 129,
        "",
        None,
        123,
    ],
)
def test_unsafe_request_ids_are_rejected(bad):
    """SQL injection 시도는 쿼리에 섞이지 않고 즉시 실패해야 한다."""
    with pytest.raises(ValueError):
        build_usage_query([bad])
