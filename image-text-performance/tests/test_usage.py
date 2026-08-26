"""ai_gateway.usage 쿼리 빌더 테스트 (src/cost/usage.py)."""

import pytest
from src.cost.usage import USAGE_COLUMNS, build_usage_query, fetch_usage


def test_query_selects_all_usage_columns():
    q = build_usage_query(["req-1"])
    for col in USAGE_COLUMNS:
        assert col in q
    assert "FROM system.ai_gateway.usage" in q
    assert "ORDER BY event_time DESC" in q


def test_query_quotes_every_request_id():
    q = build_usage_query(["req-1", "req-2"])
    assert "WHERE request_id IN ('req-1', 'req-2')" in q


def test_query_is_stripped_single_statement():
    q = build_usage_query(["req-1"])
    assert q == q.strip()
    assert q.count("SELECT") == 1


def test_empty_request_ids_returns_no_rows_query(caplog):
    q = build_usage_query([])
    assert q == "SELECT * FROM system.ai_gateway.usage WHERE 1=0"


def test_empty_request_ids_warns(caplog):
    with caplog.at_level("WARNING"):
        build_usage_query([])
    assert "request_ids is empty" in caplog.text


def test_fetch_usage_is_phase1_stub():
    with pytest.raises(NotImplementedError):
        fetch_usage(["req-1"])
