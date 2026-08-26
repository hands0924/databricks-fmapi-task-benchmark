"""ai_gateway.usage 테이블 조인 (plan §10).

벤치마크의 request_id를 system.ai_gateway.usage와 연결해
실제 호출의 latency·토큰 통계를 가져온다.

**권한 상태** (plan §10):
    - system.ai_gateway.usage: 접근 가능 (현재)
    - system.billing: 접근 불가 → pricing.yaml이 1차 소스, 권한 확보 시 사후 검증 가능

**데이터 흐름**:
    1. 벤치마크 실행 시 각 FMAPI 호출의 request_id를 결과에 저장
    2. 결과들을 모아 request_id 리스트 생성
    3. build_usage_query()로 SQL 작성
    4. fetch_usage()로 실행해 usage 행 조회
    5. 원본 결과와 merge 해 latency·토큰 통계 추가
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# request_id는 FMAPI 응답(헤더 x-request-id 또는 body id)에서 오는 외부 입력이다.
# SQL에 리터럴로 들어가므로 보수적 allowlist로 검증한다.
_REQUEST_ID_RE = re.compile(r"\A[A-Za-z0-9._:-]{1,128}\Z")


# === 상수: ai_gateway.usage 컬럼 목록 (plan §10) ===
# Phase 1에서 실제 쿼리 실행 시 이 컬럼들을 SELECT 한다.
USAGE_COLUMNS = [
    "latency_ms",
    "time_to_first_byte_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "token_details",
    "endpoint_name",
    "status_code",
    "event_time",
    "request_id",
]

# Phase 0에서 추가로 필요한 nested 컬럼들 (복합 사용)
USAGE_NESTED_COLUMNS = [
    "token_details.cache_read_input_tokens",
    "token_details.cache_creation_input_tokens",
    "token_details.output_reasoning_tokens",
    "token_details.output_token_details",  # completion_tokens_details 역할
]


def build_usage_query(
    request_ids: list[str],
    warehouse_id: str = "2c4aa6fec2649553",
) -> str:
    """system.ai_gateway.usage에서 request_id로 조인할 SQL 작성.

    Args:
        request_ids: 조회할 request_id 리스트
        warehouse_id: Databricks warehouse ID (기본값: plan §10의 공식 warehouse)

    Returns:
        SQL 쿼리 문자열. 형식:
            SELECT <USAGE_COLUMNS> FROM system.ai_gateway.usage
            WHERE request_id IN (...)
            ORDER BY event_time DESC

    Raises:
        ValueError: request_id가 allowlist(영숫자·`.`·`_`·`:`·`-`, 128자 이내)를 벗어날 때.

    Note:
        - request_id는 외부 입력이므로 SQL에 넣기 전 allowlist로 검증한다(injection 방지).
        - warehouse_id는 쿼리 실행 시 지정(build_usage_query는 SELECT만 생성).
    """
    if not request_ids:
        logger.warning("request_ids is empty; returning empty result query")
        return "SELECT * FROM system.ai_gateway.usage WHERE 1=0"

    quoted_ids = ", ".join(f"'{_validate_request_id(rid)}'" for rid in request_ids)

    # USAGE_COLUMNS를 쿼리에 포함
    columns_str = ", ".join(USAGE_COLUMNS)

    query = f"""
SELECT
    {columns_str}
FROM system.ai_gateway.usage
WHERE request_id IN ({quoted_ids})
ORDER BY event_time DESC
""".strip()

    return query


def _validate_request_id(request_id: Any) -> str:
    """SQL 리터럴로 쓸 수 있는 request_id만 통과시킨다."""
    if not isinstance(request_id, str) or not _REQUEST_ID_RE.match(request_id):
        raise ValueError(f"unsafe request_id for SQL literal: {request_id!r}")
    return request_id


def fetch_usage(
    request_ids: list[str],
    profile: str = "ai_devtools",
    warehouse_id: str = "2c4aa6fec2649553",
) -> list[dict[str, Any]]:
    """system.ai_gateway.usage에서 usage 행 조회 (Phase 1).

    Args:
        request_ids: 조회할 request_id 리스트
        profile: Databricks CLI 프로파일명 (기본값: ai_devtools)
        warehouse_id: 실행할 warehouse ID (기본값: plan §10 공식값)

    Returns:
        [
            {
                'request_id': str,
                'latency_ms': float,
                'input_tokens': int,
                'output_tokens': int,
                'reasoning_tokens': int (nested),
                'token_details': dict,
                ...
            },
            ...
        ]

    Raises:
        NotImplementedError: Phase 1에서 구현 예정.

    Note:
        실행 방식:
            1. build_usage_query(request_ids)로 SQL 생성
            2. databricks api post /api/2.0/sql/statements로 실행
               (profile 로드 → host·token·warehouse_id 결정)
            3. response['result']['data_array'] 파싱
            4. latency_ms, token_details 등 nested 구조 정규화

        Permission 상태:
            - system.ai_gateway.usage: 접근 가능 (현 권한)
            - system.billing: 접근 불가 (미권한) — pricing.yaml이 1차 소스

        Phase 0 테스트:
            - build_usage_query만 수행, SQL 문자열 검증
            - mock usage dict로 cost 계산 테스트
    """
    raise NotImplementedError(
        "Phase 1: ai_gateway.usage 조인 실행 — "
        "Databricks SQL API(/api/2.0/sql/statements) 호출, "
        "token_details nested 구조 파싱 필요. "
        "Phase 0에서는 build_usage_query 및 mock data로 검증."
    )
