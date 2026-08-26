"""TXT-3 표 구조 추출 (Cell-F1 메트릭).

이 태스크는 표 구조 추출을 텍스트 기반으로 구현한다.
- PubTabNet 데이터셋에서 HTML 테이블을 로드
- 모델에게 표의 텍스트 내용을 제시하고 HTML 표 마크업 생성을 요청
- 예측 HTML과 참조 HTML을 파싱해 Cell-F1 메트릭으로 평가

Cell-F1: 셀 단위 퍼지 매칭 (텍스트 정규화 및 0.8 이상 유사도 기준)
"""

from __future__ import annotations

import html.parser
import re
from typing import Any

from fuzzywuzzy import fuzz

from src.adapters.fmapi import FMAPIClient, build_text_message
from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry
from src.tasks.base import Task, Sample, register


def parse_html_table(html_str: str) -> list[tuple[int, int, str]]:
    """HTML 테이블을 파싱해 (row, col, text) 셀 리스트로 변환.

    Args:
        html_str: HTML 테이블 마크업 (예: "<table><tr><td>...</td></tr></table>")

    Returns:
        [(row_idx, col_idx, cell_text), ...] 리스트
        row_idx, col_idx는 0-based 인덱스
    """
    if not isinstance(html_str, str):
        raise TypeError(f"html_str must be a string, got {type(html_str).__name__}")

    cells = []

    # 간단한 정규식 기반 파싱 (html.parser보다 빠르고 견고)
    # <tr>...</tr> 블록 추출
    tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
    td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL | re.IGNORECASE)

    rows = tr_pattern.findall(html_str)
    for row_idx, row_html in enumerate(rows):
        col_idx = 0
        for col_idx, cell_match in enumerate(td_pattern.finditer(row_html)):
            cell_text = cell_match.group(1)
            # HTML 태그 제거 및 공백 정규화
            cell_text = re.sub(r'<[^>]+>', '', cell_text)
            cell_text = re.sub(r'\s+', ' ', cell_text).strip()

            if cell_text:  # 비어있지 않은 셀만 포함
                cells.append((row_idx, col_idx, cell_text))

    return cells


def cell_f1_score(pred_cells: list[tuple[int, int, str]],
                  gold_cells: list[tuple[int, int, str]]) -> float:
    """Cell-F1 점수 계산 (셀 단위 퍼지 매칭).

    알고리즘:
    1. 위치(row, col)와 텍스트 유사도(fuzzywuzzy, 임계값 0.8)로 매칭
    2. 매칭된 셀 수 / 전체 셀 수로 F1 계산

    Args:
        pred_cells: 예측 셀 리스트 [(row, col, text), ...]
        gold_cells: 정답 셀 리스트 [(row, col, text), ...]

    Returns:
        F1 점수 [0, 1]
    """
    if not gold_cells:
        return 1.0 if not pred_cells else 0.0
    if not pred_cells:
        return 0.0

    # 매칭 전략: 각 gold 셀에 대해 가장 유사한 pred 셀 찾기
    matched_gold = set()
    matched_pred = set()

    for gold_idx, (gold_row, gold_col, gold_text) in enumerate(gold_cells):
        best_pred_idx = -1
        best_score = 0.0

        for pred_idx, (pred_row, pred_col, pred_text) in enumerate(pred_cells):
            if pred_idx in matched_pred:
                continue

            # 위치 가까울수록, 텍스트 유사할수록 높은 점수
            # 텍스트 유사도 (fuzzywuzzy)
            text_sim = fuzz.ratio(gold_text.lower(), pred_text.lower()) / 100.0

            # 위치 유사도 (같은 위치일수록 1.0)
            pos_sim = 1.0 if (gold_row == pred_row and gold_col == pred_col) else 0.5

            # 조합 점수 (텍스트를 더 중시)
            combined_score = 0.7 * text_sim + 0.3 * pos_sim

            if combined_score > best_score and text_sim >= 0.8:
                best_score = combined_score
                best_pred_idx = pred_idx

        if best_pred_idx >= 0:
            matched_gold.add(gold_idx)
            matched_pred.add(best_pred_idx)

    # F1 계산
    tp = len(matched_gold)
    fp = len(pred_cells) - tp
    fn = len(gold_cells) - tp

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


@register
class Txt3Task(Task):
    """표 구조 추출 태스크 (TXT-3)."""

    task_id: str = "TXT-3"
    kind: str = "extraction"
    is_vision: bool = False

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """PubTabNet 데이터셋에서 HTML 테이블 샘플 로드.

        각 샘플:
        - inputs: {"table_text": 셀 텍스트를 행 순서로 평문 렌더링}
        - reference: 정답 HTML 테이블 마크업

        NOTE: 실제 데이터셋이 로드되지 않으면 예외를 발생시킴. 합성 데이터 폴백 없음.
        """
        registry = load_registry()
        config = self.config

        if "datasets" not in config:
            raise ValueError("config에 datasets 맵이 없음")

        datasets_map = config["datasets"]  # {en: table_struct}
        samples = []
        sample_id = 0

        for lang, dataset_key in datasets_map.items():
            dataset_entry = resolve_dataset_entry(registry, dataset_key)
            hf_id = dataset_entry["hf_id"]
            split = dataset_entry.get("split", "train")
            config_name = dataset_entry.get("config", None)

            # 실제 데이터셋 로드 (실패 시 예외 발생)
            hf_ds = load_hf_split(hf_id, split, n, seed, config_name)

            # 컬럼명 감지
            col_html = self._detect_html_column(hf_ds)

            for idx, row in enumerate(hf_ds):
                html_table = row.get(col_html, "")

                if not html_table or not isinstance(html_table, str):
                    # 이 샘플은 건너뜀 (데이터 품질 문제)
                    continue

                # 정답 HTML에서 셀 추출
                gold_cells = parse_html_table(html_table)
                if not gold_cells:
                    # 파싱 실패 샘플은 건너뜀
                    continue

                # 셀 텍스트를 행 순서로 평문 렌더링 (모델 입력)
                table_text = self._render_table_text(gold_cells)

                sample = Sample(
                    sample_id=sample_id,
                    inputs={"table_text": table_text},
                    reference=html_table,  # 정답은 원본 HTML
                    lang=lang,
                    meta={
                        "dataset": dataset_key,
                        "source_idx": idx,
                        "n_cells": len(gold_cells),
                    },
                )
                samples.append(sample)
                sample_id += 1

                if sample_id >= n:
                    break

            if sample_id >= n:
                break

        return samples

    def _detect_html_column(self, hf_ds: Any) -> str:
        """HTML 테이블 컬럼명 감지.

        load_hf_split은 항상 list[dict]를 반환함.
        """
        if not hf_ds:
            raise ValueError("데이터셋이 비어있음")

        columns = hf_ds[0].keys()

        # 우선순위: html_table > html (PubTabNet에서 사용)
        for col in ["html_table", "html", "table", "content"]:
            if col in columns:
                return col

        raise ValueError(f"HTML 테이블 컬럼을 찾을 수 없음. 사용 가능한 컬럼: {list(columns)}")

    def _render_table_text(self, cells: list[tuple[int, int, str]]) -> str:
        """셀 리스트를 행 순서 평문으로 렌더링.

        형식:
        Row 0: [cell 0,0] [cell 0,1] ...
        Row 1: [cell 1,0] [cell 1,1] ...
        """
        if not cells:
            return ""

        # 최대 행/열 인덱스 찾기
        max_row = max(r for r, c, _ in cells)
        max_col = max(c for r, c, _ in cells)

        # 행별로 그룹화
        grid = {}
        for row, col, text in cells:
            if row not in grid:
                grid[row] = {}
            grid[row][col] = text

        # 평문 렌더링
        lines = []
        for row in range(max_row + 1):
            row_cells = []
            for col in range(max_col + 1):
                cell_text = grid.get(row, {}).get(col, "")
                row_cells.append(f"[{cell_text}]")
            lines.append(" ".join(row_cells))

        return "\n".join(lines)

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """표 셀 텍스트를 제시하고 HTML 테이블 생성 요청.

        모델에게:
        1. 셀 텍스트 제시 (평문)
        2. HTML 테이블 마크업 생성 요청
        """
        table_text = sample.inputs["table_text"]

        prompt = f"""Given the following table cell contents in row-major order, generate a valid HTML table with proper <table>, <tr>, and <td> tags.

Table cells:
{table_text}

Generate the HTML table structure. Respond with ONLY the HTML code, no explanation:"""

        return build_text_message(prompt)

    def parse_output(self, raw_text: str, sample: Sample) -> str:
        """모델 응답을 HTML 테이블로 파싱.

        응답에서 <table>...</table> 부분 추출.
        """
        # 응답에서 HTML 테이블 블록 추출
        match = re.search(r'<table[^>]*>.*?</table>', raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0)

        # 테이블 태그가 없으면 응답 전체 반환 (파싱 실패로 낮은 F1)
        return raw_text.strip()

    def score(self, parsed: list[str], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측 HTML에 대해 Cell-F1 계산.

        Returns:
            {cell_f1: float, n_evaluated: int, notes: str}
        """
        f1_scores = []

        for pred_html, sample in zip(parsed, samples):
            gold_html = sample.reference

            # 셀 파싱
            pred_cells = parse_html_table(pred_html)
            gold_cells = parse_html_table(gold_html)

            # Cell-F1 계산
            f1 = cell_f1_score(pred_cells, gold_cells)
            f1_scores.append(f1)

        # 평균 F1
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return {
            "cell_f1": avg_f1,
            "n_evaluated": len(f1_scores),
            "notes": "Cell-F1 v1: 셀 텍스트 퍼지 매칭 (threshold=0.8). TEDS는 후속 예정.",
        }


if __name__ == "__main__":
    """End-to-end 테스트: 3샘플로 FMAPI 호출 및 Cell-F1 계산."""
    import sys

    test_config = {
        "datasets": {
            "en": "table_struct",
        }
    }

    registry = load_registry()
    task = Txt3Task(test_config, registry)

    print("=" * 70)
    print("TXT-3 Table Structure Extraction - End-to-End Test")
    print("=" * 70)
    print("\nLoading 3 samples...")

    try:
        samples = task.load_samples(n=3, seed=42)
        print(f"✓ Loaded {len(samples)} samples\n")

        for sample in samples:
            cells = parse_html_table(sample.reference)
            table_text_preview = sample.inputs["table_text"][:80].replace("\n", " ")
            print(
                f"  [{sample.sample_id}] {len(cells)} cells, "
                f"table_text={table_text_preview}..."
            )

        print("\n" + "=" * 70)
        print("Calling FMAPIClient (databricks-gpt-5-6-sol)...")
        print("=" * 70)

        with FMAPIClient(
            profile="ai_devtools", timeout_seconds=60, max_retries=3
        ) as client:
            parsed_outputs = []

            for sample in samples:
                messages = task.build_prompt(sample)

                print(f"\n[Sample {sample.sample_id}]")
                table_text_preview = sample.inputs["table_text"][:60].replace("\n", " ")
                print(f"  Table text: {table_text_preview}...")

                # FMAPI 호출 (sol 모델, reasoning 최소화, max_tokens 증가)
                response = client.chat(
                    endpoint="databricks-gpt-5-6-sol",
                    messages=messages,
                    max_tokens=512,
                    extra_params={"reasoning_effort": "none"},
                )

                html_preview = response.text[:100].replace("\n", " ")
                print(f"  Generated HTML: {html_preview}...")

                # 파싱
                parsed = task.parse_output(response.text, sample)
                parsed_outputs.append(parsed)

                # 즉각 채점
                gold_cells = parse_html_table(sample.reference)
                pred_cells = parse_html_table(parsed)
                f1 = cell_f1_score(pred_cells, gold_cells)
                print(f"  Cell-F1: {f1:.4f} (gold={len(gold_cells)}, pred={len(pred_cells)})")

            # 전체 채점
            print("\n" + "=" * 70)
            print("Computing Cell-F1 Scores...")
            print("=" * 70)

            scores = task.score(parsed_outputs, samples)

            print(f"\nCell-F1 Metrics:")
            print(f"  Cell-F1: {scores['cell_f1']:.4f}")
            print(f"  Samples evaluated: {scores['n_evaluated']}")
            print(f"  Notes: {scores['notes']}")

            print("\n" + "=" * 70)
            print("✓ Test completed successfully")
            print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
