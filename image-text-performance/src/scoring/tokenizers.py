"""언어별 토크나이저 (plan 부록 P0).

한국어는 교착어라 공백 분리로 토큰화하면 ROUGE·Token-F1이 조용히 왜곡된다.
따라서 형태소 분석기(KoNLPy Mecab)를 쓴다. 영어 등은 공백 분리.

Mecab은 시스템 의존성(mecab, mecab-ko-dic)이 필요하다. 미설치 환경에서 벤치마크가
죽지 않도록, Mecab이 없으면 **음절 단위(syllable)** fallback으로 자동 전환한다.
음절 단위는 공백 분리보다 훨씬 정확하지만 형태소만큼은 아니므로, 어느 방식이 쓰였는지
`korean_tokenizer_backend()`로 조회해 리포트에 명시할 수 있다.

시스템 의존성 설치:
- macOS: brew install mecab mecab-ko mecab-ko-dic
- Ubuntu/Debian: sudo apt-get install mecab libmecab-dev mecab-ko-dic
"""

from __future__ import annotations

import re
import sys

# Mecab 인스턴스 캐시 (최초 1회만 초기화 시도)
_mecab = None
_mecab_tried = False
_backend = "unknown"  # "mecab" | "syllable"


def _get_mecab():
    """Mecab을 지연 초기화. 실패하면 None (fallback 신호)."""
    global _mecab, _mecab_tried, _backend
    if _mecab_tried:
        return _mecab
    _mecab_tried = True
    try:
        from konlpy.tag import Mecab

        m = Mecab()
        m.morphs("테스트")  # 실제 동작 확인 (백엔드 없으면 여기서 예외)
        _mecab = m
        _backend = "mecab"
    except Exception as e:  # konlpy backends may raise varied exception types
        _mecab = None
        _backend = "syllable"
        print(
            f"Mecab 초기화 실패 ({type(e).__name__}: {e}); "
            "음절 토큰화로 대체해 채점합니다.",
            file=sys.stderr,
        )
    return _mecab


# 한글 음절 + 영숫자 토큰 (음절 fallback용)
_SYLLABLE_RE = re.compile(r"[가-힣]|[a-zA-Z0-9]+")


def _syllable_tokenize(text: str) -> list[str]:
    """한글은 음절 하나씩, 영숫자 덩어리는 하나로. 형태소 대체 fallback."""
    return _SYLLABLE_RE.findall(text)


def korean_tokenizer_backend() -> str:
    """현재 한국어 토큰화에 쓰이는 백엔드: 'mecab' | 'syllable' | 'unknown'.

    리포트에 어떤 방식으로 채점했는지 명시하기 위함. tokenize(ko) 최초 호출 후 확정.
    """
    return _backend


def tokenize(text: str, lang: str) -> list[str]:
    """언어별 토큰화.

    - ko: Mecab 형태소. 없으면 음절 단위 fallback(자동).
    - en/기타: 공백 분리.
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {type(text)}")

    if lang == "ko":
        mecab = _get_mecab()
        if mecab is not None:
            return mecab.morphs(text)
        return _syllable_tokenize(text)
    return text.split()
