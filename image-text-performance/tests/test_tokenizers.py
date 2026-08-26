"""언어별 토크나이저 테스트 (src/scoring/tokenizers.py).

Mecab은 시스템 의존성이라 CI/로컬에 없을 수 있다. 모듈 캐시(_mecab/_mecab_tried)를
직접 조작해 mecab 경로와 음절 fallback 경로를 둘 다 검증한다.
"""

import pytest
from src.scoring import tokenizers


@pytest.fixture(autouse=True)
def reset_backend_cache(monkeypatch):
    """각 테스트는 초기화되지 않은 캐시 상태에서 시작한다."""
    monkeypatch.setattr(tokenizers, "_mecab", None)
    monkeypatch.setattr(tokenizers, "_mecab_tried", False)
    monkeypatch.setattr(tokenizers, "_backend", "unknown")


class FakeMecab:
    def __init__(self):
        self.calls = 0

    def morphs(self, text):
        self.calls += 1
        return text.replace("은", " 은").split()


def force_mecab(monkeypatch, mecab):
    monkeypatch.setattr(tokenizers, "_mecab", mecab)
    monkeypatch.setattr(tokenizers, "_mecab_tried", True)
    monkeypatch.setattr(tokenizers, "_backend", "mecab")


def force_syllable(monkeypatch):
    monkeypatch.setattr(tokenizers, "_mecab_tried", True)
    monkeypatch.setattr(tokenizers, "_backend", "syllable")


def test_english_is_whitespace_split():
    assert tokenizers.tokenize("the quick  brown fox", "en") == ["the", "quick", "brown", "fox"]


def test_unknown_language_falls_back_to_whitespace():
    assert tokenizers.tokenize("a b", "ja") == ["a", "b"]


def test_korean_uses_mecab_when_available(monkeypatch):
    mecab = FakeMecab()
    force_mecab(monkeypatch, mecab)
    assert tokenizers.tokenize("서울은 수도", "ko") == ["서울", "은", "수도"]
    assert mecab.calls == 1
    assert tokenizers.korean_tokenizer_backend() == "mecab"


def test_korean_syllable_fallback_splits_per_syllable(monkeypatch):
    force_syllable(monkeypatch)
    assert tokenizers.tokenize("서울 GPT4", "ko") == ["서", "울", "GPT4"]
    assert tokenizers.korean_tokenizer_backend() == "syllable"


def test_syllable_fallback_drops_punctuation(monkeypatch):
    force_syllable(monkeypatch)
    assert tokenizers.tokenize("안녕, 세계!", "ko") == ["안", "녕", "세", "계"]


def test_get_mecab_switches_backend_when_konlpy_missing(monkeypatch):
    """konlpy/Mecab 초기화 실패 시 죽지 않고 음절 백엔드로 전환한다."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("konlpy"):
            raise ImportError("no konlpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert tokenizers._get_mecab() is None
    assert tokenizers.korean_tokenizer_backend() == "syllable"
    assert tokenizers.tokenize("가", "ko") == ["가"]


def test_get_mecab_is_only_attempted_once(monkeypatch):
    attempts = []

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("konlpy"):
            attempts.append(name)
            raise ImportError("no konlpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    tokenizers._get_mecab()
    tokenizers._get_mecab()
    assert len(attempts) == 1


def test_backend_is_unknown_before_any_korean_call():
    assert tokenizers.korean_tokenizer_backend() == "unknown"


@pytest.mark.parametrize("bad", [None, 42, ["a"]])
def test_non_string_input_raises(bad):
    with pytest.raises(ValueError):
        tokenizers.tokenize(bad, "en")
