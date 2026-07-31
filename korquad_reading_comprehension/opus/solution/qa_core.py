"""Core utilities for KorQuAD-style extractive QA without pretrained models.

Char-level F1 metric: whitespace removed, multiset of characters.
"""
import re
from collections import Counter

# ---------------------------------------------------------------- metric
def norm_chars(s):
    return re.sub(r"\s+", "", str(s))


def char_f1(pred, gold):
    p = Counter(norm_chars(pred))
    g = Counter(norm_chars(gold))
    if not p or not g:
        return float(p == g)
    common = sum((p & g).values())
    if common == 0:
        return 0.0
    prec = common / sum(p.values())
    rec = common / sum(g.values())
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------- tokenize
TOK_RE = re.compile(r"[^\s]+")


def tokens_with_offsets(text):
    return [(m.group(0), m.start(), m.end()) for m in TOK_RE.finditer(text)]


# trailing josa / endings, longest first
JOSA = [
    "으로부터", "에서부터", "이라고는", "이라는", "라고는", "에서는", "에게서", "으로서", "으로써",
    "이라고", "라고", "이라", "에서", "에게", "에는", "에도", "으로", "만큼", "까지", "부터",
    "이며", "이고", "이다", "였다", "이었다", "인", "의", "은", "는", "이", "가", "을", "를",
    "에", "로", "와", "과", "도", "만", "께", "란", "라", "며", "고", "나", "든",
]
JOSA = sorted(set(JOSA), key=len, reverse=True)

PUNCT_TRAIL = ".,!?;:·…\"'”’)]}》〉』」>＞"
PUNCT_LEAD = "\"'“‘([{《〈『「<＜«´"


def strip_span(s, strip_josa=True):
    s = s.strip()
    # leading punctuation
    while s and s[0] in PUNCT_LEAD:
        s = s[1:]
    while s and s[-1] in PUNCT_TRAIL:
        s = s[:-1]
    if strip_josa and len(s) >= 3:
        for j in JOSA:
            if s.endswith(j) and len(s) - len(j) >= 2:
                cand = s[: -len(j)]
                # do not strip if it breaks a pure-digit / latin token oddly
                s = cand
                break
    while s and s[-1] in PUNCT_TRAIL:
        s = s[:-1]
    return s.strip()


# ---------------------------------------------------------------- sentences
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_spans(text):
    spans = []
    pos = 0
    for m in SENT_SPLIT.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return [(a, b) for a, b in spans if b > a]


# ---------------------------------------------------------------- char ngrams
def ngrams(s, n=2):
    s = norm_chars(s)
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def bigram_f1(a, b):
    ca, cb = Counter(ngrams(a)), Counter(ngrams(b))
    if not ca or not cb:
        return 0.0
    common = sum((ca & cb).values())
    if not common:
        return 0.0
    return 2 * common / (sum(ca.values()) + sum(cb.values()))
