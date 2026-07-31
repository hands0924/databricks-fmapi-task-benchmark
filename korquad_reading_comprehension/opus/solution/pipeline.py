"""Feature-based extractive QA pipeline for KorQuAD (no pretrained models).

Candidate spans = runs of 1..MAXT whitespace tokens (with josa/punct stripping)
taken from the top-K sentences ranked by question overlap.  A gradient boosted
regressor scores each candidate against the char-level F1 target.
"""
import math
import re
from collections import Counter

import numpy as np

from qa_core import (
    bigram_f1,
    char_f1,
    norm_chars,
    sentence_spans,
    strip_span,
    tokens_with_offsets,
)

MAXT = 4
TOPK_SENT = 4
MAX_CAND = 220

HANJA = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"\d")
YEAR = re.compile(r"(1[0-9]{3}|20[0-9]{2})")
NUMUNIT = re.compile(r"\d+\s*(명|개|회|번|차|세|원|년|월|일|시간|분|초|%|퍼센트|만|억|천|백|점|위|kg|km|m|cm|mm|t|일간|주|개월|권|편|척|대|마리|가지|종)")

LOC_SUF = ("시", "도", "군", "구", "국", "주", "성", "산", "강", "역", "현", "읍", "면", "리", "동", "섬", "해", "만", "양", "로", "가", "부", "촌", "지방", "지역", "대륙", "반도", "공화국")
ORG_SUF = ("사", "회", "부", "원", "청", "당", "교", "단", "팀", "소", "과", "실", "위원회", "협회", "연맹", "대학교", "고등학교", "중학교", "학교", "대학", "그룹", "재단", "센터", "연구소", "은행", "방송", "신문")
PERSON_SUF = ("씨", "왕", "공", "제", "군", "후", "백", "자", "니", "스", "프", "르", "노", "코", "토", "키")

STOP_TOK = set("그, 이, 저, 및, 또, 또한, 하지만, 그러나, 그리고, 이는, 그는, 그의, 이후, 당시, 등, 등의, 있다., 없다., 것으로, 것이다., 대한, 위해, 통해, 함께, 따라".split(", "))

QWORDS = {
    "who": ["누구", "누가"],
    "when": ["언제", "몇 년", "몇년", "연도", "년도", "날짜", "몇 월", "며칠", "시기", "어느 해", "몇 세기", "몇시"],
    "where": ["어디", "어느 나라", "어느 지역", "장소", "지역은", "도시는", "나라는", "출신"],
    "num": ["몇", "얼마", "개수", "몇 명", "비율", "얼마나", "수는", "숫자"],
    "what": ["무엇", "뭐", "어떤", "무슨", "어느", "이름", "명칭"],
    "why": ["왜", "이유", "까닭", "때문", "원인"],
    "how": ["어떻게", "방법", "어떠한"],
}

Q_TAIL_RE = re.compile(
    r"(은|는|이|가|을|를|의|과|와|로|으로|에|에서)?\s*"
    r"(무엇인가요|무엇인가|무엇입니까|무엇이었나|무엇일까|무엇은|무엇을|무엇|누구인가요|누구인가|누구입니까|누구였는가|누구|어디인가요|어디인가|어디|언제인가|언제|어느것|얼마인가|얼마)?\s*[\?\.!]*$"
)


def stem_tok(t):
    s = strip_span(t)
    if len(s) >= 1:
        return s
    return re.sub(r"[^\w가-힣]", "", t)


def build_idf(contexts):
    df = Counter()
    seen = set()
    n = 0
    for c in contexts:
        if c in seen:
            continue
        seen.add(c)
        n += 1
        df.update(set(stem_tok(t) for t, _, _ in tokens_with_offsets(c)))
    return df, max(n, 1)


class Idf:
    def __init__(self, contexts):
        self.df, self.n = build_idf(contexts)
        self.cache = {}

    def __call__(self, w):
        v = self.cache.get(w)
        if v is None:
            v = math.log((self.n + 1) / (self.df.get(w, 0) + 1)) + 1.0
            if len(w) <= 1:
                v *= 0.3
            self.cache[w] = v
        return v


# ------------------------------------------------------------------ question
def question_info(q, idf):
    q = str(q)
    toks = [t for t, _, _ in tokens_with_offsets(q)]
    stems = [stem_tok(t) for t in toks]
    qtype = {}
    for k, pats in QWORDS.items():
        qtype[k] = int(any(p in q for p in pats))
    # head noun: last token before interrogative tail
    head = ""
    for t in reversed(toks):
        s = Q_TAIL_RE.sub("", t)
        s = strip_span(s)
        if len(s) >= 2:
            head = s
            break
    stemset = {}
    for s in stems:
        if len(s) >= 1:
            stemset[s] = idf(s)
    total = sum(stemset.values()) + 1e-9
    return {
        "q": q,
        "toks": toks,
        "stems": stems,
        "stemset": stemset,
        "qidf_total": total,
        "qtype": qtype,
        "head": head,
        "qlen": len(q),
        "ntok": len(toks),
    }


# ------------------------------------------------------------------ context
class Ctx:
    __slots__ = ("text", "toks", "stems", "sents", "tok_sent", "nchar", "sent_of_char")

    def __init__(self, text):
        self.text = text
        self.toks = tokens_with_offsets(text)
        self.stems = [stem_tok(t) for t, _, _ in self.toks]
        self.sents = sentence_spans(text)
        self.nchar = len(text)
        ts = []
        si = 0
        for _, a, b in self.toks:
            while si < len(self.sents) - 1 and a >= self.sents[si][1]:
                si += 1
            ts.append(si)
        self.tok_sent = ts


_ctx_cache = {}


def get_ctx(text):
    c = _ctx_cache.get(text)
    if c is None:
        if len(_ctx_cache) > 4000:
            _ctx_cache.clear()
        c = Ctx(text)
        _ctx_cache[text] = c
    return c


NEXT_COPULA = ("이다", "였다", "이었다", "다.", "한다", "했다", "된다", "되었다", "이라고", "라고", "라는", "이라는")


def type_flags(txt):
    nc = len(norm_chars(txt))
    return np.array(
        [
            nc,
            1.0 if DIGIT.search(txt) else 0.0,
            1.0 if YEAR.search(txt) else 0.0,
            1.0 if NUMUNIT.search(txt) else 0.0,
            1.0 if HANJA.search(txt) else 0.0,
            1.0 if LATIN.search(txt) else 0.0,
            1.0 if txt.endswith(LOC_SUF) else 0.0,
            1.0 if txt.endswith(ORG_SUF) else 0.0,
            1.0 if (2 <= nc <= 4 and re.fullmatch(r"[가-힣]+", txt) is not None) else 0.0,
            1.0 if len(txt.split()) == 1 else 0.0,
        ],
        dtype=np.float32,
    )


NTYPE = 10


class AnswerTypeStats:
    """Backoff target-encoding of answer surface types keyed on the question's
    head noun (and its suffix), learned from the training answers."""

    def __init__(self, questions, answers, idf, min_count=4):
        acc = {}
        cnt = Counter()
        for q, a in zip(questions, answers):
            qi = question_info(q, idf)
            head = qi["head"]
            keys = ["__global__"]
            if len(head) >= 2:
                keys.append("H:" + head)
                keys.append("S:" + head[-2:])
                keys.append("S:" + head[-1:])
            tf = type_flags(str(a))
            for k in keys:
                if k not in acc:
                    acc[k] = np.zeros(NTYPE, dtype=np.float64)
                acc[k] += tf
                cnt[k] += 1
        self.stats = {k: acc[k] / cnt[k] for k in acc if cnt[k] >= min_count}
        self.cnt = cnt
        self.glob = self.stats.get("__global__", np.zeros(NTYPE))
        self.min_count = min_count

    def get(self, head):
        for k in (("H:" + head, "S:" + head[-2:], "S:" + head[-1:]) if len(head) >= 2 else ()):
            v = self.stats.get(k)
            if v is not None and self.cnt[k] >= self.min_count:
                return v, float(self.cnt[k]), 1.0 if k.startswith("H:") else 0.0
        return self.glob, 0.0, 0.0


def ats_features(txt, prior, n, exact):
    tf = type_flags(txt)
    match = 0.0
    for i in range(1, NTYPE):
        p = prior[i]
        match += p * tf[i] + (1 - p) * (1 - tf[i])
    match /= (NTYPE - 1)
    return [
        prior[0],
        tf[0] - prior[0],
        abs(tf[0] - prior[0]),
        match,
        float(np.log1p(n)),
        exact,
    ] + [float(prior[i] * tf[i] - (1 - prior[i]) * tf[i]) for i in range(1, NTYPE)] + [float(prior[i]) for i in range(1, NTYPE)]


N_ATS = 6 + 2 * (NTYPE - 1)


def gen_features(ctx, qi, idf, gold=None, max_cand=MAX_CAND, ats=None):
    text = ctx.text
    toks, stems, sents = ctx.toks, ctx.stems, ctx.sents
    ntok = len(toks)
    qstem = qi["stemset"]
    # token match info
    matched = np.zeros(ntok, dtype=np.float32)
    for i, s in enumerate(stems):
        w = qstem.get(s)
        if w is not None:
            matched[i] = w
        elif len(s) >= 3:
            # partial containment (compound words)
            for qs, wv in qstem.items():
                if len(qs) >= 3 and (qs in s or s in qs):
                    matched[i] = 0.6 * wv
                    break
    # sentence scores
    sent_scores = []
    for si, (a, b) in enumerate(sents):
        ids = [i for i in range(ntok) if ctx.tok_sent[i] == si]
        m = sum(matched[i] for i in ids)
        cover = m / qi["qidf_total"]
        bg = bigram_f1(text[a:b], qi["q"])
        sent_scores.append(1.0 * cover + 0.5 * bg)
    if not sent_scores:
        return [], None
    order = np.argsort(sent_scores)[::-1]
    rank_of = {int(s): r for r, s in enumerate(order)}
    keep_sents = set(int(s) for s in order[:TOPK_SENT])
    best_sent_score = max(sent_scores)

    # head-noun positions in context
    head = qi["head"]
    head_pos = []
    if len(head) >= 2:
        st = 0
        while True:
            k = text.find(head, st)
            if k < 0:
                break
            head_pos.append(k)
            st = k + 1
    # matched token char positions
    mpos = [(toks[i][1], toks[i][2], matched[i]) for i in range(ntok) if matched[i] > 0]

    qnorm = norm_chars(qi["q"])
    prior = prior_tail = None
    pn_log = 0.0
    pexact = 0.0
    if ats is not None:
        pr, pn, pexact = ats.get(head)
        prior = [float(x) for x in pr]
        prior_tail = prior[1:]
        pn_log = float(math.log1p(pn))
    rows = []
    metas = []
    for a in range(ntok):
        if ctx.tok_sent[a] not in keep_sents:
            continue
        first_tok = toks[a][0]
        if first_tok in STOP_TOK:
            continue
        si = ctx.tok_sent[a]
        for L in range(1, MAXT + 1):
            b = a + L - 1
            if b >= ntok or ctx.tok_sent[b] != si:
                break
            s_char, e_char = toks[a][1], toks[b][2]
            raw = text[s_char:e_char]
            variants = []
            v1 = strip_span(raw, True)
            v0 = strip_span(raw, False)
            if v1:
                variants.append((v1, 1))
            if v0 and v0 != v1:
                variants.append((v0, 0))
            for txt, did_strip in variants:
                nc = len(norm_chars(txt))
                if nc == 0 or nc > 40:
                    continue
                # ---- features
                inner_match = float(matched[a : b + 1].sum())
                left_d = 99.0
                right_d = 99.0
                w15 = w40 = 0.0
                for (ms, me, mv) in mpos:
                    if me <= s_char:
                        d = s_char - me
                        if d < left_d:
                            left_d = d
                        if d <= 40:
                            w40 += mv
                        if d <= 15:
                            w15 += mv
                    elif ms >= e_char:
                        d = ms - e_char
                        if d < right_d:
                            right_d = d
                        if d <= 40:
                            w40 += mv
                        if d <= 15:
                            w15 += mv
                prev_m = float(matched[a - 1]) if a > 0 and ctx.tok_sent[a - 1] == si else 0.0
                next_m = float(matched[b + 1]) if b + 1 < ntok and ctx.tok_sent[b + 1] == si else 0.0
                prev_tok = toks[a - 1][0] if a > 0 else ""
                next_tok = toks[b + 1][0] if b + 1 < ntok else ""
                hd = 999.0
                for hp in head_pos:
                    d = abs(hp - s_char)
                    if d < hd:
                        hd = d
                in_q = 1.0 if norm_chars(txt) and norm_chars(txt) in qnorm else 0.0
                f = [
                    nc,                                        # 0 char len
                    L,                                         # 1 tokens
                    did_strip,                                 # 2
                    len(raw) - nc,                             # 3 stripped chars
                    sent_scores[si],                           # 4
                    sent_scores[si] - best_sent_score,          # 5
                    rank_of[si],                               # 6
                    si,                                        # 7
                    si / max(1, len(sents) - 1),               # 8
                    s_char / max(1, ctx.nchar),                # 9
                    (s_char - sents[si][0]) / max(1, sents[si][1] - sents[si][0]),  # 10
                    a,                                         # 11
                    inner_match,                               # 12
                    inner_match / (L + 1e-9),                  # 13
                    min(left_d, 99),                           # 14
                    min(right_d, 99),                          # 15
                    min(min(left_d, right_d), 99),             # 16
                    w15,                                       # 17
                    w40,                                       # 18
                    w40 / qi["qidf_total"],                    # 19
                    prev_m,                                    # 20
                    next_m,                                    # 21
                    in_q,                                      # 22
                    bigram_f1(txt, qi["q"]),                   # 23
                    min(hd, 999),                              # 24
                    1.0 if len(head) >= 2 and (head in prev_tok or prev_tok in head and len(prev_tok) > 1) else 0.0,  # 25
                    1.0 if len(head) >= 2 and (head in next_tok or next_tok in head and len(next_tok) > 1) else 0.0,  # 26
                    bigram_f1(prev_tok, head) if head else 0.0,  # 27
                    bigram_f1(next_tok, head) if head else 0.0,  # 28
                    1.0 if prev_tok.endswith("의") else 0.0,      # 29
                    1.0 if any(next_tok.startswith(x) or next_tok == x for x in NEXT_COPULA) else 0.0,  # 30
                    1.0 if DIGIT.search(txt) else 0.0,          # 31
                    1.0 if txt.replace(",", "").replace(".", "").isdigit() else 0.0,  # 32
                    1.0 if YEAR.search(txt) else 0.0,           # 33
                    1.0 if NUMUNIT.search(txt) else 0.0,        # 34
                    1.0 if HANJA.search(txt) else 0.0,          # 35
                    1.0 if LATIN.search(txt) else 0.0,          # 36
                    1.0 if txt.endswith(LOC_SUF) else 0.0,      # 37
                    1.0 if txt.endswith(ORG_SUF) else 0.0,      # 38
                    1.0 if (2 <= nc <= 4 and re.fullmatch(r"[가-힣]+", txt) is not None) else 0.0,  # 39
                    1.0 if (s_char > 0 and text[s_char - 1] in "《〈『「\"'“‘(") else 0.0,  # 40
                    1.0 if (e_char < ctx.nchar and text[e_char] in "》〉』」\"'”’)") else 0.0,  # 41
                    text.count(txt) if nc <= 12 else 1,         # 42
                    qi["qtype"]["who"],                        # 43
                    qi["qtype"]["when"],                       # 44
                    qi["qtype"]["where"],                      # 45
                    qi["qtype"]["num"],                        # 46
                    qi["qtype"]["what"],                       # 47
                    qi["qtype"]["why"],                        # 48
                    qi["qtype"]["how"],                        # 49
                    qi["ntok"],                                # 50
                    qi["qlen"],                                # 51
                    len(sents),                                # 52
                    ctx.nchar,                                 # 53
                    1.0 if a == 0 or ctx.tok_sent[a - 1] != si else 0.0,  # 54 sent start
                    1.0 if b + 1 >= ntok or ctx.tok_sent[b + 1] != si else 0.0,  # 55 sent end
                    1.0 if txt.endswith(PERSON_SUF) else 0.0,  # 56
                    (sents[si][1] - sents[si][0]),             # 57
                    1.0 if "," in raw else 0.0,                # 58
                    idf(head) if head else 0.0,                # 59
                    min(left_d, 99) + min(right_d, 99),        # 60
                    prev_m * next_m,                           # 61
                    float(inner_match > 0),                    # 62
                ]
                if ats is not None:
                    tfl = (nc, f[31], f[33], f[34], f[35], f[36], f[37], f[38], f[39],
                           1.0 if L == 1 else 0.0)
                    match = 0.0
                    for k in range(1, NTYPE):
                        pk = prior[k]
                        tk = tfl[k]
                        match += pk * tk + (1.0 - pk) * (1.0 - tk)
                    f.append(prior[0])
                    f.append(nc - prior[0])
                    f.append(abs(nc - prior[0]))
                    f.append(match / (NTYPE - 1))
                    f.append(pn_log)
                    f.append(pexact)
                    for k in range(1, NTYPE):
                        f.append((2.0 * prior[k] - 1.0) * tfl[k])
                    f.extend(prior_tail)
                rows.append(f)
                metas.append((txt, s_char, e_char))
    if not rows:
        return [], None
    X = np.asarray(rows, dtype=np.float32)
    y = None
    if gold is not None:
        y = np.array([char_f1(m[0], gold) for m in metas], dtype=np.float32)
    if len(metas) > max_cand:
        # prune by cheap heuristic, always keeping best-target rows in training
        heur = heuristic_scores(X)
        idxs = np.argsort(heur)[::-1][:max_cand]
        if y is not None:
            keep_best = np.argsort(y)[::-1][:5]
            idxs = np.unique(np.concatenate([idxs, keep_best]))
        X = X[idxs]
        metas = [metas[i] for i in idxs]
        if y is not None:
            y = y[idxs]
    return (X, metas) if y is None else (X, metas, y)


def heuristic_scores(X):
    """Cheap prefilter / fallback score."""
    return (
        1.5 * X[:, 4]                      # sentence score
        - 0.6 * np.minimum(X[:, 16], 60) / 20.0   # distance to matched q word
        + 0.4 * X[:, 19]                   # matched idf mass nearby
        - 0.8 * X[:, 22]                   # appears in question
        - 0.9 * X[:, 23]                   # bigram overlap with question
        - 0.05 * np.abs(X[:, 0] - 6)       # prefer ~6 char answers
        + 0.3 * X[:, 30] + 0.2 * X[:, 29]
        + 0.3 * (X[:, 25] + X[:, 26])
    )


NBASE = 63
NFEAT = NBASE + N_ATS
FEATURE_NAMES = [f"f{i}" for i in range(NFEAT)]
