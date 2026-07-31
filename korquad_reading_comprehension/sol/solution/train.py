#!/usr/bin/env python3
"""Train a lightweight extractive Korean QA ranker and create a submission."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


SENT_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣一-龥]+|[^\w\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
SUFFIXES = (
    "에서는", "으로부터", "이라고", "라는", "으로", "에서", "에게", "부터", "까지",
    "와의", "과의", "들의", "에는", "에도", "보다", "처럼", "라고", "이며", "이고",
    "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "만",
)
Q_STOP = {
    "무엇", "무슨", "어떤", "어디", "누구", "언제", "얼마", "몇", "왜", "어떻게",
    "것", "곳", "사람", "이름", "명칭", "시기", "년도", "해", "인가", "했는가",
}


def normalize(text: str) -> str:
    return SPACE_RE.sub("", str(text)).lower()


def char_f1(pred: str, gold: str) -> float:
    p, g = Counter(normalize(pred)), Counter(normalize(gold))
    if not p or not g:
        return float(p == g)
    common = sum((p & g).values())
    return 2.0 * common / (sum(p.values()) + sum(g.values()))


def ngrams(text: str, n: int = 2) -> set[str]:
    text = normalize(text)
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a))


def split_sentences(context: str) -> list[tuple[int, int, str]]:
    out = []
    for match in SENT_RE.finditer(context):
        start, end = match.span()
        while start < end and context[start].isspace():
            start += 1
        while end > start and context[end - 1].isspace():
            end -= 1
        if start < end:
            out.append((start, end, context[start:end]))
    return out or [(0, len(context), context)]


def question_terms(question: str) -> tuple[set[str], set[str]]:
    words = set(re.findall(r"[0-9A-Za-z가-힣]+", question.lower())) - Q_STOP
    chars = ngrams(question)
    return words, chars


def sentence_score(question: str, sentence: str) -> float:
    words, qgrams = question_terms(question)
    swords = set(re.findall(r"[0-9A-Za-z가-힣]+", sentence.lower()))
    word_score = len(words & swords) / max(1, len(words))
    return 2.2 * word_score + overlap(qgrams, ngrams(sentence))


def sentence_candidates(context: str, question: str, limit: int = 4):
    sentences = split_sentences(context)
    ranked = sorted(sentences, key=lambda x: sentence_score(question, x[2]), reverse=True)
    return ranked[:limit]


def candidate_spans(sentence: str) -> list[tuple[int, int]]:
    tokens = list(TOKEN_RE.finditer(sentence))
    spans: set[tuple[int, int]] = set()
    for i, first in enumerate(tokens):
        if not re.search(r"[0-9A-Za-z가-힣一-龥]", first.group()):
            continue
        for j in range(i, min(len(tokens), i + 5)):
            last = tokens[j]
            if not re.search(r"[0-9A-Za-z가-힣一-龥]", last.group()):
                continue
            start, end = first.start(), last.end()
            text = sentence[start:end].strip(" \t\n,.;:!?()[]{}<>\"'《》〈〉「」『』")
            if text:
                real_start = sentence.find(text, start, end)
                spans.add((real_start, real_start + len(text)))
            last_text = last.group()
            for suffix in SUFFIXES:
                if last_text.endswith(suffix) and len(last_text) > len(suffix):
                    spans.add((start, last.end() - len(suffix)))
                    break
    return sorted(spans)


def qtypes(question: str) -> list[float]:
    q = normalize(question)
    return [
        float(any(x in q for x in ("누구", "인물", "사람", "이름", "자는?"))),
        float(any(x in q for x in ("언제", "몇년", "몇월", "시기", "연도", "태어난해"))),
        float(any(x in q for x in ("어디", "장소", "지역", "국가", "나라", "출생지"))),
        float(any(x in q for x in ("얼마", "몇명", "몇개", "수는", "기간"))),
        float(any(x in q for x in ("무엇", "무슨", "어떤", "뭐"))),
        float(any(x in q for x in ("왜", "이유", "원인"))),
    ]


def features(context: str, question: str, sent_start: int, sent: str, start: int, end: int, cache):
    answer = sent[start:end]
    before, after = sent[max(0, start - 45) : start], sent[end : end + 45]
    qg, ag = cache["qg"], ngrams(answer)
    sg, bg, fg = cache["sg"], ngrams(before), ngrams(after)
    qword = cache["qword"]
    aword = set(re.findall(r"[0-9A-Za-z가-힣]+", answer.lower()))
    clean = answer.strip()
    numeric = bool(re.search(r"\d", clean))
    qt = cache["qt"]
    qnorm = normalize(question)
    left8, right8 = ngrams(sent[max(0, start - 8) : start]), ngrams(sent[end : end + 8])
    left20, right20 = ngrams(sent[max(0, start - 20) : start]), ngrams(sent[end : end + 20])
    around80 = ngrams(sent[max(0, start - 80) : start] + sent[end : end + 80])
    return np.asarray([
        len(clean), len(normalize(clean)), len(clean.split()),
        (sent_start + start) / max(1, len(context)), start / max(1, len(sent)),
        cache["rank"], cache["sscore"], overlap(qg, sg), overlap(qg, bg), overlap(qg, fg),
        overlap(qg, bg | fg), overlap(qg, ag), overlap(ag, qg),
        len(qword & aword) / max(1, len(aword)),
        float(numeric), float(clean.isdigit()), float(bool(re.fullmatch(r"\d{3,4}년", clean))),
        float(bool(re.search(r"\d+년", clean))), float(bool(re.search(r"\d+[월일명개차세%]", clean))),
        float(clean.endswith(("년", "월", "일", "시", "분", "명", "개", "곳"))),
        float(clean.startswith(("《", "〈", "「", "『", "\"", "'"))),
        float(any(c in clean for c in ",.;:!?")),
        float(clean.endswith(SUFFIXES)),
        qt[0] * float(not numeric and len(clean) <= 15),
        qt[1] * float(numeric), qt[1] * float("년" in clean),
        qt[2] * float(clean.endswith(("시", "도", "군", "구", "국", "주", "섬"))),
        qt[3] * float(numeric),
        overlap(qg, left8), overlap(qg, right8),
        overlap(qg, left20), overlap(qg, right20), overlap(qg, around80),
        float("직업" in qnorm), float("이름" in qnorm or "명칭" in qnorm),
        float("태어" in qnorm), float("제작" in qnorm or "만든" in qnorm),
        float("몇" in qnorm), float("누가" in qnorm or "누구" in qnorm),
        float("어디" in qnorm or "어느곳" in qnorm),
        float("이유" in qnorm or "원인" in qnorm or "왜" in qnorm),
        *qt,
    ], dtype=np.float32)


def row_candidates(context: str, question: str, sentence_limit: int = 4):
    result = []
    seen = set()
    qg = ngrams(question)
    qword = set(re.findall(r"[0-9A-Za-z가-힣]+", question.lower()))
    qt = qtypes(question)
    for rank, (sent_start, _, sent) in enumerate(sentence_candidates(context, question, sentence_limit)):
        cache = {
            "qg": qg, "qword": qword, "qt": qt, "sg": ngrams(sent),
            "sscore": sentence_score(question, sent), "rank": rank,
        }
        for start, end in candidate_spans(sent):
            answer = sent[start:end].strip()
            key = (sent_start + start, sent_start + end)
            if answer and key not in seen and len(answer) <= 45:
                seen.add(key)
                result.append((answer, features(context, question, sent_start, sent, start, end, cache)))
    return result


def build_training(rows: pd.DataFrame, rng: np.random.Generator, max_rows: int):
    xs, ys = [], []
    recalls = []
    if len(rows) > max_rows:
        rows = rows.sample(max_rows, random_state=123)
    for row in rows.itertuples(index=False):
        candidates = row_candidates(str(row.context), str(row.question), sentence_limit=2)
        if not candidates:
            continue
        scores = np.asarray([char_f1(a, str(row.answer)) for a, _ in candidates])
        recalls.append(float(scores.max()))
        useful = np.flatnonzero(scores > 0.0)
        negatives = np.flatnonzero(scores == 0.0)
        if len(useful) > 16:
            # Retain the best spans plus varied partial matches without swamping
            # each question with many nearly identical overlapping candidates.
            best = useful[np.argsort(scores[useful])[-8:]]
            rest = np.setdiff1d(useful, best)
            useful = np.concatenate((best, rng.choice(rest, min(8, len(rest)), replace=False)))
        keep_neg = rng.choice(negatives, min(36, len(negatives)), replace=False)
        keep = np.unique(np.concatenate((useful, keep_neg)))
        for i in keep:
            xs.append(candidates[i][1])
            ys.append(scores[i])
    print(f"training rows={len(rows)} examples={len(ys)} candidate_oracle_f1={np.mean(recalls):.4f}")
    return np.vstack(xs), np.asarray(ys, dtype=np.float32)


def predict(model, rows: pd.DataFrame) -> list[str]:
    answers = []
    for idx, row in enumerate(rows.itertuples(index=False), 1):
        candidates = row_candidates(str(row.context), str(row.question), sentence_limit=2)
        if candidates:
            x = np.vstack([f for _, f in candidates])
            scores = model.predict(x)
            # Small tie-breaker favors concise spans, as required by character F1.
            scores -= 0.00015 * np.asarray([len(a) for a, _ in candidates])
            answer = candidates[int(np.argmax(scores))][0]
            # Extractive candidates sometimes retain a case marker. Multi-character
            # markers and these four unambiguous one-character markers are rarely
            # part of a KorQuAD gold span.
            for suffix in ("에서는", "으로부터", "에서", "으로", "에게", "에는", "에도", "이라고", "라고", "이며", "이고", "의", "은", "는", "을", "를"):
                if answer.endswith(suffix) and len(answer) >= len(suffix) + 2:
                    answer = answer[: -len(suffix)].rstrip()
                    break
            answers.append(answer)
        else:
            answers.append("")
        if idx % 2000 == 0:
            print(f"predicted {idx}/{len(rows)}")
    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--output", default="outputs/submission.csv")
    parser.add_argument("--max-train-rows", type=int, default=8000)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(args.train, dtype=str).fillna("")
    test = pd.read_csv(args.test, dtype=str).fillna("")
    rng = np.random.default_rng(123)

    fit_rows = train
    valid = None
    if args.validate:
        contexts = train["context"].drop_duplicates().sample(frac=0.1, random_state=123)
        valid_mask = train["context"].isin(set(contexts))
        fit_rows, valid = train[~valid_mask], train[valid_mask]
        valid = valid.sample(min(600, len(valid)), random_state=123)

    x, y = build_training(fit_rows, rng, args.max_train_rows)
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.075, max_iter=180, max_leaf_nodes=31,
        l2_regularization=2.0, min_samples_leaf=35, random_state=123,
    )
    model.fit(x, y)

    if valid is not None:
        pred = predict(model, valid)
        score = np.mean([char_f1(p, g) for p, g in zip(pred, valid["answer"])])
        print(json.dumps({"validation_char_f1": score, "rows": len(valid)}, ensure_ascii=False))
        for q, g, p in list(zip(valid["question"], valid["answer"], pred))[:15]:
            print(f"Q={q} | G={g} | P={p} | F1={char_f1(p, g):.3f}")
        return

    pred = predict(model, test)
    output = pd.DataFrame({"id": test["id"], "answer": pred})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(output)} empty={sum(a == '' for a in pred)}")


if __name__ == "__main__":
    main()
