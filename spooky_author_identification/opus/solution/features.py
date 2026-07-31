"""Build and cache feature matrices."""
import numpy as np, pandas as pd, re, os, time, pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse

CLASSES = ['EAP', 'HPL', 'MWS']
CACHE = 'work/cache'
os.makedirs(CACHE, exist_ok=True)


def load():
    tr = pd.read_csv('train.csv'); te = pd.read_csv('test.csv')
    y = tr.author.map({c: i for i, c in enumerate(CLASSES)}).values
    return tr, te, y


def _cache(name, fn):
    p = f'{CACHE}/{name}.pkl'
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    t0 = time.time()
    obj = fn()
    with open(p, 'wb') as f:
        pickle.dump(obj, f)
    print(f'  built {name} {getattr(obj,"shape",None)} in {time.time()-t0:.1f}s', flush=True)
    return obj


def get_texts():
    tr, te, y = load()
    return pd.concat([tr.text, te.text]).values, len(tr)


# ---------------- sparse text features ----------------
def word_tfidf():
    def f():
        txt, n = get_texts()
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                            strip_accents='unicode', token_pattern=r'\w{1,}')
        return v.fit_transform(txt)
    return _cache('word_tfidf', f)


def word_tfidf1():
    def f():
        txt, n = get_texts()
        v = TfidfVectorizer(ngram_range=(1, 1), min_df=1, sublinear_tf=True,
                            strip_accents='unicode', token_pattern=r'\w{1,}')
        return v.fit_transform(txt)
    return _cache('word_tfidf1', f)


def char_tfidf():
    def f():
        txt, n = get_texts()
        v = TfidfVectorizer(ngram_range=(2, 6), min_df=3, sublinear_tf=True,
                            strip_accents='unicode', analyzer='char_wb', max_features=300000)
        return v.fit_transform(txt)
    return _cache('char_tfidf', f)


def char_full_tfidf():
    """char analyzer (not _wb) captures cross-word patterns."""
    def f():
        txt, n = get_texts()
        v = TfidfVectorizer(ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                            analyzer='char', max_features=300000, lowercase=True)
        return v.fit_transform(txt)
    return _cache('char_full_tfidf', f)


def word_counts():
    def f():
        txt, n = get_texts()
        v = CountVectorizer(ngram_range=(1, 2), min_df=2, strip_accents='unicode',
                            token_pattern=r'\w{1,}')
        return v.fit_transform(txt)
    return _cache('word_counts', f)


def char_counts():
    def f():
        txt, n = get_texts()
        v = CountVectorizer(ngram_range=(2, 5), min_df=3, analyzer='char_wb',
                            strip_accents='unicode', max_features=200000)
        return v.fit_transform(txt)
    return _cache('char_counts', f)


def pos_tfidf():
    """Crude POS-ish / shape-based token sequence n-grams (no external deps)."""
    def f():
        txt, n = get_texts()
        FUNC = set("""a about above after again against all am an and any are as at be because been before
being below between both but by cannot could did do does doing down during each few for from further had
has have having he her here hers herself him himself his how i if in into is it its itself me more most
my myself nor not of off on once only or other ought our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why with would you your yours
yourself yourselves shall may might must upon whose thou thee thy hath doth ere nay yet still""".split())

        def shape(tok):
            if tok in FUNC:
                return tok
            if tok.isdigit():
                return 'NUM'
            if tok[0].isupper():
                return 'CAP'
            if tok.endswith('ly'):
                return 'ADV'
            if tok.endswith('ing'):
                return 'ING'
            if tok.endswith('ed'):
                return 'ED'
            if tok.endswith('ion') or tok.endswith('ions'):
                return 'ION'
            if tok.endswith('ness') or tok.endswith('ity') or tok.endswith('ment'):
                return 'ABS'
            if tok.endswith('s'):
                return 'PLS'
            return 'W%d' % min(len(tok) // 3, 4)

        docs = []
        for t in txt:
            toks = re.findall(r"[A-Za-z']+|[.,;:!?\"()\-]", t)
            docs.append(' '.join(shape(tk) if tk[0].isalpha() or tk[0] == "'" else 'P' + tk for tk in toks))
        v = TfidfVectorizer(ngram_range=(1, 4), min_df=3, sublinear_tf=True,
                            token_pattern=r'\S+', lowercase=False)
        return v.fit_transform(docs)
    return _cache('pos_tfidf', f)


# ---------------- dense handcrafted features ----------------
def hand_feats():
    def f():
        txt, n = get_texts()
        STOP = set("""the of and to in a i was that it he his my me her she as for with but not this had
you all is at be so were which have on him by from they what them one no there we been would their or
their""".split())
        rows = []
        for t in txt:
            L = len(t)
            words = re.findall(r"[A-Za-z']+", t)
            nw = max(len(words), 1)
            lw = [w.lower() for w in words]
            wl = [len(w) for w in words] or [0]
            uniq = len(set(lw))
            row = [
                L, nw, np.mean(wl), np.std(wl), max(wl), uniq / nw,
                sum(c.isupper() for c in t) / L,
                sum(c in ',' for c in t), sum(c in ';' for c in t), sum(c in ':' for c in t),
                sum(c in '.' for c in t), sum(c in '!?' for c in t), sum(c in '"' for c in t),
                t.count("'"), t.count('-'), t.count('('),
                sum(c in ',' for c in t) / nw, sum(c in ';' for c in t) / nw,
                sum(w in STOP for w in lw) / nw,
                sum(len(w) > 7 for w in words) / nw,
                np.mean([len(s) for s in re.split(r'[.!?]', t) if s.strip()] or [0]),
                len([s for s in re.split(r'[.!?]', t) if s.strip()]),
                sum(1 for w in lw if w.endswith('ly')) / nw,
                sum(1 for w in lw if w.endswith('ing')) / nw,
                sum(1 for w in words if w[0].isupper()) / nw,
                len(set(t)) / L,
            ]
            rows.append(row)
        A = np.array(rows, dtype=np.float64)
        A[:, 0] = np.log1p(A[:, 0]); A[:, 1] = np.log1p(A[:, 1])
        return A
    return _cache('hand_feats', f)


def svd_feats(k=180):
    def f():
        from sklearn.decomposition import TruncatedSVD
        X = sparse.hstack([word_tfidf(), char_tfidf()]).tocsr()
        s = TruncatedSVD(k, random_state=0)
        return s.fit_transform(X).astype(np.float32)
    return _cache(f'svd{k}', f)


def stopw_tfidf():
    """Function-word-only stream (authorship attribution classic)."""
    def f():
        txt, n = get_texts()
        FUNC = set("""a about above across after against all almost alone along already also although
always am among an and another any anybody anyone anything anywhere are around as at be because been
before behind being below beneath beside besides between beyond both but by can cannot could dare did do
does doing done down during each either else elsewhere enough even ever every everybody everyone
everything except far few for former forth from further had hardly has have having he hence her here hers
herself him himself his hither how however i if in indeed inside instead into is it its itself just least
less lest like little many may me might mine more moreover most much must my myself near neither never
nevertheless next no nobody none nor not nothing notwithstanding now nowhere of off often on once one only
onto or other others otherwise ought our ours ourselves out over own perhaps rather round same save shall
she should since so some somebody somehow someone something sometimes somewhat somewhere still such than
that the thee their theirs them themselves then thence there thereby therefore thus these they thine this
those thou though through throughout thy till to together too toward towards under unless until unto up
upon us very was we well were what whatever when whence whenever where whereas whereby wherein whereupon
wherever whether which while whither who whoever whom whose why will with within without would ye yet you
your yours yourself yourselves hath doth ere nay art wilt shalt""".split())
        docs = []
        for t in txt:
            toks = re.findall(r"[A-Za-z']+|[.,;:!?\"()\-]", t.lower())
            docs.append(' '.join(tk if tk in FUNC else ('P' + tk if not tk[0].isalpha() else 'X')
                                 for tk in toks))
        v = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True,
                            token_pattern=r'\S+', lowercase=False)
        return v.fit_transform(docs)
    return _cache('stopw_tfidf', f)


def charcase_tfidf():
    """char_wb n-grams with case preserved (captures capitalisation habits)."""
    def f():
        txt, n = get_texts()
        v = TfidfVectorizer(ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                            analyzer='char_wb', lowercase=False, max_features=250000)
        return v.fit_transform(txt)
    return _cache('charcase_tfidf', f)


def word3_tfidf():
    def f():
        txt, n = get_texts()
        v = TfidfVectorizer(ngram_range=(1, 3), min_df=3, sublinear_tf=True,
                            strip_accents='unicode', token_pattern=r'\w{1,}')
        return v.fit_transform(txt)
    return _cache('word3_tfidf', f)
