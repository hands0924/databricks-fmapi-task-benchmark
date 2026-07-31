"""Load cached base-model probabilities, choose blend weights by *post-constraint*
OOF accuracy (the competition metric), apply the premise-group constraint and
write outputs/submission.csv."""
import sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, 'solution')
from constraint import apply_constraint, known_from, CLASSES

KEYS = ['lin', 'gb', 'mlp']
LAMS = [0, -1, -2, -3, -4, -6, -10]

tr = pd.read_csv('train.csv'); te = pd.read_csv('test.csv')
y = np.load('cache/y.npy', allow_pickle=True)
O = np.nan_to_num(np.load('cache/oof_all.npy'), nan=1 / 3)
T = np.nan_to_num(np.load('cache/te_all.npy'), nan=1 / 3)
O = O / O.sum(2, keepdims=True); T = T / T.sum(2, keepdims=True)
acc = lambda P: float((CLASSES[P.argmax(1)] == y).mean())
for i, k in enumerate(KEYS):
    print('oof %-4s %.4f' % (k, acc(O[i])))
LO = np.log(np.clip(O, 1e-9, None)); LT = np.log(np.clip(T, 1e-9, None))

# sibling label counts for the training rows (leave-one-out)
full = known_from(tr.sentence1.values, y)
sib = []
for p, l in zip(tr.sentence1.values, y):
    c = Counter(full[p]); c[l] -= 1
    sib.append({k: v for k, v in c.items() if v > 0})
has = np.array([len(s) > 0 for s in sib])
pen = np.array([[s.get(c, 0) for c in CLASSES] for s in sib], dtype=np.float64)

best = (None, None, -1.0)
g = np.arange(0, 1.0001, 0.1)
for w1 in g:
    for w2 in np.arange(0, 1.0001 - w1 + 1e-9, 0.1):
        w = np.array([w1, w2, 1 - w1 - w2])
        Z = np.tensordot(w, LO, 1)
        for lam in LAMS:
            a = float((CLASSES[(Z + lam * pen).argmax(1)] == y).mean())
            if a > best[2]:
                best = (w, lam, a)
W, LAM, bacc = best
Z = np.tensordot(W, LO, 1)
print('weights', np.round(W, 2), 'lam', LAM)
print('oof: plain blend %.4f -> with constraint %.4f  (sibling subset %.4f)'
      % (acc(np.exp(Z)), bacc,
         float((CLASSES[(Z + LAM * pen).argmax(1)][has] == y[has]).mean())))

Zt = np.tensordot(W, LT, 1)
Pt = np.exp(Zt - Zt.max(1, keepdims=True)); Pt /= Pt.sum(1, keepdims=True)
pred = apply_constraint(Pt, te.sentence1.values, full, LAM, groups=True)
base = CLASSES[Pt.argmax(1)]
print('changed by constraint', int((pred != base).sum()))

# exact (premise, hypothesis) duplicates between train and test -> copy the label
dup = {(a, b): l for a, b, l in zip(tr.sentence1, tr.sentence2, y)}
nd = 0
for i, (a, b) in enumerate(zip(te.sentence1, te.sentence2)):
    if (a, b) in dup:
        pred[i] = dup[(a, b)]; nd += 1
print('exact duplicate overrides', nd)

sub = pd.DataFrame({'id': te.id, 'label': pred})
assert sub.id.tolist() == te.id.tolist() and sub.label.notna().all()
assert len(sub) == len(te) and set(sub.label) <= set(CLASSES)
sub.to_csv('outputs/submission.csv', index=False)
print(sub.label.value_counts().to_dict(), 'written outputs/submission.csv')
