"""Tier 2.5: feature-based matrix factorization (collaborative-filtering framing).

Drug-response is a matrix-completion problem: a (cell_line x drug) matrix of
sensitivities, mostly observed here but the framing is standard. Because test
cell lines are unseen, the line embedding must come from genomic SIDE-INFO:

    pred[i, j] = b_j + (x_i · W) · v_j

  x_i : genomic feature vector for line i (curated + tissue one-hot + bias)
  W   : F x K  (maps genomics -> a K-dim line embedding)
  v_j : K       drug embedding (free)
  b_j : drug bias

Trained by minibatch SGD on observed train pairs. A held-out line gets an
embedding W^T x_i and is scored against every drug — pure matrix-completion.
"""

from __future__ import annotations

import numpy as np

from experiments.harness import build_truth


def _feature_matrix(U, cell_cols, tissue_vals):
    X = U["feats"][cell_cols].to_numpy(dtype=np.float32)
    tiss = U["tissue_arr"]
    onehot = np.zeros((len(tiss), len(tissue_vals)), dtype=np.float32)
    tindex = {t: k for k, t in enumerate(tissue_vals)}
    for i, t in enumerate(tiss):
        if t in tindex:
            onehot[i, tindex[t]] = 1.0
    bias = np.ones((len(tiss), 1), dtype=np.float32)
    return np.hstack([X, onehot, bias])


def run_mf(U, train_idx, eval_idx, cell_cols, truth=None, K=16, epochs=25,
           lr=0.02, reg=0.02, seed=0):
    if truth is None:
        truth = build_truth(U["targets_raw"], U["drug_ids"], train_idx)
    ids = U["drug_ids"]
    didx = {d: j for j, d in enumerate(ids)}
    D = len(ids)
    Xf = _feature_matrix(U, cell_cols, U["tissue_vals"])
    F = Xf.shape[1]

    # observed train (line, drug) pairs
    rows_i, rows_j, ys = [], [], []
    tr_set = set(train_idx.tolist())
    for d in ids:
        t = truth[d]
        j = didx[d]
        for i in tr_set:
            if not np.isnan(t[i]):
                rows_i.append(i)
                rows_j.append(j)
                ys.append(t[i])
    rows_i = np.asarray(rows_i)
    rows_j = np.asarray(rows_j)
    ys = np.asarray(ys, dtype=np.float32)

    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((F, K)) * 0.05).astype(np.float32)
    V = (rng.standard_normal((D, K)) * 0.05).astype(np.float32)
    b = np.zeros(D, dtype=np.float32)

    n = len(ys)
    bs = 8192
    for _ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            i = rows_i[idx]
            j = rows_j[idx]
            y = ys[idx]
            xi = Xf[i]                      # (B, F)
            u = xi @ W                      # (B, K)
            vj = V[j]                       # (B, K)
            pred = b[j] + np.sum(u * vj, axis=1)
            e = (pred - y).astype(np.float32)       # (B,)
            B = len(idx)
            # grads
            gb = e
            gV = e[:, None] * u             # wrt v_j
            gU = e[:, None] * vj            # wrt u_i -> W
            # apply (accumulate per-index for b, V)
            np.add.at(b, j, -lr * (gb + reg * b[j]))
            np.add.at(V, j, -lr * (gV + reg * vj))
            W -= lr * (xi.T @ gU) / B + lr * reg * W

    # predict eval pairs
    pred = {}
    for d in ids:
        t = truth[d]
        j = didx[d]
        ev = [i for i in eval_idx if not np.isnan(t[i])]
        if not ev:
            continue
        u = Xf[ev] @ W
        p = b[j] + u @ V[j]
        pred[d] = {int(i): float(pp) for i, pp in zip(ev, p, strict=True)}
    return pred
