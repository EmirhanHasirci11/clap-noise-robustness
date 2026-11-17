"""Zero-shot classifier, Tip-Adapter style cache and linear-probe control (all on frozen CLAP embeddings).

Embeddings from LAION-CLAP are already L2-normalised, so dot products are cosine similarities.
"""
from __future__ import annotations

import warnings

import numpy as np


def onehot(y: np.ndarray, n_cls: int) -> np.ndarray:
    out = np.zeros((len(y), n_cls), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def zs_logits(A: np.ndarray, T: np.ndarray, scale: float) -> np.ndarray:
    """Zero-shot logits: scaled cosine similarity between audio and class-prompt embeddings."""
    return scale * A @ T.T


def tip_logits(A, keys, key_y, n_cls, alpha, beta, zs) -> np.ndarray:
    """Tip-Adapter (Zhang et al., 2022, Eq. 3-4) with audio keys instead of image keys:

        affinity = exp(-beta * (1 - A @ keys^T))
        logits   = zs + alpha * affinity @ onehot(key_y)
    """
    aff = np.exp(-beta * (1.0 - A @ keys.T))
    return zs + alpha * aff @ onehot(key_y, n_cls)


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=1) == y))


def sample_support(labels, folds, pool_folds, k, rng) -> np.ndarray:
    """k clip indices per class, drawn only from `pool_folds`."""
    idx = []
    in_pool = np.isin(folds, pool_folds)
    for c in np.unique(labels):
        cand = np.where((labels == c) & in_pool)[0]
        idx.append(rng.choice(cand, size=k, replace=False))
    return np.concatenate(idx)


def tune_tip(val_A, val_y, keys, key_y, T, n_cls, scale, alphas, betas):
    """Grid search on the validation fold (clean audio). Returns (alpha, beta, val_acc)."""
    zs = zs_logits(val_A, T, scale)
    sims = val_A @ keys.T
    L = onehot(key_y, n_cls)
    best = (None, None, -1.0)
    for b in betas:
        cache = np.exp(-b * (1.0 - sims)) @ L
        for a in alphas:
            acc = accuracy(zs + a * cache, val_y)
            if acc > best[2] + 1e-12:
                best = (a, b, acc)
    return best


def fit_lp(X, y, C, seed):
    from sklearn.linear_model import LogisticRegression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)
        clf.fit(X, y)
    return clf


def tune_lp(X, y, val_A, val_y, Cs, seed):
    """Select C on the validation fold. Returns (fitted model, C, val_acc)."""
    best = (None, None, -1.0)
    for C in Cs:
        clf = fit_lp(X, y, C, seed)
        acc = float(np.mean(clf.predict(val_A) == val_y))
        if acc > best[2] + 1e-12:
            best = (clf, C, acc)
    return best
