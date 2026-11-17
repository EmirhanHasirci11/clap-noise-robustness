"""Experiments on saved embeddings: reproduction table and few-shot robustness runs.

Runs on CPU in a few minutes; needs only results/embeddings/*.npz and the ESC-50 metadata.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import fewshot as fs
from . import noise as nz
from .data import val_fold_for
from .embed import emb_dir, load_emb

FOLDS = [1, 2, 3, 4, 5]


def conditions(cfg) -> list[str]:
    n = cfg["noise"]
    return ["clean"] + [nz.condition_name(k, s) for k in n["test_types"] for s in n["test_snrs_db"]]


# ---------------------------------------------------------------- reproduction

def reproduction_table(cfg, labels, folds) -> pd.DataFrame:
    rows = []
    for name in cfg["reproduction"]["checkpoints"]:
        fa, ft = emb_dir(cfg) / f"{name}__clean.npz", emb_dir(cfg) / f"{name}__text.npz"
        if not (fa.exists() and ft.exists()):
            print(f"[eval] skipping {name}: embeddings not found")
            continue
        A = load_emb(fa)
        texts = np.load(ft)
        T_all, templates = texts["emb"].astype(np.float32), [str(t) for t in texts["templates"]]
        for t_i, tmpl in enumerate(templates):
            sims = A @ T_all[t_i].T
            rank = np.argsort(-sims, axis=1)
            pos = np.argmax(rank == labels[:, None], axis=1)  # 0 = correct class ranked first
            fold_acc = [float(np.mean(pos[folds == f] == 0)) for f in FOLDS]
            f1 = folds == 1
            rows.append({
                "checkpoint": name,
                "fusion": "fusion" in name,
                "template": tmpl,
                "acc_all": float(np.mean(pos == 0)),
                "acc_fold_mean": float(np.mean(fold_acc)),
                "acc_fold_std": float(np.std(fold_acc, ddof=1)),
                "fold1_R@1": float(np.mean(pos[f1] < 1)),
                "fold1_R@5": float(np.mean(pos[f1] < 5)),
                "fold1_mAP@10": float(np.mean(np.where(pos[f1] < 10, 1.0 / (pos[f1] + 1), 0.0))),
                **{f"acc_fold{f}": a for f, a in zip(FOLDS, fold_acc)},
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- few-shot robustness

def fewshot_runs(cfg, labels, folds):
    name, fcfg = cfg["main_checkpoint"], cfg["fewshot"]
    d = emb_dir(cfg)
    texts = np.load(d / f"{name}__text.npz")
    templates = [str(t) for t in texts["templates"]]
    T = texts["emb"].astype(np.float32)[templates.index(cfg["main_template"])]
    E = {c: load_emb(d / f"{name}__{c}.npz") for c in conditions(cfg)}
    E_aug = load_emb(d / f"{name}__aug.npz")
    n_cls, scale = T.shape[0], fcfg["zs_logit_scale"]

    rows, timing = [], []
    preds = {}  # (method, condition) -> per-clip predictions, filled for seed 0 and the largest K
    k_pred = max(fcfg["shots"])

    for f in FOLDS:
        v = val_fold_for(f)
        pool = [p for p in FOLDS if p not in (f, v)]
        te, va = np.where(folds == f)[0], np.where(folds == v)[0]
        val_A, val_y = E["clean"][va], labels[va]

        for cond in E:
            lg = fs.zs_logits(E[cond][te], T, scale)
            rows.append(dict(test_fold=f, seed=-1, k=0, method="zero-shot", condition=cond,
                             acc=fs.accuracy(lg, labels[te])))
            preds.setdefault(("zero-shot", cond), np.full(len(labels), -1))[te] = lg.argmax(1)

        for seed in fcfg["seeds"]:
            for k in fcfg["shots"]:
                rng = np.random.default_rng([cfg["seed"], seed, f, k])
                sup = fs.sample_support(labels, folds, pool, k, rng)
                y_sup = labels[sup]
                variants = {
                    "clean": (E["clean"][sup], y_sup),
                    "aug": (np.concatenate([E["clean"][sup], E_aug[sup]]), np.concatenate([y_sup, y_sup])),
                    "noisy-only": (E_aug[sup], y_sup),
                }
                for vname, (keys, key_y) in variants.items():
                    method = f"tip-{vname}"
                    t0 = time.perf_counter()
                    a, b, vacc = fs.tune_tip(val_A, val_y, keys, key_y, T, n_cls, scale,
                                             fcfg["tip_alpha"], fcfg["tip_beta"])
                    t_tune = time.perf_counter() - t0
                    for cond in E:
                        A = E[cond][te]
                        t0 = time.perf_counter()
                        lg = fs.tip_logits(A, keys, key_y, n_cls, a, b, fs.zs_logits(A, T, scale))
                        t_inf = time.perf_counter() - t0
                        rows.append(dict(test_fold=f, seed=seed, k=k, method=method, condition=cond,
                                         acc=fs.accuracy(lg, labels[te]), alpha=a, beta=b, val_acc=vacc))
                        if seed == 0 and k == k_pred:
                            preds.setdefault((method, cond), np.full(len(labels), -1))[te] = lg.argmax(1)
                        if cond == "clean":
                            timing.append(dict(method=method, k=k, tune_s=t_tune,
                                               infer_ms_per_clip=1000 * t_inf / len(te),
                                               cache_entries=len(keys), cache_MB=keys.nbytes / 2**20))

                for vname in ("clean", "aug"):
                    method = f"lp-{vname}"
                    X, yy = variants[vname]
                    t0 = time.perf_counter()
                    clf, C, vacc = fs.tune_lp(X, yy, val_A, val_y, fcfg["lp_C"], seed)
                    t_tune = time.perf_counter() - t0
                    for cond in E:
                        A = E[cond][te]
                        t0 = time.perf_counter()
                        p = clf.predict(A)
                        t_inf = time.perf_counter() - t0
                        rows.append(dict(test_fold=f, seed=seed, k=k, method=method, condition=cond,
                                         acc=float(np.mean(p == labels[te])), C=C, val_acc=vacc))
                        if seed == 0 and k == k_pred:
                            preds.setdefault((method, cond), np.full(len(labels), -1))[te] = p
                        if cond == "clean":
                            timing.append(dict(method=method, k=k, tune_s=t_tune,
                                               infer_ms_per_clip=1000 * t_inf / len(te),
                                               cache_entries=len(X), cache_MB=X.nbytes / 2**20))
                print(f"  fold {f} seed {seed} k {k:2d} done")

        # zero-shot timing (no tuning)
        t0 = time.perf_counter()
        fs.zs_logits(E["clean"][te], T, scale).argmax(1)
        timing.append(dict(method="zero-shot", k=0, tune_s=0.0,
                           infer_ms_per_clip=1000 * (time.perf_counter() - t0) / len(te),
                           cache_entries=0, cache_MB=0.0))

    pred_df = pd.DataFrame({f"{m}|{c}": v for (m, c), v in preds.items()})
    return pd.DataFrame(rows), pd.DataFrame(timing), pred_df


def run_eval_stage(cfg, meta):
    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    labels, folds = meta["label"].to_numpy(), meta["fold"].to_numpy()

    print("[eval] reproduction table")
    rep = reproduction_table(cfg, labels, folds)
    rep.to_csv(out / "reproduction.csv", index=False)
    print(rep[["checkpoint", "template", "acc_all", "fold1_R@1"]].to_string(index=False))

    print("[eval] few-shot robustness runs")
    runs, timing, preds = fewshot_runs(cfg, labels, folds)
    runs.to_csv(out / "fewshot_runs.csv", index=False)
    timing.groupby(["method", "k"]).mean(numeric_only=True).reset_index().to_csv(
        out / "resources_adapters.csv", index=False)
    preds.insert(0, "filename", meta["filename"])
    preds.insert(1, "label", labels)
    preds.insert(2, "fold", folds)
    preds.to_csv(out / f"predictions_k{max(cfg['fewshot']['shots'])}_seed0.csv", index=False)
    with open(out / "eval_done.json", "w") as fh:
        json.dump({"n_runs": int(len(runs))}, fh)
