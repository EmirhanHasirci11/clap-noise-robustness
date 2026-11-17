"""CPU-only test of the evaluation and reporting code on synthetic embeddings.

No model, dataset or GPU needed:  python -m tests.test_synthetic
It checks that every stage after `embed` runs and produces the expected files.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import noise as nz
from src.embed import save_emb
from src.evaluate import run_eval_stage
from src.report import run_report_stage


def make_fake(cfg, rng):
    n_cls, per_fold, d = 50, 8, 512
    rows = [dict(filename=f"{f}-{c}-{i}.wav", fold=f, label=c, class_name=f"class {c}")
            for c in range(n_cls) for f in range(1, 6) for i in range(per_fold)]
    meta = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    y = meta["label"].to_numpy()

    def unit(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    centers = unit(rng.standard_normal((n_cls, d)))
    clean = unit(centers[y] + 0.9 * rng.standard_normal((len(y), d)) / np.sqrt(d) * 8)
    noise_dir = unit(rng.standard_normal(d))
    out = Path(cfg["results_dir"]) / "embeddings"
    name = cfg["main_checkpoint"]
    templates = cfg["reproduction"]["templates"]
    text = np.stack([unit(centers + 0.3 * rng.standard_normal((n_cls, d)) / np.sqrt(d) * 8) for _ in templates])
    for ck in cfg["reproduction"]["checkpoints"]:
        save_emb(out / f"{ck}__text.npz", text, templates=np.array(templates))
        save_emb(out / f"{ck}__clean.npz", clean)
    for kind in cfg["noise"]["test_types"]:
        for snr in cfg["noise"]["test_snrs_db"]:
            w = 1.0 / (1.0 + 10 ** (snr / 20))
            save_emb(out / f"{name}__{nz.condition_name(kind, snr)}.npz", unit((1 - w) * clean + w * noise_dir))
    save_emb(out / f"{name}__aug.npz", unit(0.7 * clean + 0.3 * noise_dir))
    (Path(cfg["results_dir"]) / "resources_encoder.json").write_text('{"device": "synthetic"}')
    return meta


def test_pipeline():
    cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg["results_dir"] = str(tmp / "results")
        cfg["reproduction"]["checkpoints"] = [cfg["main_checkpoint"]]
        cfg["noise"]["test_types"] = ["white", "pink"]
        cfg["noise"]["test_snrs_db"] = [10, 0]
        cfg["fewshot"]["shots"] = [1, 4]
        cfg["fewshot"]["seeds"] = [0, 1]
        meta = make_fake(cfg, np.random.default_rng(0))
        run_eval_stage(cfg, meta)
        run_report_stage(cfg, meta)
        res = Path(cfg["results_dir"])
        for f in ["reproduction.csv", "fewshot_runs.csv", "summary_fewshot.csv", "table3_paired.csv",
                  "per_class.csv", "RESULTS.md", "figures/fig1_acc_vs_snr.png", "figures/fig2_acc_vs_shots.png"]:
            assert (res / f).exists(), f
        runs = pd.read_csv(res / "fewshot_runs.csv")
        assert runs["acc"].between(0, 1).all()
        # 5 folds x 5 conditions zero-shot + 5 folds x 2 seeds x 2 shots x 5 methods x 5 conditions
        assert len(runs) == 25 + 5 * 2 * 2 * 5 * 5, len(runs)
        print("synthetic pipeline test passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_noise_snr():
    rng = np.random.default_rng(0)
    clean = (0.1 * rng.standard_normal(48000)).astype(np.float32)
    for kind in nz.NOISE_TYPES:
        for snr in [20, 10, 0]:
            y = nz.test_condition(3, kind, snr, clean)
            n = y - clean
            meas = 10 * np.log10(np.mean(clean.astype(float) ** 2) / np.mean(n.astype(float) ** 2))
            assert abs(meas - snr) < 0.05, (kind, snr, meas)
            assert np.array_equal(y, nz.test_condition(3, kind, snr, clean))  # deterministic
    print("noise SNR test passed")


if __name__ == "__main__":
    test_noise_snr()
    test_pipeline()
