"""Single entry point for every experiment in the report.

    python run_all.py --stage all            # full run (GPU recommended, ~30-60 min incl. downloads)
    python run_all.py --smoke                # 5-minute end-to-end check on 500 clips, writes results_smoke/
    python run_all.py --stage eval report    # recompute every table/figure from saved embeddings (CPU only)

Stages: data -> embed -> eval -> report.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

from src.data import class_names, load_meta, load_waveforms

PACKAGES = ["torch", "torchaudio", "torchvision", "laion_clap", "transformers", "numpy", "librosa",
            "scikit-learn", "scipy", "pandas", "matplotlib", "soundfile", "torchlibrosa"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def record_environment(cfg, args, device):
    env = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "command": " ".join(sys.argv),
           "python": sys.version, "platform": platform.platform(), "device": device, "packages": {}}
    for p in PACKAGES:
        try:
            env["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            env["packages"][p] = None
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        env["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        env["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        env["git_commit"] = None
    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def smoke_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["results_dir"] = "results_smoke"
    cfg["reproduction"]["checkpoints"] = [cfg["main_checkpoint"]]
    cfg["noise"]["test_types"] = ["white"]
    cfg["noise"]["test_snrs_db"] = [10]
    cfg["fewshot"]["shots"] = [1]
    cfg["fewshot"]["seeds"] = [0]
    cfg["embed"]["latency_clips"] = 10
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stage", nargs="+", default=["all"], choices=["all", "data", "embed", "eval", "report"])
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run on 2 clips/class/fold")
    ap.add_argument("--device", default=None, help="cuda:0 or cpu (default: cuda if available)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.smoke:
        cfg = smoke_config(cfg)
    stages = ["data", "embed", "eval", "report"] if "all" in args.stage else args.stage
    set_seed(cfg["seed"])

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if "embed" in stages:
        record_environment(cfg, args, device)

    meta = load_meta(cfg)
    if args.smoke:  # 2 clips per class per fold -> 500 clips
        meta = meta.groupby(["label", "fold"], group_keys=False).head(2).sort_values("filename").reset_index(drop=True)

    t_start = time.time()
    if "data" in stages or "embed" in stages:
        waves = load_waveforms(cfg, load_meta(cfg))
        if args.smoke:
            full = load_meta(cfg)
            keep = full.index[full["filename"].isin(meta["filename"])].to_numpy()
            waves = np.asarray(waves[keep])
    if "embed" in stages:
        from src.embed import run_embed_stage
        run_embed_stage(cfg, waves, class_names(meta), device)
    if "eval" in stages:
        from src.evaluate import run_eval_stage
        run_eval_stage(cfg, meta)
    if "report" in stages:
        from src.report import run_report_stage
        run_report_stage(cfg, meta)
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Results: {cfg['results_dir']}/")


if __name__ == "__main__":
    main()
