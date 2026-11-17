"""LAION-CLAP model loading, embedding extraction and resource measurement."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import noise as nz
from .data import download


# ---------------------------------------------------------------- checkpoints

def checkpoint_path(cfg, name: str) -> Path:
    path = Path(cfg["checkpoint_dir"]) / f"{name}.pt"
    download(cfg["checkpoint_url"].format(name=name), path)
    return path


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_model(cfg, name: str, device: str):
    import laion_clap  # imported lazily: it downloads tokenizers on import

    path = checkpoint_path(cfg, name)
    fusion = "fusion" in name
    model = laion_clap.CLAP_Module(enable_fusion=fusion, amodel="HTSAT-tiny", device=device)
    model.load_ckpt(str(path), verbose=False)
    model.eval()
    return model, path


# ---------------------------------------------------------------- embeddings

def embed_text(model, prompts: list[str]) -> np.ndarray:
    import torch

    with torch.no_grad():
        return np.asarray(model.get_text_embedding(prompts), dtype=np.float32)


def embed_audio(model, waves, batch_size: int, transform=None, desc: str = "") -> np.ndarray:
    """Embed every row of `waves`; `transform(i, wave)` can corrupt clip i first."""
    import torch

    n = len(waves)
    out = []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, batch_size):
            idx = range(s, min(n, s + batch_size))
            batch = np.stack([
                transform(i, np.asarray(waves[i], dtype=np.float32)) if transform else np.asarray(waves[i], dtype=np.float32)
                for i in idx
            ])
            out.append(np.asarray(model.get_audio_embedding_from_data(batch, use_tensor=False), dtype=np.float32))
            done = idx[-1] + 1
            if done % (batch_size * 20) == 0 or done == n:
                print(f"  [{desc}] {done}/{n} clips, {time.time() - t0:6.1f} s")
    return np.concatenate(out, axis=0)


def save_emb(path: Path, arr: np.ndarray, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, emb=arr.astype(np.float16), **extra)


def load_emb(path: Path) -> np.ndarray:
    return np.load(path)["emb"].astype(np.float32)


def emb_dir(cfg) -> Path:
    """Embeddings live in results/embeddings unless a config points elsewhere (e.g. a re-analysis)."""
    return Path(cfg.get("embeddings_dir") or Path(cfg["results_dir"]) / "embeddings")


# ---------------------------------------------------------------- resources

def measure_resources(model, waves, device: str, n_clips: int, batch_size: int) -> dict:
    """Audio-encoder latency (batch 1), throughput (batch B) and peak GPU memory."""
    import torch

    cuda = device.startswith("cuda") and torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    clips = np.stack([np.asarray(waves[i], dtype=np.float32) for i in range(min(n_clips, len(waves)))])
    res = {"device": device, "n_clips": int(len(clips)), "batch_size": batch_size}
    with torch.no_grad():
        for i in range(3):  # warm-up
            model.get_audio_embedding_from_data(clips[i:i + 1])
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(len(clips)):
            sync()
            t = time.perf_counter()
            model.get_audio_embedding_from_data(clips[i:i + 1])
            sync()
            times.append((time.perf_counter() - t) * 1000)
        res["latency_bs1_ms_median"] = float(np.median(times))
        res["latency_bs1_ms_p90"] = float(np.percentile(times, 90))
        if cuda:
            res["peak_mem_bs1_MB"] = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
        sync()
        t = time.perf_counter()
        for s in range(0, len(clips), batch_size):
            model.get_audio_embedding_from_data(clips[s:s + batch_size])
        sync()
        total = time.perf_counter() - t
        res["throughput_clips_per_s"] = len(clips) / total
        res["per_clip_ms_batched"] = total * 1000 / len(clips)
        if cuda:
            res[f"peak_mem_bs{batch_size}_MB"] = torch.cuda.max_memory_allocated() / 2**20
            res["gpu"] = torch.cuda.get_device_name(0)
    return res


# ---------------------------------------------------------------- stage

def run_embed_stage(cfg, waves, names: list[str], device: str, only_main: bool = False):
    """Embeds clean audio for every checkpoint (+ templates), and noisy/augmented audio for the main one."""
    out_dir = emb_dir(cfg)
    templates = cfg["reproduction"]["templates"]
    ckpts = [cfg["main_checkpoint"]] if only_main else cfg["reproduction"]["checkpoints"]
    ckpt_info = {}
    for name in ckpts:
        print(f"\n[embed] checkpoint {name}")
        model, path = load_model(cfg, name, device)
        ckpt_info[name] = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}

        text = np.stack([embed_text(model, [t.format(c) for c in names]) for t in templates])
        save_emb(out_dir / f"{name}__text.npz", text, templates=np.array(templates))

        f = out_dir / f"{name}__clean.npz"
        if not f.exists():
            save_emb(f, embed_audio(model, waves, cfg["embed"]["batch_size"], desc=f"{name} clean"))

        if name == cfg["main_checkpoint"]:
            ncfg = cfg["noise"]
            for kind in ncfg["test_types"]:
                for snr in ncfg["test_snrs_db"]:
                    cond = nz.condition_name(kind, snr)
                    f = out_dir / f"{name}__{cond}.npz"
                    if f.exists():
                        continue
                    tf = (lambda k, s: (lambda i, w: nz.test_condition(i, k, s, w)))(kind, snr)
                    save_emb(f, embed_audio(model, waves, cfg["embed"]["batch_size"], tf, desc=f"{name} {cond}"))
            f = out_dir / f"{name}__aug.npz"
            if not f.exists():
                snrs = np.zeros(len(waves), dtype=np.float32)

                def tf_aug(i, w):
                    y, snrs[i] = nz.augmentation(i, ncfg["aug_type"], tuple(ncfg["aug_snr_range_db"]), w)
                    return y

                emb = embed_audio(model, waves, cfg["embed"]["batch_size"], tf_aug, desc=f"{name} aug")
                save_emb(f, emb, snr_db=snrs)

            print("[embed] measuring encoder latency / memory ...")
            res = measure_resources(model, waves, device, cfg["embed"]["latency_clips"], cfg["embed"]["batch_size"])
            Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
            with open(Path(cfg["results_dir"]) / "resources_encoder.json", "w") as fh:
                json.dump(res, fh, indent=2)
            print(json.dumps(res, indent=2))

        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    info_path = Path(cfg["results_dir"]) / "checkpoints.json"
    old = json.loads(info_path.read_text()) if info_path.exists() else {}
    old.update(ckpt_info)
    info_path.write_text(json.dumps(old, indent=2))
