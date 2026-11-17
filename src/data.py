"""ESC-50 download, resampling to 48 kHz and fold handling."""
from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100.0, 100.0 * done / total_size)
        sys.stdout.write(f"\r  {done / 1e6:8.1f} MB / {total_size / 1e6:8.1f} MB ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r  {done / 1e6:8.1f} MB")
    sys.stdout.flush()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[data] already downloaded: {dest}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[data] downloading {url}")
    urllib.request.urlretrieve(url, tmp, _progress)
    print()
    shutil.move(tmp, dest)
    return dest


def esc50_root(cfg) -> Path:
    return Path(cfg["data_dir"]) / "ESC-50-master"


def ensure_esc50(cfg) -> Path:
    root = esc50_root(cfg)
    if (root / "meta" / "esc50.csv").exists():
        return root
    zip_path = download(cfg["esc50_url"], Path(cfg["data_dir"]) / "esc50.zip")
    print("[data] extracting ESC-50 ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cfg["data_dir"])
    if not (root / "meta" / "esc50.csv").exists():
        raise FileNotFoundError(f"ESC-50 metadata not found under {root}")
    return root


def load_meta(cfg) -> pd.DataFrame:
    """Metadata sorted by filename, with a class index and a readable class name."""
    root = ensure_esc50(cfg)
    meta = pd.read_csv(root / "meta" / "esc50.csv").sort_values("filename").reset_index(drop=True)
    meta["label"] = meta["target"].astype(int)
    meta["class_name"] = meta["category"].str.replace("_", " ")
    return meta


def class_names(meta: pd.DataFrame) -> list[str]:
    by_idx = meta.drop_duplicates("label").sort_values("label")
    names = by_idx["class_name"].tolist()
    assert len(names) == 50 and by_idx["label"].tolist() == list(range(50))
    return names


def load_waveforms(cfg, meta: pd.DataFrame) -> np.ndarray:
    """All clips at 48 kHz as a (N, 240000) float32 array, cached on disk."""
    sr = cfg["sample_rate"]
    n_samples = sr * cfg["clip_seconds"]
    cache = Path(cfg["data_dir"]) / f"esc50_{sr // 1000}k.npy"
    if cache.exists():
        arr = np.load(cache, mmap_mode="r")
        if arr.shape == (len(meta), n_samples):
            return arr
    import librosa  # same loader/resampler as laion_clap.get_audio_embedding_from_filelist

    root = esc50_root(cfg)
    print(f"[data] resampling {len(meta)} clips to {sr} Hz (one-time, cached in {cache}) ...")
    out = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32, shape=(len(meta), n_samples))
    for i, fn in enumerate(meta["filename"]):
        y, _ = librosa.load(root / "audio" / fn, sr=sr, mono=True)
        y = y[:n_samples]
        if len(y) < n_samples:
            y = np.pad(y, (0, n_samples - len(y)))
        out[i] = y
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(meta)}")
    out.flush()
    del out
    return np.load(cache, mmap_mode="r")


def val_fold_for(test_fold: int) -> int:
    """Fold used for hyper-parameter selection when `test_fold` is held out (1->2, ..., 5->1)."""
    return test_fold % 5 + 1
