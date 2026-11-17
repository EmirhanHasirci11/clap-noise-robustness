"""Additive background noise at a controlled SNR.

Noise is generated deterministically from (seed space, clip index, condition), so every
method and every checkpoint sees exactly the same corrupted waveform.
"""
from __future__ import annotations

import numpy as np

NOISE_TYPES = ("white", "pink", "brown")
TEST_SEED_SPACE = 12345   # noise added to evaluation clips
AUG_SEED_SPACE = 67890    # noise added to support (cache) clips, disjoint from test noise


def colored_noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance white (flat), pink (1/f power) or brown (1/f^2 power) noise."""
    white = rng.standard_normal(n)
    if kind == "white":
        x = white
    elif kind in ("pink", "brown"):
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(n)
        f[0] = f[1]  # avoid division by zero at DC
        spec = spec / (np.sqrt(f) if kind == "pink" else f)
        x = np.fft.irfft(spec, n)
    else:
        raise ValueError(f"unknown noise type {kind}")
    x = x - x.mean()
    return (x / (x.std() + 1e-12)).astype(np.float32)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """clean + g*noise with 10*log10(P_clean / P_noise) = snr_db over the whole clip.

    If the mixture would clip (|y| > 1) the whole mixture is scaled down, which keeps the SNR.
    """
    p_clean = float(np.mean(clean.astype(np.float64) ** 2))
    p_clean = max(p_clean, 1e-10)  # (almost) silent clips still get a defined noise level
    p_noise = float(np.mean(noise.astype(np.float64) ** 2))
    gain = np.sqrt(p_clean / (p_noise * 10.0 ** (snr_db / 10.0)))
    y = clean + gain * noise
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def test_condition(clip_idx: int, kind: str, snr_db: float, clean: np.ndarray) -> np.ndarray:
    type_id = NOISE_TYPES.index(kind)
    rng = np.random.default_rng([TEST_SEED_SPACE, clip_idx, type_id, int(round(snr_db * 10)) + 1000])
    return mix_at_snr(clean, colored_noise(kind, len(clean), rng), snr_db)


def augmentation(clip_idx: int, kind: str, snr_range: tuple[float, float], clean: np.ndarray):
    rng = np.random.default_rng([AUG_SEED_SPACE, clip_idx])
    snr = float(rng.uniform(*snr_range))
    return mix_at_snr(clean, colored_noise(kind, len(clean), rng), snr), snr


def condition_name(kind: str, snr_db: float) -> str:
    return f"{kind}_{int(snr_db)}dB"
