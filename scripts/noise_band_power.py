"""Fraction of each noise type's power that falls inside the CLAP mel band (50 Hz - 14 kHz).

Explains why 'brown 0 dB' barely changes accuracy: the nominal SNR is computed over the full
band, but almost all brown-noise power lies below the 50 Hz lower edge of the mel filterbank.

    python scripts/noise_band_power.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.noise import colored_noise  # noqa: E402

N, SR, FMIN, FMAX = 240000, 48000, 50, 14000  # HTSAT-tiny audio_cfg in laion_clap 1.1.7

f = np.fft.rfftfreq(N, 1 / SR)
band = (f >= FMIN) & (f <= FMAX)
for kind in ["white", "pink", "brown"]:
    frac = np.mean([
        (lambda P: P[band].sum() / P.sum())(np.abs(np.fft.rfft(colored_noise(kind, N, np.random.default_rng(s)))) ** 2)
        for s in range(5)])
    print(f"{kind:6s} in-band power fraction {frac:.4f} -> in-band SNR is {-10 * np.log10(frac):4.1f} dB above nominal")
