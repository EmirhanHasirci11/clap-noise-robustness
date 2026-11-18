# CLAP under background noise: zero-shot reproduction and a noise-augmented few-shot cache

YZM5110 Research Project, Emirhan Hasırcı.

**Reproduced paper:** Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, S. Dubnov.
*Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation.*
ICASSP 2023. [arXiv:2211.06687](https://arxiv.org/abs/2211.06687)

## Research question

1. **Reproduction.** Do the released LAION-CLAP checkpoints reach the ESC-50 zero-shot accuracy reported in the paper
   (Table 4: 89.1 / 88.0 / 91.0 %), and does feature fusion matter on 5-second clips?
2. **Improvement.** Zero-shot CLAP has no way to adapt to a noisy deployment. A training-free few-shot cache
   (Tip-Adapter, applied to audio as in the Treff adapter) adapts with K labelled clips per class.
   Does building the cache from *clean + noise-augmented* support clips make the classifier more robust to
   background noise than a cache of clean clips only, without hurting clean accuracy?
   Controls: zero-shot, clean cache, noisy-only cache, and linear probes trained on the same clean / clean+noisy support data.

## Main results (measured on an RTX 3080; full tables in `results/RESULTS.md`, report in `report/Hasirci_YZM5110_Research.pdf`)

| | Result |
|---|---|
| Zero-shot ESC-50, 4 released checkpoints, README prompt | 90.8 - 92.3 % (paper Table 4: 88.0 - 91.0 %) |
| `630k-audioset-best`, fold 1, R@1 | 93.25 % (README: 92.75 %) |
| Bare class name instead of a sentence prompt | -5.6 to -10.3 points |
| Zero-shot at 0 dB white / pink noise | 66.4 % / 68.8 % (clean 92.3 %) |
| Noisy cache vs clean cache, K=16, white / pink noise (avg over SNRs) | +2.83 / +2.21 points, 15/15 runs better; clean: +0.15 (n.s.) |
| Same, extended alpha/beta grid (`configs/extended_grid.yaml`) | +3.51 / +3.20 points; clean: +0.03 |
| Linear probe on the same clean+noisy data, K=16 | best overall (93.1 % noisy avg, 97.2 % clean) |

The gain comes from noise-augmenting the support set; a trained linear probe uses it better than the training-free cache for K >= 4.
Brown noise turned out to be nearly clean for the model (only 0.3 % of its power lies in the 50 Hz - 14 kHz mel band, see `scripts/noise_band_power.py`).

## Quick start (Windows, NVIDIA GPU)

Python 3.11 is required (`laion_clap` needs `numpy<2`).

```powershell
cd D:\GithubRepos\clap-noise-robustness
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -m tests.test_synthetic        # 30 s, CPU only, no downloads: checks eval/report code
python run_all.py --smoke             # ~5 min: real model on 500 clips -> results_smoke/
python run_all.py --stage all         # full experiment -> results/
```

If `py -3.11` is not available: `pip install uv`, then `uv venv --python 3.11 .venv` and continue from the activate step.
Linux/macOS: same commands with `source .venv/bin/activate` (use the default PyTorch wheel for your platform).

The first run downloads ESC-50 (~600 MB, from GitHub), `roberta-base`/`bert-base-uncased` tokenizers
(Hugging Face) and four CLAP checkpoints (Hugging Face, `lukewys/laion_clap`), and caches the
48 kHz waveforms in `data/esc50_48k.npy` (~1.9 GB).

### Which results need what

| Stage | Command | Needs | Output |
|---|---|---|---|
| data | `--stage data` | internet | `data/` (not committed) |
| embed | `--stage embed` | GPU recommended, checkpoints | `results/embeddings/*.npz`, `results/resources_encoder.json`, `results/checkpoints.json` |
| eval | `--stage eval` | CPU only, saved embeddings | `results/reproduction.csv`, `results/fewshot_runs.csv`, `results/predictions_k16_seed0.csv` |
| report | `--stage report` | CPU only | `results/RESULTS.md`, `results/table*.csv`, `results/figures/*.png` |

Follow-up analysis and report (CPU only, from saved embeddings):

```powershell
python run_all.py --config configs/extended_grid.yaml --stage eval report   # wider alpha/beta grid -> results/extended_grid/
python scripts/noise_band_power.py                                           # in-band power of each noise colour
python report/make_tables.py                                                 # LaTeX tables -> report/generated/
cd report; latexmk -pdf report.tex                                           # or upload report/ to Overleaf
```

The embeddings of every condition are committed (float16, compressed), so **all tables and figures can be
regenerated on a CPU without the dataset or checkpoints**: `python run_all.py --stage eval report`
(this still reads ESC-50 metadata, which is downloaded automatically, ~600 MB).

## Experimental protocol

* **Data.** ESC-50 (2000 clips, 50 classes, 5 s, official 5 folds), resampled 44.1 -> 48 kHz with `librosa.load`
  (the loader used by `laion_clap`), then int16-quantised inside `laion_clap` as in the official API.
* **Reproduction.** Four released checkpoints (`630k-best`, `630k-audioset-best`, `630k-fusion-best`,
  `630k-audioset-fusion-best`) x three prompt templates, zero-shot top-1 on all 2000 clips and on fold 1
  (the split used in the official README example).
* **Noise.** White, pink (1/f) and brown (1/f^2) Gaussian noise added at 20, 10, 5 and 0 dB SNR
  (power ratio over the whole clip, `src/noise.py`). Noise is a deterministic function of the clip index and
  condition, so every method sees identical inputs.
* **Few-shot split.** For test fold *f*, validation fold is *f mod 5 + 1*; the K support clips per class
  (K in {1, 4, 16}) are drawn from the remaining three folds with 3 seeds. 5 folds x 3 seeds = 15 runs per setting.
* **Methods** (all on frozen `630k-audioset-best` embeddings, prompt `This is a sound of {}.`):
  * `zero-shot`: 100 * cos(audio, prompt).
  * `tip-clean`: Tip-Adapter cache of the K clean support clips, `logits = zs + alpha * exp(-beta(1 - a k^T)) L`.
  * `tip-aug` (**the intervention**): cache of the K clean clips **plus** the same clips with white noise at a random SNR in [0, 20] dB.
  * `tip-noisy-only`: cache of the noisy copies only (ablation).
  * `lp-clean` / `lp-aug`: logistic-regression probes on the same support sets (control: is any gain due to the
    augmented data or to the cache mechanism?).
* **No test-set tuning.** alpha, beta (Tip) and C (LP) are chosen on the **clean** validation fold only.
  Pink and brown noise are never used to build the cache, so they measure robustness to *unseen* noise.
* **Resources.** Audio-encoder latency (batch 1), throughput (batch 32), peak GPU memory; adapter tuning time,
  inference time and cache size.

## Repository layout

```
run_all.py              single entry point (stages data/embed/eval/report, --smoke)
configs/default.yaml    every setting used for the reported results
configs/extended_grid.yaml  follow-up: wider alpha/beta grid (eval/report stages only)
scripts/noise_band_power.py in-band noise power per colour
report/                 LaTeX source, generated tables, make_tables.py, final PDF
src/data.py             ESC-50 download, 48 kHz cache, fold logic
src/noise.py            noise generation and SNR mixing
src/embed.py            checkpoint download (+sha256), CLAP embeddings, latency/memory
src/fewshot.py          zero-shot, Tip-Adapter cache, linear probe, validation-fold tuning
src/evaluate.py         reproduction table, 15-run few-shot experiment, predictions
src/report.py           tables (RESULTS.md, CSV), paired 95% CIs, failure analysis, figures
tests/test_synthetic.py CPU-only test on synthetic embeddings
results/                committed outputs of the reported run (environment.json pins versions, GPU and git commit)
```

## Licences and data access

* Code: MIT (see `LICENSE`).
* ESC-50: CC BY-NC 3.0, downloaded from <https://github.com/karolpiczak/ESC-50>; audio is **not** redistributed here.
* LAION-CLAP code and checkpoints: <https://github.com/LAION-AI/CLAP>, <https://huggingface.co/lukewys/laion_clap>; checkpoints are not redistributed.

## AI-use disclosure

I selected the reproduced paper and the reviewed literature. Claude (Anthropic) was used to search for and summarise related work, write and test the experiment code, and determine the flow of the report, as well as create the charts and tables placed while I wrote the text. I ran the experiments, checked the outputs, and am responsible for all claims.

## Citation

```bibtex
@inproceedings{laionclap2023,
  title     = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author    = {Wu, Yusong and Chen, Ke and Zhang, Tianyu and Hui, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  booktitle = {ICASSP},
  year      = {2023}
}
```
