# The Fourier Shift — spatial vs. frequency-domain detection of AI-generated images

CS 7643 Deep Learning, final project.

We ask whether an AI-image detector that reads an image's **frequency fingerprint** generalizes to
unseen generators better than one that reads its **pixels**. On [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
(120k 32×32 images; 60k real CIFAR-10, 60k latent-diffusion fakes) we train the *same* ResNet-18
backbone on two input representations — raw RGB, and the centered log-magnitude 2-D FFT — so the
representation is the only variable. Around that controlled core we add a pretrained ConvNeXt-Tiny
arm in both domains, a two-stream RGB+FFT fusion model, four ablations on the front-end and on fusion,
and two distribution-shift probes: **cross-generator** (test against DDPM samples, a generator family
never seen in training) and **post-processing robustness** (JPEG recompression, Gaussian blur). Labels are
fixed as `REAL = 0, FAKE = 1` throughout; F1 and AUROC treat FAKE as the positive class, and AUROC is
reported unflipped — a value below 0.5 is a result, not a bug.

**Convention note for readers of the results:** the FFT front-end (`fft2` → `fftshift` → `log1p|·|` →
standardize) is a *fixed, non-learned* transform. Phase is discarded. That is a deliberate information
bottleneck, and it is the thing the experiments are about.

---

## Setup

Requires Python 3.12 and, for practical runtimes, an NVIDIA GPU (developed on an RTX 4060, 8 GB).
CPU works but is slow. Commands below are PowerShell (Windows 11); on Linux/macOS substitute
`source .venv/bin/activate` and forward-slash paths — no code is platform-specific.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# torch first, from the pytorch index (see the note at the top of requirements.txt for other CUDA versions)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`diffusers` / `transformers` / `accelerate` are commented out in `requirements.txt`. They are needed
**only** to regenerate the DDPM evaluation set (`src/generate_fakes.py`); uncomment them, or install
them ad hoc, if you want that step.

## Dataset

CIFAKE is ~120k JPEGs, a few hundred MB. Download via `kagglehub` (needs Kaggle API credentials at
`~/.kaggle/kaggle.json`) and copy the result to `data/cifake`:

```powershell
python -c "import kagglehub; print(kagglehub.dataset_download('birdy654/cifake-real-and-ai-generated-synthetic-images'))"
# copy the printed folder to data\cifake  (copy, not symlink — the loader does not follow links on Windows)
```

The expected layout, which the loader assumes:

```
data/cifake/{train,test}/{REAL,FAKE}/*.jpg      100k train / 20k test, balanced
data/ddpm_fakes/*.jpg                           4,000 DDPM samples (cross-generator eval)
data/fft_stats.json                             cached normalization stats, written on first use
```

`data/` is gitignored — nothing here ships with the repo.

**The validation split is derived from a keyed hash of each file's path, not an RNG**, so all 5,000
validation images (2,500 per class) are identical across every run and every machine, with no seed
plumbing. Normalization statistics — both RGB and FFT — are accumulated in a single streaming pass
over the *training* split only, with validation images excluded.

Regenerating the DDPM set is optional and takes a few minutes on a GPU:

```powershell
python -m src.generate_fakes --n 4000 --steps 50
```

It samples `google/ddpm-cifar10-32` via DDIM and writes JPEGs at **quality 75, 4:2:0** — matching
CIFAKE's own encoding exactly. This matters: CIFAKE stores REAL and FAKE identically (JPEG, q75,
4:2:0), so saving the DDPM samples as PNG would have turned the cross-generator evaluation into a
compression detector rather than a generator-fingerprint detector.

## Reproducing the experiments

Smoke test first — it is the gate, and it finishes in about 90 seconds:

```powershell
python -m src.train --arch resnet18 --domain fft --epochs 1 --limit 2000 --bs 256 --run smoke
```

Then the full queue, which runs every training job and every evaluation condition end to end:

```powershell
.\run_all.ps1
```

It calls `.venv\Scripts\python.exe` directly, so it does not need the venv activated. It is
**resumable and idempotent**: a run whose `metrics.json` already exists is skipped, and evaluations
already present in `all_evals.csv` are skipped, so re-invoking it after an interruption picks up where
it stopped. Each training job is a separate process, so GPU memory is fully released between runs.

The queue trains these eleven runs (per-arch defaults for epochs / batch size / learning rate live in
`src/train.py`; timings are for an RTX 4060):

| Run name | Arch | Domain | Seeds | Purpose | Approx. time |
|---|---|---|---|---|---|
| `e1_rgb_s{0,1,2}` | resnet18 (scratch) | rgb | 0,1,2 | spatial baseline | ~8 min each |
| `e2_fft_s{0,1,2}` | resnet18 (scratch) | fft | 0,1,2 | frequency arm | ~8 min each |
| `e3_convnext_rgb_s0` | ConvNeXt-Tiny (pretrained) | rgb | 0 | transfer learning, spatial | ~6 min |
| `e4_convnext_fft_s0` | ConvNeXt-Tiny (pretrained) | fft | 0 | transfer learning, spectral | ~6 min |
| `e5_twostream_s{0,1,2}` | two-stream | both | 0,1,2 | RGB+FFT fusion | ~12 min each |
| `a1_fft_raw_s0` | resnet18 | fft | 0 | ablation: raw magnitude, no log scaling | ~8 min |
| `a2_fft_gray_s0` | resnet18 | fft | 0 | ablation: grayscale spectrum | ~8 min |
| `a3_twostream_rgb2_s0` | two-stream | rgb2 | 0 | ablation: RGB+RGB fusion — capacity control, so a two-stream gain is not just 2× the parameters | ~12 min |
| `a4_fft_masklf4_s0` | resnet18 | fft | 0 | ablation: central low-frequency disc (radius 4) masked out, so DC / brightness cannot carry the class | ~8 min |

Individual runs follow this pattern:

```powershell
python -m src.train --arch resnet18  --domain fft  --seed 0 --run e2_fft_s0
python -m src.train --arch convnext  --domain rgb  --seed 0 --run e3_convnext_rgb_s0   # inputs resized to 128
python -m src.train --arch twostream --domain both --seed 0 --run e5_twostream_s0
python -m src.train --arch resnet18  --domain fft  --seed 0 --log 0     --run a1_fft_raw_s0
python -m src.train --arch resnet18  --domain fft  --seed 0 --gray 1    --run a2_fft_gray_s0
python -m src.train --arch twostream --domain rgb2 --seed 0             --run a3_twostream_rgb2_s0
python -m src.train --arch resnet18  --domain fft  --seed 0 --mask_lf 4 --run a4_fft_masklf4_s0
```

The first epoch of any full run is much slower than the rest (~290 s vs ~30 s): the 120 MB dataset is
not yet in the OS page cache. This is expected — do not extrapolate a run's total time from epoch 1.

ConvNeXt inputs are resized to **128×128, not 224**. The sources are 32×32, so 224 is pure upsampling
cost with no added information; 128 preserves the network's stem-and-downsample geometry and peaks at
1.7 GB of VRAM.

Every trained checkpoint is then scored under all eight conditions:

| Condition | What it measures |
|---|---|
| `test` | in-distribution — full CIFAKE test set, 10k/10k |
| `cifake4k` | control for `ddpm`: the *same* 4k real images, paired with CIFAKE fakes instead of DDPM ones, so the cross-generator drop is attributable to the generator and not to the smaller real subset |
| `ddpm` | cross-generator — 4k CIFAKE reals vs. 4k DDPM samples |
| `jpeg75` | pipeline control. CIFAKE is *already* stored at quality 75, so this is very nearly a no-op and should reproduce `test` |
| `jpeg50`, `jpeg30` | robustness to JPEG **recompression** (the sources are already q75, so this is double compression, not native q50/q30) |
| `blur1`, `blur2` | robustness to Gaussian blur, σ = 1 and 2 |

```powershell
python -m src.evaluate --ckpt results\e2_fft_s0\best.pt --condition test
python -m src.evaluate --ckpt results\e2_fft_s0\best.pt --condition ddpm
python -m src.evaluate --ckpt results\e2_fft_s0\best.pt --condition jpeg30
```

`evaluate.py` reconstructs the architecture, input domain, and front-end settings from the checkpoint
itself, so those are never re-specified on the command line and cannot drift out of sync with
training. Each call appends one row to `results/all_evals.csv`.

AUROC is computed with FAKE as the positive class and is **not** flipped when it falls below 0.5. A
sub-chance AUROC under the cross-generator condition is a genuine finding — the detector's ranking is
anti-correlated, not absent — and it is reported that way deliberately.

## Figures and tables

```powershell
python -m src.figures            # all figures + LaTeX tables
python -m src.figures --fusion   # additionally compute the late-fusion baseline (needs the GPU)
```

Writes to `report_assets/`: `fig1_pipeline.pdf`, `mean_spectra.pdf`, `radial_profile.pdf`,
`samples_grid.pdf`, `curves.pdf`, `bars_generalization.pdf`, and `table1_main.tex`,
`table2_robustness.tex`, `table3_ablations.tex`.

**No number in the report is typed by hand** — every table and plot regenerates from the CSVs and JSON
in `results/`. Accuracies come from `all_evals.csv` only; `metrics.json` supplies run labels, never
numbers. Figures whose inputs are not yet on disk are skipped with a printed message rather than
failing, so the script is safe to run against a partially complete queue.

`--fusion` is opt-in because it loads checkpoints onto the GPU: it averages the softmax outputs of
`e1_rgb_s0` and `e2_fft_s0` under every condition and writes `results/late_fusion.csv`. This is the
control for the two-stream model — it shows how much of any fusion gain is available from simply
ensembling two independently trained single-arm detectors.

## Results layout

```
results/
├── all_evals.csv           one row per (checkpoint × condition): run, arch, domain, seed,
│                           condition, acc, f1, auroc, n  — the single source for every table
├── late_fusion.csv         same schema, for the RGB+FFT softmax-averaging baseline (--fusion)
├── spectra.npz             cached mean log-spectra for REAL / FAKE / DDPM; delete to recompute
└── <run_name>/
    ├── best.pt             checkpoint with the highest validation accuracy, plus its training args
    ├── curve.csv           per-epoch train/val loss and accuracy, and wall time
    └── metrics.json        final test acc / F1 / AUROC, runtime, peak VRAM, full hyperparameters
```

`curve.csv` is appended each epoch, so killing a run keeps whatever already finished. `results/` and
`report_assets/` are gitignored; regenerate them with the commands above.

When reporting, take accuracies from `all_evals.csv` rather than `metrics.json`. The two paths differ
by ~2 samples in 20,000 on the same checkpoint, because training enables cuDNN autotuning and
evaluation does not, which changes convolution algorithm selection under fp16 autocast on borderline
samples. The two numbers are not independent measurements.

## Repository layout

```
src/data.py            CIFAKE dataset, hashed val split, FFT transform, streaming stats, corruptions
src/models.py          cifar_resnet18, ConvNeXt-Tiny fine-tune, two-stream fusion, arch registry
src/train.py           training CLI — one generic loop shared by all three architectures
src/evaluate.py        score any checkpoint under any condition into all_evals.csv
src/generate_fakes.py  DDPM cross-generator set (optional deps)
src/figures.py         all figures and LaTeX tables
run_all.ps1            the full experiment queue: every run, then every evaluation
scripts/               one-off sanity checks used during development
NOTES.md               running log of failures and surprises encountered during development
```
