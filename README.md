# E. coli Detection Kit — Computer Vision Demo

> **HONESTY DISCLAIMER — read first.**
> **EDUCATIONAL DEMONSTRATION ONLY — not a medical device, not for diagnostic
> use, not the actual CISNR system or its data.**
> All plate images in this repository are **procedurally generated synthetic
> images** (drawn blobs, noise, and artifacts — no photographs, no biological
> samples). Every metric measures the demo pipeline on that synthetic data
> **only** and is **not** a claim about real-world detection accuracy,
> sensitivity, specificity, or field performance.

## What is real vs what is demo

- **Real:** Bilal Ahmad worked 14 months (Dec 2022 – Feb 2024) as a Junior
  Research Engineer at CISNR (Center for Intelligence Systems and Network
  Research, UET Peshawar) on an E. coli rapid-detection kit — an IoT
  data-acquisition layer plus a cloud-hosted AI platform with computer
  vision, integrated into a field-testing prototype.
- **Demo (everything in this repo):** a from-scratch educational
  re-implementation of the *detection/vision* side of such a kit, using
  classical computer vision (OpenCV, no machine learning) on synthetic
  images. It demonstrates the engineering approach, not the actual system.

Companion project: the [sensor-to-cloud pipeline
simulator](https://github.com/bilal4web/iot-sensor-pipeline) (project #2)
covers the data-transport side; this repo covers the vision side.

## Architecture

```
 synthetic/plates.py        vision/detect.py            vision/evaluate.py
 +----------------+        +------------------+        +------------------+
 | PlateSpec      |        | preprocess       |        | match detections |
 |  seeded RNG   +------->|  denoise+CLAHE   +------->|  to ground truth |
 |  blobs+texture |  img   | adaptive thresh  |  dets  | (greedy nearest) |
 |  dust/scratch |        | morphology       |        | TP/FP/FN,        |
 |  ground truth |        | watershed split  |        | count error      |
 +----------------+        | contour filter   |        +------------------+
                           +------------------+
                                     |
                                     v
                           analysis/plots.py  ->  results/*.png
                           run_demo.py        ->  results/dashboard.html
```

**Pipeline stages (classical CV, no ML):**
1. **Capture (synthetic):** seeded generator draws background texture,
   vignette, colony-like blobs (known positions = ground truth), plus
   dust/scratch/glare artifacts the detector must reject.
2. **Preprocess:** Gaussian denoise + CLAHE normalization (handles
   vignette/glare brightness drift).
3. **Detect:** adaptive Gaussian threshold → morphological opening →
   distance-transform watershed (splits touching blobs) → contour
   analysis with area + circularity filters.
4. **Count + evaluate:** greedy matching of detections to ground truth;
   reports count error, precision/recall *on synthetic data*.

**Known limitation (documented, not hidden):** heavily overlapping
colonies can merge into one detection, so counts underestimate at very
high densities. The accuracy-vs-density chart shows this honestly.

## How to run

```bash
pip install -r requirements.txt
python3 run_demo.py
```

Options:

```bash
python3 run_demo.py --densities 0 20 60 100 --plates-per-density 3 \
    --seed 7 --size 512
```

Outputs in `results/`: per-plate PNGs, detection overlays, charts,
`summary.json`, and `dashboard.html` (open it in a browser).

Runtime is well under a minute on a modern laptop.

## Sample results (synthetic demo, seed 7)

Typical run: 20 plates across densities 0–120 colonies. Mean absolute
count error ≈ 1–3 colonies, mean processing ≈ tens of ms per 512×512
image. Error grows with density (touching colonies merge) — see
`accuracy_vs_density.png`. **These numbers describe the demo on synthetic
data only.**

## Troubleshooting

- **`ModuleNotFoundError: No module named 'cv2'`** — install requirements:
  `pip install -r requirements.txt`. On some systems you may need
  `pip install opencv-python-headless` instead if GUI libraries are
  missing (the demo never opens GUI windows).
- **`numpy` / `matplotlib` import errors** — this repo pins `numpy<2`
  because `matplotlib<3.7` is incompatible with NumPy 2.x. If your
  environment already has NumPy 2.x, either downgrade NumPy
  (`pip install "numpy<2"`) or upgrade matplotlib
  (`pip install -U matplotlib`).
- **Plots look empty / all-black** — check that `results/*.png` were
  regenerated after your change; stale files from an interrupted run can
  linger. Delete `results/` and re-run.
- **Dashboard images broken** — open `results/dashboard.html` (not a copy
  moved elsewhere); it references sibling PNGs by relative path.
- **"Detected 0 colonies" on every plate** — the top-hat `tophat_thresh`
  (default 25) may be too high if you changed the generator's colony
  intensity range; retune `DetectorConfig` in `vision/detect.py`.
- **Poor accuracy on tiny images (<256px)** — the pipeline is tuned for
  ~512×512 plates. On much smaller images the fixed morphology kernels
  are disproportionately large; use `--size 512` (default) for best
  results.
- **High false positives on real photos** — expected: the detector is
  tuned for the synthetic generator's statistics, not real imagery. This
  demo is not intended for real plate photographs.

## Project structure

```
ecoli-detection-demo/
├── synthetic/plates.py    # seeded synthetic plate generator + ground truth
├── vision/detect.py        # classical CV detection pipeline (OpenCV)
├── vision/evaluate.py      # detection-vs-truth matching & metrics
├── analysis/plots.py       # matplotlib charts (watermarked synthetic)
├── run_demo.py             # one-command demo + dashboard generation
├── results/                # generated outputs (not committed upstream)
├── requirements.txt
└── README.md
```
