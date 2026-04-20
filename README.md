# osapiens Makeathon Challenge

This repository contains a **web frontend** and a **research** codebase for training, validating, ensembling, and exporting submissions for the osapiens Makeathon challenge.

---

## Repository layout

| Path | What it is |
| :--- | :--- |
| **`frontend/`** | Web app (see `frontend/README.md` for dev setup). |
| **`research/`** | Python ML pipeline: scripts, `models/`, `utils/`, and `notebooks/` for data prep. |

**Run Python scripts from `research/`** so imports like `models` and `utils` resolve:

```bash
cd research
python train.py --help
```

---
## 📸 Screenshots

<p align="center">
  <img src="https://github.com/user-attachments/assets/a63805f5-c3a6-40e4-b383-858bd7215a57" width="48%" />
  <img src="https://github.com/user-attachments/assets/e5e18d8d-c2f8-4994-ad53-29e102630d0f" width="48%" />
</p>

---

## 1. Pipeline overview (step by step)

Each stage feeds the next. Do them in order when running the full pipeline.

1. **Preprocessing (`research/notebooks/`):** Notebooks build per-tile **fused** multi-band GeoTIFFs and **fused** labels. Typical flow: explore or compress inputs → cloud masking → Sentinel‑1 / temporal / fusion steps → a slim stack plus labels ready for training.

2. **Training (`research/train.py`):** K-fold cross-validation on the prepared tiles. Uses random crops of 256 x 256, BCE + Dice loss, Adam, and `ReduceLROnPlateau`. 

3. **Evaluation (`research/validation.py`):** Loads best checkpoints, refits normalization on each fold’s **train** tiles, scores **val** tiles. Run test-time augmentation (TTA) and a threshold optimization.

4. **Ensembling (`research/ensemble.py`):** Merges several trained runs that share the same fold splits based on model confidence. 

5. **Inference / submission (`research/submission.py`):** Full-tile prediction on test (or GeoJSON-listed) tiles with TTA and threshold tuning. Generates the GeoJSON for upload.

---

## 2. Data cleanup 
### 2.1 Feature Engineering 
This is the core compression step. Instead of keeping raw multi-temporal imagery, it computes temporal change statistics:

AEF features (394 bands):

Loads 2020 as baseline (64 bands)
For each year 2021–2025: computes delta = AEF_year − AEF_2020 (5 × 64 = 320 bands)
Delta L2 norm per year (5 bands) — magnitude of embedding change
Cosine distance per year (5 bands) — directional change: 

We also did PCA analysis, which gave 15 channels explaining 95 percent of the data, but the Fisher discriminant ratio of that came out really low, and since the goal was to maximize separation, not variance, we prioritized that.
​
### 2.2 Label Cleaning
Step 1: Decode — read raw files, throw away old alerts
RADD: One file per tile. Values encode CDDDD (confidence digit + days since 2014-12-31). Script splits that into a confidence (2 or 3) and a date. If the date is before Jan 1, 2021 → set to 0. Otherwise keep the confidence value (2 or 3).
GLAD-L: Five files per tile (one per year 2021–2025), each with values 0/2/3. Also five companion date files with day-of-year. Script checks if each alert's date is ≥ Jan 1, 2021 (which it almost always is since the files are yearly). Merges all 5 years into one layer by taking the max confidence per pixel across years. Output: 0/2/3.
GLAD-S2: One alert file (0–4) + one date file (days since 2019-01-01). Same date cutoff logic. Output: 0/1/2/3/4.
No confidence thresholding. All confidence levels are kept.

Step 2: Outlier filter — remove lonely alert pixels (at native resolution)
For each alert pixel, look at a 7×7 window around it. Count what fraction of those 49 neighbours are also alerts. If fewer than 10% are alerts (i.e., <5 neighbours), zero it out. This kills isolated speckle — random single pixels that no nearby pixel agrees with.
Runs at ~28m for GLAD-L, ~10m for RADD and GLAD-S2.

Step 3: Morphological opening — erode then dilate (at native resolution)
Binary erosion shrinks all alert patches by 1 pixel on every edge. Then dilation grows them back. Net effect: any alert patch that was only 1 pixel wide gets deleted, but larger patches survive with smoothed boundaries. Confidence values are restored to surviving pixels.
Again at native resolution per source.

Step 4: Resample GLAD-L ~28m → ~10m (only after cleaning)
Nearest-neighbour resampling to match the RADD/GLAD-S2 10m grid. Each cleaned 28m GLAD-L pixel becomes ~6.25 10m pixels with the same confidence value. This is purely geometric alignment — no data transformation.

### 2.3 Label Fusion

A primary technical challenge of this project was resolving conflicting supervision from three independent, noisy alert systems. Our pipeline transforms these "weak labels" into a high-fidelity binary supervisory signal for UNet training.

We implemented a Radar-First Override strategy. To ensure EUDR compliance under persistent cloud cover, the system treats Sentinel-1 (RADD) as the primary ground truth. In the absence of a radar signal, an alert is only triggered if both optical sources agree.

<p align="center">
<img width="631" height="273" alt="Screenshot 2026-04-18 at 16 28 30" src="https://github.com/user-attachments/assets/0714430b-af62-4006-80da-9a55b19fa91b" />
<img width="646" height="423" alt="Screenshot 2026-04-18 at 16 26 28" src="https://github.com/user-attachments/assets/46ed8382-13a8-4bd7-886d-99f50cc4137d" />
</p>

S1 features (6 bands):

Splits monthly dB timeseries into before (2020–2021) and after (2023–2025)
Computes: before_mean, before_std, after_mean, after_std, change_mean, change_ratio
Reprojects S1 grid to AEF grid (bilinear)
Output: 400-band GeoTIFF per tile — still large but all temporal info is now compressed into statistics.

## 2. Data layout and GeoJSON

Point `--data-root` at a directory that mirrors what the notebooks produced.

Processed data is one multi-band float raster per tile under `features/`, plus a single-band uint8 mask under `labels/` (0 = background, 1 = deforestation, 255 = ignore).

**Training expects:**

| Path | Content |
| :--- | :--- |
| `features/{tile_id}_slim.tif` | Float **C×H×W** stack (channel count comes from the file). |
| `labels/{tile_id}_fused.tif` | Single-band **uint8** mask. |
| `train_tiles.geojson` or `metadata/train_tiles.geojson` | Tile list: each feature’s `properties.name` is the `tile_id`. |

Labels are aligned to the feature grid in `research/utils/dataset.py` when CRS or transform differ.

**Inference** uses `test/{tile_id}_slim.tif` (or `features/` as fallback) and optionally `metadata/test_tiles.geojson`.

---

## 3. Model (`research/models/unet.py`)

* **Architecture:** U-Net with skip connections; decoder blocks concatenate upsampled features with encoder skips.
* **Blocks:** Two convolutions per block: Conv2d → BatchNorm2d → ReLU (`_DoubleConv`).
* **Head:** 1×1 convolution to `num_classes` channels (binary segmentation uses `num_classes=1` logits).
* **Padding:** `SegLitModule` pads height/width to a multiple of `pad_multiple`. CLI `--pad-multiple` defaults to **0**, which means **16** in code; replicate padding, then logits are cropped back to the original size.

<img width="1555" height="1036" alt="image" src="https://github.com/user-attachments/assets/16676e30-c26d-4162-8de7-60639c3e1719" />

---

## 4. Loss, optimizer, metrics

### Loss (`train.py` — `BCEDiceLoss`)

* **BCE with logits:** Valid pixels only (`label < 255`). Positive class weight from **`estimate_bce_pos_weight()`** on the fold’s training tiles (ratio of valid negative to valid positive pixels, clamped to **[1, 100]**).
* **Soft Dice:** Same valid mask; ignored pixels are excluded via masking.
* **Combined:** `(1 - λ) * BCE + λ * Dice` with `λ = --dice-weight` (default 0.5).

### Optimizer

* **Adam**, learning rate `--lr` (default 1e-3).
* **`ReduceLROnPlateau`** on **`val_union_iou`** (max mode): `--lr-plateau-patience`, `--lr-plateau-factor`, `--lr-plateau-min`.

### Metrics (`research/utils/metrics.py`)

* Union IoU on pixels.

---

## 5. Training (`research/train.py`)

* **CV:** K-Fold Cross Validation. We used K = 4.
* **Normalization:** We normalize the data based on training tiles.
* **Training:** We use Random `--crop-size` crops, `--crops-per-tile` per epoch, horizontal flip, vertical flip, and rotation as augmentation.
* **Validation:** Run trained model on full tile.
* **Early stopping**.
* **Outputs:** We save the last abd best checkpoints, as well as training log in Tensorboard.

<img width="850" height="481" alt="image" src="https://github.com/user-attachments/assets/3307683a-04d9-42a5-b8c1-7c98345c8fc2" />

### Minimal training command

```bash
cd research
python train.py \
  --data-root /path/to/data \
  --geojson /path/to/data/train_tiles.geojson \
  --out-dir /path/to/experiments
```

---

## 6. Evaluation (`research/validation.py`)

Loads `best-*.ckpt` per fold (or a single run layout), refits normalization on that fold’s **train** tiles, evaluates **val** tiles.

* **No TTA:** One forward pass; threshold **0.5** on sigmoid.
* **8-way TTA:** Mean logits over eight transforms, then **0.5** on probability.
* **TTA + threshold search:** Grid from **`--thresh-min`** .. **`--thresh-max`** with **`--thresh-steps`**; pick the threshold with best mean val IoU per fold.

---

## 7. Ensembling (`research/ensemble.py`)

For each validation fold, loads `best-*.ckpt` from each run directory (same fold indices).

* **mean:** Equal average of **logits**, then sigmoid as for a single model.
* **weighted:** Normalized **`--weights`** on logits before averaging.
* **confidence:** Per-pixel weights ∝ **`max(p, 1−p)`** on each model’s probability; weighted average of **probabilities** (not logits).

Evaluation options mirror `validation.py` (no TTA / TTA@0.5 / TTA + threshold tuning).

---

## 8. Inference and submission (`research/submission.py`)

Full-tile inference on test or listed tiles.

* **Checkpoints:** **`--checkpoint`** for one file, or **`--ensemble-checkpoints-dir`** with `fold0`, `fold1`, …
* **Ensemble modes (export):** **`mean`** (average logits) or **`confidence`** (per-pixel confidence blend). There is **no** `weighted` logit mode in this script.
* **Defaults:** **`--tta`** on (8-way), **`--tune-threshold`** on; use **`--no-tune-threshold`** and **`--prob-threshold`** to fix a threshold.
* **Vectorization:** Polygons in **EPSG:4326**; **`--min-area-ha`** filters small polygons (`research/utils/submission_utils.py`).

## Future improvements would include procuring more data (2015 - 2020) and running it through the same cleaning pipeline to produce more high quality data.
---

## 9. Utilities

| Script | Purpose |
| :--- | :--- |
| `research/preview.py` | Writes **`preview/samples.png`**: false-color first three slim bands (percentile stretch) and fused labels. |
| `research/utils/submission_utils.py` | Raster → GeoJSON helper with minimum area filtering. |
