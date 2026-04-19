# osapiens Makeathon Challenge

This repository holds the training, evaluation, ensembling, and submission export code for the osapiens Makeathon challenge. Upstream notebook-based preprocessing that builds slim GeoTIFFs is **not** included here; this tree expects ready-made `*_slim.tif` features and fused labels.

**Layout:** Top-level entrypoints are `train.py`, `validation.py`, `ensemble.py`, and `submission.py`. Shared code lives under **`models/`** (e.g. `UNet`) and **`utils/`** (dataset loading, metrics, GeoJSON helpers). Run scripts from the repo root so imports resolve.

---

## 1. Pipeline Overview

1. **Preprocessing (external):** Produce per-tile slim feature stacks and fused label rasters on disk (not shipped in this repo).
2. **Training (`train.py`):** K-fold cross-validation with random crops, BCE + Dice loss, Adam, and `ReduceLROnPlateau` on `val_union_iou`. Checkpoints per fold.
3. **Evaluation (`validation.py`):** Per-fold IoU with optional 8-way TTA and a foreground probability threshold grid.
4. **Ensembling (`ensemble.py`):** Combine multiple runs that share the same fold splits; optional logit averaging or confidence-weighted probabilities.
5. **Inference (`submission.py`):** Full-tile inference on test (or listed) tiles; optional TTA and threshold tuning; GeoJSON output.

---

## 2. Data Layout and GeoJSON

Processed data is one multi-band float raster per tile under `features/`, plus a single-band uint8 mask under `labels/` (0 = background, 1 = deforestation, 255 = ignore).

**Training expects:**

| Path | Content |
| :--- | :--- |
| `features/{tile_id}_slim.tif` | Float **C×H×W** stack (channel count is read from the data). |
| `labels/{tile_id}_fused.tif` | Single-band **uint8** mask. |
| `train_tiles.geojson` or `metadata/train_tiles.geojson` | Tile list: each feature’s `properties.name` is the `tile_id`. |

Labels are warped to the feature grid in `utils.dataset.load_tile` when CRS or transform differ from the feature raster.

**Inference** uses `test/{tile_id}_slim.tif` (or `features/` as fallback) and optionally `metadata/test_tiles.geojson`; training still requires both features and labels for every tile in the train GeoJSON.

Illustrative channel families (actual band count and order follow your slim export): AEF-vs-baseline deltas, S1 summary change, S1 temporal metrics, etc.

---

## 3. Model (`models/unet.py`)

* **Architecture:** U-Net with skip connections; decoder blocks concatenate upsampled features with encoder skips.
* **Blocks:** Two convolutions per block: Conv2d → BatchNorm2d → ReLU (see `_DoubleConv`).
* **Head:** 1×1 convolution to `num_classes` channels (binary segmentation uses `num_classes=1` logits).
* **Padding:** `train.SegLitModule` pads height/width to a multiple of `pad_multiple`. CLI `--pad-multiple` defaults to **0**, which means **16** in code; replicate padding, then logits are cropped back to the original size.

---

## 4. Loss, Optimizer, Metrics

### Loss (`train.py` — `BCEDiceLoss`)

* **BCE with logits:** Valid pixels only (`label < 255`). Positive class weight is **`estimate_bce_pos_weight()`** on the fold’s training tiles (ratio of valid negative to valid positive pixels, clamped to **[1, 100]**).
* **Soft Dice:** Same valid mask: ignored pixels are masked out of the Dice numerator/denominator via zeroed predictions/targets there.
* **Combined:** `(1 - λ) * BCE + λ * Dice` with `λ = --dice-weight` (default 0.5).

### Optimizer

* **Adam**, learning rate `--lr` (default 1e-3).
* **`ReduceLROnPlateau`** on **`val_union_iou`** (max mode): `--lr-plateau-patience`, `--lr-plateau-factor`, `--lr-plateau-min`.

### Metrics (`utils/metrics.py`)

* **`mean_union_iou_batch_ignore`:** Union IoU on class **1** vs ground truth, excluding label **255**.

---

## 5. Training (`train.py`)

* **CV:** `--cv-folds` (default 5), `KFold` with `--seed`. Requires at least `K` tiles.
* **Normalization:** `--input-norm zscore` (default) or `minmax`, **fit on training tiles only** for each fold.
* **Training:** Random `--crop-size` crops, `--crops-per-tile` per epoch, **8-way** augmentation (same family as eval TTA: 90° rotations × optional horizontal flip).
* **Validation:** Full tile, **batch size 1**.
* **Early stopping:** On **`val_union_iou`** with **`--early-stop-patience`** (default 30).
* **Outputs:** Checkpoints under `{--out-dir}/checkpoints/version_N/fold{k}/best-*.ckpt`. TensorBoard logs under `{--out-dir}/tensorboard/fold*/`.

### Minimal training command

```bash
python train.py \
  --data-root /path/to/data \
  --geojson /path/to/data/train_tiles.geojson \
  --out-dir /path/to/experiments
```

---

## 6. Evaluation (`validation.py`)

Loads the best `best-*.ckpt` per `fold*` (or a single `train/` run), refits normalization on that fold’s **train** tiles, evaluates **val** tiles.

* **No TTA:** Forward pass; binary threshold **0.5** on sigmoid.
* **8-way TTA:** Mean logits over the same 8 transforms as training, then **0.5** on probability.
* **TTA + threshold search:** Grid of foreground probabilities from **`--thresh-min`** .. **`--thresh-max`** with **`--thresh-steps`** points; pick the threshold with best mean val IoU per fold.

---

## 7. Ensembling (`ensemble.py`)

For each validation fold, loads `best-*.ckpt` from each supplied run directory (same fold indices).

* **mean:** Equal average of **logits**, then sigmoid/softmax as in a single model.
* **weighted:** Normalized **`--weights`** on logits before averaging.
* **confidence:** Per-pixel weights ∝ **`max(p, 1−p)`** on each model’s foreground probability; weighted average of **probabilities** (not logits).

Evaluation mirrors `validation.py` (no TTA / TTA@0.5 / TTA + threshold grid).

---

## 8. Inference and submission (`submission.py`)

Full-tile inference on test (or GeoJSON-listed) tiles.

* **Checkpoints:** Single **`--checkpoint`**, or **`--ensemble-checkpoints-dir`** with `fold0`, `fold1`, …
* **Ensemble modes (export only):** **`mean`** (average logits) or **`confidence`** (per-pixel confidence blend — same idea as `ensemble.py`). There is **no** `weighted` logit mode in this script.
* **Defaults:** **`--tta`** on (8-way), **`--tune-threshold`** on (grid search on a train/val split consistent with the checkpoint layout); optional **`--no-tune-threshold`** and **`--prob-threshold`**.
* **Vectorization:** Polygons in **EPSG:4326**; **`--min-area-ha`** drops small polygons (via `utils/submission_utils.py`).

---

## 9. Utilities

| Script | Purpose |
| :--- | :--- |
| `preview.py` | Saves **`preview/samples.png`**: false-color first three slim bands (percentile stretch) and fused labels for quick checks. |
| `utils/submission_utils.py` | Raster → GeoJSON helper with minimum area filtering. |
