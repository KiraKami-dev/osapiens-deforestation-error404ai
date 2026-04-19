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

## 1. Pipeline overview (step by step)

Each stage feeds the next. Do them in order when building a full run from raw inputs.

1. **Preprocessing (`research/notebooks/`):** Notebooks build per-tile **slim** multi-band GeoTIFFs and **fused** label rasters on disk. Typical flow: explore or compress inputs → cloud masking → Sentinel‑1 / temporal / fusion steps → a slim stack plus labels ready for training.

2. **Training (`research/train.py`):** K-fold cross-validation on the prepared tiles. Uses random crops, BCE + Dice loss, Adam, and `ReduceLROnPlateau` on `val_union_iou`. Saves one checkpoint folder per fold.

3. **Evaluation (`research/validation.py`):** Loads best checkpoints, refits normalization on each fold’s **train** tiles, scores **val** tiles. Optional 8-way test-time augmentation (TTA) and a foreground probability threshold grid.

4. **Ensembling (`research/ensemble.py`):** Merges several trained runs that share the same fold splits. You can average logits, use confidence-weighted probabilities, or supply manual weights.

5. **Inference / submission (`research/submission.py`):** Full-tile prediction on test (or GeoJSON-listed) tiles; optional TTA and threshold tuning; writes GeoJSON for upload.

---

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

* **`mean_union_iou_batch_ignore`:** Union IoU on class **1** vs ground truth, excluding label **255**.

---

## 5. Training (`research/train.py`)

* **CV:** `--cv-folds` (default 5), `KFold` with `--seed`. Needs at least `K` tiles.
* **Normalization:** `--input-norm zscore` (default) or `minmax`, fit on **training tiles only** per fold.
* **Training:** Random `--crop-size` crops, `--crops-per-tile` per epoch, **8-way** augmentation (same family as eval TTA).
* **Validation:** Full tile, batch size 1.
* **Early stopping:** On **`val_union_iou`** with **`--early-stop-patience`** (default 30).
* **Outputs:** Checkpoints under `{--out-dir}/checkpoints/version_N/fold{k}/best-*.ckpt`. TensorBoard under `{--out-dir}/tensorboard/fold*/`.

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

Evaluation options mirror `validation.py` (no TTA / TTA@0.5 / TTA + threshold grid).

---

## 8. Inference and submission (`research/submission.py`)

Full-tile inference on test or listed tiles.

* **Checkpoints:** **`--checkpoint`** for one file, or **`--ensemble-checkpoints-dir`** with `fold0`, `fold1`, …
* **Ensemble modes (export):** **`mean`** (average logits) or **`confidence`** (per-pixel confidence blend). There is **no** `weighted` logit mode in this script.
* **Defaults:** **`--tta`** on (8-way), **`--tune-threshold`** on; use **`--no-tune-threshold`** and **`--prob-threshold`** to fix a threshold.
* **Vectorization:** Polygons in **EPSG:4326**; **`--min-area-ha`** filters small polygons (`research/utils/submission_utils.py`).

---

## 9. Utilities

| Script | Purpose |
| :--- | :--- |
| `research/preview.py` | Writes **`preview/samples.png`**: false-color first three slim bands (percentile stretch) and fused labels. |
| `research/utils/submission_utils.py` | Raster → GeoJSON helper with minimum area filtering. |
