#!/usr/bin/env python3
from __future__ import annotations

import random
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from utils.dataset import OsapiensTerraDataset

_REPO = Path(__file__).resolve().parent
DATA_ROOT = _REPO / "data"
GEOJSON = DATA_ROOT / "train_tiles.geojson"

N_SAMPLES = 5


def rgb_from_chw(x: np.ndarray, r: int, g: int, b: int) -> np.ndarray:
    rgb = np.stack([x[r], x[g], x[b]], axis=-1).astype(np.float64)
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], (2, 98))
        hi = max(hi, lo + 1.0)
        out[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo), 0.0, 1.0)
    return out


random.seed(42)
ds = OsapiensTerraDataset(
    DATA_ROOT,
    GEOJSON,
    crops_per_tile=N_SAMPLES,
    max_tiles=1,
    augment_flip=False,
    input_norm="zscore",
)
ds.fit_zscore_from_tiles()

# False-color preview: first three slim feature bands (percentile-stretched).
R_IDX, G_IDX, B_IDX = 0, 1, 2
viz_title = "Slim features (bands 0–2, percentile stretch)"

fig, axes = plt.subplots(
    2,
    N_SAMPLES,
    figsize=(3.2 * N_SAMPLES, 7.0),
    layout="constrained",
)

for i in range(N_SAMPLES):
    features, labels = ds[i]
    chw = features.detach().cpu().numpy()
    rgb = rgb_from_chw(chw, R_IDX, G_IDX, B_IDX)

    radd = labels.detach().cpu().numpy()

    tile_id = ds.tiles[i // ds.crops_per_tile]
    ax_rgb = axes[0, i]
    ax_radd = axes[1, i]
    ax_rgb.imshow(rgb)
    ax_rgb.set_title(f"{i}  {tile_id}")
    ax_rgb.axis("off")

    im = ax_radd.imshow(radd, cmap="magma", interpolation="nearest")
    ax_radd.axis("off")

fig.colorbar(im, ax=axes[1, :], shrink=0.75)
fig.suptitle(f"{viz_title}  ·  fused label below (same crop, aligned)")

os.makedirs("./preview", exist_ok=True)
out_path = "./preview/samples.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(out_path)
