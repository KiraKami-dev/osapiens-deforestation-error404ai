"""GeoTIFF tiles for the osapiens deforestation challenge.

Loads per-tile **slim** feature stacks from ``{root}/features/{tile_id}_slim.tif`` (multi-band float
rasters, e.g. AEF + SAR-derived channels).

Labels: ``{root}/labels/{tile_id}_fused.tif`` (single-band, warped to the feature grid).

**Input scaling** (choose one):

- **z-score** (default): call ``fit_zscore_from_tiles()`` on the **training** tiles only (exclude val when
  you fit on train_ids); then ``set_zscore(mean, std)`` for val/test. Per-band
  ``(x - mean_c) / (std_c + 1e-8)``, then ``nan_to_num`` to 0.
- **min–max**: ``fit_minmax_from_tiles()`` / ``set_minmax`` — each channel to ``[0, 1]`` via min/max.

**Modes**: ``mode="crop"`` (default) yields random crops for training. ``mode="full"`` yields one
**full-tile** sample per tile (validation / inference / preview); use ``batch_size=1`` if tile sizes differ.

**Labels**: integer map ``0`` / ``1`` / ``255`` (ignore). Loss functions should mask pixels with
``label == 255``.

**Augmentation**: with ``augment_flip=True``, **8-way** geometric aug (random rotation ``k*90°`` and
optional horizontal flip), matching eval TTA in ``validation.py`` — applied to random crops and to
full tiles when ``mode="full"``.

Returns float tensors on the first CUDA device if available (match ``--num-workers 0`` for training).
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.warp import Resampling, reproject
from torch.utils.data import Dataset


def _warp_to_ref(src_path: Path, ref_path: Path, resampling: Resampling) -> np.ndarray:
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_h, dst_w = ref.height, ref.width

    with rasterio.open(src_path) as src:
        same_grid = (
            src.height == dst_h
            and src.width == dst_w
            and src.transform == dst_transform
            and src.crs == dst_crs
        )
        if same_grid:
            return np.array(src.read())

        nodata = src.nodata
        out = np.zeros((src.count, dst_h, dst_w), dtype=np.dtype(src.dtypes[0]))
        for b in range(src.count):
            reproject(
                source=rasterio.band(src, b + 1),
                destination=out[b],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                src_nodata=nodata,
                dst_nodata=nodata,
            )
        return out


def _slim_band_names(root: Path, tile_id: str) -> tuple[str | None, ...] | None:
    feat_path = root / "features" / f"{tile_id}_slim.tif"
    if not feat_path.is_file():
        return None
    with rasterio.open(feat_path) as src:
        desc = src.descriptions
        names: list[str | None] = []
        for i in range(src.count):
            names.append(desc[i] if desc and i < len(desc) else None)
        return tuple(names)


def load_tile(root: Path, tile_id: str, device: torch.device):
    """Slim feature stack + fused label mask, label grid-aligned to features."""
    root = Path(root)
    feat_path = root / "features" / f"{tile_id}_slim.tif"
    label_path = root / "labels" / f"{tile_id}_fused.tif"
    if not feat_path.is_file():
        raise FileNotFoundError(f"Missing feature raster: {feat_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label raster: {label_path}")

    with rasterio.open(feat_path) as src:
        arr = np.array(src.read())

    feats = torch.from_numpy(arr).to(device, non_blocking=device.type == "cuda")

    lab = _warp_to_ref(label_path, feat_path, Resampling.nearest)
    if lab.ndim == 2:
        lab = lab[np.newaxis, ...]
    lab_t = torch.from_numpy(lab).to(device, non_blocking=device.type == "cuda")

    timestamps: list[tuple[int, int]] = []
    return feats, timestamps, lab_t


def _fit_minmax_tensors(
    tiles_data: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float]]:
    """Per-channel min/max over all pixels in loaded tiles; returns (C,1,1) tensors and flat lists."""
    if not tiles_data:
        raise RuntimeError("No tiles loaded; cannot fit min/max.")
    c = tiles_data[0][0].shape[0]
    gmin: torch.Tensor | None = None
    gmax: torch.Tensor | None = None
    pinf = torch.tensor(float("inf"), device=device, dtype=torch.float32)
    ninf = torch.tensor(float("-inf"), device=device, dtype=torch.float32)
    for feats, _ in tiles_data:
        flat = feats.float().reshape(c, -1)
        tmin = torch.where(torch.isnan(flat), pinf, flat).min(dim=1).values
        tmax = torch.where(torch.isnan(flat), ninf, flat).max(dim=1).values
        tmin = torch.nan_to_num(tmin, nan=0.0, posinf=0.0, neginf=0.0)
        tmax = torch.nan_to_num(tmax, nan=1.0, posinf=1.0, neginf=1.0)
        if gmin is None:
            gmin, gmax = tmin, tmax
        else:
            gmin = torch.minimum(gmin, tmin)
            gmax = torch.maximum(gmax, tmax)
    eps = torch.tensor(1e-6, device=gmax.device, dtype=gmax.dtype)
    gmax = torch.maximum(gmax, gmin + eps)
    mm_min = gmin.view(c, 1, 1).to(device)
    mm_max = gmax.view(c, 1, 1).to(device)
    return mm_min, mm_max, mm_min.flatten().tolist(), mm_max.flatten().tolist()


def _fit_zscore_tensors(
    tiles_data: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float]]:
    """Per-channel mean/std over all pixels in loaded tiles; std has epsilon for stability."""
    if not tiles_data:
        raise RuntimeError("No tiles loaded; cannot fit z-score stats.")
    c = tiles_data[0][0].shape[0]
    means = torch.zeros(c, device=device, dtype=torch.float32)
    stds = torch.ones(c, device=device, dtype=torch.float32)
    for b in range(c):
        vals = torch.cat([feats[b].float().reshape(-1) for feats, _ in tiles_data])
        vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        means[b] = vals.mean()
        stds[b] = vals.std() + 1e-8
    zmean = means.view(c, 1, 1)
    zstd = stds.view(c, 1, 1)
    return zmean, zstd, means.flatten().tolist(), stds.flatten().tolist()


def _apply_input_norm(
    features: torch.Tensor,
    *,
    mm_min: torch.Tensor | None,
    mm_max: torch.Tensor | None,
    zs_mean: torch.Tensor | None,
    zs_std: torch.Tensor | None,
    input_norm: str,
) -> torch.Tensor:
    if input_norm == "zscore":
        if zs_mean is None or zs_std is None:
            raise RuntimeError("z-score norm requested but set_zscore / fit_zscore_from_tiles not called.")
        out = (features - zs_mean) / zs_std
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if input_norm == "minmax":
        if mm_min is None or mm_max is None:
            raise RuntimeError("min–max norm requested but set_minmax / fit_minmax_from_tiles not called.")
        denom = mm_max - mm_min
        out = (features - mm_min) / denom
        out = out.clamp(0.0, 1.0)
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    raise ValueError(f"Unknown input_norm: {input_norm}")


def _crop_tile_item(
    index: int,
    tiles_data: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    crops_per_tile: int,
    crop_size: int,
    augment_flip: bool,
    mm_min: torch.Tensor | None,
    mm_max: torch.Tensor | None,
    zs_mean: torch.Tensor | None,
    zs_std: torch.Tensor | None,
    input_norm: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random crop, z-score or min–max norm, label map 0/1/255, optional 8-way geo aug."""
    tile_idx = index // crops_per_tile
    feats, lab = tiles_data[tile_idx]

    h, w = feats.shape[-2:]
    cs = crop_size
    if h < cs or w < cs:
        pad_w = max(0, cs - w)
        pad_h = max(0, cs - h)
        pad_spec = (0, pad_w, 0, pad_h)
        feats = F.pad(feats, pad_spec)
        lab = F.pad(lab, pad_spec)
        h, w = feats.shape[-2:]
    y0 = random.randint(0, h - cs)
    x0 = random.randint(0, w - cs)

    features = feats[:, y0 : y0 + cs, x0 : x0 + cs].float()
    features = _apply_input_norm(
        features,
        mm_min=mm_min,
        mm_max=mm_max,
        zs_mean=zs_mean,
        zs_std=zs_std,
        input_norm=input_norm,
    )
    lab_crop = lab[:, y0 : y0 + cs, x0 : x0 + cs]
    lab_hw = lab_crop.squeeze(0).long()
    labels = lab_hw
    if augment_flip:
        features, labels = _apply_geo_aug8(features, labels)
    return features, labels


def _apply_geo_aug8(features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same 8 transforms as TTA in validation.py: rot90^k and rot90^k ∘ flip_W (k=0..3)."""
    aug_idx = random.randint(0, 7)
    k = aug_idx % 4
    flip_w = aug_idx // 4
    if k:
        features = torch.rot90(features, k, dims=(-2, -1))
        labels = torch.rot90(labels, k, dims=(-2, -1))
    if flip_w:
        features = features.flip(-1)
        labels = labels.flip(-1)
    return features, labels


class OsapiensTerraDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        geojson_path: str | Path,
        *,
        crop_size: int = 256,
        crops_per_tile: int = 256,
        max_tiles: int | None = None,
        tile_ids: Sequence[str] | None = None,
        augment_flip: bool = True,
        mode: str = "crop",
        input_norm: str = "zscore",
    ) -> None:
        self.root = Path(root)
        self.crop_size = crop_size
        self.crops_per_tile = crops_per_tile
        self.augment_flip = augment_flip
        self.mode = mode
        if self.mode not in ("crop", "full"):
            raise ValueError('mode must be "crop" or "full"')
        self.input_norm = input_norm
        if self.input_norm not in ("zscore", "minmax"):
            raise ValueError('input_norm must be "zscore" or "minmax"')
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._mm_min: torch.Tensor | None = None
        self._mm_max: torch.Tensor | None = None
        self._zs_mean: torch.Tensor | None = None
        self._zs_std: torch.Tensor | None = None

        with open(geojson_path, encoding="utf-8") as f:
            data = json.load(f)
        self.tiles = [f["properties"]["name"] for f in data["features"]]
        if tile_ids is not None:
            allow = set(tile_ids)
            self.tiles = [t for t in self.tiles if t in allow]
        self.tiles = self.tiles[:max_tiles] if max_tiles is not None else self.tiles

        self.feature_band_descriptions: tuple[str | None, ...] | None = None
        if self.tiles:
            self.feature_band_descriptions = _slim_band_names(self.root, self.tiles[0])

        self._tiles_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.tile_timestamps: list[list[tuple[int, int]]] = []
        self._n_channels: int | None = None
        for tile_id in self.tiles:
            feats, timestamps, lab = load_tile(self.root, tile_id, self.device)
            if self._n_channels is None:
                self._n_channels = feats.shape[0]
            elif feats.shape[0] != self._n_channels:
                raise ValueError(
                    f"Channel mismatch for tile {tile_id}: expected {self._n_channels} bands, got {feats.shape[0]}"
                )
            self._tiles_data.append((feats, lab))
            self.tile_timestamps.append(timestamps)

    def fit_minmax_from_tiles(self) -> tuple[list[float], list[float]]:
        """Set ``_mm_min`` / ``_mm_max`` from per-channel min and max over all pixels in loaded tiles."""
        self._mm_min, self._mm_max, mn, mx = _fit_minmax_tensors(self._tiles_data, self.device)
        return mn, mx

    def fit_zscore_from_tiles(self) -> tuple[list[float], list[float]]:
        """Set ``_zs_mean`` / ``_zs_std`` from per-channel mean and std over all pixels in loaded tiles."""
        self._zs_mean, self._zs_std, m, s = _fit_zscore_tensors(self._tiles_data, self.device)
        return m, s

    def set_minmax(self, mm_min: torch.Tensor, mm_max: torch.Tensor) -> None:
        """Use the same normalization as training (e.g. val/test). Tensors are (C,), (C,1,1), or broadcastable."""
        c = self._tiles_data[0][0].shape[0] if self._tiles_data else mm_min.numel()
        self._mm_min = mm_min.to(device=self.device, dtype=torch.float32).view(c, 1, 1)
        self._mm_max = mm_max.to(device=self.device, dtype=torch.float32).view(c, 1, 1)
        if torch.any(self._mm_max <= self._mm_min):
            raise ValueError("set_minmax: each max must be > min for every channel")

    def set_zscore(self, zs_mean: torch.Tensor, zs_std: torch.Tensor) -> None:
        """Apply training z-score stats to val/test (same shape as ``fit_zscore_from_tiles``)."""
        c = self._tiles_data[0][0].shape[0] if self._tiles_data else zs_mean.numel()
        self._zs_mean = zs_mean.to(device=self.device, dtype=torch.float32).view(c, 1, 1)
        self._zs_std = zs_std.to(device=self.device, dtype=torch.float32).view(c, 1, 1)

    def estimate_bce_pos_weight(
        self,
        *,
        min_weight: float = 1.0,
        max_weight: float = 100.0,
    ) -> float:
        """``#valid background / #valid foreground`` for ``BCEWithLogitsLoss`` (PyTorch imbalance recipe).

        Counts only pixels with label ``0`` or ``1`` among ``label < 255`` (ignore nodata).
        """
        n_pos = 0
        n_neg = 0
        for _, lab in self._tiles_data:
            y = lab.squeeze(0)
            valid = y < 255
            n_pos += int(((y == 1) & valid).sum().item())
            n_neg += int(((y == 0) & valid).sum().item())
        if n_pos < 1:
            return float(max_weight)
        w = n_neg / float(n_pos)
        return float(max(min_weight, min(max_weight, w)))

    def __len__(self) -> int:
        if self.mode == "full":
            return len(self.tiles)
        return len(self.tiles) * self.crops_per_tile

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "full":
            feats, lab = self._tiles_data[index]
            features = feats.float()
            features = _apply_input_norm(
                features,
                mm_min=self._mm_min,
                mm_max=self._mm_max,
                zs_mean=self._zs_mean,
                zs_std=self._zs_std,
                input_norm=self.input_norm,
            )
            y = lab.squeeze(0).long()
            if self.augment_flip:
                features, y = _apply_geo_aug8(features, y)
            return features, y
        return _crop_tile_item(
            index,
            self._tiles_data,
            crops_per_tile=self.crops_per_tile,
            crop_size=self.crop_size,
            augment_flip=self.augment_flip,
            mm_min=self._mm_min,
            mm_max=self._mm_max,
            zs_mean=self._zs_mean,
            zs_std=self._zs_std,
            input_norm=self.input_norm,
        )
