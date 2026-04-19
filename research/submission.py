#!/usr/bin/env python3
"""Run full-tile inference with TTA + val threshold tuning (like ``validation.py``), then GeoJSON."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

from utils.dataset import OsapiensTerraDataset
from utils.metrics import mean_union_iou_batch_ignore
from utils.submission_utils import raster_to_geojson
from train import SegLitModule, load_seg_lit_from_checkpoint
from validation import (
    _resolve_geojson,
    discover_fold_dirs,
    find_best_checkpoint,
    tta_d8_logits,
)

_log = logging.getLogger(__name__)


def _confidence_combine_binary_prob_stack(prob_stack: torch.Tensor) -> torch.Tensor:
    """``prob_stack`` shape ``(M, B, H, W)`` per-model sigmoid P(fg). Returns ``(B, H, W)``.

    Same as ``ensemble.py``: weights ``w_m ∝ max(p_m, 1-p_m)``, normalized over ``m``.
    """
    conf = torch.maximum(prob_stack, 1.0 - prob_stack)
    w = conf / conf.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return (prob_stack * w).sum(dim=0)


def _pad_to_multiple(x: torch.Tensor, mult: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad CHW tensor on bottom/right so H,W are divisible by ``mult``. Returns padded tensor and (H, W) original."""
    _, h, w = x.shape
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph == 0 and pw == 0:
        return x, (h, w)
    x = F.pad(x, (0, pw, 0, ph))
    return x, (h, w)


def build_fold_runs(
    *,
    checkpoints_dir: Path | None,
    all_tiles: list[str],
    cv_folds: int | None,
    val_fraction: float,
    seed: int,
) -> list[tuple[int, list[str], list[str]]]:
    """Same splits as ``validation.py``: K-fold under ``fold*`` dirs, else ``train_test_split``."""
    if checkpoints_dir is not None and checkpoints_dir.is_dir():
        fold_dirs = discover_fold_dirs(checkpoints_dir)
        if not fold_dirs:
            raise FileNotFoundError(f"No fold*/train/ under {checkpoints_dir}")
        single_train_dir = len(fold_dirs) == 1 and fold_dirs[0][1].name == "train"
        if single_train_dir:
            train_ids, val_ids = train_test_split(
                all_tiles,
                test_size=float(val_fraction),
                random_state=seed,
                shuffle=True,
            )
            return [(0, train_ids, val_ids)]
        k = cv_folds if cv_folds is not None else len(fold_dirs)
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        splits = list(kf.split(range(len(all_tiles))))
        if len(splits) < len(fold_dirs):
            raise ValueError(f"K-fold splits={len(splits)} but need at least {len(fold_dirs)} folds.")
        fold_runs: list[tuple[int, list[str], list[str]]] = []
        for fold_idx, _fold_path in fold_dirs:
            if fold_idx >= len(splits):
                continue
            tr_idx, va_idx = splits[fold_idx]
            fold_runs.append(
                (
                    fold_idx,
                    [all_tiles[i] for i in tr_idx],
                    [all_tiles[i] for i in va_idx],
                )
            )
        if not fold_runs:
            raise ValueError("No fold runs to evaluate.")
        return fold_runs
    train_ids, val_ids = train_test_split(
        all_tiles,
        test_size=float(val_fraction),
        random_state=seed,
        shuffle=True,
    )
    return [(0, train_ids, val_ids)]


def val_tiles_for_fallback(fold_runs: list[tuple[int, list[str], list[str]]]) -> list[str]:
    """Ordered unique val tile ids (K-fold: each tile appears once)."""
    seen: set[str] = set()
    out: list[str] = []
    for _, _, va in fold_runs:
        for t in va:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def discover_test_tile_ids_from_slim_rasters(test_dir: Path) -> list[str]:
    """Tile ids from ``{id}_slim.tif`` under ``test_dir`` (sorted, unique)."""
    if not test_dir.is_dir():
        return []
    suf = "_slim.tif"
    out: list[str] = []
    seen: set[str] = set()
    for p in sorted(test_dir.glob(f"*{suf}")):
        if not p.is_file():
            continue
        name = p.name
        if not name.endswith(suf):
            continue
        tid = name[: -len(suf)]
        if tid and tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


@torch.inference_mode()
def tune_binary_threshold_single(
    lit: SegLitModule,
    val_loader: DataLoader,
    device: torch.device,
    thresholds: torch.Tensor,
    *,
    use_tta: bool,
) -> tuple[float, float]:
    """Grid-search P(fg) threshold; logits are TTA or plain forward matching ``--tta``."""
    lit.eval()
    lit.to(device)
    iou_fn = mean_union_iou_batch_ignore
    thr = thresholds.to(device=device, dtype=torch.float32)
    sum_thr = torch.zeros(thr.numel(), device=device)
    n_samples = 0
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = x.size(0)
        logits = tta_d8_logits(lit, x) if use_tta else lit(x)
        prob_fg = torch.sigmoid(logits).squeeze(1)
        for ti in range(thr.numel()):
            t = thr[ti]
            pred_thr = (prob_fg > t).long()
            sum_thr[ti] += iou_fn(pred_thr.detach(), y.detach()) * bs
        n_samples += bs
    n = max(n_samples, 1)
    mean_thr = sum_thr / n
    best_idx = int(mean_thr.argmax().item())
    return float(thr[best_idx].item()), float(mean_thr[best_idx].item())


@torch.inference_mode()
def tune_threshold_ensemble_pooled(
    lits: list[SegLitModule],
    fold_runs: list[tuple[int, list[str], list[str]]],
    *,
    data_root: Path,
    geojson_path: Path,
    device: torch.device,
    thresholds: torch.Tensor,
    input_norm: str,
    ds_kw: dict,
    num_workers: int,
    use_tta: bool,
    ensemble_mode: str,
) -> float:
    """Pooled mean val_union_iou over all CV val tiles; per-fold norm (matches ``ensemble.py`` idea)."""
    iou_fn = mean_union_iou_batch_ignore
    thr = thresholds.to(device=device, dtype=torch.float32)
    sum_tta_thr = torch.zeros(thr.numel(), device=device)
    n_samples = 0
    ref = lits[0]
    nc = int(getattr(ref.hparams, "num_classes", 1))
    if nc != 1:
        raise ValueError("Threshold tuning supports binary (num_classes=1) only.")
    for lit in lits:
        lit.eval()
        lit.to(device)

    for _fold_idx, train_ids, val_ids in fold_runs:
        fit_ds = OsapiensTerraDataset(
            data_root,
            geojson_path,
            tile_ids=train_ids,
            augment_flip=False,
            **ds_kw,
        )
        if input_norm == "zscore":
            fit_ds.fit_zscore_from_tiles()
        else:
            fit_ds.fit_minmax_from_tiles()

        val_ds = OsapiensTerraDataset(
            data_root,
            geojson_path,
            tile_ids=val_ids,
            augment_flip=False,
            **ds_kw,
        )
        if input_norm == "zscore":
            val_ds.set_zscore(fit_ds._zs_mean, fit_ds._zs_std)
        else:
            val_ds.set_minmax(fit_ds._mm_min, fit_ds._mm_max)

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            bs = x.size(0)
            if ensemble_mode == "confidence":
                if use_tta:
                    prob_list = [torch.sigmoid(tta_d8_logits(lit, x)).squeeze(1) for lit in lits]
                else:
                    prob_list = [torch.sigmoid(lit(x)).squeeze(1) for lit in lits]
                prob_fg = _confidence_combine_binary_prob_stack(torch.stack(prob_list, dim=0))
            elif use_tta:
                logits_e = torch.stack([tta_d8_logits(lit, x) for lit in lits], dim=0).mean(0)
                prob_fg = torch.sigmoid(logits_e).squeeze(1)
            else:
                logits_e = torch.stack([lit(x) for lit in lits], dim=0).mean(0)
                prob_fg = torch.sigmoid(logits_e).squeeze(1)
            for ti in range(thr.numel()):
                t = thr[ti]
                pred_thr = (prob_fg > t).long()
                sum_tta_thr[ti] += iou_fn(pred_thr.detach(), y.detach()) * bs
            n_samples += bs

    n = max(n_samples, 1)
    mean_thr = sum_tta_thr / n
    best_idx = int(mean_thr.argmax().item())
    return float(thr[best_idx].item())


@torch.inference_mode()
def predict_tile_mask(
    lit: SegLitModule,
    feats_chw: torch.Tensor,
    *,
    norm_mean: torch.Tensor,
    norm_denom: torch.Tensor,
    input_norm: str,
    prob_threshold: float = 0.5,
    use_tta: bool = True,
) -> np.ndarray:
    """Return HxW uint8 binary mask (1 = deforestation class)."""
    lit.eval()
    device = next(lit.parameters()).device
    norm_mean = norm_mean.to(device=device, dtype=torch.float32)
    norm_denom = norm_denom.to(device=device, dtype=torch.float32)
    x = feats_chw.float().to(device)
    if input_norm == "zscore":
        x = (x - norm_mean) / norm_denom
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        x = (x - norm_mean) / norm_denom
        x = x.clamp(0.0, 1.0)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

    x, (h0, w0) = _pad_to_multiple(x, 16)
    x = x.unsqueeze(0)
    if lit.hparams.input_divisor != 1.0:
        x = x / lit.hparams.input_divisor
    if use_tta:
        logits = tta_d8_logits(lit, x)
    else:
        logits = lit(x)
    nc = int(getattr(lit.hparams, "num_classes", 1))
    if nc == 1:
        prob = logits.sigmoid().squeeze(1).squeeze(0)
        pred = (prob > prob_threshold)[:h0, :w0]
    else:
        pred = logits.argmax(dim=1).squeeze(0)
        pred = pred[:h0, :w0]
        pred = pred > 0
    return pred.to(torch.uint8).cpu().numpy()


@torch.inference_mode()
def predict_tile_mask_ensemble(
    lits: list[SegLitModule],
    feats_chw: torch.Tensor,
    *,
    norm_mean: torch.Tensor,
    norm_denom: torch.Tensor,
    input_norm: str,
    prob_threshold: float = 0.5,
    use_tta: bool = True,
    ensemble_mode: str = "mean",
) -> np.ndarray:
    """Ensemble folds: ``mean`` = average logits then sigmoid; ``confidence`` = blend P(fg) like ``ensemble.py``."""
    if not lits:
        raise ValueError("ensemble requires at least one checkpoint")
    ref = lits[0]
    for lit in lits:
        lit.eval()
    device = next(ref.parameters()).device
    norm_mean = norm_mean.to(device=device, dtype=torch.float32)
    norm_denom = norm_denom.to(device=device, dtype=torch.float32)
    x = feats_chw.float().to(device)
    if input_norm == "zscore":
        x = (x - norm_mean) / norm_denom
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        x = (x - norm_mean) / norm_denom
        x = x.clamp(0.0, 1.0)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

    x, (h0, w0) = _pad_to_multiple(x, 16)
    x = x.unsqueeze(0)
    if ref.hparams.input_divisor != 1.0:
        x = x / ref.hparams.input_divisor

    nc = int(getattr(ref.hparams, "num_classes", 1))
    if ensemble_mode == "confidence":
        if nc != 1:
            raise ValueError("--ensemble-mode confidence supports binary (num_classes=1) only.")
        if use_tta:
            prob_list = [torch.sigmoid(tta_d8_logits(lit, x)).squeeze(1) for lit in lits]
        else:
            prob_list = [torch.sigmoid(lit(x)).squeeze(1) for lit in lits]
        prob = _confidence_combine_binary_prob_stack(torch.stack(prob_list, dim=0)).squeeze(0)
        pred = (prob > prob_threshold)[:h0, :w0]
        return pred.to(torch.uint8).cpu().numpy()

    if use_tta:
        logits = torch.stack([tta_d8_logits(lit, x) for lit in lits], dim=0).mean(0)
    else:
        logits = torch.stack([lit(x) for lit in lits], dim=0).mean(0)
    if nc == 1:
        prob = logits.sigmoid().squeeze(1).squeeze(0)
        pred = (prob > prob_threshold)[:h0, :w0]
    else:
        pred = logits.argmax(dim=1).squeeze(0)
        pred = pred[:h0, :w0]
        pred = pred > 0
    return pred.to(torch.uint8).cpu().numpy()


def main() -> None:
    repo = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Export GeoJSON: optional K-fold logit ensemble, TTA + val threshold tuning like validation.py."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Single Lightning checkpoint. Ignored if --ensemble-checkpoints-dir is set.",
    )
    p.add_argument(
        "--ensemble-checkpoints-dir",
        type=Path,
        default=None,
        help="Directory with fold0 … (best-*.ckpt each). Combine folds via --ensemble-mode (+ TTA if enabled).",
    )
    p.add_argument(
        "--ensemble-mode",
        choices=("mean", "confidence"),
        default="mean",
        help="With multiple fold checkpoints: mean=average logits then sigmoid (default); "
        "confidence=per-pixel blend of each model's P(fg) with weights ∝ max(p,1-p) (see ensemble.py).",
    )
    p.add_argument("--data-root", type=Path, default=repo / "data")
    p.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Directory with challenge test *_slim.tif rasters. Default: <data-root>/test.",
    )
    p.add_argument(
        "--train-geojson",
        type=Path,
        default=repo / "data/train_tiles.geojson",
        help="Train tile list (same as validation --geojson). Used for norm + threshold tuning.",
    )
    p.add_argument(
        "--input-norm",
        choices=("zscore", "minmax"),
        default="zscore",
        help="Normalization used when training the checkpoint.",
    )
    p.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="8-way TTA on logits (default: on), same as validation.",
    )
    p.add_argument(
        "--tune-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Grid-search best P(fg) on val with TTA (default: on). Use --no-tune-threshold --prob-threshold for manual.",
    )
    p.add_argument(
        "--thresh-min",
        type=float,
        default=0.01,
        help="Threshold grid min (sigmoid P(fg)); same as validation.",
    )
    p.add_argument(
        "--thresh-max",
        type=float,
        default=0.99,
        help="Threshold grid max.",
    )
    p.add_argument(
        "--thresh-steps",
        type=int,
        default=50,
        metavar="N",
        help="Number of thresholds between thresh-min and thresh-max.",
    )
    p.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help="Used when --no-tune-threshold (binary only).",
    )
    p.add_argument(
        "--predict-geojson",
        type=Path,
        default=None,
        help="Tiles to predict. Default: metadata/test_tiles.geojson; if none match, "
        "auto-discover *_slim.tif under --test-dir; else val fallback.",
    )
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=None, metavar="K")
    p.add_argument("--min-area-ha", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--crops-per-tile", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--output",
        type=Path,
        default=repo / "runs/submission_best-epoch36.geojson",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    args.data_root = args.data_root.resolve()
    ensemble_root = (
        Path(args.ensemble_checkpoints_dir).resolve() if args.ensemble_checkpoints_dir is not None else None
    )
    if ensemble_root is not None and args.checkpoint is not None:
        raise ValueError("Use either --checkpoint or --ensemble-checkpoints-dir, not both.")
    if ensemble_root is None and args.ensemble_mode != "mean":
        _log.warning(
            "Ignoring --ensemble-mode=%s (only applies with --ensemble-checkpoints-dir); using single model.",
            args.ensemble_mode,
        )
    if ensemble_root is None:
        if args.checkpoint is None:
            args.checkpoint = (
                repo / "runs/checkpoints/version_0/train/best-epoch=36-val_union_iou=0.5061.ckpt"
            )
        args.checkpoint = args.checkpoint.resolve()
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    test_dir = (args.test_dir if args.test_dir is not None else args.data_root / "test").resolve()

    geojson_path = _resolve_geojson(args.data_root, Path(args.train_geojson).resolve())
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(geojson_path, encoding="utf-8") as f:
        all_train_names = [feat["properties"]["name"] for feat in json.load(f)["features"]]

    fold_runs = build_fold_runs(
        checkpoints_dir=ensemble_root,
        all_tiles=all_train_names,
        cv_folds=args.cv_folds,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    val_fallback_ids = val_tiles_for_fallback(fold_runs)

    def resolve_feat_path(tile_id: str) -> Path | None:
        pt = test_dir / f"{tile_id}_slim.tif"
        if pt.is_file():
            return pt
        pf = args.data_root / "features" / f"{tile_id}_slim.tif"
        if pf.is_file():
            return pf
        return None

    predict_names: list[str]
    if args.predict_geojson is not None:
        pg = Path(args.predict_geojson).resolve()
        with open(pg, encoding="utf-8") as f:
            candidates = [feat["properties"]["name"] for feat in json.load(f)["features"]]
        predict_names = [t for t in candidates if resolve_feat_path(t) is not None]
        if not predict_names:
            raise FileNotFoundError(
                f"No *_slim.tif for listed tiles under {test_dir} or {args.data_root / 'features'}."
            )
    else:
        test_path = args.data_root / "metadata" / "test_tiles.geojson"
        if test_path.is_file():
            with open(test_path, encoding="utf-8") as f:
                candidates = [feat["properties"]["name"] for feat in json.load(f)["features"]]
        else:
            candidates = []

        predict_names = [t for t in candidates if resolve_feat_path(t) is not None]
        if not predict_names:
            discovered = discover_test_tile_ids_from_slim_rasters(test_dir)
            if discovered:
                predict_names = discovered
                _log.info(
                    "Using %d tile(s) from *_slim.tif in %s "
                    "(metadata/test_tiles.geojson missing or no name/file match).",
                    len(predict_names),
                    test_dir,
                )
            else:
                predict_names = [t for t in val_fallback_ids if resolve_feat_path(t) is not None]
                _log.info(
                    "No *_slim.tif under %s; predicting validation fold tiles (%d with rasters).",
                    test_dir,
                    len(predict_names),
                )
        else:
            _log.info("Predicting %d test tiles from %s.", len(predict_names), test_dir)

    ds_kw = dict(
        crop_size=args.crop_size,
        crops_per_tile=args.crops_per_tile,
        input_norm=args.input_norm,
        mode="full",
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lits: list[SegLitModule]
    if ensemble_root is not None:
        folds = discover_fold_dirs(ensemble_root)
        if not folds:
            raise FileNotFoundError(f"No fold*/train/ under ensemble dir: {ensemble_root}")
        lits = []
        for fi, fd in folds:
            ckpt = find_best_checkpoint(fd)
            _log.info("Ensemble fold %d: %s", fi, ckpt.name)
            m = load_seg_lit_from_checkpoint(str(ckpt), map_location=device, strict=False)
            m.to(device)
            m.eval()
            lits.append(m)
        _log.info(
            "Loaded %d fold models from %s (ensemble_mode=%s)",
            len(lits),
            ensemble_root,
            args.ensemble_mode,
        )
    else:
        lit = load_seg_lit_from_checkpoint(str(args.checkpoint), map_location=device, strict=False)
        lit.to(device)
        lits = [lit]

    nc = int(getattr(lits[0].hparams, "num_classes", 1))
    thresh_grid = torch.linspace(args.thresh_min, args.thresh_max, args.thresh_steps)

    prob_threshold = float(args.prob_threshold)
    if args.tune_threshold:
        if nc != 1:
            _log.warning(
                "Threshold grid search supports binary checkpoints only; num_classes=%d — using argmax (threshold ignored).",
                nc,
            )
        else:
            _log.info(
                "Tuning threshold (TTA=%s): grid [%.3f..%.3f] x%d (same protocol as validation.py)",
                args.tta,
                args.thresh_min,
                args.thresh_max,
                args.thresh_steps,
            )
            if len(lits) == 1:
                tun_train = fold_runs[0][1]
                tun_val = fold_runs[0][2]
                fit_ds = OsapiensTerraDataset(
                    args.data_root,
                    geojson_path,
                    tile_ids=tun_train,
                    augment_flip=False,
                    **ds_kw,
                )
                if args.input_norm == "zscore":
                    fit_ds.fit_zscore_from_tiles()
                else:
                    fit_ds.fit_minmax_from_tiles()
                val_ds = OsapiensTerraDataset(
                    args.data_root,
                    geojson_path,
                    tile_ids=tun_val,
                    augment_flip=False,
                    **ds_kw,
                )
                if args.input_norm == "zscore":
                    val_ds.set_zscore(fit_ds._zs_mean, fit_ds._zs_std)
                else:
                    val_ds.set_minmax(fit_ds._mm_min, fit_ds._mm_max)
                val_loader = DataLoader(
                    val_ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=False,
                )
                prob_threshold, iu_opt = tune_binary_threshold_single(
                    lits[0], val_loader, device, thresh_grid, use_tta=args.tta
                )
                _log.info(
                    "Val tuning (single model): best P(fg)=%.6f (mean val_union_iou @ that t: %.6f)",
                    prob_threshold,
                    iu_opt,
                )
            else:
                prob_threshold = tune_threshold_ensemble_pooled(
                    lits,
                    fold_runs,
                    data_root=args.data_root,
                    geojson_path=geojson_path,
                    device=device,
                    thresholds=thresh_grid,
                    input_norm=args.input_norm,
                    ds_kw=ds_kw,
                    num_workers=args.num_workers,
                    use_tta=args.tta,
                    ensemble_mode=args.ensemble_mode,
                )
                _log.info("Val tuning (ensemble, pooled CV val): best P(fg)=%.6f", prob_threshold)
    else:
        _log.info("Using fixed --prob-threshold=%.6f (no grid search)", prob_threshold)

    # Input norm for raster export: single model = same train split as tuning; K-fold ensemble = all train tiles.
    if ensemble_root is None:
        tr_sub = fold_runs[0][1]
        ds_fit = OsapiensTerraDataset(
            args.data_root,
            geojson_path,
            tile_ids=tr_sub,
            crops_per_tile=1,
            crop_size=args.crop_size,
            augment_flip=False,
            input_norm=args.input_norm,
        )
    else:
        ds_fit = OsapiensTerraDataset(
            args.data_root,
            geojson_path,
            tile_ids=all_train_names,
            crops_per_tile=1,
            crop_size=args.crop_size,
            augment_flip=False,
            input_norm=args.input_norm,
        )
    if args.input_norm == "zscore":
        ds_fit.fit_zscore_from_tiles()
        norm_mean = ds_fit._zs_mean
        norm_denom = ds_fit._zs_std
    else:
        ds_fit.fit_minmax_from_tiles()
        norm_mean = ds_fit._mm_min
        norm_denom = ds_fit._mm_max - ds_fit._mm_min
    assert norm_mean is not None and norm_denom is not None
    if ensemble_root is None:
        _log.info("Submission norm: fit on %d train tiles (same split as tuning).", len(tr_sub))
    else:
        _log.info("Submission norm: fit on all %d train tiles (ensemble export).", len(all_train_names))

    merged_features: list[dict] = []
    tmpdir = tempfile.mkdtemp(prefix="pred_tif_")

    thr_for_masks = prob_threshold if nc == 1 else 0.5

    try:
        for tile_id in predict_names:
            feat_path = resolve_feat_path(tile_id)
            if feat_path is None:
                _log.warning("Skip %s (no slim.tif under %s or features/)", tile_id, test_dir)
                continue

            with rasterio.open(feat_path) as src:
                arr = np.array(src.read())
                profile = src.profile.copy()

            feats = torch.from_numpy(arr)
            if len(lits) == 1:
                mask = predict_tile_mask(
                    lits[0],
                    feats,
                    norm_mean=norm_mean,
                    norm_denom=norm_denom,
                    input_norm=args.input_norm,
                    prob_threshold=thr_for_masks,
                    use_tta=args.tta,
                )
            else:
                mask = predict_tile_mask_ensemble(
                    lits,
                    feats,
                    norm_mean=norm_mean,
                    norm_denom=norm_denom,
                    input_norm=args.input_norm,
                    prob_threshold=thr_for_masks,
                    use_tta=args.tta,
                    ensemble_mode=args.ensemble_mode,
                )

            profile.update(count=1, dtype="uint8", nodata=None)
            tif_path = Path(tmpdir) / f"{tile_id}_pred.tif"
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.write(mask.astype(np.uint8), 1)

            try:
                gj = raster_to_geojson(
                    raster_path=tif_path,
                    output_path=None,
                    min_area_ha=args.min_area_ha,
                )
            except ValueError as e:
                _log.info("Tile %s: %s", tile_id, e)
                continue
            merged_features.extend(gj["features"])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not merged_features:
        raise RuntimeError(
            "No polygon features produced (all tiles empty or below min_area_ha). "
            "Lower --min-area-ha or check predictions."
        )

    out_gj = {"type": "FeatureCollection", "features": merged_features}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_gj, f)
    _log.info("Wrote %s (%d features)", args.output, len(merged_features))


if __name__ == "__main__":
    main()
