#!/usr/bin/env python3
"""Evaluate saved K-fold checkpoints on each fold's validation split.

Uses the same **input normalization** as training (z-score or min–max, refit on each fold's train tiles),
**full-tile** batches (batch size 1), and a **single-logit** head: **without** TTA, **with** 8-way TTA
averaging logits then sigmoid, and **with** TTA + **best foreground probability threshold** (grid search).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

from utils.dataset import OsapiensTerraDataset
from utils.metrics import mean_union_iou_batch_ignore
from train import SegLitModule, load_seg_lit_from_checkpoint

_log = logging.getLogger(__name__)


def tta_d8_logits(lit: SegLitModule, x: torch.Tensor) -> torch.Tensor:
    """Average logits over 8 transforms: rot90^k and rot90^k ∘ flip_W (k=0..3)."""
    outs: list[torch.Tensor] = []
    for idx in range(8):
        k = idx % 4
        flip_w = idx // 4
        xt = torch.rot90(x, k, dims=(-2, -1))
        if flip_w:
            xt = xt.flip(-1)
        log = lit(xt)
        if flip_w:
            log = log.flip(-1)
        log = torch.rot90(log, -k, dims=(-2, -1))
        outs.append(log)
    return torch.stack(outs, dim=0).mean(0)


def _resolve_geojson(data_root: Path, geojson: Path | None) -> Path:
    if geojson is not None:
        if not geojson.is_file():
            raise FileNotFoundError(f"--geojson not found: {geojson}")
        return geojson
    for cand in (data_root / "metadata" / "train_tiles.geojson", data_root / "train_tiles.geojson"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No train_tiles.geojson under {data_root}. Pass --geojson or add metadata/train_tiles.geojson."
    )


def find_best_checkpoint(fold_dir: Path) -> Path:
    cks = sorted(fold_dir.glob("best-*.ckpt"))
    if not cks:
        raise FileNotFoundError(f"No best-*.ckpt under {fold_dir}")
    if len(cks) > 1:

        def score(p: Path) -> float:
            m = re.search(r"val_union_iou=([\d.]+)", p.name)
            return float(m.group(1)) if m else 0.0

        cks = sorted(cks, key=score, reverse=True)
        _log.warning("Multiple best checkpoints in %s; using %s", fold_dir, cks[0].name)
    return cks[0]


def discover_fold_dirs(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    """K-fold layout: ``fold0``, ``fold1``, … — or a single ``train/`` run (non-CV)."""
    out: list[tuple[int, Path]] = []
    for p in sorted(checkpoints_dir.iterdir()):
        if not p.is_dir():
            continue
        m = re.fullmatch(r"fold(\d+)", p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    if out:
        return out
    train_p = checkpoints_dir / "train"
    if train_p.is_dir():
        return [(0, train_p)]
    return []


def _pred_from_logits(lit: SegLitModule, logits: torch.Tensor, *, thr: float | None = None):
    """Binary (1 ch) → (B,H,W) long 0/1; multi-class → argmax."""
    nc = int(getattr(lit.hparams, "num_classes", 1))
    if nc == 1:
        prob = torch.sigmoid(logits).squeeze(1)
        if thr is None:
            t = 0.5
        else:
            t = thr
        return (prob > t).long()
    if thr is not None:
        raise ValueError("Threshold path only supported for binary (num_classes=1).")
    return logits.argmax(dim=1)


@torch.inference_mode()
def evaluate_fold_no_tta_tta_and_threshold(
    lit: SegLitModule,
    val_loader: DataLoader,
    device: torch.device,
    *,
    thresholds: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Returns (iou_no_tta, iou_tta_argmax, iou_tta_best_thresh, best_threshold)."""
    lit.eval()
    lit.to(device)
    iou_fn = mean_union_iou_batch_ignore
    nc = int(getattr(lit.hparams, "num_classes", 1))
    sum_no = 0.0
    sum_tta = 0.0
    n_samples = 0
    thr = thresholds.to(device=device, dtype=torch.float32)
    sum_tta_thr = torch.zeros(thr.numel(), device=device)

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = x.size(0)
        logits_no = lit(x)
        pred_no = _pred_from_logits(lit, logits_no, thr=None)
        sum_no += iou_fn(pred_no.detach(), y.detach()) * bs

        logits_tta = tta_d8_logits(lit, x)
        pred_tta = _pred_from_logits(lit, logits_tta, thr=None)
        sum_tta += iou_fn(pred_tta.detach(), y.detach()) * bs

        if nc == 1:
            prob_fg = torch.sigmoid(logits_tta).squeeze(1)
            for ti in range(thr.numel()):
                t = thr[ti]
                pred_thr = (prob_fg > t).long()
                sum_tta_thr[ti] += iou_fn(pred_thr.detach(), y.detach()) * bs
        else:
            prob_fg = logits_tta.softmax(dim=1)[:, 1]
            for ti in range(thr.numel()):
                t = thr[ti]
                pred_thr = (prob_fg > t).long()
                sum_tta_thr[ti] += iou_fn(pred_thr.detach(), y.detach()) * bs

        n_samples += bs

    n = max(n_samples, 1)
    mean_thr = sum_tta_thr / n
    best_idx = int(mean_thr.argmax().item())
    best_t = float(thr[best_idx].item())
    iou_thr = float(mean_thr[best_idx].item())
    return sum_no / n, sum_tta / n, iou_thr, best_t


def _print_summary_table(
    fold_rows: list[tuple[int, float, float, float, float]],
    no_tta_scores: list[float],
    tta_scores: list[float],
    tta_opt_scores: list[float],
) -> None:
    sep = "-" * 92
    lines = [
        sep,
        "Per-fold val_union_iou: no TTA | TTA+argmax | best P(fg) thresh | TTA+thresh",
        sep,
        f"{'fold':>4}  {'no_tta':>12}  {'tta_argmax':>12}  {'best_t':>10}  {'tta@t':>12}",
        sep,
    ]
    for fi, nt, tt, iou_t, bt in fold_rows:
        lines.append(f"{fi:>4}  {nt:>12.6f}  {tt:>12.6f}  {bt:>10.4f}  {iou_t:>12.6f}")
    if no_tta_scores and tta_scores and tta_opt_scores:
        lines.append(sep)
        lines.append(
            f"{'mean':>4}  {sum(no_tta_scores) / len(no_tta_scores):>12.6f}  "
            f"{sum(tta_scores) / len(tta_scores):>12.6f}  {'—':>10}  "
            f"{sum(tta_opt_scores) / len(tta_opt_scores):>12.6f}"
        )
    lines.append(sep)
    table = "\n".join(lines)
    _log.info("\n%s", table)
    print(table)


def main() -> None:
    repo = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="K-fold eval: val_union_iou (no TTA, TTA+argmax, TTA+best P(fg) threshold)"
    )
    p.add_argument("--data-root", type=Path, default=repo / "data")
    p.add_argument("--geojson", type=Path, default=None)
    p.add_argument(
        "--version",
        type=int,
        default=None,
        metavar="N",
        help="Use <repo>/exps/checkpoints/version_N. Ignored if --checkpoints-dir is set.",
    )
    p.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Directory with fold0, fold1, … Default: exps/checkpoints/version_1, or version_N with --version.",
    )
    p.add_argument("--cv-folds", type=int, default=None, metavar="K")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="When checkpoints use train/ (single split), val tiles = this fraction; must match training.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--crops-per-tile", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--input-norm",
        choices=("zscore", "minmax"),
        default="zscore",
        help="Must match training (fit on fold train tiles only).",
    )
    p.add_argument(
        "--thresh-min",
        type=float,
        default=0.01,
        help="Min P(foreground) for threshold grid (sigmoid for binary, else softmax ch1).",
    )
    p.add_argument(
        "--thresh-max",
        type=float,
        default=0.99,
        help="Max P(foreground) for threshold grid.",
    )
    p.add_argument(
        "--thresh-steps",
        type=int,
        default=50,
        metavar="N",
        help="Number of evenly spaced thresholds between thresh-min and thresh-max.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data_root = args.data_root.resolve()
    if args.checkpoints_dir is not None:
        checkpoints_dir = args.checkpoints_dir.resolve()
    elif args.version is not None:
        checkpoints_dir = (repo / "exps" / "checkpoints" / f"version_{args.version}").resolve()
    else:
        checkpoints_dir = (repo / "exps" / "checkpoints" / "version_1").resolve()
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"--checkpoints-dir not found: {checkpoints_dir}")

    fold_dirs = discover_fold_dirs(checkpoints_dir)
    if not fold_dirs:
        raise FileNotFoundError(
            f"No fold* or train/ directory under {checkpoints_dir}"
        )

    geojson_path = _resolve_geojson(data_root, args.geojson)
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(geojson_path, encoding="utf-8") as f:
        all_tiles = [feat["properties"]["name"] for feat in json.load(f)["features"]]

    single_train_dir = len(fold_dirs) == 1 and fold_dirs[0][1].name == "train"
    if single_train_dir:
        k = 1
        train_ids, val_ids = train_test_split(
            all_tiles,
            test_size=float(args.val_fraction),
            random_state=args.seed,
            shuffle=True,
        )
        fold_runs: list[tuple[int, Path, list[str], list[str]]] = [
            (0, fold_dirs[0][1], train_ids, val_ids)
        ]
    else:
        k = args.cv_folds if args.cv_folds is not None else len(fold_dirs)
        kf = KFold(n_splits=k, shuffle=True, random_state=args.seed)
        splits = list(kf.split(range(len(all_tiles))))
        if len(splits) < len(fold_dirs):
            raise ValueError(f"K-fold splits={len(splits)} but need at least {len(fold_dirs)} folds.")
        fold_runs = []
        for fold_idx, fold_path in fold_dirs:
            if fold_idx >= len(splits):
                continue
            tr_idx, va_idx = splits[fold_idx]
            fold_runs.append(
                (
                    fold_idx,
                    fold_path,
                    [all_tiles[i] for i in tr_idx],
                    [all_tiles[i] for i in va_idx],
                )
            )
        if not fold_runs:
            raise ValueError("No fold runs to evaluate.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds_kw = dict(
        crop_size=args.crop_size,
        crops_per_tile=args.crops_per_tile,
        input_norm=args.input_norm,
        mode="full",
    )

    no_tta_scores: list[float] = []
    tta_scores: list[float] = []
    tta_opt_scores: list[float] = []
    best_thresholds: list[float] = []
    fold_rows: list[tuple[int, float, float, float, float]] = []

    thresh_grid = torch.linspace(args.thresh_min, args.thresh_max, args.thresh_steps)

    _log.info(
        "checkpoints_dir=%s K=%d device=%s threshold_grid=[%.3f..%.3f] x%d",
        checkpoints_dir,
        k,
        device,
        args.thresh_min,
        args.thresh_max,
        args.thresh_steps,
    )

    for fold_idx, fold_path, train_ids, val_ids in fold_runs:
        ckpt = find_best_checkpoint(fold_path)

        _log.info("Fold %d: checkpoint=%s val_tiles=%d", fold_idx, ckpt.name, len(val_ids))

        fit_ds = OsapiensTerraDataset(
            data_root,
            geojson_path,
            tile_ids=train_ids,
            augment_flip=False,
            **ds_kw,
        )
        if args.input_norm == "zscore":
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

        lit = load_seg_lit_from_checkpoint(str(ckpt), map_location=device, strict=False)

        iou_no, iou_tta, iou_opt, best_t = evaluate_fold_no_tta_tta_and_threshold(
            lit, val_loader, device, thresholds=thresh_grid
        )
        no_tta_scores.append(iou_no)
        tta_scores.append(iou_tta)
        tta_opt_scores.append(iou_opt)
        best_thresholds.append(best_t)
        fold_rows.append((fold_idx, iou_no, iou_tta, iou_opt, best_t))

        _log.info(
            "Fold %d: no_tta=%.6f tta_argmax=%.6f tta@best_t=%.6f (best P(fg)=%.4f)",
            fold_idx,
            iou_no,
            iou_tta,
            iou_opt,
            best_t,
        )

    if fold_rows:
        _print_summary_table(fold_rows, no_tta_scores, tta_scores, tta_opt_scores)

    if no_tta_scores:
        _log.info(
            "Mean val_union_iou no_tta: %.6f (per-fold: %s)",
            sum(no_tta_scores) / len(no_tta_scores),
            [round(s, 6) for s in no_tta_scores],
        )
    if tta_scores:
        _log.info(
            "Mean val_union_iou tta_argmax: %.6f (per-fold: %s)",
            sum(tta_scores) / len(tta_scores),
            [round(s, 6) for s in tta_scores],
        )
    if tta_opt_scores:
        _log.info(
            "Mean val_union_iou tta@best_thresh: %.6f (per-fold: %s)",
            sum(tta_opt_scores) / len(tta_opt_scores),
            [round(s, 6) for s in tta_opt_scores],
        )
    if best_thresholds:
        _log.info(
            "Best P(fg) threshold per fold: %s (mean=%.4f; each fold tuned independently on its val split)",
            [round(t, 4) for t in best_thresholds],
            sum(best_thresholds) / len(best_thresholds),
        )


if __name__ == "__main__":
    main()
