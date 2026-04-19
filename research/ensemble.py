#!/usr/bin/env python3
"""Ensemble multiple K-fold experiments (same folds / seed): combine predictions, then val_union_iou.

For each fold, loads ``best-*.ckpt`` from every experiment under ``fold{i}/``. Combination modes:

- **mean** — equal average of logits, then sigmoid / softmax as in training.
- **weighted** — ``--weights`` (normalized) average of logits.
- **confidence** — per-pixel weights from each model's confidence ``max(p, 1-p)`` on foreground
  probability ``p`` (binary); weights are normalized across models at each pixel, then ensemble
  probability is ``sum_m w_m * p_m`` (no logit average).

Metrics match ``validation.py``: no TTA @ 0.5, TTA @ 0.5, TTA + threshold grid on ensemble ``P(fg)``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

from utils.dataset import OsapiensTerraDataset
from utils.metrics import mean_union_iou_batch_ignore
from train import SegLitModule, load_seg_lit_from_checkpoint
from validation import (
    _pred_from_logits,
    _resolve_geojson,
    discover_fold_dirs,
    find_best_checkpoint,
    tta_d8_logits,
)

_log = logging.getLogger(__name__)


def _combine_logits(logits_list: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """Weighted or unweighted average along model dimension 0.

    ``weights`` has shape ``(M,)`` and sums to 1; ``logits_list`` has length ``M``.
    """
    stacked = torch.stack(logits_list, dim=0)
    w = weights.to(device=stacked.device, dtype=stacked.dtype).view(-1, *([1] * (stacked.ndim - 1)))
    return (stacked * w).sum(0)


def _confidence_combine_binary_prob_stack(prob_stack: torch.Tensor) -> torch.Tensor:
    """``prob_stack`` shape ``(M, B, H, W)`` with per-model ``sigmoid`` probs. Returns ``(B, H, W)``.

    Per-pixel weights ``w_m ∝ max(p_m, 1-p_m)``, normalized over ``m``.
    """
    conf = torch.maximum(prob_stack, 1.0 - prob_stack)
    w = conf / conf.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return (prob_stack * w).sum(dim=0)


@torch.inference_mode()
def evaluate_ensemble_fold(
    lits: list[SegLitModule],
    val_loader: DataLoader,
    device: torch.device,
    *,
    thresholds: torch.Tensor,
    weights: torch.Tensor,
    ensemble_mode: str,
) -> tuple[float, float, float, float]:
    """Returns (iou_no_tta, iou_tta_05, iou_tta_best_thresh, best_threshold).

    ``ensemble_mode``: ``mean`` | ``weighted`` (use ``weights`` on logits) | ``confidence`` (ignore ``weights``).
    """
    if not lits:
        raise ValueError("Need at least one model for ensemble.")
    for lit in lits:
        lit.eval()
        lit.to(device)
    nc = int(getattr(lits[0].hparams, "num_classes", 1))
    for lit in lits[1:]:
        if int(getattr(lit.hparams, "num_classes", 1)) != nc:
            raise ValueError("All ensemble checkpoints must use the same num_classes.")
    ref = lits[0]
    iou_fn = mean_union_iou_batch_ignore
    sum_no = 0.0
    sum_tta = 0.0
    n_samples = 0
    thr = thresholds.to(device=device, dtype=torch.float32)
    sum_tta_thr = torch.zeros(thr.numel(), device=device)

    use_confidence = ensemble_mode == "confidence"
    if use_confidence and nc != 1:
        raise ValueError(
            "--ensemble-mode confidence uses per-pixel max(p,1-p) weights on P(foreground); "
            "only num_classes==1 is supported."
        )

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = x.size(0)

        if use_confidence:
            prob_stack_no = torch.stack(
                [torch.sigmoid(lit(x)).squeeze(1) for lit in lits], dim=0
            )
            p_no = _confidence_combine_binary_prob_stack(prob_stack_no)
            pred_no = (p_no > 0.5).long()
            prob_stack_t = torch.stack(
                [torch.sigmoid(tta_d8_logits(lit, x)).squeeze(1) for lit in lits], dim=0
            )
            p_tta = _confidence_combine_binary_prob_stack(prob_stack_t)
            pred_tta = (p_tta > 0.5).long()
            sum_no += iou_fn(pred_no.detach(), y.detach()) * bs
            sum_tta += iou_fn(pred_tta.detach(), y.detach()) * bs
            prob_fg = p_tta
            for ti in range(thr.numel()):
                t = thr[ti]
                pred_thr = (prob_fg > t).long()
                sum_tta_thr[ti] += iou_fn(pred_thr.detach(), y.detach()) * bs
        else:
            logits_no = _combine_logits([lit(x) for lit in lits], weights)
            pred_no = _pred_from_logits(ref, logits_no, thr=None)
            sum_no += iou_fn(pred_no.detach(), y.detach()) * bs

            logits_tta = _combine_logits([tta_d8_logits(lit, x) for lit in lits], weights)
            pred_tta = _pred_from_logits(ref, logits_tta, thr=None)
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
        "Ensemble per-fold val_union_iou (logit combine): no TTA | TTA@0.5 | best P(fg) | TTA@best_t",
        sep,
        f"{'fold':>4}  {'no_tta':>12}  {'tta_0.5':>12}  {'best_t':>10}  {'tta@t':>12}",
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


def _assert_matching_folds(exp_dirs: list[Path], ref_folds: list[tuple[int, Path]]) -> None:
    ref_idx = {i for i, _ in ref_folds}
    for ed in exp_dirs:
        fds = discover_fold_dirs(ed)
        idx = {i for i, _ in fds}
        if idx != ref_idx:
            raise ValueError(
                f"Fold mismatch: {exp_dirs[0].resolve()} has folds {sorted(ref_idx)}, "
                f"but {ed.resolve()} has {sorted(idx)}. Same K-fold layout is required."
            )


def main() -> None:
    repo = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Ensemble logits from multiple checkpoint runs (same folds), eval with TTA + threshold"
    )
    p.add_argument(
        "--experiments",
        type=Path,
        nargs="+",
        required=True,
        help="Checkpoint roots, each with fold0 … fold{K-1} (or train/ for a single split).",
    )
    p.add_argument("--data-root", type=Path, default=repo / "data")
    p.add_argument("--geojson", type=Path, default=None)
    p.add_argument("--cv-folds", type=int, default=None, metavar="K")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Single train/ layout: must match each training run.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--crops-per-tile", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--input-norm",
        choices=("zscore", "minmax"),
        default="zscore",
        help="Must match training.",
    )
    p.add_argument("--thresh-min", type=float, default=0.01)
    p.add_argument("--thresh-max", type=float, default=0.99)
    p.add_argument("--thresh-steps", type=int, default=50, metavar="N")
    p.add_argument(
        "--ensemble-mode",
        choices=("mean", "weighted", "confidence"),
        default="mean",
        help="mean: equal logit average. weighted: --weights on logits. "
        "confidence: binary only — per-pixel weights ∝ max(p,1-p) on each model's P(fg), then blend probs.",
    )
    p.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        metavar="W",
        help="Required for weighted mode: one weight per --experiments. Ignored for mean/confidence.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    exp_dirs = [e.resolve() for e in args.experiments]
    for ed in exp_dirs:
        if not ed.is_dir():
            raise FileNotFoundError(f"Experiment dir not found: {ed}")

    m = len(exp_dirs)
    if args.ensemble_mode == "weighted":
        if args.weights is None or len(args.weights) != m:
            raise ValueError(
                f"--ensemble-mode weighted requires exactly {m} --weights (one per experiment), "
                f"got {args.weights!r}"
            )
        w_arr = np.array(args.weights, dtype=np.float64)
        if np.any(w_arr < 0) or w_arr.sum() <= 0:
            raise ValueError("--weights must be non-negative and not all zero.")
        w_arr = w_arr / w_arr.sum()
        weights_cpu = torch.tensor(w_arr, dtype=torch.float32)
        _log.info("Ensemble weights (normalized): %s", w_arr.tolist())
    elif args.ensemble_mode == "confidence":
        if args.weights is not None:
            _log.warning(
                "Ignoring --weights for --ensemble-mode confidence "
                "(per-pixel weights from max(P(fg), 1-P(fg)))."
            )
        weights_cpu = torch.full((m,), 1.0 / m, dtype=torch.float32)
    else:
        if args.weights is not None:
            _log.warning("Ignoring --weights because --ensemble-mode mean (equal 1/%d each).", m)
        weights_cpu = torch.full((m,), 1.0 / m, dtype=torch.float32)

    data_root = args.data_root.resolve()
    geojson_path = _resolve_geojson(data_root, args.geojson)
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ref_folds = discover_fold_dirs(exp_dirs[0])
    if not ref_folds:
        raise FileNotFoundError(f"No fold* or train/ under {exp_dirs[0]}")
    _assert_matching_folds(exp_dirs[1:], ref_folds)

    with open(geojson_path, encoding="utf-8") as f:
        all_tiles = [feat["properties"]["name"] for feat in json.load(f)["features"]]

    single_train_dir = len(ref_folds) == 1 and ref_folds[0][1].name == "train"
    if single_train_dir:
        k = 1
        train_ids, val_ids = train_test_split(
            all_tiles,
            test_size=float(args.val_fraction),
            random_state=args.seed,
            shuffle=True,
        )
        fold_runs: list[tuple[int, list[Path], list[str], list[str]]] = [
            (0, [ed / "train" for ed in exp_dirs], train_ids, val_ids)
        ]
    else:
        k = args.cv_folds if args.cv_folds is not None else len(ref_folds)
        kf = KFold(n_splits=k, shuffle=True, random_state=args.seed)
        splits = list(kf.split(range(len(all_tiles))))
        if len(splits) < len(ref_folds):
            raise ValueError(f"K-fold splits={len(splits)} but need at least {len(ref_folds)} folds.")
        fold_runs = []
        for fold_idx, _ in ref_folds:
            if fold_idx >= len(splits):
                continue
            tr_idx, va_idx = splits[fold_idx]
            fold_paths = [ed / f"fold{fold_idx}" for ed in exp_dirs]
            fold_runs.append(
                (
                    fold_idx,
                    fold_paths,
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

    thresh_grid = torch.linspace(args.thresh_min, args.thresh_max, args.thresh_steps)

    _log.info(
        "Ensemble %d experiments | mode=%s | K=%d | device=%s | threshold=[%.3f..%.3f] x%d",
        len(exp_dirs),
        args.ensemble_mode,
        k,
        device,
        args.thresh_min,
        args.thresh_max,
        args.thresh_steps,
    )
    for i, ed in enumerate(exp_dirs):
        _log.info("  exp[%d]: %s", i, ed)

    no_tta_scores: list[float] = []
    tta_scores: list[float] = []
    tta_opt_scores: list[float] = []
    best_thresholds: list[float] = []
    fold_rows: list[tuple[int, float, float, float, float]] = []

    for fold_idx, fold_paths, train_ids, val_ids in fold_runs:
        lits: list[SegLitModule] = []
        for fp in fold_paths:
            ckpt = find_best_checkpoint(fp)
            _log.info(
                "Fold %d: load %s from %s",
                fold_idx,
                ckpt.name,
                fp.resolve(),
            )
            lit = load_seg_lit_from_checkpoint(str(ckpt), map_location=device, strict=False)
            lits.append(lit)

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

        iou_no, iou_tta, iou_opt, best_t = evaluate_ensemble_fold(
            lits,
            val_loader,
            device,
            thresholds=thresh_grid,
            weights=weights_cpu,
            ensemble_mode=args.ensemble_mode,
        )
        no_tta_scores.append(iou_no)
        tta_scores.append(iou_tta)
        tta_opt_scores.append(iou_opt)
        best_thresholds.append(best_t)
        fold_rows.append((fold_idx, iou_no, iou_tta, iou_opt, best_t))

        _log.info(
            "Fold %d ensemble: no_tta=%.6f tta@0.5=%.6f tta@best_t=%.6f (best P(fg)=%.4f)",
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
            "Mean ensemble val_union_iou no_tta: %.6f (per-fold: %s)",
            sum(no_tta_scores) / len(no_tta_scores),
            [round(s, 6) for s in no_tta_scores],
        )
    if tta_scores:
        _log.info(
            "Mean ensemble val_union_iou tta@0.5: %.6f (per-fold: %s)",
            sum(tta_scores) / len(tta_scores),
            [round(s, 6) for s in tta_scores],
        )
    if tta_opt_scores:
        _log.info(
            "Mean ensemble val_union_iou tta@best_thresh: %.6f (per-fold: %s)",
            sum(tta_opt_scores) / len(tta_opt_scores),
            [round(s, 6) for s in tta_opt_scores],
        )
    if best_thresholds:
        _log.info(
            "Best P(fg) per fold (ensemble): %s (mean=%.4f)",
            [round(t, 4) for t in best_thresholds],
            sum(best_thresholds) / len(best_thresholds),
        )


if __name__ == "__main__":
    main()
