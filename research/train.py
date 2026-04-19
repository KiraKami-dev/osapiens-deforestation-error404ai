#!/usr/bin/env python3
"""Train segmentation on GeoTIFF tiles: compact U-Net (see ``utils.dataset``)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from models.unet import UNet
from utils.dataset import OsapiensTerraDataset
from utils.metrics import mean_union_iou_batch_ignore

_log = logging.getLogger(__name__)


def _log_input_norm_tensorboard(
    logger: TensorBoardLogger,
    *,
    input_norm: str,
    mn: list[float] | None = None,
    mx: list[float] | None = None,
    zm: list[float] | None = None,
    zs: list[float] | None = None,
) -> None:
    """Record train-computed normalization stats under ``input_norm``."""
    exp = logger.experiment
    tag = "input_norm"
    if input_norm == "zscore" and zm is not None and zs is not None:
        for i in range(len(zm)):
            exp.add_scalar(f"{tag}/ch{i:02d}_mean", zm[i], 0)
            exp.add_scalar(f"{tag}/ch{i:02d}_std", zs[i], 0)
        payload = json.dumps({"source": "train_tiles_zscore", "mean": zm, "std": zs}, indent=2)
    elif input_norm == "minmax" and mn is not None and mx is not None:
        for i in range(len(mn)):
            exp.add_scalar(f"{tag}/ch{i:02d}_min", mn[i], 0)
            exp.add_scalar(f"{tag}/ch{i:02d}_max", mx[i], 0)
        payload = json.dumps({"source": "train_tiles_minmax", "min": mn, "max": mx}, indent=2)
    else:
        payload = "{}"
    exp.add_text(f"{tag}/computed", payload, 0)


class BCEDiceLoss(nn.Module):
    """Binary segmentation: weighted BCE-with-logits + soft Dice on valid pixels (target 255 = ignore)."""

    def __init__(self, pos_weight: float = 5.0, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        valid = targets < 255
        if not valid.any():
            return logits.sum() * 0.0
        lv = logits[valid]
        tv = targets[valid].float().clamp(0.0, 1.0)
        pw = self.pos_weight.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(lv, tv, pos_weight=pw)
        p = torch.sigmoid(logits) * valid.float()
        t = torch.where(valid, targets.float().clamp(0.0, 1.0), torch.zeros_like(logits))
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dice_loss = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
        return (1.0 - self.dice_weight) * bce + self.dice_weight * dice_loss


def _count_params(module: nn.Module, *, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _build_segmentation_net(in_channels: int, num_classes: int, base_channels: int) -> nn.Module:
    return UNet(in_channels, num_classes, base_channels=base_channels)


def load_seg_lit_from_checkpoint(
    ckpt_path: str | Path,
    map_location: torch.device | str | None = None,
    *,
    strict: bool = False,
) -> SegLitModule:
    """Load ``SegLitModule`` from a checkpoint path."""
    return SegLitModule.load_from_checkpoint(
        str(ckpt_path),
        map_location=map_location,
        strict=strict,
    )


class SegLitModule(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        lr: float,
        *,
        num_classes: int = 1,
        base_channels: int = 64,
        pad_multiple: int = 16,
        input_divisor: float = 1.0,
        pos_weight: float = 5.0,
        dice_weight: float = 0.5,
        input_norm: str = "zscore",
        lr_plateau_patience: int = 10,
        lr_plateau_factor: float = 0.5,
        lr_plateau_min: float = 1e-6,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs"])
        self.net = _build_segmentation_net(in_channels, num_classes, base_channels)
        self.criterion = BCEDiceLoss(pos_weight=pos_weight, dice_weight=dice_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hparams.input_divisor != 1.0:
            x = x / self.hparams.input_divisor
        _, _, h, w = x.shape
        pm = int(self.hparams.pad_multiple)
        pad_h = (pm - h % pm) % pm
        pad_w = (pm - w % pm) % pm
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            logits = self.net(x)
            return logits[..., :h, :w]
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
        )
        nc = int(self.hparams.num_classes)
        if nc == 1:
            pred = (logits.sigmoid() > 0.5).squeeze(1).long()
        else:
            pred = logits.argmax(dim=1)
        batch_iou = mean_union_iou_batch_ignore(pred.detach(), y.detach())
        bs = x.size(0)
        self._val_union_iou_sum += batch_iou * bs
        self._val_union_iou_count += bs
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_union_iou_sum = 0.0
        self._val_union_iou_count = 0

    def on_validation_epoch_end(self) -> None:
        if self._val_union_iou_count > 0:
            self.log(
                "val_union_iou",
                self._val_union_iou_sum / self._val_union_iou_count,
                prog_bar=True,
                sync_dist=False,
            )

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = ReduceLROnPlateau(
            opt,
            mode="max",
            factor=float(self.hparams.lr_plateau_factor),
            patience=int(self.hparams.lr_plateau_patience),
            min_lr=float(self.hparams.lr_plateau_min),
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_union_iou",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_start(self) -> None:
        tr = _count_params(self.net, trainable_only=True)
        tot = _count_params(self.net, trainable_only=False)
        _log.info("Model params: trainable=%s total=%s", f"{tr:,}", f"{tot:,}")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="U-Net segmentation (K-fold CV, Adam optimizer)")
    p.add_argument(
        "--data-root",
        type=Path,
        default=repo / "data",
        help="Dataset root with slim layout: features/{tile}_slim.tif, labels/{tile}_fused.tif.",
    )
    p.add_argument(
        "--geojson",
        type=Path,
        default=None,
        help="Tile list (properties.name). Default: data-root/metadata/train_tiles.geojson or train_tiles.geojson.",
    )
    p.add_argument("--out-dir", type=Path, default=repo / "exps")
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        metavar="K",
        help="Number of folds for cross-validation (default: 5). Checkpoints: out-dir/checkpoints/version_N/fold{0..K-1}/.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Train DataLoader batch size (random --crop-size crops batched together).",
    )
    p.add_argument("--num-workers", type=int, default=0, help="Use 0 if dataset keeps tensors on GPU.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--lr-plateau-patience",
        type=int,
        default=10,
        metavar="N",
        help="ReduceLROnPlateau: epochs with no val_union_iou improvement before LR *= factor.",
    )
    p.add_argument(
        "--lr-plateau-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau: multiply LR by this when plateau triggers.",
    )
    p.add_argument(
        "--lr-plateau-min",
        type=float,
        default=1e-6,
        help="ReduceLROnPlateau: minimum learning rate.",
    )
    p.add_argument("--crop-size", type=int, default=256, help="Random crop side length.")
    p.add_argument(
        "--crops-per-tile",
        type=int,
        default=128,
        help="Random crops per tile per epoch.",
    )
    p.add_argument(
        "--input-divisor",
        type=float,
        default=1.0,
        help="Divide inputs in the model after dataset normalization (e.g. 10000 for raw uint16). Use 1 after scaling.",
    )
    p.add_argument(
        "--input-norm",
        choices=("zscore", "minmax"),
        default="zscore",
        help="Per-band z-score (train-fit) or min–max to [0,1]. Fit uses training tiles only.",
    )
    p.add_argument(
        "--dice-weight",
        type=float,
        default=0.5,
        help="Weight of Dice term in BCE+Dice loss (0 = BCE only).",
    )
    p.add_argument("--base-channels", type=int, default=64, help="UNet base channel width.")
    p.add_argument(
        "--pad-multiple",
        type=int,
        default=0,
        metavar="P",
        help="Replicate-pad H,W to a multiple of P before the net (crop after). 0 = auto (16).",
    )
    p.add_argument("--early-stop-patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_one_fold(
    args: argparse.Namespace,
    geojson_path: Path,
    train_ids: list[str],
    val_ids: list[str],
    run_dir: Path,
    *,
    tb_name: str,
) -> tuple[float | None, Path | None]:
    """Train one split. Normalization is fit on ``train_ids`` only (val tiles excluded).

    Validation is **full-tile** (one batch item per val tile; use ``batch_size=1``).
    """
    ds_kw = dict(
        crop_size=args.crop_size,
        crops_per_tile=args.crops_per_tile,
        input_norm=args.input_norm,
    )
    fit_ds = OsapiensTerraDataset(
        args.data_root,
        geojson_path,
        tile_ids=train_ids,
        augment_flip=False,
        mode="crop",
        **ds_kw,
    )
    if args.input_norm == "zscore":
        zm_list, zs_list = fit_ds.fit_zscore_from_tiles()
        _log.info("Input z-score from train tiles: mean=%s std=%s", zm_list, zs_list)
        norm_lists = (zm_list, zs_list)
    else:
        mn_list, mx_list = fit_ds.fit_minmax_from_tiles()
        _log.info("Input min/max from train tiles: min=%s max=%s", mn_list, mx_list)
        norm_lists = (mn_list, mx_list)

    train_ds = OsapiensTerraDataset(
        args.data_root,
        geojson_path,
        tile_ids=train_ids,
        augment_flip=True,
        mode="crop",
        **ds_kw,
    )
    if args.input_norm == "zscore":
        train_ds.set_zscore(fit_ds._zs_mean, fit_ds._zs_std)
    else:
        train_ds.set_minmax(fit_ds._mm_min, fit_ds._mm_max)

    val_ds = OsapiensTerraDataset(
        args.data_root,
        geojson_path,
        tile_ids=val_ids,
        augment_flip=False,
        mode="full",
        **ds_kw,
    )
    if args.input_norm == "zscore":
        val_ds.set_zscore(fit_ds._zs_mean, fit_ds._zs_std)
    else:
        val_ds.set_minmax(fit_ds._mm_min, fit_ds._mm_max)

    if getattr(train_ds, "feature_band_descriptions", None):
        _log.info("Feature bands (first tile): %s", train_ds.feature_band_descriptions)

    pos_weight = fit_ds.estimate_bce_pos_weight()
    _log.info(
        "BCE pos_weight=%.4f (valid neg / valid pos on train tiles; clamped 1..100)",
        pos_weight,
    )
    train_bs = args.batch_size
    _log.info(
        "Training: random crops %d×%d (%d steps/epoch), augment_flip=True",
        args.crop_size,
        args.crop_size,
        len(train_ds),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    x0, _ = train_ds[0]
    pad_multiple = int(args.pad_multiple)
    if pad_multiple == 0:
        pad_multiple = 16
    lit_kw: dict = dict(
        in_channels=x0.shape[0],
        lr=args.lr,
        num_classes=1,
        base_channels=args.base_channels,
        pad_multiple=pad_multiple,
        input_divisor=args.input_divisor,
        pos_weight=pos_weight,
        dice_weight=args.dice_weight,
        input_norm=args.input_norm,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor,
        lr_plateau_min=args.lr_plateau_min,
    )
    lit = SegLitModule(**lit_kw)
    epochs = args.epochs
    tb_logger = TensorBoardLogger(str(args.out_dir), name=tb_name)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best-{epoch:02d}-{val_union_iou:.4f}",
        monitor="val_union_iou",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early = EarlyStopping(
        monitor="val_union_iou",
        patience=args.early_stop_patience,
        mode="max",
    )
    if args.input_norm == "zscore":
        _log_input_norm_tensorboard(
            tb_logger, input_norm="zscore", zm=norm_lists[0], zs=norm_lists[1]
        )
    else:
        _log_input_norm_tensorboard(
            tb_logger, input_norm="minmax", mn=norm_lists[0], mx=norm_lists[1]
        )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[ckpt_cb, early],
        logger=tb_logger,
        default_root_dir=str(run_dir),
        log_every_n_steps=10,
    )
    trainer.fit(lit, train_loader, val_loader)
    best = ckpt_cb.best_model_score
    best_path = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else None
    _log.info("Fold run finished. Best val_union_iou: %s (under %s)", best, run_dir.resolve())
    return (float(best) if best is not None else None), best_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args.data_root = args.data_root.resolve()

    def _resolve_geojson() -> Path:
        if args.geojson is not None:
            if not args.geojson.is_file():
                raise FileNotFoundError(f"--geojson not found: {args.geojson}")
            return args.geojson
        for cand in (args.data_root / "metadata" / "train_tiles.geojson", args.data_root / "train_tiles.geojson"):
            if cand.is_file():
                return cand
        raise FileNotFoundError(
            f"No train_tiles.geojson under {args.data_root}. Pass --geojson or add metadata/train_tiles.geojson."
        )

    geojson_path = _resolve_geojson()

    pl.seed_everything(args.seed, workers=True)

    with open(geojson_path, encoding="utf-8") as f:
        all_tiles = [feat["properties"]["name"] for feat in json.load(f)["features"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ck_root = args.out_dir / "checkpoints"
    nums: list[int] = []
    if ck_root.is_dir():
        for p in ck_root.iterdir():
            if p.is_dir() and p.name.startswith("version_"):
                try:
                    nums.append(int(p.name.removeprefix("version_")))
                except ValueError:
                    pass
    run_ver = max(nums, default=-1) + 1
    version_name = f"version_{run_ver}"
    checkpoint_version_root = args.out_dir / "checkpoints" / version_name
    checkpoint_version_root.mkdir(parents=True, exist_ok=True)
    _log.info(
        "Run output: %s — checkpoints in checkpoints/%s/; TensorBoard logs under %s/tensorboard/fold*/version_*",
        version_name,
        version_name,
        args.out_dir.resolve(),
    )

    k = int(args.cv_folds)
    if k < 2:
        raise ValueError("--cv-folds must be >= 2")
    if len(all_tiles) < k:
        raise ValueError(f"Need at least {k} tiles for {k}-fold CV, got {len(all_tiles)}")
    _log.info(
        "K-fold CV: K=%d tiles=%d epochs=%d batch_size=%d input_norm=%s data_root=%s out_dir=%s",
        k,
        len(all_tiles),
        args.epochs,
        args.batch_size,
        args.input_norm,
        args.data_root,
        args.out_dir,
    )
    kf = KFold(n_splits=k, shuffle=True, random_state=args.seed)
    fold_scores: list[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(all_tiles)))):
        train_ids = [all_tiles[i] for i in train_idx]
        val_ids = [all_tiles[i] for i in val_idx]
        _log.info(
            "Fold %d/%d: %d train tiles, %d val tiles",
            fold_idx + 1,
            k,
            len(train_ids),
            len(val_ids),
        )
        run_dir = checkpoint_version_root / f"fold{fold_idx}"
        best, _ = run_one_fold(
            args,
            geojson_path,
            train_ids,
            val_ids,
            run_dir,
            tb_name=f"tensorboard/fold{fold_idx}",
        )
        if best is not None:
            fold_scores.append(best)
    if fold_scores:
        mean_iou = sum(fold_scores) / len(fold_scores)
        _log.info(
            "CV finished. Per-fold best val_union_iou: %s | mean=%.6f",
            [round(s, 6) for s in fold_scores],
            mean_iou,
        )
    else:
        _log.warning("CV finished but no fold reported a best score.")
    _log.info(
        "Checkpoints & trainer files under %s (fold0 … fold%d)",
        checkpoint_version_root.resolve(),
        k - 1,
    )
    _log.info(
        "TensorBoard: tensorboard --logdir %s",
        (args.out_dir / "tensorboard").resolve(),
    )


if __name__ == "__main__":
    main()
