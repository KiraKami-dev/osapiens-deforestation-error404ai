"""Per-image union IoU (binary Jaccard on foreground) in pure PyTorch."""

from __future__ import annotations

import torch


def _binary_union_iou(
    pred_fg: torch.Tensor,
    tgt_fg: torch.Tensor,
    *,
    zero_division: float = 1.0,
) -> torch.Tensor:
    """Scalar IoU from boolean or 0/1 masks of equal shape: :math:`|P \\cap G| / |P \\cup G|`."""
    p = pred_fg.float().reshape(-1)
    t = tgt_fg.float().reshape(-1)
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    if union <= 0:
        return torch.tensor(float(zero_division), dtype=p.dtype, device=p.device)
    return inter / union


def mean_union_iou_batch(
    pred_labels: torch.Tensor,
    gt_labels: torch.Tensor,
    *,
    foreground_class: int = 1,
    zero_division: float = 1.0,
) -> float:
    """Mean per-crop union IoU (Jaccard) for foreground vs. ground truth."""

    if pred_labels.ndim != 3 or gt_labels.ndim != 3:
        raise ValueError("Expected pred and gt with shape (B, H, W)")
    b = pred_labels.shape[0]
    parts = [
        _binary_union_iou(
            pred_labels[i].eq(foreground_class),
            gt_labels[i].eq(foreground_class),
            zero_division=zero_division,
        )
        for i in range(b)
    ]
    return torch.stack(parts).mean().item()


def mean_union_iou_batch_ignore(
    pred_labels: torch.Tensor,
    gt_labels: torch.Tensor,
    *,
    ignore_index: int = 255,
    foreground_class: int = 1,
    zero_division: float = 1.0,
) -> float:
    """Mean per-crop union IoU on foreground, ignoring ``ignore_index`` in the target."""

    if pred_labels.ndim != 3 or gt_labels.ndim != 3:
        raise ValueError("Expected pred and gt with shape (B, H, W)")
    b = pred_labels.shape[0]
    parts: list[torch.Tensor] = []
    for i in range(b):
        valid = gt_labels[i] != ignore_index
        if not valid.any():
            continue
        pred_fg = pred_labels[i].eq(foreground_class) & valid
        tgt_fg = gt_labels[i].eq(foreground_class) & valid
        parts.append(_binary_union_iou(pred_fg, tgt_fg, zero_division=zero_division))
    if not parts:
        return float(zero_division)
    return torch.stack(parts).mean().item()
