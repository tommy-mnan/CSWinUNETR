"""Foreground batch Dice and cross-entropy loss, skeleton recall, and deep supervision."""

import torch
from torch.nn import functional as F


def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average Dice loss over present foreground classes, combined with cross entropy."""
    probabilities = logits.float().softmax(1)[:, 1:]
    labels = F.one_hot(target, logits.shape[1]).movedim(-1, 1)[:, 1:].float()
    dims = (0, *range(2, logits.ndim))
    intersection = (probabilities * labels).sum(dims)
    target_sum = labels.sum(dims)
    denominator = probabilities.sum(dims) + target_sum
    present = target_sum > 0
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    dice_loss = 1 - dice[present].mean() if present.any() else logits.sum() * 0
    return 0.5 * dice_loss + 0.5 * F.cross_entropy(logits.float(), target)


def skeleton_recall_loss(logits: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
    """Soft skeleton recall loss (Kirchhoff et al., ECCV 2024), as 1 - recall.

    `skeleton` holds the (tubed) skeleton of the ground truth with class labels, like the
    segmentation target. Recall is the share of skeleton pixels covered by the predicted
    foreground probability, computed over the whole batch (like `dice_ce_loss`) with
    smooth = 1e-5 as in the official nnU-Net trainer. Background is excluded. The official
    code returns -recall; 1 - recall has the same gradient and stays nonnegative.
    """
    probabilities = logits.float().softmax(1)[:, 1:]
    labels = F.one_hot(skeleton, logits.shape[1]).movedim(-1, 1)[:, 1:].float()
    dims = (0, *range(2, logits.ndim))
    recall = ((probabilities * labels).sum(dims) + 1e-5) / (labels.sum(dims) + 1e-5)
    return 1 - recall.mean()


def downsample_target(target: torch.Tensor, factor: int, method: str) -> torch.Tensor:
    """Downsample an integer label map (N, H, W) by an integer factor.

    "maxpool" keeps a label wherever any pixel of the factor x factor block has it, so
    1-2 px structures survive at every scale. "nearest" samples one pixel per block
    (nearest-exact, as nnU-Net's deep-supervision targets), which can drop thin lines.
    """
    if factor == 1:
        return target
    if method == "maxpool":
        return F.max_pool2d(target[:, None].float(), factor)[:, 0].long()
    if method == "nearest":
        size = [s // factor for s in target.shape[1:]]
        return F.interpolate(target[:, None].float(), size=size, mode="nearest-exact")[
            :, 0
        ].long()
    raise ValueError("method must be 'maxpool' or 'nearest'")


def deep_supervision_weights(levels: int) -> list[float]:
    """nnU-Net style weights 1, 1/2, 1/4, ... (full resolution first), normalised to sum 1."""
    weights = [0.5**level for level in range(levels)]
    return [w / sum(weights) for w in weights]


def training_loss(
    outputs,
    target: torch.Tensor,
    skeleton: torch.Tensor | None = None,
    *,
    srl_weight: float = 0.0,
    srl_scales: str = "final",
    downsample: str = "maxpool",
) -> torch.Tensor:
    """Total training loss for every configuration of the study.

    loss = sum_i w_i * DiceCE_i + srl_weight * sum_j v_j * SRL_j

    `outputs` is one logits tensor, or a list ordered from full resolution to coarsest
    (deep supervision). Level i is supervised at its own resolution with targets
    downsampled by `downsample`. w are the deep-supervision weights over all levels. SRL is
    applied to the final output only (`srl_scales="final"`, v = [1]) or to every level
    (`"all"`, v = w), so the Dice+CE terms and the SRL terms each keep a total weight of 1
    and `srl_weight` respectively in every configuration.
    """
    outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    weights = deep_supervision_weights(len(outputs))
    use_srl = srl_weight > 0
    if use_srl and skeleton is None:
        raise ValueError("skeleton targets are required when srl_weight > 0")
    if srl_scales not in ("final", "all"):
        raise ValueError("srl_scales must be 'final' or 'all'")
    full_size = target.shape[-1]
    total = outputs[0].new_zeros((), dtype=torch.float32)
    for level, (logits, weight) in enumerate(zip(outputs, weights)):
        if full_size % logits.shape[-1]:
            raise ValueError("Side outputs must divide the target size")
        factor = full_size // logits.shape[-1]
        total = total + weight * dice_ce_loss(logits, downsample_target(target, factor, downsample))
        if use_srl and (level == 0 or srl_scales == "all"):
            srl_share = 1.0 if srl_scales == "final" else weight
            level_skeleton = downsample_target(skeleton, factor, downsample)
            total = total + srl_weight * srl_share * skeleton_recall_loss(logits, level_skeleton)
    return total
