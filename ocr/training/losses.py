"""Detector losses.

`bce_loss`  the original objective: per-pixel BCE against the full-box mask.
`DBLoss`    DBNet's objective (Liao et al., 2020) against the shrunk targets from
            ocr/data/targets.py. Three terms:

    L = L_prob + alpha * L_binary + beta * L_thresh

    L_prob    balanced BCE on the probability map P vs the shrunk mask. Hard
              negative mining keeps every text pixel but only the 3x hardest
              background pixels, so the sea of easy background cannot drown
              out the thin gaps between lines.
    L_binary  Dice loss on the approximate binary map
                  B = sigmoid(k * (P - T)),  k = 50
              vs the shrunk mask. Because B is computed from P AND the predicted
              threshold T, this term back-propagates through the binarization
              step ("differentiable binarization") and moves the boundary.
    L_thresh  L1 between the predicted threshold map T and its target, applied
              only inside the border bands (thresh_mask).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_loss(prob_logits: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
    loss = F.binary_cross_entropy_with_logits(prob_logits, gt)
    return {"loss": loss, "loss_prob": loss.detach()}


def balanced_bce(prob_logits: torch.Tensor, gt: torch.Tensor, negative_ratio: float = 3.0) -> torch.Tensor:
    """BCE over all positives plus the hardest `negative_ratio` x as many negatives."""
    positive = gt > 0.5
    negative = ~positive
    num_pos = int(positive.sum())
    num_neg = min(int(negative.sum()), int(num_pos * negative_ratio)) if num_pos else int(negative.sum())

    per_pixel = F.binary_cross_entropy_with_logits(prob_logits, gt, reduction="none")
    pos_loss = per_pixel[positive].sum()
    neg_losses = per_pixel[negative]
    if num_neg < neg_losses.numel():
        neg_losses = torch.topk(neg_losses, k=num_neg).values  # hardest negatives
    neg_loss = neg_losses.sum()
    return (pos_loss + neg_loss) / max(1, num_pos + num_neg)


def dice_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - 2|A∩B| / (|A| + |B|), on probabilities; region-based, so class balance is irrelevant."""
    intersection = (pred * gt).sum()
    return 1.0 - (2.0 * intersection + eps) / (pred.sum() + gt.sum() + eps)


class DBLoss:
    def __init__(self, alpha: float = 1.0, beta: float = 10.0, k: float = 50.0, negative_ratio: float = 3.0) -> None:
        self.alpha, self.beta, self.k, self.negative_ratio = alpha, beta, k, negative_ratio

    def __call__(
        self,
        prob_logits: torch.Tensor,
        thresh_logits: torch.Tensor,
        gt: torch.Tensor,
        thresh_map: torch.Tensor,
        thresh_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prob_logits = prob_logits.float()
        thresh_logits = thresh_logits.float()

        loss_prob = balanced_bce(prob_logits, gt, self.negative_ratio)

        prob = torch.sigmoid(prob_logits)
        thresh = torch.sigmoid(thresh_logits)
        binary = torch.sigmoid(self.k * (prob - thresh))
        loss_binary = dice_loss(binary, gt)

        band = thresh_mask.sum()
        loss_thresh = (torch.abs(thresh - thresh_map) * thresh_mask).sum() / band if band > 0 else thresh.sum() * 0.0

        total = loss_prob + self.alpha * loss_binary + self.beta * loss_thresh
        return {
            "loss": total,
            "loss_prob": loss_prob.detach(),
            "loss_binary": loss_binary.detach(),
            "loss_thresh": loss_thresh.detach(),
        }
