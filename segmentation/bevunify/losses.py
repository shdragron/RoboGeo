"""CVT center loss (the segmentation and GaussianLSS losses come from GaussianLSS.losses)."""
import torch
import torch.nn as nn
from fvcore.nn import sigmoid_focal_loss


class FocalCenterLoss(nn.Module):
    """Sigmoid focal loss on the raw center logits against the center heatmap, averaged
    over cells with visibility >= min_visibility (cross_view_transformer.losses.CenterLoss)."""

    def __init__(self, min_visibility=0, alpha=-1.0, gamma=2.0, key="vehicle"):
        super().__init__()
        self.min_visibility = min_visibility
        self.alpha = alpha
        self.gamma = gamma
        self.key = key

    def forward(self, pred_dict, batch, eps=1e-6):
        k = f"{self.key}_center"
        pred = pred_dict[k]
        target = batch[k]
        loss = sigmoid_focal_loss(pred, target, self.alpha, self.gamma, reduction="none")
        mask = torch.ones_like(target, dtype=torch.bool)
        if self.min_visibility > 0:
            vis_mask = batch[f"{self.key}_visibility"] >= self.min_visibility
            mask = mask * vis_mask[:, None]
        return (loss * mask).sum() / (mask.sum() + eps)
