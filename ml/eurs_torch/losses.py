"""Multi-task loss: restoration (L1 + SSIM) and segmentation (weighted CE + soft Dice)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = 255


def _gauss(win=11, sigma=1.5, ch=3, device="cpu"):
    x = torch.arange(win, dtype=torch.float32, device=device) - win // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    return g.expand(ch, 1, win, win).contiguous()


def ssim(a, b, win=11):
    ch = a.shape[1]
    w = _gauss(win, ch=ch, device=a.device).to(a.dtype)
    mu_a = F.conv2d(a, w, padding=win // 2, groups=ch)
    mu_b = F.conv2d(b, w, padding=win // 2, groups=ch)
    s_aa = F.conv2d(a * a, w, padding=win // 2, groups=ch) - mu_a ** 2
    s_bb = F.conv2d(b * b, w, padding=win // 2, groups=ch) - mu_b ** 2
    s_ab = F.conv2d(a * b, w, padding=win // 2, groups=ch) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m.mean(dim=(1, 2, 3))


def dice_loss(logits, target, num_classes):
    valid = (target != IGNORE).unsqueeze(1).float()
    t = target.clone()
    t[t == IGNORE] = 0
    onehot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float() * valid
    p = logits.softmax(1) * valid
    inter = (p * onehot).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    present = onehot.sum((0, 2, 3)) > 0
    dice = (2 * inter + 1) / (denom + 1)
    return 1 - dice[present].mean() if present.any() else logits.sum() * 0


class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes=5, class_weights=None, w_rest=1.0, w_seg=1.0, w_ssim=0.2, w_dice=0.5):
        super().__init__()
        self.nc = num_classes
        self.register_buffer("cw", torch.as_tensor(class_weights, dtype=torch.float32)
                             if class_weights is not None else torch.ones(num_classes))
        self.w_rest, self.w_seg, self.w_ssim, self.w_dice = w_rest, w_seg, w_ssim, w_dice

    def forward(self, restored, logits, clean, mask, has_clean, has_mask):
        logs = {}
        total = restored.sum() * 0
        if has_clean.any():
            r, c = restored[has_clean], clean[has_clean]
            l1 = F.l1_loss(r, c)
            ls = (1 - ssim(r, c)).mean()
            total = total + self.w_rest * (l1 + self.w_ssim * ls)
            logs.update(l1=l1.item(), ssim_loss=ls.item())
        if has_mask.any():
            lg, m = logits[has_mask], mask[has_mask]
            if (m != IGNORE).any():
                ce = F.cross_entropy(lg, m, weight=self.cw, ignore_index=IGNORE)
                dl = dice_loss(lg, m, self.nc)
                total = total + self.w_seg * (ce + self.w_dice * dl)
                logs.update(ce=ce.item(), dice=dl.item())
        logs["total"] = total.item()
        return total, logs
