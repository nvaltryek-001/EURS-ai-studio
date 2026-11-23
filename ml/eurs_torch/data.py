"""PyTorch dataset over the split manifest (paired degraded / clean / mask, any may be missing)."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from eurs_ml.discover import read_manifest
from eurs_ml.io import MaskDecoder, read_rgb

IGNORE = 255


class EURSDataset(Dataset):
    def __init__(self, manifest: str | Path, root: str | Path, split: str, size: int = 384,
                 train: bool = True, decoder: MaskDecoder | None = None):
        self.rows = read_manifest(manifest, split)
        self.root = Path(root)
        self.size = size
        self.train = train
        self.decode = decoder or MaskDecoder()

    def __len__(self):
        return len(self.rows)

    def _crop_params(self, h, w):
        if not self.train:
            return 0, 0, h, w
        scale = random.uniform(0.5, 1.0)
        ch, cw = max(32, int(h * scale)), max(32, int(w * scale))
        return random.randint(0, h - ch), random.randint(0, w - cw), ch, cw

    def __getitem__(self, i):
        r = self.rows[i]
        x = read_rgb(self.root / r["input"])
        h, w = x.shape[:2]
        y = read_rgb(self.root / r["clean"]) if r["clean"] else None
        m = self.decode(self.root / r["mask"]) if r["mask"] else None
        if y is not None and y.shape[:2] != (h, w):
            y = cv2.resize(y, (w, h), interpolation=cv2.INTER_AREA)
        if m is not None and m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        t, l, ch, cw = self._crop_params(h, w)
        flip = self.train and random.random() < 0.5
        S = self.size

        def tf(img, interp):
            img = img[t:t + ch, l:l + cw]
            img = cv2.resize(img, (S, S), interpolation=interp)
            return img[:, ::-1] if flip else img

        x = tf(x, cv2.INTER_AREA)
        out = {"image": torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float() / 255,
               "has_clean": y is not None, "has_mask": m is not None, "id": r["id"]}
        y = tf(y, cv2.INTER_AREA) if y is not None else np.zeros_like(x)
        m = tf(m, cv2.INTER_NEAREST) if m is not None else np.full((S, S), IGNORE, np.uint8)
        out["clean"] = torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1))).float() / 255
        out["mask"] = torch.from_numpy(np.ascontiguousarray(m)).long()
        return out


def class_weights_from_fraction(frac: dict, power: float = 0.5, cap: float = 10.0) -> list[float]:
    """Inverse-frequency^power class weights, normalised to mean 1 and capped."""
    f = np.array([max(float(frac[str(i)]), 1e-6) for i in range(len(frac))])
    w = np.minimum((1 / f) ** power, cap * np.min((1 / f) ** power))
    return (w / w.mean()).tolist()
