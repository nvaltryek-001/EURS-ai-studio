"""Procedural *synthetic* EURS-like data for smoke tests, CI and demos.

Clean scenes J with pixel-exact 5-class masks are rendered procedurally, then degraded
with the simplified underwater image formation model (Jaffe-McGlamery / Akkaynak-Treibitz):

    I_c(x) = J_c(x) * t_c(x) + B_c * (1 - t_c(x)),   t_c(x) = exp(-beta_c * z(x))

plus forward-scatter blur, non-uniform artificial lighting and sensor noise.
These images are NOT real EURS data; never report metrics on them as benchmark results.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .io import write_mask, write_rgb


def _noise(rng, h, w, scale):
    small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _fbm(rng, h, w, octaves=4, base=32):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        out += amp * _noise(rng, h, w, max(2, base // (2 ** o)))
        tot += amp
        amp *= 0.5
    return out / tot


def _paint(img, mask, region, color, cls, rng, tex=0.15):
    t = _fbm(rng, *region.shape, octaves=3, base=8)[..., None]
    col = np.clip(np.array(color, np.float32)[None, None] * (1 - tex + 2 * tex * t), 0, 1)
    img[region] = col[region]
    mask[region] = cls


def render_scene(rng: np.random.Generator, h: int = 240, w: int = 320):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Background: clear "in-air" colours of water column + sandy floor (class 0).
    horizon = rng.uniform(0.35, 0.65) * h + 12 * np.sin(xx / w * rng.uniform(2, 6) + rng.uniform(0, 6))
    water_top, water_bot = np.array([0.55, 0.75, 0.9]), np.array([0.35, 0.55, 0.75])
    a = (yy / h)[..., None]
    img = (water_top * (1 - a) + water_bot * a).astype(np.float32)
    floor = yy > horizon
    sand = np.array([0.86, 0.78, 0.6]) * (0.75 + 0.35 * _fbm(rng, h, w, 5, 24))[..., None]
    img[floor] = sand[floor]
    mask = np.zeros((h, w), np.uint8)

    # Other (class 4): rocks and kelp.
    for _ in range(rng.integers(1, 4)):
        cx, cy = rng.uniform(0, w), rng.uniform(horizon.mean(), h)
        r = rng.uniform(12, 40)
        ang = np.arctan2(yy - cy, xx - cx)
        rr = r * (1 + 0.25 * np.sin(3 * ang + rng.uniform(0, 6)) + 0.15 * np.sin(5 * ang))
        reg = np.hypot(xx - cx, (yy - cy) * 1.4) < rr
        g = rng.uniform(0.35, 0.55)
        _paint(img, mask, reg, [g, g * 0.95, g * 0.85], 4, rng, 0.3)
    for _ in range(rng.integers(0, 3)):
        x0 = rng.uniform(0, w)
        wave = x0 + 6 * np.sin(yy / rng.uniform(12, 25) + rng.uniform(0, 6))
        reg = (np.abs(xx - wave) < rng.uniform(2, 4)) & (yy > rng.uniform(0.1, 0.5) * h)
        _paint(img, mask, reg, [0.35, 0.6, 0.25], 4, rng, 0.2)

    # Coral (class 1): textured blobs anchored on the floor.
    palette = [[1.0, 0.45, 0.35], [0.95, 0.55, 0.75], [0.7, 0.4, 0.85], [1.0, 0.8, 0.3], [0.95, 0.6, 0.2]]
    for _ in range(rng.integers(1, 5)):
        cx, cy = rng.uniform(0, w), rng.uniform(horizon.mean() + 10, h + 10)
        n = _fbm(rng, h, w, 4, 16)
        d = np.hypot((xx - cx) / rng.uniform(25, 60), (yy - cy) / rng.uniform(18, 45))
        reg = (d + 0.6 * (n - 0.5)) < 1.0
        _paint(img, mask, reg, palette[rng.integers(len(palette))], 1, rng, 0.35)

    # Fish (class 2): ellipse body + triangular tail, some with stripes.
    for _ in range(rng.integers(0, 7)):
        cx, cy = rng.uniform(20, w - 20), rng.uniform(15, h - 15)
        L, H = rng.uniform(10, 28), 0
        H = L * rng.uniform(0.35, 0.55)
        d = 1 if rng.random() < 0.5 else -1
        th = rng.uniform(-0.3, 0.3)
        xr = (xx - cx) * math.cos(th) + (yy - cy) * math.sin(th)
        yr = -(xx - cx) * math.sin(th) + (yy - cy) * math.cos(th)
        body = (xr / L) ** 2 + (yr / H) ** 2 < 1
        tx = -d * xr - L * 0.8
        tail = (tx > 0) & (tx < L * 0.6) & (np.abs(yr) < tx * 0.9)
        reg = body | tail
        col = [[1.0, 0.85, 0.2], [0.8, 0.82, 0.85], [0.2, 0.45, 0.95], [1.0, 0.5, 0.15]][rng.integers(4)]
        _paint(img, mask, reg, col, 2, rng, 0.15)
        if rng.random() < 0.4:
            stripes = reg & (np.sin(xr / 2.5) > 0.6)
            img[stripes] *= 0.3

    # Elasmobranch (class 3): ray (kite + tail) or shark (spindle + fins).
    if rng.random() < 0.45:
        cx, cy = rng.uniform(40, w - 40), rng.uniform(20, h * 0.8)
        s = rng.uniform(22, 48)
        if rng.random() < 0.5:  # ray
            reg = (np.abs(xx - cx) / s + np.abs(yy - cy) / (0.6 * s)) < 1
            reg |= (np.abs(xx - cx) < 1.5) & (yy > cy) & (yy < cy + 1.6 * s)
            col = [0.45, 0.42, 0.38]
        else:  # shark
            xr, yr = (xx - cx) / (1.4 * s), (yy - cy) / (0.28 * s)
            reg = xr ** 2 + yr ** 2 < 1
            reg |= (yy < cy) & (yy > cy - 0.55 * s) & (np.abs(xx - cx) < (yy - (cy - 0.55 * s)) * 0.7)
            reg |= (xx > cx + 1.2 * s) & (xx < cx + 1.7 * s) & (np.abs(yy - cy) < (xx - cx - 1.2 * s) * 0.9)
            col = [0.55, 0.6, 0.66]
        _paint(img, mask, reg, col, 3, rng, 0.12)

    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    return np.clip(img, 0, 1), mask


def degrade(rng: np.random.Generator, clean: np.ndarray) -> np.ndarray:
    h, w, _ = clean.shape
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32) / h
    # Range map: farther at the top (open water), nearer at the bottom (floor), plus jitter.
    z = rng.uniform(1.5, 4.0) + rng.uniform(3, 10) * (1 - yy) ** 1.5 + 1.5 * _fbm(rng, h, w, 3, 64)
    beta = np.array([rng.uniform(0.25, 0.6), rng.uniform(0.06, 0.16), rng.uniform(0.04, 0.12)], np.float32)
    if rng.random() < 0.3:  # greener, turbid coastal water
        beta = beta * np.array([1.0, 0.8, 1.6], np.float32)
    t = np.exp(-beta[None, None] * z[..., None])
    B = np.array([rng.uniform(0.0, 0.15), rng.uniform(0.35, 0.6), rng.uniform(0.45, 0.75)], np.float32)
    light = 0.75 + 0.4 * np.exp(-(((np.mgrid[0:h, 0:w][1] / w) - rng.uniform(0.3, 0.7)) ** 2
                                  + (yy - rng.uniform(0.2, 0.6)) ** 2) / 0.12)[..., None]
    img = clean * t * light + B * (1 - t)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), rng.uniform(0.6, 1.6))
    img = img + rng.normal(0, rng.uniform(0.005, 0.02), img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def generate(out_dir: str | Path, n: int = 300, h: int = 240, w: int = 320, seed: int = 7) -> Path:
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    n_dives = max(1, n // 6)
    for i in range(n):
        dive = i % n_dives  # consecutive "frames" from one dive share an id prefix
        sid = f"dive{dive:03d}_{i // n_dives:04d}"
        clean, mask = render_scene(rng, h, w)
        deg = degrade(rng, clean)
        write_rgb(out / "degraded" / f"{sid}.png", (deg * 255).astype(np.uint8))
        write_rgb(out / "clean" / f"{sid}.png", (clean * 255).astype(np.uint8))
        write_mask(out / "masks" / f"{sid}.png", mask)
    (out / "SYNTHETIC_DATA_README.txt").write_text(
        "Procedurally generated smoke-test data. NOT the official EURS dataset.\n"
        "Do not report metrics computed on this data as benchmark results.\n")
    return out
