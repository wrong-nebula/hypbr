"""Optional-PyTorch compute backend for the procedural pipeline.

PyTorch is an optional extra (``pip install "hypbr[gpu]"`` / ``"hypbr[neural]"``). When it is
importable, hypbr runs the exact same torch kernels as before (so output is unchanged and
CUDA/MPS acceleration still applies). Without it, the NumPy implementations below reproduce the
torch semantics (``F.pad`` modes, ``conv2d`` cross-correlation, ``max_pool2d``, ``interpolate``
and ``grid_sample``) to float32 rounding.

Set ``HYPBR_TORCH=0`` to force the NumPy path even when torch is installed (used by the
torch-free test job and handy for A/B checks).
"""
from __future__ import annotations

import functools
import os

import numpy as np

_TORCH = None
_TORCH_TRIED = False
_DISABLED_VALUES = ("0", "false", "no", "off")


def torch_disabled_by_env() -> bool:
    return os.environ.get("HYPBR_TORCH", "").strip().lower() in _DISABLED_VALUES


def get_torch():
    """Return the `torch` module, or `None` when it is not installed (or `HYPBR_TORCH=0`)."""
    global _TORCH, _TORCH_TRIED
    if not _TORCH_TRIED:
        _TORCH_TRIED = True
        if not torch_disabled_by_env():
            try:
                import torch

                _TORCH = torch
            except (ImportError, OSError):
                _TORCH = None
    return _TORCH


def has_torch() -> bool:
    return get_torch() is not None


def require_torch(feature: str = "This feature"):
    """Return `torch` or raise an `ImportError` explaining how to install it."""
    torch = get_torch()
    if torch is None:
        why = " (disabled by HYPBR_TORCH=0)" if torch_disabled_by_env() else ""
        raise ImportError(
            f"{feature} requires PyTorch{why}. Install it with `pip install \"hypbr[neural]\"` "
            "(or `pip install torch`). Procedural synthesis (`hypbr synth`) works without it."
        )
    return torch


def no_grad(fn):
    """`torch.no_grad()` decorator that is a no-op when torch is absent (resolved at call time)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        torch = get_torch()
        if torch is None:
            return fn(*args, **kwargs)
        with torch.no_grad():
            return fn(*args, **kwargs)

    return wrapper


class CpuDevice:
    """Minimal stand-in for `torch.device("cpu")` when torch is not installed."""

    type = "cpu"
    index = None

    def __repr__(self) -> str:
        return "device(type='cpu')"

    def __str__(self) -> str:
        return "cpu"

    def __eq__(self, other) -> bool:
        return str(other) == "cpu"

    def __hash__(self) -> int:
        return hash("cpu")


def resolve_device():
    """Pick the compute device: `HYPBR_DEVICE` override, else CUDA > MPS > CPU (CPU without torch)."""
    torch = get_torch()
    env = os.environ.get("HYPBR_DEVICE", "").strip().lower()
    if torch is None:
        return CpuDevice()
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# NumPy equivalents of the torch ops used on the procedural path
# =============================================================================
_NP_PAD_MODE = {"reflect": "reflect", "circular": "wrap", "replicate": "edge", "constant": "constant"}


def pad2d(a: np.ndarray, left: int, right: int, top: int, bottom: int, mode: str) -> np.ndarray:
    """`F.pad(t[None, None], (left, right, top, bottom), mode)` on the last two axes of `a`."""
    torch = get_torch()
    if torch is not None and a.ndim == 2:
        F = torch.nn.functional
        return F.pad(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None],
                     (left, right, top, bottom), mode=mode)[0, 0].numpy()
    width = [(0, 0)] * (a.ndim - 2) + [(top, bottom), (left, right)]
    return np.pad(np.asarray(a, dtype=np.float32), width, mode=_NP_PAD_MODE[mode])


def correlate3x3(a: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    """Pad by 1 with `mode`, then `F.conv2d` (cross-correlation) with a 3x3 `kernel` (float32)."""
    torch = get_torch()
    k = np.asarray(kernel, dtype=np.float32)
    if torch is not None:
        F = torch.nn.functional
        t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
        t = F.pad(t, (1, 1, 1, 1), mode=mode)
        return F.conv2d(t, torch.from_numpy(k).view(1, 1, 3, 3))[0, 0].numpy()
    p = pad2d(a, 1, 1, 1, 1, mode)
    H, W = a.shape
    out = np.zeros((H, W), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            w = k[dy, dx]
            if w != 0.0:
                out += w * p[dy:dy + H, dx:dx + W]
    return out


def separable_conv(arr: np.ndarray, ker: np.ndarray, circular: bool, device=None) -> np.ndarray:
    """Horizontal-then-vertical 1D correlation with `ker` (odd length) and reflect/circular padding."""
    ker = np.asarray(ker, dtype=np.float32)
    rad = (ker.shape[0] - 1) // 2
    mode = "circular" if circular else "reflect"
    torch = get_torch()
    if torch is not None:
        F = torch.nn.functional
        dev = device if device is not None else torch.device("cpu")
        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(dev)[None, None]
        k_h = torch.from_numpy(ker).to(dev)[None, None, None, :]
        k_v = torch.from_numpy(ker).to(dev)[None, None, :, None]
        t = F.pad(t, (rad, rad, 0, 0), mode=mode)
        t = F.conv2d(t, k_h)
        t = F.pad(t, (0, 0, rad, rad), mode=mode)
        t = F.conv2d(t, k_v)
        return t[0, 0].cpu().numpy()
    a = np.asarray(arr, dtype=np.float32)
    H, W = a.shape
    np_mode = _NP_PAD_MODE[mode]
    p = np.pad(a, ((0, 0), (rad, rad)), mode=np_mode)
    tmp = ker[0] * p[:, 0:W]
    for i in range(1, ker.shape[0]):
        tmp += ker[i] * p[:, i:i + W]
    p = np.pad(tmp, ((rad, rad), (0, 0)), mode=np_mode)
    out = ker[0] * p[0:H]
    for i in range(1, ker.shape[0]):
        out += ker[i] * p[i:i + H]
    return out


def max_filter(arr: np.ndarray, radius: int, mode: str) -> np.ndarray:
    """`F.max_pool2d(F.pad(t, (r, r, r, r), mode), 2r+1, stride=1)` (separable, exact)."""
    k = 2 * radius + 1
    torch = get_torch()
    if torch is not None:
        F = torch.nn.functional
        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
        return F.max_pool2d(F.pad(t, (radius, radius, radius, radius), mode=mode), k, stride=1)[0, 0].numpy()
    a = np.asarray(arr, dtype=np.float32)
    H, W = a.shape
    p = pad2d(a, radius, radius, radius, radius, mode)
    rows = p[:, 0:W].copy()
    for i in range(1, k):
        np.maximum(rows, p[:, i:i + W], out=rows)
    out = rows[0:H].copy()
    for i in range(1, k):
        np.maximum(out, rows[i:i + H], out=out)
    return out


def _cubic_weights(t: np.ndarray, A: float = -0.75):
    """PyTorch `upsample_bicubic2d` / `grid_sample(mode="bicubic")` Keys-cubic tap weights."""
    x1 = t + 1.0
    w0 = ((A * x1 - 5.0 * A) * x1 + 8.0 * A) * x1 - 4.0 * A
    w1 = ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0
    x2 = 1.0 - t
    w2 = ((A + 2.0) * x2 - (A + 3.0)) * x2 * x2 + 1.0
    x3 = 2.0 - t
    w3 = ((A * x3 - 5.0 * A) * x3 + 8.0 * A) * x3 - 4.0 * A
    return w0, w1, w2, w3


def resize(g: np.ndarray, H: int, W: int, mode: str) -> np.ndarray:
    """`F.interpolate(t[None, None], size=(H, W), mode=mode, align_corners=False)` for 2D `g`."""
    torch = get_torch()
    if torch is not None:
        F = torch.nn.functional
        return F.interpolate(torch.from_numpy(np.ascontiguousarray(g, dtype=np.float32))[None, None],
                             size=(H, W), mode=mode, align_corners=False)[0, 0].numpy()
    g = np.asarray(g, dtype=np.float32)
    h_in, w_in = g.shape
    if mode == "bilinear":
        def axis(n_out: int, n_in: int):
            src = np.float32(n_in / n_out) * (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) - np.float32(0.5)
            src = np.maximum(src, np.float32(0.0))
            i0 = src.astype(np.int64)
            i1 = np.minimum(i0 + 1, n_in - 1)
            l1 = (src - i0).astype(np.float32)
            return i0, i1, (np.float32(1.0) - l1), l1

        y0, y1, wy0, wy1 = axis(H, h_in)
        x0, x1, wx0, wx1 = axis(W, w_in)
        top = g[y0] * wy0[:, None] + g[y1] * wy1[:, None]
        return (top[:, x0] * wx0[None, :] + top[:, x1] * wx1[None, :]).astype(np.float32)
    if mode == "bicubic":
        def axis(n_out: int, n_in: int):
            src = np.float32(n_in / n_out) * (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) - np.float32(0.5)
            i = np.floor(src).astype(np.int64)
            t = (src - i).astype(np.float32)
            idx = [np.clip(i + o, 0, n_in - 1) for o in (-1, 0, 1, 2)]
            return idx, _cubic_weights(t)

        iy, wy = axis(H, h_in)
        ix, wx = axis(W, w_in)
        rows = sum(g[iy[j]] * wy[j][:, None] for j in range(4))
        return sum(rows[:, ix[j]] * wx[j][None, :] for j in range(4)).astype(np.float32)
    raise ValueError(f"Unsupported resize mode {mode!r}")


def _reflect_coords_ac(x: np.ndarray, size: int) -> np.ndarray:
    """`reflect_coordinates(x, 0, 2*(size-1))` + clip (grid_sample, align_corners=True)."""
    if size <= 1:
        return np.zeros_like(x)
    span = np.float32(size - 1)
    x = np.abs(x)
    extra = np.fmod(x, span)
    flips = np.floor(x / span)
    out = np.where(np.fmod(flips, 2.0) == 0.0, extra, span - extra)
    return np.clip(out, 0.0, span)


def grid_sample(img: np.ndarray, gx: np.ndarray, gy: np.ndarray, mode: str = "bilinear",
                padding_mode: str = "border", align_corners: bool = True) -> np.ndarray:
    """`F.grid_sample(img[None], stack([gx, gy], -1)[None], ...)` for `img` of shape `(H, W)` or `(C, H, W)`.

    Supports `bilinear` (border/reflection, align_corners=True) and `bicubic` (border, align_corners=False),
    the combinations used by hypbr.
    """
    torch = get_torch()
    squeeze = img.ndim == 2
    torch_img = np.ascontiguousarray(img[None] if squeeze else img, dtype=np.float32)
    if torch is not None:
        F = torch.nn.functional
        grid = torch.from_numpy(np.stack([gx, gy], axis=-1)[None]).float()
        out = F.grid_sample(torch.from_numpy(torch_img)[None], grid, mode=mode,
                            padding_mode=padding_mode, align_corners=align_corners)[0].numpy()
        return out[0] if squeeze else out
    _, h_in, w_in = torch_img.shape
    gx = np.asarray(gx, dtype=np.float32)
    gy = np.asarray(gy, dtype=np.float32)
    if mode == "bilinear":
        if not align_corners:
            raise NotImplementedError("bilinear grid_sample fallback implements align_corners=True only")
        ix = (gx + np.float32(1.0)) / np.float32(2.0) * np.float32(w_in - 1)
        iy = (gy + np.float32(1.0)) / np.float32(2.0) * np.float32(h_in - 1)
        if padding_mode == "border":
            ix = np.clip(ix, 0.0, w_in - 1)
            iy = np.clip(iy, 0.0, h_in - 1)
        elif padding_mode == "reflection":
            ix = _reflect_coords_ac(ix, w_in)
            iy = _reflect_coords_ac(iy, h_in)
        else:
            raise NotImplementedError(f"padding_mode {padding_mode!r}")
        x0 = np.floor(ix)
        y0 = np.floor(iy)
        tx = (ix - x0).astype(np.float32)
        ty = (iy - y0).astype(np.float32)
        x0 = x0.astype(np.int64)
        y0 = y0.astype(np.int64)
        x1 = np.minimum(x0 + 1, w_in - 1)
        y1 = np.minimum(y0 + 1, h_in - 1)
        w_nw = (1.0 - tx) * (1.0 - ty)
        w_ne = tx * (1.0 - ty)
        w_sw = (1.0 - tx) * ty
        w_se = tx * ty
        out = (torch_img[:, y0, x0] * w_nw + torch_img[:, y0, x1] * w_ne
               + torch_img[:, y1, x0] * w_sw + torch_img[:, y1, x1] * w_se).astype(np.float32)
        return out[0] if squeeze else out
    if mode == "bicubic":
        if align_corners or padding_mode != "border":
            raise NotImplementedError("bicubic grid_sample fallback implements border/align_corners=False only")
        ix = ((gx + np.float32(1.0)) * np.float32(w_in) - np.float32(1.0)) / np.float32(2.0)
        iy = ((gy + np.float32(1.0)) * np.float32(h_in) - np.float32(1.0)) / np.float32(2.0)
        x0 = np.floor(ix)
        y0 = np.floor(iy)
        wx = _cubic_weights((ix - x0).astype(np.float32))
        wy = _cubic_weights((iy - y0).astype(np.float32))
        x0 = x0.astype(np.int64)
        y0 = y0.astype(np.int64)
        out = np.zeros((torch_img.shape[0],) + gx.shape, dtype=np.float32)
        for j, oy in enumerate((-1, 0, 1, 2)):
            yy = np.clip(y0 + oy, 0, h_in - 1)
            row = np.zeros_like(out)
            for i, ox in enumerate((-1, 0, 1, 2)):
                xx = np.clip(x0 + ox, 0, w_in - 1)
                row += torch_img[:, yy, xx] * wx[i]
            out += row * wy[j]
        return out[0] if squeeze else out
    raise ValueError(f"Unsupported grid_sample mode {mode!r}")


__all__ = [
    "CpuDevice", "correlate3x3", "get_torch", "grid_sample", "has_torch", "max_filter", "no_grad",
    "pad2d", "require_torch", "resize", "resolve_device", "separable_conv", "torch_disabled_by_env",
]

