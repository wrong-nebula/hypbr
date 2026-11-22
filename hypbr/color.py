"""Exact IEC 61966-2-1 sRGB transfer functions shared by every albedo read/write.

Engines (glTF, Three.js `SRGBColorSpace`, Blender `sRGB`, Unity/Unreal sRGB textures) decode
albedo with the piecewise sRGB curve, not a pure 2.2 power, so hypbr encodes with the same curve.
"""
from __future__ import annotations

import numpy as np


def srgb_to_linear(c):
    """sRGB-encoded values in [0, 1] -> linear light (float32 array, or float for scalars)."""
    c = np.clip(np.asarray(c, dtype=np.float32), 0.0, 1.0)
    out = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)
    return float(out) if out.ndim == 0 else out


def linear_to_srgb(x):
    """Linear light in [0, 1] -> sRGB-encoded values (float32 array, or float for scalars)."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    out = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)
    return float(out) if out.ndim == 0 else out


def srgb_u8_to_linear(u8: np.ndarray) -> np.ndarray:
    """8-bit sRGB image -> linear float32."""
    return srgb_to_linear(np.asarray(u8, dtype=np.float32) / 255.0)


def linear_to_srgb_u8(x: np.ndarray) -> np.ndarray:
    """Linear float image -> 8-bit sRGB (rounded)."""
    return np.clip(linear_to_srgb(x) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def linear_to_srgb_torch(x):
    """Torch tensor variant of `linear_to_srgb` (stays on the tensor's device)."""
    import torch

    x = torch.clamp(x, 0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)
