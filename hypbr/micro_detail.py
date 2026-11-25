"""Curl-free 3-scale neural micro-detail refiner (`MicroUNet-dh-1K`, 229.6 KB fp16).

Predicts a physically integrable scalar micro-height residual `dh(x, y)` from full-res RGB albedo
plus macro normal & height maps, then derives exact curl-free micro-normals via crisp finite
differences and RNM-blends them onto the base normal map (cross-lab 6.09 deg, val PSNR 31.12 dB,
0.323 deg flat-trap leak; matches the WebGPU compute weights `microunet_dh_1k_f16.bin`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import _backend as bk
from ._backend import no_grad

PKG_DIR = Path(__file__).resolve().parent
WEIGHTS_F16_PATH = Path(os.environ.get("HYPBR_MICROUNET_BIN", PKG_DIR / "weights" / "microunet_dh_1k_f16.bin"))
WEIGHTS_META_PATH = Path(os.environ.get("HYPBR_MICROUNET_META", PKG_DIR / "weights" / "microunet_dh_1k_meta.json"))


def build_microunet_dh():
    """Construct the 117,561-param 3-scale circular curl-free MicroUNet-dh architecture."""
    torch = bk.require_torch("`hypbr.micro_detail` (MicroUNet-dh-1K)")
    import torch.nn as nn
    import torch.nn.functional as F

    def _pad(x, p: int, mode: str):
        return F.pad(x, (p, p, p, p), mode=mode)

    class ResBlock(nn.Module):
        def __init__(self, ch: int, dilation: int = 1):
            super().__init__()
            self.d = dilation
            self.c1 = nn.Conv2d(ch, ch, kernel_size=3, padding=0, dilation=dilation)
            self.c2 = nn.Conv2d(ch, ch, kernel_size=1)

        def forward(self, x, pad_mode: str = "circular"):
            return x + self.c2(F.silu(self.c1(_pad(x, self.d, pad_mode))))

    def height_to_dxy(dh, pad_mode: str = "circular"):
        p = _pad(dh, 1, pad_mode)
        dx_c = (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]) * 0.5
        dy_c = (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]) * 0.5
        dx_s = (
            (p[:, :, :-2, 2:] + 2.0 * p[:, :, 1:-1, 2:] + p[:, :, 2:, 2:])
            - (p[:, :, :-2, :-2] + 2.0 * p[:, :, 1:-1, :-2] + p[:, :, 2:, :-2])
        ) * 0.125
        dy_s = (
            (p[:, :, 2:, :-2] + 2.0 * p[:, :, 2:, 1:-1] + p[:, :, 2:, 2:])
            - (p[:, :, :-2, :-2] + 2.0 * p[:, :, :-2, 1:-1] + p[:, :, :-2, 2:])
        ) * 0.125
        return torch.cat([-(0.75 * dx_c + 0.25 * dx_s), (0.75 * dy_c + 0.25 * dy_s)], dim=1)

    def rnm_blend(n_base, dxy):
        t = torch.cat([n_base[:, :2], n_base[:, 2:3] + 1.0], dim=1)
        dz = torch.sqrt((1.0 - (dxy[:, 0:1] ** 2 + dxy[:, 1:2] ** 2)).clamp(min=0.05))
        u = torch.cat([-dxy[:, 0:1], -dxy[:, 1:2], dz], dim=1)
        dot = (t * u).sum(dim=1, keepdim=True)
        return F.normalize(t * (dot / t[:, 2:3].clamp(min=1e-5)) - u, dim=1, eps=1e-6)

    class MicroUNetDH(nn.Module):
        def __init__(self, in_ch: int = 10, c0: int = 24, c1: int = 36, c2: int = 48):
            super().__init__()
            self.stem = nn.Conv2d(in_ch, c0, 3, padding=0)
            self.e0 = ResBlock(c0, 1)
            self.d1_conv = nn.Conv2d(c0, c1, 3, padding=0)
            self.e1 = ResBlock(c1, 1)
            self.d2_conv = nn.Conv2d(c1, c2, 3, padding=0)
            self.b0 = ResBlock(c2, 1)
            self.b1 = ResBlock(c2, 2)
            self.u1_fuse = nn.Conv2d(c2 + c1, c1, 1)
            self.u1_blk = ResBlock(c1, 1)
            self.u0_fuse = nn.Conv2d(c1 + c0, c0, 1)
            self.u0_blk = ResBlock(c0, 1)
            self.head1 = nn.Conv2d(c0, 16, 3, padding=0)
            self.head2 = nn.Conv2d(16, 1, 1)

        def forward(self, alb, n_lo, h_lo, tileable: bool = True, strength: float = 1.0):
            pm = "circular" if tileable else "reflect"
            lum = 0.299 * alb[:, 0:1] + 0.587 * alb[:, 1:2] + 0.114 * alb[:, 2:3]
            lp3 = F.avg_pool2d(_pad(lum, 1, pm), 3, stride=1)
            lp9 = F.avg_pool2d(_pad(lum, 4, pm), 9, stride=1)
            hp3 = ((lum - lp3) * 6.0).clamp(-2.5, 2.5)
            hp9 = ((lum - lp9) * 4.0).clamp(-2.5, 2.5)
            lp = _pad(lum, 1, pm)
            gx = ((lp[:, :, 1:-1, 2:] - lp[:, :, 1:-1, :-2]) * 4.0).clamp(-2.5, 2.5)
            gy = ((lp[:, :, 2:, 1:-1] - lp[:, :, :-2, 1:-1]) * 4.0).clamp(-2.5, 2.5)
            h_std = h_lo.std(dim=(2, 3), keepdim=True) + 1e-4
            h_n = ((h_lo - h_lo.mean(dim=(2, 3), keepdim=True)) / h_std).clamp(-3.0, 3.0) / 3.0
            x = torch.cat([alb * 2.0 - 1.0, n_lo[:, :2], h_n, hp3, hp9, gx, gy], dim=1)

            s0 = self.e0(F.silu(self.stem(_pad(x, 1, pm))), pm)
            s1 = self.e1(F.silu(self.d1_conv(_pad(F.avg_pool2d(s0, 2), 1, pm))), pm)
            b = self.b1(self.b0(F.silu(self.d2_conv(_pad(F.avg_pool2d(s1, 2), 1, pm))), pm), pm)

            u1 = F.interpolate(b, size=s1.shape[-2:], mode="bilinear", align_corners=False)
            u1 = self.u1_blk(F.silu(self.u1_fuse(torch.cat([u1, s1], dim=1))), pm)
            u0 = F.interpolate(u1, size=s0.shape[-2:], mode="bilinear", align_corners=False)
            u0 = self.u0_blk(F.silu(self.u0_fuse(torch.cat([u0, s0], dim=1))), pm)

            dh = torch.tanh(self.head2(F.silu(self.head1(_pad(u0, 1, pm))))) * 2.2
            dxy = (height_to_dxy(dh, pm) * float(strength)).clamp(-0.85, 0.85)
            return rnm_blend(n_lo, dxy), dh

    return MicroUNetDH()


def load_microunet_dh(bin_path: Path | str = WEIGHTS_F16_PATH, meta_path: Path | str = WEIGHTS_META_PATH, device=None):
    """Load the shipped 229.6 KB fp16 `MicroUNet-dh-1K` weights into PyTorch."""
    torch = bk.require_torch("`hypbr.micro_detail` (MicroUNet-dh-1K)")
    model = build_microunet_dh()
    raw = np.frombuffer(Path(bin_path).read_bytes(), dtype=np.float16).astype(np.float32)
    meta = json.loads(Path(meta_path).read_text())
    sd = model.state_dict()
    for t in meta["tensors"]:
        name, shape, off, cnt = t["name"], tuple(t["shape"]), int(t["offset"]), int(t["count"])
        sd[name].copy_(torch.from_numpy(raw[off : off + cnt].reshape(shape)))
    dev = bk.resolve_device() if device is None else device
    return model.to(dev).eval()


@no_grad
def refine_micro_detail(
    model,
    rgb_uint8: np.ndarray,
    normal_gl_f32: np.ndarray,
    height_f32: np.ndarray,
    *,
    tileable: bool = True,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(refined_normal_gl_f32 (H,W,3), micro_height_dh_f32 (H,W))`."""
    torch = bk.require_torch("`hypbr.micro_detail` (MicroUNet-dh-1K)")
    dev = next(model.parameters()).device
    alb = torch.from_numpy(np.ascontiguousarray(rgb_uint8).astype(np.float32).transpose(2, 0, 1) / 255.0)[None].to(dev)
    nb = torch.from_numpy(np.ascontiguousarray(normal_gl_f32).astype(np.float32).transpose(2, 0, 1))[None].to(dev)
    hb = torch.from_numpy(np.ascontiguousarray(height_f32).astype(np.float32))[None, None].to(dev)
    n_out, dh = model(alb, nb, hb, tileable=tileable, strength=strength)
    return (
        n_out[0].permute(1, 2, 0).cpu().numpy().astype(np.float32),
        dh[0, 0].cpu().numpy().astype(np.float32),
    )
