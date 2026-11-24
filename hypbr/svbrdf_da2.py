"""Optional Neural SVBRDF Decomposition for `hypbr photo` (Albedo Delighting + Roughness + Metalness).

Shares the frozen Apache-2.0 DINOv2 ViT-S backbone from `Depth-Anything-V2-Small-hf` (zero extra
base-model download!) with a 3.15 MB (1.65M-param fp16) skip-connected SVBRDF decoder trained on
882 CC0 Poly Haven + ambientCG SVBRDF scans under physically-based Cook-Torrance GGX lighting.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import _backend as bk
from ._backend import no_grad

PKG_DIR = Path(__file__).resolve().parent
SVBRDF_PATCH_PATH = Path(os.environ.get("HYPBR_SVBRDF_PATCH", PKG_DIR / "weights" / "svbrdf_da2s.safetensors"))
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_svbrdf_decoder():
    """Construct the 1.65M-param skip-connected SVBRDF decoder (albedo, roughness, metalness)."""
    torch = bk.require_torch("`hypbr photo --svbrdf`")
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvAct(nn.Module):
        def __init__(self, ci: int, co: int, k: int = 3):
            super().__init__()
            self.conv = nn.Conv2d(ci, co, k, padding=k // 2)
            self.gn = nn.GroupNorm(min(8, co), co)

        def forward(self, x):
            return F.silu(self.gn(self.conv(x)))

    class SVBRDFDecoder(nn.Module):
        def __init__(self, vit_dim: int = 384, fpn_ch: int = 128):
            super().__init__()
            self.proj = nn.ModuleList([nn.Conv2d(vit_dim, fpn_ch, 1) for _ in range(4)])
            self.fuse = nn.Sequential(ConvAct(fpn_ch * 4, 192, 3), ConvAct(192, 128, 3))
            self.rgb_enc = nn.Sequential(ConvAct(3, 32, 3), ConvAct(32, 48, 3))
            self.up1 = nn.Sequential(ConvAct(128, 96, 3), ConvAct(96, 64, 3))
            self.final_fuse = nn.Sequential(ConvAct(64 + 48, 80, 3), ConvAct(80, 64, 3))
            self.head_alb = nn.Sequential(ConvAct(64, 32, 3), nn.Conv2d(32, 3, 1))
            self.head_rm = nn.Sequential(ConvAct(64, 32, 3), nn.Conv2d(32, 2, 1))

        def forward(self, rgb_01, feats, ph: int, pw: int):
            B, _, H, W = rgb_01.shape
            maps = []
            for idx, f in enumerate(feats):
                tok = f[:, -ph * pw :, :].transpose(1, 2).reshape(B, -1, ph, pw) if f.ndim == 3 else f
                maps.append(self.proj[idx](tok.float()))
            z = self.fuse(torch.cat(maps, dim=1))
            z = F.interpolate(z, size=(H // 2, W // 2), mode="bilinear", align_corners=False)
            z = self.up1(z)
            z = F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)
            r_feat = self.rgb_enc(rgb_01 * 2.0 - 1.0)
            h = self.final_fuse(torch.cat([z, r_feat], dim=1))
            alb_pred = (rgb_01 + torch.tanh(self.head_alb(h)) * 0.65).clamp(0.0, 1.0)
            rm = torch.sigmoid(self.head_rm(h))
            return alb_pred, rm[:, 0:1], rm[:, 1:2]

    return SVBRDFDecoder()


def load_svbrdf_decoder(patch_path: Path | str = SVBRDF_PATCH_PATH, device=None):
    """Load the 3.15 MB SVBRDF decoder onto `device` (or auto-resolved device)."""
    torch = bk.require_torch("`hypbr photo --svbrdf`")
    p = Path(patch_path)
    if not p.exists():
        raise FileNotFoundError(f"SVBRDF weights not found: {p}")
    dec = build_svbrdf_decoder()
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(p))
    else:
        sd = torch.load(p, map_location="cpu", weights_only=True)
    dec.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    dev = bk.resolve_device() if device is None else device
    return dec.to(dev).eval()


@no_grad
def infer_svbrdf(da2_model, decoder, rgb_uint8: np.ndarray, res: int = 518) -> dict[str, np.ndarray]:
    """Predict delighted `albedo_u8` (H,W,3), `roughness_u8` (H,W), and `metalness_u8` (H,W)."""
    torch = bk.require_torch("`hypbr photo --svbrdf`")
    import torch.nn.functional as F

    dev = next(decoder.parameters()).device
    mean = torch.tensor(_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=dev).view(1, 3, 1, 1)
    H0, W0 = rgb_uint8.shape[:2]

    rgb = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).to(dev).float().permute(2, 0, 1)[None] / 255.0
    rgb_r = F.interpolate(rgb, size=(res, res), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    feats = da2_model.backbone.forward_with_filtered_kwargs((rgb_r - mean) / std).feature_maps
    ph, pw = res // da2_model.config.patch_size, res // da2_model.config.patch_size
    ap, rp, mp = decoder(rgb_r, feats, ph, pw)

    ap = F.interpolate(ap, size=(H0, W0), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).cpu().numpy()
    rp = F.interpolate(rp, size=(H0, W0), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    mp = F.interpolate(mp, size=(H0, W0), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return {
        "albedo": np.clip(ap * 255.0 + 0.5, 0, 255).astype(np.uint8),
        "roughness": np.clip(rp * 255.0 + 0.5, 0, 255).astype(np.uint8),
        "metalness": np.clip(mp * 255.0 + 0.5, 0, 255).astype(np.uint8),
    }
