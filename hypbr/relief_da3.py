"""Optional Pro macro-relief backend for `hypbr photo`: Apache-2.0 Depth Anything 3 Mono Large.

Uses `depth-anything/DA3MONO-LARGE` (Apache-2.0) + an optional hypbr DPT head relief patch
(`HYPBR_DA3_PATCH`, e.g. `exp032_da3mono_full_800scans_s101_da3mono_patch.pt`: cross-lab
det +0.405 / slope +0.365 / hf +0.428, 7.32 deg nMAE, 0.010 albedo leak at 1008 px).

Sign convention: `DA3MONO-LARGE` outputs depth (larger = farther); relief height is `-depth`
so larger = closer = higher, matching `relief_da2.infer_relief` and `depthpro_pbr.infer_inv_depth`.
"""
from __future__ import annotations

import os
from pathlib import Path
import types
import sys

import numpy as np

from . import _backend as bk
from ._backend import no_grad

MODEL_ID = "depth-anything/DA3MONO-LARGE"
INFER_RES = 504
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _to_f32(obj):
    torch = bk.require_torch("`hypbr photo --depth-model da3mono`")
    if torch.is_tensor(obj):
        return obj.float()
    if isinstance(obj, tuple):
        return tuple(_to_f32(x) for x in obj)
    if isinstance(obj, list):
        return [_to_f32(x) for x in obj]
    return obj


def load_da3mono_relief(patch_path: Path | str | None = None):
    """Load Apache-2.0 DA3MONO-LARGE and apply an optional hypbr DPT head relief patch."""
    torch = bk.require_torch("`hypbr photo --depth-model da3mono`")
    sys.modules.setdefault("depth_anything_3.utils.export", types.SimpleNamespace(export=None))
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise ImportError(
            "`--depth-model da3mono` requires the Apache-2.0 Depth-Anything-3 package: "
            "pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
        ) from exc

    dev = bk.resolve_device()
    wrapper = DepthAnything3.from_pretrained(MODEL_ID).to(dev).eval()
    net = wrapper.model
    p_str = str(patch_path) if patch_path else os.environ.get("HYPBR_DA3_PATCH", "").strip()
    n_loaded = 0
    if p_str:
        p = Path(p_str)
        if not p.exists():
            raise FileNotFoundError(f"HYPBR_DA3_PATCH not found: {p}")
        if p.suffix == ".safetensors":
            from safetensors.torch import load_file
            patch = load_file(str(p))
        else:
            patch = torch.load(p, map_location="cpu", weights_only=True)
        sd = net.state_dict()
        for k, v in patch.items():
            if k in sd and sd[k].shape == v.shape:
                sd[k].copy_(v.to(device=sd[k].device, dtype=sd[k].dtype))
                n_loaded += int(v.numel())
    return net, n_loaded


@no_grad
def infer_relief(net, rgb_uint8: np.ndarray, tileable: bool = False, rot4: bool = False, res: int = INFER_RES) -> np.ndarray:
    """Relief height (H, W) float32 for an RGB uint8 photo; larger = higher."""
    torch = bk.require_torch("`hypbr photo --depth-model da3mono`")
    import torch.nn.functional as F

    from .depthpro_pbr import detrend_surface
    from .weightcode import tileable_4quad_blend

    dev = next(net.parameters()).device
    mean = torch.tensor(_MEAN, device=dev).view(1, 1, 3, 1, 1)
    std = torch.tensor(_STD, device=dev).view(1, 1, 3, 1, 1)

    def _once(img: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(img)).to(dev).float().permute(2, 0, 1)[None] / 255.0
        x = F.interpolate(x, size=(res, res), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        x5 = (x.unsqueeze(1) - mean) / std
        feats, _ = net.backbone(x5, cam_token=None, export_feat_layers=[])
        out = net.head(_to_f32(feats), res, res, patch_start_idx=0)
        d = out["depth"] if isinstance(out, dict) else out.depth
        d4 = d[:, :1].float() if d.ndim == 4 else d[:, None].float()
        h = F.interpolate(-d4, size=img.shape[:2], mode="bicubic", align_corners=False)
        return h[0, 0].cpu().numpy().astype(np.float32)

    def _run(img: np.ndarray) -> np.ndarray:
        if rot4:
            hs = [np.rot90(_once(np.rot90(img, k)), (4 - k) % 4) for k in range(4)]
            return np.mean([detrend_surface(h, degree=2) for h in hs], axis=0).astype(np.float32)
        out = _once(img)
        return detrend_surface(out, degree=2) if tileable else out

    out = tileable_4quad_blend(rgb_uint8, _run) if tileable else _run(rgb_uint8)
    if dev.type == "mps":
        torch.mps.empty_cache()
    return out.astype(np.float32)
