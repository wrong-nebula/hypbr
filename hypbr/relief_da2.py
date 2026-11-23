"""Macro relief for `hypbr photo` from Depth Anything V2 Small + hypbr's relief patch (the default backend).

The base model (`depth-anything/Depth-Anything-V2-Small-hf`, Apache-2.0, ~99 MB) is fetched from Hugging Face on
first use. hypbr ships only the 5 MB patch `weights/da2s_relief_pbr.safetensors`: the model's DPT neck + head
(2.73M params) fine-tuned on 344 CC0 Poly Haven scans to predict surface relief from studio-lit close-up photos,
with the DINOv2 ViT-S backbone left untouched. On held-out scans it beats the Apple Depth Pro 2 path (research-only
license) on every relief metric; see `weights/NOTICE.md`.

The output follows the same convention as `depthpro_pbr.infer_inv_depth`: larger = closer to the camera = higher.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from . import _backend as bk
from ._backend import no_grad

PKG_DIR = Path(__file__).resolve().parent
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
CONFIG_PATH = PKG_DIR / "weights" / "da2s_config.json"
PATCH_PATH = Path(os.environ.get("HYPBR_DA2_PATCH", PKG_DIR / "weights" / "da2s_relief_pbr.safetensors"))
INFER_RES = 518  # the patch was trained on 378 / 518 px inputs (multiples of the 14 px ViT patch)
# Shape + detail blend: the patched head sharpens fine relief but loses some broad shapes the untouched model gets
# right (overlapping boards, rock plates; on bark it even inverts). Above BLEND_SIGMA_512 px (at 512 px output) the
# result takes BLEND_LOW of its shape from the original head; below it, all detail from the patched head. Held-out
# score 0.266 -> 0.283 (bark -0.27 -> +0.02, boards 0.50 -> 0.62), cross-lab unchanged within 0.005.
BLEND_SIGMA_512 = 16.0
BLEND_LOW = 0.3
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _transformers():
    bk.require_torch("`hypbr photo` (Depth Anything V2 relief)")
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "`hypbr photo` needs Hugging Face transformers for Depth Anything V2: pip install 'hypbr[neural]'"
        ) from exc
    return transformers


def build_da2_model(pretrained: bool = True):
    """Depth Anything V2 Small, downloaded once from Hugging Face (`pretrained=False`: random init, offline)."""
    tf = _transformers()
    if pretrained:
        return tf.AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    cfg = tf.DepthAnythingConfig.from_dict(json.loads(CONFIG_PATH.read_text()))
    return tf.DepthAnythingForDepthEstimation(cfg)


@no_grad
def apply_relief_patch(model, patch_path: Path | str = PATCH_PATH) -> int:
    """Copy the patch tensors into `model` (strict: every key must exist with the same shape). Returns #params."""
    from safetensors.torch import load_file

    patch = load_file(str(patch_path))
    sd = model.state_dict()
    bad = [k for k, v in patch.items() if k not in sd or sd[k].shape != v.shape]
    if bad:
        raise ValueError(f"relief patch {patch_path} does not fit this model: {bad[:3]}")
    for k, v in patch.items():
        sd[k].copy_(v.to(device=sd[k].device, dtype=sd[k].dtype))
    return int(sum(v.numel() for v in patch.values()))


def load_da2_relief(patch_path: Path | str | None = PATCH_PATH, pretrained: bool = True):
    """Base model + relief patch on `HYPBR_DEVICE` (auto: cuda > mps > cpu), in eval mode.

    The original neck + head are kept alongside as `model.hypbr_base_head` (2.7M params) for the shape blend.
    """
    model = build_da2_model(pretrained=pretrained)
    base = (copy.deepcopy(model.neck), copy.deepcopy(model.head)) if patch_path else None
    n = apply_relief_patch(model, patch_path) if patch_path else 0
    dev = bk.resolve_device()
    if base is not None:
        model.hypbr_base_head = tuple(m.to(dev).eval() for m in base)
    return model.to(dev).eval(), n


def blend_shape_detail(base: np.ndarray, patched: np.ndarray, sigma: float, low: float = BLEND_LOW) -> np.ndarray:
    """Broad shape from `base` (Gaussian low-pass) + fine detail from `patched` (high-pass), in `patched`'s units."""
    from .depthpro_pbr import detrend_surface, gaussian_blur_np

    b = detrend_surface(base, degree=2)
    t = detrend_surface(patched, degree=2)
    scale = float(t.std())
    if low <= 0.0 or scale < 1e-12 or float(b.std()) < 1e-12:
        return patched.astype(np.float32)
    lo = gaussian_blur_np(b / b.std(), sigma, circular=False)
    tn = t / scale
    hi = tn - gaussian_blur_np(tn, sigma, circular=False)
    out = low * lo / (lo.std() + 1e-8) + (1.0 - low) * hi / (hi.std() + 1e-8)
    return (out * scale).astype(np.float32)


@no_grad
def infer_relief(model, rgb_uint8: np.ndarray, tileable: bool = False, rot4: bool = False,
                 res: int = INFER_RES, blend_low: float = BLEND_LOW) -> np.ndarray:
    """Relief height (H, W) float32 for an RGB uint8 photo; larger = higher.

    `rot4` averages the four 90-degree rotations; `tileable` evaluates four half-tile cyclic shifts blended with an
    exact partition of unity, so the result has no seam at the tile border (both as in the Depth Pro path).
    `blend_low` > 0 mixes in the original head's broad shape (see BLEND_LOW); 0 uses the patched head alone.
    """
    torch = bk.require_torch("`hypbr photo` (Depth Anything V2 relief)")
    import torch.nn.functional as F

    from .depthpro_pbr import detrend_surface
    from .weightcode import tileable_4quad_blend

    dev = next(model.parameters()).device
    mean = torch.tensor(_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=dev).view(1, 3, 1, 1)

    base = getattr(model, "hypbr_base_head", None) if blend_low > 0.0 else None

    def _up(y, hw) -> np.ndarray:
        y = F.interpolate(y[:, None].float(), size=hw, mode="bicubic", align_corners=False)
        return y[0, 0].cpu().numpy().astype(np.float32)

    def _once(img: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(img)).to(dev).float().permute(2, 0, 1)[None] / 255.0
        x = F.interpolate(x, size=(res, res), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        x = (x - mean) / std
        if base is None:
            return _up(model(pixel_values=x).predicted_depth, img.shape[:2])
        feats = model.backbone.forward_with_filtered_kwargs(x).feature_maps  # one backbone pass, two heads
        ph, pw = x.shape[-2] // model.config.patch_size, x.shape[-1] // model.config.patch_size
        t = _up(model.head(model.neck(feats, ph, pw), ph, pw), img.shape[:2])
        b = _up(base[1](base[0](feats, ph, pw), ph, pw), img.shape[:2])
        return blend_shape_detail(b, t, BLEND_SIGMA_512 * img.shape[0] / 512.0, blend_low)

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
