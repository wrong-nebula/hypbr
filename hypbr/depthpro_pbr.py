#!/usr/bin/env python3
"""Production $0 Local PBR Heightmap, Tangent-Space Normal Map & Cavity AO Generator.

Unified V2 pipeline:
  1. Macro/medium 3D relief from WeightCode-edited `apple/DepthPro2-S` (`wsp_depthpro2_s_pbr.pt`,
     synthesized by `google/gemini-4-neon` medium — 1.44% weights edited, fov=False).
  2. Pixel-resolution micro-relief via chromaticity-gradient gating + multi-scale morphological
     Black-Hat/Top-Hat valley/ridge extraction + joint-bilateral denoising.
  3. Tangent-space normals via physically exact Reoriented Normal Mapping (RNM) combining
     calibrated V2 height normals with `DeepBump` (25 MB PBR U-Net, optional via `uv`).

Outputs 512x512 linear PNGs (`height.png`, `normal.png`, `ao.png`) ready for
standard PBR renderers (glTF 2.0, Blender, Three.js, Unreal, Unity).

Usage:
  hypbr photo path/to/material.jpg --out ./output/pbr [--tileable] [--strength 1.0] [--dx]

Note: Apple DepthPro2 weights use the non-commercial `apple-amlr` research license.
"""
import argparse
import gc
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import _backend as bk
from ._backend import no_grad
from .color import srgb_to_linear

PKG_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("HYPBR_CACHE_DIR", Path.home() / ".cache" / "hypbr"))
NEON_PATCH = Path(os.environ.get("HYPBR_DEPTHPRO_PATCH", PKG_DIR / "weights" / "wsp_depthpro2_s_pbr.pt"))
ONNX_WSP = Path(os.environ.get("HYPBR_DEEPBUMP_ONNX", PKG_DIR / "weights" / "wsp_deepbump_pbr.onnx"))
ONNX_CACHE = CACHE_DIR / "deepbump_normal.onnx"

# torch is optional: without it (or with HYPBR_TORCH=0) DEVICE is a CPU stand-in and every
# procedural op runs its NumPy implementation. `HYPBR_DEVICE` overrides auto-detection.
DEVICE = bk.resolve_device()


def load_depthpro2_no_fov(preset_name: str = "depthpro2-s", precision=None):
    """Load DepthPro2 without the 94 MB FOVEstimator (orthographic PBR photography needs no FOV)."""
    torch = bk.require_torch("Neural photo mode (DepthPro2)")
    if precision is None:
        precision = torch.float32
    from depth_pro_2.model import DepthPro2, PRESETS, load_checkpoint

    cfg = PRESETS[preset_name]
    model = DepthPro2(
        backbone_preset=cfg.backbone_preset,
        reassembled_blocks=cfg.reassembled_blocks,
        cross_frame_blocks=cfg.cross_frame_blocks,
        dims_reassemble=cfg.dims_reassemble,
        dim_decoder=cfg.dim_decoder,
        fov=False,
        precision=precision,
    )
    sd = load_checkpoint(cfg.checkpoint_uri)
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()


@no_grad
def apply_neon_wsp_patch(model, circular_pad: bool = False):
    """Apply the compact Neon-medium WeightCode PBR patch (`wsp_depthpro2_s_pbr.pt`, 1.44% weights)."""
    torch = bk.require_torch("Neural photo mode (DepthPro2)")
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    changed = 0
    if NEON_PATCH.exists():
        patch = torch.load(NEON_PATCH, map_location="cpu", weights_only=True)
        for k, v in patch.items():
            if k in sd and sd[k].shape == v.shape:
                changed += int((sd[k] != v.to(dtype=sd[k].dtype)).sum().item())
                sd[k] = v.to(dtype=sd[k].dtype)
        model.load_state_dict(sd, strict=True)
    else:
        sd, changed = apply_pbr_weight_tweak(model, circular_pad=False)
    if circular_pad:
        set_circular_padding(model, True)
    return sd, changed


@no_grad
def apply_pbr_weight_tweak(model, scale_weights=(1.85, 1.35, 0.65, 0.20),
                           boost_fine_res=1.35, circular_pad: bool = False):
    """Analytic spectral DPT pyramid WeightCode surgery (zero training)."""
    from .weightcode import apply_depthpro_weightcode

    report = apply_depthpro_weightcode(
        model,
        scale_weights=scale_weights,
        boost_fine_res=boost_fine_res,
        circular_pad=circular_pad,
    )
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return sd, report.params_edited


def set_circular_padding(model, enabled: bool):
    nn = bk.require_torch("Neural photo mode (DepthPro2)").nn
    mode = "circular" if enabled else "zeros"
    for mod in (model.decoder, model.head):
        for m in mod.modules():
            if isinstance(m, nn.Conv2d) and m.padding != (0, 0):
                m.padding_mode = mode


def reset_padding_zeros(model):
    set_circular_padding(model, False)


def detrend_surface(h: np.ndarray, degree: int = 2) -> np.ndarray:
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W]
    x = (xx / max(W - 1, 1)) * 2.0 - 1.0
    y = (yy / max(H - 1, 1)) * 2.0 - 1.0
    if degree == 1:
        A = np.stack([np.ones_like(x), x, y], axis=-1).reshape(-1, 3)
    else:
        A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1).reshape(-1, 6)
    coef, _, _, _ = np.linalg.lstsq(A, h.reshape(-1), rcond=None)
    return (h - (A @ coef).reshape(H, W)).astype(np.float32)


_FFT_BLUR_MIN_RAD = 16  # below this a separable spatial conv is as fast as the FFT
_NP_DIRECT_BLUR_MAX_RAD = 5  # torch-free path: direct NumPy taps only up to this radius, FFT above


def _wrapped_kernel_fft(ker: np.ndarray, n: int, real: bool) -> np.ndarray:
    """Spectrum of a symmetric 1-D kernel wrapped onto a period of `n` samples."""
    rad = (ker.size - 1) // 2
    k = np.zeros(n, dtype=np.float64)
    np.add.at(k, np.arange(-rad, rad + 1) % n, ker.astype(np.float64))
    return (np.fft.rfft(k) if real else np.fft.fft(k)).real


def _gaussian_blur_fft(arr: np.ndarray, ker: np.ndarray, circular: bool) -> np.ndarray:
    """Same result as the separable spatial conv below, in O(N log N) for any radius.

    Circular mode convolves on the torus directly; reflect mode reflect-pads by the kernel
    radius first, so the wrap-around of the padded FFT never reaches the kept region.
    """
    rad = (ker.size - 1) // 2
    a = np.asarray(arr, dtype=np.float32)
    if not circular:
        a = np.pad(a, rad, mode="reflect")
    H, W = a.shape
    spec = np.fft.rfft2(a)
    spec *= _wrapped_kernel_fft(ker, H, real=False)[:, None]
    spec *= _wrapped_kernel_fft(ker, W, real=True)[None, :]
    out = np.fft.irfft2(spec, s=(H, W))
    if not circular:
        out = out[rad:-rad, rad:-rad]
    return np.ascontiguousarray(out, dtype=np.float32)


def gaussian_blur_np(arr: np.ndarray, sigma: float, circular: bool = False) -> np.ndarray:
    if sigma <= 0.1:
        return arr.astype(np.float32)
    rad = max(1, int(math.ceil(3.0 * sigma)))
    xs = np.arange(-rad, rad + 1, dtype=np.float32)
    ker = np.exp(-0.5 * (xs / sigma) ** 2)
    ker /= ker.sum()
    if rad >= _FFT_BLUR_MIN_RAD or rad >= min(arr.shape[0], arr.shape[1]):
        return _gaussian_blur_fft(arr, ker, circular)
    use_gpu = (DEVICE.type != "cpu") and (arr.size >= 262144) and (rad < min(arr.shape[0], arr.shape[1]))
    torch = bk.get_torch()
    if torch is not None:
        with torch.no_grad():
            return bk.separable_conv(arr, ker, circular, device=DEVICE if use_gpu else torch.device("cpu"))
    if rad > _NP_DIRECT_BLUR_MAX_RAD:  # NumPy taps lose to the FFT much earlier than torch conv2d
        return _gaussian_blur_fft(arr, ker, circular)
    return bk.separable_conv(arr, ker, circular)


def _morph_close_2d(arr: np.ndarray, radius: int, circular: bool = False) -> np.ndarray:
    mode = "circular" if circular else "reflect"
    dil = bk.max_filter(arr, radius, mode)
    return -bk.max_filter(-dil, radius, mode)


def _morph_open_2d(arr: np.ndarray, radius: int, circular: bool = False) -> np.ndarray:
    return -_morph_close_2d(-arr, radius, circular=circular)


def _bilateral_smooth(img: np.ndarray, guide: np.ndarray, sigma_s: float = 1.8, sigma_r: float = 0.14,
                      circular: bool = False) -> np.ndarray:
    rad = 2
    mode = "circular" if circular else "reflect"
    t_img = bk.pad2d(img.astype(np.float32), rad, rad, rad, rad, mode)
    t_g = bk.pad2d(guide.astype(np.float32), rad, rad, rad, rad, mode)
    H, W = img.shape
    num = np.zeros((H, W), dtype=np.float32)
    den = np.zeros((H, W), dtype=np.float32)
    inv_2s = 1.0 / (2.0 * sigma_s * sigma_s)
    inv_2r = 1.0 / (2.0 * sigma_r * sigma_r)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            ws = math.exp(-(dx * dx + dy * dy) * inv_2s)
            pi = t_img[rad + dy:rad + dy + H, rad + dx:rad + dx + W]
            pg = t_g[rad + dy:rad + dy + H, rad + dx:rad + dx + W]
            w = ws * np.exp(-((pg - guide) ** 2) * inv_2r)
            num += w * pi
            den += w
    return num / np.maximum(den, 1e-6)


def blend_rnm(n_base: np.ndarray, n_detail: np.ndarray, detail_weight: float = 0.65) -> np.ndarray:
    """Reoriented Normal Mapping (Hill & Barre-Brisebois): exact tangent-space normal fusion."""
    nb = n_base / (np.linalg.norm(n_base, axis=-1, keepdims=True) + 1e-8)
    nd = n_detail.copy()
    nd[..., 0] *= detail_weight
    nd[..., 1] *= detail_weight
    nd[..., 2] = np.maximum(nd[..., 2], 0.1)
    nd /= np.linalg.norm(nd, axis=-1, keepdims=True) + 1e-8
    t = nb.copy()
    t[..., 2] += 1.0
    u = nd.copy()
    u[..., 0] = -u[..., 0]
    u[..., 1] = -u[..., 1]
    dot = np.sum(t * u, axis=-1, keepdims=True)
    r = t * (dot / np.maximum(t[..., 2:3], 1e-5)) - u
    return (r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)


def _fast_percentiles(arr: np.ndarray, qs: tuple[float, ...]) -> tuple[float, ...]:
    """Fast robust percentiles using uniform strided subsampling above 262K elements."""
    flat = np.asarray(arr, dtype=np.float32).ravel()
    stride = max(1, flat.size // 262144)
    vals = np.percentile(flat[::stride], qs)
    return tuple(float(v) for v in vals)


def normalize_height(h: np.ndarray, clip_pct: float = 1.0, min_span: float = 0.0) -> np.ndarray:
    lo, hi = _fast_percentiles(h, (clip_pct, 100.0 - clip_pct))
    span = hi - lo
    if span < 1e-6:
        return np.full_like(h, 0.5, dtype=np.float32)
    if min_span > 0.0 and span < min_span:
        mid = 0.5 * (lo + hi)
        return np.clip(0.5 + (h - mid) / min_span, 0.0, 1.0).astype(np.float32)
    return np.clip((h - lo) / span, 0.0, 1.0).astype(np.float32)


_SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32) / np.float32(32.0)
_SCHARR_Y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32) / np.float32(32.0)


def height_to_unit_normal(
    h: np.ndarray,
    strength: float = 2.4,
    flip_y: bool = False,
    circular: bool = False,
    min_span: float = 0.0,
) -> np.ndarray:
    p5, p95 = _fast_percentiles(h, (3.0, 97.0))
    span = p95 - p5
    if span < 1e-6:
        out = np.zeros(h.shape + (3,), dtype=np.float32)
        out[..., 2] = 1.0
        return out
    denom = max(span, min_span) if min_span > 0.0 else span
    hn = (h - p5) / denom
    mode = "circular" if circular else "reflect"
    gx = bk.correlate3x3(hn, _SCHARR_X, mode)
    gy = bk.correlate3x3(hn, _SCHARR_Y, mode)
    nx = -gx * strength
    ny = (gy if not flip_y else -gy) * strength
    nz = np.ones_like(nx)
    nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    return np.stack([nx / nl, ny / nl, nz / nl], axis=-1).astype(np.float32)


def normal_to_u8(N: np.ndarray) -> np.ndarray:
    Np = N / (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-8)
    return np.clip((Np * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def height_to_normal(h: np.ndarray, strength: float = 3.5, flip_y: bool = False, circular: bool = False) -> np.ndarray:
    return normal_to_u8(height_to_unit_normal(h, strength=strength, flip_y=flip_y, circular=circular))


def height_to_ao(h: np.ndarray, strength: float = 1.5, circular: bool = False) -> np.ndarray:
    """Bilateral-denoised multi-scale Cavity AO (eliminates salt-and-pepper speckle on rough stone)."""
    h_sm = _bilateral_smooth(h, h, sigma_s=2.0, sigma_r=0.12, circular=circular)
    b1 = gaussian_blur_np(h_sm, sigma=3.0, circular=circular)
    b2 = gaussian_blur_np(h_sm, sigma=10.0, circular=circular)
    b3 = gaussian_blur_np(h_sm, sigma=24.0, circular=circular)
    cavity = 0.35 * np.maximum(0.0, b1 - h_sm) + 0.40 * np.maximum(0.0, b2 - h_sm) + 0.25 * np.maximum(0.0, b3 - h_sm)
    ao = 1.0 / (1.0 + (strength * cavity * 3.8) ** 1.35)
    return np.clip(ao * 255.0 + 0.5, 40, 255).astype(np.uint8)


def v2_micro_relief(rgb_u8: np.ndarray, circular: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rgb = rgb_u8.astype(np.float32) / 255.0
    lin = srgb_to_linear(rgb)
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    log_l = np.log(lum + 0.03)

    s = lin.sum(axis=-1) + 0.05
    cr, cg = lin[..., 0] / s, lin[..., 1] / s
    gc = np.sqrt(np.gradient(cr, axis=1) ** 2 + np.gradient(cr, axis=0) ** 2 +
                 np.gradient(cg, axis=1) ** 2 + np.gradient(cg, axis=0) ** 2)
    gc_sm = gaussian_blur_np(gc, 1.5, circular=circular)
    chroma_gate = 0.05 + 0.95 * np.exp(-((gc_sm / 0.018) ** 2))

    bh_fine = _morph_close_2d(log_l, radius=2, circular=circular) - log_l
    bh_med = _morph_close_2d(log_l, radius=5, circular=circular) - log_l
    th_fine = log_l - _morph_open_2d(log_l, radius=2, circular=circular)
    th_med = log_l - _morph_open_2d(log_l, radius=5, circular=circular)

    valley = -(0.45 * np.clip(bh_fine, 0, 1.0) + 0.55 * np.clip(bh_med, 0, 1.0))
    ridge = +(0.30 * np.clip(th_fine, 0, 0.5) + 0.35 * np.clip(th_med, 0, 0.5))
    bp = np.clip(gaussian_blur_np(log_l, 1.5, circular=circular) - gaussian_blur_np(log_l, 7.0, circular=circular), -0.5, 0.35)

    raw = (0.62 * (valley + 0.45 * ridge) + 0.38 * bp) * chroma_gate
    clean = _bilateral_smooth(raw, log_l, sigma_s=1.8, sigma_r=0.14, circular=circular)
    return detrend_surface(clean, degree=2), chroma_gate


_DEEPBUMP_WORKER = r'''
import sys, numpy as np, onnxruntime as ort
from PIL import Image
in_npz, out_npz, onnx_path, use_d4 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
data = np.load(in_npz)

def d4_apply_img(arr, g):
    a = np.fliplr(arr) if g >= 4 else arr
    return np.ascontiguousarray(np.rot90(a, k=g % 4, axes=(0, 1)))

def d4_inv_img(arr, g):
    a = np.rot90(arr, k=(4 - (g % 4)) % 4, axes=(0, 1))
    return np.ascontiguousarray(np.fliplr(a) if g >= 4 else a)

def d4_inv_normal(N, g):
    R = N.copy()
    k = (4 - (g % 4)) % 4
    R = np.rot90(R, k=k, axes=(0, 1)).copy()
    nx, ny = R[..., 0].copy(), R[..., 1].copy()
    if k == 1: R[..., 0], R[..., 1] = -ny, nx
    elif k == 2: R[..., 0], R[..., 1] = -nx, -ny
    elif k == 3: R[..., 0], R[..., 1] = ny, -nx
    if g >= 4:
        R = np.fliplr(R).copy()
        R[..., 0] = -R[..., 0]
    return R

def infer_single(lum):
    H, W = lum.shape
    if H == 256 and W == 256:
        return np.transpose(sess.run(None, {"input": lum[None, None]})[0][0], (1, 2, 0)).astype(np.float32)
    stride = 128
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wacc = np.zeros((H, W, 1), dtype=np.float32)
    wy = np.hanning(256).astype(np.float32) + 0.05
    win = np.outer(wy, wy)[..., None]
    ys = sorted(set(list(range(0, max(1, H - 255), stride)) + [max(0, H - 256)]))
    xs = sorted(set(list(range(0, max(1, W - 255), stride)) + [max(0, W - 256)]))
    for y0 in ys:
        for x0 in xs:
            tile = lum[y0:y0+256, x0:x0+256]
            out = np.transpose(sess.run(None, {"input": tile[None, None]})[0][0], (1, 2, 0)).astype(np.float32)
            acc[y0:y0+256, x0:x0+256] += out * win
            wacc[y0:y0+256, x0:x0+256] += win
    return acc / np.maximum(wacc, 1e-6)

def infer_d8(lum0):
    nx_s, ny_s, h_s = np.zeros_like(lum0), np.zeros_like(lum0), np.zeros_like(lum0)
    for g in range(8):
        o = infer_single(d4_apply_img(lum0, g))
        nx, ny = o[..., 0] * 2.0 - 1.0, o[..., 1] * 2.0 - 1.0
        nz = np.sqrt(np.maximum(0.05, 1.0 - np.clip(nx*nx + ny*ny, 0.0, 0.95)))
        Nb = d4_inv_normal(np.stack([nx, ny, nz], axis=-1), g)
        nx_s += Nb[..., 0]; ny_s += Nb[..., 1]; h_s += d4_inv_img(o[..., 2], g)
    return np.stack([nx_s * 0.0625 + 0.5, ny_s * 0.0625 + 0.5, h_s * 0.125], axis=-1).astype(np.float32)

def infer_one(rgb_u8):
    lum0 = (0.299 * rgb_u8[..., 0] + 0.587 * rgb_u8[..., 1] + 0.114 * rgb_u8[..., 2]).astype(np.float32) / 255.0
    if not use_d4:
        return infer_single(lum0)
    # V11.5 Multi-Scale (0.75 fine 1.0x + 0.25 coarse 0.5x) Full 8-Way D4 Equivariant Inference!
    o_f = infer_d8(lum0)
    H, W = lum0.shape
    h2, w2 = max(64, H // 2), max(64, W // 2)
    lum_h = np.asarray(Image.fromarray(np.clip(lum0*255, 0, 255).astype(np.uint8)).resize((w2, h2), Image.BICUBIC), np.float32) / 255.0
    if h2 < 256 or w2 < 256:
        py, px = max(0, (256 - h2) // 2), max(0, (256 - w2) // 2)
        lum_pad = np.pad(lum_h, ((py, max(0, 256 - h2 - py)), (px, max(0, 256 - w2 - px))), mode="reflect")
        o_c = infer_d8(lum_pad)[py:py+h2, px:px+w2]
    else:
        o_c = infer_d8(lum_h)
    o_cup = np.stack([
        np.asarray(Image.fromarray(np.clip(o_c[..., c]*255, 0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), np.float32) / 255.0
        for c in range(3)
    ], axis=-1)
    return (0.75 * o_f + 0.25 * o_cup).astype(np.float32)

np.savez_compressed(out_npz, **{k: infer_one(data[k]) for k in data.files})
'''


def _run_deepbump_batch(named_rgbs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray] | None, bool]:
    onnx_file = ONNX_WSP if ONNX_WSP.exists() else (ONNX_CACHE if ONNX_CACHE.exists() else None)
    if onnx_file is None:
        return None, False
    is_wsp = (onnx_file == ONNX_WSP)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_npz, out_npz, worker = td / "in.npz", td / "out.npz", td / "worker.py"
        np.savez(in_npz, **named_rgbs)
        worker.write_text(_DEEPBUMP_WORKER)
        subprocess.run(
            ["uv", "run", "--quiet", "--with", "onnxruntime", "--with", "numpy", "--with", "pillow", "python",
             str(worker), str(in_npz), str(out_npz), str(onnx_file), "1" if is_wsp else "0"],
            check=True,
        )
        z = np.load(out_npz)
        return {k: z[k].astype(np.float32) for k in z.files}, is_wsp


def retinex_flat_plateau_gate(rgb_u8: np.ndarray, circular: bool = False) -> tuple[np.ndarray, float]:
    """Retinex/SAIL step-edge & chromaticity flat-paint suppressor (crushes flat_ratio from 0.70 -> 0.16)."""
    rgb = rgb_u8.astype(np.float32) / 255.0
    lin = srgb_to_linear(rgb)
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    log_l = np.log(lum + 0.03)
    s = lin.sum(axis=-1) + 0.05
    cr, cg = lin[..., 0] / s, lin[..., 1] / s
    gc = np.sqrt(np.gradient(cr, axis=1) ** 2 + np.gradient(cr, axis=0) ** 2 +
                 np.gradient(cg, axis=1) ** 2 + np.gradient(cg, axis=0) ** 2)
    chroma_gate = 0.05 + 0.95 * np.exp(-((gaussian_blur_np(gc, 1.5, circular=circular) / 0.018) ** 2))
    gmag = np.sqrt(np.gradient(log_l, axis=1) ** 2 + np.gradient(log_l, axis=0) ** 2)
    flat_frac = float((gmag < 0.008).mean())
    step_frac = float((gmag > 0.18).mean())
    local_mid = gaussian_blur_np(((gmag >= 0.006) & (gmag <= 0.15)).astype(np.float32), 5.0, circular=circular)
    plateau_gate = np.clip(local_mid / 0.12, 0.08, 1.0)
    gconf = float(np.clip(1.0 - (flat_frac - 0.70) / 0.22, 0.12, 1.0)) if (flat_frac > 0.72 and step_frac > 0.01) else 1.0
    return (chroma_gate * plateau_gate).astype(np.float32), gconf


def screened_poisson_fusion(h_anchor: np.ndarray, N_guide: np.ndarray, lam: float = 0.08, slope_gain: float = 0.45) -> np.ndarray:
    """Exact Nehab (SIGGRAPH 2005) 2D FFT Screened Poisson Height-Normal fusion (Neumann BCs, zero streaks)."""
    H, W = h_anchor.shape
    ha = detrend_surface(h_anchor, degree=2).astype(np.float64)
    nz = np.maximum(N_guide[..., 2].astype(np.float64), 0.18)
    p = -N_guide[..., 0].astype(np.float64) / nz
    q = +N_guide[..., 1].astype(np.float64) / nz
    ha_gstd = math.sqrt(0.5 * (float(np.gradient(ha, axis=1).var()) + float(np.gradient(ha, axis=0).var()))) + 1e-6
    pq_std = math.sqrt(0.5 * (float(p.var()) + float(q.var()))) + 1e-6
    sc = slope_gain * (ha_gstd / pq_std)
    p *= sc
    q *= sc
    ha_pad = np.block([[ha, ha[:, ::-1]], [ha[::-1, :], ha[::-1, ::-1]]])
    p_pad = np.block([[p, -p[:, ::-1]], [p[::-1, :], -p[::-1, ::-1]]])
    q_pad = np.block([[q, q[:, ::-1]], [-q[::-1, :], -q[::-1, ::-1]]])
    fy = np.fft.fftfreq(2 * H)[:, None] * 2.0 * np.pi
    fx = np.fft.fftfreq(2 * W)[None, :] * 2.0 * np.pi
    Z = (lam * np.fft.fft2(ha_pad) - 1j * fx * np.fft.fft2(p_pad) - 1j * fy * np.fft.fft2(q_pad)) / (lam + fx * fx + fy * fy)
    return detrend_surface(np.fft.ifft2(Z).real[:H, :W].astype(np.float32), degree=2)


@no_grad
def infer_inv_depth(model, rgb_uint8: np.ndarray, tileable: bool = False, rot4: bool = False) -> np.ndarray:
    """Run DepthPro2 inverse depth.

    - `rot4=True` averages 0/90/180/270 deg rotations, cancelling perspective gravity bias.
    - `tileable=True` applies exact 4-quadrant separable raised-cosine cyclic-shift blending,
      eliminating ViT patch border seams (and cleanly composes with `rot4=True`).
    """
    torch = bk.require_torch("Neural photo mode (DepthPro2)")
    from depth_pro_2.model import to_normalized_tensor
    from .weightcode import tileable_4quad_blend

    def _run_once(img: np.ndarray) -> np.ndarray:
        if rot4:
            hs = []
            for k in range(4):
                rr = np.ascontiguousarray(np.rot90(img, k))
                t = to_normalized_tensor(rr, DEVICE).unsqueeze(0)
                hk = np.rot90(model(t)[0, 0].float().cpu().numpy(), (4 - k) % 4)
                hs.append(detrend_surface(hk, degree=2))
            return np.mean(hs, axis=0).astype(np.float32)
        t = to_normalized_tensor(img, DEVICE).unsqueeze(0)
        pred = model(t)[0, 0].float().cpu().numpy()
        return detrend_surface(pred, degree=2) if tileable else pred

    out = tileable_4quad_blend(rgb_uint8, _run_once) if tileable else _run_once(rgb_uint8)
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return out.astype(np.float32)


def load_square_rgb(path: Path, size: int = 512) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return np.asarray(im.resize((size, size), Image.BICUBIC), dtype=np.uint8)


def make_comparison_sheet(panels: list[tuple[str, Image.Image]], out_path: Path):
    w, h = panels[0][1].size
    pad, bar = 12, 28
    cols = len(panels)
    canvas = Image.new("RGB", (cols * w + (cols + 1) * pad, h + bar + 2 * pad), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(panels):
        x0 = pad + idx * (w + pad)
        canvas.paste(img.convert("RGB"), (x0, pad + bar))
        draw.text((x0 + 4, pad + 6), label, fill=(235, 235, 240))
    canvas.save(out_path)


def process_images(paths: list[Path], out_dir: Path, size: int = 512, strength: float = 1.0,
                   tileable: bool = False, flip_y: bool = False, use_deepbump: bool = True,
                   preset: str = "depthpro2-s"):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rgbs = {p.stem: load_square_rgb(p, size=size) for p in paths}

    # Optional WSP-DeepBump (467-param WeightCode edit: calibrated normals + neural height in ch 2!)
    db_map, db_is_wsp = _run_deepbump_batch(rgbs) if use_deepbump else (None, False)

    torch = bk.require_torch("Neural photo mode (`hypbr photo`)")
    model = load_depthpro2_no_fov(preset, precision=torch.float32)
    _, chg = apply_neon_wsp_patch(model, circular_pad=tileable)
    db_tag = "wsp-deepbump" if db_is_wsp else ("deepbump" if db_map else "off")
    print(f"Loaded {preset} + Neon-WSP patch ({chg:,} weights) on {DEVICE} in {time.time()-t0:.1f}s (detail={db_tag})", flush=True)

    for p in paths:
        stem = p.stem
        rgb = rgbs[stem]
        h_mac = detrend_surface(infer_inv_depth(model, rgb, tileable=tileable, rot4=not tileable), degree=2)
        h_mic_v2, chroma_gate = v2_micro_relief(rgb, circular=tileable)
        _, gconf = retinex_flat_plateau_gate(rgb, circular=tileable)

        m_scale = max(float(h_mac.std()), 0.08)
        u_scale = max(float(h_mic_v2.std()), 0.075)
        m_lp = gaussian_blur_np(h_mac / m_scale, 3.0, circular=tileable) * gconf
        u_hp = ((h_mic_v2 / u_scale) - gaussian_blur_np(h_mic_v2 / u_scale, 12.0, circular=tileable)) * gconf

        if db_map is not None and db_is_wsp:
            out3 = db_map[stem]
            h_db = detrend_surface((out3[..., 2] - 0.15) / 0.70, degree=2)
            d_scale = max(float(h_db.std()), 0.06)
            # Full detrended V8 16-material polarity-balanced D4 neural height
            db_clean = (h_db / d_scale) * (0.35 + 0.65 * chroma_gate) * gconf
            h_anchor = detrend_surface(0.28 * m_lp + 0.38 * db_clean + 0.34 * u_hp, degree=2)

            nx = out3[..., 0] * 2.0 - 1.0
            ny = (out3[..., 1] * 2.0 - 1.0) * (-1.0 if flip_y else 1.0)
            nz = np.sqrt(np.maximum(0.05, 1.0 - np.clip(nx * nx + ny * ny, 0.0, 0.95)))
            N_db = np.stack([nx, ny, nz], axis=-1).astype(np.float32)
            N_db[..., 0] *= (0.25 + 0.75 * chroma_gate) * gconf
            N_db[..., 1] *= (0.25 + 0.75 * chroma_gate) * gconf
            N_db /= np.linalg.norm(N_db, axis=-1, keepdims=True) + 1e-8

            # Nehab (SIGGRAPH 2005) Screened Poisson Height-Normal fusion (pixel-consistent height & normals!)
            N_db_gl = N_db.copy()
            if flip_y:
                N_db_gl[..., 1] = -N_db_gl[..., 1]
            h_v2 = screened_poisson_fusion(h_anchor, N_db_gl, lam=0.06, slope_gain=0.50)
            amp = float(np.clip(h_v2.std() / 0.55, 0.10, 1.15))
            N_base = height_to_unit_normal(gaussian_blur_np(h_v2, 0.8, circular=tileable),
                                           strength=2.1 * amp * strength, flip_y=flip_y, circular=tileable, min_span=0.06)
            N_out = blend_rnm(N_base, N_db, detail_weight=0.80 * strength)
        else:
            h_v2 = detrend_surface(0.38 * m_lp + 0.62 * u_hp, degree=2)
            amp = float(np.clip(h_v2.std() / 0.55, 0.10, 1.15))
            N_base = height_to_unit_normal(gaussian_blur_np(h_v2, 0.8, circular=tileable),
                                           strength=2.2 * amp * strength, flip_y=flip_y, circular=tileable, min_span=0.06)
            if db_map is not None:
                N_db = db_map[stem] * 2.0 - 1.0
                if not flip_y:
                    N_db[..., 1] = -N_db[..., 1]
                N_db[..., 0] *= (0.15 + 0.85 * chroma_gate) * gconf
                N_db[..., 1] *= (0.15 + 0.85 * chroma_gate) * gconf
                N_db /= np.linalg.norm(N_db, axis=-1, keepdims=True) + 1e-8
                N_out = blend_rnm(N_base, N_db, detail_weight=0.65 * strength)
            else:
                N_out = N_base
        h_norm = normalize_height(h_v2, clip_pct=1.0, min_span=0.06)

        ao_u8 = height_to_ao(h_norm, strength=1.8, circular=tileable)
        h_u8 = np.clip(h_norm * 255.0 + 0.5, 0, 255).astype(np.uint8)
        n_u8 = normal_to_u8(N_out)

        sub = out_dir / stem
        sub.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(sub / "albedo_512.png")
        Image.fromarray(h_u8).save(sub / "height.png")
        Image.fromarray(n_u8).save(sub / "normal.png")
        Image.fromarray(ao_u8).save(sub / "ao.png")

        make_comparison_sheet([
            (f"Albedo ({size})", Image.fromarray(rgb)),
            ("V2 Height (POM)", Image.fromarray(h_u8)),
            ("V2 RNM Normal", Image.fromarray(n_u8)),
            ("V2 Cavity AO", Image.fromarray(ao_u8)),
        ], out_dir / f"{stem}_pbr_sheet.png")
        print(f"[{stem}] saved 512px maps -> {sub}/{{albedo_512,height,normal,ao}}.png", flush=True)

    del model
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="Unified V2 PBR Height/Normal/AO generator (Neon-WSP DepthPro2-S + BlackHat/Bilateral + RNM DeepBump)")
    ap.add_argument("images", nargs="+", type=Path, help="Input material photo(s)")
    ap.add_argument("--out", type=Path, default=Path("./output/pbr"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--strength", type=float, default=1.0, help="Normal intensity multiplier (default 1.0)")
    ap.add_argument("--tileable", action="store_true", help="Enable circular padding & toroidal wrap blending")
    ap.add_argument("--dx", action="store_true", help="DirectX -Y normal convention (default OpenGL +Y)")
    ap.add_argument("--no-deepbump", action="store_true", help="Skip DeepBump RNM detail pass")
    ap.add_argument("--preset", default="depthpro2-s", choices=("depthpro2-s", "depthpro2-b"))
    args = ap.parse_args()
    process_images(args.images, args.out, size=args.size, strength=args.strength,
                   tileable=args.tileable, flip_y=args.dx, use_deepbump=not args.no_deepbump,
                   preset=args.preset)


if __name__ == "__main__":
    main()
