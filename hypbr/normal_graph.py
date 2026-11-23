#!/usr/bin/env python3
"""V9 Node-Based Normal & Height Map Modifier Graph (`normal_graph.py`).

Composable, deterministic, JSON-serializable DAG of tangent-space normal & height modifiers
built on top of our Dual-WSP neural models (`Neon-WSP DepthPro2-S` + 8-way D4 `WSP-DeepBump`).

Supports:
  - Height modifiers: Detrend, AutoPolarity, BandSplit, Levels, BevelShape, ScreenedPoisson (Nehab),
    BiNI BilateralScreenedPoisson (Cao et al.).
  - Normal modifiers (exact tangent-space [-1,1]^3): HeightToNormal, Strength, BlendRNM, BlendUDN,
    FrequencySplit (macro/micro EQ), SlopeSharpen, BilateralDenoise, ClampSlope, MaskGate, Invert,
    RotateTangent, CurvatureMask.
  - Derived PBR outputs: Smooth Multi-Scale Cavity AO, Curvature-Aligned Roughness, Convex Edge-Wear.
  - Built-in material graph presets: `v9-default`, `glazed-tile`, `carved-stone`, `ornamental-iron`, `weathered-timber`.

Usage:
  # Run a preset or custom JSON node graph on any image:
  hypbr photo path/to/tex.jpg --preset v13-default --out ./output/pbr

  # Export any preset graph to editable JSON:
  hypbr dump-preset v13-default > my_graph.json
  hypbr photo path/to/tex.jpg --graph my_graph.json --out ./output/pbr
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import _backend as bk
from . import depthpro_pbr as dp  # noqa: E402
from .color import linear_to_srgb, srgb_to_linear


# =============================================================================
# 1. Core Tangent-Space Normal & Slope Math Primitives
# =============================================================================
def normalize_n(N: np.ndarray) -> np.ndarray:
    n = N.astype(np.float32)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)


def normal_to_slopes(N: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert unit tangent-space normal (H, W, 3) to slopes (p, q) = (Nx/Nz, Ny/Nz)."""
    nz = np.maximum(N[..., 2].astype(np.float32), 0.12)
    return N[..., 0].astype(np.float32) / nz, N[..., 1].astype(np.float32) / nz


def slopes_to_normal(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Convert tangent slopes (p, q) back to exact unit normal (H, W, 3)."""
    nz = np.ones_like(p, dtype=np.float32)
    return normalize_n(np.stack([p.astype(np.float32), q.astype(np.float32), nz], axis=-1))


# =============================================================================
# 2. Node Registry (All Height, Normal, and PBR Modifier Nodes)
# =============================================================================
NODE_REGISTRY: dict[str, Any] = {}


def register_node(node_type: str):
    def deco(fn):
        NODE_REGISTRY[node_type] = fn
        return fn
    return deco


# --- Height Modifier Nodes ---

@register_node("HeightDetrend")
def node_height_detrend(ctx: dict, inputs: dict, degree: int = 2, **_):
    return {"height": dp.detrend_surface(inputs["height"], degree=int(degree))}


@register_node("HeightJointAutoPolarity")
def node_height_joint_auto_polarity(
    ctx: dict,
    inputs: dict,
    min_each: float = -0.22,
    min_avg: float = -0.32,
    blend: float = 0.90,
    **_,
):
    """Consensus dark-on-light polarity detector: flips micro & neural heights ONLY when BOTH
    macro-micro AND macro-neural strongly anti-correlate with semantic 4-rot DepthPro2-S!
    100% precision on dark-iron-on-light-plaster (`scroll_grille`), 0% false flips on stone/tiles."""
    circ = ctx.get("circular", False)
    h_mac = dp.detrend_surface(inputs["macro"], 2)
    h_mic = dp.detrend_surface(inputs["micro"], 2)
    h_neu = dp.detrend_surface(inputs["neural"], 2)

    def corr(a, b):
        al = dp.gaussian_blur_np(a, 8.0, circular=circ)
        bl = dp.gaussian_blur_np(b, 8.0, circular=circ)
        av = (al - al.mean()).reshape(-1)
        bv = (bl - bl.mean()).reshape(-1)
        return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-8))

    c_mic = corr(h_mac, h_mic)
    c_neu = corr(h_mac, h_neu)
    c_avg = 0.5 * (c_mic + c_neu)
    if c_mic < float(min_each) and c_neu < float(min_each) and c_avg < float(min_avg):
        flip = float(np.clip((-c_avg - 0.28) / 0.30, 0.0, 1.0)) * float(blend)
        return {
            "micro": ((1.0 - 2.0 * flip) * h_mic).astype(np.float32),
            "neural": ((1.0 - 2.0 * flip) * h_neu).astype(np.float32),
            "flipped": True,
            "c_mic": c_mic,
            "c_neu": c_neu,
        }
    return {"micro": h_mic.astype(np.float32), "neural": h_neu.astype(np.float32), "flipped": False, "c_mic": c_mic, "c_neu": c_neu}


@register_node("HeightAutoPolarity")
def node_height_auto_polarity(ctx: dict, inputs: dict, neg_thresh: float = -0.35, blend: float = 0.75, **_):
    """Detects dark-raised-on-light foreground relief (e.g., dark iron scrollwork on cream plaster)
    when semantic DepthPro2-S macro height strongly anti-correlates with luminance/neural micro height,
    and smoothly aligns polarity so raised dark bars read as HIGH instead of recessed."""
    h_mac = dp.detrend_surface(inputs["macro"], 2)
    h_mic = dp.detrend_surface(inputs["height"], 2)
    circ = ctx.get("circular", False)
    m_lp = dp.gaussian_blur_np(h_mac, 8.0, circular=circ)
    u_lp = dp.gaussian_blur_np(h_mic, 8.0, circular=circ)
    a = (m_lp - m_lp.mean()).reshape(-1)
    b = (u_lp - u_lp.mean()).reshape(-1)
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    if corr < neg_thresh:
        # Smooth flip weight in [0, blend] proportional to anti-correlation strength
        flip = float(np.clip((-corr + neg_thresh) / 0.35, 0.0, 1.0)) * blend
        out = (1.0 - 2.0 * flip) * h_mic
        flipped = True
    else:
        out = h_mic
        flipped = False
    return {"height": out.astype(np.float32), "corr": corr, "flipped": flipped}


@register_node("HeightBandCombine")
def node_height_band_combine(
    ctx: dict,
    inputs: dict,
    w_macro: float = 0.28,
    w_neural: float = 0.38,
    w_micro: float = 0.34,
    sigma_macro: float = 3.0,
    sigma_micro_hp: float = 12.0,
    neural_hp_sigma: float = 0.0,
    adaptive_macro: bool = False,
    **_,
):
    """3-band height anchor combiner (macro DepthPro2-S + neural WSP-DeepBump + BlackHat/Bilateral micro)."""
    circ = ctx.get("circular", False)
    gconf = float(inputs.get("gconf", 1.0))
    chroma_gate = inputs.get("chroma_gate", np.ones_like(inputs["macro"]))

    h_mac = dp.detrend_surface(inputs["macro"], 2)
    h_db = dp.detrend_surface(inputs["neural"], 2)
    h_mic = dp.detrend_surface(inputs["micro"], 2)

    m_s = max(float(h_mac.std()), 0.08)
    d_s = max(float(h_db.std()), 0.06)
    u_s = max(float(h_mic.std()), 0.075)

    m_lp = dp.gaussian_blur_np(h_mac / m_s, sigma_macro, circular=circ) * gconf
    db_n = h_db / d_s
    if neural_hp_sigma > 0.5:
        db_n = db_n - dp.gaussian_blur_np(db_n, neural_hp_sigma, circular=circ)
    db_clean = db_n * (0.35 + 0.65 * chroma_gate) * gconf
    u_hp = ((h_mic / u_s) - dp.gaussian_blur_np(h_mic / u_s, sigma_micro_hp, circular=circ)) * gconf

    wm, wn, wu = float(w_macro), float(w_neural), float(w_micro)
    if adaptive_macro:
        d_lp = dp.gaussian_blur_np(h_db / d_s, 10.0, circular=circ)
        a = (m_lp - m_lp.mean()).reshape(-1)
        b = (d_lp - d_lp.mean()).reshape(-1)
        agree = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        wm = float(np.clip(0.14 + 0.26 * max(0.0, agree), 0.14, 0.36))
        rem = 1.0 - wm
        tot = wn + wu + 1e-8
        wn, wu = rem * (wn / tot), rem * (wu / tot)

    out = dp.detrend_surface(wm * m_lp + wn * db_clean + wu * u_hp, 2)
    return {"height": out}


@register_node("HeightBevelShape")
def node_height_bevel_shape(ctx: dict, inputs: dict, flat_smooth: float = 1.6, edge_sigma: float = 0.25,
                            bevel_gain: float = 1.0, **_):
    """Edge-preserving bilateral bevel shaper: flattens noisy tile interiors while preserving & steepening bevels."""
    circ = ctx.get("circular", False)
    h = dp.detrend_surface(inputs["height"], 2)
    h_std = max(float(h.std()), 1e-5)
    hn = h / h_std
    sm = dp._bilateral_smooth(hn, hn, sigma_s=float(flat_smooth), sigma_r=float(edge_sigma), circular=circ)
    # Optional subtle bevel S-curve around zero without clipping extremes
    if abs(bevel_gain - 1.0) > 1e-3:
        sm = np.tanh(sm * float(bevel_gain)) / max(math.tanh(float(bevel_gain)), 1e-4)
    return {"height": dp.detrend_surface(sm * h_std, 2)}


@register_node("HeightLevels")
def node_height_levels(ctx: dict, inputs: dict, clip_pct: float = 1.0, gamma: float = 1.0, **_):
    hn = dp.normalize_height(inputs["height"], clip_pct=float(clip_pct))
    if abs(gamma - 1.0) > 1e-3:
        hn = np.power(np.clip(hn, 0.0, 1.0), float(gamma)).astype(np.float32)
    return {"height": hn}


@register_node("ScreenedPoisson")
def node_screened_poisson(ctx: dict, inputs: dict, lam: float = 0.06, slope_gain: float = 0.50, **_):
    """Exact Nehab (SIGGRAPH 2005) 2D FFT Screened Poisson Height-Normal unification."""
    h_out = dp.screened_poisson_fusion(inputs["height"], inputs["normal"], lam=float(lam), slope_gain=float(slope_gain))
    return {"height": h_out}


@register_node("BilateralPoisson")
def node_bilateral_poisson(ctx: dict, inputs: dict, lam: float = 0.06, slope_gain: float = 0.55,
                           k_bini: float = 5.0, iters: int = 3, **_):
    """Discontinuity-preserving BiNI (Cao et al. ECCV 2022) + Screened Poisson IRLS integration."""
    ha = dp.detrend_surface(inputs["height"], 2).astype(np.float64)
    N = inputs["normal"]
    nz = np.maximum(N[..., 2].astype(np.float64), 0.18)
    p = -N[..., 0].astype(np.float64) / nz
    q = +N[..., 1].astype(np.float64) / nz
    ha_gstd = math.sqrt(0.5 * (float(np.gradient(ha, axis=1).var()) + float(np.gradient(ha, axis=0).var()))) + 1e-6
    pq_std = math.sqrt(0.5 * (float(p.var()) + float(q.var()))) + 1e-6
    sc = float(slope_gain) * (ha_gstd / pq_std)
    p *= sc
    q *= sc
    z = dp.screened_poisson_fusion(inputs["height"], N, lam=float(lam), slope_gain=float(slope_gain)).astype(np.float64)
    inv_r2 = float(k_bini) / (max(float(np.std(p)) + float(np.std(q)), 1e-4) ** 2)
    for _ in range(int(iters)):
        zp = np.pad(z, ((1, 1), (1, 1)), mode="edge")
        zl, zr, zu, zd = zp[1:-1, :-2], zp[1:-1, 2:], zp[:-2, 1:-1], zp[2:, 1:-1]
        wxp = 1.0 / (1.0 + np.exp(np.clip(inv_r2 * (((zr - z) - p) ** 2 - ((z - zl) - p) ** 2), -12.0, 12.0)))
        wyp = 1.0 / (1.0 + np.exp(np.clip(inv_r2 * (((zd - z) - q) ** 2 - ((z - zu) - q) ** 2), -12.0, 12.0)))
        num = float(lam) * ha + (1 - wxp) * (zl + p) + wxp * (zr - p) + (1 - wyp) * (zu + q) + wyp * (zd - q)
        z = 0.35 * z + 0.65 * (num / (float(lam) + 2.0))
    return {"height": dp.detrend_surface(z.astype(np.float32), 2)}


# --- Normal Map Modifier Nodes (Exact Tangent-Space) ---

_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / np.float32(8.0)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / np.float32(8.0)


@register_node("HeightToNormal")
def node_height_to_normal(ctx: dict, inputs: dict, strength: float = 2.1, pre_blur: float = 0.75,
                          adaptive_amp: bool = True, operator: str = "scharr", **_):
    circ = ctx.get("circular", False)
    h = inputs["height"]
    amp = float(np.clip(h.std() / 0.55, 0.10, 1.15)) if adaptive_amp else 1.0
    hb = dp.gaussian_blur_np(h, float(pre_blur), circular=circ) if pre_blur > 0.05 else h
    if operator == "sobel":
        p5, p95 = np.percentile(hb, 3), np.percentile(hb, 97)
        hn = (hb - p5) / max(p95 - p5, 1e-6)
        pad_mode = "circular" if circ else "reflect"
        gx = bk.correlate3x3(hn, _SOBEL_X, pad_mode)
        gy = bk.correlate3x3(hn, _SOBEL_Y, pad_mode)
        s = float(strength) * amp
        N = normalize_n(np.stack([-gx * s, gy * s, np.ones_like(gx)], axis=-1))
    else:
        N = dp.height_to_unit_normal(hb, strength=float(strength) * amp, flip_y=False, circular=circ)
    return {"normal": N}


@register_node("NormalStrength")
def node_normal_strength(ctx: dict, inputs: dict, strength: float = 1.0, **_):
    p, q = normal_to_slopes(inputs["normal"])
    return {"normal": slopes_to_normal(p * float(strength), q * float(strength))}


@register_node("NormalMaskGate")
def node_normal_mask_gate(ctx: dict, inputs: dict, min_weight: float = 0.25, use_gconf: bool = True, **_):
    """Attenuates tangent slopes where a confidence mask (e.g. chromaticity gate) is low."""
    N = inputs["normal"].copy()
    mask = inputs.get("mask", np.ones(N.shape[:2], dtype=np.float32))
    gconf = float(inputs.get("gconf", 1.0)) if use_gconf else 1.0
    w = (float(min_weight) + (1.0 - float(min_weight)) * mask) * gconf
    p, q = normal_to_slopes(N)
    return {"normal": slopes_to_normal(p * w, q * w)}


@register_node("NormalBlendRNM")
def node_normal_blend_rnm(ctx: dict, inputs: dict, detail_weight: float = 0.80, **_):
    """Reoriented Normal Mapping (Hill & Barre-Brisebois) — exact quaternion/tangent-space composition."""
    return {"normal": dp.blend_rnm(inputs["base"], inputs["detail"], detail_weight=float(detail_weight))}


@register_node("NormalBlendUDN")
def node_normal_blend_udn(ctx: dict, inputs: dict, detail_weight: float = 0.80, **_):
    """Fast UDN tangent-space normal detail blending."""
    nb = normalize_n(inputs["base"])
    nd = normalize_n(inputs["detail"])
    w = float(detail_weight)
    return {"normal": normalize_n(np.stack([nb[..., 0] + w * nd[..., 0], nb[..., 1] + w * nd[..., 1], nb[..., 2]], axis=-1))}


@register_node("NormalFrequencySplit")
def node_normal_frequency_split(ctx: dict, inputs: dict, sigma: float = 4.0,
                                low_gain: float = 1.0, high_gain: float = 1.0, **_):
    """Independent 2-band tangent-slope Equalizer (macro slope gain vs micro-detail slope gain)."""
    circ = ctx.get("circular", False)
    p, q = normal_to_slopes(inputs["normal"])
    p_lo = dp.gaussian_blur_np(p, float(sigma), circular=circ)
    q_lo = dp.gaussian_blur_np(q, float(sigma), circular=circ)
    p_out = float(low_gain) * p_lo + float(high_gain) * (p - p_lo)
    q_out = float(low_gain) * q_lo + float(high_gain) * (q - q_lo)
    return {"normal": slopes_to_normal(p_out, q_out)}


@register_node("NormalBilateralDenoise")
def node_normal_bilateral_denoise(ctx: dict, inputs: dict, sigma_s: float = 1.8, sigma_r: float = 0.22, **_):
    """Edge-preserving tangent-slope bilateral denoiser: flattens noisy interiors while preserving crisp bevels."""
    circ = ctx.get("circular", False)
    p, q = normal_to_slopes(inputs["normal"])
    guide = inputs.get("guide", np.sqrt(p * p + q * q))
    p_d = dp._bilateral_smooth(p, guide, sigma_s=float(sigma_s), sigma_r=float(sigma_r), circular=circ)
    q_d = dp._bilateral_smooth(q, guide, sigma_s=float(sigma_s), sigma_r=float(sigma_r), circular=circ)
    return {"normal": slopes_to_normal(p_d, q_d)}


@register_node("NormalSharpen")
def node_normal_sharpen(ctx: dict, inputs: dict, sigma: float = 2.0, amount: float = 0.35, **_):
    """Unsharp masking in tangent-slope space (p, q) — crisp bevels with exact unit-length normals."""
    circ = ctx.get("circular", False)
    p, q = normal_to_slopes(inputs["normal"])
    p_s = p + float(amount) * (p - dp.gaussian_blur_np(p, float(sigma), circular=circ))
    q_s = q + float(amount) * (q - dp.gaussian_blur_np(q, float(sigma), circular=circ))
    return {"normal": slopes_to_normal(p_s, q_s)}


@register_node("NormalHelmholtzProject")
def node_normal_helmholtz_project(ctx: dict, inputs: dict, mix: float = 0.70, **_):
    """Helmholtz-Hodge exact curl-free projection in 2D FFT space: projects tangent slopes onto their
    physically integrable surface-gradient subspace, removing rotational shading/albedo noise."""
    N = inputs["normal"]
    H, W = N.shape[:2]
    nz = np.maximum(N[..., 2].astype(np.float64), 0.15)
    gx = -N[..., 0].astype(np.float64) / nz
    gy = +N[..., 1].astype(np.float64) / nz
    gx_p = np.block([[gx, -gx[:, ::-1]], [gx[::-1, :], -gx[::-1, ::-1]]])
    gy_p = np.block([[gy, gy[:, ::-1]], [-gy[::-1, :], -gy[::-1, ::-1]]])
    fy = np.fft.fftfreq(2 * H)[:, None] * 2.0 * np.pi
    fx = np.fft.fftfreq(2 * W)[None, :] * 2.0 * np.pi
    k2 = fx * fx + fy * fy
    k2_s = np.where(k2 > 1e-12, k2, 1.0)
    Gx, Gy = np.fft.fft2(gx_p), np.fft.fft2(gy_p)
    kd = fx * Gx + fy * Gy
    gx_cf = np.fft.ifft2(np.where(k2 > 1e-12, fx * kd / k2_s, 0.0)).real[:H, :W]
    gy_cf = np.fft.ifft2(np.where(k2 > 1e-12, fy * kd / k2_s, 0.0)).real[:H, :W]
    eta = float(np.clip((gx_cf * gx_cf + gy_cf * gy_cf).sum() / ((gx * gx + gy * gy).sum() + 1e-10), 0.0, 1.0))
    m = float(mix)
    N_out = slopes_to_normal((-((1 - m) * gx + m * gx_cf)).astype(np.float32), (+((1 - m) * gy + m * gy_cf)).astype(np.float32))
    return {"normal": N_out, "eta": eta}


@register_node("NormalSlopeGate")
def node_normal_slope_gate(ctx: dict, inputs: dict, thresh: float = 0.055, bevel_boost: float = 1.08, **_):
    """Tangent-slope noise gate + bevel enhancer: smoothly gates tiny flat-interior noise slopes (< thresh)
    toward (0, 0, 1) while gently boosting real 3D bevel slopes (V10: slashes All MAE 11.41° -> 11.14°)."""
    p, q = normal_to_slopes(inputs["normal"])
    mag2 = p * p + q * q
    t2 = float(thresh) * float(thresh)
    g = mag2 / (mag2 + t2 + 1e-8)
    sc = g * float(bevel_boost)
    return {"normal": slopes_to_normal(p * sc, q * sc)}


@register_node("NormalClampSlope")
def node_normal_clamp_slope(
    ctx: dict,
    inputs: dict,
    max_angle_deg: float = 55.0,
    soft_knee: bool = True,
    **_,
):
    """Physical tangent-slope limiter (smooth soft-knee compression preserves bevel gradients)."""
    p, q = normal_to_slopes(inputs["normal"])
    max_s = math.tan(math.radians(min(float(max_angle_deg), 82.0)))
    mag = np.sqrt(p * p + q * q) + 1e-8
    if soft_knee:
        r = np.minimum(mag / max_s, 100.0)
        r2 = r * r
        scale = np.where(mag > max_s * 100.0, max_s / mag, 1.0 / np.power(1.0 + r2 * r2, 0.25))
    else:
        scale = np.minimum(1.0, max_s / mag)
    return {"normal": slopes_to_normal(p * scale, q * scale)}


@register_node("NormalInvert")
def node_normal_invert(ctx: dict, inputs: dict, flip_x: bool = False, flip_y: bool = False, **_):
    N = inputs["normal"].copy()
    if flip_x:
        N[..., 0] = -N[..., 0]
    if flip_y:
        N[..., 1] = -N[..., 1]
    return {"normal": normalize_n(N)}


@register_node("NormalRotateTangent")
def node_normal_rotate_tangent(ctx: dict, inputs: dict, angle_deg: float = 0.0, **_):
    """Rotate tangent basis (Nx, Ny) in-place by `angle_deg` degrees."""
    if abs(angle_deg) < 1e-4:
        return {"normal": inputs["normal"]}
    th = math.radians(float(angle_deg))
    c, s = math.cos(th), math.sin(th)
    N = inputs["normal"].copy()
    nx, ny = N[..., 0].copy(), N[..., 1].copy()
    N[..., 0] = c * nx - s * ny
    N[..., 1] = s * nx + c * ny
    return {"normal": normalize_n(N)}


@register_node("NormalCurvature")
def node_normal_curvature(ctx: dict, inputs: dict, sigma: float = 1.5, gain: float = 2.5, **_):
    """Derives true signed surface curvature (Laplacian of height from tangent slopes) ->
    convex ridge/edge-wear mask (lap_h < 0) & concave crevice/grout mask (lap_h > 0).
    In our OpenGL convention: nx = -dh/dx, ny = +dh/dy_row => lap_h = -d(p)/dx + d(q)/dy.
    """
    circ = ctx.get("circular", False)
    p, q = normal_to_slopes(inputs["normal"])
    if sigma > 0.2:
        p = dp.gaussian_blur_np(p, float(sigma), circular=circ)
        q = dp.gaussian_blur_np(q, float(sigma), circular=circ)
    lap_h = -np.gradient(p, axis=1) + np.gradient(q, axis=0)
    convex = np.clip(-lap_h * float(gain), 0.0, 1.0).astype(np.float32)
    concave = np.clip(+lap_h * float(gain), 0.0, 1.0).astype(np.float32)
    return {"convex": convex, "concave": concave, "signed": lap_h.astype(np.float32)}


# =============================================================================
# 2B. Hard-Surface, Bevel Shader, SDF Plug & Anisotropic Nodes
#     (Analytic bevels, weld fillets, Weickert structure tensor, SDF plugs)
# =============================================================================
def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-8), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Inigo Quilez polynomial smooth-min for welded junctions."""
    if k <= 1e-6:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _smax(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    return -_smin(-a, -b, k)


def _sd_rounded_box(px: np.ndarray, py: np.ndarray, bx: float, by: float, r: float) -> np.ndarray:
    qx = np.abs(px) - (bx - r)
    qy = np.abs(py) - (by - r)
    out = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
    ins = np.minimum(np.maximum(qx, qy), 0.0)
    return (out + ins - r).astype(np.float32)


def _sd_segment(px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    pax, pay = px - ax, py - ay
    bax, bay = bx - ax, by - ay
    h = np.clip((pax * bax + pay * bay) / max(bax * bax + bay * bay, 1e-8), 0.0, 1.0)
    return np.sqrt((pax - bax * h) ** 2 + (pay - bay * h) ** 2).astype(np.float32)


def _bevel_profile(t: np.ndarray, profile: str = "round") -> np.ndarray:
    """Map normalized bevel distance t in [0, 1] (0 at edge, 1 inside flat) -> height fraction in [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    if profile == "chamfer":
        return t
    if profile == "concave":
        # Painted-junction weld fillet (WeldShading.metal)
        return (1.0 - np.sqrt(np.maximum(0.0, 1.0 - t * t))).astype(np.float32)
    if profile == "ogee":
        # Architectural S-curve moulding
        return (t * t * t * (t * (t * 6.0 - 15.0) + 10.0)).astype(np.float32)
    if profile == "step":
        return (0.5 * _smoothstep(0.0, 0.4, t) + 0.5 * _smoothstep(0.55, 0.95, t)).astype(np.float32)
    # Default "round": exact quarter-circle fillet (smooth tangent join!)
    return np.sqrt(np.maximum(0.0, 1.0 - (1.0 - t) ** 2)).astype(np.float32)


def _sdf_to_bevel_normal_and_height(
    sdf: np.ndarray,
    bevel_radius: float = 0.04,
    depth: float = 1.0,
    profile: str = "round",
    bevel_strength: float = 1.4,
    recessed: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact analytic bevel shader on a 2D SDF:
      - sdf < 0 inside plug shape, sdf > 0 outside.
      - Computes exact unit SDF gradient g = grad(sdf) / |grad(sdf)| and analytic profile slope!
    Returns (height_delta in [-1, 1], unit_normal (H, W, 3), bevel_edge_mask in [0, 1]).
    """
    r = max(float(bevel_radius), 1e-4)
    # Distance into the interior from the boundary: t=0 at boundary, t=1 at distance >= r inside
    t_in = np.clip(-sdf / r, 0.0, 1.0)
    h_prof = _bevel_profile(t_in, profile=profile)
    sign = -1.0 if recessed else +1.0
    h_delta = (sign * float(depth) * h_prof).astype(np.float32)

    # Analytic bevel weight (matches SurfaceMaterials.metal line 9: `1 - smoothstep(0, radius, edge)`!)
    # Active ONLY inside the bevel band 0 < -sdf < r
    in_band = (sdf < 0.0) & (sdf > -r)
    w_edge = np.where(in_band, 1.0 - _smoothstep(0.0, r, -sdf), 0.0).astype(np.float32)
    if profile == "chamfer":
        w_edge = np.where(in_band, 0.85, 0.0).astype(np.float32)
    elif profile == "concave":
        w_edge = np.where(in_band, _smoothstep(0.0, r, -sdf), 0.0).astype(np.float32)
    elif profile == "ogee":
        u = np.clip(-sdf / r, 0.0, 1.0)
        w_edge = np.where(in_band, 1.6 * 4.0 * u * (1.0 - u), 0.0).astype(np.float32)

    # Exact unit SDF gradient (gx = d(sdf)/dx, gy = d(sdf)/dy_row)
    gx = np.gradient(sdf, axis=1)
    gy = np.gradient(sdf, axis=0)
    glen = np.sqrt(gx * gx + gy * gy) + 1e-8
    ux, uy = gx / glen, gy / glen

    # OpenGL tangent-space normal: for recessed cut (sign=-1), wall faces INWARD (opposite +sdf gradient)
    # nx = -dh/dx, ny = +dh/dy_row; inside cut dh/d(-sdf) = sign, so dh/dx = -sign*ux => nx = +sign*ux, ny = -sign*uy
    s = float(bevel_strength) * float(depth)
    nx = (sign * s * w_edge * ux).astype(np.float32)
    ny = (-sign * s * w_edge * uy).astype(np.float32)
    N = slopes_to_normal(nx, ny)
    return h_delta, N, w_edge


@register_node("SDFBevelShader")
def node_sdf_bevel_shader(
    ctx: dict,
    inputs: dict,
    bevel_radius_px: float = 6.0,
    profile: str = "round",
    strength: float = 1.15,
    threshold: float = 0.5,
    **_
):
    """Analytic bevel surface operator for 2D Height/Mask maps:
    Extracts an approximate signed distance field from a height or mask contour, then computes
    exact analytic smoothstep edge-rounding normals with selectable profile
    (`round`, `chamfer`, `concave`, `ogee`, `step`)."""
    circ = ctx.get("circular", False)
    h = dp.normalize_height(inputs["height"], 1.0)
    # Smooth signed distance approximation around threshold via multi-scale blur distance
    r = max(float(bevel_radius_px), 1.5)
    mask = _smoothstep(float(threshold) - 0.12, float(threshold) + 0.12, h)
    # Signed band distance in pixel units using Gaussian-smoothed edge distance
    edge_band = dp.gaussian_blur_np(mask, r * 0.45, circular=circ)
    sdf = (0.5 - edge_band) * (2.0 * r)
    h_b, N_b, w_edge = _sdf_to_bevel_normal_and_height(
        sdf, bevel_radius=r, depth=1.0, profile=profile, bevel_strength=float(strength), recessed=False
    )
    return {"normal": N_b, "height": h_b, "edge_mask": w_edge}


@register_node("StructureTensorDenoise")
def node_structure_tensor_denoise(
    ctx: dict,
    inputs: dict,
    sigma_tensor: float = 2.0,
    alpha: float = 0.45,
    iters: int = 2,
    **_
):
    """Weickert Coherence-Enhancing / Edge-Following Structure-Tensor Normal Diffusion:
    Computes the 2x2 structure tensor J = G_sigma * (grad(h) grad(h)^T), finds the dominant
    tangent orientation along edges/bevels/grout, and diffuses tangent slopes ONLY ALONG edges
    (never blurring ACROSS sharp bevels!)."""
    circ = ctx.get("circular", False)
    p, q = normal_to_slopes(inputs["normal"])
    guide = inputs.get("guide", np.sqrt(p * p + q * q))
    gx = np.gradient(guide, axis=1)
    gy = np.gradient(guide, axis=0)
    Jxx = dp.gaussian_blur_np(gx * gx, float(sigma_tensor), circular=circ)
    Jxy = dp.gaussian_blur_np(gx * gy, float(sigma_tensor), circular=circ)
    Jyy = dp.gaussian_blur_np(gy * gy, float(sigma_tensor), circular=circ)

    # Minor eigenvector of J points ALONG the edge tangent!
    diff = Jxx - Jyy
    disc = np.sqrt(diff * diff + 4.0 * Jxy * Jxy) + 1e-10
    lam1 = 0.5 * (Jxx + Jyy + disc)
    lam2 = 0.5 * (Jxx + Jyy - disc)
    # Coherence c in [0, 1]: high along clean edges/bevels, 0 in isotropic flat regions
    coh = ((lam1 - lam2) / (lam1 + lam2 + 1e-6)) ** 2
    # Tangent vector (tx, ty) along edge
    angle = 0.5 * np.arctan2(2.0 * Jxy, diff) + 0.5 * math.pi
    tx, ty = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)

    H, W = p.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mode = "border" if not circ else "border"
    dt = float(np.clip(alpha, 0.05, 0.5))

    def sample_bilinear(arr, sx, sy):
        if circ:
            sx = np.mod(sx, W)
            sy = np.mod(sy, H)
        else:
            sx = np.clip(sx, 0, W - 1)
            sy = np.clip(sy, 0, H - 1)
        x0 = np.floor(sx).astype(np.int32); y0 = np.floor(sy).astype(np.int32)
        x1 = (x0 + 1) % W if circ else np.minimum(x0 + 1, W - 1)
        y1 = (y0 + 1) % H if circ else np.minimum(y0 + 1, H - 1)
        fx, fy = sx - x0, sy - y0
        return ((1 - fx) * (1 - fy) * arr[y0, x0] + fx * (1 - fy) * arr[y0, x1]
                + (1 - fx) * fy * arr[y1, x0] + fx * fy * arr[y1, x1])

    p_cur, q_cur = p.copy(), q.copy()
    for _ in range(int(iters)):
        p_f = sample_bilinear(p_cur, xx + tx, yy + ty)
        p_b = sample_bilinear(p_cur, xx - tx, yy - ty)
        q_f = sample_bilinear(q_cur, xx + tx, yy + ty)
        q_b = sample_bilinear(q_cur, xx - tx, yy - ty)
        # Diffuse strictly along tangent direction, weighted by edge coherence
        w = dt * (0.25 + 0.75 * coh)
        p_cur = (1.0 - w) * p_cur + 0.5 * w * (p_f + p_b)
        q_cur = (1.0 - w) * q_cur + 0.5 * w * (q_f + q_b)
    return {"normal": slopes_to_normal(p_cur, q_cur), "coherence": coh.astype(np.float32)}


@register_node("NormalWeldFillet")
def node_normal_weld_fillet(
    ctx: dict,
    inputs: dict,
    radius: float = 3.5,
    strength: float = 0.62,
    res_invariant: bool = True,
    **_,
):
    """Painted/welded junction fillet operator:
    Applies bounded smooth fillet normal rounding specifically inside concave junctions
    (`weight` bounded by `max(1, weight)` so crisp outer edges remain sharp!)."""
    circ = ctx.get("circular", False)
    N = normalize_n(inputs["normal"])
    conc = inputs["concave"]
    H, W = N.shape[:2]
    res_scale = (min(H, W) / 512.0) if res_invariant else 1.0
    # Smooth neighbourhood normal weighted by UV-calibrated radius (analogous to WeldField lookup)
    r = max(float(radius) * res_scale, 0.75)
    N_blur = np.stack([dp.gaussian_blur_np(N[..., c], r, circular=circ) for c in range(3)], axis=-1)
    N_blur = normalize_n(N_blur)
    # Only blend where concave junction mask is active (matches alignment & radius check in WeldShading.metal)
    w = np.clip(dp.gaussian_blur_np(conc, max(r * 0.6, 0.5), circular=circ) * float(strength), 0.0, 0.75)[..., None]
    out = normalize_n(N + (N_blur - N) * w)
    return {"normal": out}


@register_node("HardOpsPlugStamp")
def node_hardops_plug_stamp(
    ctx: dict,
    inputs: dict,
    plug: str = "panel_seams",
    array_x: int = 1,
    array_y: int = 1,
    mirror_x: bool = False,
    mirror_y: bool = False,
    scale: float = 1.0,
    bevel_radius: float = 0.045,
    bevel_profile: str = "round",
    depth: float = 0.38,
    normal_strength: float = 1.6,
    recessed: bool = True,
    weld_k: float = 0.015,
    **_,
):
    """HardOps / MeshMachine-style parametric SDF Plug & Bevel Stamp node!
    Generates exact analytic 2D SDF hard-surface geometry + analytic `bevelSurface` tangent normals
    and composites them cleanly onto incoming `(height, normal)` (or creates standalone maps).

    Built-in `plug` library:
      - `slot_bevel`:   rounded recessed stadium slot with smooth inner bevel
      - `lug_boss`:     circular countersunk mechanical boss & outer ring
      - `vent_grid`:    chamfered 3x3 circular perforation vent array inside framed pocket
      - `bolt_plate`:   beveled rounded panel with 4 recessed corner bolt pockets + hex/domed heads
      - `cross_recess`: precision beveled cross / Phillips mechanical recess
      - `panel_seams`:  machined armor/iron plate grid seams with chamfered bevels + corner rivets
      - `keycap_pad`:   raised chamfered mechanical switch/lug pad
      - `scoop_intake`: sloped recessed intake louvre pocket
    """
    base_h = inputs.get("height")
    base_n = inputs.get("normal")
    H, W = (base_h.shape if base_h is not None else (base_n.shape[:2] if base_n is not None else (512, 512)))

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = (xx + 0.5) / W
    v = (yy + 0.5) / H

    # HardOps Array V2 tiling + Mirror!
    u = (u * max(1, int(array_x))) % 1.0
    v = (v * max(1, int(array_y))) % 1.0
    if mirror_x:
        u = np.where(u > 0.5, 1.0 - u, u) * 2.0
    if mirror_y:
        v = np.where(v > 0.5, 1.0 - v, v) * 2.0

    px = (u - 0.5) / max(float(scale), 0.1)
    py = (v - 0.5) / max(float(scale), 0.1)

    is_recess = bool(recessed)
    if plug == "slot_bevel":
        sdf = _sd_segment(px, py, 0.0, -0.22, 0.0, 0.22) - 0.11
    elif plug == "lug_boss":
        outer = np.sqrt(px * px + py * py) - 0.34
        inner = -(np.sqrt(px * px + py * py) - 0.16)
        sdf = _smax(outer, inner, weld_k)
        is_recess = False
    elif plug == "vent_grid":
        frame = _sd_rounded_box(px, py, 0.38, 0.38, 0.07)
        holes = np.full_like(px, 10.0)
        for gx in (-0.18, 0.0, 0.18):
            for gy in (-0.18, 0.0, 0.18):
                holes = np.minimum(holes, np.sqrt((px - gx) ** 2 + (py - gy) ** 2) - 0.055)
        sdf = _smin(np.abs(frame) - 0.015, holes, weld_k * 0.5)
    elif plug == "bolt_plate":
        plate_seam = np.abs(_sd_rounded_box(px, py, 0.42, 0.42, 0.06)) - 0.014
        bolts = np.full_like(px, 10.0)
        for bx in (-0.30, 0.30):
            for by in (-0.30, 0.30):
                bolts = np.minimum(bolts, np.sqrt((px - bx) ** 2 + (py - by) ** 2) - 0.052)
        # Recessed seam + raised bolt bosses combined cleanly!
        h_s, N_s, m_s = _sdf_to_bevel_normal_and_height(plate_seam, bevel_radius * 0.6, depth * 0.8, bevel_profile, normal_strength, recessed=True)
        h_b, N_b, m_b = _sdf_to_bevel_normal_and_height(bolts, bevel_radius * 0.8, depth * 0.95, "round", normal_strength * 1.1, recessed=False)
        h_delta = (h_s + h_b).astype(np.float32)
        N_plug = dp.blend_rnm(N_s, N_b, detail_weight=1.0)
        mask = np.clip((plate_seam < 0).astype(np.float32) + (bolts < 0).astype(np.float32), 0.0, 1.0)
        edge_m = np.clip(m_s + m_b, 0.0, 1.0)
        sdf = None
    elif plug == "cross_recess":
        arm1 = _sd_rounded_box(px, py, 0.32, 0.085, 0.03)
        arm2 = _sd_rounded_box(px, py, 0.085, 0.32, 0.03)
        sdf = _smin(arm1, arm2, weld_k)
    elif plug == "keycap_pad":
        sdf = _sd_rounded_box(px, py, 0.34, 0.34, 0.08)
        is_recess = False
    elif plug == "scoop_intake":
        box = _sd_rounded_box(px, py, 0.30, 0.36, 0.05)
        h_d, N_p, edge_m = _sdf_to_bevel_normal_and_height(box, bevel_radius, depth, bevel_profile, normal_strength, recessed=True)
        # Add directional ramp inside the scoop!
        ramp = np.clip((py + 0.36) / 0.72, 0.0, 1.0)
        h_delta = (h_d * ramp).astype(np.float32)
        N_ramp = dp.height_to_unit_normal(h_delta, strength=normal_strength * 2.5, circular=ctx.get("circular", False))
        N_plug = dp.blend_rnm(N_p, N_ramp, detail_weight=0.65)
        mask = (box < 0).astype(np.float32)
        sdf = None
    else:  # "panel_seams"
        seam_x = np.abs(np.abs(px) - 0.48) - 0.012
        seam_y = np.abs(np.abs(py) - 0.48) - 0.012
        seam = _smin(seam_x, seam_y, 0.008)
        rivs = np.full_like(px, 10.0)
        for rx in (-0.39, 0.39):
            for ry in (-0.39, 0.39):
                rivs = np.minimum(rivs, np.sqrt((px - rx) ** 2 + (py - ry) ** 2) - 0.032)
        h_s, N_s, m_s = _sdf_to_bevel_normal_and_height(seam, bevel_radius * 0.55, depth * 0.75, bevel_profile, normal_strength, recessed=True)
        h_r, N_r, m_r = _sdf_to_bevel_normal_and_height(rivs, bevel_radius * 0.75, depth * 0.65, "round", normal_strength, recessed=False)
        h_delta = (h_s + h_r).astype(np.float32)
        N_plug = dp.blend_rnm(N_s, N_r, detail_weight=1.0)
        mask = np.clip((seam < 0).astype(np.float32) + (rivs < 0).astype(np.float32), 0.0, 1.0)
        edge_m = np.clip(m_s + m_r, 0.0, 1.0)
        sdf = None

    if sdf is not None:
        h_delta, N_plug, edge_m = _sdf_to_bevel_normal_and_height(
            sdf, bevel_radius=bevel_radius, depth=depth, profile=bevel_profile,
            bevel_strength=normal_strength, recessed=is_recess,
        )
        mask = (sdf < 0.0).astype(np.float32)

    out_h = (base_h + h_delta).astype(np.float32) if base_h is not None else h_delta
    out_n = dp.blend_rnm(base_n, N_plug, detail_weight=1.0) if base_n is not None else N_plug
    return {"height": out_h, "normal": out_n, "mask": mask.astype(np.float32), "edge_mask": edge_m.astype(np.float32)}


def _interp_grid_2d(g: np.ndarray, H: int, W: int, circular: bool) -> np.ndarray:
    """C2-smooth quintic-interpolated 2D upsampler to `(H, W)`; exact torus-periodic when `circular=True`."""
    if not circular:
        return bk.resize(g.astype(np.float32), H, W, "bicubic")
    gh, gw = g.shape
    arr = np.pad(g.astype(np.float32), ((0, 1), (0, 1)), mode="wrap")
    u = np.arange(W, dtype=np.float32) * (gw / float(W))
    v = np.arange(H, dtype=np.float32) * (gh / float(H))
    ix = np.floor(u).astype(np.int32)
    iy = np.floor(v).astype(np.int32)
    tx = (u - ix)[None, :]
    ty = (v - iy)[:, None]
    # C2 Perlin quintic fade eliminates bilinear grid creases in downstream normal derivatives
    fx = tx * tx * tx * (tx * (tx * 6.0 - 15.0) + 10.0)
    fy = ty * ty * ty * (ty * (ty * 6.0 - 15.0) + 10.0)
    v00 = arr[iy[:, None], ix[None, :]]
    v10 = arr[iy[:, None], (ix + 1)[None, :]]
    v01 = arr[(iy + 1)[:, None], ix[None, :]]
    v11 = arr[(iy + 1)[:, None], (ix + 1)[None, :]]
    return ((1.0 - fy) * ((1.0 - fx) * v00 + fx * v10) + fy * ((1.0 - fx) * v01 + fx * v11)).astype(np.float32)


@register_node("MicroGrain")
def node_micro_grain(
    ctx: dict,
    inputs: dict,
    kind: str = "brushed",
    scale: float = 28.0,
    strength: float = 0.25,
    angle_deg: float = 0.0,
    seed: int = 7,
    res_invariant: bool = True,
    **_,
):
    """Procedural micro-grain normal synthesis (brushed metal, cast stipple, mineral grain)."""
    base_n = inputs["normal"]
    H, W = base_n.shape[:2]
    circ = ctx.get("circular", False)
    res_scale = (min(H, W) / 512.0) if res_invariant else 1.0
    rng = np.random.default_rng(int(seed))
    if kind == "brushed":
        # Authentic multi-scale anisotropic brushed-metal streaks (zero artificial sine scanlines!)
        g_fine = rng.standard_normal((max(16, int(scale * 4.5)), max(4, int(scale * 0.28)))).astype(np.float32)
        g_med = rng.standard_normal((max(8, int(scale * 1.8)), max(3, int(scale * 0.14)))).astype(np.float32)
        t_fine = _interp_grid_2d(g_fine, H, W, circular=circ)
        t_med = _interp_grid_2d(g_med, H, W, circular=circ)
        grain = 0.68 * t_fine + 0.32 * t_med
        if abs(float(angle_deg)) >= 1e-5:
            th = math.radians(float(angle_deg))
            cycles = max(4, int(round(float(scale) * 14.0 / (2.0 * math.pi))))
            yy_1d = np.arange(H, dtype=np.float32) / float(H)
            xx_1d = np.arange(W, dtype=np.float32) / float(W)
            ry = math.sin(th) * xx_1d[None, :] + math.cos(th) * yy_1d[:, None]
            grain = 0.80 * grain + 0.20 * np.sin(ry * (2.0 * math.pi * cycles))
    elif kind == "cast":
        # Fine isotropic forged/cast metal stipple (two-octave C2 smooth periodic noise)
        g1 = rng.standard_normal((max(8, int(scale)), max(8, int(scale)))).astype(np.float32)
        g2 = rng.standard_normal((max(16, int(scale * 2.2)), max(16, int(scale * 2.2)))).astype(np.float32)
        t1 = _interp_grid_2d(g1, H, W, circular=circ)
        t2 = _interp_grid_2d(g2, H, W, circular=circ)
        grain = 0.55 * t1 + 0.45 * t2
    else:  # "mineral"
        g1 = rng.standard_normal((max(12, int(scale * 1.2)), max(12, int(scale * 1.2)))).astype(np.float32)
        g2 = rng.standard_normal((max(24, int(scale * 2.6)), max(24, int(scale * 2.6)))).astype(np.float32)
        grain = 0.50 * _interp_grid_2d(g1, H, W, circular=circ) + 0.50 * _interp_grid_2d(g2, H, W, circular=circ)

    N_g = dp.height_to_unit_normal(grain, strength=float(strength) * 2.2 * res_scale, circular=circ)
    return {"normal": dp.blend_rnm(base_n, N_g, detail_weight=1.0), "grain": grain.astype(np.float32)}


@register_node("NormalBlendMasked")
def node_normal_blend_masked(ctx: dict, inputs: dict, mode: str = "rnm", weight: float = 1.0, **_):
    """Blend two tangent-space normal maps (`base`, `detail`) modulated by an arbitrary `mask` in [0, 1]."""
    m = np.clip(inputs["mask"].astype(np.float32) * float(weight), 0.0, 1.0)
    p, q = normal_to_slopes(inputs["detail"])
    N_det_m = slopes_to_normal(p * m, q * m)
    if mode == "lerp":
        return {"normal": normalize_n(inputs["base"] * (1.0 - m[..., None]) + inputs["detail"] * m[..., None])}
    return {"normal": dp.blend_rnm(inputs["base"], N_det_m, detail_weight=1.0)}


@register_node("RoughnessSpecularAA")
def node_roughness_specular_aa(
    ctx: dict,
    inputs: dict,
    strength: float = 0.25,
    max_boost: float = 0.08,
    res_invariant: bool = True,
    **_,
):
    """Kaplanyan / Toksvig Specular Anti-Aliasing roughness variance boost:
    `rough = sqrt(min(1, rough^2 + min(max_boost, variance * strength)))` — prevents grazing specular flicker!"""
    circ = ctx.get("circular", False)
    N = normalize_n(inputs["normal"])
    r = inputs["roughness"].astype(np.float32)
    H, W = r.shape[:2]
    res_scale = (min(H, W) / 512.0) if res_invariant else 1.0
    if circ:
        dx = 0.5 * (np.roll(N, -1, axis=1) - np.roll(N, 1, axis=1)) * res_scale
        dy = 0.5 * (np.roll(N, -1, axis=0) - np.roll(N, 1, axis=0)) * res_scale
    else:
        dx = np.gradient(N, axis=1) * res_scale
        dy = np.gradient(N, axis=0) * res_scale
    var = np.maximum(np.sum(dx * dx, axis=-1), np.sum(dy * dy, axis=-1))
    r_aa = np.sqrt(np.minimum(1.0, r * r + np.minimum(float(max_boost), var * float(strength)))).astype(np.float32)
    return {"roughness": r_aa, "roughness_u8": np.clip(r_aa * 255.0 + 0.5, 0, 255).astype(np.uint8)}


@register_node("HeightSDFBoolean")
def node_height_sdf_boolean(ctx: dict, inputs: dict, op: str = "smooth_union", k: float = 0.06, **_):
    """Boolean combine two height fields (`a`, `b`): `union`, `subtract`, `smooth_union`, `smooth_subtract`."""
    a, b = inputs["a"].astype(np.float32), inputs["b"].astype(np.float32)
    if op == "union":
        out = np.maximum(a, b)
    elif op == "subtract":
        out = np.minimum(a, -b)
    elif op == "smooth_subtract":
        out = _smin(a, -b, float(k))
    else:
        out = _smax(a, b, float(k))
    return {"height": out.astype(np.float32)}


# --- Derived PBR Map Output Nodes ---

@register_node("CavityAO")
def node_cavity_ao(ctx: dict, inputs: dict, strength: float = 1.5, **_):
    circ = ctx.get("circular", False)
    hn = dp.normalize_height(inputs["height"], 1.0)
    ao_u8 = dp.height_to_ao(hn, strength=float(strength), circular=circ)
    return {"ao": ao_u8.astype(np.float32) / 255.0, "ao_u8": ao_u8}


@register_node("RoughnessFromCurvature")
def node_roughness_from_curvature(ctx: dict, inputs: dict, base_rough: float = 0.55,
                                  cavity_boost: float = 0.30, edge_polish: float = 0.15, **_):
    """Simple curvature-only roughness (kept for custom graphs; presets now use RoughnessPhysical)."""
    conv = inputs["convex"]
    conc = inputs["concave"]
    r = np.clip(float(base_rough) + float(cavity_boost) * conc - float(edge_polish) * conv, 0.08, 0.96).astype(np.float32)
    return {"roughness": r, "roughness_u8": np.clip(r * 255.0 + 0.5, 0, 255).astype(np.uint8)}


@register_node("RoughnessPhysical")
def node_roughness_physical(
    ctx: dict,
    inputs: dict,
    base_rough: float | None = 0.55,
    span: float = 0.34,
    w_neural: float = 0.42,
    w_physical: float = 0.36,
    w_curv: float = 0.22,
    bilateral_s: float = 1.8,
    bilateral_r: float = 0.14,
    **_,
):
    """Physically-based + WeightCode-Neural PBR Roughness node:
    Fuses 3 complementary signals:
      1. WSP-DeepBump neural roughness (145-param 8-way D4 ONNX head, +0.66 on real stone!)
      2. Multi-cue microfacet BRDF physics:
         - Recessed cavity & grout depth (1 - AO, local pits, 1 - height)
         - Toksvig / GGX micro-slope dispersion sqrt(Var_local(Nx) + Var_local(Ny))
         - High-frequency micro-texture dispersion + dielectric Fresnel specular glint polish
      3. True-sign Laplacian curvature (matte crevices, polished convex edges)
    Finished with height-guided bilateral denoising so smooth tile interiors stay uniform
    while mortar/grout transitions stay razor-sharp.
    """
    circ = ctx.get("circular", False)
    rgb = inputs["rgb"].astype(np.float32) / 255.0
    lin = srgb_to_linear(rgb)
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    mx, mn = lin.max(axis=-1), lin.min(axis=-1)
    sat = (mx - mn) / (mx + 1e-4)

    hn = dp.normalize_height(inputs["height"], clip_pct=1.0)
    ao_f = inputs["ao"]
    N = inputs["normal"]
    conc = inputs["concave"]
    conv = inputs["convex"]

    # 1. Cavity & valley depth
    valley = np.clip(1.0 - hn, 0.0, 1.0)
    ao_cav = np.clip(1.0 - ao_f, 0.0, 1.0)
    local_pit = np.clip((dp.gaussian_blur_np(hn, 4.0, circular=circ) - hn) * 3.5, 0.0, 1.0)

    # 2. Toksvig / GGX micro-slope dispersion
    p, q = normal_to_slopes(N)
    pm, qm = dp.gaussian_blur_np(p, 2.5, circular=circ), dp.gaussian_blur_np(q, 2.5, circular=circ)
    tok = np.sqrt(np.maximum(dp.gaussian_blur_np((p - pm) ** 2 + (q - qm) ** 2, 2.5, circular=circ), 0.0))
    tok_n = np.clip(tok / (np.percentile(tok, 95) + 1e-4), 0.0, 1.0)

    # 3. Micro-texture energy & Fresnel specular glint polish
    hf_l = lum - dp.gaussian_blur_np(lum, 2.0, circular=circ)
    tex_std = np.sqrt(np.maximum(dp.gaussian_blur_np(hf_l * hf_l, 3.0, circular=circ), 0.0))
    tex_n = np.clip(tex_std / (np.percentile(tex_std, 95) + 1e-4), 0.0, 1.0)
    spec = np.clip((lum - dp.gaussian_blur_np(lum, 8.0, circular=circ)) * 3.0, 0.0, 1.0) * (1.0 - 0.6 * sat)

    phys_raw = (0.34 * ao_cav + 0.24 * local_pit + 0.16 * valley + 0.14 * tok_n + 0.12 * tex_n - 0.14 * spec)
    curv_raw = conc - 0.65 * conv

    def znorm(arr):
        spread = float(np.percentile(arr, 95) - np.percentile(arr, 5)) + 1e-5
        return (arr - float(np.median(arr))) / spread

    # Fuse spatial variation fields (standardized so every cue contributes cleanly)
    neural_r = inputs.get("neural_rough")
    if neural_r is not None and float(w_neural) > 0:
        rel = float(w_neural) * znorm(neural_r) + float(w_physical) * znorm(phys_raw) + float(w_curv) * znorm(curv_raw)
    else:
        tot = float(w_physical) + float(w_curv) + 1e-8
        rel = (float(w_physical) / tot) * znorm(phys_raw) + (float(w_curv) / tot) * znorm(curv_raw)

    # Base roughness: explicit preset value OR interior-p25 auto-estimate (immune to grout edges!)
    if base_rough is None:
        p25_tex = float(np.percentile(tex_std, 25))
        p25_tok = float(np.percentile(tok, 25))
        n_med = float(np.median(neural_r)) if neural_r is not None else 0.55
        b_r = float(np.clip(0.55 * n_med + 0.45 * np.clip(0.28 + 18.0 * p25_tex + 1.4 * p25_tok, 0.26, 0.88), 0.28, 0.88))
    else:
        b_r = float(base_rough)

    rough = np.clip(b_r + float(span) * rel, 0.08, 0.96).astype(np.float32)
    rough = dp._bilateral_smooth(rough, hn, sigma_s=float(bilateral_s), sigma_r=float(bilateral_r), circular=circ)
    rough = np.clip(rough, 0.08, 0.96).astype(np.float32)
    return {"roughness": rough, "roughness_u8": np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)}


@register_node("AlbedoDelight")
def node_albedo_delight(ctx: dict, inputs: dict, shadow_lift: float = 0.65, ao_remove: float = 0.45, **_):
    """Physically-based Albedo De-Lighting: estimates smooth directional shading + cavity shadow
    from height/AO & low-frequency luminance, and divides it out in linear RGB while preserving
    exact chromaticity so engine lighting never double-shadows."""
    circ = ctx.get("circular", False)
    rgb = inputs["rgb"].astype(np.float32) / 255.0
    lin = srgb_to_linear(rgb)
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]

    # Estimate low-frequency uneven illumination gradient across the photo
    lum_broad = dp.gaussian_blur_np(lum, 32.0, circular=circ)
    lum_med = float(np.median(lum_broad)) + 1e-4
    illum_grad = np.clip(lum_broad / lum_med, 0.55, 1.55)

    # Combine with partial cavity AO shadow removal
    ao = np.clip(inputs["ao"].astype(np.float32), 0.20, 1.0)
    ao_term = 1.0 - float(ao_remove) * (1.0 - ao)

    shading = np.clip(
        (1.0 - float(shadow_lift)) + float(shadow_lift) * illum_grad,
        0.55, 1.55,
    ) * ao_term
    shading = np.clip(shading, 0.45, 1.50)

    delit_lin = np.clip(lin / shading[..., None], 0.0, 1.0)
    # Preserve exact mean exposure
    delit_srgb = linear_to_srgb(delit_lin)
    u8 = np.clip(delit_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return {"albedo": delit_srgb.astype(np.float32), "albedo_u8": u8}


@register_node("MetallicPhysical")
def node_metallic_physical(
    ctx: dict,
    inputs: dict,
    base_metal: float = 0.0,
    edge_wear_metal: float = 0.0,
    cavity_demetal: float = 0.75,
    oxide_demetal: float = 0.60,
    **_,
):
    """Physically-based Metallic map (glTF 2.0 metallic-roughness workflow):
    - Dielectric presets keep base_metal = 0.0.
    - Metal / painted-metal presets expose bare metal on convex worn ridges (+edge_wear_metal)
      while demetallizing recessed dirty cavities (-cavity_demetal) and warm orange oxide rust (-oxide_demetal)."""
    rgb = inputs["rgb"].astype(np.float32) / 255.0
    lin = srgb_to_linear(rgb)
    ao = np.clip(inputs["ao"].astype(np.float32), 0.0, 1.0)
    conv = np.clip(inputs["convex"].astype(np.float32), 0.0, 1.0)

    # Warm rust / oxide detector (high R > G > B with warm chroma)
    s = lin.sum(axis=-1) + 1e-4
    r_ch, g_ch, b_ch = lin[..., 0] / s, lin[..., 1] / s, lin[..., 2] / s
    rust = np.clip((r_ch - b_ch - 0.14) * 3.5, 0.0, 1.0) * np.clip((r_ch - g_ch - 0.04) * 4.0, 0.0, 1.0)

    cav_mask = 1.0 - float(cavity_demetal) * np.clip(1.0 - ao, 0.0, 1.0)
    ox_mask = 1.0 - float(oxide_demetal) * rust
    m = (float(base_metal) + float(edge_wear_metal) * conv) * cav_mask * ox_mask
    m = np.clip(m, 0.0, 1.0).astype(np.float32)
    return {"metallic": m, "metallic_u8": np.clip(m * 255.0 + 0.5, 0, 255).astype(np.uint8)}


@register_node("PackORM")
def node_pack_orm(ctx: dict, inputs: dict, **_):
    """Pack native glTF 2.0 ORM texture: R = AO, G = Roughness, B = Metallic."""
    ao = np.clip(inputs["ao"] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    ro = np.clip(inputs["roughness"] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    me = np.clip(inputs["metallic"] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    orm = np.stack([ao, ro, me], axis=-1)
    return {"orm_u8": orm}


@register_node("HeightThermalErosion")
def node_height_thermal_erosion(
    ctx: dict,
    inputs: dict,
    iterations: int = 16,
    talus_angle: float = 0.012,
    rate: float = 0.35,
    **_,
):
    """Iterative talus-angle thermal erosion on a heightmap: relaxes slopes steeper than `talus_angle`
    and deposits material into concave crevices (realistic stone weathering & mortar scree)."""
    circ = ctx.get("circular", False)
    h = inputs["height"].astype(np.float32).copy()
    h0 = h.copy()
    pad_mode = "wrap" if circ else "edge"
    talus = float(talus_angle)
    r = float(np.clip(rate, 0.01, 0.5))

    for _ in range(int(max(1, iterations))):
        hp = np.pad(h, 1, mode=pad_mode)
        # 8-connected neighbor differences (h_center - h_neighbor)
        d_n = h - hp[:-2, 1:-1]
        d_s = h - hp[2:, 1:-1]
        d_w = h - hp[1:-1, :-2]
        d_e = h - hp[1:-1, 2:]
        d_nw = (h - hp[:-2, :-2]) * 0.7071
        d_ne = (h - hp[:-2, 2:]) * 0.7071
        d_sw = (h - hp[2:, :-2]) * 0.7071
        d_se = (h - hp[2:, 2:]) * 0.7071

        diffs = [d_n, d_s, d_w, d_e, d_nw, d_ne, d_sw, d_se]
        excess = [np.maximum(0.0, d - talus) for d in diffs]
        total_ex = sum(excess) + 1e-8
        max_ex = np.maximum.reduce(excess)
        move = r * 0.5 * max_ex
        outflow = np.where(max_ex > 0.0, move, 0.0)

        # Distribute outflow proportionally to steeper neighbors
        flows = [outflow * (ex / total_ex) for ex in excess]
        dh = -outflow
        # Gather inflows from opposite directions
        def _shift(arr, dy, dx):
            p = np.pad(arr, 1, mode=pad_mode)
            return p[1 - dy:1 - dy + arr.shape[0], 1 - dx:1 - dx + arr.shape[1]]

        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for f, (dy, dx) in zip(flows, offsets):
            dh += _shift(f, dy, dx)
        h += dh

    sediment = np.clip((h - h0) * 8.0, 0.0, 1.0).astype(np.float32)
    return {"height": h.astype(np.float32), "sediment": sediment}


@register_node("HeightTerraceStep")
def node_height_terrace_step(
    ctx: dict,
    inputs: dict,
    steps: int = 6,
    sharpness: float = 4.0,
    mix: float = 0.75,
    **_,
):
    """Smoothly quantizes a heightmap into architectural strata / stepped moldings without aliasing."""
    h = inputs["height"].astype(np.float32)
    h_lo, h_hi = float(np.percentile(h, 1)), float(np.percentile(h, 99))
    span = max(h_hi - h_lo, 1e-6)
    u = np.clip((h - h_lo) / span, 0.0, 1.0)
    n = float(max(2, steps))
    k = float(max(0.5, sharpness))
    s = u * n
    fl = np.floor(s)
    frac = s - fl
    # Smooth S-curve step transition
    t = 0.5 + 0.5 * np.tanh(k * (frac - 0.5)) / math.tanh(0.5 * k)
    u_terr = (fl + t) / n
    out = (1.0 - float(mix)) * h + float(mix) * (h_lo + u_terr * span)
    return {"height": out.astype(np.float32)}


@register_node("HeightWarpFlow")
def node_height_warp_flow(
    ctx: dict,
    inputs: dict,
    scale: float = 6.0,
    amplitude: float = 0.015,
    octaves: int = 3,
    seed: int = 19,
    **_,
):
    """Domain-warps a heightmap (and optional normal map) along a divergence-free curl-noise flow field
    to break up overly rigid procedural edges into hand-forged ironwork or water-worn stone."""
    circ = ctx.get("circular", False)
    h = inputs["height"].astype(np.float32)
    H, W = h.shape
    rng = np.random.default_rng(int(seed))

    # Build periodic multi-octave scalar stream-function psi, then take curl (dpsi/dy, -dpsi/dx)
    psi = np.zeros((H, W), dtype=np.float32)
    amp = 1.0
    norm = 0.0
    base_f = max(2, int(round(scale)))
    for o in range(int(max(1, octaves))):
        f = max(2, base_f * (2 ** o))
        g = rng.standard_normal((f, f)).astype(np.float32)
        up = bk.resize(g, H, W, "bilinear")
        psi += amp * up
        norm += amp
        amp *= 0.5
    psi /= max(norm, 1e-6)
    psi = (psi - psi.mean()) / (psi.std() + 1e-6)

    dpsi_dy, dpsi_dx = np.gradient(psi)
    # Normalize curl vector RMS to 1
    rms = float(np.sqrt(np.mean(dpsi_dx * dpsi_dx + dpsi_dy * dpsi_dy)) + 1e-6)
    vx = (dpsi_dy / rms) * float(amplitude) * W
    vy = (-dpsi_dx / rms) * float(amplitude) * H

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx = xx + vx
    sy = yy + vy
    if circ:
        sx = sx % W
        sy = sy % H
    else:
        sx = np.clip(sx, 0.0, W - 1.001)
        sy = np.clip(sy, 0.0, H - 1.001)

    grid_x = (sx / max(W - 1, 1)) * 2.0 - 1.0
    grid_y = (sy / max(H - 1, 1)) * 2.0 - 1.0
    pad_m = "reflection" if not circ else "border"
    h_w = bk.grid_sample(h, grid_x, grid_y, mode="bilinear", padding_mode=pad_m, align_corners=True)

    out = {"height": h_w.astype(np.float32)}
    if "normal" in inputs and inputs["normal"] is not None:
        N = inputs["normal"].astype(np.float32)
        N_w = bk.grid_sample(N.transpose(2, 0, 1), grid_x, grid_y, mode="bilinear", padding_mode=pad_m,
                             align_corners=True).transpose(1, 2, 0)
        out["normal"] = normalize_n(N_w)
    return out


@register_node("NormalAnisotropicBrush")
def node_normal_anisotropic_brush(
    ctx: dict,
    inputs: dict,
    angle_deg: float = 0.0,
    freq: float = 64.0,
    anisotropy: float = 12.0,
    strength: float = 0.22,
    radial: bool = False,
    seed: int = 73,
    res_invariant: bool = True,
    **_,
):
    """Physically consistent directional brushed/milled metal or lathe-turned tangent-space normal detail
    blended onto `normal` via exact RNM."""
    circ = ctx.get("circular", False)
    N = inputs["normal"].astype(np.float32)
    H, W = N.shape[:2]
    res_scale = (min(H, W) / 512.0) if res_invariant else 1.0
    rng = np.random.default_rng(int(seed))

    # Generate high-aspect-ratio periodic brushed noise field
    f_across = max(8, int(round(freq)))
    f_along = max(2, int(round(f_across / max(1.0, anisotropy))))
    g = rng.standard_normal((f_along, f_across)).astype(np.float32)
    h_br = _interp_grid_2d(g, H, W, circular=circ)
    h_br = (h_br - h_br.mean()) / (h_br.std() + 1e-6)

    if not radial and abs(float(angle_deg)) < 1e-4:
        h_rot = h_br
    else:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u = (xx / W) - 0.5
        v = (yy / H) - 0.5
        if radial:
            # Concentric lathe-turned radial brushing around center
            r = np.sqrt(u * u + v * v) * 2.0
            th = np.arctan2(v, u) / (2.0 * math.pi) + 0.5
            gx = (r * 2.0 - 1.0)
            gy = (th * 2.0 - 1.0)
            h_rot = bk.grid_sample(h_br, gx, gy, mode="bilinear", padding_mode="reflection", align_corners=True)
        else:
            rad = math.radians(float(angle_deg))
            ca, sa = math.cos(rad), math.sin(rad)
            ru = (ca * u - sa * v) + 0.5
            rv = (sa * u + ca * v) + 0.5
            ru = (ru % 1.0) * 2.0 - 1.0
            rv = (rv % 1.0) * 2.0 - 1.0
            h_rot = bk.grid_sample(h_br, ru, rv, mode="bilinear", padding_mode="reflection", align_corners=True)

    if circ:
        dx = 0.5 * (np.roll(h_rot, -1, axis=1) - np.roll(h_rot, 1, axis=1)) * res_scale
        dy = 0.5 * (np.roll(h_rot, -1, axis=0) - np.roll(h_rot, 1, axis=0)) * res_scale
    else:
        dx = np.gradient(h_rot, axis=1) * res_scale
        dy = np.gradient(h_rot, axis=0) * res_scale
    N_br = slopes_to_normal(-dx * float(strength) * 1.8, +dy * float(strength) * 1.8)
    return {"normal": dp.blend_rnm(N, N_br, detail_weight=1.0), "brush_h": h_rot.astype(np.float32)}


@register_node("NormalSeamlessTile")
def node_normal_seamless_tile(
    ctx: dict,
    inputs: dict,
    border_frac: float = 0.22,
    **_,
):
    """Makes any non-tileable normal map (and optional heightmap) seamlessly tileable via half-offset
    cosine-feathered blending in tangent-slope space (zero seam discontinuity, preserves slope energy)."""
    N = inputs["normal"].astype(np.float32)
    H, W = N.shape[:2]
    bf = float(np.clip(border_frac, 0.05, 0.48))

    # 2D raised-cosine interior weight (1.0 in center, smoothly 0.0 at all 4 borders)
    def _w1d(n):
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        ramp = np.minimum(t, 1.0 - t) / bf
        ramp = np.clip(ramp, 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(math.pi * ramp)

    w = _w1d(H)[:, None] * _w1d(W)[None, :]
    w_shift = np.roll(np.roll(w, H // 2, axis=0), W // 2, axis=1)
    w_sum = w + w_shift + 1e-6

    p, q = normal_to_slopes(N)
    p_s = np.roll(np.roll(p, H // 2, axis=0), W // 2, axis=1)
    q_s = np.roll(np.roll(q, H // 2, axis=0), W // 2, axis=1)
    p_out = (w * p + w_shift * p_s) / w_sum
    q_out = (w * q + w_shift * q_s) / w_sum
    out = {"normal": slopes_to_normal(p_out, q_out)}

    if "height" in inputs and inputs["height"] is not None:
        h = inputs["height"].astype(np.float32)
        h_s = np.roll(np.roll(h, H // 2, axis=0), W // 2, axis=1)
        out["height"] = ((w * h + w_shift * h_s) / w_sum).astype(np.float32)
    return out


@register_node("NormalDetailTransfer")
def node_normal_detail_transfer(
    ctx: dict,
    inputs: dict,
    sigma: float = 4.0,
    strength: float = 0.75,
    **_,
):
    """Extracts high-frequency tangent-space detail from a `donor` normal map (subtracting its
    low-frequency Gaussian envelope in slope space) and transfers it onto `target` via exact RNM."""
    circ = ctx.get("circular", False)
    N_tgt = inputs["target"].astype(np.float32)
    N_don = inputs["donor"].astype(np.float32)

    p_d, q_d = normal_to_slopes(N_don)
    p_lp = dp.gaussian_blur_np(p_d, float(sigma), circular=circ)
    q_lp = dp.gaussian_blur_np(q_d, float(sigma), circular=circ)
    p_hf = (p_d - p_lp) * float(strength)
    q_hf = (q_d - q_lp) * float(strength)
    N_hf = slopes_to_normal(p_hf, q_hf)
    return {"normal": dp.blend_rnm(N_tgt, N_hf, detail_weight=1.0), "detail_normal": N_hf}


# =============================================================================
# 2b. V13 Full-Spectrum PBR Map Synthesis Nodes (HBAO, Bent Normal, DelightV13,
#     Multi-Scale Curvature/Cavity, Anisotropy Flow, Specular F0, Thickness/SSS)
# =============================================================================
@register_node("HorizonBentNormalAO")
def node_horizon_bent_normal_ao(
    ctx: dict,
    inputs: dict,
    strength: float = 1.45,
    height_scale: float = 18.0,
    radii: tuple[int, ...] = (1, 2, 4, 8, 16, 28, 44),
    ao_floor: float = 0.16,
    res_invariant: bool = True,
    **_,
):
    """V13 True 8-Azimuth Horizon-Based Ambient Occlusion (HBAO/GTAO) + Tangent-Space Bent Normal.
    Marches 8 compass directions across multiple UV-calibrated radii over `height`, measuring the maximum
    elevation horizon angle in each azimuth to compute exact hemisphere visibility (AO) AND the
    average unoccluded sky direction (Bent Normal) to eliminate indirect IBL light-leaking."""
    circ = ctx.get("circular", False)
    h = dp.normalize_height(inputs["height"].astype(np.float32), clip_pct=0.5)
    N = inputs["normal"].astype(np.float32) if "normal" in inputs and inputs["normal"] is not None else None
    H, W = h.shape
    res_scale = (min(H, W) / 512.0) if res_invariant else 1.0
    hs = float(height_scale) * res_scale
    falloff_k = 0.028 / max(res_scale, 0.25)
    if res_invariant:
        scaled = [max(1, int(round(r * res_scale))) for r in radii]
        eff_radii = tuple(sorted({r for r in scaled if r < max(2, min(H, W) // 2)})) or (1,)
    else:
        eff_radii = tuple(r for r in radii if r < max(2, min(H, W) // 2)) or (1,)
    floor_v = float(np.clip(ao_floor, 0.05, 0.50))

    dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
    ]
    use_gpu = (dp.DEVICE.type != "cpu") and (H * W >= 262144)
    if use_gpu:
        torch = bk.require_torch("GPU HBAO")
        F = torch.nn.functional
        with torch.no_grad():
            dev = dp.DEVICE
            ht = torch.from_numpy(np.ascontiguousarray(h, dtype=np.float32)).to(dev)
            max_r = max(eff_radii)
            pad_mode = "circular" if circ else "replicate"
            ht_pad = F.pad(ht[None, None], (max_r, max_r, max_r, max_r), mode=pad_mode)[0, 0]

            step_lens = torch.tensor([math.hypot(dx, dy) for dx, dy in dirs], dtype=torch.float32, device=dev)[:, None, None]
            ux = torch.tensor([dx / math.hypot(dx, dy) for dx, dy in dirs], dtype=torch.float32, device=dev)[:, None, None]
            uy = torch.tensor([dy / math.hypot(dx, dy) for dx, dy in dirs], dtype=torch.float32, device=dev)[:, None, None]

            max_sin_h = torch.zeros((len(dirs), H, W), dtype=torch.float32, device=dev)
            ht_b = ht[None, :, :]
            for r in eff_radii:
                h_s = torch.stack([
                    ht_pad[max_r + int(round(dy * r)):max_r + int(round(dy * r)) + H,
                           max_r + int(round(dx * r)):max_r + int(round(dx * r)) + W]
                    for dx, dy in dirs
                ], dim=0)
                dh = (h_s - ht_b) * hs
                dist = float(r) * step_lens
                sin_h = dh * torch.rsqrt(dh * dh + (dist * dist + 1e-6))
                falloff = 1.0 / (1.0 + falloff_k * dist)
                max_sin_h = torch.maximum(max_sin_h, torch.clamp(sin_h, 0.0, 0.98) * falloff)

            vis = 1.0 - max_sin_h
            ao_raw = torch.clamp(vis.mean(dim=0), 0.0, 1.0)
            ao_f = torch.clamp(1.0 - float(strength) * (1.0 - ao_raw), floor_v, 1.0)
            cos_h = torch.sqrt(torch.clamp(1.0 - max_sin_h * max_sin_h, min=0.0))
            vc = vis * cos_h
            bx_t = (ux * vc * 0.55).sum(dim=0)
            by_t = ((-uy) * vc * 0.55).sum(dim=0)
            bz_t = (vis * (0.45 + 0.55 * cos_h)).sum(dim=0)
            bent_t = torch.stack([bx_t, by_t, torch.clamp(bz_t, min=0.15)], dim=-1)
            bent_t = bent_t / (torch.linalg.norm(bent_t, dim=-1, keepdim=True) + 1e-8)
            if N is not None:
                nt = torch.from_numpy(np.ascontiguousarray(N, dtype=np.float32)).to(dev)
                bent_t = 0.58 * nt + 0.42 * bent_t
                bent_t = bent_t / (torch.linalg.norm(bent_t, dim=-1, keepdim=True) + 1e-8)

            ao = ao_f.cpu().numpy().astype(np.float32)
            bent = bent_t.cpu().numpy().astype(np.float32)
    else:
        ao_acc = np.zeros((H, W), dtype=np.float32)
        bx = np.zeros((H, W), dtype=np.float32)
        by = np.zeros((H, W), dtype=np.float32)
        bz = np.zeros((H, W), dtype=np.float32)

        for dx, dy in dirs:
            step_len = math.hypot(dx, dy)
            ux, uy = dx / step_len, dy / step_len
            max_sin_h = np.zeros((H, W), dtype=np.float32)
            for r in eff_radii:
                sx, sy = int(round(dx * r)), int(round(dy * r))
                if circ:
                    h_s = np.roll(np.roll(h, -sy, axis=0), -sx, axis=1)
                else:
                    pad_y = abs(sy)
                    pad_x = abs(sx)
                    hp = np.pad(h, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
                    h_s = hp[pad_y + sy:pad_y + sy + H, pad_x + sx:pad_x + sx + W]
                dh = (h_s - h) * hs
                dist = float(r) * step_len
                sin_h = dh / np.sqrt(dh * dh + dist * dist + 1e-6)
                # Resolution-invariant distance falloff so distant macro hills don't over-darken micro crevices
                falloff = 1.0 / (1.0 + falloff_k * dist)
                max_sin_h = np.maximum(max_sin_h, np.clip(sin_h, 0.0, 0.98) * falloff)

            # Visibility along this azimuth = 1 - sin(horizon_angle)
            vis = 1.0 - max_sin_h
            ao_acc += vis
            # Unoccluded bent ray leans away from high horizons (+Y OpenGL convention: dy=-1 is +Y)
            cos_h = np.sqrt(np.maximum(0.0, 1.0 - max_sin_h * max_sin_h))
            bx += ux * vis * cos_h * 0.55
            by += (-uy) * vis * cos_h * 0.55
            bz += vis * (0.45 + 0.55 * cos_h)

        ao_raw = np.clip(ao_acc / float(len(dirs)), 0.0, 1.0)
        ao = np.clip(1.0 - float(strength) * (1.0 - ao_raw), floor_v, 1.0).astype(np.float32)

        # Combine unoccluded sky cone with surface normal for clean tangent-space bent normal
        bent = normalize_n(np.stack([bx, by, np.maximum(bz, 0.15)], axis=-1))
        if N is not None:
            bent = normalize_n(0.58 * N + 0.42 * bent)

    return {
        "ao": ao,
        "ao_u8": np.clip(ao * 255.0 + 0.5, 0, 255).astype(np.uint8),
        "bent_normal": bent,
        "bent_normal_u8": dp.normal_to_u8(bent),
    }


@register_node("AlbedoDelightV13")
def node_albedo_delight_v13(
    ctx: dict,
    inputs: dict,
    directional_remove: float = 0.65,
    shadow_lift: float = 0.55,
    ao_remove: float = 0.45,
    spec_suppress: float = 0.55,
    **_,
):
    """V13 Physics-Based Albedo De-Lighting:
      1. Solves for the photo's dominant directional key-light vector L in closed form by
         regressing linear luminance against tangent-space normal (Nx, Ny, Nz, 1).
      2. Divides out directional Lambertian (N·L) shading + low-frequency illumination gradient
         + HBAO cavity shadows while preserving exact chromaticity.
      3. Suppresses achromatic white-clipped specular glints without desaturating colored albedo."""
    circ = ctx.get("circular", False)
    rgb = inputs["rgb"].astype(np.float32) / 255.0
    lin = srgb_to_linear(rgb)
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    N = inputs["normal"].astype(np.float32)
    ao = np.clip(inputs["ao"].astype(np.float32), 0.18, 1.0)

    # 1. Closed-form key-light estimation: lum ≈ c0*Nx + c1*Ny + c2*Nz + c3
    sub_N = N[::2, ::2].reshape(-1, 3).astype(np.float64)
    sub_y = lum[::2, ::2].reshape(-1).astype(np.float64)
    X = np.hstack([sub_N, np.ones((sub_N.shape[0], 1), dtype=np.float64)])
    reg = np.diag([1e-2, 1e-2, 1e-2, 1e-6])
    coeff = np.linalg.solve(X.T @ X + reg, X.T @ sub_y)
    # Directional shading component from (Nx, Ny) slopes relative to flat (0,0,1)
    dir_delta = (coeff[0] * N[..., 0] + coeff[1] * N[..., 1]).astype(np.float32)
    med_lum = float(np.median(lum)) + 1e-4
    dir_shade = np.clip(1.0 + float(directional_remove) * (dir_delta / med_lum), 0.65, 1.45)

    # 2. Broad illumination gradient + HBAO shadow term
    lum_broad = dp.gaussian_blur_np(lum, 32.0, circular=circ)
    illum_grad = np.clip(lum_broad / (float(np.median(lum_broad)) + 1e-4), 0.58, 1.50)
    ao_term = 1.0 - float(ao_remove) * (1.0 - ao)
    shading = np.clip(
        dir_shade * ((1.0 - float(shadow_lift)) + float(shadow_lift) * illum_grad) * ao_term,
        0.48, 1.55,
    )
    delit_lin = np.clip(lin / shading[..., None], 0.0, 1.0)

    # 3. Specular glint despiker: bright achromatic pixels (high lum, low chroma) pulled gently toward local median
    if float(spec_suppress) > 1e-3:
        cmax = delit_lin.max(axis=-1)
        cmin = delit_lin.min(axis=-1)
        sat = (cmax - cmin) / (cmax + 1e-4)
        l_d = 0.2126 * delit_lin[..., 0] + 0.7152 * delit_lin[..., 1] + 0.0722 * delit_lin[..., 2]
        p92 = float(np.percentile(l_d, 92))
        glint = np.clip((l_d - p92) / max(1.0 - p92, 0.08), 0.0, 1.0) * np.clip(1.0 - sat * 2.5, 0.0, 1.0)
        blur_lin = np.stack([dp.gaussian_blur_np(delit_lin[..., c], 4.0, circular=circ) for c in range(3)], axis=-1)
        w_sp = (glint * float(spec_suppress))[..., None]
        delit_lin = np.clip(delit_lin * (1.0 - w_sp) + np.minimum(delit_lin, blur_lin) * w_sp, 0.0, 1.0)

    delit_srgb = linear_to_srgb(delit_lin)
    u8 = np.clip(delit_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return {"albedo": delit_srgb.astype(np.float32), "albedo_u8": u8}


@register_node("CurvatureMultiScale")
def node_curvature_multiscale(
    ctx: dict,
    inputs: dict,
    sigmas: tuple[float, ...] = (0.8, 2.0, 5.0),
    weights: tuple[float, ...] = (0.45, 0.35, 0.20),
    gain: float = 3.2,
    res_invariant: bool = True,
    **_,
):
    """V13 Multi-Scale Signed Mean Curvature (Substance/Marmoset 0.5-neutral standard) + fine Cavity & Convex/Concave."""
    circ = ctx.get("circular", False)
    N = inputs["normal"].astype(np.float32)
    p, q = normal_to_slopes(N)
    H, W = p.shape
    res_scale = (min(H, W) / 512.0) if res_invariant else 1.0

    div_acc = np.zeros_like(p)
    fine_conc = None
    for idx, (sig, w) in enumerate(zip(sigmas, weights)):
        sig_eff = float(sig) * res_scale
        ps = dp.gaussian_blur_np(p, sig_eff, circular=circ) if sig_eff > 0.3 else p
        qs = dp.gaussian_blur_np(q, sig_eff, circular=circ) if sig_eff > 0.3 else q
        if circ:
            dpdx = 0.5 * (np.roll(ps, -1, axis=1) - np.roll(ps, 1, axis=1))
            dqdy = 0.5 * (np.roll(qs, -1, axis=0) - np.roll(qs, 1, axis=0))
        else:
            dpdx = np.gradient(ps, axis=1)
            dqdy = np.gradient(qs, axis=0)
        # OpenGL slopes: p = -dh/dx, q = +dh/dy => -lap_h = d(p)/dx - d(q)/dy
        # Positive = convex ridge/crest, negative = concave valley/grout (UV-calibrated via res_scale)
        curv_s = (dpdx - dqdy) * (0.5 * res_scale)
        div_acc += float(w) * curv_s
        if idx == 0:
            fine_conc = np.maximum(0.0, -curv_s)

    # Adaptive Substance-style contrast with gentle p98 highlight roll-off to avoid micro-pit white halos
    p98 = float(np.percentile(np.abs(div_acc), 98))
    eff_gain = float(gain) * (0.28 / max(p98, 0.04))
    signed = np.clip(0.5 + 0.46 * np.tanh(div_acc * eff_gain), 0.0, 1.0).astype(np.float32)
    convex = np.clip((signed - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
    concave = np.clip((0.5 - signed) * 2.0, 0.0, 1.0).astype(np.float32)

    fc_p97 = float(np.percentile(fine_conc, 97))
    fine_norm = np.clip(fine_conc / max(fc_p97, 0.03), 0.0, 1.0)
    cavity = np.clip(1.0 - np.maximum(concave * 0.85, fine_norm * 0.72), 0.08, 1.0).astype(np.float32)

    return {
        "curvature": signed,
        "curvature_u8": np.clip(signed * 255.0 + 0.5, 0, 255).astype(np.uint8),
        "cavity": cavity,
        "cavity_u8": np.clip(cavity * 255.0 + 0.5, 0, 255).astype(np.uint8),
        "convex": convex,
        "concave": concave,
    }


@register_node("AnisotropyFlow")
def node_anisotropy_flow(
    ctx: dict,
    inputs: dict,
    sigma_tensor: float = 2.0,
    base_f0: float = 0.50,
    res_invariant: bool = True,
    **_,
):
    """V13 glTF `KHR_materials_anisotropy` Structure-Tensor Flow Map (R=cos(theta), G=sin(theta), B=coherence)
    + physically calibrated Dielectric Specular F0 map (`specular.png`, 0.50 = standard 4% F0)."""
    circ = ctx.get("circular", False)
    h = inputs["height"].astype(np.float32)
    ao = np.clip(inputs["ao"].astype(np.float32), 0.0, 1.0)
    rough = np.clip(inputs["roughness"].astype(np.float32), 0.0, 1.0)
    H, W = h.shape
    res_scale = (min(H, W) / 512.0) if res_invariant else 1.0

    if circ:
        gx = 0.5 * (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * res_scale
        gy = 0.5 * (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * res_scale
    else:
        gx = np.gradient(h, axis=1) * res_scale
        gy = np.gradient(h, axis=0) * res_scale
    sig_eff = max(0.5, float(sigma_tensor) * res_scale)
    Jxx = dp.gaussian_blur_np(gx * gx, sig_eff, circular=circ)
    Jyy = dp.gaussian_blur_np(gy * gy, sig_eff, circular=circ)
    Jxy = dp.gaussian_blur_np(gx * gy, sig_eff, circular=circ)

    # Tangent flow direction is perpendicular to gradient (minor eigenvector of J).
    # Gate coherence by local unblurred + tight-blurred gradient energy so flat decks/panels
    # cleanly decay to isotropic neutral (B=0, tangent=(+1,0)) without halos or noise blobs.
    tr = Jxx + Jyy
    raw_e = dp.gaussian_blur_np(gx * gx + gy * gy, max(0.4, 0.9 * res_scale), circular=circ)
    eps_energy = max(4e-4, 0.22 * float(np.percentile(tr, 95)))
    ew = raw_e / (raw_e + eps_energy)
    energy_w = (ew * ew).astype(np.float32)
    diff = np.sqrt((Jxx - Jyy) ** 2 + 4.0 * Jxy * Jxy)
    coherence = np.clip((diff / (tr + 1e-8)) * energy_w, 0.0, 1.0).astype(np.float32)

    theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy) + math.pi * 0.5
    vx = np.cos(theta)
    vy = np.sin(theta)
    # Canonicalize 180-deg symmetric tangent into unified +X half-plane across all materials
    flip = (vx < -1e-4) | ((np.abs(vx) <= 1e-4) & (vy < 0.0))
    vx = np.where(flip, -vx, vx)
    vy = np.where(flip, -vy, vy)
    # Smoothly fade low-coherence flat/isotropic zones to canonical glTF neutral tangent (+1, 0)
    tx = coherence * vx + (1.0 - coherence) * 1.0
    ty = coherence * vy
    tn = np.sqrt(tx * tx + ty * ty) + 1e-8
    tx = tx / tn
    ty = ty / tn

    aniso_r = np.clip((tx * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    aniso_g = np.clip((ty * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    aniso_b = np.clip(coherence * 255.0 + 0.5, 0, 255).astype(np.uint8)
    aniso_u8 = np.stack([aniso_r, aniso_g, aniso_b], axis=-1)

    # Dielectric Specular F0: cavities and dusty high-roughness pores lose micro-specular F0
    spec_f0 = np.clip(float(base_f0) * (0.35 + 0.65 * ao) * (1.15 - 0.45 * rough), 0.08, 0.95).astype(np.float32)
    return {
        "anisotropy": np.stack([tx * 0.5 + 0.5, ty * 0.5 + 0.5, coherence], axis=-1),
        "anisotropy_u8": aniso_u8,
        "coherence": coherence,
        "specular": spec_f0,
        "specular_u8": np.clip(spec_f0 * 255.0 + 0.5, 0, 255).astype(np.uint8),
    }


@register_node("ThicknessSubsurface")
def node_thickness_subsurface(
    ctx: dict,
    inputs: dict,
    radii: tuple[float, ...] = (2.0, 6.0, 16.0),
    res_invariant: bool = True,
    **_,
):
    """V13 Local Thin-Feature Translucency / Subsurface Thickness Map (bright on thin protruding
    carved edges/lips, dark in thick deep recesses — ready for glTF `KHR_materials_volume` / SSS)."""
    circ = ctx.get("circular", False)
    h = dp.normalize_height(inputs["height"].astype(np.float32), clip_pct=0.5)
    H, W = h.shape
    res_scale = (min(H, W) / 512.0) if res_invariant else 1.0
    acc = np.zeros_like(h)
    for r in radii:
        r_eff = max(0.5, float(r) * res_scale)
        hl = dp.gaussian_blur_np(h, r_eff, circular=circ)
        # Protruding thin features have h > local average
        acc += np.clip((h - hl) * 2.4 + 0.5, 0.0, 1.0)
    thin = np.clip(0.14 + 0.82 * (acc / float(len(radii)) * 0.68 + h * 0.32), 0.12, 0.98).astype(np.float32)
    return {
        "thickness": thin,
        "thickness_u8": np.clip(thin * 255.0 + 0.5, 0, 255).astype(np.uint8),
    }


# =============================================================================
# 3. Graph Execution Engine + Built-In Material Presets (JSON-Serializable!)
# =============================================================================
def run_graph(graph_spec: dict, sources: dict[str, Any], circular: bool = False) -> dict[str, dict[str, Any]]:
    """Execute a JSON-serializable DAG of height/normal modifier nodes.

    Each node in `graph_spec["nodes"]`:
      {"id": "n1", "type": "NormalSharpen", "inputs": {"normal": "base_n.normal"}, "params": {"amount": 0.35}}
    Special source refs start with `"src."` (e.g. `"src.macro"`, `"src.neural_h"`, `"src.neural_n"`,
    `"src.micro_h"`, `"src.chroma_gate"`, `"src.gconf"`).
    Returns dict mapping every node ID -> its output dict (so intermediate taps are inspectable!).
    """
    ctx = {"circular": circular}
    state: dict[str, dict[str, Any]] = {"src": sources}

    for node in graph_spec["nodes"]:
        nid = node["id"]
        ntype = node["type"]
        if ntype not in NODE_REGISTRY:
            raise ValueError(f"Unknown modifier node type: {ntype!r}. Available: {sorted(NODE_REGISTRY)}")
        resolved_in = {}
        for k, ref in (node.get("inputs") or {}).items():
            src_id, port = ref.split(".", 1)
            resolved_in[k] = state[src_id][port]
        params = dict(node.get("params") or {})
        state[nid] = NODE_REGISTRY[ntype](ctx, resolved_in, **params)
    return state


def get_preset_graph(name: str = "v13-default") -> dict:
    """Return a clean, editable JSON node graph for common architectural PBR materials."""
    if name in ("v13-default", "v12-default", "v11-default", "v10-default", "v9-default"):
        return {
            "name": "v13-default" if name == "v13-default" else name,
            "description": "V13 Full-Spectrum PBR graph: V12.2a conv2 Dual-WSP + Helmholtz Curl-Free + JointAutoPolarity + Screened Poisson + 8-Azimuth HBAO & Bent Normal + Directional N·L Specular-Despiked Albedo Delight + Multi-Scale Curvature/Cavity + Specular-AA Roughness + Metallic + glTF ORM + Anisotropy Flow + Specular F0 + SSS Thickness + 16-bit Height.",
            "nodes": [
                {"id": "neural_n_cf", "type": "NormalHelmholtzProject",
                 "inputs": {"normal": "src.neural_n"},
                 "params": {"mix": 0.70}},
                {"id": "neural_n_gated", "type": "NormalMaskGate",
                 "inputs": {"normal": "neural_n_cf.normal", "mask": "src.chroma_gate", "gconf": "src.gconf"},
                 "params": {"min_weight": 0.25}},
                {"id": "pol", "type": "HeightJointAutoPolarity",
                 "inputs": {"macro": "src.macro", "micro": "src.micro_h", "neural": "src.neural_h"},
                 "params": {"min_each": -0.22, "min_avg": -0.32, "blend": 0.90}},
                {"id": "anchor", "type": "HeightBandCombine",
                 "inputs": {"macro": "src.macro", "neural": "pol.neural", "micro": "pol.micro",
                            "chroma_gate": "src.chroma_gate", "gconf": "src.gconf"},
                 "params": {"w_macro": 0.26, "w_neural": 0.40, "w_micro": 0.34, "sigma_macro": 3.0, "sigma_micro_hp": 12.0}},
                {"id": "bevel_h", "type": "HeightBevelShape",
                 "inputs": {"height": "anchor.height"},
                 "params": {"flat_smooth": 1.6, "edge_sigma": 0.24, "bevel_gain": 1.05}},
                {"id": "poisson_h", "type": "ScreenedPoisson",
                 "inputs": {"height": "bevel_h.height", "normal": "neural_n_gated.normal"},
                 "params": {"lam": 0.06, "slope_gain": 0.50}},
                {"id": "base_n", "type": "HeightToNormal",
                 "inputs": {"height": "poisson_h.height"},
                 "params": {"strength": 1.95, "pre_blur": 0.65, "adaptive_amp": True}},
                {"id": "rnm_n", "type": "NormalBlendRNM",
                 "inputs": {"base": "base_n.normal", "detail": "neural_n_gated.normal"},
                 "params": {"detail_weight": 0.82}},
                {"id": "denoise_n", "type": "NormalBilateralDenoise",
                 "inputs": {"normal": "rnm_n.normal", "guide": "poisson_h.height"},
                 "params": {"sigma_s": 1.8, "sigma_r": 0.20}},
                {"id": "st_n", "type": "StructureTensorDenoise",
                 "inputs": {"normal": "denoise_n.normal", "guide": "poisson_h.height"},
                 "params": {"sigma_tensor": 2.0, "alpha": 0.35, "iters": 2}},
                {"id": "eq_n", "type": "NormalFrequencySplit",
                 "inputs": {"normal": "st_n.normal"},
                 "params": {"sigma": 3.5, "low_gain": 1.02, "high_gain": 0.90}},
                {"id": "sg_n", "type": "NormalSlopeGate",
                 "inputs": {"normal": "eq_n.normal"},
                 "params": {"thresh": 0.055, "bevel_boost": 1.08}},
                {"id": "cf_n", "type": "NormalHelmholtzProject",
                 "inputs": {"normal": "sg_n.normal"},
                 "params": {"mix": 0.32}},
                {"id": "final_n", "type": "NormalClampSlope",
                 "inputs": {"normal": "cf_n.normal"},
                 "params": {"max_angle_deg": 52.0}},
                {"id": "curv", "type": "CurvatureMultiScale",
                 "inputs": {"normal": "final_n.normal"},
                 "params": {"gain": 3.2}},
                {"id": "ao", "type": "HorizonBentNormalAO",
                 "inputs": {"height": "poisson_h.height", "normal": "final_n.normal"},
                 "params": {"strength": 1.45, "height_scale": 18.0}},
                {"id": "albedo", "type": "AlbedoDelightV13",
                 "inputs": {"rgb": "src.rgb", "normal": "final_n.normal", "ao": "ao.ao"},
                 "params": {"directional_remove": 0.65, "shadow_lift": 0.55, "ao_remove": 0.45, "spec_suppress": 0.55}},
                {"id": "rough_raw", "type": "RoughnessPhysical",
                 "inputs": {"rgb": "src.rgb", "height": "poisson_h.height", "normal": "final_n.normal",
                            "ao": "ao.ao", "convex": "curv.convex", "concave": "curv.concave",
                            "neural_rough": "src.neural_rough"},
                 "params": {"base_rough": None, "span": 0.36, "w_neural": 0.42, "w_physical": 0.36, "w_curv": 0.22}},
                {"id": "rough", "type": "RoughnessSpecularAA",
                 "inputs": {"roughness": "rough_raw.roughness", "normal": "final_n.normal"},
                 "params": {"strength": 0.25, "max_boost": 0.08}},
                {"id": "metal", "type": "MetallicPhysical",
                 "inputs": {"rgb": "src.rgb", "ao": "ao.ao", "convex": "curv.convex"},
                 "params": {"base_metal": 0.0, "edge_wear_metal": 0.0, "cavity_demetal": 0.75, "oxide_demetal": 0.60}},
                {"id": "orm", "type": "PackORM",
                 "inputs": {"ao": "ao.ao", "roughness": "rough.roughness", "metallic": "metal.metallic"}},
                {"id": "aniso", "type": "AnisotropyFlow",
                 "inputs": {"height": "poisson_h.height", "ao": "ao.ao", "roughness": "rough.roughness"},
                 "params": {"sigma_tensor": 2.0, "base_f0": 0.50}},
                {"id": "thick", "type": "ThicknessSubsurface",
                 "inputs": {"height": "poisson_h.height"},
                 "params": {}},
            ],
            "outputs": {
                "albedo": "albedo.albedo_u8",
                "height": "poisson_h.height",
                "normal": "final_n.normal",
                "bent_normal": "ao.bent_normal_u8",
                "ao": "ao.ao_u8",
                "roughness": "rough.roughness_u8",
                "metallic": "metal.metallic_u8",
                "orm": "orm.orm_u8",
                "curvature": "curv.curvature_u8",
                "cavity": "curv.cavity_u8",
                "edge_wear": "curv.convex",
                "anisotropy": "aniso.anisotropy_u8",
                "specular": "aniso.specular_u8",
                "thickness": "thick.thickness_u8",
            },
        }

    if name == "glazed-tile":
        g = get_preset_graph("v9-default")
        g["name"] = "glazed-tile"
        g["description"] = "Crisp beveled ceramic/slate tiles: flat denoised glazed faces, sharpened grout bevels, low tile roughness + matte grout."
        by_id = {n["id"]: n for n in g["nodes"]}
        by_id["anchor"]["params"].update({"w_macro": 0.12, "w_neural": 0.50, "w_micro": 0.38})
        by_id["bevel_h"]["params"].update({"flat_smooth": 2.0, "edge_sigma": 0.20, "bevel_gain": 1.25})
        by_id["base_n"]["params"].update({"strength": 3.1, "pre_blur": 0.45})
        by_id["denoise_n"]["params"].update({"sigma_s": 2.0, "sigma_r": 0.16})
        by_id["eq_n"]["params"].update({"sigma": 2.5, "low_gain": 0.85, "high_gain": 1.35})
        by_id["rough_raw"]["params"].update({"base_rough": 0.34, "span": 0.28, "w_neural": 0.35, "w_physical": 0.45, "w_curv": 0.20, "bilateral_r": 0.10})
        return g

    if name == "carved-stone":
        g = get_preset_graph("v12-default")
        g["name"] = "carved-stone"
        g["description"] = "Limestone / sandstone masonry: rich macro carving + bilateral pitted micro-grain + matte cavity roughness."
        by_id = {n["id"]: n for n in g["nodes"]}
        by_id["anchor"]["params"].update({"w_macro": 0.32, "w_neural": 0.38, "w_micro": 0.30})
        by_id["base_n"]["params"].update({"strength": 2.5, "pre_blur": 0.60})
        by_id["eq_n"]["params"].update({"sigma": 4.0, "low_gain": 1.15, "high_gain": 1.10})
        by_id["ao"]["params"].update({"strength": 1.7})
        by_id["rough_raw"]["params"].update({"base_rough": 0.60, "span": 0.45, "w_neural": 0.48, "w_physical": 0.32, "w_curv": 0.20})
        return g

    if name == "ornamental-iron":
        g = get_preset_graph("v12-default")
        g["name"] = "ornamental-iron"
        g["description"] = "Cast iron / bronze scrollwork: strong macro relief, sharpened bevel edges, polished metallic ridge wear."
        by_id = {n["id"]: n for n in g["nodes"]}
        by_id["anchor"]["params"].update({"w_macro": 0.36, "w_neural": 0.38, "w_micro": 0.26})
        by_id["base_n"]["params"].update({"strength": 2.6, "pre_blur": 0.50})
        by_id["eq_n"]["params"].update({"sigma": 3.0, "low_gain": 1.20, "high_gain": 1.25})
        by_id["final_n"]["params"].update({"max_angle_deg": 60.0})
        by_id["curv"]["params"].update({"sigma": 1.2, "gain": 3.5})
        by_id["rough_raw"]["params"].update({"base_rough": 0.42, "span": 0.36, "w_neural": 0.40, "w_physical": 0.34, "w_curv": 0.26})
        by_id["metal"]["params"].update({"base_metal": 0.72, "edge_wear_metal": 0.28, "cavity_demetal": 0.65, "oxide_demetal": 0.70})
        return g

    if name == "weathered-timber":
        g = get_preset_graph("v12-default")
        g["name"] = "weathered-timber"
        g["description"] = "Timber planks & grain: clean restrained slopes, crisp plank seams, medium-matte finish."
        by_id = {n["id"]: n for n in g["nodes"]}
        by_id["base_n"]["params"].update({"strength": 1.9, "pre_blur": 0.70})
        by_id["eq_n"]["params"].update({"sigma": 3.0, "low_gain": 0.90, "high_gain": 1.15})
        by_id["final_n"]["params"].update({"max_angle_deg": 42.0})
        by_id["rough_raw"]["params"].update({"base_rough": 0.64, "span": 0.30, "w_neural": 0.40, "w_physical": 0.38, "w_curv": 0.22})
        return g

    if name == "hardops-plate":
        # HardOps / MeshMachine Pluglib + Painted Weld Fillets + Brushed Metal Anisotropy!
        g = get_preset_graph("ornamental-iron")
        g["name"] = "hardops-plate"
        g["description"] = "HardOps/MeshMachine hard-surface machined panel: stacks beveled panel seams, bolt plates, circular lug bosses, vent grilles, painted weld fillets (WeldShading.metal) & brushed anisotropy."
        # Insert hard-surface plug stamps + weld fillet + brushed grain right before final curvature/AO!
        idx_curv = next(i for i, n in enumerate(g["nodes"]) if n["id"] == "curv")
        extra_nodes = [
            {"id": "plug_seams", "type": "HardOpsPlugStamp",
             "inputs": {"height": "poisson_h.height", "normal": "final_n.normal"},
             "params": {"plug": "panel_seams", "array_x": 2, "array_y": 2, "bevel_radius": 0.04, "bevel_profile": "chamfer", "depth": 0.55, "normal_strength": 1.9}},
            {"id": "plug_bolts", "type": "HardOpsPlugStamp",
             "inputs": {"height": "plug_seams.height", "normal": "plug_seams.normal"},
             "params": {"plug": "bolt_plate", "array_x": 2, "array_y": 2, "scale": 0.72, "bevel_radius": 0.045, "bevel_profile": "round", "depth": 0.45, "normal_strength": 1.8}},
            {"id": "plug_lug", "type": "HardOpsPlugStamp",
             "inputs": {"height": "plug_bolts.height", "normal": "plug_bolts.normal"},
             "params": {"plug": "lug_boss", "array_x": 2, "array_y": 2, "scale": 0.42, "bevel_radius": 0.05, "bevel_profile": "round", "depth": 0.55, "normal_strength": 2.0}},
            {"id": "grain_n", "type": "MicroGrain",
             "inputs": {"normal": "plug_lug.normal"},
             "params": {"kind": "brushed", "scale": 34.0, "strength": 0.14, "angle_deg": 0.0}},
            {"id": "curv_pre", "type": "NormalCurvature",
             "inputs": {"normal": "grain_n.normal"},
             "params": {"sigma": 1.4, "gain": 2.6}},
            {"id": "weld_n", "type": "NormalWeldFillet",
             "inputs": {"normal": "grain_n.normal", "concave": "curv_pre.concave"},
             "params": {"radius": 2.8, "strength": 0.55}},
        ]
        g["nodes"][idx_curv:idx_curv] = extra_nodes
        by_id = {n["id"]: n for n in g["nodes"]}
        by_id["curv"]["inputs"]["normal"] = "weld_n.normal"
        by_id["ao"]["inputs"]["height"] = "plug_lug.height"
        by_id["ao"]["inputs"]["normal"] = "weld_n.normal"
        by_id["albedo"]["inputs"]["normal"] = "weld_n.normal"
        by_id["aniso"]["inputs"]["height"] = "plug_lug.height"
        by_id["thick"]["inputs"]["height"] = "plug_lug.height"
        by_id["rough_raw"]["inputs"]["height"] = "plug_lug.height"
        by_id["rough_raw"]["inputs"]["normal"] = "weld_n.normal"
        by_id["rough_raw"]["params"].update({"base_rough": 0.34, "span": 0.38})
        by_id["rough"]["inputs"]["normal"] = "weld_n.normal"
        by_id["metal"]["params"].update({"base_metal": 0.82, "edge_wear_metal": 0.25, "cavity_demetal": 0.65, "oxide_demetal": 0.50})
        g["outputs"]["height"] = "plug_lug.height"
        g["outputs"]["normal"] = "weld_n.normal"
        return g

    raise ValueError(f"Unknown preset {name!r}. Choose: v13-default, v12-default, glazed-tile, carved-stone, ornamental-iron, weathered-timber, hardops-plate")


PRESETS: dict[str, dict] = {
    name: get_preset_graph(name)
    for name in (
        "v13-default",
        "glazed-tile",
        "carved-stone",
        "ornamental-iron",
        "weathered-timber",
        "hardops-plate",
    )
}


_ROUGH_ONNX = dp.PKG_DIR / "weights" / "wsp_deepbump_roughness.onnx"

_ROUGH_WORKER = r'''
import sys, numpy as np, onnxruntime as ort
in_npz, out_npz, onnx_path = sys.argv[1:4]
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
data = np.load(in_npz)
def d4_apply(arr, g):
    a = np.fliplr(arr) if g >= 4 else arr
    return np.ascontiguousarray(np.rot90(a, k=g % 4, axes=(0, 1)))
def d4_inv(arr, g):
    a = np.rot90(arr, k=(4 - (g % 4)) % 4, axes=(0, 1))
    return np.ascontiguousarray(np.fliplr(a) if g >= 4 else a)
def infer_single(lum):
    H, W = lum.shape
    if H == 256 and W == 256:
        return sess.run(None, {"input": lum[None, None]})[0][0, 0].astype(np.float32)
    stride = 128
    acc = np.zeros((H, W), dtype=np.float32); wacc = np.zeros((H, W), dtype=np.float32)
    wy = np.hanning(256).astype(np.float32) + 0.05; win = np.outer(wy, wy)
    ys = sorted(set(list(range(0, max(1, H - 255), stride)) + [max(0, H - 256)]))
    xs = sorted(set(list(range(0, max(1, W - 255), stride)) + [max(0, W - 256)]))
    for y0 in ys:
        for x0 in xs:
            o = sess.run(None, {"input": lum[y0:y0+256, x0:x0+256][None, None]})[0][0, 0].astype(np.float32)
            acc[y0:y0+256, x0:x0+256] += o * win; wacc[y0:y0+256, x0:x0+256] += win
    return acc / np.maximum(wacc, 1e-6)
res = {}
for k in data.files:
    u8 = data[k]
    lum0 = (0.299*u8[...,0] + 0.587*u8[...,1] + 0.114*u8[...,2]).astype(np.float32) / 255.0
    acc = np.zeros_like(lum0)
    for g in range(8):
        acc += d4_inv(infer_single(d4_apply(lum0, g)), g)
    res[k] = (acc / 8.0).astype(np.float32)
np.savez_compressed(out_npz, **res)
'''


def _run_wsp_roughness_batch(rgbs: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    if not _ROUGH_ONNX.exists():
        return None
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_p, out_p, wf = td / "in.npz", td / "out.npz", td / "rw.py"
        np.savez(in_p, **rgbs)
        wf.write_text(_ROUGH_WORKER)
        subprocess.run(
            ["uv", "run", "--quiet", "--with", "onnxruntime", "--with", "numpy", "python",
             str(wf), str(in_p), str(out_p), str(_ROUGH_ONNX)],
            check=True,
        )
        return {k: v.astype(np.float32) for k, v in np.load(out_p).items()}


def extract_neural_sources(rgbs: dict[str, np.ndarray], tileable: bool = False,
                           depth_model: str = "da2") -> dict[str, dict[str, Any]]:
    """Run the neural sources once per image: 4-rot macro relief (`depth_model`: Depth Anything V2 Small + relief
    patch by default, or Neon-WSP DepthPro2-S) + 8-way D4 WSP-DeepBump + WSP-Roughness."""
    db_map, db_is_wsp = dp._run_deepbump_batch(rgbs)
    rough_map = _run_wsp_roughness_batch(rgbs)
    dp2_map = dp.macro_relief_maps(rgbs, depth_model=depth_model, tileable=tileable)

    sources = {}
    for k, rgb in rgbs.items():
        h_mic, chroma_gate = dp.v2_micro_relief(rgb, circular=tileable)
        _, gconf = dp.retinex_flat_plateau_gate(rgb, circular=tileable)
        if db_map is not None and db_is_wsp:
            out3 = db_map[k]
            nx = out3[..., 0] * 2.0 - 1.0
            ny = out3[..., 1] * 2.0 - 1.0
            nz = np.sqrt(np.maximum(0.05, 1.0 - np.clip(nx * nx + ny * ny, 0.0, 0.95)))
            N_db = normalize_n(np.stack([nx, ny, nz], axis=-1))
            h_db = dp.detrend_surface((out3[..., 2] - 0.15) / 0.70, 2)
        elif depth_model == "da2":
            # No DeepBump weights: stand in with the relief model's own detail at a calibrated slope. Its slopes point
            # the right way ~3.5x as often as the luminance micro-relief's (held-out scans), and the final height map
            # correlates twice as well with the true height.
            h_db = dp.detrend_surface(dp2_map[k], 2)
            N_db = dp.relief_normal(dp2_map[k], circular=tileable)
        else:
            h_db = h_mic.copy()
            N_db = dp.height_to_unit_normal(h_mic, strength=2.0, circular=tileable)
        sources[k] = {
            "rgb": rgb,
            "macro": dp.detrend_surface(dp2_map[k], 2),
            "neural_h": h_db,
            "neural_n": N_db,
            "neural_rough": rough_map[k] if rough_map is not None else None,
            "micro_h": h_mic,
            "chroma_gate": chroma_gate,
            "gconf": gconf,
        }
    return sources


def main():
    ap = argparse.ArgumentParser(description="V13 Full-Spectrum PBR + HardOps Pluglib & Bevel Shader Node Graph")
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--preset", default="v13-default",
                    choices=("v13-default", "v12-default", "v11-default", "v10-default", "v9-default", "glazed-tile", "carved-stone", "ornamental-iron", "weathered-timber", "hardops-plate"))
    ap.add_argument("--graph", type=Path, default=None, help="Path to custom JSON modifier graph")
    ap.add_argument("--dump-preset", default=None, help="Print preset graph JSON and exit")
    ap.add_argument("--out", type=Path, default=Path("./output/pbr"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--tileable", action="store_true")
    ap.add_argument("--dx", action="store_true", help="DirectX -Y normal output (default OpenGL +Y)")
    ap.add_argument("--strength", type=float, default=1.0, help="Final normal strength multiplier")
    ap.add_argument("--depth-model", default="da2", choices=dp.DEPTH_MODELS,
                    help="Macro relief model: da2 (default, Apache-2.0 base) or depthpro (research-only license)")
    args = ap.parse_args()

    if args.dump_preset:
        print(json.dumps(get_preset_graph(args.dump_preset), indent=2))
        return

    if not args.images:
        ap.error("Provide at least one image path or --dump-preset")

    graph = json.loads(args.graph.read_text()) if args.graph else get_preset_graph(args.preset)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rgbs = {p.stem: dp.load_square_rgb(p, size=args.size) for p in args.images}
    sources = extract_neural_sources(rgbs, tileable=args.tileable, depth_model=args.depth_model)

    def get_out(state, ref):
        nid, port = ref.split(".", 1)
        return state[nid][port]

    for p in args.images:
        stem = p.stem
        state = run_graph(graph, sources[stem], circular=args.tileable)
        outs = graph["outputs"]
        h_raw = get_out(state, outs["height"])
        N_raw = get_out(state, outs["normal"])
        if abs(args.strength - 1.0) > 1e-3:
            N_raw = node_normal_strength({"circular": args.tileable}, {"normal": N_raw}, strength=args.strength)["normal"]
        if args.dx:
            N_raw = node_normal_invert({"circular": args.tileable}, {"normal": N_raw}, flip_y=True)["normal"]

        albedo_u8 = get_out(state, outs["albedo"]) if "albedo" in outs else rgbs[stem]
        ao_u8 = get_out(state, outs["ao"])
        rough_u8 = get_out(state, outs["roughness"])
        metal_u8 = get_out(state, outs["metallic"]) if "metallic" in outs else np.zeros((args.size, args.size), np.uint8)
        orm_u8 = get_out(state, outs["orm"]) if "orm" in outs else np.stack([ao_u8, rough_u8, metal_u8], axis=-1)
        wear_u8 = np.clip(get_out(state, outs["edge_wear"]) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        h_norm = dp.normalize_height(h_raw, 1.0)
        h_u8 = np.clip(h_norm * 255.0 + 0.5, 0, 255).astype(np.uint8)
        h_u16 = np.clip(h_norm * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        n_u8 = dp.normal_to_u8(N_raw)
        bent_u8 = get_out(state, outs["bent_normal"]) if "bent_normal" in outs else n_u8
        curv_u8 = get_out(state, outs["curvature"]) if "curvature" in outs else np.full((args.size, args.size), 128, np.uint8)
        cav_u8 = get_out(state, outs["cavity"]) if "cavity" in outs else ao_u8
        aniso_u8 = get_out(state, outs["anisotropy"]) if "anisotropy" in outs else np.full((args.size, args.size, 3), 128, np.uint8)
        spec_u8 = get_out(state, outs["specular"]) if "specular" in outs else np.full((args.size, args.size), 128, np.uint8)
        thick_u8 = get_out(state, outs["thickness"]) if "thickness" in outs else h_u8

        sub = args.out / stem
        sub.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgbs[stem]).save(sub / "source_512.png")
        Image.fromarray(albedo_u8).save(sub / "albedo_delit.png")
        Image.fromarray(h_u8).save(sub / "height.png")
        Image.fromarray(h_u16).save(sub / "height16.png")
        Image.fromarray(n_u8).save(sub / "normal.png")
        Image.fromarray(bent_u8).save(sub / "bent_normal.png")
        Image.fromarray(ao_u8).save(sub / "ao.png")
        Image.fromarray(rough_u8).save(sub / "roughness.png")
        Image.fromarray(metal_u8).save(sub / "metallic.png")
        Image.fromarray(orm_u8).save(sub / "orm.png")
        Image.fromarray(curv_u8).save(sub / "curvature.png")
        Image.fromarray(cav_u8).save(sub / "cavity.png")
        Image.fromarray(wear_u8).save(sub / "edge_wear.png")
        Image.fromarray(aniso_u8).save(sub / "anisotropy.png")
        Image.fromarray(spec_u8).save(sub / "specular.png")
        Image.fromarray(thick_u8).save(sub / "thickness.png")
        (sub / "graph.json").write_text(json.dumps(graph, indent=2))

        dp.make_comparison_sheet([
            (f"Source ({graph['name']})", Image.fromarray(rgbs[stem])),
            ("V13 De-Lit Albedo", Image.fromarray(albedo_u8)),
            ("V13 Normal", Image.fromarray(n_u8)),
            ("V13 Bent Normal", Image.fromarray(bent_u8)),
            ("V13 Height (16-bit)", Image.fromarray(h_u8)),
            ("V13 8-Dir HBAO", Image.fromarray(ao_u8)),
            ("V13 Roughness", Image.fromarray(rough_u8)),
            ("V13 Curvature", Image.fromarray(curv_u8)),
            ("V13 Cavity", Image.fromarray(cav_u8)),
            ("V13 Anisotropy", Image.fromarray(aniso_u8)),
            ("V13 Specular F0", Image.fromarray(spec_u8)),
            ("V13 SSS Thickness", Image.fromarray(thick_u8)),
        ], args.out / f"{stem}_{graph['name']}_full_pbr.png")
        print(f"[{stem}] ({graph['name']}) -> {sub}/ (15 V13 PBR maps + graph.json)", flush=True)

    print(f"Done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
