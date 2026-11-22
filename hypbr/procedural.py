#!/usr/bin/env python3
"""Everyday Real-World Surface Details PBR Normal/Height/Roughness Generator.

Generates 12 seamlessly tileable everyday-life surface details (things encountered daily:
cardboard, knurled metal, denim twill, quilted leather, pegboard, sidewalk concrete,
beveled subway tile, rubber stair tread, corduroy, radiator fins, herringbone parquet,
and 2x2 carbon fiber weave) x 3 wear states (0_fresh, 1_lived_in, 2_weathered)
= 36 complete 512x512 seamless PBR texture sets!

Outputs per (detail, wear_state):
  - `normal.png`    (512x512 OpenGL +Y tangent-space normal, seamlessly tileable)
  - `height.png`    (512x512 linear POM heightmap, seamlessly tileable)
  - `ao.png`        (512x512 multi-scale cavity ambient occlusion)
  - `roughness.png` (512x512 calibrated physical roughness + Specular AA)
  - `edge_wear.png` (512x512 convex ridge wear / scuff / lint highlight mask)
  - `preview.png`   (512x512 shaded PBR clay/material lit preview so details read instantly!)

Usage:
  hypbr synth --out ./output/pbr --size 512
"""
import argparse
import base64
import functools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import _backend as bk
from . import depthpro_pbr as dp  # noqa: E402
from . import normal_graph as ng  # noqa: E402
from ._parallel import map_rows, row_slices
from .color import linear_to_srgb


# =============================================================================
# 1. Seamless Periodic Noise & Helpers
# =============================================================================
@functools.lru_cache(maxsize=96)
def _cr_periodic_basis_np(n: int, f_sz: int) -> np.ndarray:
    """Exact 1D periodic Catmull-Rom cubic interpolation matrix of shape `(n, f_sz)`."""
    pos = np.linspace(0.0, float(f_sz) * (1.0 - 1.0 / n), n, dtype=np.float32)
    i0 = np.floor(pos).astype(np.int64) % f_sz
    t = pos - np.floor(pos)
    t2 = t * t
    t3 = t2 * t
    mat = np.zeros((n, f_sz), dtype=np.float32)
    rows = np.arange(n)
    np.add.at(mat, (rows, (i0 - 1) % f_sz), -0.5 * t3 + t2 - 0.5 * t)
    np.add.at(mat, (rows, i0), 1.5 * t3 - 2.5 * t2 + 1.0)
    np.add.at(mat, (rows, (i0 + 1) % f_sz), -1.5 * t3 + 2.0 * t2 + 0.5 * t)
    np.add.at(mat, (rows, (i0 + 2) % f_sz), 0.5 * t3 - 0.5 * t2)
    return mat


@functools.lru_cache(maxsize=96)
def _cr_periodic_basis_gpu(n: int, f_sz: int, dev_str: str):
    torch = bk.require_torch("GPU periodic noise")
    return torch.from_numpy(_cr_periodic_basis_np(n, f_sz)).to(dev_str)


def _periodic_fbm(H: int, W: int, seed: int, octaves=(4, 8, 16, 32, 64)) -> np.ndarray:
    """100% mathematically seamless multi-octave periodic noise for ANY integer frequency f.

    Uses cached separable 1D periodic Catmull-Rom GEMMs `(My @ g) @ Mx.T` (14x faster on
    MPS/CUDA and 9x faster on CPU BLAS than `grid_sample`, with >0.997 correlation and
    exact circular periodicity for arbitrary integer frequencies).
    """
    rng = np.random.default_rng(seed)
    use_gpu = (dp.DEVICE.type != "cpu") and (H * W >= 262144) and all(f <= 256 for f in octaves)
    amp = 1.0
    norm = 0.0

    if use_gpu:
        torch = bk.require_torch("GPU periodic noise")
        with torch.no_grad():
            dev = dp.DEVICE
            dev_str = str(dev)
            out_t = torch.zeros((H, W), dtype=torch.float32, device=dev)
            for f in octaves:
                g = torch.from_numpy(rng.standard_normal((f, f)).astype(np.float32)).to(dev)
                my = _cr_periodic_basis_gpu(H, f, dev_str)
                mx = _cr_periodic_basis_gpu(W, f, dev_str)
                out_t.add_((my @ g) @ mx.T, alpha=amp)
                norm += amp
                amp *= 0.52
            out_t.div_(max(norm, 1e-6))
            out_t = (out_t - out_t.mean()) / (out_t.std() + 1e-6)
            return out_t.cpu().numpy().astype(np.float32)

    out = np.zeros((H, W), dtype=np.float32)
    for f in octaves:
        g = rng.standard_normal((f, f)).astype(np.float32)
        if f <= 256:
            my = _cr_periodic_basis_np(H, f)
            mx = _cr_periodic_basis_np(W, f)
            up = (my @ g) @ mx.T
        else:
            up = _fbm_octave_grid_sample(g, H, W, f)
        out += amp * up
        norm += amp
        amp *= 0.52
    out /= max(norm, 1e-6)
    return (out - out.mean()) / (out.std() + 1e-6)


def _fbm_octave_grid_sample(g: np.ndarray, H: int, W: int, f: int) -> np.ndarray:
    """Bicubic periodic upsample of an `(f, f)` lattice for octaves above the GEMM basis limit."""
    torch = bk.get_torch()
    if torch is not None:
        F = torch.nn.functional
        yy = torch.linspace(0.0, 1.0 - 1.0 / H, H, dtype=torch.float32)
        xx = torch.linspace(0.0, 1.0 - 1.0 / W, W, dtype=torch.float32)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        t_pad = F.pad(torch.from_numpy(g)[None, None], (2, 2, 2, 2), mode="circular")
        n_pad = float(f + 4)
        u_c = ((2.0 + gx * float(f) + 0.5) / n_pad) * 2.0 - 1.0
        v_c = ((2.0 + gy * float(f) + 0.5) / n_pad) * 2.0 - 1.0
        grid = torch.stack([u_c, v_c], dim=-1)[None]
        return F.grid_sample(t_pad, grid, mode="bicubic", padding_mode="border", align_corners=False)[0, 0].numpy()
    yy = np.linspace(0.0, 1.0 - 1.0 / H, H, dtype=np.float32)
    xx = np.linspace(0.0, 1.0 - 1.0 / W, W, dtype=np.float32)
    gy, gx = np.meshgrid(yy, xx, indexing="ij")
    t_pad = np.pad(g.astype(np.float32), 2, mode="wrap")
    n_pad = np.float32(f + 4)
    u_c = ((np.float32(2.0) + gx * np.float32(f) + np.float32(0.5)) / n_pad) * np.float32(2.0) - np.float32(1.0)
    v_c = ((np.float32(2.0) + gy * np.float32(f) + np.float32(0.5)) / n_pad) * np.float32(2.0) - np.float32(1.0)
    return bk.grid_sample(t_pad, u_c, v_c, mode="bicubic", padding_mode="border", align_corners=False)


def _periodic_grad(h: np.ndarray):
    """Exact periodic 3x3 isotropic Scharr gradient operator (zero boundary seam & optimal diagonal AA!)."""
    h_l = np.roll(h, 1, axis=1)
    h_r = np.roll(h, -1, axis=1)
    h_u = np.roll(h, 1, axis=0)
    h_d = np.roll(h, -1, axis=0)
    h_ul = np.roll(h_u, 1, axis=1)
    h_ur = np.roll(h_u, -1, axis=1)
    h_dl = np.roll(h_d, 1, axis=1)
    h_dr = np.roll(h_d, -1, axis=1)
    # Scharr 3x3 weights [3, 10, 3] / 32 preserve exact magnitude of 2-tap central diff with isotropic AA
    gx = (3.0 * (h_ur - h_ul) + 10.0 * (h_r - h_l) + 3.0 * (h_dr - h_dl)) * (1.0 / 32.0)
    gy = (3.0 * (h_dl - h_ul) + 10.0 * (h_d - h_u) + 3.0 * (h_dr - h_ur)) * (1.0 / 32.0)
    return gx.astype(np.float32), gy.astype(np.float32)


def _height_to_normal_periodic(h: np.ndarray, slope: float = 18.0) -> np.ndarray:
    """Convert periodic heightfield to OpenGL +Y tangent-space normal (resolution-invariant to 512px, Scharr AA)."""
    gx, gy = _periodic_grad(h)
    eff_slope = float(slope) * (min(h.shape[0], h.shape[1]) / 512.0)
    return ng.slopes_to_normal(-gx * eff_slope, +gy * eff_slope)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    if abs(e1 - e0) < 1e-6:
        return (x >= e0).astype(np.float32)
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _smin(a: np.ndarray, b: np.ndarray, k: float = 0.08) -> np.ndarray:
    """C1-smooth polynomial minimum (eliminates sharp V-creases at SDF intersections)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (b * (1.0 - h) + a * h - k * h * (1.0 - h)).astype(np.float32)


def _smax(a: np.ndarray, b: np.ndarray, k: float = 0.08) -> np.ndarray:
    """C1-smooth polynomial maximum."""
    return (-_smin(-a, -b, k)).astype(np.float32)


def _periodic_worley(u: np.ndarray, v: np.ndarray, n: int, seed: int):
    """Fast vectorized 100% periodic 2D Worley/Voronoi F1 & F2 distances on [0,1)^2."""
    rng = np.random.default_rng(seed)
    jx = rng.uniform(0.15, 0.85, (n, n)).astype(np.float32)
    jy = rng.uniform(0.15, 0.85, (n, n)).astype(np.float32)
    gx, gy = u * n, v * n
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)
    fx, fy = gx - ix, gy - iy
    f1 = np.full_like(u, 1e6, dtype=np.float32)
    f2 = np.full_like(u, 1e6, dtype=np.float32)
    for dy in (-1, 0, 1):
        ny = (iy + dy) % n
        for dx in (-1, 0, 1):
            nx = (ix + dx) % n
            ddx = dx + jx[ny, nx] - fx
            ddy = dy + jy[ny, nx] - fy
            d = np.sqrt(ddx * ddx + ddy * ddy)
            m1 = d < f1
            f2 = np.where(m1, f1, np.minimum(f2, d))
            f1 = np.where(m1, d, f1)
    return f1, f2


def _periodic_hammer_peen(u: np.ndarray, v: np.ndarray, n: int, seed: int, beta: float = 4.6):
    """C-infinity smooth, order-independent 5x5 LogSumExp lower envelope of 3D ball-peen hammer strikes (ZERO Voronoi creases!)."""
    rng = np.random.default_rng(seed)
    jx = rng.uniform(0.18, 0.82, (n, n)).astype(np.float32)
    jy = rng.uniform(0.18, 0.82, (n, n)).astype(np.float32)
    # Natural artisan strike depth & peen curvature radius variance
    z_off = rng.uniform(-0.08, 0.10, (n, n)).astype(np.float32)
    r_inv2 = rng.uniform(1.15, 1.72, (n, n)).astype(np.float32)
    gx, gy = u * n, v * n
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)
    fx, fy = gx - ix, gy - iy
    # Log-sum-exp smooth-min over full 5x5 periodic neighborhood (exact symmetry + zero 3x3 boundary steps!)
    b_min = np.full_like(u, 1e6, dtype=np.float32)
    bowls = []
    for dy in (-2, -1, 0, 1, 2):
        ny = (iy + dy) % n
        for dx in (-2, -1, 0, 1, 2):
            nx = (ix + dx) % n
            ddx = dx + jx[ny, nx] - fx
            ddy = dy + jy[ny, nx] - fy
            bowl = z_off[ny, nx] + (ddx * ddx + ddy * ddy) * r_inv2[ny, nx]
            bowls.append(bowl)
            b_min = np.minimum(b_min, bowl)
    acc = np.zeros_like(u, dtype=np.float32)
    for bowl in bowls:
        acc += np.exp(np.clip(-beta * (bowl - b_min), -40.0, 0.0))
    out = b_min - np.log(np.maximum(acc, 1e-12)) / beta
    # Exact full-range min/max normalization (ZERO percentile flat-disc clipping!)
    lo, hi = float(out.min()), float(out.max())
    return ((out - lo) / max(hi - lo, 1e-4)).astype(np.float32)


def _periodic_worley_id(u: np.ndarray, v: np.ndarray, n: int, seed: int):
    """Fast vectorized 100% periodic 2D Worley/Voronoi returning (F1, F2, cell_hash in [0,1])."""
    rng = np.random.default_rng(seed)
    jx = rng.uniform(0.14, 0.86, (n, n)).astype(np.float32).ravel()
    jy = rng.uniform(0.14, 0.86, (n, n)).astype(np.float32).ravel()
    ch = rng.uniform(0.0, 1.0, (n, n)).astype(np.float32).ravel()
    u, v = np.broadcast_arrays(np.asarray(u), np.asarray(v))
    if u.ndim == 0:
        return _worley_core(u, v, n, jx, jy, ch)
    # Purely per-pixel kernel: row bands on the shared thread pool give bit-identical output
    row_px = int(np.prod(u.shape[1:])) if u.ndim > 1 else 1
    parts = map_rows(lambda s: _worley_core(u[s], v[s], n, jx, jy, ch), row_slices(u.shape[0], row_px))
    if len(parts) == 1:
        return parts[0]
    return tuple(np.concatenate([p[i] for p in parts], axis=0) for i in range(3))


def _worley_core(u: np.ndarray, v: np.ndarray, n: int, jx: np.ndarray, jy: np.ndarray, ch: np.ndarray):
    gx, gy = u * n, v * n
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)
    fx, fy = gx - ix, gy - iy
    f1 = np.full_like(u, 1e6, dtype=np.float32)
    f2 = np.full_like(u, 1e6, dtype=np.float32)
    cid = np.zeros_like(u, dtype=np.float32)
    for dy in (-1, 0, 1):
        row = ((iy + dy) % n) * n
        for dx in (-1, 0, 1):
            flat = row + (ix + dx) % n  # flat index into the n*n jitter/hash tables (cheaper than 2-D fancy indexing)
            ddx = dx + jx.take(flat) - fx
            ddy = dy + jy.take(flat) - fy
            d = np.sqrt(ddx * ddx + ddy * ddy)
            m1 = d < f1
            f2 = np.where(m1, f1, np.minimum(f2, d))
            f1 = np.where(m1, d, f1)
            cid = np.where(m1, ch.take(flat), cid)
    return f1, f2, cid


def _lit_preview(N: np.ndarray, ao: np.ndarray, rough: np.ndarray, tint: tuple[float, float, float]) -> np.ndarray:
    """Fast physically-inspired Blinn-Phong + rim preview renderer so normal/roughness detail pops."""
    H, W = N.shape[:2]
    L1 = np.array([0.48, 0.52, 0.71], dtype=np.float32); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.60, -0.30, 0.74], dtype=np.float32); L2 /= np.linalg.norm(L2)
    V = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    H1 = L1 + V; H1 /= np.linalg.norm(H1)

    ndotl1 = np.clip(N[..., 0] * L1[0] + N[..., 1] * L1[1] + N[..., 2] * L1[2], 0.0, 1.0)
    ndotl2 = np.clip(N[..., 0] * L2[0] + N[..., 1] * L2[1] + N[..., 2] * L2[2], 0.0, 1.0)
    ndoth1 = np.clip(N[..., 0] * H1[0] + N[..., 1] * H1[1] + N[..., 2] * H1[2], 0.0, 1.0)

    shin = np.clip(2.0 / (np.maximum(rough, 0.08) ** 2.4) - 2.0, 2.0, 180.0)
    spec = (ndoth1 ** shin) * (1.0 - rough * 0.82) * 0.65

    base = np.array(tint, dtype=np.float32)[None, None, :]
    col = base * (0.20 + 0.78 * ndotl1[..., None] + 0.22 * ndotl2[..., None]) * (0.35 + 0.65 * ao[..., None])
    col += spec[..., None] * (0.4 + 0.6 * ao[..., None])
    col = linear_to_srgb(col)
    return np.clip(col * 255.0 + 0.5, 0, 255).astype(np.uint8)


# =============================================================================
# 2. 12 Everyday Real-World Seamless Surface Details
# =============================================================================
DAILY_CATALOG = [
    ("corrugated_cardboard",     "Fluted kraft shipping cardboard with sinusoidal ribs & paper fibers",     "paper",   (0.62, 0.48, 0.34)),
    ("knurled_metal_grip",       "Diamond cross-hatch machined knurling (flashlights, dials, tool grips)", "metal",   (0.58, 0.60, 0.63)),
    ("woven_denim_twill",        "Diagonal 3x1 twill denim/canvas weave with interlocking yarn twist",     "fabric",  (0.24, 0.34, 0.48)),
    ("quilted_leather_stitch",   "Padded diamond-quilted booth/bag leather with recessed stitch seams",    "leather", (0.34, 0.21, 0.16)),
    ("acoustic_pegboard",        "Workshop pegboard / acoustic panel with chamfered circular hole grid",   "wood",    (0.68, 0.59, 0.46)),
    ("sidewalk_expansion_joint", "Broom-finished concrete sidewalk slabs with chamfered control joints",   "concrete",(0.62, 0.62, 0.60)),
    ("subway_metro_tile",        "Beveled running-bond ceramic subway tiles with recessed sanded grout",   "ceramic", (0.84, 0.86, 0.85)),
    ("rubber_tread_stair",       "Anti-slip metro/staircase coin-dot rubber floor mat with bevel studs",   "rubber",  (0.22, 0.23, 0.25)),
    ("ribbed_corduroy_velvet",   "Vertical wales corduroy fabric with rounded ridges & cross-weave",       "fabric",  (0.46, 0.31, 0.22)),
    ("waffle_radiator_fins",     "Apartment cast-iron radiator / HVAC vent fins with rounded ribs",        "metal",   (0.76, 0.75, 0.72)),
    ("wood_parquet_herringbone", "Oak herringbone parquet floorboards with micro-bevel seams & wood grain","wood",    (0.56, 0.40, 0.26)),
    ("carbon_fiber_weave",       "2x2 twill carbon-fiber tech weave with crisp over-under tow relief",     "carbon",  (0.18, 0.19, 0.21)),
    ("street_storm_grate",       "Street cast-iron storm-drain trench grate with staggered beveled slots", "metal",   (0.46, 0.48, 0.51)),
    ("diamond_tread_plate",      "Industrial/subway steel diamond checkerplate with alternating raised bars","metal", (0.50, 0.53, 0.56)),
    ("hvac_slotted_vent",        "Stamped architectural HVAC return-air louver vent grille with ribs",     "metal",   (0.68, 0.69, 0.68)),
    # --- 13 New Small-Scale Vents, Grilles, Plates, Meshes & Micro-Treads ---
    ("speaker_mesh_hex",         "Fine hexagonal perforated speaker / intercom / laptop mesh grille",      "metal",   (0.50, 0.53, 0.57)),
    ("expanded_metal_mesh",      "Diamond expanded-metal security screen & industrial walkway mesh",       "metal",   (0.42, 0.44, 0.46)),
    ("stamped_louver_micro",     "Fine stamped electronics / power-box cooling louver slot array",         "metal",   (0.36, 0.38, 0.40)),
    ("chainlink_wire_mesh",      "Tight woven galvanized wire mesh screen with over-under crossings",      "metal",   (0.56, 0.58, 0.60)),
    ("circular_perf_sheet",      "Staggered micro-punched stainless steel perforated vent sheet",          "metal",   (0.64, 0.66, 0.68)),
    ("corrugated_sheet_metal",   "Fine trapezoidal corrugated steel shutter / utility siding ribs",        "metal",   (0.54, 0.56, 0.58)),
    ("ribbed_rubber_mat",        "Fine parallel V-ribbed escalator / bus-aisle anti-slip rubber mat",      "rubber",  (0.055, 0.058, 0.064)),
    ("brushed_anodized_plate",   "Fine directional brushed aluminum elevator / appliance trim seam",       "metal",   (0.70, 0.72, 0.75)),
    ("embossed_braille_strip",   "Tactile railing / elevator braille dots & directional guide ribs",       "metal",   (0.66, 0.62, 0.50)),
    ("stamped_vent_slots_stagger","Staggered rounded-pill vent slots on street transformer cabinets",      "metal",   (0.46, 0.49, 0.46)),
    ("riveted_sheet_seam",       "Overlapping bus/train/aircraft sheet-metal lap seam with flush rivets",  "metal",   (0.60, 0.62, 0.65)),
    ("micro_hex_grip",           "Molded silicone / tool handle / camera body micro-hex grip pattern",     "rubber",  (0.068, 0.074, 0.084)),
    ("woven_nylon_strap",        "Tight herringbone ribbed seatbelt / backpack nylon webbing weave",       "fabric",  (0.10, 0.14, 0.11)),
    # --- 8 More Small-Scale Seamless Everyday Micro-Surfaces (36 Total!) ---
    ("knurled_straight_flutes",  "Fine vertical straight-knurled dial / pen / camera barrel flutes",       "metal",   (0.58, 0.60, 0.64)),
    ("dimpled_golf_grip",        "Concave spherical micro-dimpled grip / acoustic / sport handle surface", "rubber",  (0.032, 0.035, 0.042)),
    ("perforated_leather_auto",  "Car-seat / headphone micro-perforated pebble leather upholstery",        "leather", (0.085, 0.065, 0.055)),
    ("stair_nosing_grooves",     "Machined aluminum anti-slip stair-nosing tread channels with grit",      "metal",   (0.62, 0.64, 0.66)),
    ("heatsink_extrusion_fins",  "Tight extruded aluminum electronics heatsink cooling fin array",         "metal",   (0.32, 0.34, 0.37)),
    ("window_blinds_slats",      "Overlapping horizontal venetian-blind slats with ladder-cord ribs",      "metal",   (0.70, 0.71, 0.70)),
    ("acoustic_slat_wood",       "Vertical micro-slatted oak acoustic wall paneling over dark felt",       "wood",    (0.58, 0.44, 0.30)),
    ("drip_tray_sieve_grate",    "Stainless-steel coffee/espresso drip-tray bridge-lanced slotted grate",  "metal",   (0.70, 0.72, 0.75)),
    # --- 8 More Small-Scale Seamless Everyday Micro-Surfaces (44 Total!) ---
    ("modular_stud_grid",        "Precision-molded interlocking circular plastic stud grid",               "ceramic", (0.76, 0.22, 0.20)),
    ("waffle_silicone_pad",      "Molded kitchen/coaster square-waffle silicone grip mat",                 "rubber",  (0.24, 0.34, 0.38)),
    ("keyboard_keycap_grid",     "Tight grid of gently dished sculpted matte-plastic keycaps",             "rubber",  (0.22, 0.23, 0.26)),
    ("solar_cell_grid",          "Monocrystalline solar panel cells with silver busbars & micro-fingers",  "ceramic", (0.12, 0.17, 0.28)),
    ("pcb_solder_mask_traces",   "Electronics PCB surface with parallel copper traces & micro-vias",       "ceramic", (0.12, 0.34, 0.22)),
    ("acoustic_pyramid_foam",    "Studio/packaging convoluted acoustic pyramid foam micro-peaks",          "rubber",  (0.18, 0.19, 0.21)),
    ("terry_velcro_loops",       "Fine micro-looped hook-and-loop / terry textile surface",                "fabric",  (0.58, 0.54, 0.48)),
    ("reeded_coin_milling",      "Ultra-fine milled coin-edge / watch-bezel parallel micro-reeding",       "metal",   (0.95, 0.68, 0.24)),
    # --- 12 New Material Categories (Glass, Terrazzo, Rattan, Cork, Hammered Copper, Knit Wool, Mosaics, Asphalt, Plaster, Felt) ---
    ("fluted_reeded_glass",      "Architectural vertical fluted/reeded privacy glass ribs",                "glass",   (0.78, 0.86, 0.85)),
    ("terrazzo_speckled_stone",  "Honed terrazzo cement with multi-colored angular marble aggregate",      "stone",   (0.76, 0.74, 0.70)),
    ("woven_rattan_cane",        "Parisian cafe Thonet woven rattan cane webbing with natural fibers",     "natural", (0.68, 0.54, 0.34)),
    ("corkboard_pin_surface",    "Natural granulated office corkboard with compressed bark cells",         "natural", (0.58, 0.42, 0.27)),
    ("hammered_copper_dimple",   "Hand-planished hammered copper with organic concave hammer facets",      "copper",  (0.80, 0.44, 0.28)),
    ("cable_knit_wool",          "Chunky V-stitch knit wool sweater textile with interlocking yarn loops", "fabric",  (0.62, 0.58, 0.52)),
    ("penny_round_mosaic",       "Glazed teal circular penny-round ceramic mosaic tiles with grout",       "ceramic", (0.16, 0.44, 0.46)),
    ("hex_mosaic_tile",          "Classic 1-inch matte hexagonal ceramic bathroom floor mosaic tiles",     "ceramic", (0.86, 0.86, 0.84)),
    ("fine_asphalt_tarmac",      "Fine-aggregate street asphalt/tarmac with sharp dark mineral gravel",    "asphalt", (0.14, 0.14, 0.15)),
    ("venetian_plaster_trowel",  "Hand-troweled architectural lime plaster with sweeping trowel ridges",   "plaster", (0.82, 0.77, 0.70)),
    ("frosted_prism_diffuser",   "Ceiling prismatic acrylic light-diffuser panel with micro-pyramids",     "glass",   (0.84, 0.88, 0.92)),
    ("acoustic_kerf_felt",       "Compressed heather-gray PET acoustic felt with fine kerf slits",         "fabric",  (0.42, 0.44, 0.47)),
    # --- 12 More Material Categories (Patina Bronze, Marble, Tweed, Waffle Linen, Damascus, Fishscale, Burlap, Cast Iron, Velvet, Bamboo, Travertine, Gilded Tin) ---
    ("patinated_verdigris_bronze","Aged architectural bronze with turquoise verdigris patina in crevices", "bronze",  (0.48, 0.36, 0.22)),
    ("carrara_veined_marble",    "Polished white Italian Carrara marble with feathery blue-gray veins",    "stone",   (0.88, 0.88, 0.87)),
    ("herringbone_tweed_wool",   "Tailored charcoal & oatmeal diagonal chevron herringbone tweed weave",   "fabric",  (0.38, 0.36, 0.33)),
    ("waffle_cotton_linen",      "Deep 3D honeycomb waffle-weave sage cotton spa textile",                 "fabric",  (0.46, 0.54, 0.48)),
    ("damascus_folded_steel",    "Acid-etched folded Damascus steel with wavy topographic grain bands",    "metal",   (0.54, 0.56, 0.59)),
    ("mermaid_fishscale_tile",   "Glazed cobalt-emerald Moroccan scalloped fish-scale ceramic tiles",      "ceramic", (0.10, 0.36, 0.44)),
    ("burlap_jute_sackcloth",    "Coarse open-weave raw jute burlap with slubby golden fibers",            "natural", (0.58, 0.46, 0.30)),
    ("cast_iron_skillet_pebble", "Seasoned black cast-iron cookware pebble grain with satin oil sheen",    "metal",   (0.14, 0.14, 0.15)),
    ("tufted_emerald_velvet",    "Plush diamond-tufted emerald velvet upholstery with recessed buttons",   "fabric",  (0.08, 0.28, 0.22)),
    ("woven_bamboo_tatami",      "Traditional Japanese woven tatami / bamboo strip mat with reed grain",   "natural", (0.64, 0.60, 0.40)),
    ("travertine_porous_stone",  "Honed Roman travertine limestone with banded horizontal erosion pores",  "stone",   (0.76, 0.69, 0.58)),
    ("gilded_tin_ceiling_tile",  "Embossed vintage pressed-tin ceiling relief in antique champagne brass", "brass",   (0.74, 0.60, 0.34)),
]

WEAR_STATES = ("0_fresh", "1_lived_in", "2_weathered")


def build_daily_detail(detail_id: str, H: int, W: int, seed: int):
    """Returns (height [0..1], slope_scale, cavity_mask [0..1], base_roughness, ridge_mask [0..1])."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / float(W)
    v = yy / float(H)

    if detail_id == "corrugated_cardboard":
        # 24 smooth vertical fluted kraft-liner ripples + subtle recycled cellulose grain (zero cross-hatch burlap weave!)
        w_wave = 0.008 * _periodic_fbm(H, W, seed + 1, (2, 4))
        flutes = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 24.0 + w_wave))
        flutes_s = np.power(np.clip(flutes, 0.0, 1.0), 0.72)
        # Subtle recycled kraft pulp grain (reduced cross-axis noise so vertical flutes read crisp & silky paperboard)
        fibers = _periodic_fbm(H, W, seed + 2, (16, 32, 64))
        crush = 0.032 * _periodic_fbm(H, W, seed + 4, (2, 4, 8))
        h = 0.22 + 0.62 * flutes_s + 0.036 * fibers + crush
        cav = np.clip((1.0 - flutes_s) * 0.80, 0.0, 1.0)
        ridge = np.clip(flutes_s ** 1.5, 0.0, 1.0)
        return h.astype(np.float32), 13.5, cav.astype(np.float32), 0.84, ridge.astype(np.float32)

    if detail_id == "knurled_metal_grip":
        # Classic 45-deg diamond machined knurling: chamfered flat-topped pyramids + smooth-filleted V-grooves
        a = (u + v) * 14.0
        b = (u - v) * 14.0
        fa = ((a % 1.0) - 0.5) * 2.0
        fb = ((b % 1.0) - 0.5) * 2.0
        # C1-smooth abs eliminates razor 1-texel V-notch flips at groove centers
        da = np.sqrt(fa * fa + 0.018) / math.sqrt(1.018)
        db = np.sqrt(fb * fb + 0.018) / math.sqrt(1.018)
        pyr = _smin(1.0 - da, 1.0 - db, 0.08)
        h = 0.82 * _smoothstep(0.02, 0.66, pyr)
        cav = _smoothstep(0.28, 0.02, h)
        ridge = _smoothstep(0.65, 0.82, h)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.28, ridge.astype(np.float32)

    if detail_id == "woven_denim_twill":
        # Authentic fine 3x1 diagonal jeans denim twill: dense 56x56 subtle indigo warp & fine diagonal twill cords (ZERO chunky tweed blocks!)
        diag_cord = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 32.0 + v * 32.0))
        warp_rib = 0.5 + 0.5 * np.cos(2.0 * math.pi * u * 64.0)
        weft_rib = 0.5 + 0.5 * np.cos(2.0 * math.pi * v * 64.0)
        # Subtle diagonal slubby cotton yarn texture (characteristic indigo jeans look!)
        slub_u = 0.06 * _periodic_fbm(H, W, seed + 3, (4, 32))
        slub_v = 0.04 * _periodic_fbm(H, W, seed + 4, (32, 4))
        fine_nap = 0.035 * _periodic_fbm(H, W, seed + 5, (32, 64))
        weave = 0.52 * diag_cord + 0.28 * warp_rib + 0.20 * weft_rib
        h = 0.28 + 0.52 * weave + slub_u + slub_v + fine_nap
        cav = _smoothstep(0.52, 0.28, h)
        ridge = _smoothstep(0.68, 0.90, h)
        return h.astype(np.float32), 6.5, cav.astype(np.float32), 0.86, ridge.astype(np.float32)

    if detail_id == "quilted_leather_stitch":
        # 4x4 diamond-quilted full-grain saddle leather: plump pillows + Voronoi pebble pores + stitch-hole puckering + twisted thread
        a = (u + v) * 4.0
        b = (u - v) * 4.0
        fa = np.abs((a % 1.0) - 0.5) * 2.0
        fb = np.abs((b % 1.0) - 0.5) * 2.0
        da = 1.0 - fa
        db = 1.0 - fb
        seam_d = np.minimum(da, db)
        # Organic subtle leather sag/wrinkle modulation so diamonds look hand-upholstered, not rigid polygons
        sag = 0.025 * _periodic_fbm(H, W, seed + 4, (4, 8, 16))
        pillow = np.sqrt(np.clip(1.0 - 0.72 * (fa * fa + fb * fb) * 0.5, 0.06, 1.0)) + sag
        seam_cut = np.exp(-((seam_d / 0.042) ** 2))
        # Stitch-hole tension puckering radiating softly away from each stitch seam
        pucker_a = np.exp(-((da / 0.14) ** 2)) * np.cos(2.0 * math.pi * b * 10.0)
        pucker_b = np.exp(-((db / 0.14) ** 2)) * np.cos(2.0 * math.pi * a * 10.0)
        pucker = 0.018 * (pucker_a + pucker_b) * (1.0 - seam_cut)
        # Crisp recessed needle holes + raised 2-ply twisted saddle-stitch dashes
        st_phase_a = np.sin(2.0 * math.pi * b * 10.0)
        st_phase_b = np.sin(2.0 * math.pi * a * 10.0)
        twist_a = 0.85 + 0.15 * np.sin(2.0 * math.pi * b * 50.0)
        twist_b = 0.85 + 0.15 * np.sin(2.0 * math.pi * a * 50.0)
        st_a = np.exp(-((da / 0.015) ** 2)) * _smoothstep(0.18, 0.52, st_phase_a) * twist_a
        st_b = np.exp(-((db / 0.015) ** 2)) * _smoothstep(0.18, 0.52, st_phase_b) * twist_b
        stitches = np.maximum(st_a, st_b)
        # True Voronoi pebble-grain leather pores (organic cellular hides!)
        f1_p, f2_p = _periodic_worley(u, v, 44, seed + 5)
        pore_crease = 1.0 - _smoothstep(0.02, 0.18, f2_p - f1_p)
        pebble = np.clip(1.0 - 0.55 * pore_crease - 0.25 * f1_p, 0.0, 1.0)
        h = pillow * (1.0 - 0.48 * seam_cut) + pucker + 0.17 * stitches + 0.036 * pebble * (1.0 - 0.75 * seam_cut)
        cav = np.clip(seam_cut * 1.05 + 0.12 * pore_crease * (1.0 - seam_cut), 0.0, 1.0)
        ridge = np.clip(stitches * 1.25 + _smoothstep(0.74, 0.96, pillow) * 0.42, 0.0, 1.0)
        return h.astype(np.float32), 15.0, cav.astype(np.float32), 0.34, ridge.astype(np.float32)

    if detail_id == "acoustic_pegboard":
        # 6x6 2-stage chamfered acoustic birch-ply/MDF pegboard: crisp milled bevel rims + fine directional birch wood fiber grain
        n = 6.0
        fu = (u * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        # Crisp deep bore (r < 0.160), clean 45-deg countersunk chamfer wall (0.160..0.240), crisp birch face (> 0.240)
        bore = _smoothstep(0.150, 0.178, r)
        chamfer = _smoothstep(0.172, 0.242, r)
        hole_prof = 0.38 * bore + 0.62 * chamfer
        w_warp = _periodic_fbm(H, W, seed + 6, (4, 8, 16))
        # Crisp directional elongated birch veneer grain & fine wood vessels (zero blurry cloud noise!)
        wood_grain = 0.020 * np.sin(2.0 * math.pi * (u * 36.0 + 0.18 * w_warp)) + 0.012 * np.sin(2.0 * math.pi * (u * 72.0 - 0.12 * w_warp))
        wood_pores = 0.022 * _periodic_fbm(H, W, seed + 7, (32, 64, 96))
        h = 0.05 + 0.85 * hole_prof + (wood_grain + wood_pores) * chamfer
        cav = np.clip(1.0 - hole_prof, 0.0, 1.0)
        ridge = _smoothstep(0.195, 0.242, r) * (1.0 - _smoothstep(0.245, 0.300, r))
        return h.astype(np.float32), 21.5, cav.astype(np.float32), 0.52, ridge.astype(np.float32)

    if detail_id == "sidewalk_expansion_joint":
        # Authentic broomed Portland-cement sidewalk slabs: packed fine aggregate + spalled control joint + broom lines + micro-pits
        # Quarter-cell phase shift places joints cleanly at 0.25/0.75 away from wrap borders -> zero normal seam ratio spike!
        w_edge = 0.0045 * _periodic_fbm(H, W, seed + 8, (8, 24, 48))
        us = (u + 0.25) % 0.5
        vs = (v + 0.25) % 0.5
        du = np.minimum(us, 0.5 - us)
        dv = np.minimum(vs, 0.5 - vs)
        d_joint = np.minimum(du, dv) + w_edge
        # Slightly rounded troweled joint lip + irregular micro-spalling along concrete joint edge
        joint = _smoothstep(0.005, 0.018, d_joint)
        trowel_lip = _smoothstep(0.016, 0.032, d_joint) * (1.0 - _smoothstep(0.036, 0.060, d_joint))
        # Fine stiff-bristle broom finish running horizontally across slab interior (not inside troweled border)
        broom_zone = _smoothstep(0.032, 0.055, d_joint)
        broom_n = _periodic_fbm(H, W, seed + 9, (4, 16))
        broom = (0.020 * np.sin(2.0 * math.pi * v * 72.0 + 0.5 * broom_n) + 0.010 * np.sin(2.0 * math.pi * v * 144.0)) * broom_zone
        # Multi-scale packed sand & fine gravel mineral aggregate (crisp Worley pebbles!)
        f1_a, f2_a, _ = _periodic_worley_id(u, v, 38, seed + 10)
        f1_b, f2_b, _ = _periodic_worley_id((u + 0.021) % 1.0, (v + 0.037) % 1.0, 68, seed + 11)
        agg = 0.030 * np.clip(1.0 - f1_a * 1.4, 0.0, 1.0) + 0.024 * np.clip(1.0 - f1_b * 1.5, 0.0, 1.0)
        grit_fine = _periodic_fbm(H, W, seed + 12, (24, 48, 96))
        # Subtle entrained air-bubble pinholes in cured concrete
        pits = _smoothstep(1.35, 2.05, -grit_fine) * 0.14
        h = 0.16 + 0.68 * joint + 0.024 * trowel_lip + (agg + broom + 0.028 * grit_fine - pits) * joint
        cav = np.clip((1.0 - joint) + 0.55 * pits + 0.15 * _smoothstep(0.95, 1.7, -grit_fine), 0.0, 1.0)
        ridge = np.clip(trowel_lip * 0.75 + _smoothstep(0.034, 0.050, agg) * 0.65 * joint, 0.0, 1.0)
        return h.astype(np.float32), 14.5, cav.astype(np.float32), 0.84, ridge.astype(np.float32)

    if detail_id == "subway_metro_tile":
        # Classic 2x4 running-bond beveled ceramic subway tiles (phase-centered off UV border)
        rows, cols = 6.0, 3.0
        vs = (v + 0.5 / rows) % 1.0
        row_idx = np.floor(vs * rows)
        # Quarter-tile U phase keeps BOTH even (fu=0.25) and odd (fu=0.75) running-bond grout lines off the UV border!
        u_shift = (u + 0.25 / cols + 0.5 * (row_idx % 2.0) / cols) % 1.0
        fu = (u_shift * cols) % 1.0
        fv = (vs * rows) % 1.0
        du = np.minimum(fu, 1.0 - fu) / cols
        dv = np.minimum(fv, 1.0 - fv) / rows
        d_edge = np.minimum(du, dv)
        grout = _smoothstep(0.004, 0.009, d_edge)
        bevel = _smoothstep(0.009, 0.032, d_edge)
        undul = 0.028 * _periodic_fbm(H, W, seed + 13, (4, 8))
        grout_grit = 0.04 * _periodic_fbm(H, W, seed + 14, (24, 48)) * (1.0 - grout)
        h = 0.14 + 0.24 * grout + 0.58 * bevel + undul * grout + grout_grit
        cav = 1.0 - grout
        ridge = _smoothstep(0.026, 0.034, d_edge) * (1.0 - _smoothstep(0.034, 0.048, d_edge))
        return h, 16.5, cav, 0.10, ridge

    if detail_id == "rubber_tread_stair":
        # Industrial anti-slip coin-dot vulcanized rubber mat: beveled flat-top studs + recessed rim step + tactile micro-pore rubber grain
        n = 8.0
        row = np.floor(v * n)
        u_s = (u + 0.5 * (row % 2.0) / n) % 1.0
        fu = (u_s * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        # 2-stage molded rubber stud: crisp beveled wall + slight rounded shoulder + subtle top-surface wear crown
        coin_base = 1.0 - _smoothstep(0.30, 0.375, r)
        coin_top = 1.0 - _smoothstep(0.25, 0.315, r)
        coin = 0.58 * coin_base + 0.42 * coin_top
        ring = 0.055 * np.exp(-(((r - 0.21) / 0.032) ** 2))
        # Tactile vulcanized rubber stipple + micro-scuffs on worn stud tops (smooth C1 micro-pebble, zero cell-edge cliffs)
        f1_p, _ = _periodic_worley(u, v, 72, seed + 15)
        grit = 0.026 * _periodic_fbm(H, W, seed + 14, (32, 64, 96)) + 0.016 * (0.5 - np.clip(f1_p, 0.0, 1.0))
        h = 0.14 + 0.72 * coin - ring + grit
        cav = np.clip(1.0 - coin_base, 0.0, 1.0)
        # Encode beveled stud rim + worn top face in ridge for semi-gloss catchlight contrast
        ridge = np.clip(0.85 * coin_top + 0.65 * (coin_base - coin_top), 0.0, 1.0)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.50, ridge.astype(np.float32)

    if detail_id == "ribbed_corduroy_velvet":
        # 20 plush rounded cotton-velvet corduroy wales + crisp micro-fiber pile normals + recessed plain-weave trough threads (zero blotchy clouds!)
        w_drift = 0.005 * _periodic_fbm(H, W, seed + 16, (4, 8))
        wales = 0.5 + 0.5 * np.cos(2.0 * math.pi * (u * 20.0 + w_drift))
        wales_r = np.power(np.clip(wales, 0.0, 1.0), 0.56)
        # Crisp fine vertical-brushed cotton-velvet micro-tufts along the plush wale crown (smooth domes, zero 1px cliff noise)
        f1_t, _ = _periodic_worley(u, v, 84, seed + 17)
        pile_fuzz = 0.036 * _periodic_fbm(H, W, seed + 18, (48, 96, 128)) + 0.020 * (0.5 - np.clip(f1_t, 0.0, 1.0))
        cross = 0.045 * np.sin(2.0 * math.pi * v * 72.0) * (0.20 + 0.80 * (1.0 - wales_r))
        h = 0.14 + 0.70 * wales_r + cross + pile_fuzz * (0.4 + 0.6 * wales_r)
        cav = np.clip(1.0 - wales_r, 0.0, 1.0)
        ridge = _smoothstep(0.58, 0.94, wales_r)
        return h.astype(np.float32), 13.0, cav.astype(np.float32), 0.86, ridge.astype(np.float32)

    if detail_id == "waffle_radiator_fins":
        # Vertical cast-iron radiator columns + horizontal tie-manifold ribs
        cols = np.power(np.clip(0.5 + 0.5 * np.cos(2.0 * math.pi * u * 10.0), 0.0, 1.0), 0.55)
        tie1 = np.exp(-(((v - 0.18) / 0.045) ** 2))
        tie2 = np.exp(-(((v - 0.82) / 0.045) ** 2))
        ties = np.maximum(tie1, tie2)
        h = np.maximum(0.12 + 0.75 * cols, 0.45 + 0.38 * ties)
        cav = _smoothstep(0.45, 0.15, h)
        ridge = _smoothstep(0.78, 0.92, h)
        return h, 20.0, cav, 0.34, ridge

    if detail_id == "wood_parquet_herringbone":
        # Authentic elongated 4:1 oak herringbone parquet in 45-deg rotated periodic lattice (N=8 -> EXACT torus periodicity!)
        N = 8.0
        rx = (u + v) * N
        ry = (u - v) * N
        ix = np.floor(rx)
        iy = np.floor(ry)
        fx = rx - ix
        fy = ry - iy
        d = (ix - iy) % 8.0
        is_h = d < 4.0
        px = np.where(is_h, d + fx, (d - 4.0) + fy)
        py = np.where(is_h, fy, fx)
        # Under u->u+1: (ix,iy)->(ix+8,iy+8); under v->v+1: (ix,iy)->(ix+8,iy-8) => mod N is 100% invariant!
        pid_x = np.where(is_h, (ix - d) % N, ix % N)
        pid_y = np.where(is_h, iy % N, (iy - (d - 4.0)) % N)
        p_hash1 = np.sin(pid_x * 13.7 + pid_y * 29.3) * 0.5 + 0.5
        p_hash2 = np.cos(pid_x * 37.1 - pid_y * 19.7) * 0.5 + 0.5
        seam_d = _smin(np.minimum(px, 4.0 - px), np.minimum(py, 1.0 - py), 0.04)
        # Anti-aliased tongue-and-groove V-bevel (wider smooth transition removes diagonal 1px staircase specks)
        bevel = _smoothstep(0.020, 0.105, seam_d)
        # Per-plank seeded flowing oak growth rings + fine elongated oak open-pore vessels
        w1 = _periodic_fbm(H, W, seed + 21, (4, 8, 16))
        w2 = _periodic_fbm(H, W, seed + 22, (16, 32, 64))
        freq_p = 3.2 + 2.4 * p_hash2
        phase_p = 6.283 * p_hash1
        g_h = np.sin(2.0 * math.pi * (ry * freq_p + 0.28 * w1) + phase_p + 0.70 * w2)
        g_v = np.sin(2.0 * math.pi * (rx * freq_p + 0.28 * w1) + phase_p + 0.70 * w2)
        grain = np.where(is_h, g_h, g_v)
        # Fine recessed dark oak wood pores aligned with plank grain
        pores = _smoothstep(0.75, 1.35, -w2) * (0.4 + 0.6 * (0.5 - 0.5 * grain))
        plank_step = 0.022 * (p_hash1 - 0.5)
        h = 0.42 + 0.44 * bevel + (0.022 * grain - 0.028 * pores + plank_step) * bevel
        cav = np.clip((1.0 - bevel) * 0.85 + 0.22 * pores * bevel, 0.0, 1.0)
        ridge = _smoothstep(0.055, 0.11, seam_d) * (1.0 - _smoothstep(0.11, 0.18, seam_d))
        return h.astype(np.float32), 9.5, cav.astype(np.float32), 0.30, ridge.astype(np.float32)

    if detail_id == "carbon_fiber_weave":
        # Tight glossy clearcoat 2x2 twill carbon-fiber epoxy weave (28x28 tows = 7 twill repeats, smooth wet clearcoat!)
        n = 28.0
        us = (u + 0.5 / n) % 1.0
        vs = (v + 0.5 / n) % 1.0
        iu, iv = np.floor(us * n), np.floor(vs * n)
        fu, fv = (us * n) % 1.0, (vs * n) % 1.0
        twill = ((iu + iv) % 4.0) < 2.0
        # Smooth resin-encapsulated flat-crowned carbon tows with soft shallow micro-valleys
        tow_u = np.sin(math.pi * fu) ** 0.24 * (0.88 + 0.12 * np.sin(math.pi * fv))
        tow_v = np.sin(math.pi * fv) ** 0.24 * (0.88 + 0.12 * np.sin(math.pi * fu))
        h = np.where(twill, 0.52 + 0.40 * tow_u, 0.48 + 0.38 * tow_v)
        cav = _smoothstep(0.62, 0.48, h) * 0.65
        ridge = _smoothstep(0.76, 0.92, h)
        return h.astype(np.float32), 5.5, cav.astype(np.float32), 0.07, ridge.astype(np.float32)

    if detail_id == "street_storm_grate":
        # Heavy cast-iron street drainage trench grate: staggered rectangular slots + beveled bars
        cols, rows = 8.0, 4.0
        r_idx = np.floor(v * rows)
        u_s = (u + 0.5 * (r_idx % 2.0) / cols) % 1.0
        fu = np.abs((u_s * cols) % 1.0 - 0.5) * 2.0  # 0 at slot center, 1 at bar center
        fv = np.abs((v * rows) % 1.0 - 0.5) * 2.0
        # Rounded-rect drainage opening inside each cell
        d_slot = np.maximum(fu - 0.56, fv - 0.72)
        bar = _smoothstep(-0.08, 0.08, d_slot)  # 0 inside open slot, 1 on cast-iron bar
        # Slight raised anti-slip crown along bar centers
        crown = 0.08 * np.clip(d_slot * 2.5, 0.0, 1.0)
        h = 0.06 + 0.80 * bar + crown * bar
        cav = 1.0 - bar
        ridge = _smoothstep(0.04, 0.14, d_slot) * (1.0 - _smoothstep(0.14, 0.26, d_slot))
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.28, ridge.astype(np.float32)

    if detail_id == "diamond_tread_plate":
        # Classic street/subway steel diamond checkerplate (alternating perpendicular raised lozenge bars)
        n = 8.0
        iu, iv = np.floor(u * n), np.floor(v * n)
        fu = (u * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        # Rotate 45 deg inside each cell, alternating +45 / -45 by checkerboard parity
        par = ((iu + iv) % 2.0) < 0.5
        a1 = (fu + fv) * 0.70710678
        b1 = (fv - fu) * 0.70710678
        lx = np.where(par, a1, b1)
        ly = np.where(par, b1, a1)
        # Tapered almond/lozenge diamond bar SDF
        d_loz = np.sqrt((lx / 0.34) ** 2 + (ly / 0.115) ** 2)
        bar = 1.0 - _smoothstep(0.72, 1.05, d_loz)
        h = 0.20 + 0.72 * bar
        cav = _smoothstep(0.35, 0.22, h)
        ridge = _smoothstep(0.65, 0.90, bar)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.32, ridge.astype(np.float32)

    if detail_id == "hvac_slotted_vent":
        # Crisp extruded/stamped architectural brushed-aluminum HVAC return-air louver grille (ZERO blotchy orange-peel!)
        cols, rows = 3.0, 10.0
        fu = np.abs(((u * cols + 0.5) % 1.0) - 0.5) * 2.0
        fv = (v * rows + 0.5) % 1.0
        in_col = 1.0 - _smoothstep(0.83, 0.90, fu)
        # Clean asymmetric angled louver blade: crisp tilted linear ramp -> sharp rolled front nose -> deep open plenum drop
        slat_ramp = np.power(np.clip(fv / 0.74, 0.0, 1.0), 0.85)
        slat_drop = 1.0 - _smoothstep(0.71, 0.80, fv)
        slat = slat_ramp * slat_drop
        rib = 1.0 - in_col
        # Clean linear horizontal brushed-aluminum micro-streaks (no blotchy Perlin wobble!)
        brush_h = 0.006 * np.sin(2.0 * math.pi * v * 160.0 + 0.3 * _periodic_fbm(H, W, seed + 29, (4, 16)))
        h_blade = 0.04 + 0.84 * slat
        h_rib = 0.82 * rib
        h = np.maximum(h_blade * in_col, h_rib) + brush_h * (1.0 - (1.0 - slat_drop) * in_col)
        cav = np.clip((1.0 - slat_drop) * in_col, 0.0, 1.0)
        # Bright metallic specular catch along rolled louver front nose + beveled vertical stile shoulders
        stile_edge = _smoothstep(0.78, 0.86, fu) * (1.0 - _smoothstep(0.88, 0.95, fu))
        ridge = np.clip(_smoothstep(0.58, 0.72, fv) * slat_drop * in_col + 0.75 * stile_edge, 0.0, 1.0)
        return h.astype(np.float32), 19.0, cav.astype(np.float32), 0.24, ridge.astype(np.float32)

    if detail_id == "speaker_mesh_hex":
        # Die-stamped hexagonal honeycomb perforated speaker/intercom mesh: deep dark hex holes + rolled bevel rims + brushed anodized web
        nx, ny = 14.0, 12.0
        row = np.floor(v * ny)
        u_s = (u + 0.5 * (row % 2.0) / nx) % 1.0
        fu = (u_s * nx) % 1.0 - 0.5
        fv = (v * ny) % 1.0 - 0.5
        ax, ay = np.abs(fu), np.abs(fv)
        d_hex = np.maximum(ax * 0.866025 + ay * 0.5, ay)
        # 2-stage stamped hex web: crisp void drop + curved die-punched bevel shoulder
        hole_void = 1.0 - _smoothstep(0.28, 0.35, d_hex)
        roll_bevel = _smoothstep(0.30, 0.44, d_hex)
        brush = 0.010 * np.sin(2.0 * math.pi * v * 128.0 + 0.4 * _periodic_fbm(H, W, seed + 30, (8, 24)))
        h = 0.04 + 0.58 * (1.0 - hole_void) + 0.26 * roll_bevel + brush * (1.0 - hole_void)
        cav = hole_void
        ridge = _smoothstep(0.33, 0.40, d_hex) * (1.0 - _smoothstep(0.41, 0.48, d_hex))
        return h.astype(np.float32), 21.0, cav.astype(np.float32), 0.20, ridge.astype(np.float32)

    if detail_id == "expanded_metal_mesh":
        # Stretched diamond expanded-metal mesh with C1 beveled strand relief (zero diagonal staircase aliasing)
        a = (u * 12.0 + v * 6.0)
        b = (u * 12.0 - v * 6.0)
        fa = ((a % 1.0) - 0.5) * 2.0
        fb = ((b % 1.0) - 0.5) * 2.0
        da = np.sqrt(fa * fa + 0.015) / math.sqrt(1.015)
        db = np.sqrt(fb * fb + 0.015) / math.sqrt(1.015)
        d_m = _smax(da, db, 0.06)
        strand = _smoothstep(0.58, 0.84, d_m)
        twist = 0.12 * np.sin(2.0 * math.pi * a) * strand
        h = 0.08 + 0.76 * strand + twist
        cav = 1.0 - strand
        ridge = _smoothstep(0.74, 0.92, d_m)
        return h.astype(np.float32), 17.5, cav.astype(np.float32), 0.38, ridge.astype(np.float32)

    if detail_id == "stamped_louver_micro":
        # Fine electronics / street cabinet stamped louver rows (6 cols x 18 rows, C1 raised-cosine hoods)
        cols, rows = 6.0, 18.0
        fu = np.abs((u * cols) % 1.0 - 0.5) * 2.0
        fv = (v * rows + 0.36) % 1.0
        in_slot = 1.0 - _smoothstep(0.72, 0.88, fu)
        hood = 0.5 * (1.0 - np.cos(2.0 * math.pi * np.clip(fv / 0.72, 0.0, 1.0))) * (fv < 0.72)
        open_gap = _smoothstep(0.66, 0.78, fv) * (1.0 - _smoothstep(0.86, 0.97, fv)) * in_slot
        h = 0.45 + 0.40 * hood * in_slot - 0.34 * open_gap
        cav = open_gap
        ridge = _smoothstep(0.60, 0.95, hood) * in_slot
        return h.astype(np.float32), 16.5, cav.astype(np.float32), 0.34, ridge.astype(np.float32)

    if detail_id == "chainlink_wire_mesh":
        # Authentic interlocking helical twisted galvanized steel chain-link fence with zinc spangle over deep open void!
        nx, ny = 6.0, 6.0
        zv_a = 0.265 * (2.0 / math.pi) * np.arcsin(0.94 * np.sin(2.0 * math.pi * v * ny))
        zv_b = -zv_a
        da = np.abs((((u * nx - 0.25 - zv_a) + 0.5) % 1.0) - 0.5)
        db = np.abs((((u * nx - 0.75 - zv_b) + 0.5) % 1.0) - 0.5)
        r_wire = 0.105
        wa = np.sqrt(np.clip(1.0 - (da / r_wire) ** 2, 0.0, 1.0))
        wb = np.sqrt(np.clip(1.0 - (db / r_wire) ** 2, 0.0, 1.0))
        # Pronounced helical over-under hook depth + tight twisted collar at every diamond knuckle!
        phase_k = np.cos(2.0 * math.pi * v * ny)
        za = 0.72 + 0.27 * phase_k
        zb = 0.72 - 0.27 * phase_k
        ha = wa * za
        hb = wb * zb
        knuckle = np.minimum(wa, wb)
        twist_collar = 0.18 * knuckle * (0.7 + 0.3 * np.sin(2.0 * math.pi * (u * nx * 3.0 + v * ny * 3.0)))
        # Galvanized zinc crystal micro-spangle (tapered at crystal boundaries so zero 1px normal cliffs)
        f1_z, f2_z, zinc_sp = _periodic_worley_id(u, v, 42, seed + 39)
        zinc_taper = _smoothstep(0.0, 0.14, f2_z - f1_z)
        wire_h = np.clip(np.maximum(ha, hb) + twist_collar, 0.0, 1.0)
        wire_mask = _smoothstep(0.04, 0.20, np.maximum(wa, wb))
        h = 0.02 + 0.88 * wire_h + 0.015 * (zinc_sp - 0.5) * zinc_taper * wire_mask
        cav = 1.0 - wire_mask
        ridge = np.clip((_smoothstep(0.60, 0.90, wire_h) + 0.35 * knuckle) * wire_mask, 0.0, 1.0)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.24, ridge.astype(np.float32)

    if detail_id == "circular_perf_sheet":
        # Die-punched staggered stainless-steel perforated sheet (14x14): rolled inward punch bevel rims + directional brushed metal!
        n = 14.0
        row = np.floor(v * n)
        u_s = (u + 0.5 * (row % 2.0) / n) % 1.0
        fu = (u_s * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        # 2-stage die-punched profile: crisp punched hole + rolled specular collar bevel around each hole!
        hole_void = 1.0 - _smoothstep(0.24, 0.30, r)
        roll_bevel = _smoothstep(0.25, 0.39, r)
        brush = 0.012 * np.sin(2.0 * math.pi * v * 144.0 + 0.4 * _periodic_fbm(H, W, seed + 36, (8, 24)))
        h = 0.04 + 0.58 * (1.0 - hole_void) + 0.26 * roll_bevel + brush * (1.0 - hole_void)
        cav = hole_void
        ridge = _smoothstep(0.28, 0.35, r) * (1.0 - _smoothstep(0.36, 0.44, r))
        return h.astype(np.float32), 21.0, cav.astype(np.float32), 0.20, ridge.astype(np.float32)

    if detail_id == "corrugated_sheet_metal":
        # Smooth sinusoidal-trapezoidal galvanized corrugated steel panel (10 ribs, flat-plateau border-aligned)
        t = (u * 10.0 + 0.43) % 1.0
        trap = np.minimum(_smoothstep(0.06, 0.30, t), 1.0 - _smoothstep(0.56, 0.80, t))
        galv = 0.015 * _periodic_fbm(H, W, seed + 37, (8, 16, 32))
        h = 0.16 + 0.72 * trap + galv
        cav = 1.0 - trap
        ridge = _smoothstep(0.82, 0.99, trap)
        return h.astype(np.float32), 12.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)

    if detail_id == "ribbed_rubber_mat":
        # Crisp extruded parallel trapezoidal-rib anti-slip industrial rubber mat (14 bold ribs + rounded crests + vulcanized micro-grain)
        t_raw = np.abs(((v * 14.0) % 1.0) - 0.5) * 2.0
        # Flat deep floor + steep beveled side walls + gently rounded rubber crest catches crisp studio highlights!
        rib_wall = 1.0 - _smoothstep(0.32, 0.68, t_raw)
        rib_crown = np.sqrt(np.clip(1.0 - (t_raw / 0.48) ** 2, 0.0, 1.0))
        rib = 0.62 * rib_wall + 0.38 * rib_crown
        f1_p, _ = _periodic_worley(u, v, 72, seed + 38)
        grain = 0.025 * _periodic_fbm(H, W, seed + 39, (32, 64, 96)) + 0.015 * (0.5 - np.clip(f1_p, 0.0, 1.0))
        h = 0.10 + 0.78 * rib + grain
        cav = np.clip(1.0 - rib_wall, 0.0, 1.0)
        ridge = _smoothstep(0.55, 0.92, rib)
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.52, ridge.astype(np.float32)

    if detail_id == "brushed_anodized_plate":
        # Crisp strictly linear directional brushed aluminum elevator panel (ZERO wavy noise!) + expansion seam & screws
        # 1D periodic high-freq streak table along v, gently broken up along u
        rng_b = np.random.default_rng(seed + 41)
        b_row = rng_b.standard_normal(H).astype(np.float32)
        # Mild 3-tap periodic smoothing so streaks are anti-aliased yet crisp
        b_row = 0.25 * np.roll(b_row, -1) + 0.50 * b_row + 0.25 * np.roll(b_row, 1)
        b_mod = 0.65 + 0.35 * _periodic_fbm(H, W, seed + 42, (2, 4))
        brush = 0.011 * b_row[:, None] * b_mod
        seam_v = np.exp(-(((v - 0.5) / 0.008) ** 2))
        seam_u = np.exp(-(((u - 0.5) / 0.008) ** 2))
        seams = np.maximum(seam_v, seam_u)
        screws = np.zeros_like(u)
        for sx in (0.44, 0.56):
            for sy in (0.44, 0.56):
                r = np.sqrt((u - sx) ** 2 + (v - sy) ** 2)
                screws = np.maximum(screws, 1.0 - _smoothstep(0.012, 0.018, r))
        h = 0.62 + brush - 0.42 * seams - 0.18 * screws
        cav = np.clip(seams + 0.6 * screws, 0.0, 1.0)
        ridge = _smoothstep(0.012, 0.020, np.minimum(np.abs(u - 0.5), np.abs(v - 0.5))) * (1.0 - _smoothstep(0.020, 0.032, np.minimum(np.abs(u - 0.5), np.abs(v - 0.5))))
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)

    if detail_id == "embossed_braille_strip":
        # Tactile brushed stainless-steel transit braille railing strip: clean convex guide ribs + crisp hemispherical braille domes sitting flush on brushed plate (NO recessed crater moats!)
        rib_pos = (0.14, 0.22, 0.78, 0.86)
        ribs = np.zeros_like(u)
        fillet = np.zeros_like(u)
        for rp in rib_pos:
            dr = np.abs(v - rp)
            ribs = np.maximum(ribs, np.sqrt(np.clip(1.0 - (dr / 0.020) ** 2, 0.0, 1.0)))
            fillet = np.maximum(fillet, np.exp(-(((dr - 0.021) / 0.0045) ** 2)))
        dots = np.zeros_like(u)
        rng = np.random.default_rng(seed + 45)
        for ci in range(8):
            cx0 = (ci + 0.5) / 8.0
            for dx in (-0.022, 0.022):
                for dy in (-0.075, 0.0, 0.075):
                    if rng.random() > 0.35:
                        r = np.sqrt((u - (cx0 + dx)) ** 2 + (v - (0.5 + dy)) ** 2)
                        dots = np.maximum(dots, np.sqrt(np.clip(1.0 - (r / 0.0185) ** 2, 0.0, 1.0)))
                        fillet = np.maximum(fillet, np.exp(-(((r - 0.0195) / 0.0040) ** 2)))
        # Fine crisp linear brushed stainless-steel micro-grooves running horizontally along railing
        brush = 0.014 * np.sin(2.0 * math.pi * v * 160.0 + 0.35 * _periodic_fbm(H, W, seed + 46, (4, 16)))
        emb = np.maximum(ribs, dots)
        h = 0.22 + 0.68 * emb + brush * (1.0 - 0.7 * emb)
        cav = np.clip(fillet * (1.0 - emb) * 0.55 + 0.10 * (1.0 - emb), 0.0, 1.0)
        ridge = np.clip(_smoothstep(0.35, 0.92, emb), 0.0, 1.0)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.20, ridge.astype(np.float32)

    if detail_id == "stamped_vent_slots_stagger":
        # Die-punched staggered pill ventilation slots on sheet-metal cabinets: rolled inward punch bevel + brushed sheet grain
        cols, rows = 4.0, 12.0
        r_idx = np.floor(v * rows)
        u_s = (u + 0.5 * (r_idx % 2.0) / cols) % 1.0
        fu = (u_s * cols) % 1.0 - 0.5
        fv = (v * rows) % 1.0 - 0.5
        px_c = np.maximum(np.abs(fu) - 0.25, 0.0)
        d_pill = np.sqrt(px_c * px_c + (fv * 0.36) ** 2)
        # 2-stage die-punched profile: crisp dark hole + rolled stamped depression collar around each pill!
        hole_void = 1.0 - _smoothstep(0.062, 0.088, d_pill)
        roll_bevel = _smoothstep(0.068, 0.145, d_pill)
        # Subtle directional rolled-steel sheet micro-texture & slight stamping pillow between slots
        brush = 0.012 * _periodic_fbm(H, W, seed + 43, (8, 32, 64))
        h = 0.04 + 0.62 * (1.0 - hole_void) + 0.22 * roll_bevel + brush * (1.0 - hole_void)
        cav = np.clip(hole_void, 0.0, 1.0)
        ridge = _smoothstep(0.082, 0.118, d_pill) * (1.0 - _smoothstep(0.125, 0.175, d_pill))
        return h.astype(np.float32), 21.0, cav.astype(np.float32), 0.30, ridge.astype(np.float32)

    if detail_id == "riveted_sheet_seam":
        # Seamless aerospace/train riveted aluminum lap-strap butt joint with double flush rivet rows
        strap = _smoothstep(0.32, 0.35, v) * (1.0 - _smoothstep(0.65, 0.68, v))
        seam_groove = np.exp(-(((v - 0.50) / 0.008) ** 2))
        strap_edges = np.exp(-(((v - 0.335) / 0.007) ** 2)) + np.exp(-(((v - 0.665) / 0.007) ** 2))
        rivets = np.zeros_like(u)
        for ry in (0.415, 0.585):
            fu = (u * 10.0) % 1.0 - 0.5
            r = np.sqrt((fu / 10.0) ** 2 + (v - ry) ** 2)
            dome = np.sqrt(np.clip(1.0 - (r / 0.020) ** 2, 0.0, 1.0))
            ring = np.exp(-(((r - 0.021) / 0.004) ** 2)) * 0.25
            rivets = np.maximum(rivets, dome - ring)
        roll_grain = 0.014 * _periodic_fbm(H, W, seed + 47, (8, 32))
        h = 0.42 + 0.16 * strap - 0.20 * seam_groove - 0.06 * strap_edges + 0.26 * rivets + roll_grain
        cav = np.clip(seam_groove + 0.6 * strap_edges, 0.0, 1.0)
        ridge = np.clip(rivets + 0.4 * strap * (1.0 - seam_groove), 0.0, 1.0)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.24, ridge.astype(np.float32)

    if detail_id == "micro_hex_grip":
        # Modern molded silicone / tool / camera body micro-hex grip (16x14 raised chamfered hex islands, Nyquist-clean!)
        nx, ny = 16.0, 14.0
        row = np.floor(v * ny)
        u_s = (u + 0.5 * (row % 2.0) / nx) % 1.0
        fu = (u_s * nx) % 1.0 - 0.5
        fv = (v * ny) % 1.0 - 0.5
        ax, ay = np.abs(fu), np.abs(fv)
        d_hex = _smax(ax * 0.866025 + ay * 0.5, ay, 0.03)
        pad_h = 1.0 - _smoothstep(0.29, 0.44, d_hex)
        h = 0.16 + 0.72 * pad_h
        cav = 1.0 - pad_h
        ridge = _smoothstep(0.75, 0.98, pad_h)
        return h.astype(np.float32), 16.0, cav.astype(np.float32), 0.66, ridge.astype(np.float32)

    if detail_id == "woven_nylon_strap":
        # Clean band-limited herringbone woven seatbelt nylon webbing (zero moire, tactile matte fabric)
        cols = 4.0
        us = (u + 0.5 / cols) % 1.0
        col = np.floor(us * cols)
        sgn = np.where((col % 2.0) < 0.5, 1.0, -1.0)
        rib = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 20.0 + sgn * us * 8.0))
        cross = 0.12 * (0.5 + 0.5 * np.sin(2.0 * math.pi * us * 32.0))
        h = 0.22 + 0.60 * (rib ** 0.78) + cross
        cav = 1.0 - rib
        ridge = _smoothstep(0.72, 0.98, rib)
        return h.astype(np.float32), 13.0, cav.astype(np.float32), 0.76, ridge.astype(np.float32)

    if detail_id == "knurled_straight_flutes":
        # Crisp machined straight-knurled dial / camera barrel flutes (24 C1-continuous metallic ridges)
        t_raw = np.abs(((u * 24.0 + 0.25) % 1.0) - 0.5) * 2.0
        t = np.sqrt(t_raw * t_raw + 0.022) / math.sqrt(1.022)
        flute = np.clip(1.0 - t, 0.0, 1.0) ** 0.78
        h = 0.18 + 0.70 * flute
        cav = 1.0 - flute
        ridge = _smoothstep(0.74, 0.96, flute)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.20, ridge.astype(np.float32)

    if detail_id == "dimpled_golf_grip":
        # Tactile vulcanized rubber concave dimpled grip (12x12 deep spherical dimples + crisp chamfer rims + elastomer micro-stipple!)
        n = 12.0
        row = np.floor(v * n)
        u_s = (u + 0.5 * (row % 2.0) / n) % 1.0
        fu = (u_s * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        # Deep spherical concavity + crisp subtly raised chamfer rim lip catching studio key light
        dimple = np.sqrt(np.clip(1.0 - (r / 0.35) ** 2, 0.0, 1.0))
        rim_lip = np.exp(-(((r - 0.365) / 0.042) ** 2))
        f1_p, _ = _periodic_worley(u, v, 68, seed + 49)
        grain = 0.026 * _periodic_fbm(H, W, seed + 50, (32, 64, 96)) + 0.015 * (0.5 - np.clip(f1_p, 0.0, 1.0))
        h = 0.78 - 0.64 * dimple + 0.065 * rim_lip + grain
        cav = np.clip(dimple * 0.92, 0.0, 1.0)
        ridge = np.clip(rim_lip * 0.95 + 0.25 * (1.0 - _smoothstep(0.32, 0.42, r) * 0.0) * (1.0 - dimple), 0.0, 1.0)
        return h.astype(np.float32), 18.5, cav.astype(np.float32), 0.56, ridge.astype(np.float32)

    if detail_id == "perforated_leather_auto":
        # Luxury dark charcoal automotive micro-perforated Nappa leather: plump rounded pebble domes + indented punch bevels + deep dark bores (ZERO mud-crack Voronoi!)
        n = 12.0
        row = np.floor(v * n)
        u_s = (u + 0.5 * (row % 2.0) / n) % 1.0
        fu = (u_s * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        hole = 1.0 - _smoothstep(0.130, 0.195, r)
        # Subtle indented punch dimple collar around each perforated hole catches crisp leather specular rims
        punch_bevel = _smoothstep(0.165, 0.295, r)
        # Plump softly rounded natural Nappa pebble domes (soft curved profiles, NO flat polygonal mud-cracks!)
        wu = 0.009 * _periodic_fbm(H, W, seed + 52, (6, 12))
        wv = 0.009 * _periodic_fbm(H, W, seed + 53, (6, 12))
        f1, f2 = _periodic_worley((u + wu) % 1.0, (v + wv) % 1.0, 46, seed + 51)
        f1b, f2b = _periodic_worley((u - wu) % 1.0, (v + wv) % 1.0, 74, seed + 54)
        dome1 = np.sqrt(np.clip(1.0 - (f1 * 1.18) ** 2, 0.0, 1.0))
        dome2 = np.sqrt(np.clip(1.0 - (f1b * 1.18) ** 2, 0.0, 1.0))
        crease = 1.0 - _smoothstep(0.0, 0.28, f2 - f1)
        pebble = np.clip(0.68 * dome1 + 0.32 * dome2 - 0.12 * crease, 0.0, 1.0)
        h = 0.05 + (0.60 + 0.16 * punch_bevel + 0.12 * pebble) * (1.0 - hole)
        cav = np.clip(hole + 0.14 * crease * (1.0 - hole), 0.0, 1.0)
        ridge = np.clip((_smoothstep(0.45, 0.88, pebble) * 0.72 + _smoothstep(0.195, 0.265, r) * (1.0 - _smoothstep(0.27, 0.36, r)) * 0.75) * (1.0 - hole), 0.0, 1.0)
        return h.astype(np.float32), 16.0, cav.astype(np.float32), 0.30, ridge.astype(np.float32)

    if detail_id == "stair_nosing_grooves":
        # Machined aluminum anti-slip stair-nosing: clean extruded ribs & fine parallel micro-serrated channels
        t = (v * 10.0) % 1.0
        rib = np.minimum(_smoothstep(0.06, 0.14, t), 1.0 - _smoothstep(0.42, 0.50, t))
        # Fine 1D extruded micro-serrations inside each anti-slip channel (zero crumpled FBM distortion)
        serr = (0.5 + 0.5 * np.sin(2.0 * math.pi * v * 120.0)) * (1.0 - rib)
        h = 0.22 + 0.66 * rib + 0.045 * serr
        cav = 1.0 - rib
        ridge = _smoothstep(0.85, 0.99, rib)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)

    if detail_id == "heatsink_extrusion_fins":
        # Deep high-aspect-ratio extruded aluminum heatsink cooling fins: 8 crisp flat-topped chamfered blades + deep dark shadowed cooling slots (flat trough centered on u=0/1 for exact wrap seamlessness!)
        t = (u * 8.0) % 1.0
        dt = np.abs(t - 0.5)
        # Wide crisp 45° extruded flanks (0.12..0.24) let directional light naturally illuminate left flank bright and right flank dark (ZERO symmetric fake borders!)
        fin_wall = 1.0 - _smoothstep(0.12, 0.24, dt)
        fin_top = 1.0 - _smoothstep(0.10, 0.14, dt)
        trough_floor = _smoothstep(0.22, 0.28, dt)
        fin = 0.72 * fin_wall + 0.28 * fin_top
        # Clean purely longitudinal vertical extrusion streaks (strictly low-Nyquist u-only harmonics, ZERO cross-hatching!)
        ext_lines = 0.0022 * (np.sin(2.0 * math.pi * u * 56.0) + 0.6 * np.sin(2.0 * math.pi * u * 96.0 + 1.1))
        h = 0.02 + 0.92 * fin + ext_lines * fin_top
        cav = np.clip(0.32 * (1.0 - fin_wall) + 0.68 * trough_floor, 0.0, 1.0)
        # Keep ridge strictly on flat fin top so left/right angled flanks get pure asymmetric 3D normal shading!
        ridge = np.clip(fin_top, 0.0, 1.0)
        return h.astype(np.float32), 32.0, cav.astype(np.float32), 0.16, ridge.astype(np.float32)

    if detail_id == "window_blinds_slats":
        # Overlapping horizontal venetian-blind slats (12 phase-centered slats) + twin vertical ladder-cord ribs
        fv = (v * 12.0 + 0.5) % 1.0
        slat = np.clip(fv / 0.88, 0.0, 1.0) * (1.0 - _smoothstep(0.86, 0.96, fv))
        cord = np.maximum(
            1.0 - _smoothstep(0.008, 0.016, np.abs(u - 0.25)),
            1.0 - _smoothstep(0.008, 0.016, np.abs(u - 0.75)),
        )
        h = np.maximum(0.16 + 0.70 * slat, 0.58 * cord)
        cav = _smoothstep(0.84, 0.94, fv)
        ridge = _smoothstep(0.75, 0.86, fv) * (1.0 - cav)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.28, ridge.astype(np.float32)

    if detail_id == "acoustic_slat_wood":
        # Architectural vertical micro-slatted oak acoustic wall paneling: per-slat wood grain/vessels + chamfered edges + fuzzy felt
        n_slats = 12.0
        sid = np.floor(u * n_slats) % n_slats
        t = (u * n_slats) % 1.0
        # 2-stage chamfered oak slat profile: steep felt side walls + crisp micro-chamfered top corners
        slat_core = np.minimum(_smoothstep(0.13, 0.19, t), 1.0 - _smoothstep(0.77, 0.83, t))
        slat_top = np.minimum(_smoothstep(0.18, 0.24, t), 1.0 - _smoothstep(0.72, 0.78, t))
        slat_prof = 0.78 * slat_core + 0.22 * slat_top
        # Per-slat unique wood-grain phase & cathedral warp + fine vertical oak vessels
        s_hash = np.sin(sid * 19.3 + 4.1) * 0.5 + 0.5
        s_hash2 = np.cos(sid * 31.7 + 1.9) * 0.5 + 0.5
        w_flow = _periodic_fbm(H, W, seed + 61, (4, 8, 16))
        w_fine = _periodic_fbm(H, W, seed + 62, (16, 48, 96))
        grain_wave = np.sin(2.0 * math.pi * (u * (48.0 + 24.0 * s_hash2) + 0.16 * w_flow) + 6.283 * s_hash)
        oak_pores = _smoothstep(0.80, 1.45, -w_fine) * (0.45 + 0.55 * (0.5 - 0.5 * grain_wave))
        felt_fib = 0.055 * _periodic_fbm(H, W, seed + 63, (32, 64, 96)) * (1.0 - slat_core)
        slat_step = 0.022 * (s_hash - 0.5)
        h = 0.08 + 0.76 * slat_prof + (0.028 * grain_wave - 0.030 * oak_pores + slat_step) * slat_top + felt_fib
        cav = np.clip((1.0 - slat_core) + 0.22 * oak_pores * slat_top, 0.0, 1.0)
        # Highlight along chamfered oak slat corners
        chamfer_edge = np.clip(slat_core - slat_top, 0.0, 1.0)
        ridge = np.clip(chamfer_edge * 1.8 + 0.30 * slat_top * (0.5 + 0.5 * grain_wave), 0.0, 1.0)
        return h.astype(np.float32), 19.0, cav.astype(np.float32), 0.44, ridge.astype(np.float32)

    if detail_id == "drip_tray_sieve_grate":
        # Brushed stainless-steel espresso drip-tray bridge-lanced grate: rolled punched slot bevels + directional brush grain
        cols, rows = 10.0, 6.0
        r_idx = np.floor(v * rows)
        u_s = (u + 0.5 * (r_idx % 2.0) / cols) % 1.0
        fu = (u_s * cols) % 1.0 - 0.5
        fv = (v * rows) % 1.0 - 0.5
        py_c = np.maximum(np.abs(fv) - 0.25, 0.0)
        d_pill = np.sqrt((fu * 0.62) ** 2 + py_c * py_c)
        # 2-stage punched stainless profile: deep dark slot + curved stamped bevel lip
        slot_void = 1.0 - _smoothstep(0.082, 0.118, d_pill)
        roll_lip = _smoothstep(0.090, 0.175, d_pill)
        # Fine brushed stainless-steel micro-lines along horizontal axis (u)
        brush = 0.016 * np.sin(2.0 * math.pi * v * 128.0 + 0.6 * _periodic_fbm(H, W, seed + 64, (8, 24)))
        h = 0.04 + 0.60 * (1.0 - slot_void) + 0.24 * roll_lip + brush * (1.0 - slot_void)
        cav = np.clip(slot_void, 0.0, 1.0)
        ridge = _smoothstep(0.105, 0.145, d_pill) * (1.0 - _smoothstep(0.150, 0.210, d_pill))
        return h.astype(np.float32), 21.0, cav.astype(np.float32), 0.20, ridge.astype(np.float32)

    if detail_id == "modular_stud_grid":
        # High-gloss injection-molded 8x8 interlocking ABS plastic studs: crisp chamfered crown + recessed injection-gate ring + smooth curved fillets
        n = 8.0
        fu = (u * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        # 2-stage molded ABS cylinder: steep wall + rounded glossy top shoulder + faint recessed top mold detail
        stud_base = 1.0 - _smoothstep(0.28, 0.345, r)
        stud_top = 1.0 - _smoothstep(0.22, 0.285, r)
        stud = 0.72 * stud_base + 0.28 * stud_top
        # Subtle injection-molded circular top bevel lip & tiny center sprue dimple
        gate_ring = 0.024 * np.exp(-(((r - 0.16) / 0.026) ** 2))
        sprue = 0.014 * np.exp(-((r / 0.045) ** 2))
        mold_pol = 0.004 * _periodic_fbm(H, W, seed + 66, (6, 12, 24))
        h = 0.14 + 0.74 * stud - gate_ring - sprue + mold_pol
        cav = np.clip(1.0 - stud_base, 0.0, 1.0)
        # Crisp shoulder catchlight ring around top of each glossy ABS stud
        ridge = np.clip((stud_base - stud_top) * 1.9 + 0.35 * stud_top, 0.0, 1.0)
        return h.astype(np.float32), 21.0, cav.astype(np.float32), 0.13, ridge.astype(np.float32)

    if detail_id == "waffle_silicone_pad":
        # Everyday kitchen/desk square-waffle molded silicone mat (10x10 phase-centered crisp pockets)
        n = 10.0
        fu = np.abs(((u * n + 0.5) % 1.0) - 0.5) * 2.0
        fv = np.abs(((v * n + 0.5) % 1.0) - 0.5) * 2.0
        wall = _smoothstep(0.66, 0.84, np.maximum(fu, fv))
        h = 0.16 + 0.72 * wall
        cav = 1.0 - wall
        ridge = _smoothstep(0.82, 0.98, np.maximum(fu, fv))
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.48, ridge.astype(np.float32)

    if detail_id == "keyboard_keycap_grid":
        # Tight 6x6 grid of sculpted PBT keycaps: gently spherical finger dish + micro-stippled PBT texture + crisp chamfer rim
        n = 6.0
        fu = (u * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        # Soft rounded-square superellipse (p=4) keycap top silhouette (not harsh sharp rectangle!)
        d_box = ((np.abs(fu) ** 4.0) + (np.abs(fv) ** 4.0)) ** 0.25
        cap = 1.0 - _smoothstep(0.35, 0.46, d_box)
        top_lip = 1.0 - _smoothstep(0.29, 0.37, d_box)
        dish = 0.15 * np.exp(-((fu * fu + fv * fv) / 0.075)) * top_lip
        pbt_stipple = 0.016 * _periodic_fbm(H, W, seed + 68, (32, 64, 96))
        h = 0.08 + (0.76 * cap + 0.06 * top_lip - dish) + pbt_stipple * cap
        cav = 1.0 - cap
        ridge = _smoothstep(0.27, 0.35, d_box) * cap
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.38, ridge.astype(np.float32)

    if detail_id == "solar_cell_grid":
        # Monocrystalline/polycrystalline solar panel cells: silicon crystal facets + metallic busbars + crisp micro-fingers
        n = 2.0
        fu = np.abs((u * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((v * n) % 1.0 - 0.5) * 2.0
        cell_gap = np.maximum(np.maximum(fu, fv) - 0.94, (fu + fv) - 1.72)
        in_cell = 1.0 - _smoothstep(-0.018, 0.018, cell_gap)
        busbars = np.maximum(
            1.0 - _smoothstep(0.012, 0.022, np.abs(fu - 0.38)),
            1.0 - _smoothstep(0.012, 0.022, np.abs(fu - 0.78)),
        ) * in_cell
        # Crisp metallic collector micro-fingers running perpendicular to busbars
        f_wave = 0.5 + 0.5 * np.cos(2.0 * math.pi * v * 72.0)
        fingers = _smoothstep(0.80, 0.95, f_wave) * in_cell
        # Subtle polycrystalline silicon wafer crystal facets (smoothly tapered at grain edges -> zero glass-laminate cliff!)
        f1_s, f2_s, si_facet = _periodic_worley_id(u, v, 10, seed + 69)
        si_taper = _smoothstep(0.0, 0.10, f2_s - f1_s)
        h = 0.20 + 0.56 * in_cell + 0.14 * busbars + 0.045 * fingers + 0.010 * (si_facet - 0.5) * si_taper * in_cell
        cav = 1.0 - in_cell
        ridge = np.clip(busbars + 0.55 * fingers, 0.0, 1.0)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.10, ridge.astype(np.float32)

    if detail_id == "pcb_solder_mask_traces":
        # Authentic asymmetric electronics PCB: non-symmetrical routed copper differential buses + off-center QFP IC pads + smooth ENIG gold pads + drilled vias!
        # Keep v=0 and v=1 strictly flat vertical bus crossings so wrap seam is EXACTLY zero at any resolution (256..2048)!
        # Asymmetric upper (+2 lanes) vs lower (-1 lane) 45-deg bus jogs: strictly periodic in BOTH u and v (zero wrap seam!)
        jog_upper = np.clip((v - 0.14) / 0.14, 0.0, 1.0) * (1.0 - np.clip((v - 0.40) / 0.10, 0.0, 1.0)) * 0.125
        jog_lower = -np.clip((v - 0.56) / 0.14, 0.0, 1.0) * (1.0 - np.clip((v - 0.80) / 0.10, 0.0, 1.0)) * 0.0625
        u_routed = (u + jog_upper + jog_lower + 0.03125) % 1.0
        # 16 copper lanes with irregular realistic bus grouping (power rail + differential pairs + clock gaps)
        t_cell = (u_routed * 16.0) % 1.0 - 0.5
        t_idx = np.floor(u_routed * 16.0).astype(np.int32) % 16
        bus_keep = np.isin(t_idx, [0, 1, 2, 4, 5, 7, 9, 10, 11, 13, 14]).astype(np.float32)
        # Sharper etched copper trace shoulder transition (0.14 -> 0.19)
        trace_v = (1.0 - _smoothstep(0.14, 0.195, np.abs(t_cell))) * bus_keep
        # Asymmetric interior horizontal routing buses & stub traces (strictly interior -> zero wrap seam!)
        trace_h1 = (1.0 - _smoothstep(0.007, 0.012, np.abs(v - 0.42))) * _smoothstep(0.08, 0.11, u) * (1.0 - _smoothstep(0.68, 0.71, u))
        trace_h2 = (1.0 - _smoothstep(0.007, 0.012, np.abs(v - 0.46))) * _smoothstep(0.16, 0.19, u) * (1.0 - _smoothstep(0.84, 0.87, u))
        trace_h3 = (1.0 - _smoothstep(0.009, 0.015, np.abs(v - 0.84))) * _smoothstep(0.32, 0.35, u) * (1.0 - _smoothstep(0.92, 0.95, u))
        # Asymmetric off-center SMD QFP pad arrays (different positions/lengths on top-left vs bottom-right!)
        smd_x = np.abs(((u * 14.0 + 0.5) % 1.0) - 0.5)
        smd_a = (1.0 - _smoothstep(0.018, 0.025, np.abs(v - 0.24))) * (1.0 - _smoothstep(0.20, 0.27, smd_x)) * ((u > 0.10) & (u < 0.44)).astype(np.float32)
        smd_b = (1.0 - _smoothstep(0.018, 0.025, np.abs(v - 0.66))) * (1.0 - _smoothstep(0.20, 0.27, smd_x)) * ((u > 0.54) & (u < 0.90)).astype(np.float32)
        # Larger rectangular power-FET pad on lower-left for authentic PCB component diversity!
        fet_pad = (1.0 - _smoothstep(0.036, 0.044, np.abs(u - 0.24))) * (1.0 - _smoothstep(0.028, 0.036, np.abs(v - 0.74)))
        # Asymmetric staggered gold-ringed vias (all interior!)
        via_pts = [
            (0.11, 0.12), (0.34, 0.14), (0.61, 0.18), (0.85, 0.15),
            (0.19, 0.38), (0.50, 0.42), (0.76, 0.34), (0.88, 0.46),
            (0.14, 0.58), (0.41, 0.61), (0.64, 0.52), (0.47, 0.84),
            (0.72, 0.84), (0.86, 0.88),
        ]
        pad = np.maximum(np.maximum(smd_a, smd_b), fet_pad)
        hole = np.zeros_like(u)
        for vx, vy in via_pts:
            rv = np.sqrt((u - vx) ** 2 + (v - vy) ** 2)
            pad = np.maximum(pad, 1.0 - _smoothstep(0.018, 0.023, rv))
            hole = np.maximum(hole, 1.0 - _smoothstep(0.0075, 0.0115, rv))
        # Authentic 4-layer FR4 PCB stackup: horizontal traces run on an INNER buried copper plane (no top-layer visual short-circuits!), vertical differential buses run on TOP layer!
        inner_traces = np.maximum(trace_h1, np.maximum(trace_h2, trace_h3))
        fr4_weave = 0.014 * (np.sin(2.0 * math.pi * u * 64.0) * np.sin(2.0 * math.pi * v * 64.0))
        top_trace = np.clip(trace_v, 0.0, 1.0)
        inner_vis = np.clip(inner_traces * (1.0 - top_trace), 0.0, 1.0)
        pad_exposed = np.clip(pad * (1.0 - hole), 0.0, 1.0)
        bare_board = np.clip(1.0 - np.maximum(top_trace, pad_exposed), 0.0, 1.0)
        h = 0.40 + fr4_weave * bare_board + 0.055 * inner_vis * (1.0 - pad_exposed) + 0.23 * top_trace * (1.0 - pad_exposed) + 0.34 * pad_exposed - 0.44 * hole
        cav = hole
        # Encode inner-layer buried traces [0.10..0.22], top-layer copper traces [0.28..0.48], and exposed ENIG gold pads [0.65..1.0] via ridge!
        ridge = np.clip(np.maximum(inner_vis * 0.20, np.maximum(top_trace * 0.46, pad_exposed * 1.0)) * (1.0 - hole), 0.0, 1.0)
        return h.astype(np.float32), 16.5, cav.astype(np.float32), 0.14, ridge.astype(np.float32)

    if detail_id == "acoustic_pyramid_foam":
        # Studio open-cell polyurethane acoustic pyramid foam (8x8 pyramids + genuine open-cell Voronoi foam pores!)
        n = 8.0
        fu = np.abs(((u * n + 0.25) % 1.0) - 0.5) * 2.0
        fv = np.abs(((v * n + 0.25) % 1.0) - 0.5) * 2.0
        pyr = np.clip(1.0 - np.maximum(fu, fv), 0.0, 1.0) ** 0.92
        # Open-cell foam micro-pores: sharp cell walls (f2-f1 small) and concave dark air-bubble pockets (f1 small)
        f1_p, f2_p = _periodic_worley(u, v, 56, seed + 71)
        cell_wall = 1.0 - _smoothstep(0.02, 0.16, f2_p - f1_p)
        pore_pit = 1.0 - _smoothstep(0.08, 0.36, f1_p)
        foam_tex = 0.048 * cell_wall - 0.042 * pore_pit + 0.022 * _periodic_fbm(H, W, seed + 72, (32, 64))
        h = 0.14 + 0.74 * pyr + foam_tex
        cav = np.clip((1.0 - pyr) * 0.85 + 0.35 * pore_pit, 0.0, 1.0)
        ridge = np.clip(_smoothstep(0.72, 0.96, pyr) * 0.85 + 0.35 * cell_wall, 0.0, 1.0)
        return h.astype(np.float32), 16.0, cav.astype(np.float32), 0.88, ridge.astype(np.float32)

    if detail_id == "terry_velcro_loops":
        # Authentic plush woven cotton bath-towel terry cloth: dense 3-layer C-infinity overlapping isotropic curled cotton yarn tufts (~98% plush coverage, ZERO flat voids or diagonal bias!)
        wu = 0.008 * _periodic_fbm(H, W, seed + 73, (4, 8))
        wv = 0.008 * _periodic_fbm(H, W, seed + 74, (4, 8))
        uw = (u + wu) % 1.0
        vw = (v + wv) % 1.0
        beta_t = 5.5
        acc = np.zeros_like(u, dtype=np.float32)
        for nx, ny, off_u, off_v, amp_scale, s_off in (
            (18, 18, 0.0, 0.0, 1.00, 75),
            (22, 22, 0.027, 0.033, 0.94, 85),
            (26, 26, 0.051, 0.019, 0.86, 95),
        ):
            rng = np.random.default_rng(seed + s_off)
            jx = rng.uniform(0.14, 0.86, (ny, nx)).astype(np.float32)
            jy = rng.uniform(0.14, 0.86, (ny, nx)).astype(np.float32)
            ang = rng.uniform(0.0, 2.0 * math.pi, (ny, nx)).astype(np.float32)
            amp = rng.uniform(0.80, 1.05, (ny, nx)).astype(np.float32) * amp_scale
            gx = ((uw + off_u) % 1.0) * nx
            gy = ((vw + off_v) % 1.0) * ny
            ix = np.floor(gx).astype(np.int32)
            iy = np.floor(gy).astype(np.int32)
            fx, fy = gx - ix, gy - iy
            for dy in (-1, 0, 1):
                ny_i = (iy + dy) % ny
                for dx in (-1, 0, 1):
                    nx_i = (ix + dx) % nx
                    ddx = (dx + jx[ny_i, nx_i]) - fx
                    ddy = (dy + jy[ny_i, nx_i]) - fy
                    ca = np.cos(ang[ny_i, nx_i])
                    sa = np.sin(ang[ny_i, nx_i])
                    lx = ddx * ca + ddy * sa
                    ly = -ddx * sa + ddy * ca
                    # Softly curled plush cotton yarn arch (isotropic random angle, NO diagonal stripe bias!)
                    ly_b = ly - 0.28 * (lx * lx)
                    d2 = (lx * 0.92) ** 2 + (ly_b * 1.25) ** 2
                    lobe = np.exp(-2.2 * d2) * amp[ny_i, nx_i]
                    acc += np.exp(beta_t * lobe) - 1.0
        loops = np.log1p(acc) / beta_t
        lo_l, hi_l = float(loops.min()), float(loops.max())
        loops = (loops - lo_l) / max(hi_l - lo_l, 1e-4)
        # Soft isotropic cotton nap micro-fuzz (band-limited to 32 -> zero crunchy normal striping!)
        fuzz = 0.068 * _periodic_fbm(H, W, seed + 77, (8, 16, 32))
        h = np.clip(0.16 + 0.72 * loops + fuzz * (0.40 + 0.60 * loops), 0.05, 0.95)
        cav = _smoothstep(0.56, 0.22, h)
        ridge = _smoothstep(0.60, 0.88, h)
        return h.astype(np.float32), 7.6, cav.astype(np.float32), 0.93, ridge.astype(np.float32)

    if detail_id == "reeded_coin_milling":
        # Crisp bright milled golden-brass coin-edge ridges (20 C1-smooth band-limited flutes -> gleaming metallic sheen!)
        t = 0.5 + 0.5 * np.cos(2.0 * math.pi * u * 20.0)
        # Offset before fractional exponent eliminates infinite-slope valley pinch at t->0
        h = 0.14 + 0.76 * (((t + 0.04) / 1.04) ** 0.62)
        cav = _smoothstep(0.45, 0.0, t)
        ridge = _smoothstep(0.68, 0.98, t)
        return h.astype(np.float32), 14.5, cav.astype(np.float32), 0.08, ridge.astype(np.float32)

    if detail_id == "fluted_reeded_glass":
        # Architectural vertical fluted/reeded privacy glass (12 smooth optical cylindrical ribs with crisp Fresnel shoulders)
        fu = ((u * 12.0 + 0.5) % 1.0) - 0.5
        rib = np.sqrt(np.clip(1.0 - 0.90 * (fu * 2.0) ** 2, 0.0, 1.0))
        # Subtle optical flute waviness along height (drawn glass micro-ripple)
        draw_w = 0.008 * _periodic_fbm(H, W, seed + 80, (4, 12))
        h = 0.16 + 0.76 * rib + draw_w
        # cav marks deep refractive glass edge-absorption troughs; ridge marks bright cylindrical caustic crown
        cav = _smoothstep(0.54, 0.28, rib)
        ridge = _smoothstep(0.82, 0.99, rib)
        return h.astype(np.float32), 14.5, cav.astype(np.float32), 0.05, ridge.astype(np.float32)

    if detail_id == "terrazzo_speckled_stone":
        # Authentic Venetian honed terrazzo: densely packed 3-scale angular marble chips + fine honed cement micro-pitting!
        uw = (u + 0.006 * _periodic_fbm(H, W, seed + 79, (12, 24))) % 1.0
        vw = (v + 0.006 * _periodic_fbm(H, W, seed + 80, (12, 24))) % 1.0
        f1_a, f2_a, cid_a = _periodic_worley_id(uw, vw, 10, seed + 81)
        f1_b, f2_b, cid_b = _periodic_worley_id((uw + 0.037) % 1.0, (vw + 0.053) % 1.0, 20, seed + 82)
        f1_c, f2_c, cid_c = _periodic_worley_id((uw + 0.019) % 1.0, (vw + 0.071) % 1.0, 36, seed + 83)
        # Sharp polygonal chiseled chips via (f2 - f1) border distance across 3 chip sizes (~62% total stone fill!)
        m_a = (cid_a < 0.44).astype(np.float32) * _smoothstep(0.06, 0.15, f2_a - f1_a)
        m_b = (cid_b < 0.48).astype(np.float32) * _smoothstep(0.07, 0.17, f2_b - f1_b) * (1.0 - m_a)
        m_c = (cid_c < 0.52).astype(np.float32) * _smoothstep(0.08, 0.19, f2_c - f1_c) * (1.0 - np.maximum(m_a, m_b))
        chip_mask = np.clip(m_a + m_b + m_c, 0.0, 1.0)
        # Encode mineral palette bucket (0.20..1.00) cleanly across all 3 scales
        bucket = np.where(m_a > 0.5, cid_a / 0.44, np.where(m_b > 0.5, cid_b / 0.48, cid_c / 0.52))
        cem_grit = _periodic_fbm(H, W, seed + 84, (24, 48, 96))
        cem_pits = _smoothstep(1.25, 1.95, -cem_grit) * (1.0 - chip_mask)
        # Dead-flat honed flush surface with subtle micro-pits only in the cement binder
        h = 0.56 + 0.022 * chip_mask + 0.012 * cem_grit * (1.0 - 0.65 * chip_mask) - 0.065 * cem_pits
        cav = np.clip(bucket * chip_mask, 0.0, 1.0)
        ridge = chip_mask
        return h.astype(np.float32), 4.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)

    if detail_id == "woven_rattan_cane":
        # Classic Parisian Thonet octagonal cane webbing with true over-under strand interleaving!
        n = 8.0
        us = (u + 0.5 / n) % 1.0
        vs = (v + 0.5 / n) % 1.0
        iu, iv = np.floor(us * n), np.floor(vs * n)
        fu = np.abs((us * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((vs * n) % 1.0 - 0.5) * 2.0
        fd1 = np.abs(((us + vs) * n * 0.5) % 1.0 - 0.5) * 2.0
        fd2 = np.abs(((us - vs) * n * 0.5) % 1.0 - 0.5) * 2.0
        # Over-under alternation for orthogonal & diagonal cane strips
        par_hv = ((iu + iv) % 2.0) < 0.5
        par_d = (np.floor((us + vs) * n * 0.5) + np.floor((us - vs) * n * 0.5)) % 2.0 < 0.5
        su = np.sqrt(np.clip(1.0 - (fu / 0.28) ** 2, 0.0, 1.0)) * np.where(par_hv, 0.96, 0.68)
        sv = np.sqrt(np.clip(1.0 - (fv / 0.28) ** 2, 0.0, 1.0)) * np.where(par_hv, 0.68, 0.96)
        sd1 = np.sqrt(np.clip(1.0 - (fd1 / 0.22) ** 2, 0.0, 1.0)) * np.where(par_d, 0.84, 0.58)
        sd2 = np.sqrt(np.clip(1.0 - (fd2 / 0.22) ** 2, 0.0, 1.0)) * np.where(par_d, 0.58, 0.84)
        cane = np.maximum.reduce([su, sv, sd1, sd2])
        # Directional longitudinal palm-cane silica striations along each strand
        f_noise = _periodic_fbm(H, W, seed + 86, (16, 48, 96))
        s_u = np.sin(2.0 * math.pi * fu * 6.0 + 0.5 * f_noise) * (su >= np.maximum(sv, np.maximum(sd1, sd2)))
        s_v = np.sin(2.0 * math.pi * fv * 6.0 + 0.5 * f_noise) * (sv > np.maximum(su, np.maximum(sd1, sd2)))
        fiber = 0.022 * f_noise + 0.016 * (s_u + s_v)
        h = 0.08 + 0.82 * cane + fiber * _smoothstep(0.08, 0.25, cane)
        cav = _smoothstep(0.24, 0.06, cane)
        ridge = _smoothstep(0.68, 0.92, cane)
        return h.astype(np.float32), 14.5, cav.astype(np.float32), 0.42, ridge.astype(np.float32)

    if detail_id == "corkboard_pin_surface":
        # Authentic compressed natural corkboard: heavily organic domain-warped bark flakes + subtle compressed joints + sparse torn bark gaps (ZERO straight Voronoi polygons!)
        wu = 0.028 * _periodic_fbm(H, W, seed + 83, (6, 12, 24)) + 0.010 * _periodic_fbm(H, W, seed + 830, (24, 48))
        wv = 0.028 * _periodic_fbm(H, W, seed + 84, (6, 12, 24)) + 0.010 * _periodic_fbm(H, W, seed + 840, (24, 48))
        uw, vw = (u + wu) % 1.0, (v + wv) % 1.0
        f1_a, f2_a, cid_a = _periodic_worley_id(uw, vw, 18, seed + 85)
        f1_b, f2_b, cid_b = _periodic_worley_id((uw + 0.041) % 1.0, (vw + 0.029) % 1.0, 32, seed + 86)
        # Soft narrow compressed bark-granule contacts + sparse irregular torn cork voids + fine interior spongy micro-pits
        seam_a = 1.0 - _smoothstep(0.004, 0.032, f2_a - f1_a)
        fine = _periodic_fbm(H, W, seed + 87, (32, 64, 96))
        micro_spores = _smoothstep(0.95, 1.65, -fine)
        # Sparse genuine torn bark voids only on ~10% of granules
        torn_pit = _smoothstep(0.84, 0.96, cid_b) * (1.0 - _smoothstep(0.10, 0.30, f1_b))
        # Organic spongy bark micro-grain across beveled flake faces (tapered at seams -> zero 1px step cliffs!)
        taper_a = _smoothstep(0.0, 0.14, f2_a - f1_a)
        taper_b = _smoothstep(0.0, 0.14, f2_b - f1_b)
        flake_h = 0.56 + 0.095 * (cid_a - 0.5) * taper_a + 0.065 * (cid_b - 0.5) * taper_b - 0.028 * seam_a - 0.22 * torn_pit - 0.09 * micro_spores + 0.055 * fine
        h = np.clip(flake_h, 0.10, 0.94)
        cav = np.clip(0.09 * seam_a + 0.65 * torn_pit + 0.32 * micro_spores, 0.0, 1.0)
        # Blend two organic bark flake hashes so granules overlap naturally in color!
        ridge = np.clip(cid_a * 0.62 + cid_b * 0.38, 0.0, 1.0)
        return h.astype(np.float32), 13.0, cav.astype(np.float32), 0.87, ridge.astype(np.float32)

    if detail_id == "hammered_copper_dimple":
        # Authentic hand-planished copper: C-infinity 5x5 LogSumExp ball-peen spherical strikes with softly rolled metallic shoulders (ZERO Voronoi crease webs!)
        wu = 0.014 * _periodic_fbm(H, W, seed + 88, (3, 6))
        wv = 0.014 * _periodic_fbm(H, W, seed + 89, (3, 6))
        uw, vw = (u + wu) % 1.0, (v + wv) % 1.0
        # Primary artisan planishing pass (low beta=2.7 -> luxuriously rolled C-infinity shoulders, ZERO Voronoi creases!)
        peen_main = _periodic_hammer_peen(uw, vw, 9, seed + 90, beta=2.7)
        peen_soft = _periodic_hammer_peen((uw + 0.073) % 1.0, (vw + 0.049) % 1.0, 6, seed + 91, beta=2.2)
        hammered = np.clip(0.84 * peen_main + 0.16 * peen_soft, 0.0, 1.0)
        # Subtle hand-forged macro-undulation & delicate burnishing variation (ZERO scanlines!)
        burnish = 0.018 * _periodic_fbm(H, W, seed + 92, (4, 8, 16))
        h = np.clip(0.18 + 0.72 * hammered + burnish, 0.05, 0.96)
        cav = np.clip(1.0 - hammered, 0.0, 1.0)
        ridge = _smoothstep(0.50, 0.88, hammered)
        return h.astype(np.float32), 13.0, cav.astype(np.float32), 0.16, ridge.astype(np.float32)

    if detail_id == "cable_knit_wool":
        # Authentic Fisherman's Aran braided cable-knit sweater: alternating twisted 2-strand rope cables & purl-rib channels!
        n_cables = 4.0
        cu = (u * n_cables) % 1.0 - 0.5  # [-0.5, 0.5] per cable repeat
        # Two braided strands weaving left-right in opposite sinusoidal phase along v
        phase = 2.0 * math.pi * v * 4.0
        s1_x = 0.17 * np.sin(phase)
        s2_x = -0.17 * np.sin(phase)
        d1 = np.abs(cu - s1_x)
        d2 = np.abs(cu - s2_x)
        # Knit stitch rib texture along each rope strand
        st1 = 0.86 + 0.14 * np.sin(2.0 * math.pi * (v * 28.0 + cu * 12.0))
        st2 = 0.86 + 0.14 * np.sin(2.0 * math.pi * (v * 28.0 - cu * 12.0))
        # Over-under height modulation so strand 1 crosses cleanly OVER strand 2 on cos > 0 and UNDER on cos < 0
        z1 = 0.78 + 0.22 * np.cos(phase)
        z2 = 0.78 - 0.22 * np.cos(phase)
        r1 = np.sqrt(np.clip(1.0 - (d1 / 0.145) ** 2, 0.0, 1.0)) * z1 * st1
        r2 = np.sqrt(np.clip(1.0 - (d2 / 0.145) ** 2, 0.0, 1.0)) * z2 * st2
        braid = np.maximum(r1, r2)
        # Flanking purl-stitch background ribs along |cu| > 0.34
        purl = 0.36 * (0.65 + 0.35 * np.sin(2.0 * math.pi * v * 20.0)) * (0.7 + 0.3 * np.cos(2.0 * math.pi * cu * 8.0))
        wool_fuzz = 0.035 * _periodic_fbm(H, W, seed + 91, (24, 48))
        h = 0.10 + 0.82 * np.maximum(braid, purl) + wool_fuzz
        cav = _smoothstep(0.42, 0.14, h)
        ridge = _smoothstep(0.68, 0.90, braid)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.84, ridge.astype(np.float32)

    if detail_id == "penny_round_mosaic":
        # Classic glazed circular penny-round mosaic tiles (12x12 hexagonal packing) with recessed grout
        nx, ny = 12.0, 10.0
        row = np.floor(v * ny)
        u_s = (u + 0.5 * (row % 2.0) / nx) % 1.0
        fu = (u_s * nx) % 1.0 - 0.5
        fv = (v * ny) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        tile = 1.0 - _smoothstep(0.36, 0.44, r)
        pillow = np.sqrt(np.clip(1.0 - (r / 0.46) ** 2, 0.0, 1.0)) * tile
        h = 0.12 + 0.78 * (0.45 * tile + 0.55 * pillow)
        cav = 1.0 - tile
        ridge = _smoothstep(0.75, 0.92, h)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.16, ridge.astype(np.float32)

    if detail_id == "hex_mosaic_tile":
        # True seamless dual-lattice pointy-top hexagonal mosaic tessellation (10x6 dual periods, ZERO gaps!)
        nx, ny = 10.0, 6.0  # ny/nx ≈ 1/sqrt(3) = 0.577 -> near-exact regular hexagons!
        sx, sy = u * nx, v * ny * math.sqrt(3.0)
        # Dual rectangular lattice A (centered at (i+0.5, j*sqrt(3)+sqrt(3)/2)) and B (shifted by (0.5, sqrt(3)/2))
        s3 = math.sqrt(3.0)
        ax = (sx % 1.0) - 0.5
        ay = (sy % s3) - 0.5 * s3
        bx = ((sx + 0.5) % 1.0) - 0.5
        by = ((sy + 0.5 * s3) % s3) - 0.5 * s3
        use_a = (ax * ax + ay * ay) < (bx * bx + by * by)
        hx = np.where(use_a, ax, bx)
        hy = np.where(use_a, ay, by)
        # Exact pointy-top hex apothem distance with C1 corner rounding (zero 1px corner/grout normal flips!)
        d_hex = _smax(np.abs(hx), np.abs(hx) * 0.5 + np.abs(hy) * (0.5 * s3), 0.025)
        tile = 1.0 - _smoothstep(0.415, 0.485, d_hex)
        bevel = 1.0 - _smoothstep(0.36, 0.455, d_hex)
        h = 0.16 + 0.34 * tile + 0.42 * bevel + 0.018 * _periodic_fbm(H, W, seed + 93, (8, 16)) * tile
        cav = 1.0 - tile
        ridge = _smoothstep(0.36, 0.43, d_hex) * tile
        return h.astype(np.float32), 13.5, cav.astype(np.float32), 0.18, ridge.astype(np.float32)

    if detail_id == "fine_asphalt_tarmac":
        # Authentic rolled road asphalt tarmac: continuous dark bitumen-sand bed with densely packed fine C1 mineral aggregate domes (ZERO Voronoi black-crack web!)
        wu = 0.006 * _periodic_fbm(H, W, seed + 93, (8, 16))
        wv = 0.006 * _periodic_fbm(H, W, seed + 94, (8, 16))
        uw, vw = (u + wu) % 1.0, (v + wv) % 1.0
        f1_a, _, cid_a = _periodic_worley_id(uw, vw, 28, seed + 95)
        f1_b, _, cid_b = _periodic_worley_id((uw + 0.023) % 1.0, (vw + 0.041) % 1.0, 44, seed + 96)
        f1_c, _, cid_c = _periodic_worley_id((uw + 0.013) % 1.0, (vw + 0.067) % 1.0, 62, seed + 97)
        # Smooth rolled aggregate crowns embedded in bituminous mortar (continuous C1 domes, no polygon seam cliffs!)
        peb_a = (1.0 - _smoothstep(0.06, 0.54, f1_a)) * (0.65 + 0.35 * cid_a)
        peb_b = (1.0 - _smoothstep(0.05, 0.54, f1_b)) * (0.58 + 0.42 * cid_b) * 0.86
        peb_c = (1.0 - _smoothstep(0.04, 0.54, f1_c)) * (0.55 + 0.45 * cid_c) * 0.70
        gravel = _smax(peb_a, _smax(peb_b, peb_c, 0.10), 0.10)
        road_und = 0.065 * _periodic_fbm(H, W, seed + 98, (4, 8, 16))
        sand = 0.042 * _periodic_fbm(H, W, seed + 99, (24, 48))
        h = np.clip(0.32 + 0.48 * gravel + road_und + sand, 0.08, 0.94)
        # cav = dark recessed bituminous mortar pockets; ridge = exposed rolled basalt/feldspar/quartz stone flecks
        cav = _smoothstep(0.42, 0.12, gravel)
        mineral_tone = np.where(peb_a >= peb_b, cid_a, cid_b)
        ridge = np.clip(_smoothstep(0.28, 0.72, gravel) * (0.35 + 0.65 * mineral_tone), 0.0, 1.0)
        return h.astype(np.float32), 9.5, cav.astype(np.float32), 0.84, ridge.astype(np.float32)

    if detail_id == "venetian_plaster_trowel":
        # Authentic Italian burnished Marmorino Venetian lime plaster: broad feathered overlapping spatula sweeps with contrast driven by glossy burnish vs matte lime (ZERO Voronoi petal/vein outlines!)
        w_lo = _periodic_fbm(H, W, seed + 97, (2, 4))
        w_md = _periodic_fbm(H, W, seed + 98, (4, 8, 16))
        fine = _periodic_fbm(H, W, seed + 101, (16, 32, 64))
        # Broad overlapping hand-troweled spatula sweeps via smooth periodic directional cosine-warped passes (100% periodic, zero cell boundary seams!)
        s1 = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 2.0 + v * 1.0) + 1.55 * w_lo + 0.55 * w_md)
        s2 = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 1.0 - v * 2.0 + 0.31) + 1.45 * w_md - 0.85 * w_lo)
        s3 = 0.5 + 0.5 * np.cos(2.0 * math.pi * (u * 3.0 + v * 2.0 + 0.17) + 1.20 * w_lo + 0.45 * w_md)
        # Soft overlapping feathered lime-putty coats (NO sharp Voronoi lines!)
        coat_a = _smoothstep(0.18, 0.82, s1)
        coat_b = _smoothstep(0.22, 0.78, s2)
        burnish = np.clip(0.48 * coat_a + 0.34 * coat_b + 0.18 * s3, 0.0, 1.0)
        # Delicate smooth steel-trowel surface undulation + fine marble-dust micro-grain
        h = 0.50 + 0.085 * burnish + 0.030 * w_lo + 0.016 * fine
        # Burnished mirror-smooth spatula passes (ridge) vs chalky matte lime undercoat (cav) -> drives authentic Marmorino gloss/matte sheen!
        ridge = np.clip(_smoothstep(0.30, 0.78, burnish), 0.0, 1.0)
        cav = np.clip(_smoothstep(0.62, 0.18, burnish) + 0.15 * _smoothstep(1.4, 2.1, -fine), 0.0, 1.0)
        return h.astype(np.float32), 4.2, cav.astype(np.float32), 0.34, ridge.astype(np.float32)

    if detail_id == "frosted_prism_diffuser":
        # Office/transit ceiling prismatic acrylic diffuser (20x20 C1-beveled diamond micro-pyramids)
        a = (u + v) * 20.0 + 0.25
        b = (u - v) * 20.0 + 0.25
        da = np.sqrt(((a % 1.0) - 0.5) ** 2 + 0.0025) * 2.0
        db = np.sqrt(((b % 1.0) - 0.5) ** 2 + 0.0025) * 2.0
        pyr = np.clip(_smin(1.0 - da, 1.0 - db, 0.06), 0.0, 1.0)
        h = 0.14 + 0.78 * pyr
        cav = _smoothstep(0.32, 0.16, h)
        ridge = _smoothstep(0.75, 0.92, h)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.18, ridge.astype(np.float32)

    if detail_id == "acoustic_kerf_felt":
        # Compressed matte heather wool-PET acoustic felt: crisp CNC-cut straight kerf slits + ultra-fine tangled needle-punched wool micro-fibers (ZERO camo/concrete clouds!)
        cols, rows = 4.0, 12.0
        r_idx = np.floor(v * rows)
        u_s = (u + 0.5 * (r_idx % 2.0) / cols) % 1.0
        fu = (u_s * cols) % 1.0 - 0.5
        fv = (v * rows) % 1.0 - 0.5
        edge_jit = 0.0035 * _periodic_fbm(H, W, seed + 98, (64, 128))
        px_c = np.maximum(np.abs(fu) - 0.31, 0.0)
        d_slit = np.sqrt(px_c * px_c + (fv * 0.56) ** 2) + edge_jit
        # Crisp narrow felt kerf slit without wide dark halo pinch
        slit_core = 1.0 - _smoothstep(0.032, 0.054, d_slit)
        slit_lip = 1.0 - _smoothstep(0.048, 0.074, d_slit)
        # Ultra-fine uniform tangled wool micro-fibers (smooth C1 domes & strands -> zero 1px cell cliffs!)
        f1_f1, _, f_tuft1 = _periodic_worley_id(u, v, 96, seed + 100)
        f1_f2, _, f_tuft2 = _periodic_worley_id((u + 0.011) % 1.0, (v + 0.019) % 1.0, 128, seed + 101)
        strand_a = np.sin(2.0 * math.pi * (u * 80.0 + v * 40.0) + 1.4 * _periodic_fbm(H, W, seed + 102, (16, 32)))
        strand_b = np.sin(2.0 * math.pi * (u * 44.0 - v * 88.0) + 1.4 * _periodic_fbm(H, W, seed + 103, (16, 32)))
        fuzz = 0.018 * (0.5 - np.clip(f1_f1, 0.0, 1.0)) + 0.016 * (0.5 - np.clip(f1_f2, 0.0, 1.0)) + 0.012 * (strand_a + strand_b)
        h = 0.76 - 0.56 * slit_core - 0.04 * slit_lip + fuzz * (1.0 - slit_core)
        cav = np.clip(slit_core + 0.06 * slit_lip, 0.0, 1.0)
        ridge = np.clip(0.50 + 0.25 * (f_tuft1 - 0.5) + 0.22 * (f_tuft2 - 0.5) + 0.14 * strand_a, 0.0, 1.0) * (1.0 - slit_core)
        return h.astype(np.float32), 9.0, cav.astype(np.float32), 0.94, ridge.astype(np.float32)

    if detail_id == "patinated_verdigris_bronze":
        # Authentic aged sand-cast architectural bronze: subtle organic cast undulations & fine micro-pitting + gravity-streaked crevice verdigris!
        cast_lo = _periodic_fbm(H, W, seed + 110, (2, 4, 8))
        cast_md = _periodic_fbm(H, W, seed + 111, (8, 16, 32))
        cast_hi = _periodic_fbm(H, W, seed + 112, (32, 64, 96))
        # Subtle shallow cast corrosion pits & gentle hand-worked bronze surface relief (ZERO raw Voronoi dimples!)
        pits = _smoothstep(0.70, 1.45, -cast_md - 0.55 * cast_hi)
        # Vertical rainwater patina streaks (higher freq along u, slow gravity flow along v)
        drip_u = _periodic_fbm(H, W, seed + 113, (6, 12, 24))
        drip = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 5.0 + 0.18 * cast_lo) + 0.65 * drip_u)
        # Verdigris & cuprite settle strictly inside recessed troughs, cast pits, and vertical rainwater channels!
        recess = np.clip(0.45 * _smoothstep(0.45, -0.65, cast_lo) + 0.42 * pits + 0.28 * (drip ** 2.2) * _smoothstep(-0.2, 0.9, cast_md), 0.0, 1.0)
        ox_halo = _smoothstep(0.24, 0.58, recess + 0.16 * cast_md)
        verd_crust = _smoothstep(0.46, 0.78, recess + 0.18 * cast_hi)
        # Raised cast bronze stays smooth satin metal; encrusted verdigris gets powdery micro-rough relief
        h = 0.62 + 0.10 * cast_lo + 0.04 * cast_md - 0.20 * pits - 0.12 * recess + 0.045 * cast_hi * verd_crust + 0.015 * cast_hi
        cav = np.clip(verd_crust, 0.0, 1.0)
        ridge = np.clip(ox_halo * (1.0 - verd_crust * 0.80), 0.0, 1.0)
        return h.astype(np.float32), 11.0, cav.astype(np.float32), 0.20, ridge.astype(np.float32)

    if detail_id == "carrara_veined_marble":
        # Authentic polished Italian Statuario/Carrara marble: pure phase-warped edge-to-edge sheared geological vein strata + broad smoky mineral diffusion (MATHEMATICALLY ZERO closed oval/kidney loops!)
        w_lo = _periodic_fbm(H, W, seed + 115, (2, 4, 8))
        w_md = _periodic_fbm(H, W, seed + 116, (6, 12, 24))
        w_hi = _periodic_fbm(H, W, seed + 117, (16, 32, 64))
        w_b  = _periodic_fbm(H, W, seed + 118, (3, 6, 12))
        # Pure phase-domain warp INSIDE sin() -> monotonic diagonal bedding flow spanning edge-to-edge with ZERO closed isoline ovals!
        p1 = (u * 1.0 + v * 1.0) + 0.16 * w_lo + 0.052 * w_md + 0.015 * w_hi
        p2 = (u * 1.0 + v * 1.0 + 0.37) + 0.18 * w_b - 0.058 * w_md + 0.014 * w_hi
        p3 = (u * 2.0 + v * 1.0 + 0.19) + 0.14 * w_lo + 0.048 * w_b + 0.016 * w_hi
        r1 = np.abs(np.sin(math.pi * p1))
        r2 = np.abs(np.sin(math.pi * p2))
        r3 = np.abs(np.sin(math.pi * p3))
        taper1 = _smoothstep(-0.75, 0.70, w_lo + 0.35 * w_md)
        taper2 = _smoothstep(-0.45, 0.85, -w_lo + 0.45 * w_b)
        taper3 = _smoothstep(-0.15, 0.90, w_md + 0.35 * w_lo)
        # Layered sharp crystalline vein cores + wispy feathered secondary filaments
        v_primary = np.exp(-((r1 / 0.055) ** 2)) * (0.42 + 0.58 * taper1)
        v_secondary = np.exp(-((r2 / 0.036) ** 2)) * (0.20 + 0.80 * taper2) * 0.68
        v_filament = np.exp(-((r3 / 0.024) ** 2)) * taper3 * 0.48
        sharp_v = np.clip(v_primary + v_secondary + v_filament, 0.0, 1.0)
        # Wide layered translucent smoky mineral clouding diffusing softly into warm-white calcite!
        cloud_v = np.clip(
            np.exp(-((r1 / 0.38) ** 2)) * 0.60 * (0.40 + 0.60 * taper1)
            + np.exp(-((r2 / 0.30) ** 2)) * 0.44 * (0.30 + 0.70 * taper2)
            + np.exp(-((r3 / 0.22) ** 2)) * 0.24 * taper3
            + np.clip(0.14 + 0.16 * w_lo + 0.08 * w_md, 0.0, 0.32),
            0.0, 1.0,
        )
        # Mirror-polished calcite slab with delicate natural micro-pitting strictly along mineral vein seams
        h = 0.74 - 0.014 * sharp_v + 0.008 * w_lo + 0.004 * w_md
        cav = cloud_v
        ridge = sharp_v
        return h.astype(np.float32), 2.4, cav.astype(np.float32), 0.06, ridge.astype(np.float32)

    if detail_id == "herringbone_tweed_wool":
        # Authentic woven Harris/Donegal herringbone tweed: C1-smooth interlocked warp/weft yarn crossings + fuzzy wool nap
        cols = 4.0
        us = (u + 0.5 / cols) % 1.0
        # Smoothly rounded triangle wave for interlocked chevron twill
        zig = 0.25 * (2.0 / math.pi) * np.arcsin(0.92 * np.sin(2.0 * math.pi * us * (cols * 0.5))) + 0.25
        # Half-cell phase-centered yarn crossings -> strictly C1 continuous at u=0/1 and v=0/1!
        nu, nv = 24.0, 24.0
        fu = ((u + 0.5 / nu) * nu) % 1.0
        fv = ((v + 0.5 / nv) * nv) % 1.0
        twill_wave = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 12.0 + zig * 6.0))
        warp_y = (0.5 - 0.5 * np.cos(2.0 * math.pi * fu)) ** 0.75
        weft_y = (0.5 - 0.5 * np.cos(2.0 * math.pi * fv)) ** 0.75
        weave = twill_wave * (0.40 + 0.55 * warp_y) + (1.0 - twill_wave) * (0.30 + 0.50 * weft_y)
        nap = 0.09 * _periodic_fbm(H, W, seed + 119, (24, 48, 96))
        h = 0.16 + 0.70 * weave + nap
        cav = _smoothstep(0.46, 0.20, h)
        ridge = _smoothstep(0.66, 0.88, h)
        return h.astype(np.float32), 10.0, cav.astype(np.float32), 0.88, ridge.astype(np.float32)

    if detail_id == "waffle_cotton_linen":
        # Authentic woven waffle-weave cotton (6x6 honeycomb pockets, 18 interlaced threads, phase-centered)
        n = 6.0
        tw = n * 3.0
        us = (u + 0.5 / tw) % 1.0
        vs = (v + 0.5 / tw) % 1.0
        fu = np.abs((us * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((vs * n) % 1.0 - 0.5) * 2.0
        wall = np.maximum(fu, fv)
        pocket = _smoothstep(0.32, 0.84, wall)
        tu = np.abs((us * tw) % 1.0 - 0.5) * 2.0
        tv = np.abs((vs * tw) % 1.0 - 0.5) * 2.0
        ix = np.floor(us * tw); iy = np.floor(vs * tw)
        over = ((ix + iy) % 2.0) < 0.5
        yu = np.sqrt(np.clip(1.0 - (tu / 0.84) ** 2, 0.0, 1.0)) * np.where(over, 1.0, 0.54)
        yv = np.sqrt(np.clip(1.0 - (tv / 0.84) ** 2, 0.0, 1.0)) * np.where(over, 0.54, 1.0)
        cloth = np.maximum(yu, yv)
        fuzz = 0.025 * _periodic_fbm(H, W, seed + 122, (32, 64))
        h = 0.14 + 0.46 * (pocket ** 1.08) + 0.32 * cloth + fuzz
        cav = np.clip((1.0 - pocket) * 0.68 + (1.0 - cloth) * 0.36, 0.0, 1.0)
        ridge = _smoothstep(0.68, 0.90, h)
        return h.astype(np.float32), 12.0, cav.astype(np.float32), 0.84, ridge.astype(np.float32)

    if detail_id == "damascus_folded_steel":
        # Authentic Japanese forged Raindrop/Ladder Damascus steel: dense high-frequency folded micro-layers + crisp acid-etch steps + belt-grind grain!
        f1 = _periodic_fbm(H, W, seed + 123, (2, 4, 8))
        f2 = _periodic_fbm(H, W, seed + 124, (6, 12, 24))
        # Raindrop forging whorls (periodic Voronoi pools) perturb folded billet layers organically!
        f1_w, _ = _periodic_worley(u, v, 5, seed + 125)
        whorl = np.exp(-((f1_w / 0.36) ** 2)) * 2.8
        # Tight 28-layer forged billet folds with crisp acid-etched topographic terrace steps
        phase = 2.0 * math.pi * (u * 22.0 + v * 5.0) + 5.2 * f1 + 1.15 * f2 + whorl
        striation = np.clip(0.5 + 0.49 * np.tanh(3.8 * np.sin(phase)) + 0.08 * np.sin(phase * 2.0 + 0.5), 0.0, 1.0)
        # Fine directional longitudinal belt-grind micro-scratches across blade face
        grind = 0.012 * np.sin(2.0 * math.pi * v * 128.0 + 0.5 * f2)
        fine_etch = 0.014 * _periodic_fbm(H, W, seed + 126, (32, 64, 96)) * (1.0 - striation)
        h = 0.46 + 0.14 * striation + grind + fine_etch
        cav = 1.0 - striation
        ridge = striation
        return h.astype(np.float32), 6.5, cav.astype(np.float32), 0.15, ridge.astype(np.float32)

    if detail_id == "mermaid_fishscale_tile":
        # Authentic Moroccan Zellige glazed fish-scale tiles: lush pillowed convex glaze domes + handmade kiln ripple + crisp recessed grout!
        nx, ny = 6.0, 10.0
        tile_h = np.full((H, W), 0.10, dtype=np.float32)
        tile_grout = np.ones((H, W), dtype=np.float32)
        tile_edge = np.zeros((H, W), dtype=np.float32)
        wobble = _periodic_fbm(H, W, seed + 126, (6, 12, 24))
        glaze_rip = _periodic_fbm(H, W, seed + 1270, (12, 24))
        base_ry = np.floor(v * ny)
        for row_off in (-2, -1, 0, 1, 2):
            ry = base_ry + row_off
            u_s = (u + 0.5 * (ry % 2.0) / nx) % 1.0
            rx = np.floor(u_s * nx)
            fu = (u_s * nx) % 1.0 - 0.5
            fv = (v * ny - ry) * 0.5
            r = np.sqrt(fu * fu + fv * fv) + 0.008 * wobble
            # Smooth C1 anti-aliased overlapping mask (eliminates hard binary r < 0.50 cliff!)
            cov = 1.0 - _smoothstep(0.455, 0.505, r)
            grout_rim = _smoothstep(0.415, 0.485, r)
            # Smooth convex ceramic glaze pillow with bounded slope near rim (zero sqrt infinite-slope spike!)
            dome = np.sqrt(np.clip(1.0 - (r / 0.53) ** 2, 0.04, 1.0))
            edge_pool = _smoothstep(0.28, 0.43, r) * (1.0 - grout_rim)
            # Slight per-tile handmade kiln tilt/height variance
            t_jit = 0.028 * np.sin((rx % nx) * 17.3 + (ry % ny) * 31.7)
            scale_h = 0.10 + 0.84 * (1.0 - grout_rim) * (0.26 + 0.70 * dome + 0.04 * glaze_rip) + t_jit * (1.0 - grout_rim)
            tile_h = tile_h * (1.0 - cov) + scale_h * cov
            tile_grout = tile_grout * (1.0 - cov) + grout_rim * cov
            tile_edge = tile_edge * (1.0 - cov) + edge_pool * cov
        h = np.clip(tile_h, 0.05, 0.98)
        cav = tile_grout
        ridge = tile_edge
        return h.astype(np.float32), 14.5, cav.astype(np.float32), 0.09, ridge.astype(np.float32)

    if detail_id == "burlap_jute_sackcloth":
        # Authentic coarse jute burlap: interlocking 3-ply twisted rough fibers + slub irregularity + dark open weave pores (C1-smooth over/under!)
        n = 12.0
        jit_u = 0.10 * _periodic_fbm(H, W, seed + 127, (4, 8))
        jit_v = 0.10 * _periodic_fbm(H, W, seed + 128, (4, 8))
        u_j = u + jit_u * 0.08; v_j = v + jit_v * 0.08
        fu = np.abs((u_j * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((v_j * n) % 1.0 - 0.5) * 2.0
        # C1-smooth sinusoidal over-under weave modulation (replaces discontinuous binary step!)
        weave_mod = np.cos(math.pi * u_j * n) * np.cos(math.pi * v_j * n)
        over_u = 0.73 + 0.27 * weave_mod
        over_v = 0.73 - 0.27 * weave_mod
        twist_u = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v_j * n * 3.0 + fu * 1.2))
        twist_v = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u_j * n * 3.0 + fv * 1.2))
        w_u = 0.56 + 0.12 * _periodic_fbm(H, W, seed + 129, (4, 8))
        w_v = 0.56 + 0.12 * _periodic_fbm(H, W, seed + 130, (4, 8))
        prof_u = np.sqrt(np.clip(1.0 - (fu / (w_u * 1.08)) ** 2, 0.02, 1.0)) * (1.0 - _smoothstep(0.80, 1.04, fu / w_u))
        prof_v = np.sqrt(np.clip(1.0 - (fv / (w_v * 1.08)) ** 2, 0.02, 1.0)) * (1.0 - _smoothstep(0.80, 1.04, fv / w_v))
        yu = prof_u * (0.74 + 0.26 * twist_u) * over_u
        yv = prof_v * (0.74 + 0.26 * twist_v) * over_v
        fibers = 0.085 * _periodic_fbm(H, W, seed + 1300, (24, 48, 64))
        yarn_h = _smax(yu, yv, 0.06)
        h = 0.10 + 0.80 * yarn_h + fibers * np.clip(yarn_h * 2.5, 0.0, 1.0)
        cav = _smoothstep(0.28, 0.11, h)
        ridge = _smoothstep(0.66, 0.88, h)
        return h.astype(np.float32), 13.5, cav.astype(np.float32), 0.84, ridge.astype(np.float32)

    if detail_id == "cast_iron_skillet_pebble":
        # Authentic Lodge seasoned sand-cast iron: fine Nyquist-safe sand-molded micro-pebbles + subtle foundry cast undulation!
        f1a, _ = _periodic_worley(u, v, 42, seed + 131)
        f1b, _ = _periodic_worley((u + 0.013) % 1.0, (v + 0.021) % 1.0, 56, seed + 132)
        peb_a = np.clip(1.0 - (f1a / 0.65) ** 2, 0.0, 1.0) ** 0.85
        peb_b = np.clip(1.0 - (f1b / 0.65) ** 2, 0.0, 1.0) ** 0.85
        peb = 0.68 * peb_a + 0.32 * peb_b
        cast_und = 0.08 * _periodic_fbm(H, W, seed + 133, (4, 8, 16))
        pits = _smoothstep(0.38, 0.12, peb)
        crowns = _smoothstep(0.56, 0.88, peb)
        h = 0.34 + 0.42 * peb + cast_und
        cav = pits
        ridge = crowns
        return h.astype(np.float32), 6.2, cav.astype(np.float32), 0.42, ridge.astype(np.float32)

    if detail_id == "tufted_emerald_velvet":
        # Authentic Chesterfield tufted velvet: plush diamond cushions + sharp radiating cloth tension folds + deep fabric buttons + brushed nap!
        nx, ny = 3.0, 3.0
        a = (u * nx + v * ny) % 1.0 - 0.5
        b = (u * nx - v * ny) % 1.0 - 0.5
        d_btn = np.sqrt(a * a + b * b)
        # Sharp folded fabric pleats running diagonally between buttons + organic plush cushion sag
        crease_a = _smoothstep(0.0, 0.11, np.abs(a))
        crease_b = _smoothstep(0.0, 0.11, np.abs(b))
        dimple = _smoothstep(0.048, 0.34, d_btn) ** 0.42
        pillow = (crease_a * crease_b) ** 0.38 * dimple
        # Sharper, asymmetric radiating cloth tension folds converging into each deep button hole
        ang = np.arctan2(b, a)
        pleat_env = np.exp(-((d_btn * 3.0) ** 2)) * _smoothstep(0.052, 0.105, d_btn)
        wrinkle = (0.14 * np.cos(8.0 * ang) + 0.07 * np.cos(16.0 * ang + 0.45) + 0.035 * np.sin(12.0 * ang)) * pleat_env
        # Deeply recessed fabric-covered button with dark collar gap
        btn_collar = np.exp(-(((d_btn - 0.064) / 0.014) ** 2))
        btn = (1.0 - _smoothstep(0.046, 0.064, d_btn)) * np.sqrt(np.clip(1.0 - (d_btn / 0.068) ** 2, 0.0, 1.0))
        # Fine brushed velvet micro-nap & subtle crushed-pile patches
        nap_macro = _periodic_fbm(H, W, seed + 132, (4, 8, 16))
        pile = 0.032 * _periodic_fbm(H, W, seed + 133, (32, 64, 96)) + 0.015 * nap_macro
        h = 0.12 + 0.72 * pillow + 0.26 * btn - 0.14 * btn_collar + wrinkle + pile
        cav = np.clip((1.0 - pillow) * 0.88 + 0.65 * btn_collar + np.clip(-wrinkle * 3.4, 0.0, 0.42), 0.0, 1.0) * (1.0 - btn)
        ridge = np.clip(_smoothstep(0.62, 0.90, pillow) + np.clip(wrinkle * 3.2, 0.0, 0.35), 0.0, 1.0)
        return h.astype(np.float32), 16.0, cav.astype(np.float32), 0.90, ridge.astype(np.float32)

    if detail_id == "woven_bamboo_tatami":
        # Authentic Japanese igusa rush tatami: 22 clearly rounded cylindrical rush stalks + longitudinal grass striations + warp tuck
        cols, rows = 8.0, 22.0
        us = (u + 0.5 / cols) % 1.0
        vs = (v + 0.5 / rows) % 1.0
        col_i = np.floor(us * cols)
        fu = (us * cols) % 1.0 - 0.5
        bind = 1.0 - 0.30 * _smoothstep(0.36, 0.49, np.abs(fu))
        fv_s = (vs * rows + 0.5 * (col_i % 2.0)) % 1.0 - 0.5
        stalk = np.sqrt(np.clip(1.0 - 0.95 * (fv_s * 2.0) ** 2, 0.0, 1.0))
        # Fine longitudinal striations running along each rush stalk
        striat = 0.045 * np.sin(2.0 * math.pi * v * rows * 4.0 + 0.6 * _periodic_fbm(H, W, seed + 135, (8, 24)))
        h = 0.18 + 0.68 * stalk * bind + striat * stalk
        cav = np.clip((1.0 - stalk) * 0.65 + (1.0 - bind) * 0.70, 0.0, 1.0)
        ridge = _smoothstep(0.72, 0.94, stalk) * bind
        return h.astype(np.float32), 11.0, cav.astype(np.float32), 0.64, ridge.astype(np.float32)

    if detail_id == "travertine_porous_stone":
        # Authentic honed Roman travertine: gritty calcareous limestone strata + terraced horizontally-elongated erosion pores!
        s1 = _periodic_fbm(H, W, seed + 137, (2, 4, 8))
        strata = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 10.0 + 0.18 * s1))
        # Horizontally elongated pore field (low freq along u, high freq along v) -> authentic sedimentary voids!
        p_elong = _periodic_fbm(H, W, seed + 138, (4, 8)) * 0.5 + _periodic_fbm(H, W, seed + 139, (12, 24)) * 0.5
        p_grit = _periodic_fbm(H, W, seed + 140, (24, 48, 96))
        band_wave = np.sin(2.0 * math.pi * (v * 14.0 + 0.16 * s1))
        cav_field = band_wave * 0.52 - p_elong * 0.55 + 0.25 * p_grit
        # Terraced 3-step cavity depth + sharp gritty micro-pinholes (zero flat camo puddles!)
        step1 = _smoothstep(0.58, 0.72, cav_field)
        step2 = _smoothstep(0.76, 0.90, cav_field)
        step3 = _smoothstep(0.94, 1.08, cav_field)
        pinholes = _smoothstep(0.82, 1.20, -p_grit) * 0.42
        total_pore = np.clip(0.32 * step1 + 0.36 * step2 + 0.32 * step3 + pinholes, 0.0, 1.0)
        # Fine calcareous limestone grit across the whole face
        grit_h = 0.045 * p_grit
        h = 0.78 + 0.025 * strata + grit_h * (1.0 - 0.5 * step1) - 0.16 * step1 - 0.22 * step2 - 0.24 * step3 - 0.14 * pinholes
        cav = total_pore
        ridge = _smoothstep(0.74, 0.84, h) * (1.0 - step1)
        return h.astype(np.float32), 13.0, cav.astype(np.float32), 0.48, ridge.astype(np.float32)

    # "gilded_tin_ceiling_tile"
    # Authentic vintage stamped pressed-tin ceiling tile: crisp C1-embossed rosette medallion + beaded frame + hand-planished patina
    n = 2.0
    fu = (u * n) % 1.0 - 0.5
    fv = (v * n) % 1.0 - 0.5
    ax, ay = np.abs(fu), np.abs(fv)
    r = np.sqrt(fu * fu + fv * fv)
    ang = np.arctan2(fv, fu)
    box_d = _smax(ax, ay, 0.025)
    d_outer = np.sqrt((box_d - 0.44) ** 2 + 0.00018)
    w_ax = _smoothstep(-0.04, 0.04, ax - ay)
    bead_coord = w_ax * fv + (1.0 - w_ax) * fu
    bead_wave = 0.68 + 0.32 * np.cos(2.0 * math.pi * bead_coord * 18.0)
    frame = (1.0 - _smoothstep(0.020, 0.052, d_outer)) * bead_wave
    d_inner = np.sqrt((box_d - 0.34) ** 2 + 0.00018)
    inner_f = 1.0 - _smoothstep(0.015, 0.042, d_inner)
    petal = 0.5 + 0.5 * np.cos(8.0 * ang)
    rosette = (1.0 - _smoothstep(0.165, 0.235, r - 0.065 * petal)) * (0.50 + 0.50 * petal)
    d_ring = np.sqrt((r - 0.12) ** 2 + 0.00015)
    ring = 1.0 - _smoothstep(0.014, 0.036, d_ring)
    boss = 1.0 - _smoothstep(0.040, 0.085, r)
    d_cor = np.sqrt((ax - 0.39) ** 2 + (ay - 0.39) ** 2)
    corner = 1.0 - _smoothstep(0.024, 0.054, d_cor)
    relief = _smax(frame * 0.86, inner_f * 0.72, 0.05)
    relief = _smax(relief, rosette * 0.90, 0.05)
    relief = _smax(relief, ring * 0.92, 0.05)
    relief = _smax(relief, boss * 0.98, 0.05)
    relief = _smax(relief, corner * 0.80, 0.05)
    micro = 0.024 * _periodic_fbm(H, W, seed + 141, (12, 24, 48))
    h = 0.16 + 0.76 * relief + micro
    cav = _smoothstep(0.30, 0.17, h)
    ridge = _smoothstep(0.65, 0.90, h)
    return h.astype(np.float32), 15.5, cav.astype(np.float32), 0.22, ridge.astype(np.float32)


def apply_daily_wear(h0, slope_scale, cav0, base_r, ridge0, mat_kind: str, wear_state: str, seed: int):
    H, W = h0.shape
    h = h0.astype(np.float32).copy()
    fbm_m = _periodic_fbm(H, W, seed + 101, (4, 8, 16))
    fbm_f = _periodic_fbm(H, W, seed + 102, (16, 32, 64))

    scuff_mask = np.zeros((H, W), dtype=np.float32)
    if wear_state == "1_lived_in":
        h += 0.035 * fbm_m + 0.022 * fbm_f
        scuff_mask = np.clip((fbm_m - 0.35) * 1.4, 0.0, 1.0)
    elif wear_state == "2_weathered":
        # Subtle warping + scratches/pits + worn-down ridges
        h = ng.node_height_warp_flow({"circular": True}, {"height": h}, scale=6.0, amplitude=0.008, octaves=2, seed=seed + 5)["height"]
        pits = np.clip((fbm_f - 0.85) * 2.2, 0.0, 1.0)
        h += 0.06 * fbm_m - 0.14 * pits
        scuff_mask = np.clip((fbm_m - 0.10) * 1.3 + pits * 0.8, 0.0, 1.0)

    N = _height_to_normal_periodic(h, slope=slope_scale)
    # Add fine material micro-grain
    g_kind = "mineral" if mat_kind in ("concrete", "ceramic") else ("brushed" if mat_kind in ("metal", "carbon") else "cast")
    g_str = {"0_fresh": 0.05, "1_lived_in": 0.10, "2_weathered": 0.17}[wear_state]
    if mat_kind in ("ceramic", "carbon"):
        g_str *= 0.45
    N = ng.node_micro_grain({"circular": True}, {"normal": N}, kind=g_kind, scale=36.0, strength=g_str, seed=seed + 7)["normal"]
    N = ng.node_normal_clamp_slope({"circular": True}, {"normal": N}, max_angle_deg=58.0)["normal"]

    h_norm = dp.normalize_height(h, clip_pct=0.5)
    ao_u8 = dp.height_to_ao(h_norm, strength=1.55, circular=True)
    ao_f = ao_u8.astype(np.float32) / 255.0

    r_add = {"0_fresh": 0.0, "1_lived_in": 0.06, "2_weathered": 0.14}[wear_state]
    # Grout/cavities are rougher; worn ridges polish slightly on hard surfaces or fuzz on fabric
    ridge_delta = +0.08 if mat_kind in ("fabric", "paper") else -0.10
    rough = np.clip(
        base_r + r_add + 0.34 * cav0 + 0.22 * (1.0 - ao_f) + 0.16 * scuff_mask + ridge_delta * ridge0,
        0.08, 0.96,
    ).astype(np.float32)
    rough = ng.node_roughness_specular_aa({"circular": True}, {"roughness": rough, "normal": N},
                                          strength=0.25, max_boost=0.08)["roughness"]

    w_gain = {"0_fresh": 0.20, "1_lived_in": 0.55, "2_weathered": 0.90}[wear_state]
    wear = np.clip(w_gain * (ridge0 * 0.8 + scuff_mask * 0.6), 0.0, 1.0).astype(np.float32)
    return h_norm, N, ao_u8, ao_f, rough, wear


def main():
    ap = argparse.ArgumentParser(description="Generate 36 Seamless Everyday-Life PBR Surface Details (12 designs x 3 wear states)")
    ap.add_argument("--out", type=Path, default=Path("./output/procedural"))
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    sheets_dir = args.out / "_sheets"
    sheets_dir.mkdir(exist_ok=True)
    manifest = {"version": "v1-daily-details-pbr", "tileable": True, "size": args.size, "states": list(WEAR_STATES), "details": []}
    cards_html = []
    showcase_rows = []

    for idx, (did, desc, mat_kind, tint) in enumerate(DAILY_CATALOG):
        h0, slope_scale, cav0, base_r, ridge0 = build_daily_detail(did, args.size, args.size, seed=2000 + idx * 41)
        entry = {"id": did, "description": desc, "material": mat_kind, "variants": {}}
        strip_panels = []

        for s_idx, ws in enumerate(WEAR_STATES):
            h_n, N, ao_u8, ao_f, rough, wear = apply_daily_wear(
                h0, slope_scale, cav0, base_r, ridge0, mat_kind, ws, seed=3000 + idx * 53 + s_idx * 11
            )
            preview = _lit_preview(N, ao_f, rough, tint)
            vdir = args.out / did / ws
            vdir.mkdir(parents=True, exist_ok=True)

            Image.fromarray(dp.normal_to_u8(N)).save(vdir / "normal.png")
            Image.fromarray(np.clip(h_n * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(vdir / "height.png")
            Image.fromarray(ao_u8).save(vdir / "ao.png")
            Image.fromarray(np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(vdir / "roughness.png")
            Image.fromarray(np.clip(wear * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(vdir / "edge_wear.png")
            Image.fromarray(preview).save(vdir / "preview.png")
            entry["variants"][ws] = str(vdir.relative_to(args.out))

            if s_idx == 0:
                strip_panels.extend([
                    ("Fresh (Lit)", preview),
                    ("Fresh (Normal)", dp.normal_to_u8(N)),
                    ("Fresh (Height)", np.stack([np.clip(h_n * 255.0 + 0.5, 0, 255).astype(np.uint8)] * 3, -1)),
                ])
            elif s_idx == 1:
                strip_panels.extend([
                    ("Lived-In (Lit)", preview),
                    ("Lived-In (Normal)", dp.normal_to_u8(N)),
                ])
            else:
                strip_panels.extend([
                    ("Weathered (Lit)", preview),
                    ("Weathered (Normal)", dp.normal_to_u8(N)),
                    ("Weathered (Rough)", np.stack([np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)] * 3, -1)),
                ])

        sz = 192
        pad = 6
        hdr = 22
        row = Image.new("RGB", (len(strip_panels) * (sz + pad) + pad, sz + hdr + pad), (18, 20, 24))
        dr = ImageDraw.Draw(row)
        for pi, (lbl, arr) in enumerate(strip_panels):
            x = pad + pi * (sz + pad)
            row.paste(Image.fromarray(arr).resize((sz, sz), Image.BICUBIC), (x, hdr))
            dr.text((x + 4, 4), f"{did}: {lbl}" if pi == 0 else lbl, fill=(220, 226, 236))
        sheet_p = sheets_dir / f"{did}.png"
        row.save(sheet_p)
        showcase_rows.append(row)

        b64 = base64.b64encode(sheet_p.read_bytes()).decode("ascii")
        cards_html.append(
            f'<div class="card"><div class="hdr"><b>{idx+1:02d}. {did}</b> '
            f'<span class="tag">{mat_kind}</span> <span class="desc">{desc}</span></div>'
            f'<img src="data:image/png;base64,{b64}"/></div>'
        )
        manifest["details"].append(entry)
        print(f"  [{idx+1:02d}/{len(DAILY_CATALOG)}] {did:26s} (3 wear states x 6 maps = 18 PNGs)", flush=True)

    # Save combined showcase PNG in output directory
    comb = Image.new("RGB", (showcase_rows[0].width, sum(r.height for r in showcase_rows)), (14, 16, 20))
    y = 0
    for r in showcase_rows:
        comb.paste(r, (0, y)); y += r.height
    comb_path = args.out / f"showcase_{len(DAILY_CATALOG)}.png"
    comb.save(comb_path)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Daily Life PBR Surface Details ({len(DAILY_CATALOG)} x 3)</title>
<style>body{{background:#111318;color:#e5e9f0;font-family:system-ui,sans-serif;margin:24px}}
h1{{margin:0 0 8px}}p{{color:#9aa4b2;margin:0 0 20px}}
.card{{background:#181b22;border:1px solid #262b36;border-radius:10px;padding:12px;margin-bottom:16px}}
.hdr{{margin-bottom:8px;font-size:14px}}.tag{{background:#253248;color:#8cc8ff;padding:2px 7px;border-radius:4px;font-size:11px;margin:0 6px}}
.desc{{color:#9aa4b2;font-size:12px}}img{{width:100%;border-radius:6px;display:block}}</style></head>
<body><h1>Everyday Real-World PBR Surface Details ({len(DAILY_CATALOG)} Tileable Details &times; 3 Wear States)</h1>
<p>100% seamlessly tileable 512&times;512 Normal + Height + Cavity AO + Specular-AA Roughness + Edge Wear + Lit Preview.</p>
{''.join(cards_html)}</body></html>"""
    cat_path = args.out / "daily_details_catalog.html"
    cat_path.write_text(html)
    print(f"\nGenerated {len(DAILY_CATALOG)*len(WEAR_STATES)} seamless 512x512 PBR sets in {time.time()-t0:.1f}s!")
    print(f"Showcase PNG -> {comb_path}")
    print(f"Catalog HTML -> {cat_path}")


if __name__ == "__main__":
    main()
