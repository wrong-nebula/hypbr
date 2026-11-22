"""Seamless micro-detail & organic structure helpers for the parametric atlas generators (v1.1).

Everything here is exactly periodic on `[0, 1)^2`:
  * periodic FBM octaves are integer frequencies (clamped below Nyquist for small previews),
  * domain warping uses wrap-around bilinear sampling of periodic fields with periodic offsets,
  * scatter primitives (Worley / ellipse cells) use modular cell lattices.

`apply_detail_pass` adds the high-frequency information the v1.0 parametric families lacked
(fine mineral grain, speckle, fibers, pile, micro-relief) without changing any family's layout.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import procedural as gd


# ---------------------------------------------------------------------------
# Periodic primitives
# ---------------------------------------------------------------------------
def fbm(H: int, W: int, seed: int, octaves: tuple[int, ...]) -> np.ndarray:
    """Periodic FBM (zero mean, unit std) with octaves clamped below Nyquist/1.5."""
    cap = max(2, min(170, min(H, W) // 3))
    octs = tuple(sorted({int(min(o, cap)) for o in octaves}))
    return gd._periodic_fbm(H, W, seed, octs)


def warp(field: np.ndarray, du: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """Sample a periodic field at (u + du, v + dv) with wrap-around bilinear interpolation."""
    H, W = field.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = xx + du * W
    y = yy + dv * H
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)[..., None] if field.ndim == 3 else (x - x0)
    fy = (y - y0)[..., None] if field.ndim == 3 else (y - y0)
    x0i = x0.astype(np.int64) % W
    y0i = y0.astype(np.int64) % H
    x1i = (x0i + 1) % W
    y1i = (y0i + 1) % H
    a = field[y0i, x0i]
    b = field[y0i, x1i]
    c = field[y1i, x0i]
    d = field[y1i, x1i]
    return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy).astype(np.float32)


def isoline(n: np.ndarray, width_uv) -> np.ndarray:
    """Constant-width line along the zero set of a periodic field (|n| / |grad n| distance)."""
    H, W = n.shape
    gx, gy = gd._periodic_grad(n)
    g = np.sqrt((gx * W) ** 2 + (gy * H) ** 2) + 1e-3
    d = np.abs(n) / g
    return np.exp(-np.power(d / np.maximum(width_uv, 1e-5), 2.0)).astype(np.float32)


def stretch(field: np.ndarray, axis: int, k: int) -> np.ndarray:
    """Periodic box filter along one axis (elongates noise into streaks / fibers), renormalized."""
    k = max(1, int(k))
    acc = np.zeros_like(field)
    for s_ in range(-k, k + 1):
        acc += np.roll(field, s_, axis=axis)
    acc /= (2 * k + 1)
    return (acc - acc.mean()) / (acc.std() + 1e-6)


def ridged(n: np.ndarray, sharp: float) -> np.ndarray:
    """Ridged transform of a zero-mean field: thin bright ridges where n crosses zero."""
    return np.exp(-np.power(np.abs(n) * sharp, 1.15)).astype(np.float32)


def speckle(u: np.ndarray, v: np.ndarray, cells: int, density: float, seed: int, radius: float = 0.32) -> tuple[np.ndarray, np.ndarray]:
    """Sparse round specks on a periodic Worley lattice. Returns (mask in [0,1], per-speck id)."""
    cells = int(max(4, cells))
    f1, _, cid = gd._periodic_worley_id(u, v, cells, seed)
    on = (cid > 1.0 - float(density)).astype(np.float32)
    r = radius * (0.65 + 0.7 * np.mod(cid * 7.13, 1.0))
    t = np.clip((f1 - r * 0.55) / (r * 0.45), 0.0, 1.0)
    m = (1.0 - t * t * (3.0 - 2.0 * t)) * on
    return m.astype(np.float32), np.mod(cid * 3.97 + 0.31, 1.0).astype(np.float32)


def ellipse_scatter(H: int, W: int, cells: int, seed: int, aspect: float = 2.6, size: float = 0.85,
                    layers: int = 2) -> dict[str, np.ndarray]:
    """Overlapping, randomly oriented ellipses (leaves, chips, needles, pebbles) on a periodic lattice.

    Returns dict with `mask` (coverage), `dist` (0 at center .. 1 at rim), `along` (signed major-axis
    coordinate), `across` (signed minor-axis coordinate), `id` (per-item hash), `layer` (top layer index).
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H
    out = {k: np.zeros((H, W), np.float32) for k in ("mask", "dist", "along", "across", "id", "layer")}
    out["dist"][:] = 9.0
    for layer in range(layers):
        n = int(cells)
        rng = np.random.default_rng(seed + 977 * layer)
        jx = rng.uniform(0.0, 1.0, (n, n)).astype(np.float32)
        jy = rng.uniform(0.0, 1.0, (n, n)).astype(np.float32)
        ang = rng.uniform(0.0, math.pi, (n, n)).astype(np.float32)
        sz = rng.uniform(0.7, 1.15, (n, n)).astype(np.float32) * size
        hid = rng.uniform(0.0, 1.0, (n, n)).astype(np.float32)
        gx, gy = u * n, v * n
        ix = np.floor(gx).astype(np.int32)
        iy = np.floor(gy).astype(np.int32)
        best = np.full((H, W), 9.0, np.float32)
        b_al = np.zeros((H, W), np.float32)
        b_ac = np.zeros((H, W), np.float32)
        b_id = np.zeros((H, W), np.float32)
        for dy in (-1, 0, 1):
            cy = (iy + dy) % n
            for dx in (-1, 0, 1):
                cx = (ix + dx) % n
                px = gx - (ix + dx + jx[cy, cx])
                py = gy - (iy + dy + jy[cy, cx])
                a = ang[cy, cx]
                ca, sa = np.cos(a), np.sin(a)
                al = (px * ca + py * sa) / sz[cy, cx]
                ac = (-px * sa + py * ca) / sz[cy, cx] * aspect
                d = np.sqrt(al * al + ac * ac)
                m = d < best
                best = np.where(m, d, best)
                b_al = np.where(m, al, b_al)
                b_ac = np.where(m, ac, b_ac)
                b_id = np.where(m, hid[cy, cx], b_id)
        cover = best < 1.0
        take = cover  # later layers lie on top
        out["mask"] = np.where(take, 1.0, out["mask"])
        out["dist"] = np.where(take, best, out["dist"])
        out["along"] = np.where(take, b_al, out["along"])
        out["across"] = np.where(take, b_ac, out["across"])
        out["id"] = np.where(take, b_id, out["id"])
        out["layer"] = np.where(take, float(layer), out["layer"])
    out["dist"] = np.clip(out["dist"], 0.0, 1.0)
    return out


def _rgb(c) -> np.ndarray:
    return np.asarray(c, dtype=np.float32)[None, None, :]


def _mix(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    if t.ndim == 2:
        t = t[..., None]
    return a * (1.0 - t) + b * t


# ---------------------------------------------------------------------------
# Rewritten organic / mineral structures
# ---------------------------------------------------------------------------
def marble_veins(H: int, W: int, seed: int, vein_freq: float, vein_sharp: float, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gated, width-modulated multi-scale vein network (primary + secondary + hairline + halo)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H
    f0 = max(1, int(round(vein_freq)))
    ky = max(1, int(round(max(1.0, vein_freq * angle))))
    drift = 0.6 * np.sin(2.0 * math.pi * (u * f0 + v * ky)).astype(np.float32)
    wu = 0.07 * fbm(H, W, seed + 62, (3, 6, 12))
    wv = 0.07 * fbm(H, W, seed + 63, (3, 6, 12))
    n1 = warp(0.8 * fbm(H, W, seed + 61, (f0, 2 * f0, 4 * f0)) + drift, wu, wv)
    n2 = warp(fbm(H, W, seed + 64, (2 * f0 + 2, 4 * f0 + 4, 8 * f0 + 8)), 0.6 * wv, 0.6 * wu)
    n3 = warp(fbm(H, W, seed + 68, (16, 32, 48)), 0.3 * wu, 0.3 * wv)
    px = 1.0 / 512.0
    base_w = (0.8 + 8.0 / max(vein_sharp, 0.5)) * px
    wmod1 = 0.25 + 1.6 * gd._smoothstep(-0.8, 1.4, fbm(H, W, seed + 69, (3, 6, 12)))
    gate1 = gd._smoothstep(-1.4, 0.1, fbm(H, W, seed + 70, (2, 4, 8)) + 0.5)
    gate2 = gd._smoothstep(-0.2, 1.0, fbm(H, W, seed + 71, (3, 6, 12)))
    gate3 = gd._smoothstep(0.2, 1.3, fbm(H, W, seed + 72, (4, 8, 16)))
    primary = isoline(n1, base_w * wmod1) * gate1
    halo = isoline(n1, base_w * wmod1 * 5.0) * gate1 * 0.45
    secondary = isoline(n2, base_w * 0.45 * (0.4 + wmod1 * 0.6)) * gate2 * 0.75 + isoline(n3, base_w * 0.22) * gate3 * 0.35
    clouds = gd._smoothstep(-1.2, 1.6, fbm(H, W, seed + 66, (2, 4, 8)) + 1.2 * halo)
    return np.clip(primary + 0.6 * halo, 0.0, 1.0).astype(np.float32), np.clip(secondary, 0.0, 1.0).astype(np.float32), clouds.astype(np.float32)


def ground_cover(cover: str, params: dict[str, Any], H: int, W: int, seed: int, c1, c2, c3):
    """Organic terrain surfaces viewed top-down. Returns (h, cav, ridge, alb, r_mod)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H
    c1, c2, c3 = _rgb(c1), _rgb(c2), _rgb(c3)
    lo = fbm(H, W, seed + 301, (2, 4, 8))
    mid = fbm(H, W, seed + 302, (12, 24, 48))
    hi = fbm(H, W, seed + 303, (64, 128, 192))

    if cover in ("leaves", "mulch", "needles"):
        if cover == "leaves":
            sc = ellipse_scatter(H, W, int(params.get("cells", 22)), seed + 310, aspect=1.7, size=0.95, layers=3)
        elif cover == "mulch":
            sc = ellipse_scatter(H, W, int(params.get("cells", 30)), seed + 310, aspect=3.2, size=0.95, layers=3)
        else:
            sc = ellipse_scatter(H, W, int(params.get("cells", 40)), seed + 310, aspect=14.0, size=1.25, layers=3)
        m, dist, pid = sc["mask"], sc["dist"], sc["id"]
        dome = np.sqrt(np.clip(1.0 - dist * dist, 0.0, 1.0))
        curl = 0.25 * sc["across"] * sc["across"] / 8.0 if cover == "leaves" else 0.0
        h = 0.10 + m * (0.30 + 0.16 * sc["layer"] + 0.22 * dome + curl) + 0.05 * mid
        rib = (1.0 - gd._smoothstep(0.03, 0.10, np.abs(sc["across"]) / 2.0)) * m if cover == "leaves" else 0.0
        fib = (0.5 + 0.5 * np.sin(2.0 * math.pi * sc["across"] * 2.5)) * m if cover == "mulch" else 0.0
        item = _mix(c1, c2, np.clip(pid * 1.15 + 0.12 * mid, 0.0, 1.0))
        item = item * (0.80 + 0.30 * np.mod(pid * 5.3, 1.0)[..., None]) * (0.92 + 0.10 * hi[..., None] * 0.3)
        edge_dark = gd._smoothstep(0.75, 1.0, dist) * m
        item = item * (1.0 - 0.25 * edge_dark[..., None]) * (1.0 - 0.30 * np.asarray(rib)[..., None] if cover == "leaves" else 1.0)
        if cover == "mulch":
            item = item * (0.85 + 0.22 * np.asarray(fib)[..., None])
        soil = c3 * (0.75 + 0.25 * (0.5 + 0.5 * mid[..., None]))
        alb = _mix(soil, item, m)
        cav = np.clip((1.0 - m) * 0.9 + 0.35 * edge_dark, 0.0, 1.0)
        ridge = gd._smoothstep(0.4, 0.9, dome * m)
        r_mod = 0.10 * cav - 0.04 * ridge + 0.03 * hi
        return h, cav, ridge, alb, r_mod

    if cover in ("grass", "moss", "lichen", "soil"):
        if cover == "grass":
            n_a, n_b = 72, 144
        elif cover == "moss":
            n_a, n_b = 56, 120
        elif cover == "lichen":
            n_a, n_b = 28, 64
        else:
            n_a, n_b = 40, 96
        f1a, f2a, ida = gd._periodic_worley_id(u, v, n_a, seed + 320)
        f1b, f2b, idb = gd._periodic_worley_id((u + 0.37) % 1.0, (v + 0.61) % 1.0, n_b, seed + 321)
        tuft_a = np.clip(1.0 - f1a * 1.25, 0.0, 1.0)
        tuft_b = np.clip(1.0 - f1b * 1.25, 0.0, 1.0)
        clump = 0.5 + 0.5 * np.tanh(1.3 * (lo + 0.6 * mid))
        if cover == "grass":
            # Blade tips: per-tuft hue jitter, sunlit tips vs. dark under-canopy & thatch
            tip = np.maximum(tuft_a, tuft_b * 0.9)
            hue = np.clip(0.5 * ida + 0.5 * idb + 0.25 * mid, 0.0, 1.0)
            g = _mix(c1, c2, hue) * (0.55 + 0.75 * tip[..., None])
            dry = gd._smoothstep(0.55, 1.0, 0.5 + 0.5 * lo) * gd._smoothstep(0.82, 0.98, ida)
            alb = _mix(g, c3 * 1.8 + 0.05, dry * 0.6)
            gap = 1.0 - gd._smoothstep(0.03, 0.18, f2a - f1a) * gd._smoothstep(0.02, 0.14, f2b - f1b)
            alb = _mix(alb, c3, gap * 0.55)
            h = 0.20 + 0.35 * tip + 0.20 * clump + 0.05 * hi
        elif cover == "moss":
            tip = np.maximum(tuft_a * 0.8, tuft_b)
            hue = np.clip(0.6 * idb + 0.3 * clump + 0.1 * hi, 0.0, 1.0)
            alb = _mix(c1, c2, hue) * (0.6 + 0.6 * tip[..., None])
            gap = 1.0 - gd._smoothstep(0.02, 0.16, f2b - f1b)
            alb = _mix(alb, c3, gap * 0.45) * (0.88 + 0.24 * clump[..., None])
            h = 0.15 + 0.50 * clump + 0.18 * tip + 0.04 * hi
        elif cover == "lichen":
            branch = ridged(warp(mid, 0.02 * lo, 0.02 * hi), 2.2)
            clump = 0.5 + 0.5 * np.tanh(1.3 * (0.35 * lo + mid))
            cov = gd._smoothstep(0.30, 0.60, clump + 0.25 * branch)
            alb = _mix(c3 * 1.2, _mix(c1, c2, idb) * (0.8 + 0.3 * branch[..., None]), cov)
            h = 0.20 + 0.40 * cov * (0.6 + 0.4 * branch) + 0.05 * hi
            tip = branch * cov
        else:  # soil: crumbs + pebbles + organic flecks
            crumbs = np.maximum(tuft_a, tuft_b * 0.8)
            alb = _mix(c1, c2, np.clip(0.5 + 0.35 * mid + 0.4 * (ida - 0.5), 0.0, 1.0)) * (0.7 + 0.45 * crumbs[..., None])
            peb, pid = speckle(u, v, 64, float(params.get("pebbles", 0.10)), seed + 330, 0.30)
            alb = _mix(alb, c3 * 0 + _rgb((0.38, 0.35, 0.31)) * (0.7 + 0.6 * pid[..., None]), peb * 0.85)
            fleck, _ = speckle(u, v, 110, 0.10, seed + 331, 0.22)
            alb = _mix(alb, c2 * 1.25, fleck * 0.5)
            h = 0.25 + 0.35 * crumbs + 0.25 * clump + 0.25 * peb + 0.04 * hi
            tip = crumbs
        cav = np.clip(1.0 - gd._smoothstep(0.15, 0.75, h), 0.0, 1.0)
        ridge = gd._smoothstep(0.5, 0.95, np.asarray(tip))
        r_mod = 0.08 * cav - 0.05 * ridge + 0.03 * hi
        return h.astype(np.float32), cav, ridge, alb, r_mod

    # snow (fresh powder or trampled slush)
    drift = 0.5 + 0.5 * np.tanh(0.9 * lo + 0.4 * mid)
    f1, _, _ = gd._periodic_worley_id(u, v, int(params.get("cells", 32)), seed + 340)
    tr = float(params.get("trample", 0.0))
    ruts = ridged(warp(fbm(H, W, seed + 342, (4, 8)), 0.03 * mid, 0.03 * lo), 1.8)
    prints = gd._smoothstep(0.35, 0.9, ruts) * tr
    h = 0.30 + 0.45 * drift - 0.22 * prints + 0.03 * mid + 0.012 * hi
    sparkle, _ = speckle(u, v, 160, 0.10, seed + 341, 0.20)
    alb = _mix(c1, c2, 0.5 + 0.35 * mid)
    alb = _mix(alb, c3, (1.0 - drift) * 0.22)
    dirt = gd._smoothstep(0.2, 0.9, prints + 0.25 * (0.5 + 0.5 * mid) * float(params.get("trample", 0.0)))
    alb = _mix(alb, c3 * 0.75, dirt * 0.6)
    alb = alb * (1.0 + 0.12 * sparkle[..., None])
    cav = np.clip(1.0 - drift, 0.0, 1.0) * 0.6
    ridge = gd._smoothstep(0.6, 0.95, drift)
    r_mod = -0.10 * sparkle + 0.08 * cav - 0.25 * dirt * float(params.get("trample", 0.0))
    return h.astype(np.float32), cav.astype(np.float32), ridge, alb, r_mod.astype(np.float32)


# ---------------------------------------------------------------------------
# Generic per-family micro-detail pass
# ---------------------------------------------------------------------------
# (grain, speck_density, speck_dark, speck_light, h_micro, r_micro, fiber_axis)
_PROFILES: dict[str, tuple[float, float, float, float, float, float, str | None]] = {
    "running_bond_masonry": (0.10, 0.10, 0.40, 0.18, 0.030, 0.05, None),
    "stacked_stone": (0.08, 0.06, 0.35, 0.20, 0.025, 0.05, None),
    "cobblestone_paver": (0.08, 0.06, 0.35, 0.20, 0.025, 0.05, None),
    "herringbone_chevron": (0.05, 0.03, 0.30, 0.0, 0.010, 0.03, None),  # parquet pores handled via is_wood
    "grid_tile": (0.04, 0.03, 0.30, 0.10, 0.008, 0.02, None),
    "mosaic_tile": (0.03, 0.02, 0.25, 0.0, 0.006, 0.02, None),
    "wood_planks": (0.04, 0.02, 0.35, 0.0, 0.010, 0.03, "grain"),
    "wood_slats": (0.04, 0.02, 0.35, 0.0, 0.010, 0.03, "grain"),
    "tree_bark": (0.10, 0.08, 0.35, 0.30, 0.030, 0.05, "grain"),
    "veined_marble": (0.025, 0.05, 0.0, 0.18, 0.004, 0.015, None),
    "speckled_aggregate": (0.05, 0.14, 0.35, 0.20, 0.006, 0.02, None),
    "concrete_plaster": (0.09, 0.12, 0.45, 0.15, 0.022, 0.05, None),
    "rock_cliff": (0.10, 0.07, 0.40, 0.25, 0.035, 0.05, None),
    "terrain_ground": (0.08, 0.10, 0.35, 0.25, 0.025, 0.04, None),
    "brushed_metal": (0.03, 0.0, 0.0, 0.0, 0.004, 0.04, "brush"),
    "specialty_metal": (0.05, 0.04, 0.30, 0.0, 0.010, 0.04, None),
    "metal_cladding": (0.04, 0.02, 0.25, 0.0, 0.006, 0.03, None),
    "perforated_mesh": (0.03, 0.0, 0.0, 0.0, 0.004, 0.02, None),
    "machined_grip": (0.03, 0.0, 0.0, 0.0, 0.004, 0.02, None),
    "woven_fabric": (0.07, 0.03, 0.20, 0.15, 0.012, 0.03, "fuzz"),
    "plush_knit": (0.06, 0.02, 0.15, 0.10, 0.012, 0.03, "pile"),
    "leather_hide": (0.04, 0.06, 0.30, 0.0, 0.008, 0.04, None),
    "natural_weave": (0.07, 0.04, 0.30, 0.15, 0.015, 0.03, "fuzz"),
    "glass_optics": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None),
    "rubber_polymer": (0.05, 0.08, 0.30, 0.25, 0.008, 0.03, None),
    "ceramic_glaze": (0.03, 0.0, 0.0, 0.0, 0.004, 0.015, None),
    "roofing_shingles": (0.08, 0.10, 0.35, 0.25, 0.020, 0.05, None),
    "tech_panel": (0.03, 0.0, 0.0, 0.0, 0.004, 0.03, "scratch"),
}


def _profile(family: str, params: dict[str, Any]):
    g, sd, sk, sl, hm, rm, fx = _PROFILES.get(family, (0.04, 0.0, 0.0, 0.0, 0.006, 0.02, None))
    mode = str(params.get("mode", params.get("style", "")))
    if family == "grid_tile" and float(params.get("rough", 0.3)) > 0.5:          # terracotta / cement / slate tiles
        g, sd, sk, sl, hm = 0.08, 0.08, 0.35, 0.15, 0.015
    if family == "rubber_polymer":
        if mode == "asphalt":
            g, sd, sk, sl, hm = 0.10, 0.30, 0.30, 0.60, 0.020
        elif mode in ("carbon_weave",):
            g, sd, sk, sl, hm = 0.015, 0.0, 0.0, 0.0, 0.002
        elif mode == "molded":
            g, sd, sk, sl, hm = 0.025, 0.0, 0.0, 0.0, 0.004
    if family == "roofing_shingles" and mode == "asphalt_shingle":
        sd, sk, sl = 0.35, 0.35, 0.45
    if family == "specialty_metal" and mode == "corten_patina":
        g, sd, sk, sl, hm = 0.10, 0.12, 0.35, 0.25, 0.020
    if family == "terrain_ground" and mode == "ice":
        g, sd, sk, sl, hm = 0.02, 0.06, 0.0, 0.35, 0.004
    if family == "terrain_ground" and mode == "sand_ripples":
        g, sd, sk, sl, hm = 0.07, 0.25, 0.30, 0.35, 0.010
    if family == "concrete_plaster" and mode == "trowel":
        grit = float(params.get("grit", 0.35))
        sd = 0.04 + 0.14 * grit
        hm = 0.008 + 0.030 * grit
    return g, sd, sk, sl, hm, rm, fx


def apply_detail_pass(family: str, params: dict[str, Any], H: int, W: int, seed: int,
                      h: np.ndarray, cav: np.ndarray, ridge: np.ndarray, alb: np.ndarray,
                      met: np.ndarray, r_mod: np.ndarray, axis: str = "u"):
    """Add periodic high-frequency grain, speckle, fibers and micro-relief (scaled per family)."""
    g, sd, sk, sl, hm, rm, fx = _profile(family, params)
    scale = float(params.get("detail", 1.0))
    if scale <= 0.0 or (g + sd + hm + rm) <= 0.0:
        return h, cav, ridge, alb, met, r_mod
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H
    face = np.clip(1.0 - cav, 0.0, 1.0)
    fine = fbm(H, W, seed + 401, (48, 96, 160))
    micro = fbm(H, W, seed + 402, (128, 192, 256))
    lum = (1.0 + scale * g * (0.65 * fine + 0.45 * micro))[..., None]
    alb = alb * lum

    if fx == "grain" or fx == "brush":
        # Directional streaks along the material axis (wood fibers / brush scratches)
        n_s = 160 if fx == "brush" else 96
        streak = fbm(H, W, seed + 403, (n_s,))
        if axis == "u":
            # stretch along u: average along x with a periodic box filter
            k = max(2, W // 40)
            acc = np.zeros_like(streak)
            for s in range(-k, k + 1):
                acc += np.roll(streak, s, axis=1)
            streak = acc / (2 * k + 1)
        else:
            k = max(2, H // 40)
            acc = np.zeros_like(streak)
            for s in range(-k, k + 1):
                acc += np.roll(streak, s, axis=0)
            streak = acc / (2 * k + 1)
        streak = streak / (streak.std() + 1e-6)
        amp = 0.06 if fx == "grain" else 0.035
        alb = alb * (1.0 + scale * amp * streak * face)[..., None]
        if fx == "grain" and family in ("wood_planks", "wood_slats", "herringbone_chevron"):
            # Open vessel pores: tiny dark flecks elongated along the grain
            pm, _ = speckle(u, v, 150, 0.35 * float(params.get("pores", 1.0)), seed + 408, 0.42)
            pm = np.clip(stretch(pm, 1 if axis == "u" else 0, max(2, W // 90)) * 0.35, 0.0, 1.0)
            alb = alb * (1.0 - scale * 0.28 * pm * face)[..., None]
            h = h - scale * 0.012 * pm * face
            r_mod = r_mod + scale * 0.06 * pm
        r_mod = r_mod + scale * (0.05 if fx == "brush" else 0.03) * streak
        h = h + scale * (0.010 if fx == "grain" else 0.004) * streak * face
    elif fx == "fuzz" or fx == "pile":
        fuzz = fbm(H, W, seed + 405, (160, 256))
        alb = alb * (1.0 + scale * (0.07 if fx == "fuzz" else 0.10) * fuzz * (0.6 + 0.4 * face))[..., None]
        h = h + scale * 0.008 * fuzz
    elif fx == "scratch":
        sc = ridged(warp(fbm(H, W, seed + 406, (24, 48)), 0.01 * fine, 0.01 * micro), 9.0)
        sc = sc * gd._smoothstep(0.2, 1.0, fbm(H, W, seed + 407, (3, 6)))
        alb = alb * (1.0 + scale * 0.18 * sc * face)[..., None]
        r_mod = r_mod - scale * 0.06 * sc * face

    if sd > 0.0:
        mode = str(params.get("mode", params.get("style", "")))
        cells = 44 if (family == "rubber_polymer" and mode == "asphalt") else (70 if family == "roofing_shingles" else 90)
        d_m, d_id = speckle(u, v, cells, sd * scale, seed + 410, 0.30)
        l_m, l_id = speckle((u + 0.21) % 1.0, (v + 0.47) % 1.0, cells, sd * 0.7 * scale, seed + 411, 0.26)
        if sk > 0.0:
            alb = alb * (1.0 - sk * d_m * (0.6 + 0.4 * d_id))[..., None]
        if sl > 0.0:
            light = np.clip(alb * (1.0 + 0.9 * sl) + 0.15 * sl, 0.0, 0.9)
            alb = _mix(alb, light, l_m * (0.5 + 0.5 * l_id))
        h = h + scale * 0.25 * hm * (l_m - 0.6 * d_m)
        r_mod = r_mod + scale * 0.04 * (d_m - l_m)

    if hm > 0.0:
        h = h + scale * hm * (0.55 * fine + 0.35 * micro) * (0.35 + 0.65 * face)
    if rm > 0.0:
        r_mod = r_mod + scale * rm * (0.6 * fine + 0.4 * micro) * 0.5
    return h.astype(np.float32), cav, ridge, alb.astype(np.float32), met, r_mod.astype(np.float32)
