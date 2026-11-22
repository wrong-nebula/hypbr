"""Parametric C1/C2 Seamless Procedural PBR Generator Families for the 500-Material hypbr Atlas.

Every family is mathematically periodic on [0, 1)^2 (zero wrap seams at any resolution),
produces rich multi-tone linear RGB albedo, calibrated roughness & metallic separation,
and physically grounded height/cavity/ridge relief.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import procedural as gd


def _rgb(c: tuple[float, float, float]) -> np.ndarray:
    return np.array(c, dtype=np.float32)[None, None, :]


def _periodic_uv(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return xx / float(W), yy / float(H)


def _cell_hash(ix: np.ndarray, iy: np.ndarray, nx: float, ny: float, seed: int) -> np.ndarray:
    """Deterministic periodic hash in [0, 1] per integer cell (ix mod nx, iy mod ny)."""
    cx = np.mod(ix, nx)
    cy = np.mod(iy, ny)
    v = np.sin(cx * 127.1 + cy * 311.7 + float(seed) * 74.7) * 43758.5453
    return (v - np.floor(v)).astype(np.float32)


def _cell_hash2(ix: np.ndarray, iy: np.ndarray, nx: float, ny: float, seed: int) -> np.ndarray:
    cx = np.mod(ix, nx)
    cy = np.mod(iy, ny)
    v = np.sin(cx * 269.5 + cy * 183.3 + float(seed) * 19.3) * 28461.9137
    return (v - np.floor(v)).astype(np.float32)


def _mix_rgb(c0: np.ndarray, c1: np.ndarray, t: np.ndarray) -> np.ndarray:
    tt = np.clip(t, 0.0, 1.0)[..., None]
    return c0 * (1.0 - tt) + c1 * tt


def _chroma_drift(m1: np.ndarray, m3: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.stack([
        0.020 * scale * m3,
        0.004 * scale * m1,
        -0.020 * scale * m3,
    ], axis=-1).astype(np.float32)


def build_parametric_material(
    family: str,
    params: dict[str, Any],
    H: int,
    W: int,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize a seamless procedural material from a parametric family.

    Returns:
        (h, slope_scale, cav, base_rough, ridge, alb_lin, met, r_mod)
    """
    u, v = _periodic_uv(H, W)
    m1 = gd._periodic_fbm(H, W, seed + 11, (4, 8, 16))
    m2 = gd._periodic_fbm(H, W, seed + 23, (16, 32, 64))
    m3 = gd._periodic_fbm(H, W, seed + 37, (2, 4, 8))
    cd = _chroma_drift(m1, m3, float(params.get("chroma", 0.7)))

    c1 = _rgb(params["c1"])
    c2 = _rgb(params.get("c2", params["c1"]))
    c3 = _rgb(params.get("c3", (0.28, 0.27, 0.25)))
    base_r = float(params.get("rough", 0.55))
    slope = float(params.get("slope", 10.0))
    met_val = float(params.get("metallic", 0.0))
    met = np.full((H, W), met_val, dtype=np.float32)
    r_mod = np.zeros((H, W), dtype=np.float32)

    # ------------------------------------------------------------------
    # 1. RUNNING BOND MASONRY (bricks, blocks, ashlar, firebrick)
    # ------------------------------------------------------------------
    if family == "running_bond_masonry":
        nx = float(params.get("nx", 4))
        ny = float(params.get("ny", 10))
        stagger = float(params.get("stagger", 0.5))
        mortar_w = float(params.get("mortar", 0.075))
        rough_face = float(params.get("face_rough", 0.07))
        wy = v * ny + 0.25
        iy = np.floor(wy)
        row_shift = (iy % 2.0) * stagger
        wx = u * nx + row_shift + 0.15
        ix = np.floor(wx)
        fx = np.abs((wx % 1.0) - 0.5) * 2.0
        fy = np.abs((wy % 1.0) - 0.5) * 2.0
        # Slight hand-molded edge irregularity
        wob = 0.018 * float(params.get("wobble", 1.0)) * m1
        mx = gd._smoothstep(1.0 - mortar_w * 0.75, 1.0 - mortar_w * 0.12, fx + wob)
        my = gd._smoothstep(1.0 - mortar_w * 1.45, 1.0 - mortar_w * 0.25, fy + wob)
        mortar = np.clip(np.maximum(mx, my), 0.0, 1.0)
        brick_m = 1.0 - mortar

        tid1 = _cell_hash(ix, iy, nx, ny, seed + 1)
        tid2 = _cell_hash2(ix, iy, nx, ny, seed + 2)
        camber = (1.0 - 0.28 * fx * fx) * (1.0 - 0.22 * fy * fy)
        h = (
            0.14
            + 0.66 * brick_m * (0.80 + 0.20 * camber)
            + rough_face * brick_m * (0.55 * m1 + 0.45 * m2)
            + 0.05 * brick_m * (tid1 - 0.5)
            + 0.03 * mortar * m2
        )
        cav = mortar
        ridge = gd._smoothstep(0.55, 0.92, brick_m * camber)

        brick_col = _mix_rgb(c1, c2, 0.20 + 0.65 * tid1 + 0.15 * m1)
        brick_col = brick_col * (0.90 + 0.14 * (tid2 - 0.5)[..., None] + 0.07 * m2[..., None] + cd)
        mortar_col = c3 * (0.86 + 0.16 * m2[..., None])
        alb = _mix_rgb(brick_col, mortar_col, mortar)
        r_mod = 0.14 * mortar + 0.05 * m2 - 0.04 * (tid1 - 0.5) * brick_m
        if float(params.get("glazed", 0.0)) > 0.5:
            r_mod = -0.10 * brick_m + 0.65 * mortar + 0.02 * m1

    # ------------------------------------------------------------------
    # 2. HERRINGBONE / CHEVRON TILE OR WOOD PARQUET
    # ------------------------------------------------------------------
    elif family == "herringbone_chevron":
        freq = float(params.get("freq", 4))
        is_wood = bool(params.get("is_wood", True))
        chevron = bool(params.get("chevron", False))
        grout_w = float(params.get("grout", 0.045))

        if chevron:
            us = u * freq
            col = np.floor(us)
            fu = us % 1.0
            sign = np.where((col % 2.0) < 0.5, 1.0, -1.0)
            vy = (v * freq * 2.0 + sign * fu * 1.0 + 0.25)
            row = np.floor(vy)
            fv = vy % 1.0
            du = np.abs(fu - 0.5) * 2.0
            dv = np.abs(fv - 0.5) * 2.0
            seam = np.maximum(
                gd._smoothstep(1.0 - grout_w, 1.0 - grout_w * 0.15, du),
                gd._smoothstep(1.0 - grout_w * 1.4, 1.0 - grout_w * 0.2, dv),
            )
            tid1 = _cell_hash(col, row, freq, freq * 2.0, seed + 5)
            tid2 = _cell_hash2(col, row, freq, freq * 2.0, seed + 6)
            along = fu
            across = fv
        else:
            # Classic 45-degree rotated periodic herringbone (exact integer diagonal lattice)
            a = (u + v) * freq + 0.125
            b = (u - v) * freq + 0.125
            ia = np.floor(a)
            ib = np.floor(b)
            fa = a % 1.0
            fb = b % 1.0
            parity = np.mod(ia + ib, 2.0)
            # Split into 2x1 planks alternating orientation
            sub_a = np.floor(fa * 2.0)
            sub_b = np.floor(fb * 2.0)
            along = np.where(parity < 0.5, fb, fa)
            across = np.where(parity < 0.5, (fa * 2.0) % 1.0, (fb * 2.0) % 1.0)
            id_x = np.where(parity < 0.5, ia * 2.0 + sub_a, ia)
            id_y = np.where(parity < 0.5, ib, ib * 2.0 + sub_b)
            du = np.abs(along - 0.5) * 2.0
            dv = np.abs(across - 0.5) * 2.0
            seam = np.maximum(
                gd._smoothstep(1.0 - grout_w, 1.0 - grout_w * 0.15, du),
                gd._smoothstep(1.0 - grout_w * 1.5, 1.0 - grout_w * 0.2, dv),
            )
            tid1 = _cell_hash(id_x, id_y, freq * 2.0, freq * 2.0, seed + 7)
            tid2 = _cell_hash2(id_x, id_y, freq * 2.0, freq * 2.0, seed + 8)

        if is_wood:
            grain = 0.5 + 0.5 * np.sin(2.0 * math.pi * (across * 7.0 + 0.8 * tid2 + 0.14 * m1))
            pores = 0.5 + 0.5 * np.sin(2.0 * math.pi * (across * 22.0 + 0.35 * m2))
            surf = 0.055 * grain + 0.025 * pores
        else:
            surf = 0.03 * m1 + 0.02 * m2

        h = 0.18 + 0.68 * (1.0 - seam) + surf * (1.0 - seam) + 0.025 * (tid1 - 0.5) * (1.0 - seam)
        cav = seam
        ridge = gd._smoothstep(0.55, 0.90, 1.0 - seam)
        face_col = _mix_rgb(c1, c2, 0.20 + 0.60 * tid1 + (0.20 * surf / 0.08 if is_wood else 0.15 * m1))
        face_col = face_col * (0.94 + 0.06 * m2[..., None] + cd)
        alb = _mix_rgb(face_col, c3 * (0.85 + 0.15 * m2[..., None]), seam)
        r_mod = 0.28 * seam + (0.04 * (1.0 - surf / 0.08) if is_wood else 0.03 * m1) - 0.03 * (tid2 - 0.5)

    # ------------------------------------------------------------------
    # 3. GRID TILE / ZELLIGE / LARGE SLAB / QUARRY TILE
    # ------------------------------------------------------------------
    elif family == "grid_tile":
        nx = float(params.get("nx", 4))
        ny = float(params.get("ny", 4))
        grout_w = float(params.get("grout", 0.05))
        bevel = float(params.get("bevel", 0.14))
        handmade = float(params.get("handmade", 0.0))
        stagger = float(params.get("stagger", 0.0))

        wy = v * ny + 0.25
        iy = np.floor(wy)
        wx = u * nx + (iy % 2.0) * stagger + 0.25
        ix = np.floor(wx)
        fx = np.abs((wx % 1.0) - 0.5) * 2.0
        fy = np.abs((wy % 1.0) - 0.5) * 2.0
        wob = handmade * 0.035 * m1
        mx = gd._smoothstep(1.0 - grout_w, 1.0 - grout_w * 0.12, fx + wob)
        my = gd._smoothstep(1.0 - grout_w, 1.0 - grout_w * 0.12, fy + wob)
        grout = np.clip(np.maximum(mx, my), 0.0, 1.0)
        tile_m = 1.0 - grout

        bx = 1.0 - gd._smoothstep(1.0 - grout_w - bevel, 1.0 - grout_w * 0.5, fx + wob)
        by = 1.0 - gd._smoothstep(1.0 - grout_w - bevel, 1.0 - grout_w * 0.5, fy + wob)
        dome = np.minimum(bx, by)
        tid1 = _cell_hash(ix, iy, nx, ny, seed + 13)
        tid2 = _cell_hash2(ix, iy, nx, ny, seed + 14)

        wave = handmade * 0.06 * m1 + 0.018 * m2
        h = 0.15 + 0.70 * tile_m * (0.76 + 0.24 * dome) + wave * tile_m + 0.03 * handmade * (tid1 - 0.5) * tile_m
        cav = grout
        ridge = gd._smoothstep(0.45, 0.90, tile_m * (1.0 - dome * 0.5))

        tile_col = _mix_rgb(c1, c2, 0.15 + 0.70 * tid1 + 0.15 * m1)
        tile_col = tile_col * (0.94 + 0.08 * (tid2 - 0.5)[..., None] + 0.04 * m2[..., None] + cd)
        # Subtle kiln rim glaze darkening for ceramics
        if handmade > 0.25:
            rim = tile_m * (1.0 - dome)
            tile_col = tile_col * (1.0 - 0.22 * rim[..., None])
        grout_col = c3 * (0.86 + 0.16 * m2[..., None])
        alb = _mix_rgb(tile_col, grout_col, grout)
        r_mod = float(params.get("grout_rmod", 0.55)) * grout + 0.03 * m1 * tile_m - 0.03 * (tid2 - 0.5) * tile_m
        if met_val > 0.2:
            met = np.clip(met_val * (1.0 - grout), 0.0, 0.99).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. HEX / DIAMOND / PENNY / ARABESQUE MOSAIC TILE
    # ------------------------------------------------------------------
    elif family == "mosaic_tile":
        mode = str(params.get("mode", "hex"))
        freq = float(params.get("freq", 8))
        grout_w = float(params.get("grout", 0.10))

        if mode == "diamond":
            a = (u + v) * freq + 0.25
            b = (u - v) * freq + 0.25
            ia, ib = np.floor(a), np.floor(b)
            da = np.abs((a % 1.0) - 0.5) * 2.0
            db = np.abs((b % 1.0) - 0.5) * 2.0
            edge = np.maximum(da, db)
            grout = gd._smoothstep(1.0 - grout_w * 1.4, 1.0 - grout_w * 0.2, edge)
            dome = np.clip(1.0 - edge, 0.0, 1.0)
            tid1 = _cell_hash(ia, ib, freq, freq, seed + 17)
            tid2 = _cell_hash2(ia, ib, freq, freq, seed + 18)
        elif mode == "penny":
            # Hex-packed circular penny rounds (strictly periodic: even rows)
            nx = freq
            ny = freq
            vy = v * ny + 0.25
            iy = np.floor(vy)
            ux = u * nx + (iy % 2.0) * 0.5 + 0.25
            ix = np.floor(ux)
            fx = (ux % 1.0) - 0.5
            fy = (vy % 1.0) - 0.5
            r = np.sqrt(fx * fx + fy * fy) * 2.0
            grout = gd._smoothstep(0.84 - grout_w * 0.6, 0.94, r)
            dome = np.sqrt(np.clip(1.0 - (r / 0.92) ** 2, 0.0, 1.0))
            tid1 = _cell_hash(ix, iy, nx, ny, seed + 19)
            tid2 = _cell_hash2(ix, iy, nx, ny, seed + 20)
        else:
            # Periodic Worley-aligned crisp hexagonal tiles (even ny for exact periodicity)
            nx = max(2.0, round(freq))
            ny = max(2.0, round(freq / 2.0) * 2.0)
            uu = u * nx + 0.25
            vv = v * ny + 0.25
            best_d = np.full((H, W), 9.0, dtype=np.float32)
            second_d = np.full((H, W), 9.0, dtype=np.float32)
            best_id = np.zeros((H, W), dtype=np.float32)
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    # Exact honeycomb centers on periodic (nx, ny) lattice
                    cy_i = np.floor(vv) + oy
                    shift = (np.mod(cy_i, 2.0)) * 0.5
                    cx_i = np.floor(uu - shift) + ox
                    cx = cx_i + shift + 0.5
                    cy = cy_i + 0.5
                    dx = uu - cx
                    dy = (vv - cy) * 1.12
                    d = np.sqrt(dx * dx + dy * dy)
                    cid = _cell_hash(cx_i, cy_i, nx, ny, seed + 21)
                    is_better = d < best_d
                    second_d = np.where(is_better, best_d, np.minimum(second_d, d))
                    best_id = np.where(is_better, cid, best_id)
                    best_d = np.where(is_better, d, best_d)
            edge_dist = second_d - best_d
            grout = 1.0 - gd._smoothstep(grout_w * 0.25, grout_w * 1.15, edge_dist)
            dome = gd._smoothstep(0.0, 0.35, edge_dist)
            tid1 = best_id
            tid2 = np.mod(best_id * 3.71 + 0.23, 1.0)

        tile_m = 1.0 - grout
        h = 0.14 + 0.72 * tile_m * (0.74 + 0.26 * dome) + 0.02 * m2 * tile_m
        cav = grout
        ridge = gd._smoothstep(0.45, 0.90, tile_m * dome)
        tile_col = _mix_rgb(c1, c2, 0.15 + 0.70 * tid1 + 0.15 * m1) * (0.94 + 0.08 * (tid2 - 0.5)[..., None] + cd)
        alb = _mix_rgb(tile_col, c3 * (0.88 + 0.12 * m2[..., None]), grout)
        r_mod = 0.52 * grout + 0.025 * m1 * tile_m
        if met_val > 0.2:
            met = np.clip(met_val * tile_m, 0.0, 0.99).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. STACKED STONE / LEDGER / DRY-STACK WALL
    # ------------------------------------------------------------------
    elif family == "stacked_stone":
        nx = float(params.get("nx", 3))
        ny = float(params.get("ny", 14))
        wy = v * ny + 0.18 * m1 + 0.25
        iy = np.floor(wy)
        row_off = _cell_hash(iy, iy * 0.0, ny, 1.0, seed + 25) * 0.85
        wx = u * nx + row_off + 0.10 * m1
        ix = np.floor(wx)
        fx = np.abs((wx % 1.0) - 0.5) * 2.0
        fy = np.abs((wy % 1.0) - 0.5) * 2.0
        seam = np.maximum(
            gd._smoothstep(0.90, 0.99, fx),
            gd._smoothstep(0.84, 0.98, fy),
        )
        tid1 = _cell_hash(ix, iy, nx, ny, seed + 26)
        tid2 = _cell_hash2(ix, iy, nx, ny, seed + 27)
        cleft = 0.18 * m1 + 0.12 * m2 + 0.24 * (tid1 - 0.5)
        h = np.clip(0.12 + 0.68 * (1.0 - seam) * (0.78 + 0.22 * (1.0 - fy * fy)) + cleft * (1.0 - seam), 0.02, 0.98)
        cav = np.clip(seam + 0.22 * gd._smoothstep(0.55, 0.95, -m2), 0.0, 1.0)
        ridge = gd._smoothstep(0.58, 0.92, h)
        stone = _mix_rgb(c1, c2, 0.15 + 0.68 * tid1 + 0.17 * m1) * (0.88 + 0.16 * (tid2 - 0.5)[..., None] + 0.08 * m2[..., None] + cd)
        alb = _mix_rgb(stone, c3 * (0.75 + 0.25 * m2[..., None]), seam)
        r_mod = 0.16 * seam + 0.07 * m2 - 0.05 * ridge

    # ------------------------------------------------------------------
    # 6. COBBLESTONE / FLAGSTONE / RIVER PEBBLE PAVERS
    # ------------------------------------------------------------------
    elif family == "cobblestone_paver":
        cells = int(params.get("cells", 10))
        gap_w = float(params.get("gap", 0.12))
        roundness = float(params.get("roundness", 0.65))
        f1, f2, cid = gd._periodic_worley_id(u, v, cells, seed + 31)
        edge = f2 - f1
        gap = 1.0 - gd._smoothstep(gap_w * 0.25, gap_w * 1.25, edge)
        stone_m = 1.0 - gap
        dome = np.power(np.clip(gd._smoothstep(0.0, 0.42, edge), 0.0, 1.0), 0.45 + 0.45 * roundness)
        h = 0.10 + 0.72 * stone_m * (0.65 + 0.35 * dome) + 0.06 * stone_m * (m1 + 0.6 * m2) + 0.08 * (cid - 0.5) * stone_m
        cav = gap
        ridge = gd._smoothstep(0.50, 0.92, stone_m * dome)
        stone = _mix_rgb(c1, c2, 0.15 + 0.70 * cid + 0.15 * m1) * (0.90 + 0.10 * m2[..., None] + cd)
        alb = _mix_rgb(stone, c3 * (0.82 + 0.18 * m2[..., None]), gap)
        r_mod = 0.24 * gap - 0.10 * ridge + 0.06 * m1

    # ------------------------------------------------------------------
    # 7. WOOD PLANKS & GRAIN (hardwoods, shiplap, teak deck, barnwood, bamboo)
    # ------------------------------------------------------------------
    elif family == "wood_planks":
        planks = float(params.get("planks", 6))
        joints = float(params.get("joints", 2))
        groove_w = float(params.get("groove", 0.045))
        ring_freq = float(params.get("ring_freq", 10.0))
        caulk = bool(params.get("caulk", False))

        # Horizontal planks along u, stacked along v (or vertical if rotated)
        vert = bool(params.get("vertical", False))
        uu, vv = (v, u) if vert else (u, v)
        wy = vv * planks + 0.15
        iy = np.floor(wy)
        fy = wy % 1.0
        j_off = _cell_hash(iy, iy * 0.0, planks, 1.0, seed + 41)
        wx = uu * joints + j_off
        ix = np.floor(wx)
        fx = wx % 1.0

        dy = np.abs(fy - 0.5) * 2.0
        dx = np.abs(fx - 0.5) * 2.0
        seam_y = gd._smoothstep(1.0 - groove_w * 1.4, 1.0 - groove_w * 0.15, dy)
        seam_x = gd._smoothstep(1.0 - groove_w * 0.45, 1.0 - groove_w * 0.06, dx) if joints >= 1.0 else 0.0
        seam = np.clip(np.maximum(seam_y, seam_x), 0.0, 1.0)
        plank_m = 1.0 - seam

        tid1 = _cell_hash(ix, iy, max(joints, 1.0), planks, seed + 42)
        tid2 = _cell_hash2(ix, iy, max(joints, 1.0), planks, seed + 43)

        # Periodic elongated cathedral growth rings + fine vessel pores
        warp = 0.55 * m3 + 0.25 * m1
        ring_phase = vv * planks * ring_freq * 0.32 + 1.35 * warp + 2.4 * tid2
        rings = 0.5 + 0.5 * np.sin(2.0 * math.pi * ring_phase)
        rings = np.power(rings, 1.35)
        fibers = 0.5 + 0.5 * np.sin(2.0 * math.pi * (vv * 96.0 + 0.4 * m2))
        knots = np.exp(-((np.sin(math.pi * (uu * 2.0 + tid1)) ** 2 * 14.0) + (np.sin(math.pi * (vv * planks + 0.5)) ** 2 * 10.0)))
        knot_str = float(params.get("knots", 0.25))

        h = (
            0.16
            + 0.68 * plank_m
            + 0.065 * plank_m * rings
            + 0.032 * plank_m * fibers
            + 0.04 * plank_m * knot_str * knots
            + 0.03 * (tid1 - 0.5) * plank_m
        )
        cav = np.clip(seam + 0.20 * (1.0 - fibers) * plank_m, 0.0, 1.0)
        ridge = gd._smoothstep(0.55, 0.90, plank_m * (0.7 + 0.3 * rings))

        wood_col = _mix_rgb(c1, c2, np.clip(0.18 + 0.48 * rings + 0.24 * tid1 + 0.10 * fibers, 0.0, 1.0))
        wood_col = wood_col * (0.92 + 0.10 * (tid2 - 0.5)[..., None] + 0.05 * m1[..., None] + cd)
        wood_col = wood_col * (1.0 - 0.32 * knot_str * knots[..., None])
        seam_col = c3 * (0.85 + 0.15 * m2[..., None])
        alb = _mix_rgb(wood_col, seam_col, seam)
        r_mod = (0.34 if not caulk else 0.18) * seam + 0.07 * (1.0 - rings) * plank_m - 0.04 * (tid2 - 0.5) * plank_m

    # ------------------------------------------------------------------
    # 8. WOOD SLATS / LOUVERS / BATTEN / FENCE
    # ------------------------------------------------------------------
    elif family == "wood_slats":
        slats = float(params.get("slats", 12))
        gap_ratio = float(params.get("gap_ratio", 0.26))
        vert = bool(params.get("vertical", True))
        uu, vv = (u, v) if vert else (v, u)
        ws = uu * slats + 0.25
        idx = np.floor(ws)
        fu = np.abs((ws % 1.0) - 0.5) * 2.0
        slat_m = 1.0 - gd._smoothstep(1.0 - gap_ratio - 0.12, 1.0 - gap_ratio + 0.04, fu)
        tid = _cell_hash(idx, idx * 0.0, slats, 1.0, seed + 47)
        grain = 0.5 + 0.5 * np.sin(2.0 * math.pi * (uu * slats * 5.0 + 0.35 * m1 + tid))
        h = 0.08 + 0.80 * slat_m + 0.045 * slat_m * grain + 0.02 * (1.0 - slat_m) * m2
        cav = 1.0 - slat_m
        ridge = gd._smoothstep(0.65, 0.96, slat_m)
        wood_col = _mix_rgb(c1, c2, 0.20 + 0.45 * grain + 0.35 * tid) * (0.94 + 0.06 * m2[..., None] + cd)
        alb = _mix_rgb(c3 * (0.85 + 0.15 * m2[..., None]), wood_col, slat_m)
        r_mod = 0.35 * (1.0 - slat_m) + 0.05 * (1.0 - grain) * slat_m

    # ------------------------------------------------------------------
    # 9. TREE BARK / ORGANIC CORTEX / DRIFTWOOD
    # ------------------------------------------------------------------
    elif family == "tree_bark":
        style = str(params.get("style", "furrowed"))
        freq = float(params.get("freq", 10))
        if style == "birch":
            # Horizontal papery peeling lenticels
            bands = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 18.0 + 0.45 * m1))
            lenticels = gd._smoothstep(0.55, 0.82, m2 * 0.7 + 0.4 * np.sin(2.0 * math.pi * (u * 4.0 + 1.5 * m3))) * gd._smoothstep(0.45, 0.78, bands)
            peel = gd._smoothstep(0.35, 0.75, m3 + 0.35 * m1)
            h = 0.45 + 0.25 * peel - 0.22 * lenticels + 0.08 * m2
            cav = np.clip(lenticels * 0.85 + 0.25 * (1.0 - peel), 0.0, 1.0)
            ridge = gd._smoothstep(0.55, 0.85, h)
            alb = _mix_rgb(c1, c2, 0.25 + 0.50 * peel + 0.25 * m1)
            alb = _mix_rgb(alb, c3, lenticels)
        else:
            # Deep vertical fissured bark ridges (oak, pine, redwood, cork)
            warp_u = u * freq + 0.35 * m3 + 0.18 * m1
            fissure = np.abs(np.sin(math.pi * warp_u))
            f1, f2, cid = gd._periodic_worley_id(u, v, int(max(6, freq)), seed + 51)
            plates = gd._smoothstep(0.05, 0.38, f2 - f1)
            bark_h = 0.58 * np.power(fissure, 0.65) + 0.42 * plates
            h = np.clip(0.10 + 0.72 * bark_h + 0.10 * m2 + 0.06 * (cid - 0.5), 0.02, 0.98)
            cav = np.clip(1.0 - gd._smoothstep(0.22, 0.65, bark_h), 0.0, 1.0)
            ridge = gd._smoothstep(0.55, 0.90, bark_h)
            alb = _mix_rgb(c1, c2, 0.20 + 0.55 * bark_h + 0.25 * cid) * (0.90 + 0.10 * m2[..., None] + cd)
            alb = _mix_rgb(alb, c3, cav * 0.72)
        r_mod = 0.14 * cav - 0.06 * ridge + 0.05 * m2

    # ------------------------------------------------------------------
    # 10. VEINED MARBLE / ONYX / SOAPSTONE / JADE
    # ------------------------------------------------------------------
    elif family == "veined_marble":
        vein_freq = float(params.get("vein_freq", 3.0))
        vein_sharp = float(params.get("vein_sharp", 4.5))
        angle = float(params.get("angle", 0.55))
        w1 = gd._periodic_fbm(H, W, seed + 61, (2, 4, 8, 16))
        w2 = gd._periodic_fbm(H, W, seed + 62, (4, 8, 16, 32))
        coord = (u * round(vein_freq) + v * round(max(1.0, vein_freq * angle)))
        v_raw = np.abs(np.sin(math.pi * (coord + 0.48 * w1 + 0.18 * w2)))
        vein_core = np.exp(-np.power(v_raw * vein_sharp, 1.45))
        vein_cloud = np.exp(-np.power(v_raw * (vein_sharp * 0.38), 1.25))
        v_sec = np.exp(-np.power(np.abs(np.sin(math.pi * (u * 2.0 - v * 3.0 + 0.55 * w2))) * (vein_sharp * 1.1), 1.5))
        vein_tot = np.clip(vein_core + 0.55 * v_sec, 0.0, 1.0)

        h = np.clip(0.55 + 0.16 * w1 - 0.22 * vein_tot + 0.05 * m2, 0.05, 0.95)
        cav = np.clip(vein_tot * 0.85, 0.0, 1.0)
        ridge = np.clip(vein_cloud * 0.65, 0.0, 1.0)

        base_stone = _mix_rgb(c1, c2, np.clip(0.45 + 0.35 * m3 + 0.20 * w1, 0.0, 1.0)) * (0.97 + 0.03 * m2[..., None])
        alb = _mix_rgb(base_stone, c2 * 0.82, vein_cloud * 0.45)
        alb = _mix_rgb(alb, c3, vein_tot * 0.85)
        r_mod = 0.03 * vein_cloud + 0.12 * vein_tot + 0.015 * m1

    # ------------------------------------------------------------------
    # 11. SPECKLED GRANITE / TERRAZZO / EPOXY FLAKE
    # ------------------------------------------------------------------
    elif family == "speckled_aggregate":
        cells_a = int(params.get("cells_a", 18))
        cells_b = int(params.get("cells_b", 38))
        chip_fill = float(params.get("chip_fill", 0.52))
        f1a, f2a, ida = gd._periodic_worley_id(u, v, cells_a, seed + 71)
        f1b, f2b, idb = gd._periodic_worley_id(u, v, cells_b, seed + 72)
        chip_a = gd._smoothstep(0.06, 0.20, f2a - f1a) * gd._smoothstep(1.0 - chip_fill, 1.0 - chip_fill + 0.08, ida)
        chip_b = gd._smoothstep(0.05, 0.18, f2b - f1b) * gd._smoothstep(1.0 - chip_fill * 1.1, 1.0 - chip_fill * 1.1 + 0.08, idb)
        chips = np.clip(np.maximum(chip_a, chip_b * 0.85), 0.0, 1.0)

        h = np.clip(0.46 + 0.18 * chips + 0.09 * m1 + 0.07 * m2, 0.05, 0.95)
        cav = np.clip((1.0 - chips) * 0.35 + 0.25 * gd._smoothstep(0.4, 0.9, -m2), 0.0, 1.0)
        ridge = chips

        matrix = c1 * (0.93 + 0.07 * m2[..., None] + cd)
        c_chip1 = c2 * (0.85 + 0.30 * ida[..., None])
        c_chip2 = c3 * (0.85 + 0.30 * idb[..., None])
        alb = _mix_rgb(matrix, c_chip1, chip_a * 0.92)
        alb = _mix_rgb(alb, c_chip2, chip_b * (1.0 - chip_a * 0.85) * 0.90)
        r_mod = -0.08 * chips + 0.05 * (1.0 - chips) + 0.03 * m2

    # ------------------------------------------------------------------
    # 12. CONCRETE / PLASTER / STUCCO / RAMMED EARTH / DRYWALL
    # ------------------------------------------------------------------
    elif family == "concrete_plaster":
        style = str(params.get("style", "trowel"))
        if style == "board_formed":
            boards = float(params.get("boards", 6))
            wb = v * boards + 0.25
            ib = np.floor(wb)
            fb = np.abs((wb % 1.0) - 0.5) * 2.0
            seam = gd._smoothstep(0.92, 0.99, fb)
            tid = _cell_hash(ib, ib * 0.0, boards, 1.0, seed + 81)
            wood_imprint = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 64.0 + 0.35 * m1 + tid))
            # Tie-rod cone holes (2x2 periodic)
            tu = ((u * 2.0 + 0.5) % 1.0) - 0.5
            tv = ((v * 2.0 + 0.5) % 1.0) - 0.5
            tie = 1.0 - gd._smoothstep(0.035, 0.065, np.sqrt(tu * tu + tv * tv))
            h = 0.52 + 0.14 * m1 + 0.06 * wood_imprint + 0.05 * (tid - 0.5) - 0.22 * seam - 0.38 * tie
            cav = np.clip(seam * 0.75 + tie + 0.20 * gd._smoothstep(0.45, 0.95, -m2), 0.0, 1.0)
            ridge = gd._smoothstep(0.55, 0.88, h)
            alb = _mix_rgb(c1, c2, 0.35 + 0.35 * tid + 0.30 * m1) * (0.94 + 0.06 * m2[..., None] + cd)
            alb = _mix_rgb(alb, c3, np.clip(seam * 0.55 + tie * 0.85, 0.0, 1.0))
        elif style == "rammed_earth":
            strata = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 7.0 + 0.25 * m3 + 0.08 * m1))
            strata2 = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 19.0 - 0.18 * m1))
            h = 0.45 + 0.18 * strata + 0.10 * m1 + 0.12 * m2
            cav = np.clip(0.45 * (1.0 - strata2) + 0.30 * gd._smoothstep(0.3, 0.9, -m2), 0.0, 1.0)
            ridge = gd._smoothstep(0.52, 0.88, h)
            alb = _mix_rgb(c1, c2, strata)
            alb = _mix_rgb(alb, c3, 0.32 * (1.0 - strata2)) * (0.92 + 0.08 * m2[..., None] + cd)
        else:
            # Troweled lime plaster / stucco / poured concrete / microcement
            grit = float(params.get("grit", 0.35))
            swirl = gd._periodic_fbm(H, W, seed + 83, (2, 4, 8))
            trowel = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 2.0 + v * 1.0 + 0.65 * swirl))
            f1, _, _ = gd._periodic_worley_id(u, v, 44, seed + 84)
            pores = gd._smoothstep(0.18, 0.04, f1) * float(params.get("pores", 0.3))
            h = 0.50 + 0.18 * swirl + 0.10 * trowel + grit * 0.16 * m2 - 0.28 * pores
            cav = np.clip(pores + 0.32 * gd._smoothstep(0.25, 0.85, -m1), 0.0, 1.0)
            ridge = gd._smoothstep(0.52, 0.88, trowel * 0.5 + 0.5 * np.clip(0.5 + 0.5 * swirl, 0.0, 1.0))
            alb = _mix_rgb(c1, c2, np.clip(0.35 + 0.40 * swirl + 0.25 * trowel, 0.0, 1.0)) * (0.94 + 0.06 * m2[..., None] + cd)
            alb = _mix_rgb(alb, c3, pores * 0.72)
        r_mod = 0.16 * cav - 0.06 * ridge + 0.05 * m1

    # ------------------------------------------------------------------
    # 13. ROCK / CLIFF / SEDIMENTARY / VOLCANIC / SLATE
    # ------------------------------------------------------------------
    elif family == "rock_cliff":
        layers = float(params.get("layers", 6.0))
        fracture = int(params.get("fracture", 8))
        f1, f2, cid = gd._periodic_worley_id(u, v, fracture, seed + 91)
        cracks = 1.0 - gd._smoothstep(0.02, 0.11, f2 - f1)
        strata = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * layers + 0.35 * m3 + 0.15 * m1))
        shelf = np.power(strata, 0.75)
        h = np.clip(0.18 + 0.42 * shelf + 0.22 * (1.0 - f1) + 0.14 * m1 + 0.08 * m2 - 0.26 * cracks, 0.02, 0.98)
        cav = np.clip(cracks * 0.90 + 0.30 * (1.0 - shelf), 0.0, 1.0)
        ridge = gd._smoothstep(0.56, 0.90, h)
        alb = _mix_rgb(c1, c2, np.clip(0.25 + 0.45 * shelf + 0.30 * cid, 0.0, 1.0)) * (0.90 + 0.10 * m2[..., None] + cd)
        alb = _mix_rgb(alb, c3, cav * 0.65)
        r_mod = 0.14 * cav - 0.06 * ridge + 0.06 * m2

    # ------------------------------------------------------------------
    # 14. TERRAIN / SAND / MUD / GRAVEL / SNOW / ICE / GRASS / SOIL
    # ------------------------------------------------------------------
    elif family == "terrain_ground":
        mode = str(params.get("mode", "sand_ripples"))
        if mode == "sand_ripples":
            freq = float(params.get("freq", 10.0))
            phase = v * freq + 0.38 * m3 + 0.12 * m1
            # Asymmetric wind ripple profile
            rip = 0.5 + 0.42 * np.sin(2.0 * math.pi * phase) + 0.10 * np.sin(4.0 * math.pi * phase + 0.6)
            h = np.clip(0.25 + 0.52 * rip + 0.08 * m2, 0.04, 0.96)
            cav = np.clip(1.0 - rip, 0.0, 1.0)
            ridge = gd._smoothstep(0.58, 0.92, rip)
            alb = _mix_rgb(c1, c2, 0.25 + 0.55 * rip + 0.20 * m1) * (0.95 + 0.05 * m2[..., None] + cd)
            r_mod = 0.08 * cav - 0.04 * ridge + 0.04 * m2
        elif mode == "cracked_mud":
            cells = int(params.get("cells", 7))
            f1, f2, cid = gd._periodic_worley_id(u, v, cells, seed + 101)
            crack = 1.0 - gd._smoothstep(0.025, 0.105, f2 - f1 + 0.025 * m1)
            curl = gd._smoothstep(0.08, 0.32, f2 - f1)
            h = np.clip(0.10 + 0.72 * (1.0 - crack) * (0.88 + 0.12 * (1.0 - curl)) + 0.07 * m2, 0.02, 0.98)
            cav = crack
            ridge = gd._smoothstep(0.55, 0.90, (1.0 - crack) * (1.0 - curl * 0.5))
            alb = _mix_rgb(c1, c2, 0.25 + 0.50 * cid + 0.25 * m1)
            alb = _mix_rgb(alb, c3, crack * 0.88)
            r_mod = 0.12 * crack + 0.05 * m2
        elif mode == "gravel":
            f1a, f2a, ida = gd._periodic_worley_id(u, v, 24, seed + 102)
            f1b, f2b, idb = gd._periodic_worley_id(u, v, 42, seed + 103)
            peb_a = np.clip(1.0 - f1a * 1.45, 0.0, 1.0)
            peb_b = np.clip(1.0 - f1b * 1.45, 0.0, 1.0)
            peb = np.maximum(peb_a, peb_b * 0.82)
            h = np.clip(0.12 + 0.74 * peb + 0.06 * m2, 0.02, 0.98)
            cav = np.clip(1.0 - gd._smoothstep(0.18, 0.62, peb), 0.0, 1.0)
            ridge = gd._smoothstep(0.55, 0.92, peb)
            stone_id = np.where(peb_a >= peb_b * 0.82, ida, idb)
            alb = _mix_rgb(c1, c2, stone_id)
            alb = _mix_rgb(alb, c3, cav * 0.75) * (0.94 + 0.06 * m2[..., None])
            r_mod = 0.14 * cav - 0.08 * ridge + 0.05 * m1
        elif mode == "ice":
            f1, f2, cid = gd._periodic_worley_id(u, v, 9, seed + 104)
            fract = 1.0 - gd._smoothstep(0.015, 0.075, f2 - f1 + 0.02 * m1)
            frost = gd._smoothstep(0.25, 0.85, m1 + 0.4 * m2)
            h = np.clip(0.45 + 0.18 * m3 - 0.25 * fract + 0.08 * m2, 0.05, 0.95)
            cav = fract
            ridge = frost
            alb = _mix_rgb(c1, c2, 0.30 + 0.45 * frost + 0.25 * cid)
            alb = _mix_rgb(alb, c3, fract * 0.65)
            r_mod = 0.28 * frost + 0.22 * fract - 0.05 * (1.0 - frost)
        else:
            # Organic soil / moss / grass / snow / leaf litter / mulch
            f1, _, cid = gd._periodic_worley_id(u, v, int(params.get("cells", 32)), seed + 105)
            tuft = np.clip(1.0 - f1 * 1.3, 0.0, 1.0)
            h = np.clip(0.22 + 0.36 * np.clip(0.5 + 0.5 * m3, 0.0, 1.0) + 0.24 * tuft + 0.14 * m2, 0.02, 0.98)
            cav = np.clip(1.0 - gd._smoothstep(0.22, 0.68, tuft * 0.6 + 0.4 * np.clip(0.5 + 0.5 * m1, 0.0, 1.0)), 0.0, 1.0)
            ridge = gd._smoothstep(0.52, 0.88, tuft)
            alb = _mix_rgb(c1, c2, np.clip(0.20 + 0.50 * tuft + 0.30 * cid, 0.0, 1.0)) * (0.92 + 0.08 * m1[..., None] + cd)
            alb = _mix_rgb(alb, c3, cav * 0.60)
            r_mod = 0.12 * cav - 0.05 * ridge + 0.05 * m2

    # ------------------------------------------------------------------
    # 15. BRUSHED / SATIN / ANODIZED / RADIAL METAL
    # ------------------------------------------------------------------
    elif family == "brushed_metal":
        radial = bool(params.get("radial", False))
        seam = bool(params.get("panel_seam", False))
        if radial:
            du = u - 0.5
            dv = v - 0.5
            rad = np.sqrt(du * du + dv * dv)
            streak = 0.5 + 0.5 * np.sin(2.0 * math.pi * (rad * 48.0 + 0.18 * m2))
            streak2 = 0.5 + 0.5 * np.sin(2.0 * math.pi * (rad * 96.0 + 0.10 * m1))
            brush = 0.65 * streak + 0.35 * streak2
        else:
            # Fine strictly linear directional brush lines (periodic along u)
            s1 = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 72.0 + 0.22 * m2))
            s2 = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 144.0 + 0.15 * m1))
            brush = 0.60 * s1 + 0.40 * s2

        p_seam = 0.0
        if seam:
            sx = gd._smoothstep(0.955, 0.992, np.abs((u * 2.0) % 1.0 - 0.5) * 2.0)
            sy = gd._smoothstep(0.955, 0.992, np.abs((v * 2.0) % 1.0 - 0.5) * 2.0)
            p_seam = np.maximum(sx, sy)

        h = np.clip(0.50 + 0.14 * brush + 0.08 * m3 - 0.36 * p_seam, 0.04, 0.96)
        cav = np.asarray(p_seam, dtype=np.float32) if seam else np.clip(0.25 * (1.0 - brush), 0.0, 1.0)
        ridge = gd._smoothstep(0.55, 0.90, brush)
        alb = _mix_rgb(c1, c2, 0.25 + 0.50 * brush + 0.25 * np.clip(0.5 + 0.5 * m3, 0.0, 1.0)) * (0.97 + 0.03 * m1[..., None])
        if seam:
            alb = _mix_rgb(alb, c3, p_seam * 0.85)
            met = np.clip(met_val * (1.0 - 0.45 * p_seam), 0.2, 0.99).astype(np.float32)
        r_mod = 0.045 * (1.0 - brush) + 0.025 * m3 + (0.35 * p_seam if seam else 0.0)

    # ------------------------------------------------------------------
    # 16. HAMMERED / CAST / GALVANIZED / CORTEN / PATINA / DAMASCUS METAL
    # ------------------------------------------------------------------
    elif family == "specialty_metal":
        mode = str(params.get("mode", "hammered"))
        if mode == "hammered":
            # Order-independent C-infinity LogSumExp ball-peen dimples
            cells = int(params.get("cells", 10))
            rng = np.random.default_rng(seed + 111)
            jx = rng.uniform(0.15, 0.85, (cells, cells)).astype(np.float32)
            jy = rng.uniform(0.15, 0.85, (cells, cells)).astype(np.float32)
            rad = rng.uniform(0.45, 0.72, (cells, cells)).astype(np.float32)
            ux, vy = u * cells, v * cells
            ix0 = np.floor(ux).astype(np.int32)
            iy0 = np.floor(vy).astype(np.int32)
            beta = 2.7
            acc = np.zeros((H, W), dtype=np.float32)
            for oy in (-1, 0, 1):
                cy = (iy0 + oy) % cells
                for ox in (-1, 0, 1):
                    cx = (ix0 + ox) % cells
                    dx = ux - (ix0 + ox + jx[cy, cx])
                    dy = vy - (iy0 + oy + jy[cy, cx])
                    d2 = (dx * dx + dy * dy) / (rad[cy, cx] ** 2)
                    acc += np.exp(-beta * np.minimum(d2, 9.0))
            dimple = -np.log(np.maximum(acc, 1e-6)) / beta
            dimple = (dimple - dimple.min()) / (dimple.max() - dimple.min() + 1e-6)
            h = np.clip(0.12 + 0.78 * dimple + 0.04 * m2, 0.04, 0.96)
            cav = np.clip(1.0 - dimple, 0.0, 1.0)
            ridge = gd._smoothstep(0.58, 0.92, dimple)
            alb = _mix_rgb(c1, c2, 0.25 + 0.55 * dimple + 0.20 * m1)
            r_mod = 0.10 * cav - 0.06 * ridge + 0.03 * m1
        elif mode == "galvanized":
            f1, f2, cid = gd._periodic_worley_id(u, v, int(params.get("cells", 26)), seed + 112)
            h = np.clip(0.48 + 0.14 * (cid - 0.5) + 0.06 * m2, 0.1, 0.9)
            cav = np.clip(1.0 - gd._smoothstep(0.02, 0.14, f2 - f1), 0.0, 1.0)
            ridge = cid
            alb = _mix_rgb(c1, c2, cid) * (0.94 + 0.06 * m1[..., None])
            r_mod = 0.16 * (0.5 - cid) + 0.03 * m2
        elif mode == "corten_patina":
            ox = np.clip(0.45 + 0.45 * m1 + 0.30 * m2 + 0.20 * m3, 0.0, 1.0)
            crust = gd._smoothstep(0.32, 0.78, ox)
            h = np.clip(0.28 + 0.36 * crust + 0.18 * m2 + 0.10 * m1, 0.04, 0.96)
            cav = np.clip(1.0 - crust, 0.0, 1.0)
            ridge = crust
            alb = _mix_rgb(c1, c2, ox)
            alb = _mix_rgb(alb, c3, gd._smoothstep(0.65, 0.95, ox) * 0.75)
            met = np.clip(met_val * (1.0 - 0.85 * crust), 0.05, 0.96).astype(np.float32)
            r_mod = 0.48 * crust + 0.06 * m2
        elif mode == "damascus":
            bands = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 5.0 + v * 3.0 + 0.85 * m3 + 0.28 * m1))
            etch = gd._smoothstep(0.35, 0.65, bands)
            h = np.clip(0.35 + 0.32 * etch + 0.05 * m2, 0.05, 0.95)
            cav = 1.0 - etch
            ridge = etch
            alb = _mix_rgb(c1, c2, etch) * (0.95 + 0.05 * m1[..., None])
            met = np.clip(0.86 + 0.12 * etch, 0.80, 0.99).astype(np.float32)
            r_mod = -0.06 * etch + 0.22 * (1.0 - etch)
        else:
            # Cast iron / sand-cast pebbled metal
            f1, _, cid = gd._periodic_worley_id(u, v, 56, seed + 113)
            peb = np.clip(1.0 - f1 * 1.4, 0.0, 1.0)
            h = np.clip(0.30 + 0.34 * peb + 0.16 * m2 + 0.08 * m1, 0.05, 0.95)
            cav = 1.0 - peb
            ridge = peb
            alb = _mix_rgb(c1, c2, 0.30 + 0.50 * peb + 0.20 * m1)
            r_mod = 0.12 * cav - 0.06 * ridge + 0.04 * m2

    # ------------------------------------------------------------------
    # 17. ARCHITECTURAL METAL CLADDING (standing seam, corrugated, fluted, riveted)
    # ------------------------------------------------------------------
    elif family == "metal_cladding":
        mode = str(params.get("mode", "standing_seam"))
        freq = float(params.get("freq", 8.0))
        if mode == "corrugated":
            wave = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * freq + 0.25))
            h = 0.12 + 0.76 * wave + 0.03 * m2
            cav = 1.0 - wave
            ridge = gd._smoothstep(0.60, 0.95, wave)
        elif mode == "fluted":
            fu = np.abs(((u * freq + 0.25) % 1.0) - 0.5) * 2.0
            flute = np.sqrt(np.clip(1.0 - (fu * 0.92) ** 2, 0.0, 1.0))
            h = 0.14 + 0.74 * flute + 0.025 * m2
            cav = 1.0 - flute
            ridge = gd._smoothstep(0.65, 0.96, flute)
        elif mode == "riveted":
            # Sheet-metal panel lap seams + domed rivets along borders
            sx = gd._smoothstep(0.92, 0.99, np.abs((u * 2.0 + 0.25) % 1.0 - 0.5) * 2.0)
            sy = gd._smoothstep(0.92, 0.99, np.abs((v * 2.0 + 0.25) % 1.0 - 0.5) * 2.0)
            seam = np.maximum(sx, sy)
            ru = ((u * 12.0) % 1.0) - 0.5
            rv = ((v * 2.0 + 0.25) % 1.0) - 0.5
            r_dist = np.sqrt(ru * ru + (np.abs(rv) - 0.42) ** 2 * 36.0)
            rivet = np.sqrt(np.clip(1.0 - (r_dist / 0.28) ** 2, 0.0, 1.0))
            h = np.clip(0.48 - 0.34 * seam + 0.34 * rivet + 0.05 * m1 + 0.03 * m2, 0.04, 0.96)
            cav = seam
            ridge = rivet
        else:
            # Architectural standing-seam roof / facade panels
            fu = np.abs(((u * freq + 0.5) % 1.0) - 0.5) * 2.0
            rib = gd._smoothstep(0.84, 0.94, fu)
            oilcan = 0.045 * np.sin(2.0 * math.pi * v * 2.0 + 1.2 * m3) * (1.0 - rib)
            h = np.clip(0.24 + 0.64 * rib + oilcan + 0.02 * m2, 0.04, 0.96)
            cav = gd._smoothstep(0.76, 0.86, fu) * (1.0 - rib)
            ridge = rib

        alb = _mix_rgb(c1, c2, 0.30 + 0.45 * ridge + 0.25 * np.clip(0.5 + 0.5 * m1, 0.0, 1.0)) * (0.96 + 0.04 * m2[..., None])
        alb = _mix_rgb(alb, c3, cav * 0.42)
        r_mod = -0.05 * ridge + 0.14 * cav + 0.03 * m1

    # ------------------------------------------------------------------
    # 18. PERFORATED MESH / GRILLE / LOUVER / WIRE SCREEN
    # ------------------------------------------------------------------
    elif family == "perforated_mesh":
        mode = str(params.get("mode", "round_perf"))
        freq = float(params.get("freq", 12.0))
        if mode == "wire_screen":
            wx = 0.5 + 0.5 * np.cos(2.0 * math.pi * u * freq)
            wy = 0.5 + 0.5 * np.cos(2.0 * math.pi * v * freq)
            wire_x = gd._smoothstep(0.52, 0.92, wx)
            wire_y = gd._smoothstep(0.52, 0.92, wy)
            over = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * freq * 0.5 + v * freq * 0.5))
            metal_m = np.maximum(wire_x, wire_y)
            h = 0.06 + 0.72 * metal_m + 0.14 * wire_x * wire_y * over
            hole = 1.0 - metal_m
        elif mode == "slotted":
            nx, ny = freq * 0.5, freq
            vy = v * ny + 0.25
            iy = np.floor(vy)
            ux = u * nx + (iy % 2.0) * 0.5 + 0.25
            fx = np.abs((ux % 1.0) - 0.5) * 2.0
            fy = np.abs((vy % 1.0) - 0.5) * 2.0
            slot = (1.0 - gd._smoothstep(0.58, 0.74, fx)) * (1.0 - gd._smoothstep(0.32, 0.52, fy))
            hole = slot
            h = 0.80 * (1.0 - hole) + 0.08 + 0.03 * m2 * (1.0 - hole)
        else:
            # Staggered circular perforations
            vy = v * freq + 0.25
            iy = np.floor(vy)
            ux = u * freq + (iy % 2.0) * 0.5 + 0.25
            fx = (ux % 1.0) - 0.5
            fy = (vy % 1.0) - 0.5
            r = np.sqrt(fx * fx + fy * fy) * 2.0
            hole = 1.0 - gd._smoothstep(0.46, 0.62, r)
            h = 0.80 * (1.0 - hole) + 0.08 + 0.025 * m2 * (1.0 - hole)

        metal_m = 1.0 - hole
        cav = hole
        ridge = gd._smoothstep(0.45, 0.85, metal_m)
        alb = _mix_rgb(c1, c2, 0.35 + 0.40 * ridge + 0.25 * np.clip(0.5 + 0.5 * m1, 0.0, 1.0))
        alb = _mix_rgb(alb, c3, hole * 0.94)
        met = np.clip(met_val * (1.0 - hole), 0.0, 0.99).astype(np.float32)
        r_mod = 0.62 * hole - 0.05 * ridge + 0.03 * m1 * metal_m

    # ------------------------------------------------------------------
    # 19. KNURLED / MACHINED / DIAMOND TREAD / CNC GRIP
    # ------------------------------------------------------------------
    elif family == "machined_grip":
        mode = str(params.get("mode", "knurl"))
        freq = float(params.get("freq", 14.0))
        if mode == "flutes":
            flute = 0.5 + 0.5 * np.cos(2.0 * math.pi * (u * freq + 0.25))
            h = 0.14 + 0.74 * np.power(flute, 0.75) + 0.02 * m2
            cav = 1.0 - flute
            ridge = gd._smoothstep(0.65, 0.95, flute)
        elif mode == "tread":
            nx, ny = freq * 0.5, freq * 0.5
            ux, vy = u * nx + 0.25, v * ny + 0.25
            ix, iy = np.floor(ux), np.floor(vy)
            fx, fy = (ux % 1.0) - 0.5, (vy % 1.0) - 0.5
            rot = np.where((np.mod(ix + iy, 2.0)) < 0.5, 1.0, -1.0)
            rx = (fx + rot * fy) * 0.7071
            ry = (-rot * fx + fy) * 0.7071
            bar = (1.0 - gd._smoothstep(0.24, 0.34, np.abs(rx))) * (1.0 - gd._smoothstep(0.06, 0.13, np.abs(ry)))
            h = 0.24 + 0.64 * bar + 0.03 * m2
            cav = 1.0 - bar
            ridge = bar
        else:
            # Diamond knurl pyramids (C1 smooth)
            a = 0.5 + 0.5 * np.cos(2.0 * math.pi * ((u + v) * freq + 0.25))
            b = 0.5 + 0.5 * np.cos(2.0 * math.pi * ((u - v) * freq + 0.25))
            pyr = np.minimum(a, b)
            h = 0.12 + 0.76 * pyr + 0.02 * m2
            cav = 1.0 - pyr
            ridge = gd._smoothstep(0.60, 0.94, pyr)

        alb = _mix_rgb(c1, c2, 0.25 + 0.55 * ridge + 0.20 * np.clip(0.5 + 0.5 * m1, 0.0, 1.0))
        alb = _mix_rgb(alb, c3, cav * 0.45)
        r_mod = -0.07 * ridge + 0.22 * cav + 0.03 * m1

    # ------------------------------------------------------------------
    # 20. WOVEN FABRIC (linen, canvas, denim, twill, ripstop, tweed, plaid, boucle)
    # ------------------------------------------------------------------
    elif family == "woven_fabric":
        mode = str(params.get("mode", "plain"))
        freq = float(params.get("freq", 28.0))
        if mode == "twill":
            diag = 0.5 + 0.5 * np.sin(2.0 * math.pi * ((u * 2.0 + v) * freq * 0.5))
            warp = 0.5 + 0.5 * np.sin(2.0 * math.pi * u * freq)
            weft = 0.5 + 0.5 * np.sin(2.0 * math.pi * v * freq)
            weave = 0.55 * diag + 0.25 * warp + 0.20 * weft
            yarn_t = diag
        elif mode == "ripstop":
            warp = 0.5 + 0.5 * np.sin(2.0 * math.pi * u * freq)
            weft = 0.5 + 0.5 * np.sin(2.0 * math.pi * v * freq)
            grid_x = gd._smoothstep(0.84, 0.98, np.abs(np.cos(math.pi * u * 8.0)))
            grid_y = gd._smoothstep(0.84, 0.98, np.abs(np.cos(math.pi * v * 8.0)))
            rip = np.maximum(grid_x, grid_y)
            weave = 0.42 * (warp + weft) * 0.5 + 0.58 * rip
            yarn_t = rip
        elif mode == "plaid":
            px = 0.5 + 0.5 * np.sin(2.0 * math.pi * u * 4.0)
            py = 0.5 + 0.5 * np.sin(2.0 * math.pi * v * 4.0)
            fine_x = gd._smoothstep(0.82, 0.96, np.abs(np.cos(math.pi * u * 8.0)))
            fine_y = gd._smoothstep(0.82, 0.96, np.abs(np.cos(math.pi * v * 8.0)))
            warp = 0.5 + 0.5 * np.sin(2.0 * math.pi * u * freq)
            weft = 0.5 + 0.5 * np.sin(2.0 * math.pi * v * freq)
            weave = 0.5 * warp + 0.5 * weft
            yarn_t = np.clip(0.5 * (px + py), 0.0, 1.0)
            stripe = np.maximum(fine_x, fine_y)
        elif mode == "boucle":
            f1, _, cid = gd._periodic_worley_id(u, v, int(max(18, freq)), seed + 121)
            f1b, _, cidb = gd._periodic_worley_id((u + 0.013) % 1.0, (v + 0.019) % 1.0, int(max(28, freq * 1.4)), seed + 122)
            loops = 0.6 * np.clip(1.0 - f1 * 1.35, 0.0, 1.0) + 0.4 * np.clip(1.0 - f1b * 1.35, 0.0, 1.0)
            weave = loops
            yarn_t = 0.5 * cid + 0.5 * cidb
        else:
            # Classic over-under plain linen / canvas / burlap / silk
            wx = 0.5 + 0.5 * np.cos(2.0 * math.pi * u * freq)
            wy = 0.5 + 0.5 * np.cos(2.0 * math.pi * v * freq)
            parity = 0.5 + 0.5 * np.sin(2.0 * math.pi * u * freq * 0.5) * np.sin(2.0 * math.pi * v * freq * 0.5)
            weave = wx * parity + wy * (1.0 - parity)
            yarn_t = parity

        h = np.clip(0.18 + 0.64 * weave + 0.09 * m2 + 0.05 * m1, 0.04, 0.96)
        cav = np.clip(1.0 - weave, 0.0, 1.0)
        ridge = gd._smoothstep(0.55, 0.90, weave)
        alb = _mix_rgb(c1, c2, np.clip(0.18 + 0.64 * yarn_t + 0.18 * m1, 0.0, 1.0)) * (0.92 + 0.08 * m2[..., None] + cd)
        if mode == "plaid":
            alb = _mix_rgb(alb, c3, stripe * 0.72)
        else:
            alb = _mix_rgb(alb, c3, cav * 0.25)
        r_mod = 0.06 * cav - 0.04 * ridge + 0.03 * m2

    # ------------------------------------------------------------------
    # 21. KNIT / VELVET / CORDUROY / QUILTED / CARPET / FELT
    # ------------------------------------------------------------------
    elif family == "plush_knit":
        mode = str(params.get("mode", "velvet"))
        freq = float(params.get("freq", 14.0))
        if mode == "corduroy":
            wale = 0.5 + 0.5 * np.cos(2.0 * math.pi * (u * freq + 0.25))
            nap = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * freq * 3.0 + 0.25 * m2))
            h = 0.16 + 0.70 * np.power(wale, 0.68) + 0.06 * nap * wale + 0.04 * m2
            cav = 1.0 - wale
            ridge = gd._smoothstep(0.55, 0.92, wale)
        elif mode == "quilted":
            qfreq = max(2.0, round(freq * 0.35))
            a = np.abs(((u + v) * qfreq + 0.25) % 1.0 - 0.5) * 2.0
            b = np.abs(((u - v) * qfreq + 0.25) % 1.0 - 0.5) * 2.0
            puff = np.sqrt(np.clip(1.0 - a * a, 0.0, 1.0)) * np.sqrt(np.clip(1.0 - b * b, 0.0, 1.0))
            seam = np.maximum(gd._smoothstep(0.88, 0.99, a), gd._smoothstep(0.88, 0.99, b))
            h = 0.14 + 0.74 * puff + 0.03 * m2
            cav = seam
            ridge = gd._smoothstep(0.55, 0.90, puff)
        elif mode == "knit":
            # V-stitch interlocking knit columns (integer frequencies for exact wrap)
            ufreq = max(4.0, round(freq))
            vfreq = max(4.0, round(freq * 1.2))
            col = u * ufreq
            v_off = np.abs((col % 1.0) - 0.5) * 2.0
            stitch = 0.5 + 0.5 * np.cos(2.0 * math.pi * (v * vfreq + v_off * 0.55))
            rib = 0.5 + 0.5 * np.cos(2.0 * math.pi * col)
            h = 0.16 + 0.44 * stitch + 0.28 * rib + 0.06 * m2
            cav = np.clip(1.0 - 0.5 * (stitch + rib), 0.0, 1.0)
            ridge = gd._smoothstep(0.55, 0.90, 0.55 * stitch + 0.45 * rib)
        elif mode == "carpet":
            f1, _, cid = gd._periodic_worley_id(u, v, int(max(24, freq * 2)), seed + 131)
            tuft = np.clip(1.0 - f1 * 1.35, 0.0, 1.0)
            h = 0.18 + 0.62 * tuft + 0.12 * m2 + 0.06 * m1
            cav = 1.0 - tuft
            ridge = gd._smoothstep(0.50, 0.88, tuft)
        else:
            # Crushed / plush velvet / suede / Alcantara micro-nap
            nap = 0.5 + 0.5 * m3
            micro = 0.5 + 0.5 * m2
            h = np.clip(0.38 + 0.24 * nap + 0.18 * micro + 0.10 * m1, 0.05, 0.95)
            cav = np.clip(1.0 - nap, 0.0, 1.0) * 0.55
            ridge = gd._smoothstep(0.45, 0.85, nap)

        alb = _mix_rgb(c1, c2, 0.22 + 0.56 * ridge + 0.22 * np.clip(0.5 + 0.5 * m1, 0.0, 1.0)) * (0.93 + 0.07 * m2[..., None] + cd)
        if mode == "quilted":
            alb = _mix_rgb(alb, c3, cav * 0.65)
        r_mod = 0.08 * cav - 0.05 * ridge + 0.04 * m1

    # ------------------------------------------------------------------
    # 22. LEATHER / HIDE / EMBOSSED SCALE / SUEDE
    # ------------------------------------------------------------------
    elif family == "leather_hide":
        mode = str(params.get("mode", "pebble"))
        cells = int(params.get("cells", 34))
        if mode == "croc":
            f1, f2, cid = gd._periodic_worley_id(u, (v * 2.0) % 1.0, int(max(8, cells // 3)), seed + 141)
            edge = f2 - f1
            groove = 1.0 - gd._smoothstep(0.03, 0.12, edge)
            scale = np.power(gd._smoothstep(0.02, 0.35, edge), 0.55)
            h = 0.12 + 0.74 * (1.0 - groove) * scale + 0.04 * m2
            cav = groove
            ridge = gd._smoothstep(0.55, 0.90, scale * (1.0 - groove))
        elif mode == "braided":
            a = 0.5 + 0.5 * np.cos(2.0 * math.pi * ((u + v) * 8.0 + 0.25))
            b = 0.5 + 0.5 * np.cos(2.0 * math.pi * ((u - v) * 8.0 + 0.25))
            p = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u + v) * 4.0) * np.sin(2.0 * math.pi * (u - v) * 4.0)
            weave = a * p + b * (1.0 - p)
            h = 0.14 + 0.72 * weave + 0.04 * m2
            cav = 1.0 - weave
            ridge = gd._smoothstep(0.55, 0.90, weave)
        else:
            # Full-grain pebble / nappa / distressed leather
            f1, f2, cid = gd._periodic_worley_id(u, v, cells, seed + 142)
            crease = 1.0 - gd._smoothstep(0.03, 0.15, f2 - f1)
            peb = np.sqrt(np.clip(1.0 - (f1 * 1.2) ** 2, 0.0, 1.0))
            wrinkle = 0.12 * m3 + 0.08 * m1
            h = np.clip(0.24 + 0.52 * peb * (1.0 - 0.5 * crease) + wrinkle + 0.04 * m2, 0.04, 0.96)
            cav = crease
            ridge = gd._smoothstep(0.50, 0.88, peb)

        alb = _mix_rgb(c1, c2, 0.22 + 0.54 * ridge + 0.24 * np.clip(0.5 + 0.5 * m3, 0.0, 1.0)) * (0.94 + 0.06 * m2[..., None] + cd)
        alb = _mix_rgb(alb, c3, cav * 0.55)
        r_mod = 0.18 * cav - 0.08 * ridge + 0.04 * m3

    # ------------------------------------------------------------------
    # 23. NATURAL WEAVE / BASKETRY / WICKER / THATCH / PAPER
    # ------------------------------------------------------------------
    elif family == "natural_weave":
        mode = str(params.get("mode", "wicker"))
        freq = float(params.get("freq", 10.0))
        if mode == "thatch":
            stalks = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 36.0 + 0.35 * m1))
            tiers = (v * 6.0 + 0.25) % 1.0
            step = gd._smoothstep(0.0, 0.14, tiers) * (1.0 - gd._smoothstep(0.84, 0.98, tiers))
            h = 0.16 + 0.50 * step + 0.22 * stalks + 0.08 * m2
            cav = np.clip(1.0 - step * 0.7 - stalks * 0.3, 0.0, 1.0)
            ridge = gd._smoothstep(0.55, 0.90, stalks * step)
        elif mode == "paper":
            fib = 0.5 * m1 + 0.5 * m2
            h = np.clip(0.45 + 0.18 * m3 + 0.20 * fib, 0.08, 0.92)
            cav = np.clip(0.5 - 0.5 * fib, 0.0, 1.0) * 0.45
            ridge = gd._smoothstep(0.45, 0.85, 0.5 + 0.5 * fib)
        else:
            # Basketry / wicker / seagrass / bamboo strip weave
            wx = 0.5 + 0.5 * np.cos(2.0 * math.pi * u * freq)
            wy = 0.5 + 0.5 * np.cos(2.0 * math.pi * v * freq)
            p = 0.5 + 0.5 * np.sin(2.0 * math.pi * u * freq * 0.5) * np.sin(2.0 * math.pi * v * freq * 0.5)
            weave = wx * p + wy * (1.0 - p)
            reed = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * freq * 4.0 * p + v * freq * 4.0 * (1.0 - p)))
            h = 0.14 + 0.68 * weave + 0.08 * reed + 0.04 * m2
            cav = 1.0 - weave
            ridge = gd._smoothstep(0.55, 0.90, weave)

        alb = _mix_rgb(c1, c2, 0.20 + 0.55 * ridge + 0.25 * np.clip(0.5 + 0.5 * m1, 0.0, 1.0)) * (0.93 + 0.07 * m2[..., None] + cd)
        alb = _mix_rgb(alb, c3, cav * 0.52)
        r_mod = 0.12 * cav - 0.06 * ridge + 0.04 * m2

    # ------------------------------------------------------------------
    # 24. GLASS & OPTICS (fluted, frosted, cathedral, pebbled, glass block, leaded)
    # ------------------------------------------------------------------
    elif family == "glass_optics":
        mode = str(params.get("mode", "fluted"))
        freq = float(params.get("freq", 14.0))
        if mode == "glass_block":
            nx, ny = float(params.get("nx", 4)), float(params.get("ny", 4))
            fx = np.abs(((u * nx + 0.25) % 1.0) - 0.5) * 2.0
            fy = np.abs(((v * ny + 0.25) % 1.0) - 0.5) * 2.0
            grout = np.maximum(gd._smoothstep(0.90, 0.98, fx), gd._smoothstep(0.90, 0.98, fy))
            wave = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * nx * 2.0 + 0.35 * m1)) * np.sin(2.0 * math.pi * (v * ny * 2.0 + 0.35 * m1))
            h = 0.16 + 0.64 * (1.0 - grout) * (0.78 + 0.22 * wave)
            cav = grout
            ridge = (1.0 - grout) * wave
        elif mode == "pebbled":
            f1, _, _ = gd._periodic_worley_id(u, v, int(freq), seed + 151)
            peb = np.sqrt(np.clip(1.0 - (f1 * 1.15) ** 2, 0.0, 1.0))
            h = 0.25 + 0.58 * peb + 0.04 * m1
            cav = 1.0 - peb
            ridge = peb
        elif mode == "frosted":
            h = np.clip(0.50 + 0.12 * m1 + 0.14 * m2, 0.10, 0.90)
            cav = np.clip(0.5 - 0.5 * m2, 0.0, 1.0) * 0.3
            ridge = np.clip(0.5 + 0.5 * m1, 0.0, 1.0)
        elif mode == "wired":
            wx = gd._smoothstep(0.90, 0.98, np.abs(np.cos(math.pi * u * freq)))
            wy = gd._smoothstep(0.90, 0.98, np.abs(np.cos(math.pi * v * freq)))
            wire = np.maximum(wx, wy)
            h = 0.50 + 0.16 * m1 + 0.18 * wire
            cav = wire
            ridge = 1.0 - wire
        else:
            # Fluted / reeded privacy glass
            fu = np.abs(((u * freq + 0.25) % 1.0) - 0.5) * 2.0
            rib = np.sqrt(np.clip(1.0 - (fu * 0.94) ** 2, 0.0, 1.0))
            h = 0.14 + 0.74 * rib + 0.015 * m1
            cav = 1.0 - rib
            ridge = gd._smoothstep(0.55, 0.95, rib)

        alb = _mix_rgb(c1, c2, 0.30 + 0.45 * ridge + 0.25 * np.clip(0.5 + 0.5 * m1, 0.0, 1.0))
        if mode in ("glass_block", "wired"):
            alb = _mix_rgb(alb, c3, cav * 0.85)
            r_mod = 0.58 * cav + 0.02 * m1
        else:
            r_mod = 0.04 * cav + 0.02 * m2

    # ------------------------------------------------------------------
    # 25. RUBBER / POLYMER / COMPOSITE / FOAM / TIRE TREAD / ASPHALT
    # ------------------------------------------------------------------
    elif family == "rubber_polymer":
        mode = str(params.get("mode", "tire_tread"))
        freq = float(params.get("freq", 8.0))
        if mode == "tire_tread":
            chev = np.abs(((u * 2.0) % 1.0) - 0.5) * 2.0
            lug = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * freq + chev * 1.0))
            groove_c = gd._smoothstep(0.06, 0.12, np.abs(((u * 4.0 + 0.5) % 1.0) - 0.5))
            tread = gd._smoothstep(0.32, 0.58, lug) * groove_c
            h = 0.12 + 0.74 * tread + 0.04 * m2
            cav = 1.0 - tread
            ridge = tread
        elif mode == "carbon_weave":
            # Forged or woven carbon / kevlar
            if params.get("forged", False):
                _, _, cid1 = gd._periodic_worley_id(u, v, 14, seed + 161)
                _, _, cid2 = gd._periodic_worley_id(u, v, 24, seed + 162)
                flake = 0.55 * cid1 + 0.45 * cid2
                h = 0.42 + 0.22 * flake + 0.06 * m2
                cav = 1.0 - flake
                ridge = flake
            else:
                diag = 0.5 + 0.5 * np.sin(2.0 * math.pi * ((u * 2.0 + v * 2.0) * freq * 0.5))
                tow_u = 0.5 + 0.5 * np.cos(2.0 * math.pi * u * freq)
                tow_v = 0.5 + 0.5 * np.cos(2.0 * math.pi * v * freq)
                weave = tow_u * diag + tow_v * (1.0 - diag)
                h = 0.20 + 0.64 * weave + 0.03 * m2
                cav = 1.0 - weave
                ridge = diag
        elif mode == "asphalt":
            f1a, _, ida = gd._periodic_worley_id(u, v, 56, seed + 163)
            f1b, _, idb = gd._periodic_worley_id(u, v, 96, seed + 164)
            agg = 0.58 * np.clip(1.0 - f1a * 1.45, 0.0, 1.0) + 0.42 * np.clip(1.0 - f1b * 1.45, 0.0, 1.0)
            h = np.clip(0.20 + 0.48 * agg + 0.16 * m1 + 0.12 * m2, 0.04, 0.96)
            cav = np.clip(1.0 - agg, 0.0, 1.0)
            ridge = gd._smoothstep(0.55, 0.90, agg)
        elif mode == "foam":
            f1, _, _ = gd._periodic_worley_id(u, v, 48, seed + 165)
            cell = np.clip(1.0 - f1 * 1.3, 0.0, 1.0)
            h = np.clip(0.28 + 0.46 * cell + 0.12 * m2, 0.04, 0.96)
            cav = 1.0 - cell
            ridge = cell
        else:
            # Molded ABS / matte polymer / silicone stipple
            f1, _, _ = gd._periodic_worley_id(u, v, 64, seed + 166)
            stip = np.clip(1.0 - f1 * 1.2, 0.0, 1.0)
            h = np.clip(0.45 + 0.16 * stip + 0.08 * m1 + 0.06 * m2, 0.08, 0.92)
            cav = (1.0 - stip) * 0.4
            ridge = stip

        alb = _mix_rgb(c1, c2, 0.25 + 0.50 * ridge + 0.25 * np.clip(0.5 + 0.5 * m1, 0.0, 1.0)) * (0.94 + 0.06 * m2[..., None] + cd)
        alb = _mix_rgb(alb, c3, cav * 0.40)
        r_mod = 0.18 * cav - 0.08 * ridge + 0.04 * m2

    # ------------------------------------------------------------------
    # 26. CERAMIC / STONEWARE / ENAMEL / CRACKLE GLAZE
    # ------------------------------------------------------------------
    elif family == "ceramic_glaze":
        mode = str(params.get("mode", "stoneware"))
        if mode == "crackle":
            f1, f2, cid = gd._periodic_worley_id(u, v, int(params.get("cells", 14)), seed + 171)
            crack = 1.0 - gd._smoothstep(0.015, 0.065, f2 - f1)
            h = np.clip(0.54 - 0.28 * crack + 0.10 * m1 + 0.04 * m2, 0.05, 0.95)
            cav = crack
            ridge = 1.0 - crack
            alb = _mix_rgb(c1, c2, 0.30 + 0.40 * cid + 0.30 * np.clip(0.5 + 0.5 * m1, 0.0, 1.0))
            alb = _mix_rgb(alb, c3, crack * 0.85)
            r_mod = 0.36 * crack + 0.02 * m1
        else:
            # Speckled stoneware / raw terracotta / porcelain / raku
            speck = gd._smoothstep(0.68, 0.92, m2 * 0.6 + 0.5 * m1)
            wheel = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 28.0 + 0.18 * m1))
            h = np.clip(0.42 + 0.14 * wheel + 0.12 * m1 + 0.10 * m2, 0.06, 0.94)
            cav = np.clip(0.45 * (1.0 - wheel) + 0.35 * speck, 0.0, 1.0)
            ridge = gd._smoothstep(0.45, 0.88, wheel)
            alb = _mix_rgb(c1, c2, 0.30 + 0.40 * np.clip(0.5 + 0.5 * m1, 0.0, 1.0) + 0.30 * wheel)
            alb = _mix_rgb(alb, c3, speck * float(params.get("speck_str", 0.55)))
            r_mod = 0.10 * speck + 0.04 * m1

    # ------------------------------------------------------------------
    # 27. ROOFING SHINGLES / CLAY BARREL TILES / SLATE ROOF
    # ------------------------------------------------------------------
    elif family == "roofing_shingles":
        mode = str(params.get("mode", "asphalt_shingle"))
        nx = float(params.get("nx", 5))
        ny = float(params.get("ny", 8))
        if mode == "barrel_clay":
            # Mediterranean Spanish barrel terracotta tiles
            col = u * nx + 0.25
            ix = np.floor(col)
            fu = np.abs((col % 1.0) - 0.5) * 2.0
            arch = np.sqrt(np.clip(1.0 - (fu * 0.94) ** 2, 0.0, 1.0))
            wy = v * ny + 0.25
            iy = np.floor(wy)
            fv = wy % 1.0
            step = 1.0 - gd._smoothstep(0.86, 0.98, fv)
            tid = _cell_hash(ix, iy, nx, ny, seed + 181)
            h = 0.12 + 0.56 * arch * (0.45 + 0.55 * (1.0 - fv * 0.55)) * step + 0.06 * m2
            cav = np.clip((1.0 - arch) * 0.65 + (1.0 - step) * 0.85, 0.0, 1.0)
            ridge = gd._smoothstep(0.55, 0.92, arch * step)
            alb = _mix_rgb(c1, c2, 0.18 + 0.62 * tid + 0.20 * m1) * (0.92 + 0.08 * m2[..., None] + cd)
            alb = _mix_rgb(alb, c3, cav * 0.55)
        else:
            # Overlapping rectangular / scalloped shingles (slate, cedar shake, asphalt)
            wy = v * ny + 0.25
            iy = np.floor(wy)
            fv = wy % 1.0
            wx = u * nx + (iy % 2.0) * 0.5 + 0.25
            ix = np.floor(wx)
            fx = np.abs((wx % 1.0) - 0.5) * 2.0
            if mode == "scalloped":
                r = np.sqrt(((wx % 1.0) - 0.5) ** 2 + np.maximum(fv - 0.45, 0.0) ** 2)
                edge = gd._smoothstep(0.46, 0.54, r)
            else:
                edge = np.maximum(gd._smoothstep(0.90, 0.98, fx), gd._smoothstep(0.88, 0.98, fv))
            tid = _cell_hash(ix, iy, nx, ny, seed + 182)
            slope_v = 0.65 + 0.35 * fv
            grain = 0.09 * m2 + 0.06 * m1
            h = np.clip(0.14 + 0.68 * (1.0 - edge) * slope_v + grain * (1.0 - edge) + 0.04 * (tid - 0.5), 0.04, 0.96)
            cav = edge
            ridge = gd._smoothstep(0.55, 0.90, (1.0 - edge) * slope_v)
            alb = _mix_rgb(c1, c2, 0.18 + 0.62 * tid + 0.20 * m1) * (0.90 + 0.10 * m2[..., None] + cd)
            alb = _mix_rgb(alb, c3, edge * 0.72)
        r_mod = 0.16 * cav - 0.04 * ridge + 0.05 * m2

    # ------------------------------------------------------------------
    # 28. ELECTRONICS / TECH PANEL / PCB / SOLAR / SCI-FI HULL
    # ------------------------------------------------------------------
    elif family == "tech_panel":
        mode = str(params.get("mode", "scifi_hull"))
        if mode == "pcb":
            # Parallel traces + micro vias + copper pads
            tr_u = gd._smoothstep(0.68, 0.88, np.abs(np.cos(math.pi * u * 18.0)))
            tr_v = gd._smoothstep(0.74, 0.92, np.abs(np.cos(math.pi * v * 10.0)))
            trace = np.maximum(tr_u * (1.0 - gd._smoothstep(0.45, 0.55, np.abs(v - 0.5) * 2.0)), tr_v * 0.7)
            pu = ((u * 6.0) % 1.0) - 0.5
            pv = ((v * 6.0) % 1.0) - 0.5
            r = np.sqrt(pu * pu + pv * pv)
            pad = (1.0 - gd._smoothstep(0.14, 0.19, r)) * gd._smoothstep(0.05, 0.08, r)
            hole = 1.0 - gd._smoothstep(0.05, 0.08, r)
            h = np.clip(0.42 + 0.24 * trace + 0.34 * pad - 0.36 * hole + 0.02 * m2, 0.04, 0.96)
            cav = hole
            ridge = np.maximum(trace * 0.5, pad)
            alb = _mix_rgb(c1, c2, trace * 0.75)
            alb = _mix_rgb(alb, c3, pad * 0.95)
            alb = alb * (1.0 - 0.92 * hole[..., None])
            met = np.clip(pad * 0.96 * (1.0 - hole), 0.0, 0.98).astype(np.float32)
            r_mod = -0.06 * pad - 0.03 * trace + 0.55 * hole
        elif mode == "solar":
            nx, ny = 4.0, 4.0
            fx = np.abs(((u * nx + 0.25) % 1.0) - 0.5) * 2.0
            fy = np.abs(((v * ny + 0.25) % 1.0) - 0.5) * 2.0
            gap = np.maximum(gd._smoothstep(0.92, 0.99, fx), gd._smoothstep(0.92, 0.99, fy))
            bus = gd._smoothstep(0.88, 0.96, np.abs(np.cos(math.pi * u * 12.0))) * (1.0 - gap)
            fin = gd._smoothstep(0.82, 0.94, np.abs(np.cos(math.pi * v * 48.0))) * (1.0 - gap) * 0.4
            _, _, facet = gd._periodic_worley_id(u, v, 12, seed + 191)
            h = np.clip(0.48 - 0.32 * gap + 0.14 * bus + 0.04 * fin, 0.05, 0.95)
            cav = gap
            ridge = np.maximum(bus, fin)
            alb = _mix_rgb(c1, c2, facet * 0.65 + 0.2 * m1)
            alb = _mix_rgb(alb, c3, np.maximum(bus, fin * 0.65))
            alb = _mix_rgb(alb, _rgb((0.78, 0.80, 0.82)), gap * 0.90)
            met = np.clip(0.92 * np.maximum(bus, fin * 0.5), 0.0, 0.96).astype(np.float32)
            r_mod = 0.45 * gap - 0.05 * bus
        else:
            # Sci-fi / industrial modular armor & tech hull plates with bevel seams & micro-ribs
            nx, ny = float(params.get("nx", 2)), float(params.get("ny", 2))
            fx = np.abs(((u * nx + 0.25) % 1.0) - 0.5) * 2.0
            fy = np.abs(((v * ny + 0.25) % 1.0) - 0.5) * 2.0
            seam = np.maximum(gd._smoothstep(0.91, 0.985, fx), gd._smoothstep(0.91, 0.985, fy))
            inset = np.maximum(gd._smoothstep(0.68, 0.74, fx), gd._smoothstep(0.68, 0.74, fy)) * (1.0 - seam)
            ix = np.floor(u * nx + 0.25)
            iy = np.floor(v * ny + 0.25)
            tid = _cell_hash(ix, iy, nx, ny, seed + 193)
            h = np.clip(0.56 - 0.42 * seam - 0.10 * inset + 0.04 * (tid - 0.5) + 0.03 * m2, 0.04, 0.96)
            cav = np.clip(seam + 0.35 * inset, 0.0, 1.0)
            ridge = gd._smoothstep(0.50, 0.90, 1.0 - cav)
            alb = _mix_rgb(c1, c2, 0.25 + 0.50 * tid + 0.25 * np.clip(0.5 + 0.5 * m1, 0.0, 1.0)) * (0.95 + 0.05 * m2[..., None])
            alb = _mix_rgb(alb, c3, seam * 0.82)
            met = np.clip(met_val * (1.0 - 0.35 * seam), 0.0, 0.99).astype(np.float32)
            r_mod = 0.28 * seam + 0.06 * inset - 0.04 * ridge

    else:
        raise ValueError(f"Unknown parametric material family: '{family}'")

    alb_max = np.where(met[..., None] > 0.25, 0.95, 0.855).astype(np.float32)
    alb_lin = np.clip(alb, 0.015, alb_max).astype(np.float32)
    outs = [
        np.asarray(h, dtype=np.float32),
        np.clip(cav, 0.0, 1.0).astype(np.float32),
        np.clip(ridge, 0.0, 1.0).astype(np.float32),
        alb_lin,
        np.clip(np.broadcast_to(met, (H, W)), 0.0, 0.99).astype(np.float32),
        np.asarray(np.broadcast_to(r_mod, (H, W)), dtype=np.float32),
    ]
    roll = params.get("roll")
    if roll:
        # Exact integer-pixel torus translation: keeps every map perfectly periodic while moving
        # lattice-aligned features (grooves, folds, seams) away from the texture border.
        sx = int(round(float(roll[0]) * W)) % W
        sy = int(round(float(roll[1]) * H)) % H
        outs = [np.roll(a, (sy, sx), axis=(0, 1)) for a in outs]
    h_o, cav_o, ridge_o, alb_o, met_o, rmod_o = outs
    return (h_o, slope, cav_o, base_r, ridge_o, alb_o, met_o, rmod_o)
