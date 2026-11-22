#!/usr/bin/env python3
"""V13 Material Quality Upgrade Pass for ALL 44 Seamless Daily Materials + HardOps Plate.

Upgrades every material from flat single-tint albedo & basic AO to AAA studio quality:
  1. True V13 8-Azimuth Horizon-Based AO (`HorizonBentNormalAO`) + Tangent-Space Bent Normal.
  2. Rich multi-tone physically-based Albedo per material class:
       - Glazed subway tile: warm cream ceramic faces + cool sanded mortar grout
       - Denim twill: classic indigo warp + ecru cotton weft yarns
       - PCB: emerald solder mask + bright copper/gold traces & pads + dark drilled vias
       - Solar cell: deep sapphire silicon + silver busbars & fingers + white backsheet gaps
       - Herringbone / acoustic oak: multi-tone honey/walnut wood grain + charcoal acoustic felt valleys
       - Quilted leather: rich espresso/saddle leather + warm tan contrast thread stitches
       - Machined metals / vents / grates: anisotropic steel/brass/iron tonal breakup + dark oil crevices + bright chamfer crests
       - Carbon fiber: alternating glossy graphite/charcoal tows
       - Cardboard: warm kraft pulp fiber mottling
  3. Calibrated per-pixel Roughness & Metallic separation (grout/felt/crevices matte dielectric,
     exposed crests/metal/glaze crisp low roughness) + Specular AA.
  4. Generates `hardops-plate` as a pure gunmetal machined HardOps panel
     (replacing the green palm-tree photo backdrop).
"""
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import depthpro_pbr as dp          # noqa: E402
from . import normal_graph as ng          # noqa: E402
from . import procedural as gd            # noqa: E402


def _rich_albedo_and_metal(did: str, mat_kind: str, tint: tuple, h: np.ndarray, cav: np.ndarray,
                           ridge: np.ndarray, ao: np.ndarray, seed: int):
    """Synthesize physically-grounded multi-tone linear RGB albedo, metallic, and roughness offset."""
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H

    m1 = gd._periodic_fbm(H, W, seed + 201, (4, 8, 16))
    m2 = gd._periodic_fbm(H, W, seed + 202, (16, 32, 64))
    m3 = gd._periodic_fbm(H, W, seed + 203, (2, 4, 8))
    var = 0.055 * m1 + 0.030 * m2
    # Subtle warm/cool chromatic drift breaks sterile single-RGB procedural uniformity
    chroma_drift = np.stack([0.022 * m3, 0.004 * m1, -0.022 * m3], axis=-1).astype(np.float32)

    base = np.array(tint, dtype=np.float32)[None, None, :]
    alb = base * (0.93 + var[..., None] + chroma_drift)
    met = np.zeros((H, W), dtype=np.float32)
    r_mod = np.zeros((H, W), dtype=np.float32)

    if did == "subway_metro_tile":
        # Hand-glazed warm ceramic subway tiles (subtle kiln variation + edge glaze pooling) + gritty sanded mortar grout
        tile_col = np.array([0.85, 0.855, 0.825], np.float32)[None, None, :] * (0.95 + 0.04 * m1[..., None] + 0.6 * chroma_drift)
        grout_col = np.array([0.34, 0.345, 0.335], np.float32)[None, None, :] * (0.86 + 0.14 * m2[..., None])
        alb = tile_col * (1.0 - cav[..., None]) + grout_col * cav[..., None]
        r_mod = -0.08 * (1.0 - cav) + 0.58 * cav + 0.03 * m1 * (1.0 - cav)

    elif did == "pcb_solder_mask_traces":
        # Authentic layered electronics PCB: translucent glossy emerald solder mask over woven FR4 fiberglass, brighter vivid green domed copper traces underneath, smooth gleaming ENIG gold SMD pads & via rings, dark drilled holes!
        fib_weave = 0.5 + 0.5 * np.sin(2.0 * math.pi * u * 64.0) * np.sin(2.0 * math.pi * v * 64.0)
        # ridge encodes traces in [0.15..0.44] and exposed ENIG gold pads in [0.60..1.0]
        gold_m = gd._smoothstep(0.55, 0.80, ridge) * (1.0 - cav)
        trace_under = gd._smoothstep(0.14, 0.34, ridge) * (1.0 - gold_m) * (1.0 - cav)
        via_dark = np.clip(cav * 1.35, 0.0, 1.0)
        # Translucent emerald solder mask (darker over bare FR4 substrate, noticeably brighter vivid green over raised copper traces!)
        mask_fr4 = np.array([0.016, 0.145, 0.058], np.float32)[None, None, :] * (0.88 + 0.14 * fib_weave[..., None] + 0.04 * m1[..., None])
        mask_cu  = np.array([0.046, 0.355, 0.148], np.float32)[None, None, :] * (0.96 + 0.04 * m1[..., None])
        # Smooth gleaming ENIG gold pads (NO grainy checkerboard noise!)
        gold_col = np.array([0.92, 0.72, 0.30], np.float32)[None, None, :] * (0.985 + 0.015 * m1[..., None])
        alb = mask_fr4 * (1.0 - trace_under[..., None]) + mask_cu * trace_under[..., None]
        alb = alb * (1.0 - gold_m[..., None]) + gold_col * gold_m[..., None]
        alb *= (1.0 - 0.94 * via_dark[..., None])
        # Only bare exposed ENIG gold pads & via rings are metallic (>=0.96); solder-masked traces stay glossy dielectric clearcoat!
        met = np.clip(gold_m * 0.97 * (1.0 - via_dark), 0.0, 0.98).astype(np.float32)
        # Ultra-smooth mirror-gold ENIG pads (~0.08), glossy solder-masked traces (~0.11), satin FR4 (~0.21), matte dark via holes
        r_mod = 0.07 * (1.0 - trace_under - gold_m) - 0.03 * trace_under - 0.07 * gold_m + 0.56 * via_dark + 0.015 * m1 * (1.0 - gold_m)

    elif did == "solar_cell_grid":
        # Deep sapphire crystalline silicon wafer with crystal-facet glint + bright metallic silver busbars & micro-fingers + white backsheet
        _, _, si_facet = gd._periodic_worley_id(u, v, 10, seed + 69)
        si_lo = np.array([0.028, 0.065, 0.205], np.float32)[None, None, :]
        si_hi = np.array([0.075, 0.145, 0.345], np.float32)[None, None, :]
        si_col = (si_lo * (1.0 - si_facet[..., None]) + si_hi * si_facet[..., None]) * (0.94 + 0.06 * m1[..., None])
        bus_col = np.array([0.80, 0.82, 0.85], np.float32)[None, None, :]
        gap_col = np.array([0.80, 0.81, 0.82], np.float32)[None, None, :] * (0.96 + 0.04 * m2[..., None])
        alb = si_col * (1.0 - cav[..., None]) + gap_col * cav[..., None]
        alb = alb * (1.0 - ridge[..., None]) + bus_col * ridge[..., None]
        met = np.clip(ridge * 0.95, 0.0, 0.96).astype(np.float32)
        r_mod = -0.04 * (1.0 - cav) + 0.42 * cav + 0.025 * (si_facet - 0.5) * (1.0 - cav)

    elif did == "woven_denim_twill":
        # Authentic fine indigo jeans denim twill: dense indigo-dyed warp + narrow diagonal ecru cotton weft flecks + slub fade!
        diag_f = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 32.0 + v * 32.0))
        weft_fleck = gd._smoothstep(0.68, 0.94, diag_f) * (0.5 + 0.5 * np.cos(2.0 * math.pi * u * 64.0))
        slub = 0.5 + 0.5 * gd._periodic_fbm(H, W, seed + 205, (4, 32))
        indigo_lo = np.array([0.050, 0.098, 0.225], np.float32)[None, None, :]
        indigo_hi = np.array([0.120, 0.205, 0.390], np.float32)[None, None, :]
        ecru = np.array([0.50, 0.50, 0.49], np.float32)[None, None, :]
        indigo = indigo_lo * (1.0 - 0.58 * slub[..., None]) + indigo_hi * (0.58 * slub[..., None])
        alb = indigo * (1.0 - 0.46 * weft_fleck[..., None]) + ecru * (0.46 * weft_fleck[..., None])
        alb *= (0.85 + 0.15 * ao[..., None])
        r_mod = 0.05 * cav - 0.04 * slub

    elif did == "quilted_leather_stitch":
        # Supple aniline espresso-cognac leather: warm burnished cushion crests, dark pore creases, twisted golden-tan saddle stitches!
        leath_lo = np.array([0.15, 0.070, 0.040], np.float32)[None, None, :]
        leath_hi = np.array([0.33, 0.165, 0.095], np.float32)[None, None, :]
        wear_crest = np.clip(0.38 + 0.45 * h + 0.18 * m1 + 0.10 * m3, 0.0, 1.0)[..., None]
        leath = (leath_lo * (1.0 - wear_crest) + leath_hi * wear_crest) * (0.94 + 0.06 * m2[..., None] + 0.7 * chroma_drift)
        stitch = np.array([0.76, 0.58, 0.35], np.float32)[None, None, :] * (0.92 + 0.08 * m2[..., None])
        stitch_m = np.clip(ridge * cav * 1.9, 0.0, 1.0)
        alb = leath * (1.0 - stitch_m[..., None]) + stitch * stitch_m[..., None]
        alb *= (0.78 + 0.22 * ao[..., None])
        # Natural leather sheen: glossy burnished cushions (0.26) vs matte pores/seams/thread (0.62+)
        r_mod = -0.09 * ridge + 0.26 * cav + 0.22 * stitch_m + 0.05 * m1

    elif did == "wood_parquet_herringbone":
        # Authentic multi-tone 4:1 European oak herringbone parquet: per-plank warm/cool oak hue, cathedral grain, dark oak pores, soft bevels!
        N = 8.0
        rx, ry = (u + v) * N, (u - v) * N
        ix, iy = np.floor(rx), np.floor(ry)
        d = (ix - iy) % 8.0
        is_h = d < 4.0
        pid_x = np.where(is_h, (ix - d) % N, ix % N)
        pid_y = np.where(is_h, iy % N, (iy - (d - 4.0)) % N)
        plank_id = np.sin(pid_x * 13.7 + pid_y * 29.3) * 0.5 + 0.5
        plank_hue = np.cos(pid_x * 37.1 - pid_y * 19.7) * 0.5 + 0.5
        freq_p = 3.2 + 2.4 * plank_hue
        phase_p = 6.283 * plank_id
        grain = 0.5 + 0.5 * np.where(
            is_h,
            np.sin(2.0 * math.pi * (ry * freq_p + 0.28 * m1) + phase_p + 0.70 * m2),
            np.sin(2.0 * math.pi * (rx * freq_p + 0.28 * m1) + phase_p + 0.70 * m2),
        )
        oak_honey = np.array([0.62, 0.42, 0.22], np.float32)[None, None, :]
        oak_amber = np.array([0.46, 0.29, 0.14], np.float32)[None, None, :]
        oak_walnut = np.array([0.27, 0.16, 0.08], np.float32)[None, None, :]
        p_t = np.clip(0.15 + 0.65 * plank_id + 0.20 * grain, 0.0, 1.0)[..., None]
        oak_base = oak_walnut * (1.0 - p_t) + oak_honey * p_t
        oak_base = oak_base * (0.82 + 0.30 * plank_hue[..., None]) * 0.55 + oak_amber * 0.45 * (0.88 + 0.24 * grain[..., None])
        # Subtle dark open oak wood pores + natural warm tongue-and-groove shadow (no harsh black vector lines!)
        seam_col = np.array([0.11, 0.065, 0.032], np.float32)[None, None, :]
        alb = oak_base * (1.0 - 0.62 * cav[..., None]) + seam_col * (0.62 * cav[..., None])
        alb *= (0.94 + 0.06 * m2[..., None] + 0.5 * chroma_drift)
        # Satin-varnish lacquer with subtle worn foot-traffic gloss variation & matte open pores
        r_mod = 0.24 * cav - 0.06 * grain + 0.05 * m3

    elif did == "acoustic_slat_wood":
        # Architectural slatted natural oak over fibrous charcoal needle-punch felt: per-slat wood tint & cathedral grain!
        n_slats = 12.0
        sid = np.floor(u * n_slats) % n_slats
        s_hash = np.sin(sid * 19.3 + 4.1) * 0.5 + 0.5
        s_hash2 = np.cos(sid * 31.7 + 1.9) * 0.5 + 0.5
        grain = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * (48.0 + 24.0 * s_hash2) + 0.16 * m1) + 6.283 * s_hash)
        oak_lo = np.array([0.34, 0.21, 0.11], np.float32)[None, None, :]
        oak_hi = np.array([0.65, 0.46, 0.26], np.float32)[None, None, :]
        o_mix = np.clip(0.22 + 0.48 * s_hash + 0.24 * grain + 0.08 * m1, 0.0, 1.0)[..., None]
        oak = (oak_lo * (1.0 - o_mix) + oak_hi * o_mix) * (0.94 + 0.06 * m2[..., None] + 0.5 * chroma_drift)
        # Subtle pale milled chamfer glint along slat corners
        oak = oak * (1.0 + 0.14 * ridge[..., None])
        # Fibrous heather-charcoal needle-punch acoustic felt backing
        felt_lo = np.array([0.035, 0.037, 0.042], np.float32)[None, None, :]
        felt_hi = np.array([0.095, 0.098, 0.108], np.float32)[None, None, :]
        f_t = np.clip(0.5 + 0.45 * m2, 0.0, 1.0)[..., None]
        felt = felt_lo * (1.0 - f_t) + felt_hi * f_t
        felt_m = gd._smoothstep(0.35, 0.85, cav)
        alb = oak * (1.0 - felt_m[..., None]) + felt * felt_m[..., None]
        r_mod = 0.46 * felt_m - 0.05 * grain + 0.04 * m3

    elif did == "acoustic_pegboard":
        # Natural birch-ply / tempered hardboard acoustic pegboard: crisp elongated wood veneer grain + pale milled countersunk bevel rim + deep bore (ZERO blurry clouds!)
        w_grain = 0.5 + 0.34 * np.sin(2.0 * math.pi * (u * 36.0 + 0.18 * m1) + 0.35 * m2) + 0.16 * np.sin(2.0 * math.pi * (u * 72.0 - 0.12 * m1))
        birch_lo = np.array([0.46, 0.34, 0.22], np.float32)[None, None, :]
        birch_hi = np.array([0.70, 0.57, 0.41], np.float32)[None, None, :]
        g_t = np.clip(0.18 + 0.72 * w_grain + 0.05 * m2, 0.0, 1.0)[..., None]
        board = (birch_lo * (1.0 - g_t) + birch_hi * g_t) * (0.95 + 0.05 * m2[..., None] + 0.45 * chroma_drift)
        # Warm milled birch/MDF countersunk chamfer wall + crisp rim highlight + deep dark bore
        cham_c = np.array([0.38, 0.27, 0.16], np.float32)[None, None, :]
        rim_hi = np.array([0.76, 0.64, 0.48], np.float32)[None, None, :]
        void_c = np.array([0.015, 0.014, 0.013], np.float32)[None, None, :]
        bore_m = gd._smoothstep(0.50, 0.84, cav)
        cham_m = gd._smoothstep(0.08, 0.46, cav) * (1.0 - bore_m)
        alb = board * (1.0 - 0.28 * ridge[..., None]) + rim_hi * (0.28 * ridge[..., None])
        alb = alb * (1.0 - 0.52 * cham_m[..., None]) + cham_c * (0.52 * cham_m[..., None])
        alb = alb * (1.0 - 0.96 * bore_m[..., None]) + void_c * (0.96 * bore_m[..., None])
        r_mod = 0.36 * bore_m + 0.16 * cham_m - 0.08 * ridge + 0.04 * m2

    elif did == "sidewalk_expansion_joint":
        # Authentic cured Portland-cement concrete: multi-scale quartz/basalt aggregate speckle + warm/cool weather drift + gritty joint dirt
        _, _, agg_id = gd._periodic_worley_id(u, v, 38, seed + 10)
        cem_lo = np.array([0.44, 0.435, 0.415], np.float32)[None, None, :]
        cem_hi = np.array([0.64, 0.635, 0.605], np.float32)[None, None, :]
        c_mix = np.clip(0.45 + 0.30 * m3 + 0.18 * m1 + 0.12 * (agg_id - 0.5), 0.0, 1.0)[..., None]
        conc = (cem_lo * (1.0 - c_mix) + cem_hi * c_mix) * (1.0 + 0.9 * chroma_drift)
        # Fine exposed dark basalt & bright quartz sand mineral flecks across concrete face
        dark_speck = gd._smoothstep(0.82, 0.96, agg_id)[..., None]
        pale_speck = gd._smoothstep(0.14, 0.03, agg_id)[..., None]
        conc = conc * (1.0 - 0.32 * dark_speck) + np.array([0.78, 0.77, 0.74], np.float32) * (0.24 * pale_speck)
        # Gritty dark accumulated dirt & shadow inside control joint
        joint_dirt = np.array([0.13, 0.125, 0.115], np.float32)[None, None, :] * (0.85 + 0.15 * m2[..., None])
        j_m = gd._smoothstep(0.22, 0.78, cav)
        alb = conc * (1.0 - 0.85 * j_m[..., None]) + joint_dirt * (0.85 * j_m[..., None])
        alb *= (0.86 + 0.14 * ao[..., None])
        r_mod = 0.10 * j_m - 0.08 * ridge + 0.06 * m1

    elif did == "ribbed_corduroy_velvet":
        # Plush chestnut/cognac cotton corduroy wales: clean micro-fiber velvet pile + warm grazing crest sheen + deep shadow troughs (NO muddy blotches!)
        _, _, pile_id = gd._periodic_worley_id(u, v, 84, seed + 206)
        pile_base = np.array([0.36, 0.195, 0.115], np.float32)[None, None, :] * (0.94 + 0.05 * m1[..., None] + 0.06 * (pile_id[..., None] - 0.5) + 0.35 * chroma_drift)
        pile_sheen = np.array([0.52, 0.31, 0.20], np.float32)[None, None, :]
        trough_c = np.array([0.085, 0.044, 0.026], np.float32)[None, None, :]
        wale_col = pile_base * (1.0 - 0.38 * ridge[..., None]) + pile_sheen * (0.38 * ridge[..., None])
        alb = wale_col * (1.0 - 0.74 * cav[..., None]) + trough_c * (0.74 * cav[..., None])
        alb *= (0.82 + 0.18 * ao[..., None])
        r_mod = 0.06 * cav - 0.05 * ridge + 0.02 * m2

    elif did == "carbon_fiber_weave":
        # Ultra-glossy clearcoat 2x2 twill carbon-fiber weave (28x28 tows = 7 repeats, strong orthogonal anisotropic sheen!)
        n = 28.0
        us = (u + 0.5 / n) % 1.0
        vs = (v + 0.5 / n) % 1.0
        iu, iv = np.floor(us * n), np.floor(vs * n)
        fu, fv = (us * n) % 1.0, (vs * n) % 1.0
        twill = ((iu + iv) % 4.0) < 2.0
        sheen_u = np.sin(math.pi * fu) ** 0.55
        sheen_v = np.sin(math.pi * fv) ** 0.55
        c_dark = np.array([0.024, 0.027, 0.034], np.float32)[None, None, :] * (0.82 + 0.30 * sheen_u[..., None])
        c_bright = np.array([0.23, 0.25, 0.29], np.float32)[None, None, :] * (0.72 + 0.42 * sheen_v[..., None])
        alb = np.where(twill[..., None], c_dark, c_bright)
        met = np.where(twill, 0.22, 0.62).astype(np.float32) * (1.0 - 0.35 * cav)
        r_mod = -0.03 * ridge + 0.06 * cav

    elif did == "corrugated_cardboard":
        # Authentic recycled kraft paperboard: tight fibrous pulp grain & fine dark/pale cellulose flecks (zero oversized stain blotches!)
        _, _, pulp_id = gd._periodic_worley_id(u, v, 96, seed + 207)
        kraft_lo = np.array([0.48, 0.35, 0.22], np.float32)[None, None, :]
        kraft_hi = np.array([0.59, 0.44, 0.29], np.float32)[None, None, :]
        k_t = np.clip(0.5 + 0.12 * m3 + 0.16 * m1 + 0.18 * (pulp_id - 0.5), 0.0, 1.0)[..., None]
        kraft = (kraft_lo * (1.0 - k_t) + kraft_hi * k_t) * (1.0 + 0.4 * chroma_drift)
        dark_fleck = gd._smoothstep(0.86, 0.97, pulp_id)[..., None]
        pale_fleck = gd._smoothstep(0.12, 0.02, pulp_id)[..., None]
        alb = kraft * (1.0 - 0.28 * dark_fleck) + np.array([0.74, 0.61, 0.45], np.float32) * (0.18 * pale_fleck)
        alb *= (0.84 + 0.16 * ao[..., None])
        r_mod = 0.08 * cav + 0.03 * m2

    elif did == "terrazzo_speckled_stone":
        # Authentic Venetian honed terrazzo: warm ivory-cream pitted cement + 5 rich Italian marble chip minerals with subtle veining!
        cem = np.array([0.77, 0.74, 0.68], np.float32)[None, None, :] * (0.94 + 0.06 * m2[..., None] + 0.6 * chroma_drift)
        vein_in = 0.88 + 0.20 * np.clip(0.5 + 0.5 * m2, 0.0, 1.0)[..., None]
        c_terra = np.array([0.55, 0.24, 0.15], np.float32)[None, None, :] * vein_in
        c_slate = np.array([0.14, 0.16, 0.19], np.float32)[None, None, :] * vein_in
        c_ochre = np.array([0.69, 0.52, 0.28], np.float32)[None, None, :] * vein_in
        c_white = np.array([0.84, 0.82, 0.78], np.float32)[None, None, :] * vein_in
        c_sage  = np.array([0.34, 0.42, 0.36], np.float32)[None, None, :] * vein_in
        chip = np.clip(ridge, 0.0, 1.0)
        # cav encodes mineral bucket in [0..1] inside chips
        b1 = (1.0 - gd._smoothstep(0.18, 0.22, cav))
        b2 = gd._smoothstep(0.18, 0.22, cav) * (1.0 - gd._smoothstep(0.38, 0.42, cav))
        b3 = gd._smoothstep(0.38, 0.42, cav) * (1.0 - gd._smoothstep(0.58, 0.62, cav))
        b4 = gd._smoothstep(0.58, 0.62, cav) * (1.0 - gd._smoothstep(0.78, 0.82, cav))
        b5 = gd._smoothstep(0.78, 0.82, cav)
        chip_col = (
            c_slate * b1[..., None] + c_terra * b2[..., None] + c_ochre * b3[..., None]
            + c_white * b4[..., None] + c_sage * b5[..., None]
        )
        # Subtle dark pin-pits in cement binder
        pit_dark = gd._smoothstep(0.49, 0.42, h) * (1.0 - chip)
        cem = cem * (1.0 - 0.45 * pit_dark[..., None])
        alb = cem * (1.0 - chip[..., None]) + chip_col * chip[..., None]
        # Polished marble chips are noticeably glossier (~0.13) than honed cement binder (~0.34)
        r_mod = 0.12 * (1.0 - chip) - 0.10 * chip + 0.30 * pit_dark + 0.03 * m1

    elif did == "venetian_plaster_trowel":
        # Authentic Italian burnished Marmorino Venetian lime plaster: subtle warm limestone-ivory clouding with visual drama driven by mirror-burnished vs chalky-matte spatula passes (ZERO dark marble crease veins!)
        ivory = np.array([0.81, 0.775, 0.71], np.float32)[None, None, :]
        warm_lime = np.array([0.70, 0.655, 0.585], np.float32)[None, None, :]
        t_mix = np.clip(0.35 + 0.45 * ridge - 0.20 * cav + 0.08 * m1, 0.0, 1.0)[..., None]
        alb = warm_lime * (1.0 - t_mix) + ivory * t_mix
        alb *= (0.97 + 0.03 * m2[..., None] + 0.45 * chroma_drift)
        # True Marmorino hallmark: mirror-burnished spatula sweeps (~0.11) contrasted cleanly against chalky matte lime (~0.60)!
        r_mod = -0.22 * ridge + 0.25 * cav + 0.03 * m1

    elif did == "cable_knit_wool":
        # Cozy oatmeal/cream Fisherman's Aran cable-knit wool with natural wool heather flecks & deep shadow pockets
        cream = np.array([0.73, 0.67, 0.58], np.float32)[None, None, :] * (0.92 + 0.08 * m1[..., None] + 0.5 * chroma_drift)
        oat = np.array([0.45, 0.40, 0.33], np.float32)[None, None, :] * (0.90 + 0.10 * m2[..., None])
        alb = oat * cav[..., None] + cream * (1.0 - cav[..., None])
        alb = alb * (0.78 + 0.22 * ao[..., None]) + 0.035 * ridge[..., None]
        r_mod = -0.04 * ridge + 0.08 * cav

    elif did == "penny_round_mosaic":
        # Hand-glazed deep teal/sea-glass penny tiles (per-tile glaze tonal shift + edge glaze pooling) + gritty warm-off-white grout
        glaze_lo = np.array([0.05, 0.26, 0.31], np.float32)[None, None, :]
        glaze_hi = np.array([0.14, 0.44, 0.46], np.float32)[None, None, :]
        g_t = np.clip(0.45 + 0.38 * m3 + 0.22 * m1, 0.0, 1.0)[..., None]
        glaze = (glaze_lo * (1.0 - g_t) + glaze_hi * g_t) * (0.94 + 0.06 * m2[..., None])
        grout = np.array([0.75, 0.74, 0.70], np.float32)[None, None, :] * (0.88 + 0.12 * m2[..., None])
        alb = glaze * (1.0 - cav[..., None]) + grout * cav[..., None]
        r_mod = -0.07 * (1.0 - cav) + 0.62 * cav + 0.03 * m1 * (1.0 - cav)

    elif did == "hex_mosaic_tile":
        # Satin-white 1-inch honeycomb ceramic hex mosaic tiles (subtle ceramic kiln variation) + gritty cool-gray sanded mortar grout
        white_t = np.array([0.82, 0.815, 0.79], np.float32)[None, None, :] * (0.95 + 0.045 * m1[..., None] + 0.5 * chroma_drift)
        grout = np.array([0.33, 0.345, 0.36], np.float32)[None, None, :] * (0.86 + 0.14 * m2[..., None])
        alb = white_t * (1.0 - cav[..., None]) + grout * cav[..., None]
        r_mod = -0.05 * (1.0 - cav) + 0.60 * cav + 0.03 * m3 * (1.0 - cav)

    elif did == "hammered_copper_dimple":
        # Hand-planished luminous warm rose copper: authentic metallic copper F0 with subtle antique bronze-rose patina in bowl centers
        cop = np.array([0.94, 0.53, 0.33], np.float32)[None, None, :] * (0.94 + 0.06 * m1[..., None] + 0.55 * chroma_drift)
        pat = np.array([0.48, 0.21, 0.12], np.float32)[None, None, :] * (0.92 + 0.08 * m2[..., None])
        alb = pat * (0.30 * cav[..., None]) + cop * (1.0 - 0.30 * cav[..., None])
        met = np.clip(0.98 * (1.0 - 0.08 * cav), 0.86, 0.99).astype(np.float32)
        r_mod = -0.04 * ridge + 0.14 * cav + 0.035 * m1

    elif did == "fluted_reeded_glass":
        # Architectural optical fluted privacy glass: crystal-aqua bright caustic bands + deep emerald-teal internal edge absorption!
        # Subtle refractive background light-dark vertical bands distorted across each cylindrical reed
        refract_band = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 24.0 + 0.22 * m3) + 0.35 * np.sin(2.0 * math.pi * v * 2.0))
        c_deep = np.array([0.16, 0.34, 0.33], np.float32)[None, None, :]
        c_mid  = np.array([0.52, 0.70, 0.68], np.float32)[None, None, :]
        c_caust = np.array([0.82, 0.85, 0.84], np.float32)[None, None, :]
        glass = c_mid * (0.78 + 0.28 * refract_band[..., None])
        glass = glass * (1.0 - 0.72 * cav[..., None]) + c_deep * (0.72 * cav[..., None])
        glass = glass * (1.0 - 0.42 * ridge[..., None]) + c_caust * (0.42 * ridge[..., None])
        alb = glass
        r_mod = -0.01 * ridge + 0.03 * cav

    elif did == "frosted_prism_diffuser":
        # Luminous frosted crystal acrylic prism diffuser with crisp bright/cool facet light-scatter contrast
        c_lo = np.array([0.58, 0.66, 0.71], np.float32)[None, None, :]
        c_hi = np.array([0.84, 0.855, 0.85], np.float32)[None, None, :]
        alb = (c_lo * (1.0 - (0.35 + 0.55 * ridge)[..., None]) + c_hi * (0.35 + 0.55 * ridge)[..., None]) * (0.94 + 0.06 * ao[..., None])
        r_mod = -0.06 * ridge + 0.08 * cav

    elif did == "fine_asphalt_tarmac":
        # Authentic rolled road asphalt: dark slate-charcoal bitumen mortar + subtle fine basalt/granite/feldspar/quartz aggregate flecks!
        bit = np.array([0.068, 0.071, 0.078], np.float32)[None, None, :] * (0.88 + 0.18 * m3[..., None])
        agg_dark = np.array([0.108, 0.112, 0.120], np.float32)[None, None, :]
        agg_mid  = np.array([0.185, 0.186, 0.184], np.float32)[None, None, :]
        agg_warm = np.array([0.275, 0.252, 0.222], np.float32)[None, None, :]
        agg_qrtz = np.array([0.42, 0.415, 0.40], np.float32)[None, None, :]
        stone = agg_dark * (1.0 - ridge[..., None]) + agg_mid * ridge[..., None]
        warm_m = gd._smoothstep(0.56, 0.84, ridge) * gd._smoothstep(-0.05, 0.75, m1)
        qrtz_m = gd._smoothstep(0.84, 0.96, ridge)
        stone = stone * (1.0 - 0.55 * warm_m[..., None]) + agg_warm * (0.55 * warm_m[..., None])
        stone = stone * (1.0 - 0.55 * qrtz_m[..., None]) + agg_qrtz * (0.55 * qrtz_m[..., None])
        alb = stone * (1.0 - 0.58 * cav[..., None]) + bit * (0.58 * cav[..., None])
        # Subtle worn tire-track oil sheen variation across bitumen vs rough gritty aggregate
        r_mod = -0.14 * qrtz_m - 0.10 * cav * np.clip(0.5 + 0.5 * m3, 0.0, 1.0) + 0.06 * (1.0 - cav)

    elif did == "corkboard_pin_surface":
        # Authentic compressed natural corkboard: organic warm honey/tan/sienna bark flakes + fine spongy cellular grain (soft warm compressed joints, ZERO dark polygon borders!)
        c_pale = np.array([0.68, 0.51, 0.33], np.float32)[None, None, :]
        c_mid  = np.array([0.54, 0.38, 0.22], np.float32)[None, None, :]
        c_bark = np.array([0.36, 0.23, 0.125], np.float32)[None, None, :]
        c_pore = np.array([0.21, 0.13, 0.070], np.float32)[None, None, :]
        g1 = gd._smoothstep(0.14, 0.52, ridge)[..., None]
        g2 = gd._smoothstep(0.52, 0.90, ridge)[..., None]
        flake = c_bark * (1.0 - g1) + c_mid * g1
        flake = flake * (1.0 - g2) + c_pale * g2
        # Fine spongy cork cellular speckle inside each granule
        _, _, lenticel = gd._periodic_worley_id(u, v, 88, seed + 208)
        flake *= (0.90 + 0.08 * m2[..., None] + 0.12 * (lenticel[..., None] - 0.5) + 0.45 * chroma_drift)
        alb = flake * (1.0 - 0.28 * cav[..., None]) + c_pore * (0.28 * cav[..., None])
        alb *= (0.90 + 0.10 * ao[..., None])
        r_mod = 0.08 * cav - 0.04 * ridge + 0.04 * m1

    elif did == "patinated_verdigris_bronze":
        # Authentic aged cast architectural bronze: warm satin antique bronze -> dark chocolate cuprite -> matte seafoam/turquoise verdigris in pits & drip channels!
        bronze_lo = np.array([0.32, 0.20, 0.095], np.float32)[None, None, :]
        bronze_hi = np.array([0.78, 0.54, 0.24], np.float32)[None, None, :]
        b_mix = np.clip(0.28 + 0.62 * h + 0.18 * m1, 0.0, 1.0)[..., None]
        bronze = (bronze_lo * (1.0 - b_mix) + bronze_hi * b_mix) * (1.0 + 0.6 * chroma_drift)
        cuprite = np.array([0.14, 0.075, 0.038], np.float32)[None, None, :] * (0.88 + 0.12 * m2[..., None])
        verd_lo = np.array([0.11, 0.36, 0.30], np.float32)[None, None, :]
        verd_hi = np.array([0.27, 0.56, 0.45], np.float32)[None, None, :]
        v_mix = np.clip(0.45 + 0.45 * m2 + 0.20 * m1, 0.0, 1.0)[..., None]
        verd = verd_lo * (1.0 - v_mix) + verd_hi * v_mix
        alb = bronze * (1.0 - 0.78 * ridge[..., None]) + cuprite * (0.78 * ridge[..., None])
        alb = alb * (1.0 - 0.92 * cav[..., None]) + verd * (0.92 * cav[..., None])
        met = np.clip(0.96 * (1.0 - 0.38 * ridge) * (1.0 - 0.96 * cav), 0.0, 0.99).astype(np.float32)
        r_mod = -0.04 * (h - 0.5) + 0.22 * ridge + 0.72 * cav + 0.04 * m1

    elif did == "acoustic_kerf_felt":
        # Genuine matte heathered wool-PET acoustic felt: ultra-fine tangled wool micro-fibers + clean kerf slits (ZERO low-frequency camo/concrete blotches!)
        f_hi = np.array([0.41, 0.42, 0.44], np.float32)[None, None, :]
        f_mid = np.array([0.33, 0.34, 0.36], np.float32)[None, None, :]
        f_lo = np.array([0.25, 0.26, 0.28], np.float32)[None, None, :]
        slit_c = np.array([0.022, 0.024, 0.028], np.float32)[None, None, :]
        _, _, f_fiber1 = gd._periodic_worley_id(u, v, 112, seed + 210)
        _, _, f_fiber2 = gd._periodic_worley_id((u + 0.017) % 1.0, (v + 0.011) % 1.0, 156, seed + 212)
        _, _, f_fiber3 = gd._periodic_worley_id((u + 0.031) % 1.0, (v + 0.023) % 1.0, 208, seed + 213)
        strand_a = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 96.0 + v * 48.0) + 1.8 * f_fiber1)
        strand_b = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 52.0 - v * 104.0) + 1.8 * f_fiber2)
        # Strictly high-frequency needle-punched wool heathering (0% macro blotches!)
        heather = np.clip(0.50 + 0.34 * (f_fiber1 - 0.5) + 0.30 * (f_fiber2 - 0.5) + 0.24 * (f_fiber3 - 0.5) + 0.14 * (strand_a - 0.5) + 0.14 * (strand_b - 0.5), 0.0, 1.0)
        h1 = gd._smoothstep(0.18, 0.52, heather)[..., None]
        h2 = gd._smoothstep(0.52, 0.86, heather)[..., None]
        felt = f_lo * (1.0 - h1) + f_mid * h1
        felt = felt * (1.0 - h2) + f_hi * h2
        slit_m = gd._smoothstep(0.28, 0.80, cav)
        alb = felt * (1.0 - 0.92 * slit_m[..., None]) + slit_c * (0.92 * slit_m[..., None])
        alb *= (0.86 + 0.14 * ao[..., None])
        r_mod = 0.05 * cav + 0.055 * (f_fiber1 - 0.5) - 0.035 * (f_fiber2 - 0.5)

    elif did == "carrara_veined_marble":
        # Luminous Italian Statuario/Carrara calcite: warm/cool translucent crystalline depth + wide feathery smoky blue-grey diffusion + layered mineral veins!
        cal_warm = np.array([0.85, 0.842, 0.822], np.float32)[None, None, :]
        cal_cool = np.array([0.785, 0.798, 0.818], np.float32)[None, None, :]
        c_t = np.clip(0.45 + 0.35 * m3 + 0.15 * m1, 0.0, 1.0)[..., None]
        calcite = (cal_warm * (1.0 - c_t) + cal_cool * c_t) * (0.98 + 0.02 * m2[..., None])
        cloud_c = np.array([0.56, 0.59, 0.64], np.float32)[None, None, :]
        core_c = np.array([0.22, 0.24, 0.285], np.float32)[None, None, :]
        # Smooth translucent mineral veil blending seamlessly into polished calcite
        alb = calcite * (1.0 - 0.72 * cav[..., None]) + cloud_c * (0.72 * cav[..., None])
        alb = alb * (1.0 - 0.84 * ridge[..., None]) + core_c * (0.84 * ridge[..., None])
        # Mirror-polished calcite (~0.06) vs subtly etched mineral vein seams (~0.20)
        r_mod = 0.025 * cav + 0.14 * ridge + 0.012 * m1

    elif did == "herringbone_tweed_wool":
        # Authentic woven Harris/Donegal herringbone tweed: interlocked zigzag twill yarns + multi-tone wool heather & rust/ochre Donegal flecks
        cols = 4.0
        us = (u + 0.5 / cols) % 1.0
        zig = 0.25 * (2.0 / math.pi) * np.arcsin(0.92 * np.sin(2.0 * math.pi * us * (cols * 0.5))) + 0.25
        yarn_t = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 12.0 + zig * 6.0))
        heather = np.array([0.20, 0.195, 0.19], np.float32)[None, None, :] * (0.90 + 0.10 * m1[..., None])
        oat = np.array([0.56, 0.50, 0.41], np.float32)[None, None, :] * (0.90 + 0.10 * m2[..., None])
        fleck = gd._smoothstep(1.25, 1.68, m2)[..., None]
        rust = np.array([0.48, 0.25, 0.15], np.float32)[None, None, :]
        alb = heather * (1.0 - yarn_t[..., None]) + oat * yarn_t[..., None]
        alb = alb * (1.0 - 0.55 * fleck) + rust * (0.55 * fleck)
        alb *= (0.78 + 0.22 * ao[..., None])
        r_mod = 0.04 * cav

    elif did == "waffle_cotton_linen":
        # Natural woven sage-eucalyptus cotton with bright sunlit interlaced thread crests & warm shadowed waffle pockets
        sage = np.array([0.39, 0.48, 0.41], np.float32)[None, None, :] * (0.93 + 0.07 * m1[..., None] + 0.5 * chroma_drift)
        linen_hi = np.array([0.57, 0.63, 0.55], np.float32)[None, None, :]
        alb = sage * (1.0 - 0.38 * ridge[..., None]) + linen_hi * (0.38 * ridge[..., None])
        alb *= (0.72 + 0.28 * ao[..., None])
        r_mod = 0.05 * cav - 0.03 * ridge

    elif did == "damascus_folded_steel":
        # Authentic Japanese folded Damascus steel: gleaming bright nickel-alloy steel vs acid-etched dark matte-graphite carbon steel!
        dark_s = np.array([0.14, 0.155, 0.185], np.float32)[None, None, :] * (0.92 + 0.08 * m2[..., None])
        bright_s = np.array([0.86, 0.88, 0.92], np.float32)[None, None, :] * (0.94 + 0.06 * m1[..., None])
        alb = dark_s * (1.0 - ridge[..., None]) + bright_s * ridge[..., None]
        met = np.clip(0.88 + 0.11 * ridge, 0.84, 0.99).astype(np.float32)
        # Sharp physical roughness contrast: polished nickel bands (~0.11) vs ferric-chloride acid-etched carbon bands (~0.36)!
        r_mod = -0.05 * ridge + 0.22 * cav + 0.025 * m1

    elif did == "mermaid_fishscale_tile":
        # Handmade Moroccan Zellige glazed fish-scale tiles: lush per-tile turquoise/jade/peacock kiln variation + dark rim glaze pooling + wet mirror gloss + sandy grout!
        nx, ny = 6.0, 10.0
        wobble = gd._periodic_fbm(H, W, seed + 126, (6, 12, 24))
        tile_var = np.full((H, W), 0.5, dtype=np.float32)
        tile_var2 = np.full((H, W), 0.5, dtype=np.float32)
        base_ry = np.floor(v * ny)
        for row_off in (-2, -1, 0, 1, 2):
            ry = base_ry + row_off
            u_s = (u + 0.5 * (ry % 2.0) / nx) % 1.0
            rx = np.floor(u_s * nx)
            fu = (u_s * nx) % 1.0 - 0.5
            fv = (v * ny - ry) * 0.5
            r = np.sqrt(fu * fu + fv * fv) + 0.009 * wobble
            inside = r < 0.50
            tid = np.sin((rx % nx) * 17.3 + (ry % ny) * 31.7) * 0.5 + 0.5
            tid2 = np.cos((rx % nx) * 29.1 - (ry % ny) * 13.7) * 0.5 + 0.5
            tile_var = np.where(inside, tid, tile_var)
            tile_var2 = np.where(inside, tid2, tile_var2)
        peacock = np.array([0.025, 0.21, 0.30], np.float32)[None, None, :]
        teal = np.array([0.055, 0.36, 0.42], np.float32)[None, None, :]
        jade = np.array([0.14, 0.52, 0.46], np.float32)[None, None, :]
        deep_rim = np.array([0.015, 0.105, 0.165], np.float32)[None, None, :]
        g1 = gd._smoothstep(0.15, 0.60, tile_var)[..., None]
        g2 = gd._smoothstep(0.55, 0.92, tile_var2)[..., None]
        glaze = peacock * (1.0 - g1) + teal * g1
        glaze = glaze * (1.0 - 0.65 * g2) + jade * (0.65 * g2)
        glaze = glaze * (0.88 + 0.16 * m1[..., None])
        # Thick translucent ceramic glaze pools darker around scale edges (ridge) & gleams bright on dome center!
        glaze = glaze * (1.0 - 0.56 * ridge[..., None]) + deep_rim * (0.56 * ridge[..., None])
        grout = np.array([0.74, 0.72, 0.68], np.float32)[None, None, :] * (0.88 + 0.12 * m2[..., None])
        alb = glaze * (1.0 - cav[..., None]) + grout * cav[..., None]
        # Mirror-wet glazed ceramic domes (~0.08) vs gritty sanded mortar grout (~0.84)
        r_mod = -0.02 * (1.0 - cav) + 0.76 * cav + 0.02 * m1 * (1.0 - cav)

    elif did == "burlap_jute_sackcloth":
        # Authentic raw jute burlap: warm golden-tan twisted fibers + natural deep shadowed weave gaps
        jute_hi = np.array([0.74, 0.60, 0.42], np.float32)[None, None, :]
        jute_mid = np.array([0.52, 0.38, 0.23], np.float32)[None, None, :]
        jute_lo = np.array([0.10, 0.070, 0.040], np.float32)[None, None, :]
        j_mix = np.clip(0.32 + 0.48 * ridge + 0.16 * m1, 0.0, 1.0)[..., None]
        alb = (jute_mid * (1.0 - j_mix) + jute_hi * j_mix) * (1.0 - cav[..., None]) + jute_lo * cav[..., None]
        r_mod = 0.05 * cav - 0.03 * ridge

    elif did == "cast_iron_skillet_pebble":
        # Authentic Lodge seasoned black sand-cast iron: ultra-fine dark ebony polymerized-oil seasoning with subtle warm satin micro-glints!
        pit_c = np.array([0.020, 0.021, 0.024], np.float32)[None, None, :]
        crown_c = np.array([0.072, 0.070, 0.074], np.float32)[None, None, :] * (0.90 + 0.12 * m1[..., None] + 0.4 * chroma_drift)
        alb = pit_c * (1.0 - ridge[..., None]) + crown_c * ridge[..., None]
        alb *= (0.85 + 0.15 * ao[..., None])
        # Seasoned polymerized oil is semi-metallic black cast iron (dark satin crowns ~0.34, matte pores ~0.68)
        met = np.clip(0.72 * (1.0 - 0.58 * cav) + 0.14 * ridge, 0.22, 0.86).astype(np.float32)
        r_mod = -0.09 * ridge + 0.28 * cav + 0.05 * m3

    elif did == "tufted_emerald_velvet":
        # Rich deep jewel-toned Chesterfield emerald velvet: crushed-pile sheen patches + dark folded pinch pleats!
        crush = np.clip(0.5 + 0.45 * m3 + 0.25 * m1, 0.0, 1.0)[..., None]
        em_hi = np.array([0.042, 0.31, 0.195], np.float32)[None, None, :]
        em_mid = np.array([0.020, 0.185, 0.115], np.float32)[None, None, :]
        em_lo = np.array([0.006, 0.048, 0.030], np.float32)[None, None, :]
        pile_c = (em_mid * (1.0 - crush) + em_hi * crush) * (0.94 + 0.06 * m2[..., None])
        alb = em_lo * cav[..., None] + pile_c * (1.0 - cav[..., None])
        alb *= (0.78 + 0.22 * ao[..., None])
        r_mod = -0.05 * ridge + 0.08 * cav + 0.03 * m1

    elif did == "woven_bamboo_tatami":
        # Authentic Japanese igusa rush tatami: warm golden-olive-straw stalks aligned with 22x8 phase-centered geometry!
        cols, rows = 8.0, 22.0
        us = (u + 0.5 / cols) % 1.0
        vs = (v + 0.5 / rows) % 1.0
        stalk_id = np.sin(np.floor(vs * rows) * 23.7 + np.floor(us * cols) * 11.3) * 0.5 + 0.5
        straw = np.array([0.62, 0.54, 0.30], np.float32)[None, None, :]
        igusa = np.array([0.42, 0.46, 0.24], np.float32)[None, None, :]
        gold = np.array([0.70, 0.58, 0.32], np.float32)[None, None, :]
        t1 = np.clip(0.25 + 0.55 * stalk_id + 0.12 * m1, 0.0, 1.0)[..., None]
        alb = igusa * (1.0 - t1) + straw * t1
        alb = alb * (1.0 - 0.22 * ridge[..., None]) + gold * (0.22 * ridge[..., None])
        alb *= (0.78 + 0.22 * ao[..., None])
        r_mod = 0.08 * cav - 0.05 * ridge

    elif did == "woven_rattan_cane":
        # Parisian cafe Thonet woven rattan cane: per-strand warm honey/amber cane cortex + glossy silica sheen + deep weave shadow!
        n = 8.0
        us = (u + 0.5 / n) % 1.0
        vs = (v + 0.5 / n) % 1.0
        s_hash = np.sin(np.floor(us * n) * 17.3 + np.floor(vs * n) * 29.1) * 0.5 + 0.5
        cane_lo = np.array([0.48, 0.34, 0.18], np.float32)[None, None, :]
        cane_hi = np.array([0.74, 0.58, 0.34], np.float32)[None, None, :]
        c_t = np.clip(0.25 + 0.50 * s_hash + 0.25 * ridge + 0.10 * m1, 0.0, 1.0)[..., None]
        cane_col = (cane_lo * (1.0 - c_t) + cane_hi * c_t) * (0.94 + 0.06 * m2[..., None])
        void_c = np.array([0.032, 0.024, 0.016], np.float32)[None, None, :]
        alb = cane_col * (1.0 - 0.88 * cav[..., None]) + void_c * (0.88 * cav[..., None])
        alb *= (0.80 + 0.20 * ao[..., None])
        # Natural rattan silica cortex has crisp satin highlight on cane crowns (~0.32) vs matte recesses (~0.74)
        r_mod = -0.11 * ridge + 0.30 * cav + 0.04 * m1

    elif did == "travertine_porous_stone":
        # Authentic honed Roman travertine: warm cream-ochre calcareous strata + natural gritty mineral cavities!
        ivory = np.array([0.76, 0.68, 0.55], np.float32)[None, None, :]
        honey = np.array([0.58, 0.48, 0.35], np.float32)[None, None, :]
        vein = np.array([0.46, 0.37, 0.26], np.float32)[None, None, :]
        pore = np.array([0.22, 0.175, 0.12], np.float32)[None, None, :] * (0.86 + 0.18 * m2[..., None])
        band1 = (0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 10.0 + 0.18 * m1)))[..., None]
        band2 = (0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 20.0 - 0.14 * m1)))[..., None]
        stone = (honey * (1.0 - band1) + ivory * band1) * (0.92 + 0.08 * band2) * (0.94 + 0.06 * m2[..., None] + 0.6 * chroma_drift)
        stone = stone * 0.90 + vein * 0.10 * (1.0 - band2)
        alb = stone * (1.0 - 0.74 * cav[..., None]) + pore * (0.74 * cav[..., None])
        r_mod = 0.36 * cav - 0.06 * ridge + 0.05 * m1

    elif did == "gilded_tin_ceiling_tile":
        # Authentic warm burnished gold-leaf / antique brass pressed tin with dark umber-soot recess patina
        gold = np.array([0.92, 0.72, 0.34], np.float32)[None, None, :] * (0.94 + 0.06 * m1[..., None] + 0.5 * chroma_drift)
        patina = np.array([0.17, 0.11, 0.055], np.float32)[None, None, :]
        pat_m = gd._smoothstep(0.25, 0.78, cav)
        alb = gold * (1.0 - 0.82 * pat_m[..., None]) + patina * (0.82 * pat_m[..., None])
        met = np.clip(0.97 * (1.0 - 0.86 * pat_m), 0.08, 0.99).astype(np.float32)
        r_mod = -0.06 * ridge + 0.52 * pat_m + 0.04 * m3

    elif did == "chainlink_wire_mesh":
        # Galvanized steel chain-link mesh: Voronoi zinc crystal spangle shimmer + bright knuckle wear + deep open void!
        _, _, zinc = gd._periodic_worley_id(u, v, 42, seed + 39)
        void_c = np.array([0.015, 0.016, 0.019], np.float32)[None, None, :]
        z_lo = np.array([0.44, 0.47, 0.50], np.float32)[None, None, :]
        z_hi = np.array([0.74, 0.77, 0.80], np.float32)[None, None, :]
        wire_col = (z_lo * (1.0 - zinc[..., None]) + z_hi * zinc[..., None]) * (0.94 + 0.06 * m1[..., None])
        wire_col = wire_col * (1.0 + 0.16 * ridge[..., None])
        void_m = gd._smoothstep(0.22, 0.72, cav)
        alb = wire_col * (1.0 - 0.96 * void_m[..., None]) + void_c * (0.96 * void_m[..., None])
        met = np.clip(0.96 * (1.0 - void_m), 0.0, 0.98).astype(np.float32)
        # Galvanized zinc crystal facets vary roughness cleanly (0.16..0.34)
        r_mod = -0.08 * ridge + 0.12 * (0.5 - zinc) * (1.0 - void_m) + 0.65 * void_m

    elif did == "hvac_slotted_vent":
        # Clean satin-anodized architectural aluminum HVAC return-air louver grille: directional horizontal brush sheen + bright louver noses + deep dark plenum (NO camo mottling!)
        void_c = np.array([0.015, 0.016, 0.019], np.float32)[None, None, :]
        alu = np.array([0.56, 0.59, 0.63], np.float32)[None, None, :]
        brush_h = 0.028 * np.sin(2.0 * math.pi * v * 160.0 + 0.35 * m2)
        crest = np.array([0.82, 0.85, 0.89], np.float32)[None, None, :]
        void_m = gd._smoothstep(0.22, 0.72, cav)
        alb = alu * (0.97 + brush_h[..., None] + 0.022 * m1[..., None])
        alb = alb * (1.0 - 0.38 * ridge[..., None]) + crest * (0.38 * ridge[..., None])
        alb = alb * (1.0 - 0.96 * void_m[..., None]) + void_c * (0.96 * void_m[..., None])
        met = np.clip(0.96 * (1.0 - void_m), 0.0, 0.98).astype(np.float32)
        # Clean satin brushed aluminum (~0.24) with crisp bright noses (~0.16) and dark matte open plenum
        r_mod = -0.07 * ridge + 0.018 * np.abs(brush_h) * (1.0 - void_m) + 0.65 * void_m

    elif did in (
        "expanded_metal_mesh", "street_storm_grate",
        "stamped_vent_slots_stagger", "drip_tray_sieve_grate",
        "stamped_louver_micro", "speaker_mesh_hex", "circular_perf_sheet",
    ):
        # Open-void stamped/perforated metals: directional brushed steel + crisp die-punched rim glints + subtle patina + deep open voids!
        void_c = np.array([0.015, 0.016, 0.019], np.float32)[None, None, :]
        crest = np.clip(base * 1.32 + 0.07, 0.0, 0.95)
        void_m = gd._smoothstep(0.22, 0.72, cav)
        # Clean linear brushed metal micro-streaks for stainless/anodized perforated sheets
        if did in ("drip_tray_sieve_grate", "circular_perf_sheet", "speaker_mesh_hex"):
            brush_v = 0.036 * np.sin(2.0 * math.pi * v * 136.0 + 0.45 * m2)
        else:
            brush_v = 0.022 * m2
        spots = gd._smoothstep(0.75, 1.45, m3 + 0.45 * m1) if did in ("drip_tray_sieve_grate", "stamped_vent_slots_stagger", "street_storm_grate") else 0.0
        alb = base * (0.95 + 0.035 * m1[..., None] + brush_v[..., None] + 0.45 * chroma_drift)
        alb = alb * (1.0 - 0.08 * np.asarray(spots)[..., None])
        alb = alb * (1.0 - 0.44 * ridge[..., None]) + crest * (0.44 * ridge[..., None])
        alb = alb * (1.0 - 0.96 * void_m[..., None]) + void_c * (0.96 * void_m[..., None])
        met = np.clip(0.96 * (1.0 - void_m), 0.0, 0.98).astype(np.float32)
        r_mod = -0.10 * ridge + 0.09 * np.asarray(spots) * (1.0 - void_m) + 0.035 * m1 * (1.0 - void_m) + 0.65 * void_m

    elif did == "perforated_leather_auto":
        # Luxury dark charcoal Nappa automotive leather: plump rounded dark charcoal pebble domes + satin punched-rim bevels + deep dark bores!
        void_c = np.array([0.008, 0.009, 0.010], np.float32)[None, None, :]
        leath_lo = np.array([0.014, 0.015, 0.017], np.float32)[None, None, :]
        leath_hi = np.array([0.042, 0.044, 0.048], np.float32)[None, None, :]
        hole_m = gd._smoothstep(0.45, 0.86, cav)
        crease_m = np.clip(cav * (1.0 - hole_m) * 1.8, 0.0, 1.0)
        l_t = np.clip(0.22 + 0.68 * ridge - 0.28 * crease_m + 0.08 * m1, 0.0, 1.0)[..., None]
        alb = (leath_lo * (1.0 - l_t) + leath_hi * l_t) * (0.84 + 0.16 * ao[..., None])
        alb = alb * (1.0 - 0.96 * hole_m[..., None]) + void_c * (0.96 * hole_m[..., None])
        # Crisp tactile leather sheen: satin rounded pebble crowns & punch bevels (~0.21) vs matte soft creases (~0.48) & dark holes
        r_mod = 0.52 * hole_m + 0.18 * crease_m - 0.12 * ridge + 0.025 * m3

    elif did == "acoustic_pyramid_foam":
        # Open-cell charcoal polyurethane acoustic foam: ultra-dark matte pores + tiny lighter torn cell-wall micro-glints
        foam_lo = np.array([0.045, 0.048, 0.055], np.float32)[None, None, :]
        foam_hi = np.array([0.135, 0.142, 0.158], np.float32)[None, None, :]
        alb = foam_lo * (1.0 - 0.72 * ridge[..., None]) + foam_hi * (0.72 * ridge[..., None])
        alb *= (0.74 + 0.26 * ao[..., None]) * (0.94 + 0.06 * m1[..., None])
        r_mod = 0.08 * cav - 0.04 * ridge

    elif did == "keyboard_keycap_grid":
        # Sculpted charcoal PBT keycaps: matte stippled flanks + subtle glossy finger-oil shine pooled in concave center dish!
        n = 6.0
        fu = (u * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        oil_dish = np.exp(-((fu * fu + fv * fv) / 0.065)) * (1.0 - cav) * (0.55 + 0.45 * np.clip(0.5 + 0.5 * m3, 0.0, 1.0))
        cap_c = np.array([0.14, 0.148, 0.165], np.float32)[None, None, :] * (0.95 + 0.05 * m1[..., None])
        gap_c = np.array([0.025, 0.027, 0.032], np.float32)[None, None, :]
        alb = cap_c * (1.0 - 0.85 * cav[..., None]) + gap_c * (0.85 * cav[..., None]) + 0.025 * ridge[..., None]
        # Realistic keycap wear: matte PBT (~0.46) with glossy finger-oil center dish (~0.18)!
        r_mod = 0.10 - 0.24 * oil_dish + 0.24 * cav - 0.05 * ridge

    elif did == "modular_stud_grid":
        # High-gloss injection-molded cherry-crimson ABS plastic studs: mirror-gloss clearcoat specular + subtle handling micro-scuff breakup + deep stud-collar AO!
        abs_c = np.array([0.76, 0.115, 0.105], np.float32)[None, None, :] * (0.96 + 0.04 * m1[..., None])
        alb = abs_c * (0.76 + 0.24 * ao[..., None]) + 0.035 * ridge[..., None]
        # True glossy injection-molded ABS (~0.10-0.16 on stud crowns/deck) with faint handling scuff roughness variation (~0.04)
        scuff = gd._smoothstep(0.35, 1.15, m3 + 0.4 * m2)
        r_mod = -0.04 * (1.0 - cav) - 0.03 * ridge + 0.055 * scuff + 0.09 * cav

    elif mat_kind == "rubber":
        # Authentic vulcanized molded rubber & matte kitchen silicone: fine micro-pore speckle, light dusty chalk in recessed grip troughs, worn semi-satin rubber crests!
        _, _, r_pore = gd._periodic_worley_id(u, v, 76, seed + 211)
        if did == "rubber_tread_stair":
            # Deep vulcanized metro-staircase charcoal rubber (not sterile uniform mid-grey!)
            r_base = np.array([0.095, 0.100, 0.110], np.float32)[None, None, :]
        else:
            r_base = base
        dust_c = np.clip(r_base * 1.85 + 0.045, 0.0, 0.42)
        dust_m = np.clip(cav * (0.42 + 0.40 * np.clip(0.5 + 0.5 * m1, 0.0, 1.0)), 0.0, 0.68)
        alb = r_base * (0.86 + 0.14 * r_pore[..., None] + 0.06 * m2[..., None] + 0.4 * chroma_drift)
        alb = alb * (1.0 - 0.38 * dust_m[..., None]) + dust_c * (0.38 * dust_m[..., None])
        alb = alb * (0.74 + 0.26 * ao[..., None]) + 0.025 * ridge[..., None]
        # Strong tactile roughness split: worn semi-satin rubber crests/stud tops (~0.30-0.40) vs chalky matte dusty troughs (~0.78-0.88)!
        r_mod = 0.26 * cav - 0.18 * ridge + 0.05 * m3 + 0.03 * (r_pore - 0.5)

    elif did == "terry_velcro_loops":
        # Authentic plush woven cotton bath-towel terry loops: multi-tone warm ivory/ecru fluffy fiber tufts + deep warm weave shadow pockets!
        _, _, loop_id = gd._periodic_worley_id(u, v, 72, seed + 213)
        t_lo = np.array([0.30, 0.265, 0.215], np.float32)[None, None, :]
        t_hi = np.array([0.76, 0.72, 0.65], np.float32)[None, None, :]
        t_mix = np.clip((1.0 - cav) * 0.72 + ridge * 0.28 + 0.12 * (loop_id - 0.5), 0.0, 1.0)[..., None]
        alb = (t_lo * (1.0 - t_mix) + t_hi * t_mix) * (0.93 + 0.07 * m2[..., None] + 0.35 * chroma_drift)
        alb *= (0.72 + 0.28 * ao[..., None])
        r_mod = 0.06 * cav - 0.04 * ridge + 0.03 * m2

    elif did == "heatsink_extrusion_fins":
        # Deep extruded CNC/anodized aluminum heatsink cooling fins: pitch-dark deeply-occluded matte cooling valleys + sleek longitudinal brushed aluminum blades!
        # Purely longitudinal (vertical) extruded die-draw grain along v (u-only harmonics, ZERO horizontal cross-hatching!)
        ext_streak = 0.028 * np.sin(2.0 * math.pi * u * 48.0) + 0.018 * np.sin(2.0 * math.pi * u * 88.0 + 0.7) + 0.012 * np.sin(2.0 * math.pi * u * 128.0 + 1.9)
        chan_dark = np.array([0.008, 0.009, 0.012], np.float32)[None, None, :]
        fin_alu = np.array([0.66, 0.70, 0.76], np.float32)[None, None, :] * (0.98 + ext_streak[..., None])
        crest_hi = np.array([0.82, 0.85, 0.90], np.float32)[None, None, :] * (0.98 + ext_streak[..., None])
        # Deep dark shadowed cooling slots (high trough roughness + ultra-dark metallic F0 kills sky glare inside deep fins while keeping met >= 0.85!)
        chan_m = gd._smoothstep(0.24, 0.76, cav)
        alb = fin_alu * (1.0 - 0.97 * chan_m[..., None]) + chan_dark * (0.97 * chan_m[..., None])
        alb = alb * (1.0 - 0.35 * ridge[..., None]) + crest_hi * (0.35 * ridge[..., None])
        alb *= (0.28 + 0.72 * (ao ** 1.5)[..., None])
        met = np.clip(0.97 - 0.11 * chan_m, 0.85, 0.99).astype(np.float32)
        # Sleek brushed extruded aluminum blades (~0.13) vs deeply occluded dark matte-anodized troughs (~0.72)
        r_mod = -0.05 * ridge + 0.56 * chan_m + 0.015 * np.abs(ext_streak)

    elif did == "embossed_braille_strip":
        # Architectural brushed stainless-steel transit braille railing strip: fine horizontal linear brushed anisotropy + subtle base seam fillet + gleaming polished convex domes!
        brush_h = 0.044 * np.sin(2.0 * math.pi * v * 160.0 + 0.40 * m2) + 0.020 * np.sin(2.0 * math.pi * v * 80.0 - 0.30 * m1)
        steel = np.array([0.65, 0.68, 0.72], np.float32)[None, None, :] * (0.95 + brush_h[..., None] + 0.022 * m1[..., None])
        seam_c = np.array([0.36, 0.38, 0.41], np.float32)[None, None, :]
        dome_hi = np.array([0.92, 0.94, 0.96], np.float32)[None, None, :]
        seam_m = gd._smoothstep(0.22, 0.65, cav)
        alb = steel * (1.0 - 0.32 * seam_m[..., None]) + seam_c * (0.32 * seam_m[..., None])
        alb = alb * (1.0 - 0.44 * ridge[..., None]) + dome_hi * (0.44 * ridge[..., None])
        met = np.clip(0.97 - 0.05 * seam_m, 0.90, 0.99).astype(np.float32)
        # Gleaming finger-polished braille domes (~0.10) over fine satin horizontal brushed stainless steel (~0.23) + subtle base fillet (~0.30)
        r_mod = -0.10 * ridge + 0.08 * seam_m + 0.025 * np.abs(brush_h)

    elif did == "woven_nylon_strap":
        # Tight woven herringbone seatbelt/backpack nylon webbing: deep olive-charcoal with anisotropic twill sheen on weave crests
        n_lo = np.array([0.055, 0.082, 0.064], np.float32)[None, None, :]
        n_hi = np.array([0.145, 0.195, 0.155], np.float32)[None, None, :]
        alb = (n_lo * (1.0 - ridge[..., None]) + n_hi * ridge[..., None]) * (0.92 + 0.08 * m1[..., None])
        alb *= (0.76 + 0.24 * ao[..., None])
        # Nylon webbing has synthetic semi-satin fiber glints on crests (~0.42) vs matte valleys (~0.72)
        r_mod = -0.14 * ridge + 0.16 * cav + 0.03 * m2

    elif mat_kind == "metal":
        # Clean machined/cast metal: bare metallic across decks & grooves + warm/cool mill-scale & handling roughness breakup!
        is_warm_metal = float(base[0, 0, 0]) > float(base[0, 0, 2]) * 1.35
        oil = (np.array([0.16, 0.095, 0.032], np.float32)[None, None, :]) if is_warm_metal else np.array([0.10, 0.10, 0.12], np.float32)[None, None, :]
        crest = np.clip(base * (1.20 if is_warm_metal else 1.26) + (0.02 if is_warm_metal else 0.045), 0.0, 0.96)
        # Tight crevice curve (1-ao)^1.8 prevents wide HBAO penumbras from painting halos on flat decks
        occ_tight = np.power(np.clip(1.0 - ao, 0.0, 1.0), 1.8)
        oil_m = np.clip(occ_tight * 2.2, 0.0, 1.0) * (0.35 + 0.65 * cav)
        oil_str = 0.58 if is_warm_metal else 0.48
        alb = base * (0.95 + 0.04 * m1[..., None] + 0.03 * m2[..., None] + 0.65 * chroma_drift)
        alb = alb * (1.0 - oil_str * oil_m[..., None]) + oil * (oil_str * oil_m[..., None])
        alb = alb * (1.0 - 0.34 * ridge[..., None]) + crest * (0.34 * ridge[..., None])
        met = np.clip(0.96 - 0.10 * oil_m, 0.84, 0.99).astype(np.float32)
        # Realistic micro-handling / machining oil sheen breakup on flat decks without ruining smoothness
        r_mod = -0.08 * ridge + (0.14 if is_warm_metal else 0.16) * oil_m + 0.045 * m3

    else:
        # Clean tactile dielectric fallback with warm/cool organic drift & micro-roughness breakup
        alb = base * (0.94 + 0.04 * m1[..., None] + 0.03 * m2[..., None] + 0.7 * chroma_drift)
        alb = alb * (0.80 + 0.20 * ao[..., None]) + 0.030 * ridge[..., None]
        r_mod = 0.18 * cav - 0.06 * ridge + 0.04 * m3

    # Enforce physically plausible dielectric albedo ceiling (~238 sRGB) while allowing bright metals up to 0.95
    alb_max = np.where(met[..., None] > 0.25, 0.95, 0.855).astype(np.float32)
    alb_clamped = np.clip(alb, 0.015, alb_max).astype(np.float32)
    alb_srgb = np.power(alb_clamped, 1.0 / 2.2)
    return (
        np.clip(alb_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8),
        alb_clamped,
        np.clip(met * 255.0 + 0.5, 0, 255).astype(np.uint8),
        r_mod.astype(np.float32),
    )


def synthesize_hardops_plate(out_dir: Path, size: int = 512) -> Path:
    """Synthesize a pure gunmetal machined HardOps panel using 8-Azimuth HBAO + Bent Normal."""
    H = W = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H

    # Start with clean machined gunmetal steel plate + subtle rolled-steel variation
    h0 = 0.50 + 0.015 * gd._periodic_fbm(H, W, 901, (4, 8, 16))
    N0 = gd._height_to_normal_periodic(h0, slope=10.0)

    # Stack HardOps panel seams + bolt plate + circular lug boss + brushed micro-grain + weld fillets!
    ctx = {"circular": True}
    s1 = ng.node_hardops_plug_stamp(ctx, {"height": h0, "normal": N0}, plug="panel_seams", array_x=2, array_y=2, bevel_radius=0.04, bevel_profile="chamfer", depth=0.55, normal_strength=2.0)
    s2 = ng.node_hardops_plug_stamp(ctx, {"height": s1["height"], "normal": s1["normal"]}, plug="bolt_plate", array_x=2, array_y=2, scale=0.74, bevel_radius=0.042, bevel_profile="round", depth=0.48, normal_strength=1.9)
    s3 = ng.node_hardops_plug_stamp(ctx, {"height": s2["height"], "normal": s2["normal"]}, plug="lug_boss", array_x=2, array_y=2, scale=0.42, bevel_radius=0.048, bevel_profile="round", depth=0.55, normal_strength=2.1)
    # Derive curvature/cavity/edge_wear from clean machined geometry BEFORE brush micro-grain
    n_geom = ng.node_normal_clamp_slope(ctx, {"normal": s3["normal"]}, max_angle_deg=60.0)["normal"]
    curv = ng.node_curvature_multiscale(ctx, {"normal": n_geom}, gain=3.2)
    n_br = ng.node_normal_anisotropic_brush(ctx, {"normal": s3["normal"]}, angle_deg=0.0, freq=72.0, anisotropy=14.0, strength=0.18, seed=902)["normal"]
    n_weld = ng.node_normal_weld_fillet(ctx, {"normal": n_br, "concave": curv["concave"]}, radius=2.6, strength=0.50)["normal"]
    n_fin = ng.node_normal_clamp_slope(ctx, {"normal": n_weld}, max_angle_deg=60.0)["normal"]

    h_n = dp.normalize_height(s3["height"], clip_pct=0.5)
    hbao = ng.node_horizon_bent_normal_ao(ctx, {"height": h_n, "normal": n_fin}, strength=1.22, height_scale=13.0)
    ao = hbao["ao"]

    # Pure CNC-machined gunmetal titanium/steel albedo: brushed metallic sheen, crisp worn-silver chamfers, dark machine-oil pockets
    m1 = gd._periodic_fbm(H, W, 903, (8, 16, 32))
    m3 = gd._periodic_fbm(H, W, 904, (2, 4, 8))
    chroma = np.stack([0.014 * m3, 0.004 * m1, -0.014 * m3], axis=-1).astype(np.float32)
    base = np.array([0.28, 0.31, 0.36], np.float32)[None, None, :] * (0.92 + 0.08 * m1[..., None] + chroma)
    silver = np.array([0.82, 0.85, 0.89], np.float32)[None, None, :]
    oil = np.array([0.06, 0.068, 0.080], np.float32)[None, None, :]
    w_edge = np.clip(curv["convex"] * 0.95, 0.0, 1.0)[..., None]
    w_oil = np.clip(np.power(np.clip(1.0 - ao, 0.0, 1.0), 1.35) * 1.45, 0.0, 1.0)[..., None]
    alb_lin = np.clip(base * (1.0 - w_oil) + oil * w_oil, 0.02, 0.95)
    alb_lin = np.clip(alb_lin * (1.0 - 0.62 * w_edge) + silver * (0.62 * w_edge), 0.02, 0.95)
    alb_u8 = np.clip(np.power(alb_lin, 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    rough = np.clip(0.21 + 0.38 * (1.0 - ao) - 0.10 * curv["convex"] + 0.045 * m3 + 0.025 * m1, 0.09, 0.88).astype(np.float32)
    rough = ng.node_roughness_specular_aa(ctx, {"roughness": rough, "normal": n_fin}, strength=0.25, max_boost=0.08)["roughness"]
    deep_oil = gd._smoothstep(0.45, 0.85, 1.0 - ao)
    metal = np.clip(0.96 * (1.0 - 0.16 * deep_oil) + 0.04 * curv["convex"], 0.78, 0.99).astype(np.float32)
    aniso = ng.node_anisotropy_flow(ctx, {"height": h_n, "ao": ao, "roughness": rough})
    thick = ng.node_thickness_subsurface(ctx, {"height": h_n})
    prev_u8 = render_studio_preview(alb_lin, n_fin, rough, metal, ao)
    rough_u8 = np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)
    met_u8 = np.clip(metal * 255.0 + 0.5, 0, 255).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_maps_parallel([
        (out_dir / "albedo_delit.png", alb_u8),
        (out_dir / "source_512.png", alb_u8),
        (out_dir / "preview.png", prev_u8),
        (out_dir / "height.png", np.clip(h_n * 255.0 + 0.5, 0, 255).astype(np.uint8)),
        (out_dir / "height16.png", np.clip(h_n * 65535.0 + 0.5, 0, 65535).astype(np.uint16)),
        (out_dir / "normal.png", dp.normal_to_u8(n_fin)),
        (out_dir / "bent_normal.png", hbao["bent_normal_u8"]),
        (out_dir / "ao.png", hbao["ao_u8"]),
        (out_dir / "roughness.png", rough_u8),
        (out_dir / "metallic.png", met_u8),
        (out_dir / "orm.png", np.stack([hbao["ao_u8"], rough_u8, met_u8], -1)),
        (out_dir / "curvature.png", curv["curvature_u8"]),
        (out_dir / "cavity.png", curv["cavity_u8"]),
        (out_dir / "edge_wear.png", np.clip(curv["convex"] * 255.0 + 0.5, 0, 255).astype(np.uint8)),
        (out_dir / "anisotropy.png", aniso["anisotropy_u8"]),
        (out_dir / "specular.png", aniso["specular_u8"]),
        (out_dir / "thickness.png", thick["thickness_u8"]),
    ])
    return out_dir


def _synthesize_hardops_weathered(
    out_dir: Path,
    size: int = 512,
    seed: int | None = None,
    wear: float = 0.0,
    *,
    weather_preset: str | None = None,
    dirt: float | None = None,
    wet: float | None = None,
    moss: float | None = None,
    rust: float | None = None,
    dust: float | None = None,
) -> Path:
    """Dispatch `hardops_plate` through `synthesize_material`, applying weathering post-hoc if requested."""
    import tempfile

    from .weathering import weather_material

    w_amt = float(np.clip(wear, 0.0, 1.0))
    wants = weather_preset is not None or w_amt > 1e-4 or any(
        v is not None and float(v) > 1e-4 for v in (dirt, wet, moss, rust, dust)
    )
    if not wants:
        return synthesize_hardops_plate(out_dir, size=size)
    with tempfile.TemporaryDirectory() as td:
        fresh = synthesize_hardops_plate(Path(td) / "hardops_plate", size=size)
        return weather_material(
            fresh, out_dir, preset=weather_preset, wear=(w_amt if w_amt > 1e-4 else None),
            dirt=dirt, wet=wet, moss=moss, rust=rust, dust=dust, material_kind="metal",
            seed=901 if seed is None else int(seed),
        )


def list_materials() -> list[dict[str, str]]:
    """Return metadata for all 500 built-in PBR atlas materials.

    Each item carries `id`, `name`, `description`, `category` (the legacy material kind such as
    `metal` / `wood`, kept for backward compatibility), `atlas_category`, `subcategory` and
    `generator` (`flagship` or `parametric`). The first 68 items are the legacy flagship IDs in
    their original order; full agent metadata lives in `hypbr.atlas.get_material_spec()`.
    """
    from . import atlas_catalog as ac

    by_id = {e["id"]: e for e in ac.get_catalog()}
    legacy = [did for did, *_ in gd.DAILY_CATALOG]
    ordered = legacy + [mid for mid in by_id if mid not in set(legacy)]
    return [
        {
            "id": mid,
            "name": by_id[mid]["name"],
            "description": by_id[mid]["description"],
            "category": by_id[mid]["kind"],
            "atlas_category": by_id[mid]["category"],
            "subcategory": by_id[mid]["subcategory"],
            "generator": by_id[mid]["generator"]["kind"],
        }
        for mid in ordered
    ]


def _resolve_param_kind(family: str, kind: str) -> str:
    """Micro-grain family used by the shared post-pipeline for parametric materials."""
    if family in ("brushed_metal", "metal_cladding"):
        return "brushed"
    if kind in ("concrete", "ceramic", "stone", "plaster"):
        return "mineral"
    return "cast"


def save_maps_parallel(items: list[tuple[Path, np.ndarray]], compress_level: int = 4) -> None:
    """Save multiple lossless PNG maps concurrently (Pillow releases the GIL during zlib)."""
    def _write(pair: tuple[Path, np.ndarray]) -> None:
        p, arr = pair
        Image.fromarray(arr).save(p, compress_level=compress_level)

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(items)))) as pool:
        list(pool.map(_write, items))


@torch.no_grad()
def render_studio_preview(
    alb_lin: np.ndarray,
    N: np.ndarray,
    rough: np.ndarray,
    met: np.ndarray,
    ao: np.ndarray,
) -> np.ndarray:
    """Physically-based 3-light Cook-Torrance GGX + dual-strip studio IBL + micro-sheen preview (MPS/CPU bit-matched)."""
    if dp.DEVICE.type != "cpu" and rough.size >= 262144:
        dev = dp.DEVICE
        alb_t = torch.from_numpy(np.ascontiguousarray(alb_lin, dtype=np.float32)).to(dev)
        n_t = torch.from_numpy(np.ascontiguousarray(N, dtype=np.float32)).to(dev)
        r_t = torch.clamp(torch.from_numpy(np.ascontiguousarray(rough, dtype=np.float32)).to(dev), 0.05, 0.98)
        m3 = torch.from_numpy(np.ascontiguousarray(met, dtype=np.float32)).to(dev).unsqueeze(-1)
        ao3 = torch.from_numpy(np.ascontiguousarray(ao, dtype=np.float32)).to(dev).unsqueeze(-1)

        l1 = torch.tensor([0.48, 0.55, 0.68], dtype=torch.float32, device=dev); l1 = l1 / torch.linalg.norm(l1)
        l2 = torch.tensor([-0.62, -0.25, 0.74], dtype=torch.float32, device=dev); l2 = l2 / torch.linalg.norm(l2)
        l3 = torch.tensor([-0.30, 0.76, 0.58], dtype=torch.float32, device=dev); l3 = l3 / torch.linalg.norm(l3)
        v_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=dev)
        h1 = l1 + v_vec; h1 = h1 / torch.linalg.norm(h1)
        h2 = l2 + v_vec; h2 = h2 / torch.linalg.norm(h2)
        h3 = l3 + v_vec; h3 = h3 / torch.linalg.norm(h3)

        ndotl1 = torch.clamp((n_t * l1).sum(dim=-1), 0.0, 1.0)
        ndotl2 = torch.clamp((n_t * l2).sum(dim=-1), 0.0, 1.0)
        ndotl3 = torch.clamp((n_t * l3).sum(dim=-1), 0.0, 1.0)
        ndoth1 = torch.clamp((n_t * h1).sum(dim=-1), 0.0, 1.0)
        ndoth2 = torch.clamp((n_t * h2).sum(dim=-1), 0.0, 1.0)
        ndoth3 = torch.clamp((n_t * h3).sum(dim=-1), 0.0, 1.0)
        ndotv = torch.clamp(n_t[..., 2], 0.02, 1.0)
        vdoth1 = float(torch.clamp((v_vec * h1).sum(), 0.0, 1.0).item())
        vdoth2 = float(torch.clamp((v_vec * h2).sum(), 0.0, 1.0).item())
        vdoth3 = float(torch.clamp((v_vec * h3).sum(), 0.0, 1.0).item())

        r3 = r_t.unsqueeze(-1)
        diff_col = alb_t * (1.0 - m3)
        f0 = 0.04 * (1.0 - m3) + alb_t * m3
        # Lagarde roughness-attenuated Fresnel for IBL (eliminates milky plastic halo on rough textiles)
        fresnel_ibl = f0 + (torch.maximum(1.0 - r3, f0) - f0) * torch.pow(1.0 - ndotv, 5.0).unsqueeze(-1)

        # High-contrast studio IBL cyclorama: dark studio flag horizon + warm key softbox + cool fill reflector strip
        rx = 2.0 * ndotv * n_t[..., 0]
        ry = 2.0 * ndotv * n_t[..., 1]
        sky_grad = torch.clamp(0.5 + 0.5 * ry, 0.0, 1.0).unsqueeze(-1)
        dark_flag = torch.exp(-((ry + 0.08) / 0.24).square()).unsqueeze(-1)
        box_key = torch.clamp(1.0 - torch.hypot(rx - 0.36, ry - 0.44) * 1.18, 0.0, 1.0).unsqueeze(-1).square()
        box_rim = torch.clamp(1.0 - torch.hypot(rx + 0.56, ry - 0.18) * 1.35, 0.0, 1.0).unsqueeze(-1).square()
        c_gnd = torch.tensor([0.08, 0.09, 0.11], dtype=torch.float32, device=dev)
        c_sky = torch.tensor([0.62, 0.68, 0.78], dtype=torch.float32, device=dev)
        c_key = torch.tensor([2.55, 2.42, 2.20], dtype=torch.float32, device=dev)
        c_rim = torch.tensor([1.05, 1.22, 1.50], dtype=torch.float32, device=dev)
        c_blr = torch.tensor([0.34, 0.35, 0.38], dtype=torch.float32, device=dev)
        env_sharp = ((1.0 - sky_grad) * c_gnd + sky_grad * c_sky) * (1.0 - 0.62 * dark_flag) + box_key * c_key + box_rim * c_rim
        ibl_rad = env_sharp * torch.pow(1.0 - r3, 1.40) + c_blr * torch.pow(r3, 1.15)
        # Metallic multi-scattering energy recovery keeps brushed/etched metals gleaming instead of flat grey
        ms_boost = 1.0 + m3 * r3 * 0.42 * f0
        ibl_energy = (1.18 * torch.pow(1.0 - r3, 1.50) + 0.08 * (1.0 - r3)) * (1.0 - m3) + (0.38 + 0.98 * torch.pow(1.0 - r3, 0.68)) * m3
        spec_ibl = fresnel_ibl * ibl_rad * ibl_energy * ms_boost * (0.26 + 0.74 * ao3)

        # True Cook-Torrance Trowbridge-Reitz GGX 3-point direct microfacet specular
        a = r_t * r_t
        a2 = torch.clamp(a * a, min=1e-4)
        k = (r_t + 1.0).square() * 0.125
        d1 = a2 / (math.pi * (ndoth1 * ndoth1 * (a2 - 1.0) + 1.0).square() + 1e-6)
        d2 = a2 / (math.pi * (ndoth2 * ndoth2 * (a2 - 1.0) + 1.0).square() + 1e-6)
        d3 = a2 / (math.pi * (ndoth3 * ndoth3 * (a2 - 1.0) + 1.0).square() + 1e-6)
        vis1 = 0.25 / ((ndotl1 * (1.0 - k) + k) * (ndotv * (1.0 - k) + k) + 1e-4)
        vis2 = 0.25 / ((ndotl2 * (1.0 - k) + k) * (ndotv * (1.0 - k) + k) + 1e-4)
        vis3 = 0.25 / ((ndotl3 * (1.0 - k) + k) * (ndotv * (1.0 - k) + k) + 1e-4)
        f1 = f0 + (1.0 - f0) * ((1.0 - vdoth1) ** 5)
        f2 = f0 + (1.0 - f0) * ((1.0 - vdoth2) ** 5)
        f3 = f0 + (1.0 - f0) * ((1.0 - vdoth3) ** 5)
        c_l1 = torch.tensor([1.04, 1.00, 0.94], dtype=torch.float32, device=dev)
        c_l2 = torch.tensor([0.84, 0.92, 1.06], dtype=torch.float32, device=dev)
        c_l3 = torch.tensor([0.96, 0.98, 1.03], dtype=torch.float32, device=dev)
        spec_direct = (
            f1 * c_l1 * (d1 * vis1 * ndotl1 * 2.15).unsqueeze(-1)
            + f2 * c_l2 * (d2 * vis2 * ndotl2 * 0.72).unsqueeze(-1)
            + f3 * c_l3 * (d3 * vis3 * ndotl3 * 0.62).unsqueeze(-1)
        ) * ms_boost * (0.32 + 0.68 * ao3)

        # Warm key + cool fill + subtle grazing rim diffuse + physical velvet/fabric micro-sheen on rough dielectrics
        light_diff = (
            torch.tensor([0.21, 0.22, 0.24], dtype=torch.float32, device=dev)
            + c_l1 * (0.84 * ndotl1).unsqueeze(-1)
            + c_l2 * (0.24 * ndotl2).unsqueeze(-1)
            + c_l3 * (0.14 * ndotl3).unsqueeze(-1)
        )
        diffuse = diff_col * light_diff * (0.26 + 0.74 * ao3)
        sheen_w = torch.clamp((r3 - 0.64) / 0.28, 0.0, 1.0) * (1.0 - m3)
        sheen = sheen_w * torch.pow(1.0 - ndotv, 2.8).unsqueeze(-1) * (0.20 + 0.80 * alb_t) * (0.24 + 0.52 * ndotl1.unsqueeze(-1) + 0.34 * ndotl3.unsqueeze(-1)) * 0.36 * (0.35 + 0.65 * ao3)

        col = diffuse + sheen + spec_ibl + spec_direct
        # Smooth filmic ACES-like highlight roll-off
        col = col / (1.0 + 0.28 * col)
        out_u8 = torch.clamp(torch.pow(torch.clamp(col, 0.0, 1.0), 1.0 / 2.2) * 255.0 + 0.5, 0.0, 255.0).to(torch.uint8)
        return out_u8.cpu().numpy()

    L1 = np.array([0.48, 0.55, 0.68], np.float32); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.62, -0.25, 0.74], np.float32); L2 /= np.linalg.norm(L2)
    L3 = np.array([-0.30, 0.76, 0.58], np.float32); L3 /= np.linalg.norm(L3)
    V = np.array([0.0, 0.0, 1.0], np.float32)
    H1 = L1 + V; H1 /= np.linalg.norm(H1)
    H2 = L2 + V; H2 /= np.linalg.norm(H2)
    H3 = L3 + V; H3 /= np.linalg.norm(H3)

    ndotl1 = np.clip((N * L1[None, None, :]).sum(-1), 0.0, 1.0)
    ndotl2 = np.clip((N * L2[None, None, :]).sum(-1), 0.0, 1.0)
    ndotl3 = np.clip((N * L3[None, None, :]).sum(-1), 0.0, 1.0)
    ndoth1 = np.clip((N * H1[None, None, :]).sum(-1), 0.0, 1.0)
    ndoth2 = np.clip((N * H2[None, None, :]).sum(-1), 0.0, 1.0)
    ndoth3 = np.clip((N * H3[None, None, :]).sum(-1), 0.0, 1.0)
    ndotv = np.clip(N[..., 2], 0.02, 1.0)
    vdoth1 = float(np.clip(np.dot(V, H1), 0.0, 1.0))
    vdoth2 = float(np.clip(np.dot(V, H2), 0.0, 1.0))
    vdoth3 = float(np.clip(np.dot(V, H3), 0.0, 1.0))

    m3 = met[..., None]
    r = np.clip(rough, 0.05, 0.98)
    r3 = r[..., None]
    ao3 = ao[..., None]

    # Metal/dielectric energy split & Lagarde roughness-attenuated Fresnel IBL
    diff_col = alb_lin * (1.0 - m3)
    f0 = 0.04 * (1.0 - m3) + alb_lin * m3
    fresnel_ibl = f0 + (np.maximum(1.0 - r3, f0) - f0) * ((1.0 - ndotv) ** 5)[..., None]

    rx = 2.0 * ndotv * N[..., 0]
    ry = 2.0 * ndotv * N[..., 1]
    sky_grad = np.clip(0.5 + 0.5 * ry, 0.0, 1.0)[..., None]
    dark_flag = np.exp(-(((ry + 0.08) / 0.24) ** 2))[..., None]
    box_key = np.clip(1.0 - np.hypot(rx - 0.36, ry - 0.44) * 1.18, 0.0, 1.0)[..., None] ** 2
    box_rim = np.clip(1.0 - np.hypot(rx + 0.56, ry - 0.18) * 1.35, 0.0, 1.0)[..., None] ** 2
    env_sharp = (
        ((1.0 - sky_grad) * np.array([0.08, 0.09, 0.11], np.float32) + sky_grad * np.array([0.62, 0.68, 0.78], np.float32))
        * (1.0 - 0.62 * dark_flag)
        + box_key * np.array([2.55, 2.42, 2.20], np.float32)
        + box_rim * np.array([1.05, 1.22, 1.50], np.float32)
    )
    env_blur = np.array([0.34, 0.35, 0.38], np.float32)[None, None, :]
    ibl_rad = env_sharp * ((1.0 - r3) ** 1.40) + env_blur * (r3 ** 1.15)
    ms_boost = 1.0 + m3 * r3 * 0.42 * f0
    ibl_energy = (1.18 * ((1.0 - r3) ** 1.50) + 0.08 * (1.0 - r3)) * (1.0 - m3) + (0.38 + 0.98 * ((1.0 - r3) ** 0.68)) * m3
    spec_ibl = fresnel_ibl * ibl_rad * ibl_energy * ms_boost * (0.26 + 0.74 * ao3)

    # True Cook-Torrance Trowbridge-Reitz GGX 3-point direct microfacet specular
    a = r * r
    a2 = np.maximum(a * a, 1e-4)
    k = ((r + 1.0) ** 2) * 0.125
    d1 = a2 / (math.pi * ((ndoth1 * ndoth1 * (a2 - 1.0) + 1.0) ** 2) + 1e-6)
    d2 = a2 / (math.pi * ((ndoth2 * ndoth2 * (a2 - 1.0) + 1.0) ** 2) + 1e-6)
    d3 = a2 / (math.pi * ((ndoth3 * ndoth3 * (a2 - 1.0) + 1.0) ** 2) + 1e-6)
    vis1 = 0.25 / ((ndotl1 * (1.0 - k) + k) * (ndotv * (1.0 - k) + k) + 1e-4)
    vis2 = 0.25 / ((ndotl2 * (1.0 - k) + k) * (ndotv * (1.0 - k) + k) + 1e-4)
    vis3 = 0.25 / ((ndotl3 * (1.0 - k) + k) * (ndotv * (1.0 - k) + k) + 1e-4)
    f1 = f0 + (1.0 - f0) * ((1.0 - vdoth1) ** 5)
    f2 = f0 + (1.0 - f0) * ((1.0 - vdoth2) ** 5)
    f3 = f0 + (1.0 - f0) * ((1.0 - vdoth3) ** 5)
    c_l1 = np.array([1.04, 1.00, 0.94], np.float32)
    c_l2 = np.array([0.84, 0.92, 1.06], np.float32)
    c_l3 = np.array([0.96, 0.98, 1.03], np.float32)
    spec_direct = (
        f1 * c_l1 * (d1 * vis1 * ndotl1 * 2.15)[..., None]
        + f2 * c_l2 * (d2 * vis2 * ndotl2 * 0.72)[..., None]
        + f3 * c_l3 * (d3 * vis3 * ndotl3 * 0.62)[..., None]
    ) * ms_boost * (0.32 + 0.68 * ao3)

    light_diff = (
        np.array([0.21, 0.22, 0.24], np.float32)
        + c_l1 * (0.84 * ndotl1)[..., None]
        + c_l2 * (0.24 * ndotl2)[..., None]
        + c_l3 * (0.14 * ndotl3)[..., None]
    )
    diffuse = diff_col * light_diff * (0.26 + 0.74 * ao3)
    sheen_w = np.clip((r3 - 0.64) / 0.28, 0.0, 1.0) * (1.0 - m3)
    sheen = sheen_w * ((1.0 - ndotv) ** 2.8)[..., None] * (0.20 + 0.80 * alb_lin) * (0.24 + 0.52 * ndotl1[..., None] + 0.34 * ndotl3[..., None]) * 0.36 * (0.35 + 0.65 * ao3)

    col = diffuse + sheen + spec_ibl + spec_direct
    col = col / (1.0 + 0.28 * col)
    return np.clip(np.power(np.clip(col, 0.0, 1.0), 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def synthesize_material(
    material_id: str,
    out_dir: Path | str,
    size: int = 512,
    seed: int | None = None,
    wear: float = 0.0,
    *,
    weather_preset: str | None = None,
    dirt: float | None = None,
    wet: float | None = None,
    moss: float | None = None,
    rust: float | None = None,
    dust: float | None = None,
) -> Path:
    """Synthesize a complete seamless 15-map PBR material set into `out_dir`.

    Args:
        material_id: Built-in material ID (see `list_materials()`).
        out_dir: Destination directory.
        size: Square output resolution in pixels (default: 512).
        seed: Optional deterministic seed override.
        wear: Physical weathering intensity in `[0.0, 1.0]` (`0.0`=fresh, `0.5`=lived-in, `1.0`=weathered).
        weather_preset: Optional named multi-layer weathering preset
            (`aged`, `rain`, `mossy`, `rusty`, `dusty`, `coastal`, `abandoned`).
        dirt: Optional crevice dirt/grime intensity in `[0.0, 1.0]`.
        wet: Optional rain & standing puddle moisture intensity in `[0.0, 1.0]`.
        moss: Optional biological moss & lichen growth intensity in `[0.0, 1.0]`.
        rust: Optional gravity-driven rust/verdigris/efflorescence runoff intensity in `[0.0, 1.0]`.
        dust: Optional settled mineral dust/ash intensity in `[0.0, 1.0]`.
    """
    from .weathering import apply_weathering_maps, resolve_weather_params

    from . import atlas_catalog as ac

    catalog_map = {did: (idx, desc, mat_kind, tint) for idx, (did, desc, mat_kind, tint) in enumerate(gd.DAILY_CATALOG)}
    if material_id == "hardops_plate":
        return _synthesize_hardops_weathered(
            Path(out_dir), size=size, seed=seed, wear=wear, weather_preset=weather_preset,
            dirt=dirt, wet=wet, moss=moss, rust=rust, dust=dust,
        )
    par = None
    if material_id in catalog_map:
        idx, _desc, mat_kind, tint = catalog_map[material_id]
        s_default = 2000 + idx * 41
    else:
        par = ac.parametric_spec(material_id)
        if par is None:
            raise KeyError(f"Unknown material_id '{material_id}'. Use list_materials() to see available IDs.")
        mat_kind, tint = par["kind"], tuple(par["params"]["c1"])
        s_default = int(par["seed"])
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    H = W = size
    s_base = s_default if seed is None else seed
    w_amt = float(np.clip(wear, 0.0, 1.0))

    has_multi_weather = (
        weather_preset is not None
        or any(v is not None and float(v) > 1e-4 for v in (dirt, wet, moss, rust, dust))
    )
    wp = (
        resolve_weather_params(
            weather_preset,
            wear=(w_amt if w_amt > 1e-4 else None),
            dirt=dirt,
            wet=wet,
            moss=moss,
            rust=rust,
            dust=dust,
        )
        if has_multi_weather
        else None
    )

    if par is None:
        h0, slope_scale, cav0, base_r, ridge0 = gd.build_daily_detail(material_id, H, W, seed=s_base)
        par_out = None
    else:
        from .atlas_generators import build_parametric_material

        h0, slope_scale, cav0, base_r, ridge0, p_alb_lin, p_met, p_rmod = build_parametric_material(
            par["family"], par["params"], H, W, seed=s_base,
        )
        par_out = (p_alb_lin, p_met, p_rmod)
    h_n = dp.normalize_height(h0, clip_pct=0.5)
    N_geom = gd._height_to_normal_periodic(h0, slope=slope_scale)
    N_geom = ng.node_normal_clamp_slope({"circular": True}, {"normal": N_geom}, max_angle_deg=60.0)["normal"]
    _BRUSHED_METAL_IDS = {
        "brushed_anodized_plate",
        "corrugated_sheet_metal",
        "circular_perf_sheet",
        "louver_vent_slats",
        "diamond_tread_plate",
        "embossed_braille_strip",
        "reeded_coin_milling",
        "sheet_metal_seam",
        "machined_straight_flutes",
        "radiator_louver_fins",
    }
    if par is not None:
        g_kind = _resolve_param_kind(par["family"], mat_kind)
    elif mat_kind in ("concrete", "ceramic", "stone", "plaster"):
        g_kind = "mineral"
    elif material_id in _BRUSHED_METAL_IDS:
        g_kind = "brushed"
    else:
        g_kind = "cast"
    # Keep micro-grain physically calibrated: optical glass, smooth ENIG PCB, glossy molded ABS,
    # hand-planished copper & mirror-polished marble stay silky-crisp; rough minerals/metals get tactile micro-texture
    g_scale_fac = float(np.clip(slope_scale / 10.0, 0.22, 1.0))
    if mat_kind == "glass":
        g_str = 0.006
    elif material_id in ("pcb_solder_mask_traces", "modular_stud_grid", "heatsink_extrusion_fins"):
        g_str = 0.008
    elif material_id in ("hammered_copper_dimple", "damascus_folded_steel", "gilded_tin_ceiling_tile", "carbon_fiber_weave"):
        g_str = 0.016 * g_scale_fac
    elif mat_kind in ("ceramic", "carbon") or (mat_kind == "stone" and slope_scale <= 5.0):
        g_str = 0.020 * g_scale_fac
    elif mat_kind in ("metal", "bronze", "copper", "brass"):
        g_str = 0.036 * g_scale_fac
    else:
        g_str = 0.044 * g_scale_fac
    N = ng.node_micro_grain({"circular": True}, {"normal": N_geom}, kind=g_kind, scale=36.0, strength=g_str + 0.035 * w_amt, seed=s_base + 1000)["normal"]
    N = ng.node_normal_clamp_slope({"circular": True}, {"normal": N}, max_angle_deg=60.0)["normal"]

    # Relief-calibrated HBAO: flat slabs keep ao_mean ~0.92-0.97, deep reliefs keep crisp crevices (~0.62-0.80)
    ao_hs = float(np.clip(slope_scale * 0.68, 1.0, 14.0))
    ao_str = float(np.clip(0.55 + 0.042 * slope_scale, 0.55, 1.22))
    hbao = ng.node_horizon_bent_normal_ao(
        {"circular": True}, {"height": h_n, "normal": N},
        strength=ao_str, height_scale=ao_hs, ao_floor=0.18,
    )
    ao = hbao["ao"]
    curv = ng.node_curvature_multiscale({"circular": True}, {"normal": N_geom}, gain=2.8)
    if par_out is None:
        alb_u8, alb_lin, met_u8, r_mod = _rich_albedo_and_metal(material_id, mat_kind, tint, h_n, cav0, ridge0, ao, seed=s_base + 2000)
    else:
        alb_lin, p_met, r_mod = par_out
        alb_u8 = np.clip(np.power(alb_lin, 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        met_u8 = np.clip(p_met * 255.0 + 0.5, 0, 255).astype(np.uint8)

    r_micro = 0.022 * gd._periodic_fbm(H, W, s_base + 3000, (8, 24, 48))
    ao_r_w = 0.08 if mat_kind == "metal" else 0.14
    rough_raw = np.clip(base_r + r_mod + ao_r_w * np.power(np.clip(1.0 - ao, 0.0, 1.0), 1.4) + r_micro, 0.06, 0.94).astype(np.float32)
    # Baked Texel-Level Specular AA (Tokuyoshi/Kaplanyan variance): smoothly widens roughness right on
    # sharp 1-texel normal-gradient ridges so baked PBR maps stay sparkle-free in ANY 3D engine (glTF/Blender/Three.js)
    dn_x = np.roll(N, -1, axis=1) - np.roll(N, 1, axis=1)
    dn_y = np.roll(N, -1, axis=0) - np.roll(N, 1, axis=0)
    n_var = 0.0625 * (np.sum(dn_x * dn_x, axis=-1) + np.sum(dn_y * dn_y, axis=-1))
    k_aa = np.minimum(0.32 * n_var, 0.032).astype(np.float32)
    rough = np.clip(np.sqrt(rough_raw * rough_raw + k_aa), 0.06, 0.94).astype(np.float32)
    met_f = met_u8.astype(np.float32) / 255.0

    weather_mask_rgba = None
    puddle_mask = None
    if wp is not None:
        w_res = apply_weathering_maps(
            alb_lin, N, h_n, rough, met_f, ao, curv,
            wear=wp["wear"], dirt=wp["dirt"], wet=wp["wet"],
            moss=wp["moss"], rust=wp["rust"], dust=wp["dust"],
            material_kind=mat_kind, material_id=material_id, seed=s_base,
        )
        alb_lin = w_res["alb_lin"]
        alb_u8 = w_res["alb_u8"]
        N = w_res["N"]
        rough = w_res["rough"]
        met_f = w_res["met_f"]
        met_u8 = w_res["met_u8"]
        weather_mask_rgba = w_res["weather_mask_rgba"]
        puddle_mask = w_res["masks"].get("puddle")
    elif w_amt > 1e-4:
        w_macro = np.clip(0.5 + 0.42 * gd._periodic_fbm(H, W, s_base + 7777, (4, 8, 16)), 0.0, 1.0)
        w_fine = np.clip(0.5 + 0.46 * gd._periodic_fbm(H, W, s_base + 8888, (16, 32, 64)), 0.0, 1.0)
        w_micro = np.clip(0.5 + 0.48 * gd._periodic_fbm(H, W, s_base + 9999, (32, 64, 128)), 0.0, 1.0)

        # Geometry-anchored crevice/basin mask: settles in grout, dimples, and occluded lug bases
        # while keeping flat unoccluded faces & raised crests free of blurry airbrushed clouds.
        crevice = np.clip(
            0.42 * (1.0 - ao) + 0.34 * (1.0 - curv["cavity"]) + 0.24 * curv["concave"],
            0.0, 1.0,
        )
        # Subtle outward creep onto recessed/mid zones only at higher wear intensities
        basin = np.clip(1.0 - h_n, 0.0, 1.0)
        spread = (0.14 * (w_amt ** 1.35) * w_macro + 0.28 * w_amt * basin) * (1.0 - 0.75 * curv["convex"])
        drive = crevice * (0.95 + 0.40 * w_macro) + spread + 0.26 * w_micro - 0.32 * curv["convex"]
        thresh = 0.62 - 0.38 * w_amt
        patina = (gd._smoothstep(thresh - 0.055, thresh + 0.075, drive) * np.clip(0.75 + 0.25 * w_amt, 0.0, 1.0)).astype(np.float32)

        # Raised convex crests get burnished metal polish (metals) or micro-scuffing (dielectrics)
        edge_scuff = np.clip(curv["convex"] * (0.45 + 0.85 * w_fine) * w_amt * 1.35, 0.0, 0.88)

        # 3D granular corrosion/grime normal pitting blended via RNM inside crusted pockets
        pit_h = (0.55 * w_micro + 0.45 * w_fine).astype(np.float32)
        N_pit = gd._height_to_normal_periodic(pit_h, slope=2.4 * w_amt)
        p_pit, q_pit = ng.normal_to_slopes(N_pit)
        N_pit_m = ng.slopes_to_normal(p_pit * patina, q_pit * patina)
        N = dp.blend_rnm(N, N_pit_m, detail_weight=1.0)
        N = ng.node_normal_clamp_slope({"circular": True}, {"normal": N}, max_angle_deg=60.0)["normal"]

        is_cu = any(k in material_id for k in ("copper", "bronze", "brass")) or mat_kind in ("copper", "bronze", "brass")
        is_metal = (mat_kind in ("metal", "copper", "bronze", "brass", "gold", "steel", "iron")) or (float(met_f.mean()) > 0.35)
        if is_metal:
            if is_cu:
                # Crevice-settled cuprite oxide rim + powdery blue-green verdigris core; crests stay gleaming copper
                cuprite = np.array([0.085, 0.048, 0.032], dtype=np.float32)[None, None, :]
                verd_lo = np.array([0.090, 0.275, 0.225], dtype=np.float32)[None, None, :]
                verd_hi = np.array([0.195, 0.420, 0.340], dtype=np.float32)[None, None, :]
                verd = verd_lo * (1.0 - w_micro[..., None]) + verd_hi * w_micro[..., None]
                core_w = gd._smoothstep(0.32, 0.78, patina * (0.65 + 0.55 * crevice) * (0.55 + 0.45 * w_amt))[..., None]
                tarnish_col = cuprite * (1.0 - core_w) + verd * core_w
            else:
                # 3-stage granular iron/steel rust: dark magnetite oxide + burnt sienna + dusty ochre micro-flecks
                ox_dark = np.array([0.068, 0.042, 0.028], dtype=np.float32)[None, None, :]
                sienna = np.array([0.235, 0.082, 0.028], dtype=np.float32)[None, None, :]
                ochre = np.array([0.390, 0.175, 0.050], dtype=np.float32)[None, None, :]
                rust_t = np.clip(0.25 + 0.60 * w_fine + 0.35 * (w_micro - 0.5), 0.0, 1.0)[..., None]
                rust_warm = sienna * (1.0 - rust_t) + ochre * rust_t
                core_w = gd._smoothstep(0.22, 0.72, patina * (0.55 + 0.55 * w_micro))[..., None]
                tarnish_col = ox_dark * (1.0 - core_w) + rust_warm * core_w

            alb_lin = alb_lin * (1.0 - patina[..., None] * 0.92) + tarnish_col * (patina[..., None] * 0.92)
            # Subtle warm polish on worn high crests
            crest_col = np.clip(alb_lin * 1.18 + 0.03, 0.0, 0.94)
            alb_lin = alb_lin * (1.0 - edge_scuff[..., None] * 0.30) + crest_col * (edge_scuff[..., None] * 0.30)
            # Crisp metallic stripping on oxidized crust (dielectric patina/rust)
            ox_cut = gd._smoothstep(0.22, 0.48, patina)
            met_f = np.clip(met_f * (1.0 - ox_cut * 0.97), 0.0, 1.0).astype(np.float32)
            rough = np.clip(
                rough * (1.0 - patina) + np.clip(0.80 + 0.15 * w_micro, 0.74, 0.96) * patina - 0.12 * edge_scuff * met_f,
                0.08, 0.97,
            ).astype(np.float32)
        else:
            # Dielectric crevice grime / mineral buildup (tightly anchored to grout/bevels, zero flat smoke clouds)
            grime_lo = np.array([0.055, 0.050, 0.044], dtype=np.float32)[None, None, :]
            grime_hi = np.array([0.145, 0.132, 0.112], dtype=np.float32)[None, None, :]
            grime_col = grime_lo * (1.0 - w_micro[..., None]) + grime_hi * w_micro[..., None]
            # Anchor dielectric grime strictly to crevices/cavities + subtle fine bevel wear
            diel_pat = (patina * gd._smoothstep(0.08, 0.35, crevice + 0.25 * w_amt * basin)).astype(np.float32)
            scuff_col = np.clip(alb_lin * 1.18 + 0.05, 0.0, 0.855)
            alb_lin = alb_lin * (1.0 - diel_pat[..., None] * 0.78) + grime_col * (diel_pat[..., None] * 0.78)
            alb_lin = alb_lin * (1.0 - edge_scuff[..., None] * 0.32) + scuff_col * (edge_scuff[..., None] * 0.32)
            rough = np.clip(rough + 0.34 * diel_pat + 0.16 * edge_scuff + 0.04 * w_amt * w_micro, 0.08, 0.97).astype(np.float32)

        alb_u8 = np.clip(np.power(np.clip(alb_lin, 0.0, 1.0), 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        met_u8 = np.clip(met_f * 255.0 + 0.5, 0, 255).astype(np.uint8)

    rough = ng.node_roughness_specular_aa({"circular": True}, {"roughness": rough, "normal": N}, strength=0.25, max_boost=0.08)["roughness"]
    if puddle_mask is not None and float(puddle_mask.max()) > 1e-4:
        rough = np.clip(rough * (1.0 - puddle_mask) + 0.095 * puddle_mask, 0.05, 0.97).astype(np.float32)
    prev_u8 = render_studio_preview(alb_lin, N, rough, met_f, ao)

    aniso = ng.node_anisotropy_flow({"circular": True}, {"height": h_n, "ao": ao, "roughness": rough})
    thick = ng.node_thickness_subsurface({"circular": True}, {"height": h_n})
    rough_u8 = np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)
    orm_u8 = np.stack([hbao["ao_u8"], rough_u8, met_u8], axis=-1)

    outputs = [
        (d / "albedo_delit.png", alb_u8),
        (d / "preview.png", prev_u8),
        (d / "height.png", np.clip(h_n * 255.0 + 0.5, 0, 255).astype(np.uint8)),
        (d / "height16.png", np.clip(h_n * 65535.0 + 0.5, 0, 65535).astype(np.uint16)),
        (d / "normal.png", dp.normal_to_u8(N)),
        (d / "bent_normal.png", hbao["bent_normal_u8"]),
        (d / "ao.png", hbao["ao_u8"]),
        (d / "roughness.png", rough_u8),
        (d / "metallic.png", met_u8),
        (d / "orm.png", orm_u8),
        (d / "curvature.png", curv["curvature_u8"]),
        (d / "cavity.png", curv["cavity_u8"]),
        (d / "edge_wear.png", np.clip(curv["convex"] * 255.0 + 0.5, 0, 255).astype(np.uint8)),
        (d / "anisotropy.png", aniso["anisotropy_u8"]),
        (d / "specular.png", aniso["specular_u8"]),
        (d / "thickness.png", thick["thickness_u8"]),
    ]
    if weather_mask_rgba is not None:
        outputs.append((d / "weather_mask.png", weather_mask_rgba))
    save_maps_parallel(outputs)
    return d


def synthesize_all(
    out_dir: Path | str,
    size: int = 512,
    materials: list[str] | None = None,
    wear: float = 0.0,
    *,
    weather_preset: str | None = None,
    dirt: float | None = None,
    wet: float | None = None,
    moss: float | None = None,
    rust: float | None = None,
    dust: float | None = None,
) -> list[Path]:
    """Synthesize all (or selected) seamless materials into `out_dir/<material_id>/`."""
    root = Path(out_dir)
    ids = materials if materials is not None else [m["id"] for m in list_materials()]
    return [
        synthesize_material(
            mid, root / mid, size=size, wear=wear,
            weather_preset=weather_preset, dirt=dirt, wet=wet, moss=moss, rust=rust, dust=dust,
        )
        for mid in ids
    ]


def main():
    t0 = time.time()
    out_root = Path("./output/pbr")
    paths = synthesize_all(out_root, size=512)
    print(f"Synthesized {len(paths)} seamless PBR materials in {time.time() - t0:.1f}s -> {out_root}")


if __name__ == "__main__":
    main()
