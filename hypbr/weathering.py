"""Multi-mode physical & optional neural PBR weathering engine for hypbr (`hypbr weather`).

Supports 6 orthogonal, 100% periodic seamless weathering layers on any hypbr material
(procedural or photo-derived) plus an optional Single-Encode `DepthPro2-S` neural mask
backend when trained 5.2 MiB decode-branch patches are provided:

  - `wear` : Convex chipped/burnished edge wear (crisp smoothstepped, zero chalky veil)
  - `dirt` : Crevice-locked grime, soot, and recessed soil accumulation
  - `wet`  : Two-regime porous capillary darkening + flattened glossy standing puddles
  - `moss` : Velvety biological moss & pale lichen tufts in damp shaded joints
  - `rust` : Gravity-driven downward runoff streaks, corrosion pitting, verdigris & efflorescence
  - `dust` : Settled mineral dust / ash in recessed horizontal pockets

Outputs a complete seamless 15-map PBR directory plus a packed 4-channel
`weather_mask.png` (`RGBA8`: `R=Wear, G=Dirt/Rust, B=Wet, A=Moss`) for game engines.
"""
from __future__ import annotations

import copy
import gc
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from . import _backend as bk
from ._backend import no_grad
from . import depthpro_pbr as dp
from . import normal_graph as ng
from . import procedural as gd
from .color import linear_to_srgb_u8, srgb_u8_to_linear
from .maps import MapSource, MaterialMaps, unit_normals

WEATHER_PRESETS: dict[str, dict[str, float]] = {
    "aged":      {"wear": 0.45, "dirt": 0.45, "wet": 0.00, "moss": 0.00, "rust": 0.15, "dust": 0.25},
    "rain":      {"wear": 0.10, "dirt": 0.25, "wet": 0.80, "moss": 0.00, "rust": 0.00, "dust": 0.00},
    "mossy":     {"wear": 0.20, "dirt": 0.45, "wet": 0.35, "moss": 0.80, "rust": 0.00, "dust": 0.00},
    "rusty":     {"wear": 0.35, "dirt": 0.30, "wet": 0.00, "moss": 0.00, "rust": 0.88, "dust": 0.00},
    "dusty":     {"wear": 0.25, "dirt": 0.30, "wet": 0.00, "moss": 0.00, "rust": 0.00, "dust": 0.80},
    "coastal":   {"wear": 0.35, "dirt": 0.35, "wet": 0.60, "moss": 0.35, "rust": 0.72, "dust": 0.00},
    "abandoned": {"wear": 0.55, "dirt": 0.55, "wet": 0.40, "moss": 0.60, "rust": 0.78, "dust": 0.25},
}


# Below its knee a channel's output is (intensity / knee) x its output at the knee, so a trace does almost
# nothing and the response rises linearly into the original curve, which is unchanged from the knee up.
# Knees sit at or below the lowest value any preset uses (except rust in "aged"), so presets keep their look.
FADE_IN_KNEES: dict[str, float] = {"wear": 0.35, "dirt": 0.25, "wet": 0.35, "moss": 0.35, "rust": 0.4, "dust": 0.25}


def resolve_weather_params(
    preset: str | None = None,
    *,
    wear: float | None = None,
    dirt: float | None = None,
    wet: float | None = None,
    moss: float | None = None,
    rust: float | None = None,
    dust: float | None = None,
) -> dict[str, float]:
    """Resolve an optional named weathering preset merged with explicit per-channel overrides."""
    params = {"wear": 0.0, "dirt": 0.0, "wet": 0.0, "moss": 0.0, "rust": 0.0, "dust": 0.0}
    if preset is not None:
        key = preset.strip().lower()
        if key not in WEATHER_PRESETS:
            raise ValueError(
                f"Unknown weathering preset {preset!r}. Available: {', '.join(sorted(WEATHER_PRESETS))}"
            )
        params.update(WEATHER_PRESETS[key])
    for k, val in (("wear", wear), ("dirt", dirt), ("wet", wet), ("moss", moss), ("rust", rust), ("dust", dust)):
        if val is not None:
            params[k] = float(np.clip(val, 0.0, 1.0))
    return params

WEATHER_ARCHETYPES: tuple[str, ...] = (
    "metal",
    "copper",
    "brick",
    "stone",
    "wood",
    "concrete",
    "plaster",
    "ceramic",
    "natural",
    "fabric",
    "rubber",
    "leather",
    "glass",
)


def infer_material_kind(name: str, met_mean: float = 0.0) -> str:
    """Infer a physical weathering archetype from material name and metallic map mean."""
    s = name.lower()
    if any(k in s for k in ("copper", "bronze", "brass", "verdigris", "gilded", "tin")):
        return "copper"
    if any(k in s for k in ("brick", "terracotta", "clay")):
        return "brick"
    if any(k in s for k in ("wood", "oak", "plank", "parquet", "bamboo", "tatami", "cork", "rattan")):
        return "wood"
    if any(k in s for k in ("bark", "moss", "burlap", "jute")):
        return "natural"
    if any(k in s for k in ("concrete", "sidewalk", "cement", "asphalt", "tarmac")):
        return "concrete"
    if any(k in s for k in ("plaster", "stucco", "drywall")):
        return "plaster"
    if any(k in s for k in ("tile", "ceramic", "mosaic", "subway", "pcb", "solar")):
        return "ceramic"
    if any(k in s for k in ("stone", "cobble", "granite", "marble", "slate", "travertine", "terrazzo", "rock")):
        return "stone"
    if any(k in s for k in ("fabric", "denim", "wool", "velvet", "linen", "tweed", "felt", "corduroy", "nylon", "velcro")):
        return "fabric"
    if any(k in s for k in ("leather",)):
        return "leather"
    if any(k in s for k in ("rubber", "silicone", "foam", "grip", "keycap")):
        return "rubber"
    if any(k in s for k in ("glass", "prism")):
        return "glass"
    if met_mean > 0.35 or any(k in s for k in ("metal", "steel", "iron", "aluminum", "knurl", "grate", "vent", "mesh", "rivet", "roof", "heatsink", "coin", "hardops")):
        return "metal"
    return "stone"


def _periodic_warp_x(field: np.ndarray, shift_px: np.ndarray) -> np.ndarray:
    """100% periodic horizontal bilinear warp so downward streaks meander organically."""
    H, W = field.shape
    x = (np.arange(W, dtype=np.float32)[None, :] + shift_px) % float(W)
    x0 = np.floor(x).astype(np.int32) % W
    x1 = (x0 + 1) % W
    tx = x - np.floor(x)
    rows = np.arange(H, dtype=np.int32)[:, None]
    return (field[rows, x0] * (1.0 - tx) + field[rows, x1] * tx).astype(np.float32)


def _downward_gravity_streaks(
    source: np.ndarray,
    length_frac: float = 0.09,
    meander_field: np.ndarray | None = None,
) -> np.ndarray:
    """100% periodic anisotropic downward runoff transport (resolution-invariant in UV space).

    Uses an exact circular FFT exponential-decay kernel along +Y followed by optional
    periodic lateral meandering so drips look organic and never form rigid stripes.
    """
    H, W = source.shape
    decay_px = max(2.0, float(length_frac) * H)
    y = np.arange(H, dtype=np.float32)
    ky = np.exp(-y / decay_px)
    ky /= np.maximum(ky.sum(), 1e-6)
    src_sm = dp.gaussian_blur_np(source.astype(np.float32), max(0.7, 0.004 * W), circular=True)
    FY = np.fft.rfft(src_sm, axis=0)
    KY = np.fft.rfft(ky)[:, None]
    out = np.fft.irfft(FY * KY, n=H, axis=0).astype(np.float32)
    if meander_field is not None:
        # Lateral organic drip meander (+/- 1.6% of tile width, 100% periodic)
        out = _periodic_warp_x(out, (meander_field - 0.5) * (0.032 * W))
    p99 = float(np.percentile(out, 99.0))
    if p99 > 1e-6:
        out = np.clip(out / p99, 0.0, 1.0)
    return out.astype(np.float32)


# What a coating wears through to (linear base colour, roughness, metallic). Uncoated materials have no
# entry: their worn core is the same material, which the per-kind edge wear already lightens.
_STEEL = ((0.56, 0.57, 0.58), 0.38, 1.0)
_ALUMINIUM = ((0.91, 0.92, 0.92), 0.28, 1.0)
_RED_BRICK = ((0.40, 0.15, 0.09), 0.88, 0.0)
_CONCRETE = ((0.30, 0.29, 0.27), 0.90, 0.0)
_RED_CLAY = ((0.42, 0.18, 0.10), 0.88, 0.0)
_BISQUE = ((0.62, 0.55, 0.44), 0.88, 0.0)
_RATTAN = ((0.42, 0.30, 0.16), 0.80, 0.0)


def coating_substrate(material_id: str, kind: str) -> tuple[tuple[float, float, float], float, float] | None:
    """(base colour, roughness, metallic) exposed when a coated material wears through, else None."""
    mid = (material_id or "").lower()
    k = (kind or "").lower()
    if "anodized" in mid or "anodised" in mid:
        return _ALUMINIUM
    if "armor" in mid or "armour" in mid or ("panel" in mid and k == "rubber"):
        return _STEEL
    painted = "painted" in mid or "epoxy" in mid or "enamel" in mid
    glazed = "glaze" in mid or "glazed" in mid or "gloss" in mid
    if painted and "wicker" in mid:
        return _RATTAN
    if (painted or glazed) and "brick" in mid:
        return _RED_BRICK
    if painted or (glazed and k in ("concrete",)) or "epoxy" in mid:
        return _CONCRETE if k in ("concrete", "stone") else None
    if glazed and k in ("ceramic", "stone"):
        return _RED_CLAY if any(t in mid for t in ("terracotta", "quarry", "barrel", "fishscale")) else _BISQUE
    return None


WEAR_RELIEF_TARGET = 0.55  # top-1% edge-wear drive that every material is lifted toward (never lowered)


def material_grain(texture: np.ndarray, res_scale: float = 1.0) -> np.ndarray:
    """The material's own fine texture (wood grain lines, stone speckle, weave) as a breakup field centred
    on 0.5 with the same spread (std 0.355) as the periodic noise it stands in for.

    `texture` is albedo luminance when available (log, so dark and light materials read alike), else
    height. A ~2px periodic high-pass keeps the fine pattern and drops the shapes of bevels and joints.
    """
    t = np.asarray(texture, dtype=np.float32)
    detail = t - dp.gaussian_blur_np(t, max(1.0, 2.0 * res_scale), circular=True)
    return np.clip(0.5 + 0.355 * detail / max(float(detail.std()), 1e-6), 0.0, 1.0).astype(np.float32)


_RUST_SPOT_SCALES = ((9, 0.22, 0.42, 0.25, 0.70, 5301), (14, 0.20, 0.42, 0.16, 0.55, 5303))  # patches, pits
_MOSS_COLONY_SCALES = ((6, 0.26, 0.50, 0.30, 0.80, 5401), (13, 0.20, 0.44, 0.22, 0.70, 5403))  # colonies, tufts


def _sparse_patches(u: np.ndarray, v: np.ndarray, amount: float, anchor: np.ndarray, w_fine: np.ndarray,
                    w_micro: np.ndarray, seed: int, scales) -> np.ndarray:
    """Sparse periodic patches in [0, 1] from jittered cells at several scales.

    `scales` rows: (cells per tile, radius lo/hi in cell units, on-probability lo/hi, seed offset). Each cell
    turns on with a probability that grows with `amount` and with `anchor` in [0, 1] (e.g. near crevices) and
    gets its own radius; noise-warped coordinates make the outlines irregular instead of round.
    """
    uw = np.mod(u + 0.05 * (w_fine - 0.5), 1.0)
    vw = np.mod(v + 0.05 * (w_micro - 0.5), 1.0)
    out = np.zeros_like(u, dtype=np.float32)
    for n_cells, r_lo, r_hi, p_lo, p_hi, sd in scales:
        f1, _, cid = gd._periodic_worley_id(uw, vw, n_cells, seed + sd)
        p_on = (p_lo + (p_hi - p_lo) * amount) * (0.6 + 0.8 * anchor)
        on = gd._smoothstep(-0.03, 0.03, p_on - cid)
        radius = (r_lo + (r_hi - r_lo) * np.mod(cid * 7.13, 1.0)) * (0.65 + 0.5 * amount)
        t = np.clip((f1 - 0.6 * radius) / np.maximum(0.4 * radius, 1e-4), 0.0, 1.0)  # per-cell radius: inline smoothstep
        out = np.maximum(out, ((1.0 - t * t * (3.0 - 2.0 * t)) * on).astype(np.float32))
    return out


def compute_weathering_masks(
    h_n: np.ndarray,
    N: np.ndarray,
    ao: np.ndarray,
    curv: Mapping[str, np.ndarray],
    *,
    seed: int = 42,
    material_kind: str = "stone",
    wear: float = 0.0,
    dirt: float = 0.0,
    wet: float = 0.0,
    moss: float = 0.0,
    rust: float = 0.0,
    dust: float = 0.0,
    texture: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute exact periodic, resolution-invariant weathering masks in `[0, 1]`.

    `texture` (optional, same shape as `h_n`): the material's own fine pattern, e.g. log albedo
    luminance; edge wear breaks up along it. Height is used when it is not given.
    """
    H, W = h_n.shape
    res_scale = min(H, W) / 512.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / float(W), yy / float(H)

    w_low = np.clip(0.5 + 0.45 * gd._periodic_fbm(H, W, seed + 6666, (2, 4, 8)), 0.0, 1.0)
    w_macro = np.clip(0.5 + 0.42 * gd._periodic_fbm(H, W, seed + 7777, (4, 8, 16)), 0.0, 1.0)
    w_fine = np.clip(0.5 + 0.46 * gd._periodic_fbm(H, W, seed + 8888, (16, 32, 64)), 0.0, 1.0)
    w_micro = np.clip(0.5 + 0.48 * gd._periodic_fbm(H, W, seed + 9999, (32, 64, 128)), 0.0, 1.0)

    kind = material_kind.lower() if material_kind else "stone"
    is_metal = kind in ("metal", "copper", "bronze", "brass", "steel", "iron")

    # Fade-in below each channel's knee: build the mask as at the knee, then scale it by intensity / knee.
    raw_moss, raw_wet = float(moss), float(wet)  # cross-channel coupling keeps the requested values
    fade: dict[str, float] = {}
    eff: dict[str, float] = {}
    for name, val in (("wear", wear), ("dirt", dirt), ("wet", wet), ("moss", moss), ("rust", rust), ("dust", dust)):
        amt = float(np.clip(val, 0.0, 1.0))  # not `v`: that name is the texture-coordinate grid above
        knee = FADE_IN_KNEES[name]
        eff[name], fade[name] = (max(amt, knee), min(1.0, amt / knee)) if amt > 1e-4 else (0.0, 0.0)
    wear, dirt, wet, moss, rust, dust = (eff[n] for n in ("wear", "dirt", "wet", "moss", "rust", "dust"))

    crevice = np.clip(
        0.44 * (1.0 - ao) + 0.34 * (1.0 - curv["cavity"]) + 0.22 * curv["concave"],
        0.0,
        1.0,
    ).astype(np.float32)
    crevice_halo = dp.gaussian_blur_np(crevice, max(1.8, 9.0 * res_scale), circular=True)
    crevice_wide = dp.gaussian_blur_np(crevice, max(3.5, 22.0 * res_scale), circular=True)
    cw_p95 = float(np.percentile(crevice_wide, 95.0))
    crevice_wide_n = np.clip(crevice_wide / max(cw_p95, 0.08), 0.0, 1.0).astype(np.float32)
    basin = np.clip(1.0 - h_n, 0.0, 1.0).astype(np.float32)
    convex = np.clip(curv["convex"], 0.0, 1.0).astype(np.float32)
    flatness = np.clip(N[..., 2], 0.0, 1.0).astype(np.float32)
    relief_gate = float(np.clip((float(h_n.std()) - 0.008) / 0.055, 0.25, 1.0))

    masks: dict[str, np.ndarray] = {
        "w_low": w_low,
        "w_macro": w_macro,
        "w_fine": w_fine,
        "w_micro": w_micro,
        "crevice": crevice,
        "crevice_halo": crevice_halo,
        "basin": basin,
    }

    # 1. Convex Edge Wear (crisp smoothstepped ridge exposure, zero flat-face chalky haze)
    if wear > 1e-4:
        w_amt = float(np.clip(wear, 0.0, 1.0))
        # Breakup follows the material's own fine relief (wood grain, aggregate, weave) more than noise
        grain = material_grain(h_n if texture is None else texture, res_scale)
        # 0.55^2 + 0.835^2 = 1: same mean and spread as w_fine alone, so only the pattern changes
        breakup = 0.5 + 0.55 * (w_fine - 0.5) + 0.835 * (grain - 0.5)
        raw_wear = convex * (0.50 + 0.85 * breakup) * (0.75 + 0.35 * h_n)
        # Low-relief materials (oak planks: top-1% drive ~0.37 vs ~0.5 for brick) never reached the edge
        # threshold; lift their drive toward WEAR_RELIEF_TARGET, at most x2 so polished slabs stay unworn.
        p99 = float(np.percentile(raw_wear, 99.0))
        raw_wear = raw_wear * float(np.clip(WEAR_RELIEF_TARGET / max(p99, 1e-4), 1.0, 2.0))
        m_wear = gd._smoothstep(0.18 - 0.08 * w_amt, 0.62 - 0.14 * w_amt, raw_wear) * w_amt
        masks["wear"] = np.clip(m_wear, 0.0, 1.0).astype(np.float32)
        masks["grain"] = grain
    else:
        masks["wear"] = np.zeros((H, W), dtype=np.float32)

    # 2. Crevice Dirt / Grime (anchored to joints & occluded pockets; heavy industrial oil/soot spreads on abandoned metals)
    if dirt > 1e-4:
        d_amt = float(np.clip(dirt, 0.0, 1.0))
        if is_metal:
            heavy_grime = float(np.clip((d_amt - 0.35) / 0.45, 0.0, 1.0)) * float(np.clip(0.4 + 0.8 * (raw_moss + raw_wet), 0.0, 1.0))
            spread = (
                (0.14 * (d_amt ** 1.3) * w_macro + 0.08 * d_amt * basin) * (1.0 - 0.80 * convex)
                + 0.44 * heavy_grime * (0.55 * (1.0 - w_low) + 0.45 * w_macro) * (1.0 - 0.90 * convex)
            )
            drive = crevice * (0.95 + 0.42 * w_macro) + spread + 0.20 * w_micro - 0.34 * convex
            thresh = 0.60 - 0.34 * d_amt
            m_dirt = gd._smoothstep(thresh - 0.06, thresh + 0.08, drive) * np.clip(0.75 + 0.25 * d_amt, 0.0, 1.0)
            anchor = crevice + 0.55 * heavy_grime * (1.0 - w_low) * (1.0 - convex)
            m_dirt *= gd._smoothstep(0.08, 0.36, anchor)
        else:
            spread = (0.14 * (d_amt ** 1.3) * w_macro + 0.24 * d_amt * basin) * (1.0 - 0.80 * convex)
            drive = crevice * (0.95 + 0.42 * w_macro) + spread + 0.20 * w_micro - 0.34 * convex
            thresh = 0.60 - 0.34 * d_amt
            m_dirt = gd._smoothstep(thresh - 0.06, thresh + 0.08, drive) * np.clip(0.75 + 0.25 * d_amt, 0.0, 1.0)
            m_dirt *= gd._smoothstep(0.08, 0.36, crevice + 0.25 * d_amt * basin)
        masks["dirt"] = np.clip(m_dirt * relief_gate, 0.0, 1.0).astype(np.float32)
    else:
        masks["dirt"] = np.zeros((H, W), dtype=np.float32)

    # 3. Two-Regime Moisture: crisp standing puddles + capillary damp rim
    if wet > 1e-4:
        wt = float(np.clip(wet, 0.0, 1.0))
        if is_metal:
            # Pool water naturally inside concave dimple bowls & between raised lugs (no pinpoint droplet specks)
            puddle_field = 0.56 * basin + 0.30 * (1.0 - w_low) + 0.14 * crevice
            p_thresh = 0.70 - 0.34 * wt
            puddle = gd._smoothstep(p_thresh - 0.04, p_thresh + 0.05, puddle_field) * gd._smoothstep(0.54, 0.86, flatness + 0.28 * basin)
            puddle *= gd._smoothstep(0.28, 0.64, 1.0 - w_low + 0.42 * basin + 0.28 * crevice) * (1.0 - 0.88 * convex)
            droplets = np.zeros((H, W), dtype=np.float32)
        else:
            # Organic rain-pooled standing puddles in low basins & flat slabs with clear wet-rim contrast
            puddle_field = 0.44 * basin + 0.40 * (1.0 - w_low) + 0.16 * crevice
            p_thresh = 0.70 - 0.34 * wt
            puddle = gd._smoothstep(p_thresh - 0.04, p_thresh + 0.05, puddle_field) * gd._smoothstep(0.60, 0.88, flatness)
            droplets = np.zeros((H, W), dtype=np.float32)
        # Capillary wicking halo extends cleanly outside puddle edges
        puddle_wick = dp.gaussian_blur_np(puddle, max(2.0, 9.0 * res_scale), circular=True)
        damp_base = np.clip(
            0.44 * wt * (0.30 + 0.65 * crevice_halo + 0.40 * (1.0 - w_low)) + 1.25 * puddle_wick,
            0.0,
            1.0,
        )
        moisture = np.clip(np.maximum(0.42 * damp_base, puddle * (0.55 + 0.45 * wt)), 0.0, 1.0).astype(np.float32)
        masks["wet"] = moisture
        masks["puddle"] = np.clip(puddle * np.clip(0.65 + 0.35 * wt, 0.0, 1.0), 0.0, 1.0).astype(np.float32)
        masks["damp"] = np.clip(damp_base, 0.0, 1.0).astype(np.float32)
        masks["droplets"] = droplets
    else:
        masks["wet"] = np.zeros((H, W), dtype=np.float32)
        masks["puddle"] = np.zeros((H, W), dtype=np.float32)
        masks["damp"] = np.zeros((H, W), dtype=np.float32)
        masks["droplets"] = np.zeros((H, W), dtype=np.float32)

    # 4. Biological Moss & Lichen Growth (organic Worley cushion islands anchored strictly to crevices)
    if moss > 1e-4:
        ms = float(np.clip(moss, 0.0, 1.0))
        f1_m, _ = gd._periodic_worley(u, v, 16, seed + 5101)
        tuft_cluster = np.clip(1.0 - 0.92 * f1_m, 0.0, 1.0)
        if is_metal:
            # On bare metal, moss clings to recessed lug/joint crevices & expands into dark velvety collars
            moss_habitat = 0.72 * crevice + 0.28 * crevice_halo * (1.0 - 0.85 * convex)
            moss_drive = moss_habitat * (0.72 + 0.72 * w_low) + 0.26 * tuft_cluster * w_fine - 0.42 * convex
            m_thresh = 0.48 - 0.34 * ms
            m_moss = gd._smoothstep(m_thresh - 0.05, m_thresh + 0.08, moss_drive)
            m_moss *= gd._smoothstep(0.08, 0.34, crevice + 0.52 * crevice_halo)
        else:
            # On stone/concrete/wood, moss anchors to joints/pores & creeps into organic cushion islands
            moss_habitat = (
                0.62 * crevice
                + 0.24 * crevice_halo * (0.4 + 0.6 * basin)
                + 0.14 * np.clip(-N[..., 1] * 0.5 + 0.5, 0.0, 1.0) * (1.0 - ao)
            )
            # Colonies (real moss scans: separate tufts, the largest ~10% of the moss), not the whole joint
            # network: outside a colony joint moss is suppressed; inside, heavier growth spreads onto faces.
            colonies = _sparse_patches(u, v, ms, np.clip(0.6 * crevice_wide_n + 0.4 * basin, 0.0, 1.0),
                                       w_fine, w_micro, seed, _MOSS_COLONY_SCALES)
            moss_drive = (moss_habitat * (0.72 + 0.65 * w_low) + 0.28 * tuft_cluster * w_fine - 0.36 * convex
                          + 0.30 * (0.65 + 0.35 * ms) * colonies)
            m_thresh = 0.50 - 0.34 * ms
            m_moss = gd._smoothstep(m_thresh - 0.05, m_thresh + 0.08, moss_drive)
            joint_gate = gd._smoothstep(0.05, 0.28, crevice + 0.58 * crevice_halo)
            # Inside a colony moss creeps out from the joints in a band broken up by noise (real moss hugs
            # joints and edges); it never fills disks in the middle of a face.
            reach = 0.15 + 0.55 * gd._smoothstep(0.10, 1.0, ms)
            creep = gd._smoothstep(1.0 - reach, 1.0, crevice_wide_n * (0.55 + 0.9 * w_fine))
            m_moss *= np.maximum(joint_gate * (0.3 + 0.7 * colonies), colonies * creep)
        masks["moss"] = np.clip(m_moss * np.clip(0.70 + 0.30 * ms, 0.0, 1.0) * relief_gate, 0.0, 1.0).astype(np.float32)
        masks["moss_tuft"] = (0.60 * tuft_cluster + 0.40 * w_micro).astype(np.float32)
    else:
        masks["moss"] = np.zeros((H, W), dtype=np.float32)
        masks["moss_tuft"] = np.zeros((H, W), dtype=np.float32)

    # 5. Sparse Organic Corrosion, Rust Crusts, Efflorescence & Meandering Runoff Streaks
    if rust > 1e-4:
        rs = float(np.clip(rust, 0.0, 1.0))
        # Sparse non-periodic macro nucleation gate breaks any regular lug/tile column alignment!
        nuc_gate = gd._smoothstep(0.42, 0.78, 0.60 * w_low + 0.40 * w_macro)
        seed_src = np.clip(
            crevice * nuc_gate * gd._smoothstep(0.20, 0.60, crevice + 0.25 * w_fine),
            0.0,
            1.0,
        )
        streaks = _downward_gravity_streaks(
            seed_src,
            length_frac=0.05 + 0.09 * rs,
            meander_field=w_macro,
        )
        # Break streaks organically with fine granular noise so zero rigid vertical lines appear
        streak_mod = streaks * gd._smoothstep(0.18, 0.70, 0.55 * w_fine + 0.45 * w_micro) * (0.55 + 0.45 * nuc_gate)
        if kind in ("concrete", "brick", "plaster"):
            # Porous masonry wicks rainwater & dissolved mineral salts well outward from narrow slab joints!
            bloom_drive = (
                0.42 * crevice
                + 0.56 * crevice_wide_n * (0.45 + 0.85 * w_low)
                + 0.26 * w_macro * w_fine
                - 0.22 * convex
            )
            pocket = gd._smoothstep(0.62 - 0.36 * rs, 0.74 - 0.32 * rs, bloom_drive)
            # Also generate broad mineral-leach runoff trails below joints
            leach_streaks = _downward_gravity_streaks(
                crevice * (0.35 + 0.65 * w_low),
                length_frac=0.08 + 0.14 * rs,
                meander_field=w_macro,
            ) * (0.45 + 0.55 * w_fine)
            m_rust = np.clip(np.maximum(pocket, leach_streaks * (0.60 + 0.40 * rs)), 0.0, 1.0)
        elif kind == "wood":
            bloom_drive = (
                0.55 * crevice
                + 0.52 * crevice_wide_n * (0.50 + 0.85 * w_low)
                + 0.26 * w_macro * w_fine
                - 0.26 * convex
            )
            pocket = gd._smoothstep(0.58 - 0.34 * rs, 0.70 - 0.30 * rs, bloom_drive)
            m_rust = np.clip(np.maximum(pocket, streak_mod * (0.65 + 0.35 * rs)), 0.0, 1.0)
        else:
            # Metals/stone: separate corrosion spots and pits (real rust scans: many patches, the largest holding
            # ~10% of the rust) rather than one blob following the joint network; runoff streaks start at spots.
            spots = _sparse_patches(u, v, rs, crevice_wide_n, w_fine, w_micro, seed, _RUST_SPOT_SCALES) * (1.0 - 0.6 * convex)
            spot_streaks = _downward_gravity_streaks(spots * (0.5 + 0.5 * crevice_wide_n), length_frac=0.04 + 0.08 * rs,
                                                     meander_field=w_macro)
            spot_streaks *= gd._smoothstep(0.18, 0.70, 0.55 * w_fine + 0.45 * w_micro)
            m_rust = np.clip(np.maximum(spots, spot_streaks * (0.50 + 0.35 * rs)), 0.0, 1.0)
        masks["rust"] = np.clip(m_rust * np.clip(0.72 + 0.28 * rs, 0.0, 1.0) * relief_gate, 0.0, 1.0).astype(np.float32)
    else:
        masks["rust"] = np.zeros((H, W), dtype=np.float32)

    # 6. Settled Mineral Dust / Ash / Wood Silvering
    if dust > 1e-4:
        ds = float(np.clip(dust, 0.0, 1.0))
        up_facing = gd._smoothstep(0.50, 0.92, flatness)
        if is_metal:
            # On metals, dust settles ONLY inside occluded lug/bolt crevices (never a chalky veil over open bare steel!)
            d_drive = (0.72 * crevice + 0.28 * crevice_halo * w_low) * up_facing + 0.16 * w_fine - 0.35 * convex
            d_thresh = 0.58 - 0.28 * ds
            m_dust = gd._smoothstep(d_thresh - 0.06, d_thresh + 0.08, d_drive) * gd._smoothstep(0.12, 0.42, crevice + 0.4 * crevice_halo)
        elif kind == "wood":
            # Sun-bleached silvering affects exposed weathered wood grain across upper plank faces
            d_drive = (0.35 * (1.0 - crevice) + 0.40 * w_low + 0.25 * w_fine) * up_facing
            d_thresh = 0.64 - 0.34 * ds
            m_dust = gd._smoothstep(d_thresh - 0.08, d_thresh + 0.10, d_drive)
        else:
            d_drive = (0.42 * basin + 0.38 * crevice + 0.20 * w_low) * up_facing + 0.18 * w_fine - 0.30 * convex
            d_thresh = 0.56 - 0.32 * ds
            m_dust = gd._smoothstep(d_thresh - 0.08, d_thresh + 0.10, d_drive)
        masks["dust"] = np.clip(m_dust * np.clip(0.60 + 0.40 * ds, 0.0, 1.0), 0.0, 1.0).astype(np.float32)
    else:
        masks["dust"] = np.zeros((H, W), dtype=np.float32)

    for name, keys in (("wear", ("wear",)), ("dirt", ("dirt",)), ("wet", ("wet", "puddle", "damp", "droplets")),
                       ("moss", ("moss",)), ("rust", ("rust",)), ("dust", ("dust",))):
        if 0.0 < fade[name] < 1.0:
            for key in keys:
                masks[key] = (masks[key] * fade[name]).astype(np.float32)
    # Edge wear is thresholded again when applied; that step needs the mask as built at the knee.
    masks["wear_fade"] = np.float32(fade["wear"] if fade["wear"] > 0.0 else 1.0)
    return masks


def apply_weathering_maps(
    alb_lin: np.ndarray,
    N: np.ndarray,
    h_n: np.ndarray,
    rough: np.ndarray,
    met_f: np.ndarray,
    ao: np.ndarray,
    curv: Mapping[str, np.ndarray],
    *,
    wear: float = 0.0,
    dirt: float = 0.0,
    wet: float = 0.0,
    moss: float = 0.0,
    rust: float = 0.0,
    dust: float = 0.0,
    material_kind: str = "stone",
    material_id: str = "",
    seed: int = 42,
    custom_masks: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Apply physical multi-mode weathering layers onto linear PBR maps.

    Returns updated `{"alb_lin", "alb_u8", "N", "rough", "met_f", "met_u8", "weather_mask", "weather_mask_rgba", "masks"}`
    (`weather_mask` is the float `(H, W, 4)` RGBA mask that `weather_mask_rgba` quantizes).
    """
    alb = alb_lin.astype(np.float32).copy()
    N_out = N.astype(np.float32).copy()
    r_out = rough.astype(np.float32).copy()
    m_out = met_f.astype(np.float32).copy()

    kind = material_kind.lower() if material_kind else infer_material_kind(material_id, float(m_out.mean()))
    lum = 0.2126 * alb[..., 0] + 0.7152 * alb[..., 1] + 0.0722 * alb[..., 2]
    masks = compute_weathering_masks(
        h_n, N_out, ao, curv,
        seed=seed, material_kind=kind,
        wear=wear, dirt=dirt, wet=wet, moss=moss, rust=rust, dust=dust,
        texture=np.log(np.maximum(lum, 1e-3)).astype(np.float32),
    )
    if custom_masks:
        for k in ("wear", "dirt", "wet", "moss", "rust", "dust"):
            if k in custom_masks and custom_masks[k] is not None:
                arr = np.asarray(custom_masks[k], dtype=np.float32)
                if arr.shape != h_n.shape:
                    arr = np.asarray(
                        Image.fromarray(arr).resize((h_n.shape[1], h_n.shape[0]), Image.BILINEAR),
                        dtype=np.float32,
                    )
                scale = float(locals()[k]) if float(locals()[k]) > 1e-4 else 1.0
                masks[k] = np.clip(arr * scale, 0.0, 1.0).astype(np.float32)
        if custom_masks.get("wear") is not None:
            masks["wear_fade"] = np.float32(1.0)
        if "wet" in custom_masks and custom_masks["wet"] is not None:
            mw = masks["wet"]
            masks["puddle"] = gd._smoothstep(0.42, 0.85, mw)
            masks["damp"] = np.clip(mw * 1.45, 0.0, 1.0).astype(np.float32)

    w_low = masks["w_low"]
    w_fine = masks["w_fine"]
    w_micro = masks["w_micro"]
    crevice = masks["crevice"]

    is_cu = kind in ("copper", "bronze", "brass") or any(k in material_id.lower() for k in ("copper", "bronze", "brass"))
    is_metal = kind in ("metal", "copper", "bronze", "brass", "steel", "iron") or float(m_out.mean()) > 0.35

    # --- Layer 1: Convex Edge Wear & Chipping (crisp thresholded, zero global desaturation!) ---
    m_wear = masks["wear"]
    if float(m_wear.max()) > 1e-4:
        w_fade = float(masks.get("wear_fade", 1.0))
        wear_crisp = gd._smoothstep(0.16, 0.72, m_wear / max(w_fade, 1e-6)) * w_fade
        if is_metal:
            # Gleaming burnished exposed bare metal along sharp beveled crests
            burnish = np.clip(alb * 1.24 + 0.04, 0.0, 0.95)
            alb = alb * (1.0 - 0.38 * wear_crisp[..., None]) + burnish * (0.38 * wear_crisp[..., None])
            r_out = np.clip(r_out - 0.16 * wear_crisp * np.clip(m_out, 0.2, 1.0), 0.06, 0.97)
        elif kind == "wood":
            # Sun-bleached / splintered lighter pale grain along plank edges
            pale_wood = np.clip(alb * 1.28 + np.array([0.07, 0.06, 0.045], np.float32)[None, None, :], 0.0, 0.82)
            alb = alb * (1.0 - 0.45 * wear_crisp[..., None]) + pale_wood * (0.45 * wear_crisp[..., None])
            r_out = np.clip(r_out + 0.14 * wear_crisp, 0.08, 0.97)
        elif kind in ("brick", "stone", "concrete", "plaster", "ceramic"):
            # Chipped lighter mineral core along corners & bevel edges
            lum = 0.2126 * alb[..., 0:1] + 0.7152 * alb[..., 1:2] + 0.0722 * alb[..., 2:3]
            chipped = np.clip(0.55 * alb + 0.45 * lum + 0.09, 0.0, 0.84)
            alb = alb * (1.0 - 0.48 * wear_crisp[..., None]) + chipped * (0.48 * wear_crisp[..., None])
            r_out = np.clip(r_out + 0.22 * wear_crisp, 0.08, 0.97)
        else:
            # Subtle fabric/leather/rubber abrasion along high seams
            scuff = np.clip(alb * 1.16 + 0.03, 0.0, 0.82)
            alb = alb * (1.0 - 0.28 * wear_crisp[..., None]) + scuff * (0.28 * wear_crisp[..., None])
            r_out = np.clip(r_out + 0.12 * wear_crisp, 0.08, 0.97)

        substrate = coating_substrate(material_id, kind)
        if substrate is not None:
            # Chips through the coating: crisp, patchy cores inside the worn band (outlines follow the material's
            # texture plus fine noise), showing what lies underneath, with a slight step at the coating's edge.
            col, s_rough, s_metal = substrate
            unfaded = wear_crisp / max(w_fade, 1e-6)
            breakup = masks.get("grain", np.full_like(unfaded, 0.5))
            chip = gd._smoothstep(0.42, 0.58, unfaded * (0.55 + 0.45 * breakup + 0.55 * w_micro)) * w_fade
            if float(chip.max()) > 1e-4:
                c3 = chip[..., None]
                alb = alb * (1.0 - c3) + np.array(col, np.float32)[None, None, :] * c3
                r_out = np.clip(r_out * (1.0 - chip) + s_rough * chip, 0.06, 0.97)
                m_out = np.clip(m_out * (1.0 - chip) + s_metal * chip, 0.0, 1.0)
                N_step = gd._height_to_normal_periodic(-chip.astype(np.float32), slope=1.6)
                ps, qs = ng.normal_to_slopes(N_step)
                N_out = dp.blend_rnm(N_out, ng.slopes_to_normal(ps, qs), detail_weight=1.0)
                N_out = ng.node_normal_clamp_slope({"circular": True}, {"normal": N_out}, max_angle_deg=60.0)["normal"]
            masks["chip"] = chip.astype(np.float32)

    # --- Layer 2: Crevice Dirt & Grime ---
    m_dirt = masks["dirt"]
    if float(m_dirt.max()) > 1e-4:
        grime_lo = np.array([0.048, 0.042, 0.036], dtype=np.float32)[None, None, :]
        grime_hi = np.array([0.135, 0.118, 0.096], dtype=np.float32)[None, None, :]
        grime_col = grime_lo * (1.0 - w_micro[..., None]) + grime_hi * w_micro[..., None]
        d_vis = gd._smoothstep(0.10, 0.78, m_dirt)
        alb = alb * (1.0 - 0.80 * d_vis[..., None]) + grime_col * (0.80 * d_vis[..., None])
        m_out = np.clip(m_out * (1.0 - 0.92 * gd._smoothstep(0.24, 0.56, d_vis)), 0.0, 1.0)
        r_out = np.clip(r_out * (1.0 - d_vis) + np.clip(0.84 + 0.12 * w_micro, 0.78, 0.97) * d_vis, 0.06, 0.97)

    # --- Layer 3: Corrosion, Rust Pitting, Verdigris & Gravity Runoff Streaks ---
    m_rust = masks["rust"]
    if float(m_rust.max()) > 1e-4:
        pit_h = (0.55 * w_micro + 0.45 * w_fine).astype(np.float32)
        N_pit = gd._height_to_normal_periodic(pit_h, slope=2.6 * float(np.clip(rust if rust > 0 else 0.7, 0.2, 1.0)))
        pp, qp = ng.normal_to_slopes(N_pit)
        N_out = dp.blend_rnm(N_out, ng.slopes_to_normal(pp * m_rust, qp * m_rust), detail_weight=0.85)
        N_out = ng.node_normal_clamp_slope({"circular": True}, {"normal": N_out}, max_angle_deg=60.0)["normal"]

        if is_cu:
            cuprite = np.array([0.085, 0.048, 0.032], dtype=np.float32)[None, None, :]
            verd_lo = np.array([0.085, 0.270, 0.220], dtype=np.float32)[None, None, :]
            verd_hi = np.array([0.195, 0.425, 0.340], dtype=np.float32)[None, None, :]
            verd = verd_lo * (1.0 - w_micro[..., None]) + verd_hi * w_micro[..., None]
            core_w = gd._smoothstep(0.26, 0.74, m_rust * (0.65 + 0.45 * crevice))[..., None]
            stain_col = cuprite * (1.0 - core_w) + verd * core_w
            blend_k = 0.90
        elif is_metal:
            # Tones fitted to 8 CC0 rust scans (Lab L 24-50, hue ~64 deg yellow-brown): see tools/weathering_refs.py
            ox_dark = np.array([0.057, 0.017, 0.006], dtype=np.float32)[None, None, :]
            sienna = np.array([0.131, 0.036, 0.007], dtype=np.float32)[None, None, :]
            ochre = np.array([0.229, 0.090, 0.007], dtype=np.float32)[None, None, :]
            rt = np.clip(0.32 + 0.55 * w_fine + 0.30 * (w_micro - 0.5), 0.0, 1.0)[..., None]
            # When combined with moisture/moss (abandoned/coastal), old rust darkens into pitted black-umber scale
            aged_scale = float(np.clip(0.65 * wet + 0.55 * moss, 0.0, 0.72))
            core_w = gd._smoothstep(0.12 + 0.22 * aged_scale, 0.55 + 0.20 * aged_scale, m_rust * (0.65 + 0.55 * w_micro))[..., None]
            stain_col = ox_dark * (1.0 - core_w) + (sienna * (1.0 - rt) + ochre * rt) * core_w
            if aged_scale > 0.1:
                stain_col = stain_col * (1.0 - 0.42 * aged_scale * (0.4 + 0.6 * w_low[..., None]))
            blend_k = 0.96
        elif kind in ("brick", "concrete", "plaster"):
            # Spatially separated dark soot/rainwater runoff trails vs feathered powdery crystalline salt efflorescence!
            drip_dark = np.array([0.048, 0.043, 0.036], dtype=np.float32)[None, None, :]
            warm_rust_stain = np.array([0.165, 0.095, 0.048], dtype=np.float32)[None, None, :]
            efflor = np.array([0.825, 0.815, 0.780], dtype=np.float32)[None, None, :]
            # Powdery feathered crystalline transition using fine+micro salt grain noise
            salt_grain = 0.45 * w_low + 0.35 * w_fine + 0.20 * w_micro
            salt_zone = gd._smoothstep(0.50, 0.80, salt_grain * gd._smoothstep(0.08, 0.45, 1.0 - crevice))[..., None]
            salt_zone = salt_zone * (0.55 + 0.45 * w_micro[..., None])
            iron_tint = gd._smoothstep(0.45, 0.80, w_micro)[..., None]
            dark_stain = drip_dark * (1.0 - 0.45 * iron_tint) + warm_rust_stain * (0.45 * iron_tint)
            stain_col = dark_stain * (1.0 - salt_zone) + efflor * salt_zone
            blend_k = 0.86
        elif kind == "wood":
            # Deep dark iron-tannin / water-rot staining wicking out of plank seams + warm umber grain streaks
            rot_dark = np.array([0.038, 0.028, 0.020], dtype=np.float32)[None, None, :]
            tannin = np.array([0.105, 0.062, 0.036], dtype=np.float32)[None, None, :]
            stain_col = rot_dark * (1.0 - 0.55 * w_fine[..., None]) + tannin * (0.55 * w_fine[..., None])
            blend_k = 0.88
        else:
            # Stone / ceramic mineral iron-oxide & dark calcite runoff trails
            iron_st = np.array([0.24, 0.115, 0.045], dtype=np.float32)[None, None, :]
            dark_st = np.array([0.068, 0.062, 0.055], dtype=np.float32)[None, None, :]
            stain_col = dark_st * (1.0 - 0.60 * w_fine[..., None]) + iron_st * (0.60 * w_fine[..., None])
            blend_k = 0.82

        # Oxide crusts and patina are opaque: full cover from mask 0.6, a linear ramp below (no dead zone)
        r_cover = (blend_k * np.clip(m_rust / 0.6, 0.0, 1.0))[..., None]
        alb = alb * (1.0 - r_cover) + stain_col * r_cover
        m_out = np.clip(m_out * (1.0 - 0.96 * gd._smoothstep(0.20, 0.50, m_rust)), 0.0, 1.0)
        r_out = np.clip(r_out * (1.0 - m_rust) + np.clip(0.82 + 0.14 * w_micro, 0.76, 0.97) * m_rust, 0.06, 0.97)

    # --- Layer 4: Biological Moss & Lichen Growth ---
    m_moss = masks["moss"]
    if float(m_moss.max()) > 1e-4:
        tuft = masks.get("moss_tuft", w_micro)
        # Tones fitted to CC0 moss scans (Lab L 20-46, olive hue ~108 deg): see tools/weathering_refs.py
        moss_dark = np.array([0.026, 0.034, 0.000], dtype=np.float32)[None, None, :]
        moss_mid = np.array([0.060, 0.082, 0.000], dtype=np.float32)[None, None, :]
        lichen_pale = np.array([0.158, 0.166, 0.002], dtype=np.float32)[None, None, :]
        lich_w = gd._smoothstep(0.72, 0.92, w_fine * (0.45 + 0.55 * w_micro))[..., None]
        moss_col = moss_dark * (1.0 - tuft[..., None]) + moss_mid * tuft[..., None]
        moss_col = moss_col * (1.0 - 0.45 * lich_w) + lichen_pale * (0.45 * lich_w)

        # Tactile velvety 3D moss cushion micro-bump blended cleanly into normals
        N_moss = gd._height_to_normal_periodic(tuft, slope=4.2)
        pm, qm = ng.normal_to_slopes(N_moss)
        N_out = dp.blend_rnm(N_out, ng.slopes_to_normal(pm * m_moss, qm * m_moss), detail_weight=0.88)
        N_out = ng.node_normal_clamp_slope({"circular": True}, {"normal": N_out}, max_angle_deg=60.0)["normal"]

        # Moss cushions are opaque: full cover from mask 0.6, a linear ramp below (no threshold, so no dead zone)
        cover = (0.95 * np.clip(m_moss / 0.6, 0.0, 1.0))[..., None]
        alb = alb * (1.0 - cover) + moss_col * cover
        m_out = np.clip(m_out * (1.0 - 0.98 * gd._smoothstep(0.18, 0.48, m_moss)), 0.0, 1.0)
        r_out = np.clip(r_out * (1.0 - m_moss) + np.clip(0.90 + 0.06 * w_micro, 0.87, 0.97) * m_moss, 0.06, 0.97)

    # --- Layer 5: Settled Mineral Dust / Ash / Wood Silvering ---
    m_dust = masks["dust"]
    if float(m_dust.max()) > 1e-4:
        if kind == "wood":
            # Sun-bleached weathered silver-grey dry oak patina contrasting clearly with dark damp rot
            lum = 0.2126 * alb[..., 0:1] + 0.7152 * alb[..., 1:2] + 0.0722 * alb[..., 2:3]
            silver_wood = np.clip(lum * 1.35 + np.array([0.14, 0.145, 0.14], np.float32)[None, None, :], 0.0, 0.62)
            dust_col = silver_wood * (0.88 + 0.20 * w_micro[..., None])
            d_k = 0.82
        else:
            dust_col = np.array([0.46, 0.44, 0.40], dtype=np.float32)[None, None, :] * (0.90 + 0.12 * w_micro[..., None])
            d_k = 0.58
        alb = alb * (1.0 - d_k * m_dust[..., None]) + dust_col * (d_k * m_dust[..., None])
        m_out = np.clip(m_out * (1.0 - 0.85 * gd._smoothstep(0.25, 0.65, m_dust)), 0.0, 1.0)
        r_out = np.clip(r_out * (1.0 - 0.85 * m_dust) + 0.92 * (0.85 * m_dust), 0.06, 0.97)

    # --- Layer 6: Two-Regime Rain, Capillary Rim & Glossy Standing Puddles (always topmost!) ---
    m_wet = masks["wet"]
    if float(m_wet.max()) > 1e-4:
        puddle = masks.get("puddle", gd._smoothstep(0.42, 0.85, m_wet))
        damp = masks.get("damp", np.clip(m_wet * 1.45, 0.0, 1.0))
        droplets = masks.get("droplets", np.zeros_like(puddle))

        # 6a. Porous capillary darkening (strong dark wet rim on porous stone/concrete/wood; clear pooled-water deepening on bare metals!)
        if is_metal or kind == "glass":
            porosity = 0.04
            darken = 1.0 - porosity * np.clip(damp, 0.0, 1.0) - 0.42 * np.clip(puddle, 0.0, 1.0)
            alb = np.power(np.clip(alb, 1e-4, 1.0), 1.0 + 0.36 * np.clip(puddle, 0.0, 1.0)[..., None]) * np.clip(darken, 0.25, 1.0)[..., None]
        elif kind == "ceramic":
            porosity = 0.24
            darken = 1.0 - porosity * np.clip(damp, 0.0, 1.0)
            alb = np.power(np.clip(alb, 1e-4, 1.0), 1.0 + 0.22 * porosity * damp[..., None]) * np.clip(darken, 0.25, 1.0)[..., None]
        elif kind == "wood":
            porosity = 0.56
            darken = 1.0 - porosity * np.clip(damp, 0.0, 1.0)
            alb = np.power(np.clip(alb, 1e-4, 1.0), 1.0 + 0.22 * porosity * damp[..., None]) * np.clip(darken, 0.25, 1.0)[..., None]
        else:
            porosity = 0.50
            darken = 1.0 - porosity * np.clip(damp, 0.0, 1.0)
            alb = np.power(np.clip(alb, 1e-4, 1.0), 1.0 + 0.22 * porosity * damp[..., None]) * np.clip(darken, 0.25, 1.0)[..., None]

        # 6b. Puddle meniscus edge bevel + gentle surface capillary ripples + raindrop beads!
        # Flattening puddle normals with subtle capillary ripple + sharp meniscus catches crisp glossy specular glints
        # instead of turning into a dead flat mirror staring into a dark overhead flag.
        N_meniscus = gd._height_to_normal_periodic(puddle, slope=5.8 if is_metal else 4.8)
        N_ripple = gd._height_to_normal_periodic(0.65 * w_fine + 0.35 * w_low, slope=1.15)
        pm_r, qm_r = ng.normal_to_slopes(N_meniscus)
        pr_r, qr_r = ng.normal_to_slopes(N_ripple)
        # Water surface normal: nearly horizontal (N_z > 0.975) with crisp curved shoreline meniscus + gentle wind ripple
        edge_ring = np.clip(4.0 * puddle * (1.0 - puddle), 0.0, 1.0)
        N_water = ng.slopes_to_normal(
            pm_r * edge_ring * 0.90 + pr_r * puddle * 0.40,
            qm_r * edge_ring * 0.90 + qr_r * puddle * 0.40,
        )
        flat_blend = 0.92 * puddle if is_metal else 0.86 * puddle
        N_out = ng.normalize_n(N_out * (1.0 - flat_blend[..., None]) + N_water * flat_blend[..., None])

        # Add subtle curved raindrop micro-beads on wet non-puddled surfaces (soft slope avoids harsh white salt specks on metal)
        if float(droplets.max()) > 1e-4:
            drop_slope = 2.0 if (is_metal or kind in ("ceramic", "glass")) else 1.8
            N_drop = gd._height_to_normal_periodic(droplets, slope=drop_slope)
            pd, qd = ng.normal_to_slopes(N_drop)
            drop_mask = gd._smoothstep(0.08, 0.45, droplets)
            N_out = dp.blend_rnm(N_out, ng.slopes_to_normal(pd * drop_mask, qd * drop_mask), detail_weight=0.60)
            N_out = ng.node_normal_clamp_slope({"circular": True}, {"normal": N_out}, max_angle_deg=60.0)["normal"]

        # 6c. Glossy wet-film specular response (broad clearcoat sheen in damp areas + crisp liquid puddles)
        if is_metal:
            # Flooded puddles pool mirror-smooth water over darkened submerged metal; damp zones stay crisp metallic
            r_damp = r_out * (1.0 - 0.32 * damp)
            r_out = np.clip(r_damp * (1.0 - puddle) + 0.055 * puddle, 0.05, 0.97)
        else:
            r_damp = r_out * (1.0 - 0.54 * damp)
            r_out = np.clip(r_damp * (1.0 - puddle) + 0.105 * puddle, 0.05, 0.97)
            if float(droplets.max()) > 1e-4:
                r_out = np.clip(r_out * (1.0 - 0.55 * gd._smoothstep(0.10, 0.55, droplets)), 0.05, 0.97)

    # Physical dielectric/metallic albedo ceiling enforcement
    alb_max = np.where(m_out[..., None] > 0.25, 0.95, 0.855).astype(np.float32)
    alb = np.clip(alb, 0.015, alb_max).astype(np.float32)
    alb_u8 = linear_to_srgb_u8(alb)
    met_u8 = np.clip(m_out * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # Packed 4-channel RGBA8 weather mask for real-time engine shaders (R=Wear, G=Dirt|Rust, B=Wet, A=Moss)
    g_chan = np.maximum(masks["dirt"], masks["rust"])
    weather_mask = np.stack([masks["wear"], g_chan, masks["wet"], masks["moss"]], axis=-1)
    weather_mask_rgba = np.stack([
        np.clip(masks["wear"] * 255.0 + 0.5, 0, 255),
        np.clip(g_chan * 255.0 + 0.5, 0, 255),
        np.clip(masks["wet"] * 255.0 + 0.5, 0, 255),
        np.clip(masks["moss"] * 255.0 + 0.5, 0, 255),
    ], axis=-1).astype(np.uint8)

    return {
        "alb_lin": alb,
        "alb_u8": alb_u8,
        "N": N_out.astype(np.float32),
        "rough": r_out.astype(np.float32),
        "met_f": m_out.astype(np.float32),
        "met_u8": met_u8,
        "weather_mask": weather_mask,
        "weather_mask_rgba": weather_mask_rgba,
        "masks": masks,
    }


# =============================================================================
# Optional Single-Encode Neural Weathering Backend (DepthPro2-S + 5.2 MiB heads)
# =============================================================================
_WEATHER_BRANCH_CLS = None


def _weather_decode_branch_cls():
    """Build the torch `nn.Module` lazily so importing hypbr never requires torch."""
    global _WEATHER_BRANCH_CLS
    if _WEATHER_BRANCH_CLS is not None:
        return _WEATHER_BRANCH_CLS
    torch = bk.require_torch("Neural weathering (`--backend neural`)")
    nn = torch.nn

    class _WeatherDecodeBranch(nn.Module):
        """Lightweight (~5.2 MiB fp16) single-channel DPT decode branch sharing 1 ViT encode."""

        def __init__(self, reassemble: nn.ModuleList, decoder: nn.Module, head: nn.Module):
            super().__init__()
            self.reassemble = copy.deepcopy(reassemble)
            self.decoder = copy.deepcopy(decoder)
            self.head = copy.deepcopy(head)

        @torch.no_grad()
        def load_patch_dict(self, patch_sd: Mapping[str, torch.Tensor]) -> int:
            target_maps = [
                ("encoder.reassemble.", self.reassemble.state_dict()),
                ("reassemble.", self.reassemble.state_dict()),
                ("decoder.", self.decoder.state_dict()),
                ("head.", self.head.state_dict()),
            ]
            loaded = 0
            for k, v in patch_sd.items():
                for prefix, sub_sd in target_maps:
                    if k.startswith(prefix):
                        sub_key = k[len(prefix):]
                        if sub_key in sub_sd and sub_sd[sub_key].shape == v.shape:
                            sub_sd[sub_key].copy_(v.to(device=sub_sd[sub_key].device, dtype=sub_sd[sub_key].dtype))
                            loaded += 1
                        break
            return loaded

        def forward(self, features: list[torch.Tensor], grid_size: tuple[int, int]) -> torch.Tensor:
            num_frames = features[0].shape[0]
            pyr = [
                layer(feat.transpose(1, 2).reshape(num_frames, -1, *grid_size).contiguous())
                for layer, feat in zip(self.reassemble, features)
            ]
            dec = self.decoder(pyr).float()
            return torch.sigmoid(self.head[:-1](dec))

    _WEATHER_BRANCH_CLS = _WeatherDecodeBranch
    return _WEATHER_BRANCH_CLS


@no_grad
def predict_neural_weathering_masks(
    rgb_u8: np.ndarray,
    *,
    vit_patch: Path | str | None = None,
    patch_dir: Path | str | None = None,
    branches: tuple[str, ...] = ("wear", "dirt", "moisture"),
) -> dict[str, np.ndarray]:
    """Optional Single-Encode `DepthPro2-S` neural weathering predictor.

    Runs the locked `DepthPro2-S` ViT encoder ONCE and fans features to swappable 5.2 MiB
    `wear` / `dirt` / `moisture` decode heads. Requires optional `depth_pro_2` and trained
    branch patch files (auto-discovered via `patch_dir` or `HYPBR_WEATHER_DIR`).
    """
    pdir = Path(patch_dir) if patch_dir else Path(os.environ.get("HYPBR_WEATHER_DIR", ""))
    if not pdir.is_dir():
        raise FileNotFoundError(
            f"Neural weathering patch directory not found: '{pdir}'. "
            "Pass --patch-dir / set HYPBR_WEATHER_DIR, or use default --backend procedural."
        )

    torch = bk.require_torch("Neural weathering (`--backend neural`)")
    F = torch.nn.functional
    from depth_pro_2.model import INPUT_MULTIPLE, to_normalized_tensor

    base = dp.load_depthpro2_no_fov("depthpro2-s", precision=torch.float32)
    if vit_patch is not None and Path(vit_patch).exists():
        vsd = torch.load(vit_patch, map_location="cpu", weights_only=True)
        enc_sd = base.encoder.state_dict()
        for k, v in vsd.items():
            if not k.startswith("encoder.") or k.startswith("encoder.reassemble."):
                continue
            sub_k = k[len("encoder."):]
            if sub_k in enc_sd and enc_sd[sub_k].shape == v.shape:
                enc_sd[sub_k].copy_(v.to(device=enc_sd[sub_k].device, dtype=enc_sd[sub_k].dtype))
    elif dp.NEON_PATCH.exists():
        dp.apply_neon_wsp_patch(base, circular_pad=False)
    base.eval()

    H, W = rgb_u8.shape[:2]
    ih = max(INPUT_MULTIPLE, (H // INPUT_MULTIPLE) * INPUT_MULTIPLE)
    iw = max(INPUT_MULTIPLE, (W // INPUT_MULTIPLE) * INPUT_MULTIPLE)
    rgb_in = rgb_u8 if (ih == H and iw == W) else np.asarray(
        Image.fromarray(rgb_u8).resize((iw, ih), Image.BICUBIC), dtype=np.uint8
    )
    x = to_normalized_tensor(np.array(rgb_in, dtype=np.uint8, copy=True), dp.DEVICE).unsqueeze(0)
    features, grid_size = base.encoder.encode(x, None)

    out: dict[str, np.ndarray] = {}
    for name in branches:
        # Pick highest versioned patch file matching `{name}_v*_patch.pt` or `{name}_patch.pt`
        candidates = sorted(pdir.glob(f"{name}*_patch.pt"))
        if not candidates:
            continue
        ckpt = candidates[-1]
        branch = _weather_decode_branch_cls()(base.encoder.reassemble, base.decoder, base.head).to(dp.DEVICE).eval()
        bsd = torch.load(ckpt, map_location="cpu", weights_only=True)
        branch.load_patch_dict(bsd)
        pred = branch(features, grid_size)
        if pred.shape[-2:] != (H, W):
            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
        key = "wet" if name == "moisture" else name
        out[key] = np.clip(pred[0, 0].cpu().numpy(), 0.0, 1.0).astype(np.float32)
        del branch

    del base, features
    gc.collect()
    if dp.DEVICE.type == "mps":
        torch.mps.empty_cache()
    return out


def _load_height_f32(src: MapSource, size: tuple[int, int]) -> np.ndarray:
    if src.exists("height16.png"):
        im = src.open("height16.png")
        arr = np.asarray(im, dtype=np.float32)
        arr = arr / 65535.0 if (im.mode in ("I;16", "I") or arr.max() > 255.0) else arr / 255.0
        if im.size != size:
            arr = np.asarray(Image.fromarray(arr, mode="F").resize(size, Image.BILINEAR), dtype=np.float32)
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    im = src.open("height.png").convert("L")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR)
    return (np.asarray(im, dtype=np.float32) / 255.0).astype(np.float32)


def weather_material(
    src_dir: "Path | str | MaterialMaps",
    out_dir: Path | str,
    *,
    preset: str | None = None,
    wear: float | None = None,
    dirt: float | None = None,
    wet: float | None = None,
    moss: float | None = None,
    rust: float | None = None,
    dust: float | None = None,
    material_kind: str | None = None,
    seed: int = 42,
    backend: str = "procedural",
    patch_dir: Path | str | None = None,
    vit_patch: Path | str | None = None,
) -> Path:
    """Apply multi-mode physical weathering to any existing hypbr material directory.

    Reads the source 15-map directory (`albedo_delit.png`, `height16.png`/`height.png`,
    `normal.png`, `ao.png`, `roughness.png`, `metallic.png`) — or an in-memory `MaterialMaps` —
    applies any combination of `preset` and `wear`, `dirt`, `wet`, `moss`, `rust`, `dust`
    (using either exact periodic geometric masks or optional single-encode `DepthPro2-S` neural
    masks), and writes a verified seamless PBR directory + `weather_mask.png` (`RGBA8`) into
    `out_dir`. Equivalent to `weather_material_maps(...).save(out_dir)`.
    """
    return weather_material_maps(
        src_dir, preset=preset, wear=wear, dirt=dirt, wet=wet, moss=moss, rust=rust, dust=dust,
        material_kind=material_kind, seed=seed, backend=backend, patch_dir=patch_dir, vit_patch=vit_patch,
    ).save(out_dir)


def weather_material_maps(
    src: "Path | str | MaterialMaps",
    *,
    preset: str | None = None,
    wear: float | None = None,
    dirt: float | None = None,
    wet: float | None = None,
    moss: float | None = None,
    rust: float | None = None,
    dust: float | None = None,
    material_kind: str | None = None,
    seed: int = 42,
    backend: str = "procedural",
    patch_dir: Path | str | None = None,
    vit_patch: Path | str | None = None,
    preview: bool = True,
) -> MaterialMaps:
    """In-memory `weather_material`: weather a material directory or `MaterialMaps`, return `MaterialMaps`.

    The source maps are read at their saved 8/16-bit precision, so the result is identical
    whether `src` is a `MaterialMaps` or the directory it was saved to.
    """
    from .synthesize import render_studio_preview

    wp = resolve_weather_params(preset, wear=wear, dirt=dirt, wet=wet, moss=moss, rust=rust, dust=dust)
    src = MapSource(src)

    alb_im = src.open("albedo_delit.png").convert("RGB")
    size = alb_im.size
    alb_u8_src = np.asarray(alb_im, dtype=np.uint8)
    alb_lin = srgb_u8_to_linear(alb_u8_src)

    h_n = _load_height_f32(src, size)
    n_im = src.open("normal.png").convert("RGB")
    if n_im.size != size:
        n_im = n_im.resize(size, Image.BILINEAR)
    N = ng.normalize_n(np.asarray(n_im, dtype=np.float32) / 127.5 - 1.0)

    def _load_l(fname: str, default: float) -> np.ndarray:
        if not src.exists(fname):
            return np.full((size[1], size[0]), default, dtype=np.float32)
        im = src.open(fname).convert("L")
        if im.size != size:
            im = im.resize(size, Image.BILINEAR)
        return (np.asarray(im, dtype=np.float32) / 255.0).astype(np.float32)

    ao = _load_l("ao.png", 1.0)
    rough = _load_l("roughness.png", 0.5)
    met_f = _load_l("metallic.png", 0.0)
    curv = ng.node_curvature_multiscale({"circular": True}, {"normal": N}, gain=2.8)

    custom_masks = None
    if backend == "neural":
        rgb_in = (np.asarray(src.open("preview.png").convert("RGB").resize(size, Image.BICUBIC), dtype=np.uint8)
                  if src.exists("preview.png") else alb_u8_src)
        custom_masks = predict_neural_weathering_masks(rgb_in, vit_patch=vit_patch, patch_dir=patch_dir)
    elif backend != "procedural":
        raise ValueError(f"Unknown weathering backend {backend!r}. Choose 'procedural' or 'neural'.")

    kind = material_kind or infer_material_kind(src.name, float(met_f.mean()))
    res = apply_weathering_maps(
        alb_lin, N, h_n, rough, met_f, ao, curv,
        wear=wp["wear"], dirt=wp["dirt"], wet=wp["wet"],
        moss=wp["moss"], rust=wp["rust"], dust=wp["dust"],
        material_kind=kind, material_id=src.name, seed=seed,
        custom_masks=custom_masks,
    )

    N_out = res["N"]
    rough_out = ng.node_roughness_specular_aa({"circular": True}, {"roughness": res["rough"], "normal": N_out}, strength=0.22, max_boost=0.06)["roughness"]
    # Keep glossy standing puddles ultra-crisp after Specular AA
    puddle = res["masks"].get("puddle")
    if puddle is not None and float(puddle.max()) > 1e-4:
        rough_out = np.clip(rough_out * (1.0 - puddle) + 0.095 * puddle, 0.05, 0.97).astype(np.float32)

    # Recompute AO / bent normal if puddle flattened normals strongly, else reuse source AO + subtle wet contact
    hbao = ng.node_horizon_bent_normal_ao({"circular": True}, {"height": h_n, "normal": N_out}, strength=0.85, height_scale=5.0, ao_floor=0.18)
    ao_out = np.minimum(ao, hbao["ao"]).astype(np.float32) if src.exists("ao.png") else hbao["ao"]
    curv_out = ng.node_curvature_multiscale({"circular": True}, {"normal": N_out}, gain=2.8)
    aniso = ng.node_anisotropy_flow({"circular": True}, {"height": h_n, "ao": ao_out, "roughness": rough_out})
    thick = ng.node_thickness_subsurface({"circular": True}, {"height": h_n})
    prev_u8 = render_studio_preview(res["alb_lin"], N_out, rough_out, res["met_f"], ao_out) if preview else None

    meta = dict(src.maps.metadata) if src.maps is not None else {"material_id": src.name}
    meta.update({"kind": kind, "weather": dict(wp), "weather_backend": backend, "weather_seed": seed})
    return MaterialMaps(
        albedo=res["alb_lin"],
        normal=unit_normals(N_out),
        height=h_n,
        roughness=rough_out,
        metallic=res["met_u8"].astype(np.float32) / 255.0,
        ao=ao_out,
        bent_normal=unit_normals(hbao["bent_normal"]),
        curvature=curv_out["curvature"],
        cavity=curv_out["cavity"],
        edge_wear=curv_out["convex"],
        anisotropy=aniso["anisotropy"],
        specular=aniso["specular"],
        thickness=thick["thickness"],
        preview=prev_u8,
        weather_mask=res["weather_mask"],
        metadata=meta,
    )
