"""Reference-photo calibration for the hypbr atlas (v1.1).

Real CC0 photogrammetry / photometric-stereo scans (Poly Haven, ambientCG) are reduced to
scale-aware statistics and compared with each atlas material:

* **Albedo contrast spectrum** - std of log-luminance band-pass layers (periodic Gaussian
  pyramid, sigma = 1..64 px), aligned in *meters* using each side's real-world tile size.
* **Chroma variation** - spatial std of normalized chromaticity.
* **Normal slope** - mean tangent-space slope |n_xy| / n_z.
* **Roughness** - mean roughness (where reference roughness maps exist).

Fitted corrections (per material) are stored in `hypbr/data/atlas_calibration.json` and applied
inside `synthesize_material` as an exactly periodic FFT band equalizer on albedo luminance,
a saturation gain, a normal-slope gain and a roughness offset. Seamlessness is preserved
because every operation is either per-pixel or a circular convolution.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .color import srgb_u8_to_linear

SIGMAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)  # band k = G(sigma_{k-1}) - G(sigma_k), sigma_{-1}=0
N_BANDS = len(SIGMAS)
SAT_GAIN_RANGE = (0.7, 1.4)  # chroma-variation gain limits (was 0.85-1.2, which pinned most materials)
_CAL_PATH = Path(__file__).resolve().parent / "data" / "atlas_calibration.json"
_EPS = 0.004


# ---------------------------------------------------------------------------
# Periodic band analysis & equalization
# ---------------------------------------------------------------------------
def _gauss_fft(x: np.ndarray, sigma: float) -> np.ndarray:
    H, W = x.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    k = np.exp(-2.0 * (math.pi ** 2) * (sigma ** 2) * (fx * fx + fy * fy))
    return np.fft.irfft2(np.fft.rfft2(x) * k, s=(H, W)).astype(np.float32)


def _srgb_to_lin(a_u8: np.ndarray) -> np.ndarray:
    return srgb_u8_to_linear(a_u8)


def _lum(lin: np.ndarray) -> np.ndarray:
    return (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]).astype(np.float32)


FIT_SIZE_PX = 512  # calibration statistics and gains are measured on 512px bakes


def band_layers(logl: np.ndarray, scale: float = 1.0) -> tuple[list[np.ndarray], np.ndarray]:
    """Periodic difference-of-Gaussian bands at SIGMAS * scale pixels (scale = size / FIT_SIZE_PX)."""
    prev = logl
    bands = []
    for s in SIGMAS:
        g = _gauss_fft(logl, s * scale)
        bands.append(prev - g)
        prev = g
    return bands, prev


def albedo_stats(alb_lin: np.ndarray) -> dict[str, Any]:
    L = _lum(alb_lin)
    logl = np.log(L + _EPS)
    bands, _ = band_layers(logl)
    chrom = alb_lin / (alb_lin.sum(axis=-1, keepdims=True) + 1e-4)
    chroma_var = float(np.sqrt(chrom.var(axis=(0, 1)).sum()))
    return {"bands": [float(b.std()) for b in bands], "lum_mean": float(L.mean()), "chroma_var": chroma_var,
            "color": [float(c) for c in alb_lin.reshape(-1, 3).mean(axis=0)]}


def slope_bands(nrm_u8: np.ndarray) -> list[float]:
    """Multi-scale energy of tangent-space slope (nx/nz, ny/nz) per periodic band."""
    n = nrm_u8.astype(np.float32) / 127.5 - 1.0
    nz = np.clip(n[..., 2], 0.15, 1.0)
    out = []
    bx, _ = band_layers(n[..., 0] / nz)
    by, _ = band_layers(n[..., 1] / nz)
    for a, b in zip(bx, by):
        out.append(float(np.sqrt(a.var() + b.var())))
    return out


def equalize_height(h: np.ndarray, gains: list[float], scale: float = 1.0) -> np.ndarray:
    """Reshape relief per periodic band (scales slope energy in each band by its gain)."""
    hf = np.asarray(h, dtype=np.float32)
    bands, base = band_layers(hf, scale)
    out = base.copy()
    for b, g in zip(bands, gains):
        out += float(g) * b
    return out.astype(np.float32)


def adjust_roughness(rough: np.ndarray, shift: float, std_gain: float, lo: float = 0.04, hi: float = 0.97) -> np.ndarray:
    mu = float(rough.mean())
    return np.clip(mu + float(shift) + (rough - mu) * float(std_gain), lo, hi).astype(np.float32)


def normal_slope(nrm_u8: np.ndarray) -> float:
    n = nrm_u8.astype(np.float32) / 127.5 - 1.0
    nz = np.clip(n[..., 2], 0.15, 1.0)
    return float(np.mean(np.sqrt(n[..., 0] ** 2 + n[..., 1] ** 2) / nz))


def equalize_albedo(alb_lin: np.ndarray, gains: list[float], sat_gain: float = 1.0,
                    alb_max: np.ndarray | float = 0.855, scale: float = 1.0) -> np.ndarray:
    """Reshape albedo luminance contrast per band (periodic) and rescale chroma variation."""
    L = _lum(alb_lin)
    logl = np.log(L + _EPS)
    bands, base = band_layers(logl, scale)
    out = base.copy()
    for b, g in zip(bands, gains):
        out += float(g) * b
    L2 = np.maximum(np.exp(out) - _EPS, 1e-4)
    ratio = (L2 / np.maximum(L, 1e-4))[..., None]
    alb = alb_lin * ratio
    if abs(sat_gain - 1.0) > 1e-3:
        Lc = _lum(alb)[..., None]
        alb = Lc + (alb - Lc) * float(sat_gain)
    return np.clip(alb, 0.015, alb_max).astype(np.float32)


@lru_cache(maxsize=1)
def _table() -> dict[str, Any]:
    try:
        return json.loads(_CAL_PATH.read_text(encoding="utf-8")).get("materials", {})
    except (OSError, ValueError):
        return {}


def get_calibration(material_id: str) -> dict[str, Any] | None:
    """Fitted correction for a material, or None when no reference calibration exists."""
    c = _table().get(material_id)
    return c if c and c.get("applied", True) else None


def clear_cache() -> None:
    _table.cache_clear()


# ---------------------------------------------------------------------------
# Reference matching
# ---------------------------------------------------------------------------
# subcategory -> (include regex, exclude regex)
SUB_REFS: dict[str, tuple[str, str]] = {
    "brick": (r"brick", r"floor|paving|pavement|path|herringbone|gravel|crosswalk|pebble_bricks|plaster_brick"),
    "concrete_block": (r"block_wall|cinder", r"stone_block|sandstone"),
    "ashlar": (r"sandstone_blocks|sandstone_brick|medieval_blocks|medieval_wall|castle_wall|stone_brick_wall|defense_wall|japanese_stone|old_stone_wall|stone_block_wall|rabdentse", r"$^"),
    "stacked_stone": (r"stacked_stone|rustic_stone_wall|stone_wall|mossy_stone|yellow_stone|coral_stone|plaster_stone|rock_wall_0[2349]", r"brick|block"),
    "cobblestone": (r"cobblestone|stone_pavers|grey_stone_path|stone_pathway|rock_path|pebble_embedded|wood_stone_pathway", r"$^"),
    "paver": (r"paver|paving|pavement|brick_floor|floor_bricks|herringbone_brick|herringbone_pavement|brick_paving|brick_pavement|worn_brick_path|klinkers|t_brick_floor|short_bricks|square_brick_floor|recycled_brick", r"concrete_pavement|pebble"),
    "hardwood_floor": (r"plank_flooring|wooden_floor|wood_floor|old_wood_floor|laminate|embedded_plank|wood_planks$|wooden_planks|brown_planks|dark_planks|plank", r"wall|door|shutter|cladding|grey|dirt"),
    "parquet": (r"herringbone_parquet|parquet|plank_flooring|wooden_floor|old_wood_floor|laminate", r"$^"),
    "wood_cladding": (r"wood_plank_wall|wooden_panels|cladding|wood_shutter|weathered_brown_planks|peeling_timber|green_rough_planks|wood_planks_grey|rough_pine_door|paneled_wood|dark_wooden_planks", r"$^"),
    "wood_slats": (r"bamboo_wall|wood_shutter", r"$^"),
    "bark": (r"bark|wood_trunk", r"$^"),
    "engineered_wood": (r"synthetic_wood|veneer|stained_pine|dark_wood$|Wood", r"floor|bark"),
    "marble": (r"marble", r"cliff|mosaic"),
    "granite_quartz": (r"granite|terrazzo", r"$^"),
    "concrete": (r"concrete|cement|preconcrete|rebar|hangar", r"paver|pavement|tiles|rock_embedded|pebble|gravel"),
    "plaster_stucco": (r"plaster|stucco|beige_wall|clay_wall|painted_plaster", r"brick|stone"),
    "natural_stone": (r"slate|sandstone_tiles|seaworn_stone|volcanic_rock_tiles|stone_floor|mixed_rock_tiles|granite_tile|old_sandstone", r"$^"),
    "subway_tile": (r"tiled_wall|grey_tiles|square_tiles|blue_floor_tiles|worn_tile_floor|square_floor$", r"$^"),
    "zellige": (r"tiled_wall|blue_floor_tiles|square_tiles", r"$^"),
    "porcelain": (r"square_tiles|grey_tiles|square_floor|large_square_pattern|anti_skid_tiles|concrete_tiles", r"$^"),
    "mosaic": (r"mosaic|floor_pattern|patterned_slate|rounded_square", r"$^"),
    "terracotta": (r"terracotta|red_sandstone_tiles|floor_klinkers", r"$^"),
    "specialty_metal": (r"rust|metal_plate|rusty", r"grid|shutter"),
    "metal_cladding": (r"corrugated|container_side|shutter", r"$^"),
    "woven_textile": (r"denim|denmin|linen|hessian|jacquard|gingham|satin|stretch|terlenka|herringbone$|poly_wool|pique|caban|Fabric", r"$^"),
    "plush_textile": (r"velvet|velour|fleece|boucle|teddy|corduroy|carpet|terry|fur|jersey|melange|knitted", r"$^"),
    "leather": (r"leather|suede", r"$^"),
    "natural_fiber": (r"hessian|bamboo_wall|riet", r"$^"),
    "sand": (r"sand|beach", r"rock|stone|sandstone|gravel|mud"),
    "mud": (r"mud|cracked|dry_ground|dry_mud|cracked_red", r"concrete|plaster|rock"),
    "gravel": (r"gravel|pebble|Gravel", r"embedded|bricks|road|concrete|asphalt"),
    "ice": (r"snow", r"asphalt"),
    "ground_cover": (r"forest|grass|leaves|dirt|soil|mulch|wood_chip|snow|park_dirt|ground", r"asphalt|rocks?_|gravel|concrete"),
    "rock_cliff": (r"rock_face|rock_wall|cliff|quarry|^rock_0|rock_surface|seaside_rock|low_tide|boulder|worn_rock|Rock", r"embedded|tiles|path|trail"),
    "thatch": (r"reed_roof|riet", r"$^"),
    "asphalt": (r"asphalt|road|tarred", r"snow|flower"),
    "rubber_mat": (r"running_track|anti_slip|baseball", r"$^"),
    "foam": (r"polystyrene", r"$^"),
    "roofing": (r"roof", r"reed"),
    # Proxy references (clean/painted sheet metal scans) for machined & panel metals without direct scans
    "scifi_panel": (r"metal_plate|container_side", r"rust"),
    "machined_metal": (r"metal_plate|container_side", r"rust"),
    "perforated_metal": (r"metal_grid|metal_plate|container_side", r"$^"),
    "heatsink": (r"metal_plate|container_side", r"rust"),
    "brushed_metal": (r"metal_plate", r"rust|blue"),
}
# Material-specific reference keywords (used first when >=2 scans match)
ID_REFS: dict[str, str] = {
    "denim": r"denim|denmin", "linen": r"linen", "burlap": r"hessian", "jute": r"hessian", "canvas": r"linen|hessian",
    "tweed": r"herringbone$|poly_wool|caban", "corduroy": r"corduroy", "velvet": r"velvet|velour", "boucle": r"boucle|teddy",
    "knit": r"knitted|jersey|melange", "jersey": r"jersey|melange", "carpet": r"carpet", "terry": r"terry", "fleece": r"fleece",
    "leather": r"leather", "suede": r"suede", "alcantara": r"suede",
    "slate": r"slate", "sandstone": r"sandstone", "granite": r"granite", "marble": r"marble", "limestone": r"sandstone|stone_floor",
    "terracotta": r"terracotta", "clay": r"clay|terracotta", "asphalt": r"asphalt", "tarmac": r"asphalt|tarred",
    "cobble": r"cobblestone", "flagstone": r"stone_pavers|grey_stone_path|stone_pathway", "herringbone": r"herringbone",
    "parquet": r"parquet", "bark": r"bark", "pine": r"pine", "oak": r"oak", "plaster": r"plaster", "stucco": r"stucco",
    "concrete": r"concrete", "corrugated": r"corrugated", "rust": r"rust", "corten": r"rust", "snow": r"snow",
    "sand": r"sand", "mud": r"mud", "gravel": r"gravel", "pebble": r"pebble", "moss": r"moss", "leaves": r"leaves|forest",
    "grass": r"grass", "soil": r"soil|dirt", "dirt": r"dirt", "mulch": r"mulch|wood_chip", "lava": r"volcanic|lava",
    "reed": r"reed|riet", "thatch": r"reed|riet", "roof": r"roof", "shingle": r"roof", "brick": r"brick",
}
# Approximate real-world tile size (m) of the reference scans, by name pattern
_REF_SIZE = ((r"aerial|snow_field", 12.0), (r"denim|linen|hessian|jacquard|gingham|satin|stretch|terlenka|pique|caban|velvet|velour|fleece|boucle|teddy|corduroy|terry|fur|jersey|melange|knitted|leather|suede|Fabric", 0.6), (r"bark", 1.0), (r".*", 2.0))


def ref_size(ref_id: str) -> float:
    for pat, sz in _REF_SIZE:
        if re.search(pat, ref_id):
            return sz
    return 2.0


def _roughness_for(ref_id: str, rough_dirs: list[Path]) -> tuple[float, float] | None:
    from PIL import Image

    for d in rough_dirs:
        for cand in (d / f"{ref_id}.png", d / f"{ref_id.replace('acg_', '')}.png"):
            if cand.is_file():
                a = np.asarray(Image.open(cand).convert("L"), dtype=np.float32) / 255.0
                return float(a.mean()), float(a.std())
    return None


def reference_stats(ref_dirs: list[Path], rough_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    """Statistics for every reference scan (`*.npz` with albedo + normal)."""
    out: dict[str, dict[str, Any]] = {}
    for d in ref_dirs:
        for f in sorted(Path(d).glob("*.npz")):
            try:
                z = np.load(f)
                alb = _srgb_to_lin(z["albedo"])
                st = albedo_stats(alb)
                st["slope"] = normal_slope(z["normal"]) if "normal" in z.files else None
                st["slope_bands"] = slope_bands(z["normal"]) if "normal" in z.files else None
            except Exception:  # noqa: BLE001 - skip unreadable scans
                continue
            rid = f.stem
            # Imported scans (tools/import_refs.py) carry roughness and, when known, their physical size
            rr = None
            if "roughness" in z.files:
                r = z["roughness"].astype(np.float32) / 255.0
                rr = (float(r.mean()), float(r.std()))
            rr = rr or _roughness_for(rid, rough_dirs)
            st["rough"], st["rough_std"] = (rr if rr else (None, None))
            meta_size = None
            if "meta" in z.files:
                try:
                    meta_size = float(json.loads(str(z["meta"])).get("size_m") or 0.0) or None
                except (ValueError, TypeError):
                    meta_size = None
            st["size_m"] = meta_size or ref_size(rid)
            out[rid] = st
    return out


def ac_roles() -> tuple[str, ...]:
    from .atlas_catalog import SCENE_ROLES

    return SCENE_ROLES


def match_refs(entry: dict[str, Any], refs: dict[str, dict[str, Any]]) -> tuple[list[str], str]:
    sub = entry["subcategory"]
    generic = set(ac_roles()) | {entry["category"], entry["subcategory"], entry["kind"]}
    toks = (set(entry["id"].split("_")) | set(entry.get("tags", []))) - generic
    specific: list[str] = []
    for kw, pat in ID_REFS.items():
        if kw in toks:
            specific += [r for r in refs if re.search(pat, r)]
    inc, exc = SUB_REFS.get(sub, (None, None))
    general = [r for r in refs if inc and re.search(inc, r) and not re.search(exc, r)]
    both = sorted(set(specific) & set(general))
    if len(both) >= 2:
        return both, "material"
    if len(general) >= 2:
        return sorted(general), "subcategory"
    if not general and len(set(specific)) >= 3:
        return sorted(set(specific)), "material"
    return [], "none"


def _profile_at(bands: list[float], shift_octaves: float) -> np.ndarray:
    """Sample a band profile at our band k = reference band (k + shift) (log-linear interp)."""
    lr = np.log(np.maximum(np.asarray(bands, np.float64), 1e-5))
    idx = np.clip(np.arange(N_BANDS) + shift_octaves, 0, N_BANDS - 1)
    return np.exp(np.interp(idx, np.arange(N_BANDS), lr))


def _smooth(g: np.ndarray) -> np.ndarray:
    return np.convolve(np.pad(np.asarray(g, np.float64), 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")


def _ref_weights(our_color: list[float], refs: dict[str, dict[str, Any]], ref_ids: list[str]) -> np.ndarray:
    """Color-similarity weights (log-RGB distance) with a floor so every scan contributes."""
    f0 = np.log(np.asarray(our_color, np.float64) + 0.02)
    w = []
    for r in ref_ids:
        d = float(np.linalg.norm(np.log(np.asarray(refs[r]["color"], np.float64) + 0.02) - f0))
        w.append(math.exp(-((d / 0.7) ** 2)) + 0.12)
    w = np.asarray(w)
    return w / w.sum()


def _wgeo(profiles: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    return np.exp(np.sum(np.log(np.maximum(np.asarray(profiles), 1e-6)) * w[:, None], axis=0))


def material_targets(entry: dict[str, Any], raw: dict[str, Any], refs: dict[str, dict[str, Any]],
                     ref_ids: list[str], strength: float = 0.7) -> dict[str, Any]:
    """Target statistics = geometric blend of our raw stats and the color-weighted references."""
    size = float(entry["real_world_size_m"][0])
    w = _ref_weights(raw["color"], refs, ref_ids)
    shifts = [math.log2(max(size, 1e-3) / refs[r]["size_m"]) for r in ref_ids]
    ref_alb = _wgeo([_profile_at(refs[r]["bands"], s) for r, s in zip(ref_ids, shifts)], w)
    raw_alb = np.maximum(np.asarray(raw["bands"], np.float64), 1e-5)
    ratio = np.clip(ref_alb / raw_alb, 0.4, 3.0)
    ratio = np.where(raw_alb < 0.004, np.minimum(ratio, 1.3), ratio)  # never invent texture on flat paint/glaze
    t_alb = raw_alb * _smooth(ratio ** strength)
    is_metal = bool(entry["shader_setup"]["is_metallic"])
    cv = np.asarray([refs[r]["chroma_var"] for r in ref_ids])
    t_cv = raw["chroma_var"] * (float(np.clip((float(np.sum(cv * w)) / max(raw["chroma_var"], 1e-4)) ** 0.35, 0.88, 1.15)) if not is_metal else 1.0)
    sb_ids = [i for i, r in enumerate(ref_ids) if refs[r].get("slope_bands")]
    t_slope = None
    if sb_ids and raw.get("slope_bands"):
        ws = w[sb_ids] / w[sb_ids].sum()
        ref_sb = _wgeo([_profile_at(refs[ref_ids[i]]["slope_bands"], shifts[i]) for i in sb_ids], ws)
        raw_sb = np.maximum(np.asarray(raw["slope_bands"], np.float64), 1e-5)
        t_slope = raw_sb * _smooth(np.clip(ref_sb / raw_sb, 0.45, 2.2) ** 0.55)
    rr = [(i, refs[r]["rough"], refs[r]["rough_std"]) for i, r in enumerate(ref_ids) if refs[r].get("rough") is not None]
    t_rough = t_rough_std = None
    if len(rr) >= 2 and not is_metal:
        wr = np.asarray([w[i] for i, _, _ in rr]); wr = wr / wr.sum()
        ref_m = float(np.sum(wr * np.asarray([m for _, m, _ in rr])))
        ref_s = float(np.sum(wr * np.asarray([sd for _, _, sd in rr])))
        t_rough = raw["rough"] + float(np.clip(0.5 * (ref_m - raw["rough"]), -0.10, 0.10))
        t_rough_std = raw["rough_std"] * float(np.clip((ref_s / max(raw["rough_std"], 1e-3)) ** 0.5, 0.7, 1.6))
    return {"alb_bands": [float(x) for x in t_alb], "chroma_var": float(t_cv),
            "slope_bands": None if t_slope is None else [float(x) for x in t_slope],
            "rough": t_rough, "rough_std": t_rough_std}


def update_gains(cal: dict[str, Any], targets: dict[str, Any], measured: dict[str, Any], damping: float = 0.9) -> dict[str, Any]:
    """Multiplicative closed-loop update of the correction toward the target statistics."""
    def upd(prev, tgt, meas, lo, hi):
        r = np.asarray(tgt, np.float64) / np.maximum(np.asarray(meas, np.float64), 1e-6)
        return np.clip(np.asarray(prev, np.float64) * np.power(r, damping), lo, hi)

    cal = dict(cal)
    cal["band_gains"] = [round(float(g), 3) for g in upd(cal["band_gains"], targets["alb_bands"], measured["bands"], 0.45, 2.6)]
    # chroma_var responds to sat_gain roughly as gain**1.4..2.6 (not linearly), so a full `damping`
    # exponent overshoots and oscillates between the clamps; half the exponent converges monotonically.
    cal["sat_gain"] = round(float(np.clip(cal["sat_gain"] * (targets["chroma_var"] / max(measured["chroma_var"], 1e-4)) ** (0.5 * damping), *SAT_GAIN_RANGE)), 3)
    if targets.get("slope_bands") and measured.get("slope_bands"):
        cal["height_gains"] = [round(float(g), 3) for g in upd(cal["height_gains"], targets["slope_bands"], measured["slope_bands"], 0.5, 2.0)]
    if targets.get("rough") is not None:
        cal["rough_shift"] = round(float(np.clip(cal["rough_shift"] + damping * (targets["rough"] - measured["rough"]), -0.15, 0.15)), 3)
        cal["rough_std_gain"] = round(float(np.clip(cal["rough_std_gain"] * (targets["rough_std"] / max(measured["rough_std"], 1e-3)) ** damping, 0.6, 1.8)), 3)
    return cal


def measure_dir(d: Path, ext: str = "webp") -> dict[str, Any]:
    from PIL import Image

    alb = _srgb_to_lin(np.asarray(Image.open(d / f"albedo_delit.{ext}").convert("RGB")))
    st = albedo_stats(alb)
    nrm = np.asarray(Image.open(d / f"normal.{ext}").convert("RGB"))
    st["slope"] = normal_slope(nrm)
    st["slope_bands"] = slope_bands(nrm)
    rough = np.asarray(Image.open(d / f"orm.{ext}").convert("RGB"))[..., 1].astype(np.float32) / 255.0
    st["rough"], st["rough_std"] = float(rough.mean()), float(rough.std())
    return st


def _measure_calibrated(args: tuple[str, int]) -> tuple[str, dict[str, Any]]:
    import os
    import tempfile

    os.environ["HYPBR_DEVICE"] = "cpu"
    from ._backend import get_torch

    torch = get_torch()
    if torch is not None:
        torch.set_num_threads(2)
    from .synthesize import synthesize_material

    mid, size = args
    with tempfile.TemporaryDirectory() as td:
        d = synthesize_material(mid, Path(td) / mid, size=size)
        return mid, measure_dir(d, "png")


def fit_atlas(ref_dirs: list[Path], rough_dirs: list[Path], baked_root: Path, out_path: Path | None = None,
              strength: float = 0.7, refine_iters: int = 1, workers: int = 2, size: int = 512) -> dict[str, Any]:
    """Fit calibration for all atlas materials.

    `baked_root` must hold UNCALIBRATED 512px bakes (`<root>/<id>/{albedo_delit,normal,orm}.webp`).
    After the open-loop fit, `refine_iters` closed-loop passes synthesize every calibrated material
    (CPU, `workers` processes), re-measure and update the gains toward the stored targets.
    """
    from concurrent.futures import ProcessPoolExecutor

    from . import atlas_catalog as ac

    refs = reference_stats(ref_dirs, rough_dirs)
    table: dict[str, Any] = {}
    for e in ac.get_catalog():
        d = Path(baked_root) / e["id"]
        ids, how = match_refs(e, refs)
        if not ids or not (d / "albedo_delit.webp").is_file():
            continue
        raw = measure_dir(d)
        tg = material_targets(e, raw, refs, ids, strength)
        cal = {"band_gains": [1.0] * N_BANDS, "sat_gain": 1.0, "height_gains": [1.0] * N_BANDS,
               "rough_shift": 0.0, "rough_std_gain": 1.0}
        cal = update_gains(cal, tg, raw, damping=1.0)
        cal.update({"targets": tg, "refs": ids[:12], "n_refs": len(ids), "match": how, "iterations": 0})
        table[e["id"]] = cal

    doc = {"version": 2, "n_references": len(refs), "strength": strength, "sigmas_px": list(SIGMAS), "materials": table}

    def _write() -> None:
        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
            clear_cache()

    _write()
    for it in range(int(refine_iters)):
        ids = sorted(table)
        with ProcessPoolExecutor(max_workers=max(1, min(4, int(workers)))) as pool:
            for mid, meas in pool.map(_measure_calibrated, [(m, size) for m in ids], chunksize=2):
                c = table[mid]
                newc = update_gains(c, c["targets"], meas, damping=0.85)
                newc["iterations"] = c["iterations"] + 1
                table[mid] = newc
        _write()
    return doc
