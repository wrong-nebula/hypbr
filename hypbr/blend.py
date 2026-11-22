"""Physically-based height-lerp PBR material blending (`hypbr blend`).

Blends two seamless hypbr material directories using heightmap-driven masking
(analogous to Substance / Unreal HeightLerp), preserving physical correctness:
  - Linear-light albedo blending (sRGB -> linear -> blend -> sRGB)
  - Tangent-space normal blending with unit re-normalization
  - Height-masked roughness, metallic, AO, specular, thickness, and ORM repacking
  - Exact circular-padded curvature, cavity, and edge-wear recomputation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import depthpro_pbr as dp
from . import normal_graph as ng
from . import procedural as gd
from .synthesize import render_studio_preview
from .color import srgb_to_linear
from .maps import MapSource, MaterialMaps, unit_normals


def _load_srgb_linear(src: MapSource, fname: str, size: tuple[int, int]) -> np.ndarray:
    im = src.open(fname).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return srgb_to_linear(arr)


def _load_gray_f32(src: MapSource, fname: str, size: tuple[int, int], default: float = 0.5) -> np.ndarray:
    if not src.exists(fname):
        return np.full((size[1], size[0]), default, dtype=np.float32)
    im = src.open(fname).convert("L")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _load_height_f32(src: MapSource, size: tuple[int, int]) -> np.ndarray:
    """Prefer full-precision 16-bit height16.png when available, falling back to 8-bit height.png."""
    if src.exists("height16.png"):
        im = src.open("height16.png")
        arr = np.asarray(im, dtype=np.float32)
        if im.mode in ("I;16", "I") or arr.max() > 255.0:
            arr = arr / 65535.0
        else:
            arr = arr / 255.0
        if im.size != size:
            im_f = Image.fromarray(arr, mode="F").resize(size, Image.BILINEAR)
            arr = np.asarray(im_f, dtype=np.float32)
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    return _load_gray_f32(src, "height.png", size, 0.5)


def _load_normal_f32(src: MapSource, fname: str, size: tuple[int, int]) -> np.ndarray:
    im = src.open(fname).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR)
    n = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    return ng.normalize_n(n)


def compute_height_blend_mask(
    h_a: np.ndarray,
    h_b: np.ndarray,
    factor: float = 0.5,
    contrast: float = 0.12,
    mode: str = "height",
) -> np.ndarray:
    """Compute organic, C1-smooth HeightLerp blend weight `m in [0, 1]` selecting material B over A.

    - `mode="height"`: Unreal/Substance HeightLerp — material B rises where
      `h_b + factor` exceeds `h_a + (1 - factor)`, with organic micro-fracture breakup.
    - `mode="cavity"`: Material B settles preferentially into crevices/valleys of A,
      modulated by material B's height and periodic micro-variation to avoid uniform blobs.
    - `mode="linear"`: Uniform scalar blend `m = factor`.
    """
    f = float(np.clip(factor, 0.0, 1.0))
    c = max(float(contrast), 1e-3)
    if mode == "linear":
        return np.full_like(h_a, f, dtype=np.float32)

    H, W = h_a.shape
    res_scale = min(H, W) / 512.0
    # Smooth macro region field + crisp micro edge breakup (prevents stray noise smudges on tied flat plateaus)
    macro = gd._periodic_fbm(H, W, 5213, (2, 4))
    breakup = gd._periodic_fbm(H, W, 9137, (16, 32, 64)) * 0.045
    if mode == "cavity":
        # Strictly confine material B inside recessed local bowls/valleys of A while protecting A's raised ridges
        h_a_loc = dp.gaussian_blur_np(h_a, max(2.0, 10.0 * res_scale), circular=True)
        local_bowl = np.clip((h_a_loc - h_a) * 2.6 + 0.5, 0.0, 1.0)
        ridge_prot = np.clip((h_a - h_a_loc) * 3.5, 0.0, 1.0)
        score = (1.0 - h_a) * 0.55 + local_bowl * 0.45 - 0.45 * ridge_prot + breakup * 0.8 + (f * 1.15 - 0.575)
        m = np.clip((score - 0.5) / c + 0.5, 0.0, 1.0)
    elif mode == "height":
        # Contiguous height-modulated transition: macro region sweeps cleanly while height detail
        # carves natural mortar/edge steps along the boundary without letting isolated spikes poke
        # disconnected islands through grout lines or flat plateaus.
        h_a_env = 0.55 * dp.gaussian_blur_np(h_a, max(2.0, 9.0 * res_scale), circular=True) + 0.45 * h_a
        h_b_env = 0.55 * dp.gaussian_blur_np(h_b, max(2.0, 9.0 * res_scale), circular=True) + 0.45 * h_b
        region = (2.0 * f - 1.0) * 0.95 + macro * 0.92
        frontier = np.exp(-5.0 * region * region)
        diff = region + (h_b_env - h_a_env) * (0.28 * frontier) + breakup * frontier
        m0 = np.clip(diff / max(c * 1.4, 0.06) + 0.5, 0.0, 1.0)
        # Mild continuity filter dissolves isolated pinpoint slivers while preserving crisp boundary contours
        m_cont = dp.gaussian_blur_np(m0, max(1.0, 3.0 * res_scale), circular=True)
        m = np.clip((m_cont - 0.5) / 0.28 + 0.5, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown blend mode {mode!r}. Choose 'height', 'cavity', or 'linear'.")
    # Smoothstep for crisp yet anti-aliased transitions
    return (m * m * (3.0 - 2.0 * m)).astype(np.float32)


def blend_materials(
    dir_a: "Path | str | MaterialMaps",
    dir_b: "Path | str | MaterialMaps",
    out_dir: Path | str,
    factor: float = 0.5,
    contrast: float = 0.06,
    mode: str = "height",
) -> Path:
    """Physically blend two hypbr materials (directories or `MaterialMaps`) into a new 15-map directory.

    Equivalent to `blend_materials_maps(dir_a, dir_b, ...).save(out_dir)`.
    """
    return blend_materials_maps(dir_a, dir_b, factor=factor, contrast=contrast, mode=mode).save(out_dir)


def blend_materials_maps(
    mat_a: "Path | str | MaterialMaps",
    mat_b: "Path | str | MaterialMaps",
    factor: float = 0.5,
    contrast: float = 0.06,
    mode: str = "height",
    *,
    preview: bool = True,
) -> MaterialMaps:
    """In-memory `blend_materials`: blend two materials (directories or `MaterialMaps`) -> `MaterialMaps`.

    Inputs are read at their saved 8/16-bit precision, so the result is identical whether each
    input is a `MaterialMaps` or the directory it was saved to. `blend_mask` holds the B weight.
    """
    pa, pb = MapSource(mat_a), MapSource(mat_b)

    with pa.open("albedo_delit.png") as im0:
        size = im0.size

    h_a = _load_height_f32(pa, size)
    h_b = _load_height_f32(pb, size)
    mask = compute_height_blend_mask(h_a, h_b, factor=factor, contrast=contrast, mode=mode)

    # Unified crisp physical boundary mask so albedo, roughness, and metallic share clean edges
    # without semi-metallic gray halos or muddy/cyan transition fringes.
    if mode == "linear":
        surf_mask = mask
        met_mask = mask
    else:
        sm = np.clip((mask - 0.5) / 0.22 + 0.5, 0.0, 1.0)
        surf_mask = (sm * sm * (3.0 - 2.0 * sm)).astype(np.float32)
        mm = np.clip((mask - 0.5) / 0.14 + 0.5, 0.0, 1.0)
        met_mask = (mm * mm * (3.0 - 2.0 * mm)).astype(np.float32)
    m3 = surf_mask[..., None]

    # Contact-shadow halo along the seam boundary (gives grounded physical seating)
    H, W = surf_mask.shape
    res_scale = min(H, W) / 512.0
    rim_band = 4.0 * surf_mask * (1.0 - surf_mask)
    if mode == "linear":
        contact_ao = np.ones_like(surf_mask)
    else:
        sm_blur = dp.gaussian_blur_np(surf_mask, max(1.0, 3.0 * res_scale), circular=True)
        # Darken recessed side just outside the upper material's rim
        if mode == "height":
            seam_sh = np.clip(np.abs(sm_blur - surf_mask) * 2.4, 0.0, 1.0)
        else:
            seam_sh = np.clip((1.0 - sm_blur) * surf_mask * 2.2 + rim_band * 0.6, 0.0, 1.0)
        contact_ao = np.clip(1.0 - 0.38 * seam_sh, 0.55, 1.0).astype(np.float32)

    # 1. Linear-light albedo blend + grounded contact shading at material boundary
    alb_a = _load_srgb_linear(pa, "albedo_delit.png", size)
    alb_b = _load_srgb_linear(pb, "albedo_delit.png", size)
    alb_lin = ((1.0 - m3) * alb_a + m3 * alb_b) * (0.72 + 0.28 * contact_ao[..., None])

    # 2. Height & slope-consistent normal blend with physical step lip & soft-knee slope limit
    n_a = _load_normal_f32(pa, "normal.png", size)
    n_b = _load_normal_f32(pb, "normal.png", size)
    if mode == "height":
        h_blend = np.maximum((1.0 - surf_mask) * h_a, surf_mask * (h_b + 0.14))
        lip_slope = 2.4
    elif mode == "cavity":
        h_blend = (1.0 - surf_mask) * h_a + surf_mask * (h_a * 0.65 + h_b * 0.20)
        # Attenuate inset cavity material's micro-slope so it sits recessed rather than bulging above A's ridges
        pb_s, qb_s = ng.normal_to_slopes(n_b)
        n_b = ng.slopes_to_normal(pb_s * 0.55, qb_s * 0.55)
        lip_slope = -1.8
    else:
        h_blend = (1.0 - surf_mask) * h_a + surf_mask * h_b
        lip_slope = 0.0
    h_out = dp.normalize_height(h_blend, clip_pct=0.5)
    n_lip = gd._height_to_normal_periodic(surf_mask, slope=lip_slope)
    n_mix = ng.normalize_n((1.0 - m3) * n_a + m3 * n_b)
    n_out = dp.blend_rnm(n_mix, n_lip, detail_weight=0.55)
    n_out = ng.node_normal_clamp_slope({"circular": True}, {"normal": n_out}, max_angle_deg=60.0)["normal"]

    # 3. Roughness (with Specular AA) & crisp physical Metallic
    r_a = _load_gray_f32(pa, "roughness.png", size, 0.5)
    r_b = _load_gray_f32(pb, "roughness.png", size, 0.5)
    cav_rough_boost = (0.14 * surf_mask) if mode == "cavity" else 0.0
    rough = np.clip((1.0 - surf_mask) * r_a + surf_mask * r_b + cav_rough_boost + 0.12 * (1.0 - contact_ao), 0.06, 0.98).astype(np.float32)
    rough = ng.node_roughness_specular_aa({"circular": True}, {"roughness": rough, "normal": n_out}, strength=0.2, max_boost=0.06)["roughness"]

    met_a = _load_gray_f32(pa, "metallic.png", size, 0.0)
    met_b = _load_gray_f32(pb, "metallic.png", size, 0.0)
    met = np.clip((1.0 - met_mask) * met_a + met_mask * met_b, 0.0, 1.0).astype(np.float32)

    # 4. Recompute physical 8-azimuth HBAO + contact seam AO + bent normal + curvature + anisotropy + thickness
    hbao = ng.node_horizon_bent_normal_ao({"circular": True}, {"height": h_out, "normal": n_out}, strength=1.10, height_scale=12.0, ao_floor=0.18)
    ao = np.clip(hbao["ao"] * contact_ao, 0.16, 1.0).astype(np.float32)
    curv = ng.node_curvature_multiscale({"circular": True}, {"normal": n_out}, gain=2.8)
    aniso = ng.node_anisotropy_flow({"circular": True}, {"height": h_out, "ao": ao, "roughness": rough})
    thick = ng.node_thickness_subsurface({"circular": True}, {"height": h_out})

    # 5. Physically-based studio preview (with metallic Fresnel IBL)
    prev_u8 = render_studio_preview(alb_lin, n_out, rough, met, ao) if preview else None

    return MaterialMaps(
        albedo=alb_lin,
        normal=unit_normals(n_out),
        height=h_out,
        roughness=rough,
        metallic=met,
        ao=ao,
        bent_normal=unit_normals(hbao["bent_normal"]),
        curvature=curv["curvature"],
        cavity=curv["cavity"],
        edge_wear=curv["convex"],
        anisotropy=aniso["anisotropy"],
        specular=aniso["specular"],
        thickness=thick["thickness"],
        preview=prev_u8,
        blend_mask=surf_mask,
        metadata={
            "material_id": f"{pa.name}_blend_{pb.name}",
            "blend": {"a": pa.name, "b": pb.name, "factor": float(factor), "contrast": float(contrast), "mode": mode},
        },
    )
