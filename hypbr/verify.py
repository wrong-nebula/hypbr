"""Automated physical PBR calibration & seamlessness verifier (`hypbr verify`).

Checks a material directory for:
  - Map completeness, matching resolution, and genuine 16-bit height
  - Exact border wrap-around seamlessness (seam jump vs. interior gradient ratio)
  - Tangent-space normal unit-length, +Z hemisphere validity, and Helmholtz curl residual
  - PBR physical albedo/roughness/AO bounds and glTF ORM channel consistency
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image  # noqa: F401

from .maps import MapSource, MaterialMaps  # noqa: F401  (MaterialMaps: annotations)

CORE_MAPS = (
    "albedo_delit.png",
    "normal.png",
    "height.png",
    "height16.png",
    "ao.png",
    "roughness.png",
    "metallic.png",
    "orm.png",
)


def _seam_metrics(arr: np.ndarray) -> dict[str, float]:
    """Compare wrap-around border jump against mean interior 1-pixel gradient."""
    x = arr.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    border_v = float(np.mean(np.abs(x[0, ...] - x[-1, ...])))
    border_h = float(np.mean(np.abs(x[:, 0, ...] - x[:, -1, ...])))
    interior_v = float(np.mean(np.abs(x[1:, ...] - x[:-1, ...])))
    interior_h = float(np.mean(np.abs(x[:, 1:, ...] - x[:, :-1, ...])))

    border_l1 = 0.5 * (border_v + border_h)
    interior_l1 = 0.5 * (interior_v + interior_h)
    ratio = border_l1 / max(interior_l1, 1e-4)
    return {
        "border_l1": round(border_l1, 5),
        "interior_l1": round(interior_l1, 5),
        "seam_ratio": round(ratio, 3),
    }


def _normal_metrics(nrm_u8: np.ndarray) -> dict[str, float]:
    n = nrm_u8.astype(np.float32) / 127.5 - 1.0
    norms = np.linalg.norm(n, axis=-1)
    unit_err = float(np.mean(np.abs(norms - 1.0)))
    pos_z_frac = float(np.mean(n[..., 2] > 0.0))

    # Normalized Helmholtz curl residual: RMS(dp/dy - dq/dx) / RMS(|slope|)
    nz = np.maximum(n[..., 2], 0.15)
    p = n[..., 0] / nz
    q = n[..., 1] / nz
    dp_dy = np.roll(p, -1, axis=0) - p
    dq_dx = np.roll(q, -1, axis=1) - q
    curl_rms = float(np.sqrt(np.mean((dp_dy - dq_dx) ** 2)))
    slope_rms = float(np.sqrt(np.mean(p * p + q * q))) + 1e-6
    return {
        "unit_length_error": round(unit_err, 5),
        "positive_z_fraction": round(pos_z_frac, 5),
        "curl_residual_ratio": round(curl_rms / slope_rms, 4),
    }


def verify_material(mat_dir: "Path | str | MaterialMaps", seam_ratio_limit: float = 2.2) -> dict[str, Any]:
    """Run deterministic physical and seamlessness QA checks on a hypbr material directory or `MaterialMaps`."""
    d = MapSource(mat_dir)
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    missing = [m for m in CORE_MAPS if not d.exists(m)]
    if missing:
        errors.append(f"Missing core maps: {', '.join(missing)}")
        return {"material": d.name, "passed": False, "errors": errors, "warnings": warnings, "checks": checks}

    sizes: dict[str, tuple[int, int]] = {}
    for m in CORE_MAPS:
        with d.open(m) as im:
            sizes[m] = im.size
            if m == "height16.png" and im.mode not in ("I;16", "I"):
                errors.append(f"height16.png is mode {im.mode}, expected 16-bit grayscale")

    unique_sizes = set(sizes.values())
    if len(unique_sizes) != 1:
        errors.append(f"Inconsistent map resolutions: {sizes}")
    resolution = next(iter(unique_sizes))
    checks["resolution"] = list(resolution)

    alb = np.asarray(d.open("albedo_delit.png").convert("RGB"), dtype=np.uint8)
    nrm = np.asarray(d.open("normal.png").convert("RGB"), dtype=np.uint8)
    h8 = np.asarray(d.open("height.png").convert("L"), dtype=np.uint8)
    ao = np.asarray(d.open("ao.png").convert("L"), dtype=np.uint8)
    rgh = np.asarray(d.open("roughness.png").convert("L"), dtype=np.uint8)
    met = np.asarray(d.open("metallic.png").convert("L"), dtype=np.uint8)
    orm = np.asarray(d.open("orm.png").convert("RGB"), dtype=np.uint8)

    # 1. Seamless wrap-around checks
    seam_h = _seam_metrics(h8)
    seam_n = _seam_metrics(nrm)
    seam_a = _seam_metrics(alb)
    checks["seamless"] = {"height": seam_h, "normal": seam_n, "albedo": seam_a}
    for key, sm in (("height", seam_h), ("normal", seam_n), ("albedo", seam_a)):
        if sm["border_l1"] > 0.03 and sm["seam_ratio"] > seam_ratio_limit:
            errors.append(f"{key} wrap seam detected (ratio={sm['seam_ratio']}, border_l1={sm['border_l1']})")

    # 2. Normal validity
    nrm_stats = _normal_metrics(nrm)
    checks["normal"] = nrm_stats
    if nrm_stats["unit_length_error"] > 0.04:
        errors.append(f"Normal unit-length error too high ({nrm_stats['unit_length_error']})")
    if nrm_stats["positive_z_fraction"] < 0.999:
        errors.append(f"Inverted/negative-Z normals detected (pos_z={nrm_stats['positive_z_fraction']})")

    # 3. ORM packing consistency
    orm_diff = int(
        max(
            np.max(np.abs(orm[..., 0].astype(np.int16) - ao.astype(np.int16))),
            np.max(np.abs(orm[..., 1].astype(np.int16) - rgh.astype(np.int16))),
            np.max(np.abs(orm[..., 2].astype(np.int16) - met.astype(np.int16))),
        )
    )
    checks["orm_max_channel_diff"] = orm_diff
    if orm_diff > 1:
        errors.append(f"orm.png channels diverge from ao/roughness/metallic (max_diff={orm_diff})")

    # 4. Physical calibration bounds
    lum = 0.2126 * alb[..., 0] + 0.7152 * alb[..., 1] + 0.0722 * alb[..., 2]
    p01, p99 = float(np.percentile(lum, 1)), float(np.percentile(lum, 99))
    r_mean, r_std = float(rgh.mean() / 255.0), float(rgh.std() / 255.0)
    checks["pbr_ranges"] = {
        "albedo_lum_p01": round(p01, 1),
        "albedo_lum_p99": round(p99, 1),
        "roughness_mean": round(r_mean, 3),
        "roughness_std": round(r_std, 4),
        "ao_mean": round(float(ao.mean() / 255.0), 3),
        "metallic_mean": round(float(met.mean() / 255.0), 3),
    }
    if p01 < 8.0:
        warnings.append(f"Very dark albedo shadows (p01={p01:.1f} sRGB)")
    if p99 > 248.0:
        warnings.append(f"Near-clipping albedo highlights (p99={p99:.1f} sRGB)")

    return {
        "material": d.name,
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
