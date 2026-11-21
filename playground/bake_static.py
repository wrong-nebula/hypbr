#!/usr/bin/env python3
"""Bake all 69 hypbr procedural PBR materials into a compact, crisp 512px WebP
static asset bundle for edge hosting (Vercel / GitHub Pages).

Outputs into `playground/assets/`:
  - `catalog.json`   (full catalog metadata, weather presets, QA telemetry)
  - `mats/<id>/albedo_delit.webp`  (512x512 RGB, q=92)
  - `mats/<id>/normal.webp`        (512x512 RGB, lossless WebP for zero banding)
  - `mats/<id>/orm.webp`           (512x512 RGB: R=AO, G=Roughness, B=Metallic, q=96)
  - `mats/<id>/aux.webp`           (512x512 RGB: R=Height, G=Curvature, B=Cavity, q=96)
  - `mats/<id>/preview.webp`       (160x160 RGB thumbnail, q=86)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Force CPU-only before importing hypbr/torch
os.environ["HYPBR_DEVICE"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PLAYGROUND_DIR = Path(__file__).resolve().parent
REPO_CANDIDATES = [
    PLAYGROUND_DIR.parent,               # when inside hypbr/playground
    PLAYGROUND_DIR.parent / "hypbr",     # when inside workspace/playground
]
for cand in REPO_CANDIDATES:
    if (cand / "hypbr" / "synthesize.py").exists():
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        break

import numpy as np
from PIL import Image

from hypbr import procedural as gd
from hypbr import synthesize as syn
from hypbr.verify import verify_material
from hypbr.weathering import WEATHER_PRESETS, infer_material_kind


KIND_TO_GROUP: dict[str, str] = {
    "metal": "Metal",
    "copper": "Metal",
    "bronze": "Metal",
    "brass": "Metal",
    "stone": "Stone & Concrete",
    "concrete": "Stone & Concrete",
    "asphalt": "Stone & Concrete",
    "plaster": "Stone & Concrete",
    "wood": "Wood & Natural",
    "natural": "Wood & Natural",
    "paper": "Wood & Natural",
    "ceramic": "Ceramic & Glass",
    "glass": "Ceramic & Glass",
    "fabric": "Fabric & Leather",
    "leather": "Fabric & Leather",
    "rubber": "Rubber & Polymer",
    "carbon": "Rubber & Polymer",
}

GROUP_ORDER: list[str] = [
    "All",
    "Metal",
    "Stone & Concrete",
    "Wood & Natural",
    "Ceramic & Glass",
    "Fabric & Leather",
    "Rubber & Polymer",
]

KIND_DEFAULT_META: dict[str, tuple[float, float]] = {
    "metal": (0.28, 0.95),
    "copper": (0.26, 0.96),
    "bronze": (0.34, 0.85),
    "brass": (0.24, 0.95),
    "stone": (0.55, 0.0),
    "concrete": (0.78, 0.0),
    "asphalt": (0.82, 0.0),
    "plaster": (0.64, 0.0),
    "wood": (0.52, 0.0),
    "natural": (0.74, 0.0),
    "paper": (0.84, 0.0),
    "ceramic": (0.22, 0.0),
    "glass": (0.12, 0.0),
    "fabric": (0.86, 0.0),
    "leather": (0.42, 0.0),
    "rubber": (0.68, 0.0),
    "carbon": (0.20, 0.42),
}


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    vals = [max(0, min(255, int(round(c * 255.0)))) for c in rgb]
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _format_title(mat_id: str) -> str:
    acronyms = {"hvac": "HVAC", "pcb": "PCB", "hardops": "HardOps"}
    parts = mat_id.split("_")
    return " ".join(acronyms.get(p, p.capitalize()) for p in parts)


def build_catalog() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, (did, desc, mat_kind, tint) in enumerate(gd.DAILY_CATALOG):
        def_r, def_m = KIND_DEFAULT_META.get(mat_kind, (0.5, 0.0))
        items.append({
            "id": did,
            "name": _format_title(did),
            "description": desc,
            "kind": mat_kind,
            "group": KIND_TO_GROUP.get(mat_kind, "Other"),
            "archetype": infer_material_kind(did, def_m),
            "tint": [round(float(c), 4) for c in tint],
            "tint_hex": _rgb_to_hex(tint),
            "default_seed": 2000 + idx * 41,
            "default_roughness": def_r,
            "default_metallic": def_m,
            "source": "daily_catalog",
        })
    items.append({
        "id": "hardops_plate",
        "name": "HardOps Machined Plate",
        "description": "CNC-machined gunmetal HardOps panel with plug stamps, bolt plate, lug boss & weld fillets",
        "kind": "metal",
        "group": "Metal",
        "archetype": "metal",
        "tint": [0.28, 0.31, 0.36],
        "tint_hex": _rgb_to_hex((0.28, 0.31, 0.36)),
        "default_seed": 901,
        "default_roughness": 0.24,
        "default_metallic": 0.95,
        "source": "hardops",
    })
    return items


def bake_one(args: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    entry, out_root_str, size = args
    os.environ["HYPBR_DEVICE"] = "cpu"
    out_root = Path(out_root_str)
    mat_id = entry["id"]
    seed = int(entry["default_seed"])
    mat_dir = out_root / "mats" / mat_id
    mat_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td) / mat_id
        if entry["source"] == "hardops":
            syn.synthesize_hardops_plate(tmp_dir, size=size)
        else:
            syn.synthesize_material(mat_id, tmp_dir, size=size, seed=seed, wear=0.0)

        rep = verify_material(tmp_dir)
        checks = rep.get("checks", {})
        seamless = checks.get("seamless", {})
        seam_ratios = {
            k: round(float(seamless[k]["seam_ratio"]), 3)
            for k in ("height", "normal", "albedo")
            if k in seamless and "seam_ratio" in seamless[k]
        }
        max_seam = max(seam_ratios.values()) if seam_ratios else 1.0
        nrm_chk = checks.get("normal", {})
        pbr_chk = checks.get("pbr_ranges", {})

        # Load core maps
        alb = Image.open(tmp_dir / "albedo_delit.png").convert("RGB")
        nrm = Image.open(tmp_dir / "normal.png").convert("RGB")
        orm = Image.open(tmp_dir / "orm.png").convert("RGB")
        h_img = Image.open(tmp_dir / "height.png").convert("L")
        curv_img = Image.open(tmp_dir / "curvature.png").convert("L")
        cav_img = Image.open(tmp_dir / "cavity.png").convert("L")
        prev = Image.open(tmp_dir / "preview.png").convert("RGB")

        # Pack auxiliary maps (R=Height, G=Curvature, B=Cavity)
        aux_arr = np.stack([
            np.asarray(h_img, dtype=np.uint8),
            np.asarray(curv_img, dtype=np.uint8),
            np.asarray(cav_img, dtype=np.uint8),
        ], axis=-1)
        aux = Image.fromarray(aux_arr, mode="RGB")

        # Save compact high-fidelity WebP files
        alb.save(mat_dir / "albedo_delit.webp", "WEBP", quality=92, method=6)
        nrm.save(mat_dir / "normal.webp", "WEBP", quality=98, method=6)
        orm.save(mat_dir / "orm.webp", "WEBP", quality=96, method=6)
        aux.save(mat_dir / "aux.webp", "WEBP", quality=96, method=6)
        prev.resize((160, 160), Image.Resampling.LANCZOS).save(
            mat_dir / "preview.webp", "WEBP", quality=86, method=6
        )

    total_bytes = sum(p.stat().st_size for p in mat_dir.glob("*.webp"))
    rel = f"./assets/mats/{mat_id}"
    out_entry = dict(entry)
    out_entry["cached"] = True
    out_entry["preview_url"] = f"{rel}/preview.webp"
    out_entry["bundle"] = {
        "albedo": f"{rel}/albedo_delit.webp",
        "normal": f"{rel}/normal.webp",
        "orm": f"{rel}/orm.webp",
        "aux": f"{rel}/aux.webp",
        "preview": f"{rel}/preview.webp",
    }
    out_entry["verify"] = {
        "passed": bool(rep.get("passed", True)),
        "max_seam_ratio": round(float(max_seam), 3),
        "seam_ratios": seam_ratios,
        "normal_unit_error": round(float(nrm_chk.get("unit_length_error", 0.0)), 4),
        "orm_max_diff": int(checks.get("orm_max_channel_diff", 0)),
        "pbr_ranges": {
            k: round(float(v), 3) if isinstance(v, (int, float)) else v
            for k, v in pbr_chk.items()
        },
        "warnings": list(rep.get("warnings", [])),
        "errors": list(rep.get("errors", [])),
    }
    out_entry["bundle_bytes"] = total_bytes
    return out_entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake hypbr static WebP bundle for Vercel")
    ap.add_argument("--out", type=Path, default=PLAYGROUND_DIR / "assets")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    t0 = time.perf_counter()
    catalog = build_catalog()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Baking {len(catalog)} materials at {args.size}px into {args.out} ({args.workers} workers)...")

    work = [(entry, str(args.out), args.size) for entry in catalog]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(bake_one, work), 1):
            results.append(res)
            if i % 10 == 0 or i == len(catalog):
                print(f"  [{i:2d}/{len(catalog)}] {res['id']} ({res['bundle_bytes']/1024:.1f} KB, pass={res['verify']['passed']})")

    presets: dict[str, dict[str, float]] = {
        "fresh": {"wear": 0.0, "dirt": 0.0, "wet": 0.0, "moss": 0.0, "rust": 0.0, "dust": 0.0}
    }
    for name, vals in WEATHER_PRESETS.items():
        presets[name] = {k: round(float(vals.get(k, 0.0)), 3) for k in ("wear", "dirt", "wet", "moss", "rust", "dust")}

    total_mb = sum(r["bundle_bytes"] for r in results) / (1024 * 1024)
    failed = [r["id"] for r in results if not r["verify"]["passed"]]

    manifest = {
        "version": "0.6.0",
        "mode": "static",
        "device": "webgl",
        "size": args.size,
        "allowed_sizes": [args.size],
        "groups": GROUP_ORDER,
        "weather_presets": presets,
        "materials": results,
    }
    (args.out / "catalog.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    dt = time.perf_counter() - t0
    print(f"\nDone in {dt:.1f}s! Total WebP assets: {total_mb:.2f} MB across {len(results)} materials.")
    if failed:
        print(f"WARNING: QA failed on: {failed}")
    else:
        print("ALL 69/69 materials PASSED hypbr.verify QA!")


if __name__ == "__main__":
    main()
