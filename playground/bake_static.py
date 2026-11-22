#!/usr/bin/env python3
"""Bake all 500 hypbr atlas materials into a static, CORS-friendly WebP bundle for edge hosting.

Outputs (playground root unless --root is given):
  - `assets/mats/<id>/albedo_delit.webp`  512x512 sRGB base color
  - `assets/mats/<id>/normal.webp`        512x512 OpenGL normal
  - `assets/mats/<id>/orm.webp`           512x512 R=AO, G=Roughness, B=Metallic
  - `assets/mats/<id>/aux.webp`           512x512 R=Height, G=Curvature, B=Cavity
  - `assets/mats/<id>/preview.webp`       256x256 studio sphere thumbnail
  - `assets/catalog.json`                 lean playground catalog (search fields + QA)
  - `llms.txt`, `agent.md`, `atlas.json`, `atlas/index.json`, `atlas/categories/*.json`, `atlas/mats/*.json`
  - `<hypbr>/hypbr/data/atlas_measurements.json` (measured roughness/metallic ranges + QA, shipped in the wheel)

Crash-safety: strictly CPU (HYPBR_DEVICE=cpu, no MPS), <= 4 worker processes (default 3) that are
recycled every few materials, 1-2 torch threads per worker, gc after every material. Every material
must pass `hypbr.verify` (seams / NaN / ranges) or the bake exits non-zero.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Force CPU-only before importing hypbr/torch
os.environ["HYPBR_DEVICE"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")

PLAYGROUND_DIR = Path(__file__).resolve().parent
REPO_CANDIDATES = [PLAYGROUND_DIR.parent, PLAYGROUND_DIR.parent / "hypbr"]
REPO_DIR = None
for cand in REPO_CANDIDATES:
    if (cand / "hypbr" / "synthesize.py").exists():
        REPO_DIR = cand
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        break

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

BAKE_VERSION = 3
Q_MAPS = 86
Q_NORMAL = 88
Q_PREVIEW = 82
PREVIEW_PX = 256
VERCEL_HOBBY_SOFT_LIMIT_MB = 95.0


def _signature(entry: dict[str, Any], size: int) -> str:
    raw = json.dumps({"g": entry["generator"], "size": size, "v": BAKE_VERSION,
                      "q": [Q_MAPS, Q_NORMAL, Q_PREVIEW, PREVIEW_PX]}, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def bake_one(args: tuple[str, str, int]) -> dict[str, Any]:
    mat_id, out_root_str, size = args
    os.environ["HYPBR_DEVICE"] = "cpu"
    import torch

    torch.set_num_threads(2)
    from hypbr import synthesize as syn
    from hypbr.verify import verify_material

    t0 = time.perf_counter()
    mat_dir = Path(out_root_str) / "mats" / mat_id
    mat_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td) / mat_id
        syn.synthesize_material(mat_id, tmp_dir, size=size)
        rep = verify_material(tmp_dir)
        checks = rep.get("checks", {})
        seamless = checks.get("seamless", {})
        seam_ratios = {k: round(float(seamless[k]["seam_ratio"]), 3) for k in ("height", "normal", "albedo")
                       if k in seamless and "seam_ratio" in seamless[k]}

        alb = Image.open(tmp_dir / "albedo_delit.png").convert("RGB")
        nrm = Image.open(tmp_dir / "normal.png").convert("RGB")
        orm = Image.open(tmp_dir / "orm.png").convert("RGB")
        aux = Image.fromarray(np.stack([
            np.asarray(Image.open(tmp_dir / "height.png").convert("L"), dtype=np.uint8),
            np.asarray(Image.open(tmp_dir / "curvature.png").convert("L"), dtype=np.uint8),
            np.asarray(Image.open(tmp_dir / "cavity.png").convert("L"), dtype=np.uint8),
        ], axis=-1), mode="RGB")
        prev = Image.open(tmp_dir / "preview.png").convert("RGB")

        orm_a = np.asarray(orm, dtype=np.float32) / 255.0
        alb_a = np.asarray(alb, dtype=np.float32)
        finite = bool(np.isfinite(orm_a).all() and np.isfinite(alb_a).all())
        rough, metal = orm_a[..., 1], orm_a[..., 2]
        meas = {
            "roughness_range": [round(float(np.percentile(rough, 5)), 3), round(float(np.percentile(rough, 95)), 3)],
            "metallic_range": [round(float(np.percentile(metal, 5)), 3), round(float(np.percentile(metal, 95)), 3)],
            "metallic_mean": round(float(metal.mean()), 3),
            "roughness_mean": round(float(rough.mean()), 3),
            "albedo_mean_srgb": [int(round(float(c))) for c in alb_a.reshape(-1, 3).mean(axis=0)],
            "verify": {
                "passed": bool(rep.get("passed", False)) and finite,
                "max_seam_ratio": round(max(seam_ratios.values()) if seam_ratios else 0.0, 3),
                "seam_ratios": seam_ratios,
                "normal_unit_error": round(float(checks.get("normal", {}).get("unit_length_error", 0.0)), 4),
                "errors": list(rep.get("errors", [])) + ([] if finite else ["non-finite map values"]),
                "warnings": list(rep.get("warnings", [])),
            },
        }

        alb.save(mat_dir / "albedo_delit.webp", "WEBP", quality=Q_MAPS, method=6)
        nrm.save(mat_dir / "normal.webp", "WEBP", quality=Q_NORMAL, method=6)
        orm.save(mat_dir / "orm.webp", "WEBP", quality=Q_MAPS, method=6)
        aux.save(mat_dir / "aux.webp", "WEBP", quality=Q_MAPS, method=6)
        prev.resize((PREVIEW_PX, PREVIEW_PX), Image.Resampling.LANCZOS).save(
            mat_dir / "preview.webp", "WEBP", quality=Q_PREVIEW, method=6)
        del alb, nrm, orm, aux, prev, orm_a, alb_a, rough, metal
    gc.collect()
    meas["bytes"] = sum(p.stat().st_size for p in mat_dir.glob("*.webp"))
    meas["bake_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    meas["worker_peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
    return {"id": mat_id, **meas}


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake the 500-material hypbr atlas into a static WebP bundle")
    ap.add_argument("--root", type=Path, default=PLAYGROUND_DIR, help="Static site root (default: playground/)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=3, help="CPU worker processes (capped at 4)")
    ap.add_argument("--base-url", default="https://hypbr.vercel.app")
    ap.add_argument("--only", default=None, help="Comma-separated material IDs to (re)bake")
    ap.add_argument("--force", action="store_true", help="Rebake even if the cached signature matches")
    ap.add_argument("--no-measurements", action="store_true", help="Do not update hypbr/data/atlas_measurements.json")
    args = ap.parse_args()
    workers = max(1, min(4, int(args.workers)))

    from hypbr import atlas as atl
    from hypbr import atlas_catalog as ac
    from hypbr.weathering import WEATHER_PRESETS

    t0 = time.perf_counter()
    assets = args.root / "assets"
    (assets / "mats").mkdir(parents=True, exist_ok=True)
    cache_path = assets / ".bake_cache.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}

    catalog = ac.get_catalog()
    ids = [e["id"] for e in catalog]
    sigs = {e["id"]: _signature(e, args.size) for e in catalog}
    wanted = set(args.only.split(",")) if args.only else set(ids)
    todo = []
    for mid in ids:
        files_ok = all((assets / "mats" / mid / f).is_file() for f in atl.MAP_FILES.values())
        cached = cache.get(mid)
        if mid in wanted and (args.force or not files_ok or not cached or cached.get("sig") != sigs[mid]):
            todo.append(mid)

    # Remove stale material directories no longer in the catalog
    for d in (assets / "mats").iterdir():
        if d.is_dir() and d.name not in sigs:
            shutil.rmtree(d)
            cache.pop(d.name, None)

    print(f"Baking {len(todo)}/{len(ids)} materials at {args.size}px with {workers} CPU workers "
          f"(cached: {len(ids) - len(todo)})...", flush=True)
    peak = 0.0
    if todo:
        # No max_tasks_per_child: recycling workers can deadlock/collapse the pool. Each task frees its
        # arrays and calls gc.collect(), keeping worker RSS flat (~0.6 GB at 512px).
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for i, res in enumerate(pool.map(bake_one, [(m, str(assets), args.size) for m in todo], chunksize=1), 1):
                res["sig"] = sigs[res["id"]]
                cache[res["id"]] = res
                peak = max(peak, res["worker_peak_rss_mb"])
                if i % 20 == 0 or i == len(todo) or not res["verify"]["passed"]:
                    print(f"  [{i:3d}/{len(todo)}] {res['id']:34s} {res['bytes'] / 1024:6.1f} KB  "
                          f"pass={res['verify']['passed']}  seam={res['verify']['max_seam_ratio']}  rss={res['worker_peak_rss_mb']}MB",
                          flush=True)
                if i % 25 == 0:
                    cache_path.write_text(json.dumps(cache, separators=(",", ":")))
        cache_path.write_text(json.dumps(cache, separators=(",", ":")))

    missing = [m for m in ids if m not in cache]
    if missing:
        raise SystemExit(f"Missing bake results for: {missing[:10]}")
    meas = {m: {k: v for k, v in cache[m].items() if k not in ("sig", "bake_ms", "worker_peak_rss_mb")} for m in ids}
    failed = [m for m in ids if not meas[m]["verify"]["passed"]]

    # Measurements shipped with the Python package (so offline spec cards carry measured ranges)
    if not args.no_measurements and REPO_DIR is not None:
        mp = REPO_DIR / "hypbr" / "data" / "atlas_measurements.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        slim = {m: {k: meas[m][k] for k in ("roughness_range", "metallic_range", "metallic_mean", "roughness_mean", "albedo_mean_srgb")}
                | {"verify": {k: meas[m]["verify"][k] for k in ("passed", "max_seam_ratio")}} for m in ids}
        mp.write_text(json.dumps({"version": ac.ATLAS_VERSION, "size": args.size, "materials": slim},
                                 indent=None, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        atl.clear_measurement_cache()

    # Playground catalog (lean rows + QA)
    rows = atl.playground_entries("./assets/mats")
    for r in rows:
        v = meas[r["id"]]["verify"]
        r["cached"] = True
        r["verify"] = {"passed": v["passed"], "max_seam_ratio": v["max_seam_ratio"], "seam_ratios": v["seam_ratios"],
                       "normal_unit_error": v["normal_unit_error"], "warnings": v["warnings"], "errors": v["errors"]}
        r["bundle_bytes"] = meas[r["id"]]["bytes"]
    presets = {"fresh": {k: 0.0 for k in ("wear", "dirt", "wet", "moss", "rust", "dust")}}
    for name, vals in WEATHER_PRESETS.items():
        presets[name] = {k: round(float(vals.get(k, 0.0)), 3) for k in ("wear", "dirt", "wet", "moss", "rust", "dust")}
    counts = {c["name"]: c["count"] for c in ac.CATEGORIES}
    manifest = {
        "version": ac.ATLAS_VERSION, "mode": "static", "device": "webgl", "size": args.size, "allowed_sizes": [args.size],
        "groups": ["All"] + [c["name"] for c in ac.CATEGORIES], "group_counts": counts,
        "categories": [{"slug": c["slug"], "name": c["name"], "count": c["count"]} for c in ac.CATEGORIES],
        "weather_presets": presets, "atlas": {"llms_txt": "/llms.txt", "agent_md": "/agent.md", "manifest": "/atlas.json",
                                              "index": "/atlas/index.json", "spec": "/atlas/mats/{id}.json"},
        "materials": rows,
    }
    (assets / "catalog.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")

    # Agent atlas files at the site root
    info = atl.export_atlas_files(args.root, base_url=args.base_url, measurements=meas)

    # Bundle size / file-count report against Vercel Hobby limits
    total_bytes, n_files = 0, 0
    for sub in ("assets", "atlas"):
        for p in (args.root / sub).rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                total_bytes += p.stat().st_size
                n_files += 1
    for f in ("index.html", "app.js", "styles.css", "static_engine.js", "llms.txt", "agent.md", "atlas.json", "vercel.json"):
        p = args.root / f
        if p.is_file():
            total_bytes += p.stat().st_size
            n_files += 1
    tex_mb = sum(meas[m]["bytes"] for m in ids) / (1024 * 1024)
    total_mb = total_bytes / (1024 * 1024)
    dt = time.perf_counter() - t0
    print(f"\nDone in {dt:.1f}s. Textures: {tex_mb:.1f} MB; static bundle: {total_mb:.1f} MB in {n_files} files "
          f"(worker peak RSS this run: {peak:.0f} MB). Atlas JSON: {info['files']} files.")
    if total_mb > VERCEL_HOBBY_SOFT_LIMIT_MB:
        print(f"WARNING: bundle {total_mb:.1f} MB exceeds the {VERCEL_HOBBY_SOFT_LIMIT_MB:.0f} MB soft limit for Vercel Hobby uploads.")
    if failed:
        print(f"QA FAILED on {len(failed)} materials: {failed}")
        raise SystemExit(1)
    print(f"ALL {len(ids)}/{len(ids)} materials PASSED hypbr.verify QA (seams, NaN, ranges).")


if __name__ == "__main__":
    main()
