"""Command-line interface for hypbr."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import normal_graph as ng
from .blend import blend_materials
from .exporters import SUPPORTED_TARGETS, export_material
from .preview import serve_preview
from .synthesize import list_materials, synthesize_all, synthesize_hardops_plate, synthesize_material
from .verify import verify_material
from .weathering import WEATHER_ARCHETYPES, WEATHER_PRESETS, weather_material


def cmd_list(_args: argparse.Namespace) -> int:
    for item in list_materials():
        print(f"{item['id']:34s}  [{item['category']:8s}]  {item['description']}")
    return 0


def cmd_atlas_search(args: argparse.Namespace) -> int:
    from .atlas import search_atlas

    res = search_atlas(
        query=" ".join(args.query) or None, category=args.category, scene_role=args.role,
        metallic=(True if args.metallic else (False if args.dielectric else None)),
        max_roughness=args.max_roughness, limit=args.limit, base_url=args.base_url,
    )
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    for r in res:
        w, h = r["size_m"]
        print(f"{r['score']:6.1f}  {r['id']:34s}  {r['category']:22s}  {w:g}x{h:g} m  {', '.join(r['best_objects'][:3])}")
    if not res:
        print("No matches.")
    return 0


def cmd_atlas_info(args: argparse.Namespace) -> int:
    from .atlas import get_material_spec

    spec = get_material_spec(args.material, base_url=args.base_url)
    if args.snippet:
        print(spec["snippets"]["threejs" if args.snippet == "threejs" else "blender_bpy"])
        return 0
    if not args.full:
        spec = {k: v for k, v in spec.items() if k != "snippets"}
    print(json.dumps(spec, indent=2))
    return 0


def cmd_atlas_recommend(args: argparse.Namespace) -> int:
    from .atlas import recommend_for_object

    size = tuple(args.surface) if args.surface else None
    rec = recommend_for_object(" ".join(args.object), style=args.style, environment=args.environment,
                               limit=args.limit, surface_size_m=size, base_url=args.base_url)
    if args.json:
        print(json.dumps(rec, indent=2))
        return 0
    print(f"Object: {rec['object']}  style={rec['style']}  environment={rec['environment']}")
    for m in rec["matches"]:
        wp = m["weathering"]["hypbr_params"]
        print(f"  {m['id']:34s} repeat={m['uv_repeat'][0]:g}x{m['uv_repeat'][1]:g} on {m['surface_size_m'][0]:g}x{m['surface_size_m'][1]:g} m"
              f"  proj={m['projection']['recommended']}  weather(wear={wp['wear']}, dirt={wp['dirt']}, wet={wp['wet']}, moss={wp['moss']}, rust={wp['rust']})")
    return 0


def cmd_atlas_calibrate(args: argparse.Namespace) -> int:
    from .calibration import fit_atlas

    doc = fit_atlas([Path(p) for p in args.refs], [Path(p) for p in (args.roughness or [])], args.baked,
                    out_path=args.out, strength=args.strength, refine_iters=args.refine,
                    workers=max(1, min(4, args.workers)))
    n = len(doc["materials"])
    print(f"Calibrated {n} materials against {doc['n_references']} reference scans -> {args.out}")
    return 0


def cmd_atlas_export(args: argparse.Namespace) -> int:
    from .atlas import export_atlas_files

    t0 = time.time()
    info = export_atlas_files(args.out_dir, base_url=args.base_url)
    print(f"Exported atlas ({info['materials']} materials, {info['files']} files) to {info['out_dir']} in {time.time() - t0:.1f}s")
    return 0


def _resolve_targets(raw: str) -> list[str]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if "all" in parts:
        return list(SUPPORTED_TARGETS)
    for t in parts:
        if t not in SUPPORTED_TARGETS:
            raise ValueError(f"Unsupported target '{t}'. Available: {', '.join(SUPPORTED_TARGETS)}, all")
    return parts


def cmd_synth(args: argparse.Namespace) -> int:
    t0 = time.time()
    w_kw = dict(
        weather_preset=args.weather_preset,
        dirt=args.dirt,
        wet=args.wet,
        moss=args.moss,
        rust=args.rust,
        dust=args.dust,
    )
    if args.material == "all":
        paths = synthesize_all(args.out, size=args.size, wear=args.wear, **w_kw)
        print(f"Synthesized {len(paths)} seamless 15-map PBR sets in {time.time() - t0:.1f}s -> {args.out}")
    elif args.material == "hardops_plate" and not (args.weather_preset or args.wear > 1e-4 or any(
        v is not None and v > 1e-4 for v in (args.dirt, args.wet, args.moss, args.rust, args.dust)
    )):
        paths = [synthesize_hardops_plate(args.out / "hardops_plate", size=args.size)]
        print(f"Synthesized hardops_plate in {time.time() - t0:.2f}s -> {paths[0]}")
    else:
        paths = [
            synthesize_material(
                args.material, args.out / args.material, size=args.size, seed=args.seed, wear=args.wear, **w_kw
            )
        ]
        preset_str = f", preset={args.weather_preset}" if args.weather_preset else ""
        print(
            f"Synthesized {args.material} (15 maps, {args.size}x{args.size}, wear={args.wear:.2f}{preset_str}) "
            f"in {time.time() - t0:.2f}s -> {paths[0]}"
        )

    if args.export:
        targets = _resolve_targets(args.export)
        for mat_dir in paths:
            for tgt in targets:
                exp_dir = mat_dir / "exports" / tgt
                export_material(mat_dir, exp_dir, target=tgt, name=mat_dir.name)
            print(f"  Exported {mat_dir.name} -> {', '.join(targets)} ({mat_dir / 'exports'})")
    return 0


def cmd_weather(args: argparse.Namespace) -> int:
    t0 = time.time()
    out = weather_material(
        args.src,
        args.out,
        preset=args.preset,
        wear=args.wear,
        dirt=args.dirt,
        wet=args.wet,
        moss=args.moss,
        rust=args.rust,
        dust=args.dust,
        material_kind=args.kind,
        seed=args.seed,
        backend=args.backend,
        patch_dir=args.patch_dir,
        vit_patch=args.vit_patch,
    )
    print(f"Weathered {args.src.name} (backend={args.backend}, preset={args.preset or 'custom'}) in {time.time() - t0:.2f}s -> {out}")
    if args.export:
        targets = _resolve_targets(args.export)
        for tgt in targets:
            exp_dir = out / "exports" / tgt
            export_material(out, exp_dir, target=tgt, name=out.name)
        print(f"  Exported {out.name} -> {', '.join(targets)} ({out / 'exports'})")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    targets = _resolve_targets(args.target)
    src = Path(args.src)
    mat_name = args.name or src.name
    for tgt in targets:
        dst = args.out if (len(targets) == 1 and not args.subdir) else (args.out / tgt)
        export_material(src, dst, target=tgt, name=mat_name)
        print(f"Exported [{tgt:10s}] -> {dst}")
    return 0


def cmd_blend(args: argparse.Namespace) -> int:
    t0 = time.time()
    out = blend_materials(args.dir_a, args.dir_b, args.out, factor=args.factor, contrast=args.contrast, mode=args.mode)
    print(f"Blended {args.dir_a.name} + {args.dir_b.name} (mode={args.mode}, factor={args.factor:.2f}) in {time.time() - t0:.2f}s -> {out}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    serve_preview(args.dir, host=args.host, port=args.port, open_browser=not args.no_open)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    report = verify_material(args.dir, seam_limit=args.seam_limit)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "PASS" if report["passed"] else "FAIL"
        print(f"[{status}] {report['material']} ({report['checks'].get('resolution', '?')})")
        if "seamless" in report["checks"]:
            sm = report["checks"]["seamless"]
            print("  Seam outlier score (height / normal / albedo): "
                  f"{sm['height']['outlier_score']:.3f} / {sm['normal']['outlier_score']:.3f} / {sm['albedo']['outlier_score']:.3f}")
        if "normal" in report["checks"]:
            n = report["checks"]["normal"]
            print(f"  Normal unit err: {n['unit_length_error']:.4f} | +Z: {n['positive_z_fraction']:.4f} | Curl: {n['curl_residual_ratio']:.3f}")
        for w in report["warnings"]:
            print(f"  WARN: {w}")
        for e in report["errors"]:
            print(f"  ERR:  {e}")
    return 0 if report["passed"] else 1


def cmd_dump_preset(args: argparse.Namespace) -> int:
    if args.preset not in ng.PRESETS:
        print(f"Unknown preset '{args.preset}'. Available: {', '.join(sorted(ng.PRESETS))}", file=sys.stderr)
        return 1
    print(json.dumps(ng.PRESETS[args.preset], indent=2))
    return 0


def cmd_photo(args: argparse.Namespace) -> int:
    argv = [str(args.image), "--out", str(args.out), "--preset", args.preset, "--depth-model", args.depth_model]
    if args.graph:
        argv += ["--graph", str(args.graph)]
    if args.tileable:
        argv.append("--tileable")
    if args.dx:
        argv.append("--dx")
    from ._backend import has_torch, require_torch

    if not has_torch():
        try:
            require_torch("`hypbr photo` (neural photo -> PBR)")
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    old_argv = sys.argv
    try:
        sys.argv = ["hypbr-photo", *argv]
        ng.main()
    finally:
        sys.argv = old_argv
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hypbr",
        description="Hybrid Neural + Exact Periodic Procedural PBR Material Synthesizer (15 calibrated PBR maps).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List all 500 built-in atlas materials")
    p_list.set_defaults(func=cmd_list)

    from .atlas import DEFAULT_BASE_URL

    p_atlas = sub.add_parser("atlas", help="Query and export the 500-material agent-ready PBR atlas")
    asub = p_atlas.add_subparsers(dest="atlas_command", required=True)
    a_s = asub.add_parser("search", help="Ranked search by text, object, role and physical filters")
    a_s.add_argument("query", nargs="*", help="Free text, e.g. 'brick wall'")
    a_s.add_argument("--category", default=None, help="Category or subcategory slug filter")
    a_s.add_argument("--role", default=None, help="Scene role filter (wall, floor, road, furniture, ...)")
    a_s.add_argument("--metallic", action="store_true", help="Only metallic materials")
    a_s.add_argument("--dielectric", action="store_true", help="Only non-metallic materials")
    a_s.add_argument("--max-roughness", type=float, default=None, help="Maximum mean roughness")
    a_s.add_argument("--limit", type=int, default=20)
    a_s.add_argument("--json", action="store_true")
    a_s.add_argument("--base-url", default=DEFAULT_BASE_URL)
    a_s.set_defaults(func=cmd_atlas_search)
    a_i = asub.add_parser("info", help="Print the agent spec card for one material")
    a_i.add_argument("material", help="Atlas material ID")
    a_i.add_argument("--full", action="store_true", help="Include Three.js / Blender / glTF snippets")
    a_i.add_argument("--snippet", choices=("threejs", "blender"), default=None, help="Print only one snippet")
    a_i.add_argument("--base-url", default=DEFAULT_BASE_URL)
    a_i.set_defaults(func=cmd_atlas_info)
    a_r = asub.add_parser("recommend", help="Recommend materials for an object with UV repeat & weathering advice")
    a_r.add_argument("object", nargs="+", help="Object name, e.g. 'kitchen counter'")
    a_r.add_argument("--style", default=None)
    a_r.add_argument("--environment", default=None, help="interior, exterior, coastal, rainy, desert, winter, forest, urban, abandoned, space")
    a_r.add_argument("--surface", type=float, nargs=2, default=None, metavar=("W_M", "H_M"), help="Surface size in meters")
    a_r.add_argument("--limit", type=int, default=5)
    a_r.add_argument("--json", action="store_true")
    a_r.add_argument("--base-url", default=DEFAULT_BASE_URL)
    a_r.set_defaults(func=cmd_atlas_recommend)
    a_c = asub.add_parser("calibrate", help="Fit reference-photo calibration from real PBR scans (*.npz) vs uncalibrated bakes")
    a_c.add_argument("--refs", nargs="+", required=True, help="Directories of reference scans (*.npz with albedo/normal)")
    a_c.add_argument("--roughness", nargs="*", default=None, help="Directories of reference roughness PNGs (<scan_id>.png)")
    a_c.add_argument("--baked", type=Path, required=True, help="Uncalibrated 512px bake root (<root>/<id>/albedo_delit.webp ...)")
    a_c.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "atlas_calibration.json")
    a_c.add_argument("--strength", type=float, default=0.7, help="Exponent toward the reference statistics (0..1)")
    a_c.add_argument("--refine", type=int, default=1, help="Closed-loop refinement passes (each re-synthesizes calibrated materials)")
    a_c.add_argument("--workers", type=int, default=2, help="CPU worker processes for refinement (max 4)")
    a_c.set_defaults(func=cmd_atlas_calibrate)
    a_e = asub.add_parser("export", help="Write llms.txt, agent.md, atlas.json and atlas/ JSON shards")
    a_e.add_argument("out_dir", type=Path)
    a_e.add_argument("--base-url", default=DEFAULT_BASE_URL)
    a_e.set_defaults(func=cmd_atlas_export)

    p_synth = sub.add_parser("synth", help="Synthesize seamless procedural 15-map PBR materials")
    p_synth.add_argument("material", help="Atlas material ID (see `hypbr list` / `hypbr atlas search`) or `all` (500 materials)")
    p_synth.add_argument("--out", type=Path, default=Path("./output/pbr"), help="Output directory")
    p_synth.add_argument("--size", type=int, default=512, help="Square texture resolution (default: 512)")
    p_synth.add_argument("--seed", type=int, default=None, help="Optional deterministic seed override")
    p_synth.add_argument("--wear", type=float, default=0.0, help="Weathering intensity 0.0 (fresh) to 1.0 (weathered)")
    p_synth.add_argument(
        "--weather-preset",
        choices=sorted(WEATHER_PRESETS),
        default=None,
        help=f"Optional multi-layer weathering preset ({', '.join(sorted(WEATHER_PRESETS))})",
    )
    p_synth.add_argument("--dirt", type=float, default=None, help="Optional crevice dirt/grime intensity [0, 1]")
    p_synth.add_argument("--wet", type=float, default=None, help="Optional rain & puddle moisture intensity [0, 1]")
    p_synth.add_argument("--moss", type=float, default=None, help="Optional moss & lichen growth intensity [0, 1]")
    p_synth.add_argument("--rust", type=float, default=None, help="Optional rust/verdigris/runoff intensity [0, 1]")
    p_synth.add_argument("--dust", type=float, default=None, help="Optional settled dust/ash intensity [0, 1]")
    p_synth.add_argument(
        "--export",
        type=str,
        default=None,
        help=f"Comma-separated engine export targets ({', '.join(SUPPORTED_TARGETS)}, all)",
    )
    p_synth.set_defaults(func=cmd_synth)

    p_w = sub.add_parser("weather", help="Apply optional multi-mode physical or neural weathering to a hypbr material")
    p_w.add_argument("src", type=Path, help="Source hypbr material directory")
    p_w.add_argument("--out", type=Path, required=True, help="Output directory for weathered material")
    p_w.add_argument(
        "--preset",
        choices=sorted(WEATHER_PRESETS),
        default=None,
        help=f"Named weathering preset ({', '.join(sorted(WEATHER_PRESETS))})",
    )
    p_w.add_argument("--wear", type=float, default=None, help="Convex edge wear intensity [0, 1]")
    p_w.add_argument("--dirt", type=float, default=None, help="Crevice dirt/grime intensity [0, 1]")
    p_w.add_argument("--wet", type=float, default=None, help="Rain & standing puddle moisture intensity [0, 1]")
    p_w.add_argument("--moss", type=float, default=None, help="Biological moss & lichen intensity [0, 1]")
    p_w.add_argument("--rust", type=float, default=None, help="Rust, verdigris & gravity runoff intensity [0, 1]")
    p_w.add_argument("--dust", type=float, default=None, help="Settled mineral dust intensity [0, 1]")
    p_w.add_argument(
        "--kind",
        choices=WEATHER_ARCHETYPES,
        default=None,
        help="Optional physical material archetype override (auto-inferred if omitted)",
    )
    p_w.add_argument("--seed", type=int, default=42, help="Deterministic weathering seed (default: 42)")
    p_w.add_argument(
        "--backend",
        choices=("procedural", "neural"),
        default="procedural",
        help="Mask backend: zero-GPU 'procedural' (default) or Single-Encode DepthPro2-S 'neural'",
    )
    p_w.add_argument("--patch-dir", type=Path, default=None, help="Directory of neural *_patch.pt branch weights")
    p_w.add_argument("--vit-patch", type=Path, default=None, help="Optional DepthPro2-S ViT backbone patch (.pt)")
    p_w.add_argument(
        "--export",
        type=str,
        default=None,
        help=f"Comma-separated engine export targets ({', '.join(SUPPORTED_TARGETS)}, all)",
    )
    p_w.set_defaults(func=cmd_weather)

    p_exp = sub.add_parser("export", help="Export an existing hypbr material directory to game engines & 3D standards")
    p_exp.add_argument("src", type=Path, help="Source hypbr material directory")
    p_exp.add_argument(
        "--target",
        required=True,
        help=f"Comma-separated targets: {', '.join(SUPPORTED_TARGETS)}, all",
    )
    p_exp.add_argument("--out", type=Path, required=True, help="Destination output directory")
    p_exp.add_argument("--name", type=str, default=None, help="Optional material name override")
    p_exp.add_argument("--subdir", action="store_true", help="Always write into <out>/<target>/ even for a single target")
    p_exp.set_defaults(func=cmd_export)

    p_blend = sub.add_parser("blend", help="Physically blend two hypbr materials via height-lerp, cavity, or linear mode")
    p_blend.add_argument("dir_a", type=Path, help="Base material directory A")
    p_blend.add_argument("dir_b", type=Path, help="Overlay material directory B")
    p_blend.add_argument("--out", type=Path, required=True, help="Output directory for blended material")
    p_blend.add_argument("--factor", type=float, default=0.5, help="Blend amount of B over A in [0, 1] (default: 0.5)")
    p_blend.add_argument("--contrast", type=float, default=0.15, help="Height transition sharpness (default: 0.15)")
    p_blend.add_argument("--mode", choices=("height", "cavity", "linear"), default="height", help="Blend mode")
    p_blend.set_defaults(func=cmd_blend)

    p_prev = sub.add_parser("preview", help="Launch a local interactive Three.js PBR studio preview")
    p_prev.add_argument("dir", type=Path, help="Material directory to inspect")
    p_prev.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_prev.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    p_prev.add_argument("--no-open", action="store_true", help="Do not automatically open the browser")
    p_prev.set_defaults(func=cmd_preview)

    p_ver = sub.add_parser("verify", help="Verify seamlessness, normal validity, ORM packing, and PBR physical ranges")
    p_ver.add_argument("dir", type=Path, help="Material directory to verify")
    p_ver.add_argument("--json", action="store_true", help="Print full JSON report")
    p_ver.add_argument("--seam-limit", type=float, default=1.3,
                       help="Max border jump / 99th-percentile jump of all other rows or columns (tileable ~1)")
    p_ver.set_defaults(func=cmd_verify)

    p_dump = sub.add_parser("dump-preset", help="Dump a built-in 47-node PBR graph preset as JSON")
    p_dump.add_argument("preset", nargs="?", default="v13-default", help="Preset name (default: v13-default)")
    p_dump.set_defaults(func=cmd_dump_preset)

    p_photo = sub.add_parser("photo", help="Convert a macro photograph to 15 PBR maps via neural relief + DeepBump + 47-node graph")
    p_photo.add_argument("image", type=Path, help="Path to input photograph")
    p_photo.add_argument("--out", type=Path, default=Path("./output/photo_pbr"), help="Output directory")
    p_photo.add_argument("--preset", default="v13-default", help="Graph preset name")
    p_photo.add_argument("--graph", type=Path, default=None, help="Optional custom JSON graph file")
    p_photo.add_argument("--tileable", action="store_true", help="Enable circular-padded seamless evaluation")
    p_photo.add_argument("--dx", action="store_true", help="Output DirectX (-Y) normals instead of OpenGL (+Y)")
    p_photo.add_argument("--depth-model", default="da2", choices=("da2", "da3mono", "depthpro"),
                         help="Macro relief model: da2 = Depth Anything V2 Small + hypbr patch (Apache-2.0 base, default); "
                              "da3mono = Depth Anything 3 Mono Large (Apache-2.0, optional HYPBR_DA3_PATCH); "
                              "depthpro = Apple Depth Pro 2 (research-only license, needs depth_pro_2)")
    p_photo.set_defaults(func=cmd_photo)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
