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
        print(f"{item['id']:30s}  [{item['category']:8s}]  {item['description']}")
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
    elif args.material == "hardops_plate":
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
    report = verify_material(args.dir, seam_ratio_limit=args.seam_limit)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "PASS" if report["passed"] else "FAIL"
        print(f"[{status}] {report['material']} ({report['checks'].get('resolution', '?')})")
        if "seamless" in report["checks"]:
            sh = report["checks"]["seamless"]["height"]["seam_ratio"]
            sn = report["checks"]["seamless"]["normal"]["seam_ratio"]
            print(f"  Seam ratio (height / normal): {sh:.3f} / {sn:.3f}")
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
    argv = [str(args.image), "--out", str(args.out), "--preset", args.preset]
    if args.graph:
        argv += ["--graph", str(args.graph)]
    if args.tileable:
        argv.append("--tileable")
    if args.dx:
        argv.append("--dx")
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

    p_list = sub.add_parser("list", help="List built-in procedural materials")
    p_list.set_defaults(func=cmd_list)

    p_synth = sub.add_parser("synth", help="Synthesize seamless procedural 15-map PBR materials")
    p_synth.add_argument("material", help="Material ID (see `hypbr list`), `hardops_plate`, or `all`")
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
    p_ver.add_argument("--seam-limit", type=float, default=2.2, help="Max allowed border-to-interior seam ratio")
    p_ver.set_defaults(func=cmd_verify)

    p_dump = sub.add_parser("dump-preset", help="Dump a built-in 47-node PBR graph preset as JSON")
    p_dump.add_argument("preset", nargs="?", default="v13-default", help="Preset name (default: v13-default)")
    p_dump.set_defaults(func=cmd_dump_preset)

    p_photo = sub.add_parser("photo", help="Convert a macro photograph to 15 PBR maps via Depth Pro + DeepBump + 47-node graph")
    p_photo.add_argument("image", type=Path, help="Path to input photograph")
    p_photo.add_argument("--out", type=Path, default=Path("./output/photo_pbr"), help="Output directory")
    p_photo.add_argument("--preset", default="v13-default", help="Graph preset name")
    p_photo.add_argument("--graph", type=Path, default=None, help="Optional custom JSON graph file")
    p_photo.add_argument("--tileable", action="store_true", help="Enable circular-padded seamless evaluation")
    p_photo.add_argument("--dx", action="store_true", help="Output DirectX (-Y) normals instead of OpenGL (+Y)")
    p_photo.set_defaults(func=cmd_photo)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
