# hypbr

[![CI](https://github.com/wrong-nebula/hypbr/actions/workflows/ci.yml/badge.svg)](https://github.com/wrong-nebula/hypbr/actions/workflows/ci.yml)
[![Live 3D Playground](https://img.shields.io/badge/Live_3D_Playground-hypbr.vercel.app-d97742?style=flat&logo=vercel)](https://hypbr.vercel.app)

**500-Material Agent-Ready PBR Atlas + Hybrid Neural / Exact Periodic Procedural PBR Synthesizer**

> **Point any AI agent at one URL:** **[hypbr.vercel.app/llms.txt](https://hypbr.vercel.app/llms.txt)** — 500 physically calibrated, mathematically seamless PBR materials with CORS-enabled 512px WebP maps and JSON spec cards that tell the agent *what objects each material belongs on*, the *exact real-world tile size in meters*, *UV vs triplanar projection*, *engine shader wiring*, *weathering presets* and *pairing partners*.
>
> **Try it live:** **[hypbr.vercel.app](https://hypbr.vercel.app)** — interactive Three.js `r187` PBR studio over all 500 materials with an Atlas Inspector, real-time 6-channel weathering, HeightLerp A×B blending and PNG ZIP export.

`hypbr` synthesizes seamlessly tileable, physically calibrated **15-map PBR material sets** (glTF 2.0 ORM, 16-bit height, OpenGL/DirectX tangent normals, 8-azimuth horizon-based AO, bent normals, curvature, anisotropy, specular, and subsurface thickness) at arbitrary resolution with zero border seams.

- **500-Material Agent Atlas**: 69 hand-tuned flagship generators + 431 curated parametric materials from 28 periodic generator families across 10 scene categories (masonry, wood, stone/marble/concrete, tiles, metals, fabrics, terrain, glass, rubber/roads, roofing/tech/sci-fi). Every material passes `hypbr.verify` (zero wrap seams, finite maps, physical ranges). v1.1 adds a seamless micro-detail pass (mineral grain, aggregate speckle, wood fibers, brush streaks, textile fuzz), organic ground cover (leaf litter, needles, mulch, grass, moss, lichen, soil, snow) and a gated multi-scale marble vein network.
- **Reference-Photo Calibration**: 400 of the 500 materials are calibrated against 449 real CC0 photogrammetry / photometric-stereo scans (Poly Haven, ambientCG). A closed-loop fitter matches each material's multi-scale albedo contrast spectrum, chroma variation, relief (normal-slope) spectrum and roughness mean/variance to color-weighted matching scans, applied as exactly periodic FFT band equalizers inside `synthesize_material` (`calibrated=False` reproduces the uncalibrated output). Refit with `hypbr atlas calibrate --refs <scan dirs> --baked <uncalibrated bake>`.
- **Exact Periodic Procedural Synthesis**: Pure NumPy/PyTorch circular-padded microstructure generators across metals, stones, ceramics, fabrics, wood, and hard-surface panels with `0.00 px` seam error — about 1 s per material at 512px and 11–13 s at 2048px on CPU (see [Performance](#performance)).
- **47-Node Composable PBR Graph**: Editable JSON node graph with analytic bevels, Weickert structure tensors, Reoriented Normal Mapping (RNM), 8-azimuth Horizon-Based Ambient Occlusion (HBAO) + Bent Normals, and Kaplanyan/Toksvig Specular Anti-Aliasing.
- **Optional Neural Photo-to-PBR**: Calibrate real macro photographs through **Apple Depth Pro** (metric depth prior) and **DeepBump ONNX** (micro-relief tangent normals) before feeding into the circular-padded 47-node graph.

<p align="center">
  <img src="examples/assets/showcase_1.png" alt="hypbr PBR showcase 1" width="100%" />
</p>
<p align="center">
  <img src="examples/assets/showcase_2.png" alt="hypbr PBR showcase 2" width="100%" />
</p>
<p align="center">
  <img src="examples/assets/showcase_3.png" alt="hypbr PBR showcase 3" width="100%" />
</p>

## Agent Atlas (500 materials)

| Endpoint | What it is |
| --- | --- |
| [`/llms.txt`](https://hypbr.vercel.app/llms.txt) | LLM-optimized entrypoint (endpoints, conventions, quick start) |
| [`/agent.md`](https://hypbr.vercel.app/agent.md) | Full integration guide: picking, UV scale, projection, shader wiring, weathering |
| [`/atlas.json`](https://hypbr.vercel.app/atlas.json) | Manifest: version, count, schema, URL templates, categories |
| [`/atlas/index.json`](https://hypbr.vercel.app/atlas/index.json) | Compact searchable index of all 500 materials |
| `/atlas/categories/<slug>.json` | 10 category shards with full metadata |
| `/atlas/mats/<id>.json` | Spec card: metadata + texture URLs + Three.js / Blender `bpy` / glTF snippets |
| `/assets/mats/<id>/{albedo_delit,normal,orm,aux,preview}.webp` | 512×512 maps (sRGB albedo, OpenGL normal, ORM = R AO / G rough / B metal, aux = R height / G curvature / B cavity) |

Every spec card carries `real_world_size_m` (UV repeat = surface size / tile size), `best_objects`, `avoid_objects`, `scene_roles`, `projection` (`uv` / `triplanar` / `box`, anisotropic axis, tangents, tiling note), `shader_setup` (color spaces, packing, displacement scale, IOR, measured roughness/metallic ranges, transmission), `recommended_weathering` (wear, dirt, moisture, moss, rust, frost + `hypbr_params`), `pairs_well_with` and `tags`.

```python
from hypbr.atlas import search_atlas, recommend_for_object, get_material_spec

search_atlas("brick wall", limit=5)
recommend_for_object("kitchen counter", style="modern", environment="interior")
print(get_material_spec("calacatta_gold_marble")["snippets"]["threejs"])
```

```bash
hypbr atlas search "asphalt road"
hypbr atlas recommend sofa --environment interior
hypbr atlas info classic_red_brick --snippet blender
hypbr atlas export ./site            # llms.txt, agent.md, atlas.json, atlas/**
hypbr synth classic_red_brick --size 2048 --weather-preset aged
```

## Installation

```bash
git clone https://github.com/wrong-nebula/hypbr.git
cd hypbr
pip install -e .
```

The core install is just `numpy` + `pillow`: all 500 materials, blending, weathering, export and
verification run on pure NumPy. PyTorch is optional:

```bash
pip install -e ".[gpu]"      # optional torch: CUDA/MPS-accelerated kernels (same output on CPU)
pip install -e ".[neural]"   # neural photo-to-PBR (`hypbr photo`): torch + ONNX Runtime (+ Apple Depth Pro)
```

With torch installed hypbr uses its kernels (bit-identical to previous releases); without it the
NumPy fallbacks match to within 1/255 on <0.2% of pixels. `HYPBR_TORCH=0` forces the NumPy path,
`HYPBR_DEVICE=cpu|mps|cuda` picks the torch device.

## Performance

Measured on CPU (Apple Silicon, `HYPBR_DEVICE=cpu`, 4 threads):

| Resolution | Time per material |
|---|---|
| 512 × 512 | ~1 s |
| 2048 × 2048 | ~11–13 s |

- Large-radius Gaussian blurs run as exact FFT convolutions, so cost no longer grows with the blur radius.
- Per-pixel kernels (Worley cells, studio preview) run in row bands on a small thread pool. Set `HYPBR_THREADS=N` (default `min(4, cpu_count)`, `1` disables). Output is bit-identical for any thread count.
- `HYPBR_DEVICE=cpu|mps|cuda` selects the torch device for the remaining GPU-capable kernels (when torch is installed).
- Without torch the NumPy fallbacks run at comparable CPU speed and import much faster; `HYPBR_TORCH=0` forces them.

## Quick Start (CLI)

List built-in procedural materials:

```bash
hypbr list
```

Synthesize a single seamless 15-map PBR material at `1024x1024` (with optional physical weathering `--wear`, `--weather-preset`, or per-channel `--dirt/--wet/--moss/--rust/--dust`):

```bash
hypbr synth patinated_verdigris_bronze --size 1024 --wear 0.35 --out ./output/pbr
hypbr synth travertine_porous_stone --size 1024 --weather-preset mossy --wet 0.5 --out ./output/pbr
```

Apply optional multi-mode physical weathering (`aged`, `rain`, `mossy`, `rusty`, `dusty`, `coastal`, `abandoned`, or custom channels) to any existing `hypbr` material directory — writing updated 15 maps plus a packed `RGBA8` `weather_mask.png` (`R=Wear, G=Dirt/Rust, B=Wet, A=Moss`). Supports both zero-GPU exact periodic procedural masks (default) and optional Single-Encode `DepthPro2-S` neural masks (`--backend neural --patch-dir ...`):

```bash
hypbr weather ./output/pbr/travertine_porous_stone --preset abandoned --out ./output/pbr/travertine_abandoned
hypbr weather ./output/pbr/diamond_tread_plate --rust 0.8 --wet 0.4 --out ./output/pbr/tread_rusty
```

Physically blend two materials using Unreal/Substance-style HeightLerp (`height`, `cavity`, or `linear`) with recomputed 8-azimuth HBAO, curvature, and ORM:

```bash
hypbr blend ./output/pbr/hammered_copper_dimple ./output/pbr/venetian_plaster_trowel --mode cavity --factor 0.45 --out ./output/pbr/copper_plaster
```

Inspect any synthesized or blended material in a local interactive Three.js PBR studio (channel switcher, live displacement, UV tiling, orbit lighting):

```bash
hypbr preview ./output/pbr/patinated_verdigris_bronze
```

Synthesize and export directly to game engines & 3D tools in one command:

```bash
hypbr synth patinated_verdigris_bronze --size 1024 --export unity,godot,unreal,blender,threejs,vulkan --out ./output/pbr
```

Or export any existing `hypbr` material directory:

```bash
hypbr export ./output/pbr/patinated_verdigris_bronze --target all --out ./output/engines/bronze
```

Verify seamlessness, normal validity, ORM packing, and physical PBR bounds (CI-ready exit code):

```bash
hypbr verify ./output/pbr/patinated_verdigris_bronze --json
```

Synthesize a machined hard-surface panel (`hardops_plate`):

```bash
hypbr synth hardops_plate --size 512 --out ./output/pbr
```

Synthesize the full built-in procedural catalog:

```bash
hypbr synth all --size 512 --out ./output/pbr
```

Export and inspect a 47-node PBR graph preset as JSON:

```bash
hypbr dump-preset v13-default > my_graph.json
```

Convert a macro photograph to 15 seamless PBR maps (requires optional Depth Pro + ONNX Runtime):

```bash
hypbr photo path/to/surface.jpg --preset v13-default --tileable --out ./output/photo_pbr
```

## Python API Examples

### 1. Synthesize Seamless PBR Materials

```python
from hypbr import list_materials, synthesize_material

for item in list_materials()[:5]:
    print(item["id"], "-", item["category"])

out_dir = synthesize_material(
    "damascus_folded_steel",
    out_dir="./output/damascus",
    size=1024,
    seed=42,
)
print("Saved 15 PBR maps to:", out_dir)
```

### 2. Keep Maps in Memory (no PNG round-trips)

`synthesize_material_maps` returns a `MaterialMaps` dataclass of float32 arrays (linear-light
`albedo`, unit `normal`, `height`, `roughness`, `metallic`, `ao`, `bent_normal`, `curvature`,
`cavity`, `edge_wear`, `anisotropy`, `specular`, `thickness`, plus the `preview` render and
`metadata`). `.save(dir)` writes byte-identical files to `synthesize_material`, and
`MaterialMaps.load(dir)` reads any material directory back. `blend_materials`, `weather_material`,
`export_material` and `verify_material` accept a directory or a `MaterialMaps`; the `*_maps`
variants return `MaterialMaps` so whole pipelines stay in memory:

```python
from hypbr import (MaterialMaps, blend_materials_maps, export_material,
                   synthesize_material_maps, weather_material_maps)

brick = synthesize_material_maps("classic_red_brick", size=1024)
moss = synthesize_material_maps("mossy_cobblestone_path", size=1024)
print(brick.albedo.dtype, brick.albedo.shape, brick.roughness.mean())  # float32 (1024, 1024, 3) ...

mixed = blend_materials_maps(brick, moss, factor=0.4, mode="cavity")
aged = weather_material_maps(mixed, preset="rain", seed=7)
export_material(aged, "./output/gltf/wet_brick", target="glb")  # encodes straight to the target
aged.save("./output/wet_brick")                                  # or write the 16-map PNG set
again = MaterialMaps.load("./output/wet_brick")                  # and read it back
```

Inputs are read at their saved 8/16-bit precision, so `blend_materials_maps(a, b)` gives exactly
the same result as blending `a.save(...)` and `b.save(...)` from disk.

### 3. Export to Unity, Godot, Unreal, Blender, Three.js, Vulkan & MaterialX

```python
from hypbr import export_material, synthesize_material

raw = synthesize_material("patinated_verdigris_bronze", "./output/bronze", size=1024)
export_material(raw, "./output/unity/bronze", target="unity")
export_material(raw, "./output/godot/bronze", target="godot")
export_material(raw, "./output/unreal/bronze", target="unreal")
export_material(raw, "./output/threejs/bronze", target="threejs")
export_material(raw, "./output/blender/bronze", target="blender")
export_material(raw, "./output/vulkan/bronze", target="vulkan")
```

### 4. Inspect or Customize the Node Graph

```python
import json
from hypbr import PRESETS

graph = PRESETS["v13-default"]
print(f"Nodes in v13-default: {len(graph['nodes'])}")
with open("custom_graph.json", "w") as f:
    json.dump(graph, f, indent=2)
```

See runnable scripts in [`examples/synthesize_pbr.py`](examples/synthesize_pbr.py), [`examples/export_engines.py`](examples/export_engines.py), [`examples/blend_and_weather.py`](examples/blend_and_weather.py), and [`examples/custom_graph.py`](examples/custom_graph.py).

## Reference Scans (calibration & weathering)

Calibration (`hypbr atlas calibrate`) and the weathering comparison (`tools/weathering_refs.py`) read real scans as 512px `.npz` files. `tools/import_refs.py` builds them from more sources:

```bash
# Megascans surfaces (or any PBR folder) you downloaded from Fab: sub-folders and/or zips, read in place
python tools/import_refs.py folder ~/Downloads/megascans --source megascans --out ~/hypbr-refs
# MatSynth (CC0 items only by default), streamed shard by shard; needs: pip install -e ".[refs]"
python tools/import_refs.py matsynth --out ~/hypbr-refs --limit 200
hypbr atlas calibrate --refs ~/hypbr-refs/megascans ~/hypbr-refs/matsynth --baked <uncalibrated bake>
```

Every item is recorded in `<out>/manifest.jsonl` with its source, license and origin. DirectX normal maps are detected from their height map and converted to OpenGL. The importer refuses to write inside a git repository: these scans are for training and fitting only, and must not be committed or published. Megascans' terms allow AI training (Quixel, 5 Nov 2026) but forbid standalone redistribution.

## Engine & Standard I/O Targets

| Target | Normal Convention | Channel Packing & Ready-to-Import Assets |
|---|---|---|
| `unity` | OpenGL `+Y` | `BaseMap.png`, `NormalMap.png`, `MetallicSmoothness.png` (`R=Metallic, A=1-Roughness`), `OcclusionMap.png`, `HeightMap.png`, `unity_material.json` |
| `unity-hdrp` | OpenGL `+Y` | `BaseMap.png`, `NormalMap.png`, `MaskMap.png` (`R=Metallic, G=AO, B=Detail, A=1-Roughness`), `HeightMap.png`, `unity_material.json` |
| `godot` | OpenGL `+Y` | Godot 4 `material.tres` (`StandardMaterial3D` with deep parallax & ORM channels), `albedo.png`, `orm.png`, `normal.png`, `height.png` |
| `unreal` | DirectX `-Y` | `T_BaseColor.png`, `T_Normal_DX.png` (green-flipped), `T_BentNormal_DX.png`, `T_ORM.png`, `T_Height16.png`, `ue5_material.json` |
| `blender` | OpenGL `+Y` | `import_blender.py` (one-click Blender 4.x Principled BSDF node tree with `sRGB`/`Non-Color` setup) + textures |
| `threejs` | OpenGL `+Y` | `material.js` ES module (`MeshPhysicalMaterial` factory with `SRGBColorSpace` / `NoColorSpace` & anisotropy) + textures |
| `gltf` | OpenGL `+Y` | `material.gltf` (glTF 2.0 `pbrMetallicRoughness` + `occlusionTexture` + `normalTexture`), `baseColor.png`, `orm.png`, `normal.png` |
| `glb` | OpenGL `+Y` | Single self-contained binary `<name>.glb` preview sphere with embedded albedo, ORM & normal maps (drag-and-drop in Blender, Three.js, Godot, Quick Look) |
| `vulkan` | OpenGL `+Y` | `vulkan_manifest.json` (`VK_FORMAT_*` bindings & `ktx create` UASTC/BC5 commands), `orm_unorm.png`, `normal_rg_bc5_src.png`, `height_r16.png` |
| `materialx` | OpenGL `+Y` | `material.mtlx` (MaterialX 1.38 `standard_surface` + `displacement` for OpenUSD, Houdini, Maya, RealityKit) + textures |
| `usd` | OpenGL `+Y` | `material.usda` (OpenUSD ASCII `UsdPreviewSurface` with exact `scale=(2,2,2)` / `bias=(-1,-1,-1)` normal decoding for RealityKit, visionOS, Omniverse) + textures |

## Output Maps

Every material synthesis writes a clean, renderer-ready directory compatible with **glTF 2.0**, **Three.js**, **Blender**, **Unreal Engine**, and **Unity**:

| File | Bit Depth / Channels | Description |
|---|---|---|
| `albedo_delit.png` | 8-bit sRGB | De-lit physical base color |
| `normal.png` | 8-bit RGB | Tangent-space normal map (OpenGL `+Y`, optional DirectX `-Y`) |
| `bent_normal.png` | 8-bit RGB | Average unoccluded hemisphere direction from 8-azimuth HBAO |
| `height.png` | 8-bit Gray | Linear displacement / POM heightmap |
| `height16.png` | 16-bit Gray | Precision 16-bit linear heightmap (zero banding) |
| `ao.png` | 8-bit Gray | 8-azimuth Horizon-Based Ambient Occlusion |
| `roughness.png` | 8-bit Gray | Physical roughness with Toksvig Specular AA |
| `metallic.png` | 8-bit Gray | Physical metallic mask |
| `orm.png` | 8-bit RGB | Packed glTF 2.0 texture (`R=AO, G=Roughness, B=Metallic`) |
| `curvature.png` | 8-bit Gray | Multi-scale signed curvature (0.5 neutral) |
| `cavity.png` | 8-bit Gray | Concave crevice mask |
| `edge_wear.png` | 8-bit Gray | Convex ridge wear mask |
| `anisotropy.png` | 8-bit RGB | Tangent flow direction + anisotropy strength |
| `specular.png` | 8-bit Gray | Dielectric specular level / F0 modulation |
| `thickness.png` | 8-bit Gray | Inverted ambient thickness for subsurface scattering |
| `preview.png` | 8-bit sRGB | Shaded studio preview |

## Architecture

- [`hypbr/procedural.py`](hypbr/procedural.py): Exact periodic FBM, Voronoi, weave, tile, and metal microstructure heightfield generators.
- [`hypbr/normal_graph.py`](hypbr/normal_graph.py): Composable 47-node PBR graph engine (RNM blending, Weickert structure tensors, analytic SDF bevels, GPU-accelerated 8-azimuth HBAO + bent normals, Toksvig Specular AA, ORM packing).
- [`hypbr/synthesize.py`](hypbr/synthesize.py): High-level material synthesis API combining periodic heightfields, physically based multi-tone albedo/metallic rules, and the 15-map export pipeline.
- [`hypbr/maps.py`](hypbr/maps.py): `MaterialMaps` in-memory material (float32 maps + metadata) with byte-exact `.save()` / `.load()`, and the directory-or-memory `MapSource` reader used by blend/weather/export/verify.
- [`hypbr/_backend.py`](hypbr/_backend.py): Optional-PyTorch backend: lazy `torch` import, device selection, and NumPy equivalents of the `pad` / `conv2d` / `max_pool2d` / `interpolate` / `grid_sample` kernels on the procedural path.
- [`hypbr/weightcode.py`](hypbr/weightcode.py): Zero-training analytic weight surgery (exact 9-mode orthonormal 3×3 spatial kernel filtering, SVD energy shrinkage, circular Conv2d padding, and 4-quadrant partition-of-unity tileable blending).
- [`hypbr/depthpro_pbr.py`](hypbr/depthpro_pbr.py): Optional Apple Depth Pro + DeepBump ONNX neural relief extractor (needs `hypbr[neural]`) with 4-rotation perspective cancellation, plus the shared exact FFT / separable Gaussian, morphology and Scharr kernels (torch-accelerated when installed, `HYPBR_DEVICE=cpu|mps|cuda`).
- [`hypbr/exporters.py`](hypbr/exporters.py): Zero-dependency engine & standard exporters (`unity`, `unity-hdrp`, `godot`, `unreal`, `blender`, `threejs`, `gltf`, `glb`, `vulkan`, `materialx`, `usd`).
- [`hypbr/blend.py`](hypbr/blend.py): Physically based 16-bit HeightLerp / cavity material blending (`hypbr blend`) with linear-light albedo mixing and recomputed HBAO/curvature/ORM.
- [`hypbr/preview.py`](hypbr/preview.py): Zero-dependency local interactive Three.js PBR studio viewer (`hypbr preview`) with per-channel inspection, tiling, and displacement controls.
- [`hypbr/verify.py`](hypbr/verify.py): Automated seamlessness, tangent normal validity, Helmholtz curl, ORM consistency, and PBR calibration QA verifier (`hypbr verify`).
- [`hypbr/cli.py`](hypbr/cli.py): `hypbr` command-line interface.

## License

Code is released under the [MIT License](LICENSE).

> **Note on optional neural weights:** If you use the optional `hypbr photo` neural mode with Apple Depth Pro checkpoints, those external weights are subject to Apple's `apple-amlr` research license. Pure procedural synthesis (`hypbr synth`) has zero external model dependencies and is 100% MIT licensed.
