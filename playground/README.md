# hypbr Interactive Three.js PBR Material Playground

**Live Demo:** **[https://hypbr.vercel.app](https://hypbr.vercel.app)**

Interactive Three.js `r187` PBR material studio to inspect, weather, blend, verify, and export all **69** seamless procedural PBR materials in `hypbr` (`68` `DAILY_CATALOG` materials + `hardops_plate`).

## Dual Mode: Static Edge (Vercel) + Local CPU Server

The playground auto-detects its environment:

- **Static Edge Mode (Vercel / CDN — [hypbr.vercel.app](https://hypbr.vercel.app))**: Loads pre-baked 512×512 seamless WebP bundles (`assets/mats/<id>/{albedo_delit,normal,orm,aux,preview}.webp`) and runs real-time 6-channel physical weathering, HeightLerp A×B blending, and PNG ZIP packaging 100% client-side in `static_engine.js`.
- **Local Python Server Mode**: Synthesizes fresh/weathered/blended materials live via `hypbr` on CPU with content-addressed disk caching in `playground/.cache/`.

### Run Locally (Live Python Synthesis)

```bash
HYPBR_DEVICE=cpu python3 playground/server.py --port 8766
```

Then open **http://127.0.0.1:8766** in your browser.

### Re-Bake Static WebP Assets (All 69 Materials at 512px)

```bash
HYPBR_DEVICE=cpu python3 playground/bake_static.py --workers 4
```

## Features

- **Three.js `r187` Studio Rendering**: Linear HDR `HalfFloatType` MSAA -> `SMAAPass` -> `OutputPass` (`THREE.AgXToneMapping`), procedural softbox PMREM HDRI (rotates in lockstep with `Shift+Drag` light orbit), grounded shadow catcher + cyclorama dome, physical `clearcoat` / `sheen`, and high-subdivision shaderball meshes.
- **Studio Anti-Aliasing Suite**: 5-tap texel-filtered vertex displacement, 4-tap GPU Cubic B-Spline normal magnification (zero close-up bilinear diamonds), 4-tap Rotated Grid Super-Sampling (RGSS) minification, and dual Kaplanyan + Toksvig Specular AA.
- **Catalog Browser**: Instant search (`/`), semantic category filter pills, live thumbnails, `Auto Scene` staging, and `↑` / `↓` keyboard navigation across all 69 QA-verified materials.
- **Channel Inspector + 2D Diagnostic Sheet**: Inspect `Final PBR`, `Albedo`, `Normal`, `Roughness`, `Metallic`, `AO`, `Height`, `Curvature`, `Weather Mask`, and a full `2D All Maps` contact sheet with `2×2` seamless tile preview and lightbox inspection.
- **A/B Comparison**: Draggable **Split A/B** 3D scissor wipe (`Fresh` vs `Weathered`) and **Hold `Space` for Fresh** instant toggle.
- **6-Channel Weathering Lab, HeightLerp Blend & Export**: 8 one-click presets (`Fresh`, `Aged`, `Rain`, `Mossy`, `Rusty`, `Dusty`, `Coastal`, `Abandoned`), 6 live sliders (`Wear`, `Dirt`, `Wet`, `Moss`, `Rust`, `Dust`), A×B HeightLerp/Cavity/Linear blender, live QA verification telemetry, PNG viewport snapshots, and one-click engine-ready PBR ZIP export.
