# hypbr Atlas - Agent Integration Guide

The hypbr atlas is a static, CORS-enabled catalog of **500 seamless PBR materials**. Every material ships 512x512 WebP maps and a JSON spec card telling you exactly where it belongs, how big one tile is in the real world, how to project it and how to wire it into your engine. Use this guide to go from "texture this object" to a correctly scaled, physically wired material in one pass.

## 1. Discover
| Resource | URL |
| --- | --- |
| Manifest | `https://hypbr.vercel.app/atlas.json` |
| Compact index (500 entries) | `https://hypbr.vercel.app/atlas/index.json` |
| Category shards (full metadata) | `https://hypbr.vercel.app/atlas/categories/<category>.json` |
| Spec card (+ snippets) | `https://hypbr.vercel.app/atlas/mats/<id>.json` |
| Textures | `https://hypbr.vercel.app/assets/mats/<id>/<map>.webp` |

Categories: `architectural_masonry` (65), `wood_timber` (58), `stone_marble_concrete` (58), `tiles_ceramics` (46), `metals_industrial` (58), `fabrics_upholstery` (58), `nature_terrain` (40), `glass_optics` (35), `rubber_polymers_roads` (42), `roofing_tech_scifi` (40).

## 2. Pick a material
1. Fetch `atlas/index.json` once (it is small) and match your object against `best_objects` (exact object nouns such as `kitchen_countertop`, `exterior_building_wall`, `sofa`, `road`), then `scene_roles` (`wall`, `floor`, `ceiling`, `roof`, `ground`, `road`, `furniture`, `prop`, `trim`, `hero_asset`, `mechanical`, `foliage_bark`, `glass_window`) and `tags` (color, style, era, biome, finish).
2. Reject candidates whose `avoid_objects` contain your object.
3. Prefer exterior materials outdoors and interior materials indoors; tags such as `exterior`, `coastal`, `garden` and the category tell you which.
4. Use `pairs_well_with` (3-5 atlas IDs) to choose coordinated neighbours: wall + floor + trim + glazing.

## 3. Scale it physically (never guess UV repeat)
`real_world_size_m = [w, h]` is the size in meters covered by **one** texture tile (UV 0..1).
- UV-mapped surface of `W x H` meters: `repeat_u = W / w`, `repeat_v = H / h`.
- World-space / triplanar: tiles per meter = `1 / w` (and `1 / h`).
- Example: `classic_red_brick` is `[0.9, 0.75]`; a 4 m x 2.7 m wall needs `repeat = [4.444, 3.600]`.

## 4. Project it
`projection.recommended` is `uv`, `triplanar` or `box`.
- `uv`: use mesh UVs scaled as above. Keep directional patterns aligned: `anisotropic_axis` = `u` means the grain/brushing runs along U, `v` along V, `radial` around the tile center.
- `triplanar`: sample in world space on terrain, rock, monolithic concrete and sculpted props (no UVs needed). `triplanar_fallback: false` means triplanar will visibly break the pattern (planks, tiles, bricks) - unwrap UVs instead.
- `box`: cube-project blocky architecture (blocks, ashlar) so courses stay level on every face.
- `requires_tangents: true` means the material is directional: generate MikkTSpace tangents (or use the engine's anisotropy rotation).
- `tiling_note` holds the material-specific orientation advice.

## 5. Wire the shader
| Map | Colorspace | Content |
| --- | --- | --- |
| `albedo_delit.webp` | sRGB | De-lit base color (no baked shadows) |
| `normal.webp` | Linear | Tangent-space normal, **OpenGL +Y** |
| `orm.webp` | Linear | R = AO, G = Roughness, B = Metallic |
| `aux.webp` | Linear | R = Height (0.5 mid), G = Curvature, B = Cavity |
| `preview.webp` | sRGB | Thumbnail only |

`shader_setup` also gives `displacement_scale_m` (peak-to-peak height in meters for `aux.R`), `ior`, `is_metallic`, measured `roughness_range` / `metallic_range` and `transmission` (glass and ice).

Engine notes:
- **Three.js**: `MeshPhysicalMaterial`; pass the same `orm` texture to `aoMap`, `roughnessMap` and `metalnessMap` with `roughness = metalness = 1`. Set `albedo` to `SRGBColorSpace`, data maps to `NoColorSpace`, `wrapS = wrapT = RepeatWrapping`. Glass: `transmission`, `thickness`, `ior` from the card.
- **glTF 2.0**: `orm` is natively the `metallicRoughnessTexture` (G/B) and `occlusionTexture` (R); use `KHR_texture_transform.scale` for tiling and `KHR_materials_ior`.
- **Blender**: Image Texture nodes; albedo sRGB, others Non-Color. `orm` -> Separate Color (G -> Roughness, B -> Metallic). `normal` -> Normal Map node. `aux` -> Separate Color (R) -> Displacement (Scale = `displacement_scale_m`, Midlevel 0.5). For triplanar use Object coordinates + Box projection.
- **Unreal Engine**: normals are DirectX: tick *Flip Green Channel* on `normal.webp`. `orm` matches UE's ORM order; disable sRGB on data maps.
- **Unity**: OpenGL normals are native. URP/Built-in Standard wants smoothness = 1 - roughness (repack `orm.G` into metallic alpha); HDRP mask map = (Metallic, AO, Detail, Smoothness).

## 6. Weather it
`recommended_weathering` gives 0-1 levels for `wear`, `dirt`, `moisture`, `moss`, `rust`, `frost` (+ `dust`) and a physical note. Drive your engine's grime/wetness/moss layers with them, or regenerate a weathered texture with the Python package using `hypbr_params` directly: `hypbr synth <id> --wear 0.3 --dirt 0.4 --wet 0.15 --moss 0.12`. Environment rules: interior => no moisture/moss/frost; coastal => +moisture, +rust on metals; desert => +dust, no moisture; winter => +frost; forest => +moss; abandoned => everything up.

## 7. Example spec card (`classic_red_brick`, abridged)
```json
{
  "id": "classic_red_brick",
  "real_world_size_m": [
    0.9,
    0.75
  ],
  "best_objects": [
    "exterior_building_wall",
    "chimney",
    "garden_retaining_wall",
    "loft_accent_wall",
    "archway",
    "fireplace_surround"
  ],
  "scene_roles": [
    "wall",
    "trim"
  ],
  "projection": {
    "recommended": "uv",
    "triplanar_fallback": true,
    "world_scale_uv": true,
    "anisotropic_axis": "none",
    "requires_tangents": false,
    "tiling_note": "Keep courses horizontal: map U along the wall length and V up the wall; use box mapping for chimneys.",
    "uv_repeat_formula": "repeat_u = surface_width_m / 0.9; repeat_v = surface_height_m / 0.75",
    "triplanar_scale_formula": "triplanar_scale = 1 / 0.9 (texture tiles per meter along U), 1 / 0.75 along V"
  },
  "shader_setup": {
    "normal_space": "OpenGL (+Y)",
    "orm_packing": "R=AO, G=Roughness, B=Metallic",
    "aux_packing": "R=Height, G=Curvature, B=Cavity",
    "albedo_colorspace": "sRGB",
    "data_colorspace": "Linear",
    "displacement_scale_m": 0.012,
    "ior": 1.5,
    "is_metallic": false,
    "roughness_range": [
      0.71,
      0.965
    ],
    "metallic_range": [
      0.0,
      0.0
    ],
    "transmission": 0.0,
    "ranges_measured": true
  },
  "recommended_weathering": {
    "preset": "aged",
    "wear": 0.3,
    "dirt": 0.4,
    "moisture": 0.15,
    "moss": 0.12,
    "rust": 0.0,
    "frost": 0.0,
    "dust": 0.1,
    "note": "Grime and moss settle in recessed mortar joints; keep rust at 0 unless iron fixings drip.",
    "hypbr_params": {
      "wear": 0.3,
      "dirt": 0.4,
      "wet": 0.15,
      "moss": 0.12,
      "rust": 0.0,
      "dust": 0.1
    }
  },
  "pairs_well_with": [
    "herringbone_clay_paver",
    "natural_slate_roof",
    "gray_solar_glazing",
    "exposed_aggregate_concrete"
  ],
  "maps": {
    "albedo_delit": "https://hypbr.vercel.app/assets/mats/classic_red_brick/albedo_delit.webp",
    "normal": "https://hypbr.vercel.app/assets/mats/classic_red_brick/normal.webp",
    "orm": "https://hypbr.vercel.app/assets/mats/classic_red_brick/orm.webp",
    "aux": "https://hypbr.vercel.app/assets/mats/classic_red_brick/aux.webp",
    "preview": "https://hypbr.vercel.app/assets/mats/classic_red_brick/preview.webp"
  }
}
```

### Three.js
```js
// hypbr atlas: Classic Red Brick (classic_red_brick) | 1 tile = 0.9 m x 0.75 m
import * as THREE from 'three';
const SURFACE_W = 4, SURFACE_H = 2.7; // meters covered by the mesh UV 0..1
const base = 'https://hypbr.vercel.app/assets/mats/classic_red_brick/';
const loader = new THREE.TextureLoader().setCrossOrigin('anonymous');
function tex(file, srgb) {
  const t = loader.load(base + file);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(SURFACE_W / 0.9, SURFACE_H / 0.75);
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.anisotropy = 8;
  return t;
}
const orm = tex('orm.webp', false); // R=AO, G=Roughness, B=Metallic
const material = new THREE.MeshPhysicalMaterial({
  map: tex('albedo_delit.webp', true),
  normalMap: tex('normal.webp', false), // OpenGL +Y = three.js native
  aoMap: orm, roughnessMap: orm, metalnessMap: orm,
  roughness: 1.0, metalness: 1.0,
  displacementMap: tex('aux.webp', false), displacementScale: 0.012, displacementBias: -0.006, // aux.R height; dense mesh only
  ior: 1.5,
});
mesh.material = material;
```

### Blender (bpy)
```python
# hypbr atlas: Classic Red Brick (classic_red_brick) | 1 tile = 0.9 m x 0.75 m | Blender 3.4+
import bpy, os, tempfile, urllib.request
MAT_ID, BASE = 'classic_red_brick', 'https://hypbr.vercel.app/assets/mats/classic_red_brick/'
SURFACE_W, SURFACE_H = 4, 2.7  # meters covered by UV 0..1 (UV mode)
cache = os.path.join(tempfile.gettempdir(), 'hypbr', MAT_ID); os.makedirs(cache, exist_ok=True)
def img(name, data):
    p = os.path.join(cache, name)
    if not os.path.exists(p): urllib.request.urlretrieve(BASE + name, p)
    im = bpy.data.images.load(p, check_existing=True)
    im.colorspace_settings.name = 'Non-Color' if data else 'sRGB'
    return im
mat = bpy.data.materials.new('hypbr_' + MAT_ID); mat.use_nodes = True
N, L = mat.node_tree.nodes, mat.node_tree.links
bsdf, out = N['Principled BSDF'], N['Material Output']
tc, mp = N.new('ShaderNodeTexCoord'), N.new('ShaderNodeMapping')
mp.inputs['Scale'].default_value = (SURFACE_W / 0.9, SURFACE_H / 0.75, 1.0)
L.new(tc.outputs["UV"], mp.inputs['Vector'])
def tex(name, data):
    t = N.new('ShaderNodeTexImage'); t.image = img(name, data)
    L.new(mp.outputs['Vector'], t.inputs['Vector']); return t
L.new(tex('albedo_delit.webp', False).outputs['Color'], bsdf.inputs['Base Color'])
orm = N.new('ShaderNodeSeparateColor'); L.new(tex('orm.webp', True).outputs['Color'], orm.inputs['Color'])
L.new(orm.outputs['Green'], bsdf.inputs['Roughness']); L.new(orm.outputs['Blue'], bsdf.inputs['Metallic'])
nm = N.new('ShaderNodeNormalMap'); L.new(tex('normal.webp', True).outputs['Color'], nm.inputs['Color'])  # OpenGL +Y
L.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
aux = N.new('ShaderNodeSeparateColor'); L.new(tex('aux.webp', True).outputs['Color'], aux.inputs['Color'])
dsp = N.new('ShaderNodeDisplacement'); dsp.inputs['Scale'].default_value = 0.012; dsp.inputs['Midlevel'].default_value = 0.5
L.new(aux.outputs['Red'], dsp.inputs['Height']); L.new(dsp.outputs['Displacement'], out.inputs['Displacement'])
bsdf.inputs['IOR'].default_value = 1.5
obj = bpy.context.active_object
if obj is not None and obj.type == 'MESH': obj.data.materials.append(mat)
```

## 8. Python (optional)
```python
from hypbr.atlas import search_atlas, recommend_for_object, get_material_spec
recommend_for_object("sofa", style="mid-century", environment="interior")
search_atlas("asphalt road", limit=5)
get_material_spec("calacatta_gold_marble")["snippets"]["threejs"]
```
CLI: `hypbr atlas search "brick wall"`, `hypbr atlas info <id>`, `hypbr atlas recommend "kitchen counter" --style modern`, `hypbr atlas export ./site`.
