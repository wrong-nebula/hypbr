"""Agent-facing query API and static exporter for the 500-material hypbr PBR Atlas.

Typical agent flow::

    from hypbr.atlas import recommend_for_object, get_material_spec
    pick = recommend_for_object("kitchen counter", style="modern", environment="interior")["matches"][0]
    spec = get_material_spec(pick["id"])          # map URLs + Three.js / Blender snippets
"""
from __future__ import annotations

import copy
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import atlas_catalog as ac

DEFAULT_BASE_URL = "https://hypbr.vercel.app"
MAP_FILES: dict[str, str] = {
    "albedo_delit": "albedo_delit.webp",
    "normal": "normal.webp",
    "orm": "orm.webp",
    "aux": "aux.webp",
    "preview": "preview.webp",
}
MAP_DESCRIPTIONS: dict[str, str] = {
    "albedo_delit": "Base color, de-lit (no baked lighting). sRGB. 512x512 WebP.",
    "normal": "Tangent-space normal map, OpenGL convention (+Y up / green = up). Linear. 512x512 WebP.",
    "orm": "Packed R=Ambient Occlusion, G=Roughness, B=Metallic. Linear. Directly glTF metallicRoughness compatible.",
    "aux": "Packed R=Height (0.5 = mid-level), G=Curvature (0.5 = flat), B=Cavity. Linear.",
    "preview": "256x256 studio-lit sphere thumbnail (sRGB) for previews only; never use as a texture.",
}
_MEAS_PATH = Path(__file__).resolve().parent / "data" / "atlas_measurements.json"

TYPICAL_SURFACE_M: dict[str, tuple[float, float]] = {
    "wall": (4.0, 2.7), "facade": (10.0, 6.0), "floor": (5.0, 4.0), "ceiling": (5.0, 4.0), "roof": (8.0, 5.0),
    "road": (7.0, 20.0), "street": (7.0, 20.0), "driveway": (3.0, 6.0), "path": (1.2, 6.0), "patio": (4.0, 4.0),
    "sidewalk": (2.0, 10.0), "ground": (10.0, 10.0), "terrain": (20.0, 20.0), "cliff": (12.0, 12.0),
    "counter": (2.4, 0.65), "countertop": (2.4, 0.65), "island": (2.2, 1.0), "backsplash": (2.4, 0.6),
    "table": (1.8, 0.9), "desk": (1.6, 0.8), "sofa": (2.2, 0.9), "couch": (2.2, 0.9), "armchair": (0.9, 0.9),
    "chair": (0.5, 0.5), "bed": (2.0, 1.6), "cushion": (0.5, 0.5), "rug": (2.4, 1.7), "curtain": (1.4, 2.5),
    "door": (0.9, 2.1), "window": (1.2, 1.5), "shower": (1.0, 2.1), "pool": (8.0, 4.0), "fence": (2.4, 1.8),
    "tire": (2.0, 0.25), "handle": (0.12, 0.04), "knob": (0.12, 0.03), "corridor": (3.0, 2.8), "hull": (6.0, 3.0),
    "panel": (1.2, 1.2), "box": (0.6, 0.4), "crate": (1.0, 0.8), "trunk": (2.0, 3.0), "log": (1.5, 1.0),
    "vase": (0.6, 0.3), "screen": (1.0, 2.0), "stair": (1.2, 0.3),
}

_STOP = {"a", "an", "the", "for", "of", "with", "on", "in", "and", "my", "to", "some", "material", "texture", "surface", "pbr"}
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "sofa": ("sofa", "couch", "upholstery", "armchair"), "couch": ("sofa", "couch", "upholstery"),
    "counter": ("countertop", "counter"), "countertop": ("countertop", "counter"), "worktop": ("countertop",),
    "road": ("road", "street", "highway", "asphalt"), "street": ("street", "road"), "highway": ("highway", "road"),
    "scifi": ("scifi", "sci_fi", "spaceship", "futuristic", "hull"), "spaceship": ("spaceship", "scifi", "hull"),
    "corridor": ("corridor", "hallway", "station"), "hallway": ("hallway", "corridor"),
    "floor": ("floor", "flooring"), "flooring": ("floor",), "ground": ("ground", "terrain"), "terrain": ("terrain", "ground"),
    "bricks": ("brick",), "tiles": ("tile",), "tile": ("tile", "tiles"), "rug": ("rug", "carpet"), "carpet": ("carpet", "rug"),
    "window": ("window", "glazing", "glass"), "pane": ("glass", "window"), "metal": ("metal", "steel", "aluminum"),
    "wood": ("wood", "timber", "oak", "plank"), "wooden": ("wood",), "stone": ("stone", "rock"), "rock": ("rock", "stone", "cliff"),
    "kitchen": ("kitchen",), "bathroom": ("bathroom", "shower"), "shower": ("shower", "bathroom"),
    "outdoor": ("exterior",), "outside": ("exterior",), "exterior": ("exterior",), "indoor": ("interior",),
    "fabric": ("fabric", "textile"), "cloth": ("fabric", "textile"), "grass": ("grass", "lawn"), "lawn": ("lawn", "grass"),
    "roof": ("roof", "roofing"), "roofing": ("roof",), "car": ("car", "automotive"), "tyre": ("tire",),
}
_ENV_ALIASES = {
    "interior": "interior", "indoor": "interior", "inside": "interior", "exterior": "exterior", "outdoor": "exterior",
    "outside": "exterior", "coastal": "coastal", "beach": "coastal", "marine": "coastal", "rain": "rainy", "rainy": "rainy",
    "wet": "rainy", "desert": "desert", "arid": "desert", "abandoned": "abandoned", "ruin": "abandoned", "ruins": "abandoned",
    "post_apocalyptic": "abandoned", "winter": "winter", "snow": "winter", "alpine": "winter", "arctic": "winter",
    "forest": "forest", "jungle": "forest", "swamp": "forest", "urban": "urban", "city": "urban", "industrial": "urban",
    "space": "space", "scifi": "space",
}


# ---------------------------------------------------------------------------
# Measurements written by the static baker (measured roughness/metallic ranges + QA)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _measurements() -> dict[str, Any]:
    try:
        return json.loads(_MEAS_PATH.read_text(encoding="utf-8")).get("materials", {})
    except (OSError, ValueError):
        return {}


def clear_measurement_cache() -> None:
    _measurements.cache_clear()


def _base(base_url: str | None) -> str:
    return (base_url or DEFAULT_BASE_URL).rstrip("/")


def map_urls(material_id: str, base_url: str | None = DEFAULT_BASE_URL) -> dict[str, str]:
    """Return absolute (CORS-enabled) WebP texture URLs for one material."""
    root = f"{_base(base_url)}/assets/mats/{material_id}"
    return {k: f"{root}/{f}" for k, f in MAP_FILES.items()}


def _with_measurements(entry: dict[str, Any], meas: dict[str, Any] | None = None) -> dict[str, Any]:
    m = (meas if meas is not None else _measurements()).get(entry["id"])
    if m:
        ss = entry["shader_setup"]
        if "roughness_range" in m:
            ss["roughness_range"] = [round(float(v), 3) for v in m["roughness_range"]]
        if "metallic_range" in m:
            ss["metallic_range"] = [round(float(v), 3) for v in m["metallic_range"]]
            ss["is_metallic"] = bool(float(m["metallic_range"][1]) >= 0.5 and float(m.get("metallic_mean", 0.0)) >= 0.25)
        ss["ranges_measured"] = True
        if "verify" in m:
            entry["qa"] = m["verify"]
        if "albedo_mean_srgb" in m:
            entry["albedo_mean_srgb"] = m["albedo_mean_srgb"]
    return entry


def _entries(meas: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [_with_measurements(e, meas) for e in ac.get_catalog()]


# ---------------------------------------------------------------------------
# Snippets
# ---------------------------------------------------------------------------
def threejs_snippet(entry: dict[str, Any], base_url: str | None = DEFAULT_BASE_URL,
                    surface_m: tuple[float, float] = (4.0, 2.7)) -> str:
    w, h = entry["real_world_size_m"]
    ss, pj = entry["shader_setup"], entry["projection"]
    disp = float(ss["displacement_scale_m"])
    root = f"{_base(base_url)}/assets/mats/{entry['id']}/"
    extra = []
    if ss.get("transmission", 0.0) > 0.0:
        extra.append(f"  transmission: {ss['transmission']:g}, thickness: 0.006, // glass: albedo tints transmission")
    if pj["anisotropic_axis"] in ("u", "v"):
        rot = "0" if pj["anisotropic_axis"] == "u" else "Math.PI / 2"
        extra.append(f"  anisotropy: 0.6, anisotropyRotation: {rot}, // brushed/grain direction along {pj['anisotropic_axis'].upper()}")
    extra_s = ("\n" + "\n".join(extra)) if extra else ""
    return (
        f"// hypbr atlas: {entry['name']} ({entry['id']}) | 1 tile = {w:g} m x {h:g} m\n"
        "import * as THREE from 'three';\n"
        f"const SURFACE_W = {surface_m[0]:g}, SURFACE_H = {surface_m[1]:g}; // meters covered by the mesh UV 0..1\n"
        f"const base = '{root}';\n"
        "const loader = new THREE.TextureLoader().setCrossOrigin('anonymous');\n"
        "function tex(file, srgb) {\n"
        "  const t = loader.load(base + file);\n"
        "  t.wrapS = t.wrapT = THREE.RepeatWrapping;\n"
        f"  t.repeat.set(SURFACE_W / {w:g}, SURFACE_H / {h:g});\n"
        "  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;\n"
        "  t.anisotropy = 8;\n"
        "  return t;\n"
        "}\n"
        "const orm = tex('orm.webp', false); // R=AO, G=Roughness, B=Metallic\n"
        "const material = new THREE.MeshPhysicalMaterial({\n"
        "  map: tex('albedo_delit.webp', true),\n"
        "  normalMap: tex('normal.webp', false), // OpenGL +Y = three.js native\n"
        "  aoMap: orm, roughnessMap: orm, metalnessMap: orm,\n"
        "  roughness: 1.0, metalness: 1.0,\n"
        f"  displacementMap: tex('aux.webp', false), displacementScale: {disp:g}, displacementBias: {-disp / 2:g}, // aux.R height; dense mesh only\n"
        f"  ior: {ss['ior']:g},{extra_s}\n"
        "});\n"
        "mesh.material = material;\n"
    )


def blender_snippet(entry: dict[str, Any], base_url: str | None = DEFAULT_BASE_URL,
                    surface_m: tuple[float, float] = (4.0, 2.7)) -> str:
    w, h = entry["real_world_size_m"]
    ss, pj = entry["shader_setup"], entry["projection"]
    box = pj["recommended"] in ("triplanar", "box")
    root = f"{_base(base_url)}/assets/mats/{entry['id']}/"
    coord = 'tc.outputs["Object"]' if box else 'tc.outputs["UV"]'
    scale = f"(1 / {w:g}, 1 / {h:g}, 1 / {w:g})" if box else f"(SURFACE_W / {w:g}, SURFACE_H / {h:g}, 1.0)"
    proj = '\n    t.projection = "BOX"; t.projection_blend = 0.25' if box else ""
    trans = ""
    if ss.get("transmission", 0.0) > 0.0:
        trans = (f'\nfor key in ("Transmission Weight", "Transmission"):\n'
                 f'    if key in bsdf.inputs: bsdf.inputs[key].default_value = {ss["transmission"]:g}')
    return (
        f"# hypbr atlas: {entry['name']} ({entry['id']}) | 1 tile = {w:g} m x {h:g} m | Blender 3.4+\n"
        "import bpy, os, tempfile, urllib.request\n"
        f"MAT_ID, BASE = '{entry['id']}', '{root}'\n"
        f"SURFACE_W, SURFACE_H = {surface_m[0]:g}, {surface_m[1]:g}  # meters covered by UV 0..1 (UV mode)\n"
        "cache = os.path.join(tempfile.gettempdir(), 'hypbr', MAT_ID); os.makedirs(cache, exist_ok=True)\n"
        "def img(name, data):\n"
        "    p = os.path.join(cache, name)\n"
        "    if not os.path.exists(p): urllib.request.urlretrieve(BASE + name, p)\n"
        "    im = bpy.data.images.load(p, check_existing=True)\n"
        "    im.colorspace_settings.name = 'Non-Color' if data else 'sRGB'\n"
        "    return im\n"
        "mat = bpy.data.materials.new('hypbr_' + MAT_ID); mat.use_nodes = True\n"
        "N, L = mat.node_tree.nodes, mat.node_tree.links\n"
        "bsdf, out = N['Principled BSDF'], N['Material Output']\n"
        "tc, mp = N.new('ShaderNodeTexCoord'), N.new('ShaderNodeMapping')\n"
        f"mp.inputs['Scale'].default_value = {scale}\n"
        f"L.new({coord}, mp.inputs['Vector'])\n"
        "def tex(name, data):\n"
        f"    t = N.new('ShaderNodeTexImage'); t.image = img(name, data){proj}\n"
        "    L.new(mp.outputs['Vector'], t.inputs['Vector']); return t\n"
        "L.new(tex('albedo_delit.webp', False).outputs['Color'], bsdf.inputs['Base Color'])\n"
        "orm = N.new('ShaderNodeSeparateColor'); L.new(tex('orm.webp', True).outputs['Color'], orm.inputs['Color'])\n"
        "L.new(orm.outputs['Green'], bsdf.inputs['Roughness']); L.new(orm.outputs['Blue'], bsdf.inputs['Metallic'])\n"
        "nm = N.new('ShaderNodeNormalMap'); L.new(tex('normal.webp', True).outputs['Color'], nm.inputs['Color'])  # OpenGL +Y\n"
        "L.new(nm.outputs['Normal'], bsdf.inputs['Normal'])\n"
        "aux = N.new('ShaderNodeSeparateColor'); L.new(tex('aux.webp', True).outputs['Color'], aux.inputs['Color'])\n"
        f"dsp = N.new('ShaderNodeDisplacement'); dsp.inputs['Scale'].default_value = {ss['displacement_scale_m']:g}; dsp.inputs['Midlevel'].default_value = 0.5\n"
        "L.new(aux.outputs['Red'], dsp.inputs['Height']); L.new(dsp.outputs['Displacement'], out.inputs['Displacement'])\n"
        f"bsdf.inputs['IOR'].default_value = {ss['ior']:g}{trans}\n"
        "obj = bpy.context.active_object\n"
        "if obj is not None and obj.type == 'MESH': obj.data.materials.append(mat)\n"
    )


def gltf_material(entry: dict[str, Any], base_url: str | None = DEFAULT_BASE_URL) -> dict[str, Any]:
    """glTF 2.0 material wiring (image URIs; add KHR_texture_transform scale for tiling)."""
    u = map_urls(entry["id"], base_url)
    return {
        "name": entry["id"],
        "pbrMetallicRoughness": {"baseColorTexture": {"uri": u["albedo_delit"]},
                                 "metallicRoughnessTexture": {"uri": u["orm"], "note": "G=roughness, B=metallic (glTF native)"},
                                 "metallicFactor": 1.0, "roughnessFactor": 1.0},
        "normalTexture": {"uri": u["normal"], "note": "OpenGL +Y (glTF native)"},
        "occlusionTexture": {"uri": u["orm"], "note": "R channel"},
        "extensions": {"KHR_materials_ior": {"ior": entry["shader_setup"]["ior"]},
                       "KHR_texture_transform": {"scale": "[surface_w_m / tile_w_m, surface_h_m / tile_h_m]"}},
    }


# ---------------------------------------------------------------------------
# Public query API
# ---------------------------------------------------------------------------
def _index_entry(e: dict[str, Any], base: str) -> dict[str, Any]:
    ss = e["shader_setup"]
    rr = ss["roughness_range"]
    return {
        "id": e["id"], "name": e["name"], "category": e["category"], "subcategory": e["subcategory"],
        "scene_roles": e["scene_roles"], "best_objects": e["best_objects"], "tags": e["tags"],
        "size_m": e["real_world_size_m"], "projection": e["projection"]["recommended"],
        "metallic": ss["is_metallic"], "roughness": round((rr[0] + rr[1]) / 2.0, 3),
        "preview": f"{base}/assets/mats/{e['id']}/preview.webp", "spec": f"{base}/atlas/mats/{e['id']}.json",
    }


def get_atlas(base_url: str | None = DEFAULT_BASE_URL) -> dict[str, Any]:
    """Return the top-level atlas manifest (endpoints, categories, schema, URL templates)."""
    base = _base(base_url)
    cats = []
    for c in ac.CATEGORIES:
        cats.append({**c, "shard": f"{base}/atlas/categories/{c['slug']}.json"})
    return {
        "name": "hypbr PBR Material Atlas",
        "version": ac.ATLAS_VERSION,
        "material_count": len(ac.get_catalog()),
        "license": "MIT",
        "homepage": base,
        "repository": "https://github.com/wrong-nebula/hypbr",
        "endpoints": {
            "llms_txt": f"{base}/llms.txt",
            "agent_guide": f"{base}/agent.md",
            "manifest": f"{base}/atlas.json",
            "index": f"{base}/atlas/index.json",
            "category_shard": f"{base}/atlas/categories/{{category}}.json",
            "material_spec": f"{base}/atlas/mats/{{id}}.json",
            "texture": f"{base}/assets/mats/{{id}}/{{map}}.webp",
        },
        "maps": {k: {"file": MAP_FILES[k], "description": MAP_DESCRIPTIONS[k]} for k in MAP_FILES},
        "texture_resolution": 512,
        "categories": cats,
        "scene_roles": list(ac.SCENE_ROLES),
        "projections": list(ac.PROJECTIONS),
        "anisotropic_axes": list(ac.ANISOTROPIC_AXES),
        "shader_conventions": {
            **ac.SHADER_CONSTANTS,
            "threejs": "MeshPhysicalMaterial; aoMap/roughnessMap/metalnessMap all = orm (three reads R/G/B); normal is native.",
            "gltf": "orm is directly the glTF metallicRoughnessTexture (G,B) and occlusionTexture (R).",
            "blender": "Separate Color on orm: Green->Roughness, Blue->Metallic; Normal Map node (OpenGL).",
            "unreal": "DirectX normals: enable 'Flip Green Channel' on normal.webp; ORM packs R=AO G=Roughness B=Metallic (UE native order).",
            "unity": "OpenGL normals (native). URP/Built-in: metallic map + smoothness = 1 - roughness in alpha; HDRP mask map = (Metallic, AO, detail, Smoothness).",
        },
        "weathering_channels": {
            "wear": "Edge wear / polish on convex crests", "dirt": "Crevice grime", "moisture": "Wet darkening and puddles (hypbr 'wet')",
            "moss": "Biological growth", "rust": "Oxide / verdigris / efflorescence runoff", "frost": "Frost & rime (mapped to hypbr 'dust')",
            "dust": "Settled mineral dust",
        },
        "schema": {
            "id": "unique snake_case id", "name": "display name", "category": "one of 10 category slugs",
            "subcategory": "surface class", "description": "1-2 sentences",
            "generator": "{kind: flagship|parametric, family?, params?, seed}",
            "real_world_size_m": "[width_m, height_m] covered by ONE texture tile (UV 0..1)",
            "best_objects": "4-8 objects/meshes it belongs on", "avoid_objects": "2-4 objects where it looks wrong",
            "scene_roles": "subset of scene_roles", "projection": "{recommended, triplanar_fallback, world_scale_uv, anisotropic_axis, requires_tangents, tiling_note, uv_repeat_formula}",
            "shader_setup": "{normal_space, orm_packing, aux_packing, albedo_colorspace, data_colorspace, displacement_scale_m, ior, is_metallic, roughness_range, metallic_range, transmission}",
            "recommended_weathering": "{preset, wear, dirt, moisture, moss, rust, frost, dust, note, hypbr_params}",
            "pairs_well_with": "3-5 complementary atlas ids", "tags": "search keywords",
            "maps": "absolute texture URLs (spec cards only)", "snippets": "threejs / blender_bpy / gltf (spec cards only)",
        },
        "usage": {
            "uv_repeat": "repeat_u = surface_width_m / real_world_size_m[0]; repeat_v = surface_height_m / real_world_size_m[1]",
            "triplanar": "world-space scale = 1 / real_world_size_m (tiles per meter)",
        },
    }


def get_material_spec(material_id: str, base_url: str | None = DEFAULT_BASE_URL,
                      surface_m: tuple[float, float] | None = None) -> dict[str, Any]:
    """Return the full agent spec card: metadata, texture URLs and ready-to-paste snippets."""
    e = _with_measurements(ac.get_entry(material_id))
    base = _base(base_url)
    surf = surface_m or _default_surface(e)
    e["maps"] = map_urls(material_id, base)
    e["spec_url"] = f"{base}/atlas/mats/{material_id}.json"
    w, h = e["real_world_size_m"]
    e["uv_example"] = {"surface_m": list(surf), "repeat": [round(surf[0] / w, 3), round(surf[1] / h, 3)]}
    e["snippets"] = {
        "threejs": threejs_snippet(e, base, surf),
        "blender_bpy": blender_snippet(e, base, surf),
        "gltf": gltf_material(e, base),
    }
    return e


def _default_surface(e: dict[str, Any]) -> tuple[float, float]:
    for o in e["best_objects"]:
        for tok in reversed(o.split("_")):
            if tok in TYPICAL_SURFACE_M:
                return TYPICAL_SURFACE_M[tok]
    for r in e["scene_roles"]:
        if r in TYPICAL_SURFACE_M:
            return TYPICAL_SURFACE_M[r]
    return (2.0, 2.0)


def _norm_text(text: str) -> str:
    t = text.lower().replace("sci-fi", "scifi").replace("sci fi", "scifi")
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in _norm_text(text).split() if t and t not in _STOP]


@lru_cache(maxsize=1)
def _search_index() -> list[dict[str, Any]]:
    out = []
    for e in ac.get_catalog():
        objs = e["best_objects"]
        out.append({
            "e": e,
            "name": set(_tokens(e["id"].replace("_", " ") + " " + e["name"])),
            "tags": {_stem(t) for t in e["tags"]} | {_stem(p) for t in e["tags"] for p in t.split("_")},
            "obj_tok": {_stem(p) for o in objs for p in o.split("_")},
            "objs": [o.replace("_", " ") for o in objs],
            "roles": set(e["scene_roles"]),
            "sub": set(_tokens(e["subcategory"].replace("_", " ") + " " + ac.SUBCATEGORIES[e["subcategory"]]["label"])),
            "cat": set(_tokens(e["category"].replace("_", " "))),
            "desc": set(_tokens(e["description"])),
            "avoid": {_stem(p) for o in e["avoid_objects"] for p in o.split("_")},
            "avoid_full": [o.replace("_", " ") for o in e["avoid_objects"]],
        })
    return out


def _score(ix: dict[str, Any], toks: list[str], phrase: str) -> tuple[float, list[str]]:
    score, why, matched = 0.0, [], 0
    for t in toks:
        best, reason = 0.0, ""
        for i, s in enumerate((t,) + _SYNONYMS.get(t, ())):
            s = _stem(s)
            k = 1.0 if i == 0 else 0.8
            sc = 0.0
            if s in ix["name"]:
                sc += 3.0
            if s in ix["sub"]:
                sc += 3.0
            if s in ix["tags"]:
                sc += 2.0
            if s in ix["obj_tok"]:
                sc += 3.0
            elif len(s) >= 4 and any(o.startswith(s) or (" " + s) in (" " + o) for o in ix["objs"]):
                sc += 2.0
            if s in ix["roles"]:
                sc += 2.0
            if s in ix["cat"]:
                sc += 1.0
            if s in ix["desc"]:
                sc += 1.0
            if sc * k > best:
                best, reason = sc * k, s
        if best > 0:
            matched += 1
            why.append(reason)
        score += best
        if t in ix["avoid"] and best < 3.0:
            score -= 3.0
    if toks:
        score += 3.0 * (matched / len(toks)) ** 2
        if score > 0 and "classic" in ix["tags"]:
            score += 1.0
    if phrase and len(toks) > 1:
        if any(phrase in o for o in ix["objs"]):
            score += 4.0
        if any(phrase in o for o in ix["avoid_full"]):
            score -= 6.0
    return score, why


def search_atlas(
    query: str | None = None,
    category: str | None = None,
    object_type: str | None = None,
    scene_role: str | None = None,
    metallic: bool | None = None,
    max_roughness: float | None = None,
    limit: int = 20,
    base_url: str | None = DEFAULT_BASE_URL,
) -> list[dict[str, Any]]:
    """Ranked search over all 500 materials by free text, object, role and physical filters."""
    base = _base(base_url)
    meas = _measurements()
    q = " ".join(x for x in (query, object_type) if x)
    toks = _tokens(q)
    phrase = " ".join(_norm_text(q).split())
    results = []
    for ix in _search_index():
        e = ix["e"]
        if category and e["category"] != category and e["subcategory"] != category:
            continue
        if scene_role and scene_role not in e["scene_roles"]:
            continue
        ee = _with_measurements(copy.deepcopy(e), meas) if (metallic is not None or max_roughness is not None) else e
        if metallic is not None and bool(ee["shader_setup"]["is_metallic"]) != bool(metallic):
            continue
        if max_roughness is not None:
            rr = ee["shader_setup"]["roughness_range"]
            if (rr[0] + rr[1]) / 2.0 > max_roughness:
                continue
        if toks:
            sc, why = _score(ix, toks, phrase)
            if sc <= 0.0:
                continue
        else:
            sc, why = 0.0, []
        item = _index_entry(_with_measurements(copy.deepcopy(e), meas), base)
        item["score"] = round(sc, 2)
        item["matched"] = why
        results.append(item)
    if toks:
        results.sort(key=lambda r: (-r["score"], r["id"]))
    return results[: max(1, int(limit))]


def _env_of(e: dict[str, Any]) -> str:
    t = set(e["tags"])
    if t & ac._EXTERIOR_TAGS:
        return "exterior"
    return {"ext": "exterior", "int": "interior"}.get(ac._SUB_ENV.get(e["subcategory"], "both"), "both")


def adjust_weathering(base: dict[str, Any], environment: str | None, is_metal: bool = False) -> dict[str, Any]:
    """Adapt a material's recommended weathering to a scene environment keyword."""
    w = {k: float(base[k]) for k in ("wear", "dirt", "moisture", "moss", "rust", "frost", "dust")}
    env = _ENV_ALIASES.get(_norm_text(environment or "").replace(" ", "_"), None) if environment else None
    notes = []
    if env == "interior":
        w.update(moisture=0.0, moss=0.0, rust=min(w["rust"], 0.05), frost=0.0)
        notes.append("interior: no rain, moss, frost; minimal rust")
    elif env == "coastal":
        w["moisture"] += 0.20; w["dirt"] += 0.05
        w["rust"] += 0.30 if is_metal else 0.05
        notes.append("coastal: salt spray raises moisture and corrosion")
    elif env == "rainy":
        w["moisture"] += 0.35; w["moss"] += 0.05
        notes.append("rainy: wet darkening and puddles")
    elif env == "desert":
        w.update(moisture=0.0, moss=0.0); w["dust"] += 0.35; w["wear"] += 0.10
        notes.append("desert: sand-blasted wear and heavy dust, no moisture or moss")
    elif env == "abandoned":
        for k in ("wear", "dirt", "moss", "dust"):
            w[k] += 0.25
        w["rust"] += 0.35 if is_metal else 0.05
        notes.append("abandoned: heavy grime, growth and decay")
    elif env == "winter":
        w["frost"] += 0.45; w["moisture"] += 0.10; w["moss"] = min(w["moss"], 0.1)
        notes.append("winter: frost/rime (applied via hypbr dust channel)")
    elif env == "forest":
        w["moss"] += 0.25; w["moisture"] += 0.20
        notes.append("forest: damp shade grows moss")
    elif env == "urban":
        w["dirt"] += 0.15; w["dust"] += 0.10
        notes.append("urban: soot, grime and dust")
    elif env == "space":
        w.update(moisture=0.0, moss=0.0, rust=0.0, frost=0.0); w["wear"] += 0.10
        notes.append("space: micrometeorite wear only")
    w = {k: round(min(1.0, max(0.0, v)), 3) for k, v in w.items()}
    out = {"preset": base["preset"], **w, "note": base["note"] + (f" Environment adjustment - {'; '.join(notes)}." if notes else "")}
    out["hypbr_params"] = {"wear": w["wear"], "dirt": w["dirt"], "wet": w["moisture"], "moss": w["moss"],
                           "rust": w["rust"], "dust": round(max(w["dust"], 0.8 * w["frost"]), 3)}
    return out


def _surface_for(object_name: str, roles: list[str]) -> tuple[float, float]:
    for tok in reversed(_tokens(object_name)):
        if tok in TYPICAL_SURFACE_M:
            return TYPICAL_SURFACE_M[tok]
    for r in roles:
        if r in TYPICAL_SURFACE_M:
            return TYPICAL_SURFACE_M[r]
    return (2.0, 2.0)


def recommend_for_object(
    object_name: str,
    style: str | None = None,
    environment: str | None = None,
    limit: int = 5,
    surface_size_m: tuple[float, float] | None = None,
    base_url: str | None = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Recommend materials for an object, with computed UV repeat, projection and weathering advice."""
    base = _base(base_url)
    env = _ENV_ALIASES.get(_norm_text(environment or "").replace(" ", "_")) if environment else None
    pool = search_atlas(query=" ".join(x for x in (object_name, style) if x), limit=120, base_url=base)
    by_id = {e["id"]: e for e in ac.get_catalog()}
    obj_phrase = " ".join(_norm_text(object_name).split())
    ranked = []
    for r in pool:
        e = by_id[r["id"]]
        sc = r["score"]
        if obj_phrase and any(obj_phrase in o.replace("_", " ") for o in e["best_objects"]):
            sc += 3.0
        if env in ("interior",) and _env_of(e) == "exterior":
            sc -= 3.0
        if env in ("exterior", "coastal", "rainy", "desert", "winter", "forest", "urban", "abandoned") and _env_of(e) == "interior":
            sc -= 2.0
        ranked.append((sc, r["id"]))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    matches = []
    for sc, mid in ranked[: max(1, int(limit))]:
        e = _with_measurements(copy.deepcopy(by_id[mid]))
        surf = surface_size_m or _surface_for(object_name, e["scene_roles"])
        w, h = e["real_world_size_m"]
        matches.append({
            "id": mid, "name": e["name"], "category": e["category"], "subcategory": e["subcategory"],
            "score": round(sc, 2), "description": e["description"],
            "surface_size_m": [float(surf[0]), float(surf[1])],
            "uv_repeat": [round(surf[0] / w, 3), round(surf[1] / h, 3)],
            "projection": e["projection"],
            "weathering": adjust_weathering(e["recommended_weathering"], environment, e["shader_setup"]["is_metallic"]),
            "pairs_well_with": e["pairs_well_with"],
            "maps": map_urls(mid, base), "spec_url": f"{base}/atlas/mats/{mid}.json",
        })
    return {"object": object_name, "style": style, "environment": env or environment, "matches": matches}


# ---------------------------------------------------------------------------
# Static export (llms.txt, agent.md, atlas.json, atlas/index.json, shards, spec cards)
# ---------------------------------------------------------------------------
def _llms_txt(base: str, n: int) -> str:
    cats = "\n".join(f"- [{c['name']}]({base}/atlas/categories/{c['slug']}.json): {c['count']} materials. {c['description']}"
                     for c in ac.CATEGORIES)
    return f"""# hypbr PBR Material Atlas

> {n} physically calibrated, mathematically seamless (periodic) PBR materials with agent-ready metadata: what objects each material belongs on, exact real-world tile size in meters, UV/triplanar projection advice, engine shader wiring, weathering presets and pairing partners. All JSON and 512x512 WebP textures are static, CORS-enabled (`Access-Control-Allow-Origin: *`) and free to use (MIT).

## Endpoints
- [Agent guide]({base}/agent.md): how to pick, scale, project, wire and weather materials in any engine.
- [Manifest]({base}/atlas.json): version, count, schema, URL templates, category list.
- [Search index]({base}/atlas/index.json): all {n} materials in one compact file (id, name, category, roles, best_objects, tags, size_m, preview, spec URL).
- Material spec card: `{base}/atlas/mats/{{id}}.json` (full metadata + texture URLs + Three.js / Blender / glTF snippets).
- Textures: `{base}/assets/mats/{{id}}/{{albedo_delit|normal|orm|aux|preview}}.webp`
- Interactive playground: {base}

## Conventions
- albedo_delit = sRGB base color. normal = OpenGL +Y tangent space. orm = R AO, G Roughness, B Metallic (glTF/three.js native). aux = R Height, G Curvature, B Cavity. All data maps are linear.
- One texture tile covers `real_world_size_m = [w, h]` meters. UV repeat = surface_size_m / real_world_size_m. Triplanar scale = 1 / real_world_size_m.

## Quick start
1. GET `{base}/atlas/index.json`, filter by `best_objects`, `scene_roles`, `tags`.
2. GET `{base}/atlas/mats/<id>.json` and read `real_world_size_m`, `projection`, `shader_setup`, `recommended_weathering`.
3. Three.js: `const orm = load('orm.webp'); new THREE.MeshPhysicalMaterial({{ map: load('albedo_delit.webp', sRGB), normalMap: load('normal.webp'), aoMap: orm, roughnessMap: orm, metalnessMap: orm }})` with `texture.repeat.set(surfaceW / w, surfaceH / h)`.
4. Blender: Image Texture (sRGB) -> Base Color; orm (Non-Color) -> Separate Color: G -> Roughness, B -> Metallic; normal (Non-Color) -> Normal Map -> Normal; Mapping scale = surface / size.

## Categories
{cats}

## Optional
- [Python package](https://github.com/wrong-nebula/hypbr): `pip install git+https://github.com/wrong-nebula/hypbr`; `hypbr atlas search "brick wall"`, `hypbr atlas recommend sofa`, `hypbr atlas info <id>`; regenerate any material at any resolution with `hypbr synth <id> --size 2048`.
"""


def _agent_md(base: str, n: int) -> str:
    sample = get_material_spec("classic_red_brick", base, (4.0, 2.7))
    return f"""# hypbr Atlas - Agent Integration Guide

The hypbr atlas is a static, CORS-enabled catalog of **{n} seamless PBR materials**. Every material ships 512x512 WebP maps and a JSON spec card telling you exactly where it belongs, how big one tile is in the real world, how to project it and how to wire it into your engine. Use this guide to go from "texture this object" to a correctly scaled, physically wired material in one pass.

## 1. Discover
| Resource | URL |
| --- | --- |
| Manifest | `{base}/atlas.json` |
| Compact index ({n} entries) | `{base}/atlas/index.json` |
| Category shards (full metadata) | `{base}/atlas/categories/<category>.json` |
| Spec card (+ snippets) | `{base}/atlas/mats/<id>.json` |
| Textures | `{base}/assets/mats/<id>/<map>.webp` |

Categories: {", ".join(f"`{c['slug']}` ({c['count']})" for c in ac.CATEGORIES)}.

## 2. Pick a material
1. Fetch `atlas/index.json` once (it is small) and match your object against `best_objects` (exact object nouns such as `kitchen_countertop`, `exterior_building_wall`, `sofa`, `road`), then `scene_roles` (`{"`, `".join(ac.SCENE_ROLES)}`) and `tags` (color, style, era, biome, finish).
2. Reject candidates whose `avoid_objects` contain your object.
3. Prefer exterior materials outdoors and interior materials indoors; tags such as `exterior`, `coastal`, `garden` and the category tell you which.
4. Use `pairs_well_with` (3-5 atlas IDs) to choose coordinated neighbours: wall + floor + trim + glazing.

## 3. Scale it physically (never guess UV repeat)
`real_world_size_m = [w, h]` is the size in meters covered by **one** texture tile (UV 0..1).
- UV-mapped surface of `W x H` meters: `repeat_u = W / w`, `repeat_v = H / h`.
- World-space / triplanar: tiles per meter = `1 / w` (and `1 / h`).
- Example: `classic_red_brick` is `[0.9, 0.75]`; a 4 m x 2.7 m wall needs `repeat = [{4.0/0.9:.3f}, {2.7/0.75:.3f}]`.

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
{json.dumps({k: sample[k] for k in ("id", "real_world_size_m", "best_objects", "scene_roles", "projection", "shader_setup", "recommended_weathering", "pairs_well_with", "maps")}, indent=2)}
```

### Three.js
```js
{sample["snippets"]["threejs"]}```

### Blender (bpy)
```python
{sample["snippets"]["blender_bpy"]}```

## 8. Python (optional)
```python
from hypbr.atlas import search_atlas, recommend_for_object, get_material_spec
recommend_for_object("sofa", style="mid-century", environment="interior")
search_atlas("asphalt road", limit=5)
get_material_spec("calacatta_gold_marble")["snippets"]["threejs"]
```
CLI: `hypbr atlas search "brick wall"`, `hypbr atlas info <id>`, `hypbr atlas recommend "kitchen counter" --style modern`, `hypbr atlas export ./site`.
"""


def export_atlas_files(
    out_dir: str | Path,
    base_url: str | None = DEFAULT_BASE_URL,
    measurements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write `llms.txt`, `agent.md`, `atlas.json`, `atlas/index.json`, category shards and 500 spec cards."""
    base = _base(base_url)
    out = Path(out_dir)
    (out / "atlas" / "categories").mkdir(parents=True, exist_ok=True)
    (out / "atlas" / "mats").mkdir(parents=True, exist_ok=True)
    meas = measurements if measurements is not None else _measurements()
    entries = _entries(meas)
    n = len(entries)

    def _dump(path: Path, data: Any, pretty: bool = False) -> int:
        txt = json.dumps(data, indent=1 if pretty else None, separators=None if pretty else (",", ":"), ensure_ascii=False)
        path.write_text(txt, encoding="utf-8")
        return len(txt.encode("utf-8"))

    sizes: dict[str, int] = {}
    manifest = get_atlas(base)
    manifest["material_count"] = n
    sizes["atlas.json"] = _dump(out / "atlas.json", manifest, pretty=True)
    index = {"version": ac.ATLAS_VERSION, "count": n, "base_url": base,
             "spec_template": f"{base}/atlas/mats/{{id}}.json",
             "materials": [_index_entry(e, base) for e in entries]}
    sizes["atlas/index.json"] = _dump(out / "atlas" / "index.json", index)
    for c in ac.CATEGORIES:
        mats = []
        for e in entries:
            if e["category"] == c["slug"]:
                ee = copy.deepcopy(e)
                ee["maps"] = map_urls(e["id"], base)
                ee["spec_url"] = f"{base}/atlas/mats/{e['id']}.json"
                mats.append(ee)
        sizes[f"atlas/categories/{c['slug']}.json"] = _dump(out / "atlas" / "categories" / f"{c['slug']}.json",
                                                             {"version": ac.ATLAS_VERSION, "category": c, "count": len(mats), "materials": mats})
    total_cards = 0
    for e in entries:
        card = get_material_spec(e["id"], base)
        card = _with_measurements(card, meas)
        total_cards += _dump(out / "atlas" / "mats" / f"{e['id']}.json", card, pretty=True)
    sizes["atlas/mats/*.json"] = total_cards
    (out / "llms.txt").write_text(_llms_txt(base, n), encoding="utf-8")
    (out / "agent.md").write_text(_agent_md(base, n), encoding="utf-8")
    sizes["llms.txt"] = (out / "llms.txt").stat().st_size
    sizes["agent.md"] = (out / "agent.md").stat().st_size
    return {"out_dir": str(out), "materials": n, "bytes": sizes, "files": 4 + len(ac.CATEGORIES) + n}


# ---------------------------------------------------------------------------
# Playground catalog entries (shared by playground/bake_static.py and playground/server.py)
# ---------------------------------------------------------------------------
def playground_entries(asset_prefix: str | None = "./assets/mats") -> list[dict[str, Any]]:
    """Lean per-material rows for the playground UI (search fields + bundle URLs)."""
    from .weathering import infer_material_kind

    rows = []
    cat_names = {c["slug"]: c["name"] for c in ac.CATEGORIES}
    for e in _entries():
        rr = e["shader_setup"]["roughness_range"]
        mr = e["shader_setup"]["metallic_range"]
        def_m = 0.95 if e["shader_setup"]["is_metallic"] else round(float(mr[1]) * 0.5, 3)
        tint = e["generator"]["params"]["c1"] if e["generator"]["kind"] == "parametric" else None
        if tint is None:
            from . import procedural as gd
            tint = next((t for did, _d, _k, t in gd.DAILY_CATALOG if did == e["id"]), (0.28, 0.31, 0.36))
        srgb = [max(0, min(255, int(round((max(0.0, float(c)) ** (1 / 2.2)) * 255)))) for c in tint]
        row = {
            "id": e["id"], "name": e["name"], "description": e["description"], "kind": e["kind"],
            "group": cat_names[e["category"]], "category": e["category"], "subcategory": e["subcategory"],
            "archetype": infer_material_kind(e["id"], def_m),
            "tint": [round(float(c), 4) for c in tint], "tint_hex": "#{:02x}{:02x}{:02x}".format(*srgb),
            "default_seed": int(e["generator"]["seed"]),
            "default_roughness": round((rr[0] + rr[1]) / 2.0, 3), "default_metallic": def_m,
            "source": "hardops" if e["id"] == "hardops_plate" else ("daily_catalog" if e["generator"]["kind"] == "flagship" else "atlas"),
            "best_objects": e["best_objects"], "scene_roles": e["scene_roles"], "tags": e["tags"],
            "size_m": e["real_world_size_m"], "projection": e["projection"]["recommended"],
            "stage": _stage_for(e),
        }
        if asset_prefix:
            rel = f"{asset_prefix}/{e['id']}"
            row["preview_url"] = f"{rel}/preview.webp"
            row["bundle"] = {"albedo": f"{rel}/albedo_delit.webp", "normal": f"{rel}/normal.webp",
                             "orm": f"{rel}/orm.webp", "aux": f"{rel}/aux.webp", "preview": f"{rel}/preview.webp"}
        rows.append(row)
    return rows


def _stage_for(e: dict[str, Any]) -> dict[str, Any]:
    """Best preview stage (geometry, UV repeat, light rig) at physically sensible scale."""
    roles = e["scene_roles"]
    w = e["real_world_size_m"][0]
    rig = "studio" if e["category"] == "glass_optics" else "raking"
    if e["subcategory"] in ("woven_textile", "plush_textile", "leather", "composite_weave", "specialty_metal") and "floor" not in roles:
        geom, span = "sphere", 0.5
    elif e["subcategory"] in ("machined_metal", "tire_tread", "bark", "heatsink") and "floor" not in roles:
        geom, span = "cylinder", 0.4
    elif any(r in roles for r in ("floor", "ground", "road", "roof")):
        geom, span = "plane", 2.4
    elif "wall" in roles or "glass_window" in roles:
        geom, span = "wall2x2", 2.0
    else:
        geom, span = "box", 0.8
    rep = min(8.0, max(1.0, round((span / max(w, 1e-3)) * 2.0) / 2.0))
    return {"geom": geom, "repeat": rep, "lightRig": rig}
