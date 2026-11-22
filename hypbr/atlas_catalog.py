"""The 500-material agent-ready hypbr PBR Atlas catalog.

Every entry carries the full agent metadata contract: physical tile size, best/avoid
objects, scene roles, projection advice, engine shader wiring, weathering presets and
verified pairing partners. 69 entries are the hand-tuned flagship generators
(`procedural.DAILY_CATALOG` + `hardops_plate`); 431 are curated parametric materials
built by `atlas_generators.build_parametric_material`.

All parametric parameters respect the periodicity rules of their generator family
(integer lattice counts, even row counts for staggered layouts, integer ring cycles
for wood grain) so every map wraps with zero seams on `[0, 1)^2`.
"""
from __future__ import annotations

import colorsys
import copy
import hashlib
from functools import lru_cache
from typing import Any

from .color import linear_to_srgb

ATLAS_VERSION = "1.1.0"

# ---------------------------------------------------------------------------
# Categories (exact target counts sum to 500)
# ---------------------------------------------------------------------------
CATEGORIES: list[dict[str, Any]] = [
    {"slug": "architectural_masonry", "name": "Architectural Masonry", "count": 65,
     "description": "Bricks, concrete blocks, ashlar, stacked stone, cobbles and pavers for building shells and hardscape."},
    {"slug": "wood_timber", "name": "Wood & Timber", "count": 58,
     "description": "Plank floors, parquet, cladding, slats, charred timber, engineered boards and tree bark."},
    {"slug": "stone_marble_concrete", "name": "Stone, Marble & Concrete", "count": 58,
     "description": "Marbles, granites, terrazzo, quartz, concrete, plaster, stucco, slate and natural stone slabs."},
    {"slug": "tiles_ceramics", "name": "Tiles & Ceramics", "count": 46,
     "description": "Subway, zellige, porcelain, mosaics, crackle and stoneware glazes, terracotta."},
    {"slug": "metals_industrial", "name": "Metals & Industrial", "count": 58,
     "description": "Brushed, anodized, galvanized, hammered, corten and damascus metals, cladding, meshes, treads and grips."},
    {"slug": "fabrics_upholstery", "name": "Fabrics & Upholstery", "count": 58,
     "description": "Linen, denim, canvas, plaid, boucle, velvet, corduroy, knits, carpet, felt, leathers, wicker and cork."},
    {"slug": "nature_terrain", "name": "Nature & Terrain", "count": 40,
     "description": "Sand, cracked mud, gravel, ice, snow, grass, soil, moss, forest floor, cliffs and thatch."},
    {"slug": "glass_optics", "name": "Glass & Optics", "count": 35,
     "description": "Fluted, reeded, block, frosted, pebbled, cathedral, wired and tinted architectural glazing."},
    {"slug": "rubber_polymers_roads", "name": "Rubber, Polymers & Roads", "count": 42,
     "description": "Asphalt, tire treads, rubber mats, silicone, carbon fiber, foams and molded plastics."},
    {"slug": "roofing_tech_scifi", "name": "Roofing, Tech & Sci-Fi", "count": 40,
     "description": "Slate, shake, barrel and fishscale roofing, PCBs, solar arrays, sci-fi hull plating, heatsinks and acoustic foam."},
]
CATEGORY_BY_SLUG = {c["slug"]: c for c in CATEGORIES}

SCENE_ROLES = (
    "wall", "floor", "ceiling", "roof", "ground", "road", "furniture", "prop", "trim",
    "hero_asset", "mechanical", "foliage_bark", "glass_window",
)
PROJECTIONS = ("uv", "triplanar", "box")
ANISOTROPIC_AXES = ("none", "u", "v", "radial")

SHADER_CONSTANTS = {
    "normal_space": "OpenGL (+Y)",
    "orm_packing": "R=AO, G=Roughness, B=Metallic",
    "aux_packing": "R=Height, G=Curvature, B=Cavity",
    "albedo_colorspace": "sRGB",
    "data_colorspace": "Linear",
}

# Per-kind (roughness, metallic) defaults used when no measurement is available
KIND_DEFAULT_META: dict[str, tuple[float, float]] = {
    "metal": (0.28, 0.95), "copper": (0.26, 0.96), "bronze": (0.34, 0.85), "brass": (0.24, 0.95),
    "stone": (0.55, 0.0), "concrete": (0.78, 0.0), "asphalt": (0.82, 0.0), "plaster": (0.64, 0.0),
    "wood": (0.52, 0.0), "natural": (0.74, 0.0), "paper": (0.84, 0.0), "ceramic": (0.22, 0.0),
    "glass": (0.12, 0.0), "fabric": (0.86, 0.0), "leather": (0.42, 0.0), "rubber": (0.68, 0.0),
    "carbon": (0.20, 0.42),
}


# ---------------------------------------------------------------------------
# Subcategory profiles (shared agent metadata per surface class)
# ---------------------------------------------------------------------------
def _proj(rec: str = "uv", tri: bool = True, world: bool = True, axis: str = "none",
          tangents: bool | None = None, note: str = "") -> dict[str, Any]:
    return {
        "recommended": rec,
        "triplanar_fallback": tri,
        "world_scale_uv": world,
        "anisotropic_axis": axis,
        "requires_tangents": (axis != "none") if tangents is None else tangents,
        "tiling_note": note,
    }


def _wx(preset: str, wear: float, dirt: float, moisture: float, moss: float, rust: float,
        frost: float, dust: float, note: str) -> dict[str, Any]:
    return {"preset": preset, "wear": wear, "dirt": dirt, "moisture": moisture, "moss": moss,
            "rust": rust, "frost": frost, "dust": dust, "note": note}


def _sub(cat: str, label: str, kind: str, rough: float, slope: float, disp: float, ior: float,
         objects: list[str], avoid: list[str], roles: list[str], proj: dict[str, Any],
         weather: dict[str, Any], pairs: list[str], sentence: str, transmission: float = 0.0) -> dict[str, Any]:
    return {"category": cat, "label": label, "kind": kind, "rough": rough, "slope": slope,
            "disp": disp, "ior": ior, "objects": objects, "avoid": avoid, "roles": roles,
            "projection": proj, "weather": weather, "pairs": pairs, "sentence": sentence,
            "transmission": transmission}


_AM, _WT, _SM, _TC, _MI, _FU, _NT, _GO, _RP, _RT = (c["slug"] for c in CATEGORIES)

SUBCATEGORIES: dict[str, dict[str, Any]] = {
    # ---------------- architectural masonry ----------------
    "brick": _sub(_AM, "Brick", "stone", 0.80, 10.0, 0.012, 1.5,
        ["exterior_building_wall", "chimney", "garden_retaining_wall", "loft_accent_wall", "archway", "fireplace_surround"],
        ["kitchen_countertop", "sofa", "car_body", "window_pane"],
        ["wall", "trim"],
        _proj("uv", True, True, "none", False, "Keep courses horizontal: map U along the wall length and V up the wall; use box mapping for chimneys."),
        _wx("aged", 0.30, 0.40, 0.15, 0.12, 0.0, 0.0, 0.10, "Grime and moss settle in recessed mortar joints; keep rust at 0 unless iron fixings drip."),
        ["paver", "roofing", "glazing", "concrete", "hardwood_floor"],
        "Calibrated running-bond masonry with recessed mortar joints and per-unit color variation."),
    "concrete_block": _sub(_AM, "Concrete Block", "concrete", 0.84, 9.0, 0.010, 1.5,
        ["warehouse_wall", "basement_wall", "parking_garage_wall", "utility_building", "garden_wall", "foundation"],
        ["kitchen_countertop", "sofa", "jewelry", "window_pane"],
        ["wall"],
        _proj("box", True, True, "none", False, "Box-map blocky architecture so block courses stay level on every face."),
        _wx("aged", 0.25, 0.45, 0.20, 0.08, 0.05, 0.0, 0.20, "Porous block soaks up rain streaks and dust; add light rust only below steel lintels."),
        ["asphalt", "metal_cladding", "concrete", "roofing"],
        "Modular CMU masonry with tooled mortar joints and porous cast faces."),
    "ashlar": _sub(_AM, "Ashlar Stone", "stone", 0.76, 11.0, 0.020, 1.5,
        ["castle_wall", "cathedral_facade", "bridge_pier", "palace_exterior", "monument_base", "garden_wall"],
        ["sofa", "kitchen_backsplash", "car_body", "laptop_shell"],
        ["wall", "hero_asset"],
        _proj("box", True, True, "none", False, "Box or UV map with courses horizontal; large stones read best at 1 tile per 1.2 m."),
        _wx("aged", 0.35, 0.40, 0.20, 0.25, 0.0, 0.0, 0.10, "Centuries-old dressed stone: soften arrises with wear and let moss colonize the lower courses."),
        ["cobblestone", "roofing", "glazing", "hardwood_floor"],
        "Dressed stone blocks laid in coursed ashlar with fine lime joints."),
    "stacked_stone": _sub(_AM, "Stacked Stone", "stone", 0.82, 14.0, 0.030, 1.5,
        ["fireplace_feature_wall", "garden_retaining_wall", "landscape_pillar", "exterior_accent_wall", "outdoor_kitchen_base", "mailbox_column"],
        ["floor_tile_interior", "sofa", "tabletop", "window_pane"],
        ["wall"],
        _proj("box", True, True, "none", False, "Thin ledgers must run horizontally; avoid rotating the texture on vertical faces."),
        _wx("mossy", 0.20, 0.35, 0.20, 0.30, 0.0, 0.0, 0.05, "Deep dry-stack recesses trap dirt and moss; ideal for garden and exterior use."),
        ["wood_cladding", "paver", "roofing", "ground_cover"],
        "Dry-stacked ledgestone with deep shadow gaps and split-face relief."),
    "cobblestone": _sub(_AM, "Cobblestone", "stone", 0.74, 13.0, 0.030, 1.5,
        ["old_town_street", "plaza", "courtyard", "garden_path", "castle_bailey", "alleyway"],
        ["interior_wall", "sofa", "kitchen_countertop", "ceiling"],
        ["ground", "road", "floor"],
        _proj("uv", True, True, "none", False, "Project top-down on ground meshes; use triplanar on uneven terrain to avoid stretching."),
        _wx("rain", 0.35, 0.45, 0.45, 0.20, 0.0, 0.0, 0.05, "Street stones polish on crowns and hold rain puddles and grit between setts."),
        ["ashlar", "brick", "ground_cover", "rock_cliff"],
        "Hand-laid paving stones with domed crowns and gritty sand joints."),
    "paver": _sub(_AM, "Pavers", "stone", 0.78, 9.0, 0.010, 1.5,
        ["patio", "driveway", "garden_path", "pool_deck", "sidewalk", "plaza"],
        ["interior_wall", "sofa", "ceiling", "car_body"],
        ["ground", "floor", "road"],
        _proj("uv", True, True, "none", False, "Map top-down with U along the main walking direction."),
        _wx("aged", 0.25, 0.40, 0.25, 0.15, 0.0, 0.0, 0.15, "Outdoor pavers gather grit in joints and weeds along edges."),
        ["brick", "ground_cover", "wood_slats", "glazing"],
        "Precast or clay pavers in interlocking patterns with sand-swept joints."),
    # ---------------- wood & timber ----------------
    "hardwood_floor": _sub(_WT, "Hardwood Flooring", "wood", 0.46, 8.0, 0.004, 1.53,
        ["living_room_floor", "bedroom_floor", "hallway_floor", "stage_floor", "staircase_tread", "tabletop"],
        ["exterior_wall", "roof", "road", "shower_floor"],
        ["floor", "furniture"],
        _proj("uv", False, True, "u", True, "Grain runs along U; align U with the long room axis. Avoid triplanar, which scrambles board direction."),
        _wx("lived_in", 0.25, 0.15, 0.0, 0.0, 0.0, 0.0, 0.10, "Interior wood shows traffic wear and dust in board seams, never moss or rust."),
        ["plaster_stucco", "woven_textile", "plush_textile", "brick", "subway_tile"],
        "Tongue-and-groove boards with physically scaled growth rings, pores and staggered end joints."),
    "parquet": _sub(_WT, "Parquet", "wood", 0.42, 8.0, 0.003, 1.53,
        ["ballroom_floor", "parisian_apartment_floor", "library_floor", "gallery_floor", "hotel_lobby_floor"],
        ["exterior_wall", "roof", "road", "bathroom_wall"],
        ["floor"],
        _proj("uv", False, True, "none", False, "Herringbone and chevron read correctly only with planar UVs; keep the pattern axis parallel to walls."),
        _wx("lived_in", 0.30, 0.15, 0.0, 0.0, 0.0, 0.0, 0.10, "Heirloom parquet wears along walkways; add dust in micro-bevels."),
        ["plaster_stucco", "plush_textile", "marble", "woven_textile"],
        "Interlocking parquet blocks with micro-bevelled joints and alternating grain direction."),
    "wood_cladding": _sub(_WT, "Wood Cladding", "wood", 0.62, 9.0, 0.008, 1.53,
        ["cabin_exterior", "facade_cladding", "barn_wall", "ceiling_paneling", "sauna_interior", "porch_ceiling"],
        ["kitchen_countertop", "road", "car_body", "floor_tile_interior"],
        ["wall", "ceiling"],
        _proj("uv", False, True, "u", True, "Keep boards aligned with the building axis; UV-unwrap walls flat rather than triplanar."),
        _wx("aged", 0.30, 0.30, 0.25, 0.15, 0.0, 0.0, 0.10, "Exterior boards silver, streak with rain and grow moss near the ground."),
        ["roofing", "stacked_stone", "glazing", "metal_cladding"],
        "Shiplap, board-and-batten and siding boards with shadow grooves."),
    "wood_slats": _sub(_WT, "Wood Slats", "wood", 0.52, 9.0, 0.012, 1.53,
        ["acoustic_wall_panel", "privacy_screen", "fence", "pergola", "ceiling_baffle", "bench_seat"],
        ["road", "kitchen_countertop", "car_body", "terrain_ground"],
        ["wall", "ceiling", "furniture", "prop"],
        _proj("uv", False, True, "v", True, "Slats run along V; rotate UVs 90 degrees for horizontal installations."),
        _wx("lived_in", 0.20, 0.20, 0.05, 0.0, 0.0, 0.0, 0.15, "Dust collects in the recessed gaps; outdoor fences can take aged/mossy presets."),
        ["concrete", "plaster_stucco", "hardwood_floor", "plush_textile"],
        "Parallel slats with deep shadow gaps for acoustic walls, screens and fences."),
    "bark": _sub(_WT, "Tree Bark", "wood", 0.86, 14.0, 0.030, 1.53,
        ["tree_trunk", "log", "firewood_pile", "branch", "stump", "rustic_post"],
        ["floor", "kitchen_countertop", "interior_wall", "window_pane"],
        ["foliage_bark", "prop"],
        _proj("uv", True, False, "v", True, "Wrap cylindrical trunk UVs so fissures run along V (up the trunk); use triplanar on roots and stumps."),
        _wx("mossy", 0.10, 0.25, 0.30, 0.45, 0.0, 0.0, 0.0, "Living bark carries moss and lichen on the shaded, wetter side."),
        ["ground_cover", "rock_cliff", "gravel", "thatch"],
        "Organic bark cortex with fissured ridges and periodic plates."),
    "engineered_wood": _sub(_WT, "Engineered Wood & Board", "wood", 0.66, 6.0, 0.003, 1.53,
        ["workbench", "crate", "shelving", "subfloor", "packaging_box", "cutting_board", "construction_hoarding"],
        ["exterior_facade", "road", "roof", "window_pane"],
        ["prop", "furniture", "wall"],
        _proj("box", True, True, "none", False, "Box-map boxy props; sheet goods read correctly at real 1.22 m sheet scale."),
        _wx("lived_in", 0.30, 0.30, 0.05, 0.0, 0.0, 0.0, 0.25, "Workshop boards gather dust, scuffs and grime on edges."),
        ["concrete", "machined_metal", "brushed_metal", "molded_plastic"],
        "Manufactured timber products: strand boards, plywood, butcher block and fiber boards."),
    # ---------------- stone, marble & concrete ----------------
    "marble": _sub(_SM, "Marble & Onyx", "stone", 0.16, 3.0, 0.001, 1.486,
        ["kitchen_countertop", "bathroom_vanity", "lobby_floor", "feature_wall", "table_top", "staircase", "column"],
        ["road", "roof", "terrain_ground", "fence"],
        ["floor", "wall", "furniture", "hero_asset"],
        _proj("uv", True, True, "none", False, "Use one continuous UV island per slab so veins flow across edges; bookmatch by mirroring UVs."),
        _wx("light_use", 0.10, 0.10, 0.0, 0.0, 0.0, 0.0, 0.05, "Polished stone stays clean; add faint etch wear around sinks and thresholds."),
        ["brushed_metal", "hardwood_floor", "glazing", "plush_textile", "porcelain"],
        "Polished natural stone with physically scaled veining and soft cloud variation."),
    "granite_quartz": _sub(_SM, "Granite, Quartz & Terrazzo", "stone", 0.18, 4.0, 0.001, 1.54,
        ["kitchen_countertop", "kitchen_island", "bathroom_vanity", "lobby_floor", "bar_top", "reception_desk"],
        ["road", "roof", "terrain_ground", "sofa"],
        ["floor", "furniture", "wall"],
        _proj("uv", True, True, "none", False, "Speckled aggregates hide seams well; triplanar works on chunky countertop edges."),
        _wx("light_use", 0.10, 0.10, 0.0, 0.0, 0.0, 0.0, 0.05, "Hard aggregates resist wear; only faint scuffs and dust."),
        ["brushed_metal", "hardwood_floor", "subway_tile", "molded_plastic"],
        "Crystalline speckled aggregate slabs, engineered quartz and terrazzo."),
    "concrete": _sub(_SM, "Concrete", "concrete", 0.78, 5.0, 0.004, 1.5,
        ["building_facade", "parking_garage_column", "loft_floor", "skate_park", "bunker_wall", "staircase", "bridge_pier"],
        ["sofa", "jewelry", "window_pane", "tree_trunk"],
        ["wall", "floor", "ceiling"],
        _proj("triplanar", True, True, "none", False, "Triplanar suits monolithic pours; board-formed variants need UVs with boards horizontal."),
        _wx("aged", 0.25, 0.45, 0.25, 0.10, 0.05, 0.0, 0.20, "Concrete stains under drips, darkens when wet and grows algae at the base."),
        ["brushed_metal", "glazing", "wood_slats", "asphalt"],
        "Architectural concrete with trowel marks, pores and form-work imprints."),
    "plaster_stucco": _sub(_SM, "Plaster & Stucco", "plaster", 0.70, 4.0, 0.002, 1.5,
        ["interior_wall", "ceiling", "mediterranean_villa_exterior", "bathroom_wall", "archway", "niche"],
        ["floor", "road", "kitchen_countertop", "car_body"],
        ["wall", "ceiling"],
        _proj("triplanar", True, True, "none", False, "Triplanar or world-space UVs keep trowel swirls continuous across corners."),
        _wx("light_use", 0.15, 0.20, 0.05, 0.0, 0.0, 0.0, 0.15, "Interior plaster takes scuffs at hand height; exterior stucco takes rain streaks."),
        ["hardwood_floor", "terracotta", "wood_slats", "zellige"],
        "Hand-applied lime plaster, stucco and drywall finishes with soft trowel relief."),
    "natural_stone": _sub(_SM, "Natural Stone", "stone", 0.62, 8.0, 0.006, 1.5,
        ["patio_floor", "bathroom_floor", "hearth", "garden_wall_coping", "pool_surround", "entry_steps"],
        ["sofa", "car_body", "window_pane", "ceiling"],
        ["floor", "wall"],
        _proj("uv", True, True, "none", False, "Planar UVs on floors; triplanar on hearths and coping blocks."),
        _wx("aged", 0.20, 0.30, 0.15, 0.10, 0.0, 0.0, 0.10, "Natural stone dulls with foot traffic and holds dirt in pores and clefts."),
        ["hardwood_floor", "wood_cladding", "glazing", "paver"],
        "Cleft, honed and thermal-finished natural stone slabs."),
    # ---------------- tiles & ceramics ----------------
    "subway_tile": _sub(_TC, "Subway Tile", "ceramic", 0.14, 8.0, 0.003, 1.5,
        ["kitchen_backsplash", "bathroom_wall", "shower_surround", "metro_station_wall", "cafe_wall", "restaurant_kitchen"],
        ["road", "roof", "terrain_ground", "sofa"],
        ["wall"],
        _proj("uv", False, True, "none", False, "Planar UVs with courses horizontal; never triplanar (grout lines would shear at corners)."),
        _wx("light_use", 0.10, 0.25, 0.10, 0.0, 0.0, 0.0, 0.05, "Glazed faces stay bright while grout darkens with kitchen grease and soap scum."),
        ["hardwood_floor", "marble", "brushed_metal", "porcelain"],
        "Glazed ceramic wall tiles with bevelled edges and recessed sanded grout."),
    "zellige": _sub(_TC, "Zellige & Handmade Tile", "ceramic", 0.10, 8.0, 0.003, 1.5,
        ["kitchen_backsplash", "shower_wall", "fireplace_surround", "bar_front", "hammam_wall", "fountain"],
        ["road", "roof", "terrain_ground", "sofa"],
        ["wall", "floor"],
        _proj("uv", False, True, "none", False, "Planar UVs only; the hand-made wobble needs straight grout courses to read."),
        _wx("light_use", 0.10, 0.20, 0.10, 0.0, 0.0, 0.0, 0.05, "Glossy glaze pools on rims; only grout picks up grime."),
        ["plaster_stucco", "terracotta", "brushed_metal", "hardwood_floor"],
        "Hand-cut glazed tiles with undulating faces, kiln variation and glaze pooling."),
    "porcelain": _sub(_TC, "Porcelain Tile", "ceramic", 0.30, 5.0, 0.002, 1.5,
        ["living_room_floor", "commercial_lobby_floor", "bathroom_floor", "shower_wall", "retail_floor", "patio_floor"],
        ["roof", "terrain_ground", "sofa", "tree_trunk"],
        ["floor", "wall"],
        _proj("uv", False, True, "none", False, "Planar UVs aligned to the room grid; large formats tolerate world-scale UVs."),
        _wx("light_use", 0.10, 0.20, 0.10, 0.0, 0.0, 0.0, 0.10, "Hard porcelain resists wear; dirt shows only in thin grout joints."),
        ["plaster_stucco", "brushed_metal", "glazing", "wood_slats"],
        "Rectified porcelain tiles with tight grout joints."),
    "mosaic": _sub(_TC, "Mosaic Tile", "ceramic", 0.20, 8.0, 0.002, 1.5,
        ["bathroom_floor", "shower_floor", "pool_lining", "spa_wall", "kitchen_backsplash", "fountain_basin"],
        ["road", "roof", "sofa", "terrain_ground"],
        ["floor", "wall"],
        _proj("uv", False, True, "none", False, "Planar UVs; small chips tolerate gentle curvature on pool coves."),
        _wx("rain", 0.10, 0.30, 0.35, 0.0, 0.0, 0.0, 0.0, "Wet-area mosaics hold moisture and soap scum in the dense grout grid."),
        ["subway_tile", "marble", "brushed_metal", "glazing"],
        "Small-format mosaic chips set in a dense grout matrix."),
    "glaze_ceramic": _sub(_TC, "Glazed Ceramic & Stoneware", "ceramic", 0.18, 5.0, 0.001, 1.5,
        ["vase", "tableware", "sink_basin", "planter", "pottery_prop", "decorative_wall_tile"],
        ["road", "terrain_ground", "roof", "sofa"],
        ["prop", "hero_asset", "wall"],
        _proj("triplanar", True, False, "none", False, "Triplanar for thrown pottery and organic props; UV for flat decorative tiles."),
        _wx("light_use", 0.10, 0.10, 0.0, 0.0, 0.0, 0.0, 0.10, "Pottery wears on foot rings and rims; keep weathering subtle."),
        ["hardwood_floor", "natural_fiber", "plaster_stucco", "woven_textile"],
        "Kiln-fired glazes: celadon crackle, raku, speckled stoneware and enamel."),
    "terracotta": _sub(_TC, "Terracotta & Cement Tile", "ceramic", 0.68, 7.0, 0.003, 1.5,
        ["farmhouse_kitchen_floor", "patio_floor", "conservatory_floor", "hacienda_hallway", "pool_terrace", "wine_cellar"],
        ["roof", "sofa", "car_body", "window_pane"],
        ["floor", "wall"],
        _proj("uv", False, True, "none", False, "Planar UVs with the grout grid aligned to walls."),
        _wx("aged", 0.30, 0.30, 0.10, 0.05, 0.0, 0.0, 0.15, "Unsealed clay burnishes along paths and darkens with grime."),
        ["plaster_stucco", "hardwood_floor", "zellige", "natural_fiber"],
        "Fired clay and pigmented cement tiles with hand-made irregularity."),
    # ---------------- metals & industrial ----------------
    "brushed_metal": _sub(_MI, "Brushed & Anodized Metal", "metal", 0.28, 2.0, 0.0005, 1.5,
        ["appliance_panel", "elevator_door", "kitchen_hood", "laptop_shell", "column_cladding", "signage_plate", "handrail"],
        ["terrain_ground", "sofa", "roof_shingles", "tree_trunk"],
        ["mechanical", "trim", "prop", "wall"],
        _proj("uv", False, True, "u", True, "Brush lines run along U; align U with the longest panel edge and enable anisotropy."),
        _wx("light_use", 0.15, 0.10, 0.0, 0.0, 0.0, 0.0, 0.05, "Fingerprint smudges and fine scratches; avoid rust on stainless and aluminum."),
        ["marble", "glazing", "concrete", "porcelain"],
        "Directionally brushed metal with physically calibrated anisotropic micro-grooves."),
    "specialty_metal": _sub(_MI, "Specialty Metal Finishes", "metal", 0.34, 6.0, 0.001, 1.5,
        ["sink_basin", "bar_top", "sculpture", "facade_panel", "planter", "cookware", "blade"],
        ["terrain_ground", "sofa", "window_pane", "carpet"],
        ["prop", "hero_asset", "wall", "furniture"],
        _proj("uv", True, True, "none", False, "Organic finishes tolerate triplanar on sculpted props."),
        _wx("aged", 0.25, 0.20, 0.10, 0.0, 0.30, 0.0, 0.05, "Hammered and cast metals tarnish in hollows; corten and iron welcome extra rust."),
        ["concrete", "hardwood_floor", "natural_stone", "glazing"],
        "Hammered, galvanized, weathering-steel, damascus and cast metal surfaces."),
    "metal_cladding": _sub(_MI, "Metal Cladding & Roofing", "metal", 0.40, 8.0, 0.010, 1.5,
        ["industrial_roof", "barn_roof", "warehouse_wall", "shipping_container", "modern_facade", "shed"],
        ["sofa", "kitchen_countertop", "terrain_ground", "carpet"],
        ["roof", "wall"],
        _proj("uv", False, True, "v", True, "Profiles run along V (down the roof slope); keep ribs vertical on walls."),
        _wx("aged", 0.25, 0.30, 0.25, 0.05, 0.35, 0.0, 0.10, "Sheet metal streaks below laps and rusts at cut edges and fasteners."),
        ["concrete_block", "concrete", "wood_cladding", "glazing"],
        "Architectural sheet-metal profiles: standing seam, corrugated, fluted and riveted."),
    "perforated_metal": _sub(_MI, "Perforated Metal & Mesh", "metal", 0.32, 9.0, 0.003, 1.5,
        ["speaker_grille", "facade_screen", "ventilation_panel", "stair_tread", "locker_door", "light_fixture_shade"],
        ["terrain_ground", "sofa", "floor_tile_interior", "tree_trunk"],
        ["mechanical", "wall", "trim"],
        _proj("uv", False, True, "none", False, "Use planar UVs; holes are baked dark (or cut with alpha from the aux cavity channel)."),
        _wx("aged", 0.20, 0.35, 0.10, 0.0, 0.25, 0.0, 0.20, "Dust packs into perforations; exterior screens rust at hole edges."),
        ["concrete", "brushed_metal", "glazing", "machined_metal"],
        "Punched, slotted, stamped and woven-wire metal screens with dark through-holes."),
    "machined_metal": _sub(_MI, "Machined Metal & Tread", "metal", 0.30, 10.0, 0.002, 1.5,
        ["tool_handle", "control_knob", "flashlight_body", "industrial_floor", "truck_bed", "step_plate"],
        ["sofa", "terrain_ground", "window_pane", "ceiling"],
        ["mechanical", "floor", "prop"],
        _proj("uv", False, True, "none", True, "Cylindrical UVs on handles; planar UVs on tread plates."),
        _wx("aged", 0.40, 0.30, 0.05, 0.0, 0.15, 0.0, 0.10, "Grips polish on crests and blacken with oil in valleys."),
        ["concrete", "rubber_mat", "brushed_metal", "molded_plastic"],
        "CNC knurling, flutes and raised-bar tread plate with crisp machined relief."),
    # ---------------- fabrics & upholstery ----------------
    "woven_textile": _sub(_FU, "Woven Textile", "fabric", 0.86, 6.0, 0.0008, 1.46,
        ["sofa", "armchair", "curtain", "cushion", "bedspread", "clothing", "tablecloth", "tote_bag"],
        ["floor_tile", "exterior_wall", "road", "roof"],
        ["furniture", "prop"],
        _proj("uv", False, False, "none", False, "Unwrap garments and cushions with warp along V; keep texel density uniform across seams."),
        _wx("lived_in", 0.20, 0.15, 0.0, 0.0, 0.0, 0.0, 0.15, "Fabric pills and fades on arms and seat fronts; dust gathers in folds."),
        ["hardwood_floor", "plush_textile", "leather", "plaster_stucco"],
        "Yarn-level woven cloth with over-under warp and weft relief."),
    "plush_textile": _sub(_FU, "Plush, Knit & Carpet", "fabric", 0.90, 6.0, 0.0015, 1.46,
        ["sofa", "armchair", "headboard", "cushion", "ottoman", "car_seat", "theater_seat", "rug"],
        ["exterior_wall", "road", "roof", "kitchen_countertop"],
        ["furniture", "prop", "floor"],
        _proj("uv", False, False, "none", False, "UV-unwrap upholstery; use a sheen lobe (MeshPhysicalMaterial.sheen) for pile fabrics."),
        _wx("lived_in", 0.20, 0.15, 0.0, 0.0, 0.0, 0.0, 0.20, "Pile crushes on seats and collects dust; keep moisture/moss/rust at 0 indoors."),
        ["hardwood_floor", "woven_textile", "leather", "parquet"],
        "Pile, knit, quilted and carpet textiles with soft volumetric relief."),
    "leather": _sub(_FU, "Leather & Suede", "leather", 0.48, 6.0, 0.0008, 1.5,
        ["sofa", "armchair", "car_seat", "handbag", "boots", "office_chair", "belt", "book_cover"],
        ["road", "exterior_wall", "terrain_ground", "roof"],
        ["furniture", "prop"],
        _proj("uv", False, False, "none", False, "UV-unwrap; scale grain to real hide size so pebbles stay 1-3 mm."),
        _wx("lived_in", 0.30, 0.15, 0.0, 0.0, 0.0, 0.0, 0.05, "Leather burnishes and cracks on contact areas."),
        ["hardwood_floor", "woven_textile", "brushed_metal", "parquet"],
        "Full-grain, embossed and woven leathers with crease networks."),
    "natural_fiber": _sub(_FU, "Natural Fiber & Paper", "natural", 0.78, 8.0, 0.002, 1.5,
        ["basket", "chair_seat", "lamp_shade", "room_divider", "placemat", "shoji_screen", "headboard"],
        ["road", "exterior_wall", "kitchen_countertop", "car_body"],
        ["furniture", "prop", "wall"],
        _proj("uv", True, False, "none", False, "UV-unwrap baskets and seats; triplanar is acceptable on chunky woven props."),
        _wx("lived_in", 0.20, 0.20, 0.0, 0.0, 0.0, 0.0, 0.20, "Plant fibers fray, darken with handling and collect dust."),
        ["hardwood_floor", "plaster_stucco", "woven_textile", "terracotta"],
        "Woven rattan, seagrass, cane, cork and handmade papers."),
    # ---------------- nature & terrain ----------------
    "sand": _sub(_NT, "Sand", "natural", 0.88, 8.0, 0.020, 1.5,
        ["desert_dune", "beach", "sandbox", "terrain_ground", "riverbank", "construction_site"],
        ["interior_wall", "sofa", "kitchen_countertop", "window_pane"],
        ["ground"],
        _proj("triplanar", True, True, "none", False, "Triplanar on sculpted dunes; ripples run along U, so orient U across the wind."),
        _wx("fresh", 0.0, 0.10, 0.10, 0.0, 0.0, 0.0, 0.10, "Natural ground: add moisture near water, otherwise keep weathering near zero."),
        ["rock_cliff", "gravel", "ground_cover", "wood_slats"],
        "Wind-rippled granular sand with asymmetric crests."),
    "mud": _sub(_NT, "Mud & Cracked Earth", "natural", 0.86, 11.0, 0.020, 1.5,
        ["dry_lakebed", "desert_floor", "drought_field", "canyon_floor", "battlefield_ground"],
        ["interior_wall", "sofa", "kitchen_countertop", "window_pane"],
        ["ground"],
        _proj("triplanar", True, True, "none", False, "Triplanar on terrain; plates need world-scale tiling to keep crack spacing physical."),
        _wx("dusty", 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.40, "Dry crusts collect windblown dust; add moisture to turn them into fresh mud."),
        ["rock_cliff", "sand", "ground_cover", "gravel"],
        "Desiccation-cracked clay plates with curled edges and deep fissures."),
    "gravel": _sub(_NT, "Gravel & Aggregate", "stone", 0.82, 13.0, 0.025, 1.5,
        ["driveway", "garden_path", "railway_ballast", "parking_area", "rooftop_ballast", "zen_garden"],
        ["interior_wall", "sofa", "kitchen_countertop", "ceiling"],
        ["ground", "road"],
        _proj("triplanar", True, True, "none", False, "Triplanar or top-down planar; use the height map for parallax/displacement."),
        _wx("dusty", 0.0, 0.20, 0.10, 0.05, 0.0, 0.0, 0.25, "Loose stone fills with fines and dust."),
        ["paver", "ground_cover", "rock_cliff", "wood_slats"],
        "Loose rounded and crushed stone aggregate."),
    "ice": _sub(_NT, "Ice", "glass", 0.12, 5.0, 0.010, 1.31,
        ["frozen_lake", "glacier", "ice_rink", "frozen_waterfall", "icicle_prop", "arctic_floor"],
        ["interior_wall", "sofa", "desert_ground", "kitchen_countertop"],
        ["ground", "hero_asset"],
        _proj("triplanar", True, True, "none", False, "Triplanar on glaciers; combine with transmission for clear ice."),
        _wx("frozen", 0.0, 0.05, 0.10, 0.0, 0.0, 0.60, 0.0, "Rime and frost bloom over ice; frost maps onto the hypbr dust channel."),
        ["ground_cover", "rock_cliff", "glazing", "sand"],
        "Fractured lake and glacier ice with frost blooms.", transmission=0.35),
    "ground_cover": _sub(_NT, "Soil, Grass, Moss & Snow", "natural", 0.88, 10.0, 0.020, 1.5,
        ["terrain_ground", "forest_floor", "meadow", "garden_bed", "park_lawn", "hillside", "trail"],
        ["interior_wall", "sofa", "kitchen_countertop", "window_pane"],
        ["ground"],
        _proj("triplanar", True, True, "none", False, "Triplanar on terrain; blend several ground covers with the height-lerp blend using aux.R height."),
        _wx("fresh", 0.0, 0.10, 0.20, 0.0, 0.0, 0.0, 0.0, "Organic ground is already 'weathered'; add moisture after rain or frost in winter."),
        ["rock_cliff", "gravel", "bark", "cobblestone"],
        "Organic ground cover clumps with soil, leaf, moss and snow variation."),
    "rock_cliff": _sub(_NT, "Rock & Cliff", "stone", 0.80, 14.0, 0.080, 1.5,
        ["cliff_face", "canyon_wall", "mountain", "boulder", "cave_wall", "quarry"],
        ["interior_wall", "sofa", "kitchen_countertop", "window_pane"],
        ["ground", "wall", "hero_asset"],
        _proj("triplanar", True, True, "none", False, "Triplanar is mandatory on sculpted rock; strata run along U, keep them roughly horizontal."),
        _wx("mossy", 0.15, 0.25, 0.20, 0.30, 0.0, 0.0, 0.10, "Exposed rock carries moss on ledges and dust in fractures."),
        ["ground_cover", "gravel", "sand", "bark"],
        "Stratified and fractured bedrock with ledges and crack networks."),
    "thatch": _sub(_NT, "Thatch", "natural", 0.86, 10.0, 0.030, 1.5,
        ["cottage_roof", "hut_roof", "tiki_bar_canopy", "gazebo_roof", "rural_shelter"],
        ["floor", "kitchen_countertop", "sofa", "window_pane"],
        ["roof"],
        _proj("uv", True, True, "v", True, "Stalks run along V; keep the image top toward the ridge so the stepped tiers overlap downhill."),
        _wx("mossy", 0.20, 0.30, 0.30, 0.40, 0.0, 0.0, 0.0, "Old thatch greys, holds moisture and grows moss on north slopes."),
        ["wood_cladding", "plaster_stucco", "stacked_stone", "ground_cover"],
        "Layered reed and palm thatch with tiered stalk bundles."),
    # ---------------- glass & optics ----------------
    "fluted_glass": _sub(_GO, "Fluted & Reeded Glass", "glass", 0.06, 8.0, 0.0, 1.52,
        ["shower_screen", "cabinet_door", "room_divider", "office_partition", "window", "pendant_lamp"],
        ["floor", "road", "terrain_ground", "sofa"],
        ["glass_window", "furniture"],
        _proj("uv", False, True, "v", True, "Flutes run along V; keep them vertical. Use MeshPhysicalMaterial.transmission with the albedo as attenuation tint."),
        _wx("light_use", 0.05, 0.10, 0.10, 0.0, 0.0, 0.0, 0.05, "Glass picks up water spots and dust in flute valleys."),
        ["brushed_metal", "hardwood_floor", "marble", "wood_slats"],
        "Optically ribbed privacy glass with refractive flutes.", transmission=0.9),
    "glass_block": _sub(_GO, "Glass Block", "glass", 0.10, 8.0, 0.002, 1.52,
        ["bathroom_window_wall", "stairwell_wall", "partition_wall", "basement_window", "art_deco_facade"],
        ["floor", "road", "sofa", "terrain_ground"],
        ["glass_window", "wall"],
        _proj("uv", False, True, "none", False, "Planar UVs aligned to the block grid; mortar joints are opaque (aux cavity)."),
        _wx("aged", 0.10, 0.30, 0.10, 0.05, 0.0, 0.0, 0.10, "Mortar joints darken while glass faces stay clear."),
        ["brick", "subway_tile", "concrete", "porcelain"],
        "Pressed hollow glass blocks with wavy optical faces and mortar joints.", transmission=0.75),
    "textured_glass": _sub(_GO, "Pebbled & Cathedral Glass", "glass", 0.08, 7.0, 0.0, 1.52,
        ["front_door_sidelight", "stained_glass_window", "cabinet_door", "bathroom_window", "transom", "lamp_shade"],
        ["floor", "road", "sofa", "terrain_ground"],
        ["glass_window"],
        _proj("uv", False, True, "none", False, "Planar UVs; tint the transmission with the albedo map."),
        _wx("light_use", 0.05, 0.10, 0.05, 0.0, 0.0, 0.0, 0.05, "Keep glass weathering minimal; dust settles in hollows."),
        ["wood_cladding", "brushed_metal", "plaster_stucco", "brick"],
        "Rolled obscure, hammered and cathedral art glass.", transmission=0.85),
    "frosted_glass": _sub(_GO, "Frosted & Etched Glass", "glass", 0.40, 3.0, 0.0, 1.52,
        ["bathroom_window", "office_partition", "light_diffuser", "shower_door", "door_panel", "lamp_shade"],
        ["floor", "road", "sofa", "terrain_ground"],
        ["glass_window", "ceiling"],
        _proj("uv", True, True, "none", False, "Frosting is isotropic; triplanar is fine. Pair high roughness with transmission for diffusion."),
        _wx("light_use", 0.05, 0.10, 0.05, 0.0, 0.0, 0.0, 0.05, "Frosted glass hides dirt; keep weathering subtle."),
        ["brushed_metal", "porcelain", "subway_tile", "concrete"],
        "Acid-etched, sandblasted and opal glass with diffusing micro-relief.", transmission=0.8),
    "wired_glass": _sub(_GO, "Wired Safety Glass", "glass", 0.10, 4.0, 0.0, 1.52,
        ["fire_door_window", "school_window", "skylight", "industrial_window", "stairwell_door"],
        ["floor", "road", "sofa", "terrain_ground"],
        ["glass_window"],
        _proj("uv", False, True, "none", False, "Keep the wire grid axis-aligned with the frame."),
        _wx("aged", 0.10, 0.30, 0.10, 0.0, 0.10, 0.0, 0.15, "Industrial glazing collects grime; embedded wire can show rust bloom at edges."),
        ["concrete_block", "metal_cladding", "brushed_metal", "concrete"],
        "Georgian wired safety glass with embedded steel mesh.", transmission=0.85),
    "glazing": _sub(_GO, "Architectural Glazing", "glass", 0.04, 0.4, 0.0, 1.52,
        ["window_pane", "curtain_wall", "storefront", "skyscraper_facade", "greenhouse", "car_window"],
        ["floor", "road", "sofa", "terrain_ground"],
        ["glass_window", "wall"],
        _proj("uv", True, True, "none", False, "Near-flat float glass: tiling is irrelevant; drive reflections from the environment and tint via transmission color."),
        _wx("light_use", 0.0, 0.10, 0.15, 0.0, 0.0, 0.0, 0.05, "Exterior glass shows rain spotting and dust; interior glass stays clean."),
        ["metal_cladding", "concrete", "brushed_metal", "brick"],
        "Smooth float and tinted architectural glazing.", transmission=0.95),
    # ---------------- rubber, polymers & roads ----------------
    "asphalt": _sub(_RP, "Asphalt & Road", "asphalt", 0.86, 9.0, 0.006, 1.5,
        ["road", "highway", "parking_lot", "driveway", "runway", "racetrack"],
        ["interior_wall", "sofa", "kitchen_countertop", "window_pane"],
        ["road", "ground"],
        _proj("uv", True, True, "none", False, "Planar UVs along the road spline (U along travel); world-scale tiling keeps aggregate size physical."),
        _wx("rain", 0.20, 0.40, 0.40, 0.0, 0.0, 0.0, 0.20, "Roads polish in wheel paths, gather oil and grit, and puddle after rain."),
        ["paver", "concrete", "gravel", "ground_cover"],
        "Bituminous road surfacing with sharp mineral aggregate."),
    "tire_tread": _sub(_RP, "Tire Tread", "rubber", 0.80, 11.0, 0.006, 1.52,
        ["car_tire", "truck_tire", "wheel", "rubber_track", "bicycle_tire"],
        ["interior_wall", "floor_tile_interior", "kitchen_countertop", "window_pane"],
        ["mechanical", "prop"],
        _proj("uv", False, False, "none", True, "Map V around the tire circumference and U across the tread width."),
        _wx("dusty", 0.30, 0.40, 0.10, 0.0, 0.0, 0.0, 0.40, "Tires gather road dust and mud in grooves; crests polish."),
        ["asphalt", "machined_metal", "molded_plastic", "gravel"],
        "Molded rubber tread blocks with deep sipes and grooves."),
    "composite_weave": _sub(_RP, "Carbon & Composite Weave", "carbon", 0.16, 4.0, 0.0005, 1.5,
        ["car_body_panel", "bike_frame", "drone_arm", "helmet", "phone_case", "racing_seat_shell"],
        ["terrain_ground", "road", "sofa", "roof"],
        ["prop", "hero_asset", "mechanical"],
        _proj("uv", True, True, "none", True, "UV-unwrap with tows at 0/90 degrees; add a clearcoat layer for glossy resin."),
        _wx("light_use", 0.10, 0.10, 0.0, 0.0, 0.0, 0.0, 0.05, "Clearcoat scuffs lightly; never rusts."),
        ["brushed_metal", "molded_plastic", "leather", "machined_metal"],
        "Woven and forged fiber composites under glossy resin."),
    "foam": _sub(_RP, "Foam", "rubber", 0.92, 6.0, 0.002, 1.5,
        ["mattress", "packaging_insert", "seat_cushion", "yoga_mat", "gym_floor", "insulation_panel"],
        ["road", "exterior_wall", "roof", "kitchen_countertop"],
        ["prop", "furniture", "floor"],
        _proj("triplanar", True, False, "none", False, "Isotropic cells: triplanar works on cut foam blocks."),
        _wx("lived_in", 0.15, 0.20, 0.0, 0.0, 0.0, 0.0, 0.25, "Foam yellows and gathers dust; keep it indoors."),
        ["woven_textile", "molded_plastic", "rubber_mat", "concrete"],
        "Open- and closed-cell polymer foams."),
    "molded_plastic": _sub(_RP, "Molded Plastic & Silicone", "rubber", 0.45, 3.0, 0.0005, 1.46,
        ["appliance_housing", "toy", "phone_case", "chair_shell", "storage_bin", "dashboard", "electronics_enclosure"],
        ["road", "terrain_ground", "roof", "exterior_wall"],
        ["prop", "furniture", "mechanical"],
        _proj("triplanar", True, False, "none", False, "Mold-tech stipple is isotropic; triplanar avoids UV seams on product shells."),
        _wx("light_use", 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.10, "Plastic scuffs and shines on contact points."),
        ["brushed_metal", "composite_weave", "machined_metal", "foam"],
        "Injection-molded polymers with mold-tech stipple texture."),
    "rubber_mat": _sub(_RP, "Rubber Flooring & Mats", "rubber", 0.72, 9.0, 0.004, 1.52,
        ["gym_floor", "entrance_mat", "garage_floor", "truck_bed_liner", "playground_surface", "stair_tread"],
        ["roof", "kitchen_countertop", "window_pane", "sofa"],
        ["floor", "mechanical"],
        _proj("uv", False, True, "none", False, "Planar UVs aligned to the room; studs and ribs should stay axis-aligned."),
        _wx("dusty", 0.25, 0.35, 0.05, 0.0, 0.0, 0.0, 0.30, "Rubber floors trap grit and chalk dust around studs."),
        ["concrete", "machined_metal", "brushed_metal", "concrete_block"],
        "Resilient rubber flooring with raised studs, ribs and granules."),
    # ---------------- roofing, tech & sci-fi ----------------
    "roofing": _sub(_RT, "Roofing", "stone", 0.74, 11.0, 0.020, 1.5,
        ["house_roof", "cottage_roof", "villa_roof", "church_roof", "gazebo_roof", "dormer"],
        ["floor", "sofa", "kitchen_countertop", "interior_wall"],
        ["roof"],
        _proj("uv", False, True, "none", False, "Keep the texture upright on the roof plane: image top toward the ridge, image bottom toward the eave, so course overlaps shade correctly."),
        _wx("mossy", 0.25, 0.35, 0.25, 0.30, 0.0, 0.0, 0.10, "Roofs grow moss on shaded slopes and streak below ridges."),
        ["brick", "wood_cladding", "plaster_stucco", "metal_cladding"],
        "Overlapping roof coverings with shadowed laps."),
    "pcb": _sub(_RT, "Printed Circuit Board", "ceramic", 0.22, 6.0, 0.0003, 1.5,
        ["circuit_board", "electronics_prop", "gadget_interior", "robot_internals", "server_motherboard"],
        ["floor", "road", "sofa", "terrain_ground"],
        ["prop", "mechanical", "hero_asset"],
        _proj("uv", False, False, "none", False, "Planar UVs on flat boards at true 0.1 m scale."),
        _wx("dusty", 0.05, 0.15, 0.0, 0.0, 0.05, 0.0, 0.30, "Electronics gather dust; oxidize pads only for abandoned props."),
        ["molded_plastic", "heatsink", "brushed_metal", "scifi_panel"],
        "Solder-masked FR4 boards with copper traces, vias and gold pads."),
    "solar": _sub(_RT, "Solar Array", "ceramic", 0.14, 5.0, 0.0005, 1.5,
        ["solar_panel", "solar_farm_array", "rooftop_solar", "satellite_wing", "solar_streetlight"],
        ["floor", "sofa", "terrain_ground", "kitchen_countertop"],
        ["roof", "mechanical", "prop"],
        _proj("uv", False, True, "none", False, "Planar UVs, one tile per 0.62 m cell block; cover with glossy glass coat."),
        _wx("dusty", 0.05, 0.20, 0.10, 0.0, 0.0, 0.0, 0.35, "Panels collect dust and bird grime; clean panels are glossy."),
        ["metal_cladding", "roofing", "brushed_metal", "concrete"],
        "Photovoltaic cells with silver busbars and white backsheet gaps."),
    "scifi_panel": _sub(_RT, "Sci-Fi Hull Panel", "metal", 0.34, 8.0, 0.004, 1.5,
        ["spaceship_hull", "scifi_corridor_wall", "mech_armor", "space_station_interior", "robot_body", "scifi_door"],
        ["terrain_ground", "sofa", "tree_trunk", "carpet"],
        ["wall", "floor", "hero_asset", "mechanical"],
        _proj("uv", True, True, "none", False, "Box or UV map at hull scale; trim sheets read best with panels aligned to edges."),
        _wx("aged", 0.30, 0.30, 0.0, 0.0, 0.05, 0.0, 0.15, "Battle-worn hulls chip on edges and carry carbon scoring in seams."),
        ["brushed_metal", "machined_metal", "pcb", "glazing"],
        "Modular armor and hull plates with bevel seams and inset panels."),
    "heatsink": _sub(_RT, "Heatsink & Fins", "metal", 0.30, 10.0, 0.003, 1.5,
        ["cpu_heatsink", "amplifier_chassis", "led_light_body", "power_supply", "server_rack"],
        ["terrain_ground", "sofa", "floor_tile_interior", "roof"],
        ["mechanical", "prop"],
        _proj("uv", False, False, "v", True, "Fins run along V; align them with the extrusion direction."),
        _wx("dusty", 0.10, 0.20, 0.0, 0.0, 0.0, 0.0, 0.45, "Fins clog with dust between blades."),
        ["pcb", "molded_plastic", "brushed_metal", "scifi_panel"],
        "Extruded cooling fin arrays."),
    "acoustic_foam": _sub(_RT, "Acoustic Foam", "rubber", 0.95, 9.0, 0.020, 1.5,
        ["recording_studio_wall", "podcast_booth", "home_theater_wall", "anechoic_chamber", "rehearsal_room"],
        ["floor", "road", "exterior_wall", "kitchen_countertop"],
        ["wall", "ceiling"],
        _proj("uv", False, True, "none", False, "Planar UVs at the 0.5 m panel scale."),
        _wx("dusty", 0.05, 0.10, 0.0, 0.0, 0.0, 0.0, 0.30, "Foam fades and gathers dust on peaks."),
        ["hardwood_floor", "plush_textile", "wood_slats", "concrete"],
        "Absorbent open-cell acoustic treatment panels."),
}


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------
def C(c1, c2=None, c3=None, **kw) -> dict[str, Any]:
    """Pack a parametric parameter dict with linear-RGB colors c1/c2/c3."""
    p: dict[str, Any] = {"c1": tuple(c1)}
    if c2 is not None:
        p["c2"] = tuple(c2)
    if c3 is not None:
        p["c3"] = tuple(c3)
    p.update(kw)
    return p


def _planks(c1, c2, c3, planks: int = 6, joints: int = 2, cycles: int | None = None, **kw) -> dict[str, Any]:
    """Wood planks with an integer number of growth-ring cycles across the tile (exact wrap)."""
    n = int(cycles if cycles is not None else round(planks * 10.0 * 0.32))
    return C(c1, c2, c3, planks=planks, joints=joints, ring_freq=round(n / (0.32 * planks), 6), **kw)


_ROWS: list[dict[str, Any]] = []


def _R(sub: str, mid: str, family: str, params: dict[str, Any], size: tuple[float, float], phrase: str,
       tags: str = "", objs: tuple[str, ...] = (), kind: str | None = None, name: str | None = None,
       axis: str | None = None, roles: tuple[str, ...] = (), seed: int | None = None) -> None:
    _ROWS.append({"sub": sub, "id": mid, "family": family, "params": params, "size": size, "phrase": phrase,
                  "tags": tags.split(), "objs": list(objs), "kind": kind, "name": name, "axis": axis,
                  "roles": list(roles), "seed": seed})


# ===========================================================================
# 1. ARCHITECTURAL MASONRY (65 parametric)
# ===========================================================================
_MG = (0.32, 0.31, 0.29)  # portland mortar
_B = "running_bond_masonry"
for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("classic_red_brick", (0.34, 0.095, 0.05), (0.44, 0.15, 0.08), _MG, "Kiln-fired red clay bricks in running bond with gray portland mortar", "red classic exterior", {}, (0.9, 0.75)),
    ("london_yellow_stock_brick", (0.52, 0.38, 0.19), (0.40, 0.27, 0.13), (0.36, 0.34, 0.30), "Mottled London yellow stock bricks with soot-dusted faces and lime mortar", "yellow victorian london heritage", {"wobble": 1.6}, (0.9, 0.75)),
    ("chicago_common_brick", (0.40, 0.20, 0.11), (0.26, 0.12, 0.07), (0.34, 0.32, 0.29), "Salvaged Chicago common bricks mixing salmon, brown and charred tones", "salvaged industrial reclaimed brown", {"wobble": 1.8, "face_rough": 0.10}, (0.9, 0.75)),
    ("limewashed_brick", (0.70, 0.68, 0.64), (0.55, 0.42, 0.34), (0.62, 0.61, 0.58), "Chalky limewashed brick with red clay peeking through thin coats", "white farmhouse limewash rustic", {}, (0.9, 0.75)),
    ("painted_white_brick", (0.78, 0.78, 0.76), (0.72, 0.72, 0.70), (0.66, 0.66, 0.64), "Brick painted solid white with paint-filled mortar joints", "white loft painted interior", {"face_rough": 0.05}, (0.9, 0.75)),
    ("charcoal_ironspot_brick", (0.07, 0.06, 0.055), (0.13, 0.11, 0.10), (0.22, 0.22, 0.21), "Dark ironspot face brick with flecked manganese tones and charcoal mortar", "black charcoal modern dark", {}, (0.9, 0.75)),
    ("glazed_white_brick", (0.80, 0.80, 0.78), (0.76, 0.77, 0.76), (0.30, 0.30, 0.29), "Glossy glazed white bricks used to brighten light wells and courtyards", "white glossy glazed lightwell", {"glazed": 1, "rough": 0.30}, (0.9, 0.75)),
    ("glazed_green_brick", (0.05, 0.22, 0.12), (0.08, 0.30, 0.16), (0.30, 0.30, 0.29), "Art deco glazed bottle-green bricks with glossy kiln variation", "green glossy glazed artdeco", {"glazed": 1, "rough": 0.30}, (0.9, 0.75)),
    ("roman_brick_long", (0.36, 0.30, 0.24), (0.46, 0.38, 0.30), (0.40, 0.38, 0.34), "Long thin Roman bricks in tan-gray with raked horizontal joints", "roman prairie tan midcentury", {"nx": 3, "ny": 14}, (0.915, 0.7)),
    ("offset_bond_terracotta_brick", (0.50, 0.20, 0.08), (0.56, 0.26, 0.11), (0.42, 0.40, 0.36), "Orange wire-cut terracotta bricks laid in a quarter offset bond", "orange terracotta offset contemporary", {"stagger": 0.25}, (0.9, 0.75)),
    ("stack_bond_gray_brick", (0.24, 0.23, 0.22), (0.30, 0.29, 0.27), (0.10, 0.10, 0.10), "Gray brick in modern stack bond with dark recessed mortar", "gray modern stackbond minimal", {"stagger": 0.0}, (0.9, 0.75)),
    ("tumbled_reclaimed_brick", (0.30, 0.10, 0.06), (0.42, 0.28, 0.18), (0.38, 0.36, 0.32), "Tumbled antique bricks with rounded arrises and mixed patina", "reclaimed antique tumbled rustic", {"wobble": 2.5, "face_rough": 0.12}, (0.9, 0.75)),
    ("buff_firebrick_refractory", (0.62, 0.50, 0.32), (0.55, 0.42, 0.26), (0.58, 0.54, 0.46), "Buff refractory firebricks with thin fireclay joints for hearths and ovens", "buff fireplace pizza_oven refractory", {"ny": 8, "mortar": 0.035}, (0.94, 0.944)),
    ("sooty_firebox_brick", (0.12, 0.07, 0.05), (0.30, 0.20, 0.12), (0.10, 0.09, 0.08), "Soot-blackened firebox bricks scorched by years of open flame", "soot black fireplace scorched", {"ny": 8, "mortar": 0.035}, (0.94, 0.944)),
    ("adobe_mud_brick", (0.36, 0.24, 0.14), (0.44, 0.30, 0.18), (0.32, 0.22, 0.13), "Sun-dried adobe mud bricks with wide earthen mortar beds", "adobe southwestern earth pueblo", {"nx": 3, "ny": 8, "wobble": 2.0, "mortar": 0.10}, (1.08, 0.88)),
    ("dutch_clinker_brick", (0.16, 0.07, 0.04), (0.24, 0.10, 0.05), (0.26, 0.25, 0.24), "Hard-fired Dutch clinker bricks in deep purple-brown with flush joints", "dutch clinker purple brown", {"mortar": 0.05}, (0.9, 0.75)),
    ("brown_norman_brick", (0.22, 0.12, 0.07), (0.30, 0.17, 0.10), (0.30, 0.29, 0.27), "Elongated brown Norman bricks for low horizontal facades", "brown norman midcentury", {"nx": 3}, (0.915, 0.8)),
    ("gray_thin_brick_veneer", (0.30, 0.29, 0.28), (0.38, 0.36, 0.34), (0.55, 0.54, 0.50), "Thin gray brick veneer with light pointing for loft interiors", "gray loft veneer interior", {}, (0.9, 0.75)),
    ("pink_handmade_brick", (0.52, 0.26, 0.20), (0.60, 0.34, 0.27), (0.56, 0.52, 0.46), "Soft pink handmade colonial bricks with uneven molded edges", "pink colonial handmade georgian", {"wobble": 1.8}, (0.9, 0.75)),
    ("black_glazed_brick", (0.03, 0.03, 0.035), (0.05, 0.05, 0.055), (0.42, 0.42, 0.40), "Glossy black glazed bricks with pale contrasting mortar", "black glossy glazed station", {"glazed": 1, "rough": 0.28}, (0.9, 0.75)),
    ("cream_city_brick", (0.62, 0.52, 0.36), (0.68, 0.58, 0.42), (0.70, 0.68, 0.64), "Pale cream-city bricks with white lime pointing", "cream beige heritage milwaukee", {}, (0.9, 0.75)),
    ("moss_stained_garden_brick", (0.28, 0.11, 0.06), (0.18, 0.16, 0.08), (0.24, 0.24, 0.20), "Old garden-wall bricks blotched with green algae stains", "garden old green stained", {"wobble": 1.4}, (0.9, 0.75)),
    ("sandblasted_red_brick", (0.38, 0.13, 0.08), (0.46, 0.19, 0.12), (0.40, 0.38, 0.35), "Sandblasted red brick with an open, gritty exposed-clay face", "red sandblasted loft gritty", {"face_rough": 0.14}, (0.9, 0.75)),
    ("speckled_ivory_brick", (0.66, 0.60, 0.50), (0.56, 0.50, 0.40), (0.48, 0.46, 0.42), "Ivory face brick with fine dark speckles from manganese firing", "ivory speckled contemporary", {}, (0.9, 0.75)),
]:
    _R("brick", _mid, _B, C(_c1, _c2, _c3, **_kw), _sz, _ph, _tg + " brick")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("gray_cmu_block", (0.30, 0.29, 0.27), (0.34, 0.33, 0.31), (0.26, 0.25, 0.24), "Standard gray concrete masonry units with tooled mortar joints", "gray cmu cinder industrial", {"face_rough": 0.14}, (0.8, 0.8)),
    ("split_face_block", (0.42, 0.36, 0.27), (0.48, 0.42, 0.32), (0.38, 0.35, 0.30), "Tan split-face concrete block with rough fractured faces", "tan splitface commercial", {"face_rough": 0.22, "wobble": 1.5}, (0.8, 0.8)),
    ("charcoal_cmu_block", (0.10, 0.10, 0.10), (0.14, 0.14, 0.13), (0.18, 0.18, 0.17), "Integrally colored charcoal block with matching dark mortar", "charcoal black modern", {"face_rough": 0.12}, (0.8, 0.8)),
    ("painted_stack_block", (0.62, 0.62, 0.60), (0.66, 0.66, 0.64), (0.55, 0.55, 0.53), "Painted white concrete block in stack bond for gyms and corridors", "white painted stackbond institutional", {"stagger": 0.0, "face_rough": 0.08}, (0.8, 0.8)),
    ("glazed_cmu_block", (0.55, 0.62, 0.66), (0.50, 0.57, 0.62), (0.42, 0.42, 0.40), "Pale blue glazed-face concrete block for hygienic institutional walls", "blue glazed school hospital", {"glazed": 1, "rough": 0.30}, (0.8, 0.8)),
    ("rough_cinder_block", (0.22, 0.21, 0.20), (0.28, 0.27, 0.25), (0.24, 0.23, 0.22), "Porous cinder block with open aggregate faces", "cinder rough porous utility", {"face_rough": 0.20}, (0.8, 0.8)),
    ("large_format_ground_block", (0.40, 0.39, 0.37), (0.46, 0.45, 0.43), (0.30, 0.29, 0.28), "Ground-face large-format architectural block with polished aggregate", "polished groundface architectural", {"ny": 6, "face_rough": 0.05}, (1.2, 1.2)),
]:
    _R("concrete_block", _mid, _B, C(_c1, _c2, _c3, nx=2, mortar=0.05, **{"ny": 4, **_kw}), _sz, _ph, _tg + " block concrete")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("limestone_ashlar", (0.60, 0.55, 0.45), (0.66, 0.61, 0.50), (0.56, 0.53, 0.47), "Smooth-dressed limestone ashlar with fine lime joints", "limestone cream classical", {}, (1.24, 1.24)),
    ("red_sandstone_ashlar", (0.42, 0.18, 0.10), (0.48, 0.24, 0.14), (0.44, 0.36, 0.30), "Red sandstone ashlar blocks with tooled faces", "red sandstone victorian", {}, (1.24, 1.24)),
    ("gray_granite_ashlar", (0.30, 0.30, 0.30), (0.38, 0.37, 0.36), (0.34, 0.33, 0.31), "Hard gray granite ashlar for plinths and bridges", "gray granite civic", {"face_rough": 0.14}, (1.24, 1.24)),
    ("honey_bath_stone_ashlar", (0.62, 0.48, 0.28), (0.68, 0.54, 0.33), (0.60, 0.52, 0.40), "Warm honey oolitic bath stone in Georgian ashlar", "honey georgian bath", {}, (1.24, 1.24)),
    ("castle_rubble_ashlar", (0.26, 0.23, 0.20), (0.36, 0.33, 0.29), (0.30, 0.28, 0.25), "Roughly squared medieval castle stones with irregular coursing", "medieval castle rubble fantasy", {"nx": 3, "ny": 6, "wobble": 3.0, "face_rough": 0.18}, (1.5, 1.2)),
    ("coral_keystone_ashlar", (0.66, 0.60, 0.48), (0.72, 0.66, 0.54), (0.62, 0.58, 0.50), "Porous coral keystone blocks with fossil pits", "coral tropical keystone", {"face_rough": 0.20}, (1.24, 1.24)),
    ("basalt_ashlar_block", (0.06, 0.06, 0.065), (0.10, 0.10, 0.105), (0.16, 0.16, 0.15), "Dark volcanic basalt ashlar with crisp chiselled joints", "basalt black volcanic", {}, (1.24, 1.24)),
]:
    _R("ashlar", _mid, _B, C(_c1, _c2, _c3, mortar=0.04, **{"nx": 2, "ny": 4, "wobble": 1.5, "face_rough": 0.10, "tooled": 1.0, "chips": 0.35, **_kw}), _sz, _ph, _tg + " ashlar stone")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw in [
    ("charcoal_slate_ledgestone", (0.08, 0.085, 0.09), (0.16, 0.16, 0.17), (0.03, 0.03, 0.03), "Split charcoal slate ledgers stacked dry with deep shadow lines", "slate charcoal dark", {}),
    ("golden_quartzite_ledgestone", (0.55, 0.42, 0.22), (0.40, 0.36, 0.30), (0.12, 0.10, 0.08), "Golden quartzite ledgestone with rust and silver flashes", "gold quartzite warm", {}),
    ("limestone_drystack", (0.58, 0.54, 0.46), (0.48, 0.44, 0.37), (0.16, 0.15, 0.13), "Pale limestone dry-stack wall stones", "limestone pale cream", {}),
    ("fieldstone_drystack_wall", (0.30, 0.27, 0.22), (0.42, 0.36, 0.28), (0.08, 0.07, 0.06), "Large irregular fieldstones in a farm boundary wall", "fieldstone farm rustic", {"nx": 2, "ny": 8}),
    ("desert_sandstone_ledger", (0.52, 0.30, 0.16), (0.62, 0.42, 0.24), (0.14, 0.08, 0.05), "Banded desert sandstone ledgers in warm ochres", "sandstone desert ochre", {}),
    ("gray_granite_drystack", (0.26, 0.26, 0.26), (0.40, 0.39, 0.38), (0.06, 0.06, 0.06), "Gray granite drystack with crisp split faces", "granite gray", {}),
    ("rustic_brown_ledgestone", (0.22, 0.15, 0.09), (0.34, 0.24, 0.15), (0.06, 0.04, 0.03), "Earthy brown ledgestone for mountain lodges", "brown lodge mountain", {}),
    ("white_marble_ledger", (0.74, 0.73, 0.71), (0.62, 0.62, 0.61), (0.20, 0.20, 0.20), "Thin white marble ledger strips for bright feature walls", "white marble bright", {}),
    ("charcoal_strip_ledger", (0.06, 0.06, 0.06), (0.12, 0.115, 0.11), (0.02, 0.02, 0.02), "Narrow strip ledger panels in near-black basalt", "black strip modern", {"nx": 4, "ny": 20}),
]:
    _R("stacked_stone", _mid, "stacked_stone", C(_c1, _c2, _c3, **{"nx": 3, "ny": 14, **_kw}), (1.2, 1.2), _ph, _tg + " ledgestone stone")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("granite_setts_cobble", (0.22, 0.22, 0.22), (0.34, 0.33, 0.32), (0.12, 0.11, 0.10), "Squared granite setts with flattened, traffic-polished crowns", "granite setts street european", {"cells": 10, "roundness": 0.30, "gap": 0.10}, (1.5, 1.5)),
    ("river_pebble_cobble", (0.30, 0.27, 0.22), (0.50, 0.46, 0.40), (0.20, 0.18, 0.15), "Rounded river pebbles set in mortar for paths and drainage", "pebble river round", {"cells": 16, "roundness": 0.95, "gap": 0.16}, (0.8, 0.8)),
    ("old_town_cobblestone", (0.20, 0.18, 0.16), (0.30, 0.27, 0.24), (0.08, 0.075, 0.06), "Worn domed cobblestones of an old-town street", "oldtown medieval street", {"cells": 12, "roundness": 0.75}, (1.6, 1.6)),
    ("bluestone_flagstone", (0.14, 0.17, 0.20), (0.22, 0.24, 0.26), (0.20, 0.19, 0.17), "Irregular bluestone flagstones with sand joints", "bluestone flagstone patio", {"cells": 6, "roundness": 0.15, "gap": 0.08}, (2.4, 2.4)),
    ("sandstone_flagstone_patio", (0.55, 0.42, 0.28), (0.62, 0.50, 0.34), (0.30, 0.28, 0.24), "Warm sandstone crazy-paving flagstones", "sandstone flagstone warm", {"cells": 6, "roundness": 0.20, "gap": 0.08}, (2.4, 2.4)),
    ("grass_joint_flagstone", (0.40, 0.38, 0.34), (0.50, 0.47, 0.42), (0.07, 0.13, 0.03), "Stepping-stone flagstones with grass growing in wide joints", "flagstone grass garden", {"cells": 7, "gap": 0.22, "roundness": 0.25}, (2.4, 2.4)),
    ("basalt_lava_cobble", (0.05, 0.05, 0.05), (0.10, 0.095, 0.09), (0.18, 0.17, 0.15), "Black basalt lava cobbles with pale grout", "basalt black volcanic", {"cells": 11}, (1.4, 1.4)),
    ("belgian_block_pavers", (0.30, 0.28, 0.25), (0.38, 0.35, 0.30), (0.16, 0.15, 0.13), "Rectangular Belgian block pavers with chipped edges", "belgian block driveway", {"cells": 9, "roundness": 0.25}, (1.6, 1.6)),
    ("mossy_cobblestone_path", (0.22, 0.21, 0.18), (0.30, 0.29, 0.25), (0.05, 0.10, 0.02), "Damp cobblestone path with moss-filled joints", "moss garden damp forest", {"cells": 12, "roundness": 0.80}, (1.6, 1.6)),
    ("pink_granite_cobble", (0.40, 0.25, 0.20), (0.50, 0.34, 0.28), (0.20, 0.18, 0.16), "Pink granite cobbles with coarse feldspar color", "pink granite plaza", {"cells": 10}, (1.5, 1.5)),
]:
    _R("cobblestone", _mid, "cobblestone_paver", C(_c1, _c2, _c3, **_kw), _sz, _ph, _tg + " cobblestone paving")

for _mid, _fam, _p, _sz, _ph, _tg in [
    ("herringbone_clay_paver", "herringbone_chevron", C((0.36, 0.12, 0.06), (0.46, 0.20, 0.10), (0.30, 0.28, 0.24), freq=4, is_wood=False, grout=0.06, rough=0.78, slope=9.0), (1.6, 1.6), "Red clay pavers laid in 45-degree herringbone", "red clay herringbone driveway"),
    ("herringbone_charcoal_paver", "herringbone_chevron", C((0.10, 0.10, 0.10), (0.16, 0.155, 0.15), (0.28, 0.27, 0.25), freq=4, is_wood=False, grout=0.06, rough=0.80, slope=9.0), (1.6, 1.6), "Charcoal concrete pavers in herringbone", "charcoal concrete herringbone modern"),
    ("chevron_brick_paver", "herringbone_chevron", C((0.40, 0.16, 0.08), (0.30, 0.12, 0.06), (0.32, 0.30, 0.26), freq=4, is_wood=False, chevron=True, grout=0.06, rough=0.80, slope=9.0), (1.6, 1.6), "Brick pavers in a crisp chevron pattern", "brick chevron courtyard"),
    ("concrete_square_paver", "grid_tile", C((0.36, 0.35, 0.33), (0.42, 0.41, 0.39), (0.20, 0.19, 0.18), nx=3, ny=3, grout=0.04, bevel=0.10, rough=0.82, slope=8.0, grout_rmod=0.25), (1.8, 1.8), "Square precast concrete slabs with chamfered edges", "concrete slab sidewalk"),
    ("permeable_running_paver", "grid_tile", C((0.30, 0.27, 0.24), (0.36, 0.32, 0.28), (0.18, 0.17, 0.15), nx=2, ny=6, stagger=0.5, grout=0.09, bevel=0.08, rough=0.84, slope=9.0, grout_rmod=0.25), (0.9, 0.9), "Permeable pavers with wide gravel-filled joints", "permeable driveway eco"),
    ("tumbled_holland_paver", "grid_tile", C((0.30, 0.13, 0.07), (0.40, 0.22, 0.13), (0.24, 0.22, 0.19), nx=4, ny=8, stagger=0.5, grout=0.06, handmade=0.6, rough=0.80, slope=9.0, grout_rmod=0.25), (0.8, 0.8), "Tumbled Holland clay pavers in running bond", "clay tumbled rustic path"),
    ("granite_slab_paver", "grid_tile", C((0.28, 0.28, 0.28), (0.34, 0.33, 0.33), (0.18, 0.18, 0.17), nx=2, ny=2, grout=0.025, bevel=0.04, rough=0.68, slope=6.0, grout_rmod=0.25), (1.8, 1.8), "Large flamed granite paving slabs", "granite slab plaza civic"),
    ("travertine_pool_paver", "grid_tile", C((0.62, 0.52, 0.38), (0.68, 0.58, 0.44), (0.55, 0.50, 0.42), nx=2, ny=4, stagger=0.5, grout=0.03, bevel=0.06, handmade=0.3, rough=0.62, slope=6.0, grout_rmod=0.25), (1.2, 1.2), "Tumbled travertine pool-deck pavers", "travertine pool deck mediterranean"),
]:
    _R("paver", _mid, _fam, _p, _sz, _ph, _tg + " paver")

# ===========================================================================
# 2. WOOD & TIMBER (54 parametric)
# ===========================================================================
_DK = (0.10, 0.06, 0.03)
for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("white_oak_plank_floor", (0.50, 0.33, 0.17), (0.38, 0.23, 0.11), _DK, "Quarter-sawn white oak boards with straight grain and fine pores", "oak white_oak natural", {}, (1.8, 0.9)),
    ("red_oak_plank_floor", (0.48, 0.24, 0.12), (0.36, 0.16, 0.08), _DK, "Red oak boards with bold cathedral grain and pinkish warmth", "oak red_oak traditional", {}, (1.8, 0.9)),
    ("american_walnut_floor", (0.16, 0.08, 0.04), (0.24, 0.13, 0.07), (0.04, 0.025, 0.015), "Rich chocolate American walnut boards with purple undertones", "walnut dark luxury", {}, (1.8, 0.9)),
    ("hard_maple_floor", (0.66, 0.48, 0.30), (0.58, 0.40, 0.23), (0.30, 0.20, 0.10), "Pale hard maple boards with tight, subtle grain", "maple pale scandinavian", {"knots": 0.05}, (1.8, 0.9)),
    ("cherry_wood_floor", (0.40, 0.15, 0.07), (0.30, 0.10, 0.05), _DK, "Aged American cherry boards in deep reddish-brown", "cherry red traditional", {"knots": 0.15}, (1.8, 0.9)),
    ("ebony_stained_oak_floor", (0.035, 0.028, 0.022), (0.07, 0.055, 0.04), (0.01, 0.01, 0.01), "Oak boards stained near-black with grain still visible", "ebony black stained oak dramatic", {}, (1.8, 0.9)),
    ("barnwood_reclaimed_planks", (0.24, 0.21, 0.18), (0.36, 0.31, 0.26), (0.05, 0.045, 0.04), "Reclaimed barnwood boards weathered silver-gray with nail knots", "barnwood reclaimed rustic gray", {"knots": 0.6, "groove": 0.06}, (1.8, 0.9)),
    ("rustic_pine_plank_floor", (0.56, 0.36, 0.17), (0.46, 0.27, 0.12), (0.12, 0.07, 0.03), "Knotty yellow pine boards with honey patina", "pine knotty rustic cabin", {"knots": 0.7}, (1.8, 0.9)),
    ("wide_plank_hickory_floor", (0.52, 0.36, 0.20), (0.34, 0.20, 0.10), _DK, "Wide hickory planks with dramatic light-dark heartwood contrast", "hickory wide farmhouse", {"planks": 4}, (2.4, 0.8)),
    ("gray_washed_oak_floor", (0.40, 0.37, 0.33), (0.30, 0.27, 0.23), (0.08, 0.07, 0.06), "Whitewashed gray oak boards with a soft matte finish", "oak gray coastal washed", {}, (1.8, 0.9)),
    ("mahogany_plank_floor", (0.25, 0.08, 0.04), (0.33, 0.12, 0.06), (0.06, 0.02, 0.01), "Polished mahogany boards with ribbon figure", "mahogany red luxury", {"knots": 0.05, "rough": 0.32}, (1.8, 0.9)),
    ("scandinavian_ash_floor", (0.64, 0.54, 0.40), (0.56, 0.46, 0.32), (0.26, 0.21, 0.14), "Oiled white ash boards with pale straight grain", "ash scandinavian nordic pale", {"knots": 0.08}, (1.8, 0.9)),
    ("weathered_boardwalk_planks", (0.30, 0.27, 0.22), (0.40, 0.36, 0.30), (0.03, 0.03, 0.025), "Sun-bleached boardwalk boards with open drainage gaps", "boardwalk pier exterior weathered", {"joints": 1, "groove": 0.10, "rough": 0.74}, (1.2, 0.9)),
    ("gym_maple_strip_floor", (0.70, 0.50, 0.28), (0.62, 0.42, 0.22), (0.36, 0.24, 0.12), "Lacquered maple strip flooring of a basketball court", "maple gym basketball sports glossy", {"planks": 12, "joints": 3, "knots": 0.02, "rough": 0.24}, (2.1, 0.68)),
    ("strand_bamboo_floor", (0.52, 0.34, 0.16), (0.42, 0.26, 0.11), (0.18, 0.11, 0.05), "Strand-woven carbonized bamboo boards with compressed fiber streaks", "bamboo strand eco", {"planks": 8, "knots": 0.0}, (1.8, 0.8)),
    ("smoked_european_oak_floor", (0.22, 0.14, 0.08), (0.30, 0.20, 0.12), (0.05, 0.03, 0.02), "Fumed European oak with smoky brown depth and brushed texture", "oak smoked fumed brushed", {"knots": 0.35}, (2.2, 0.95)),
]:
    _R("hardwood_floor", _mid, "wood_planks", _planks(_c1, _c2, _c3, **{"planks": 6, "joints": 2, "rough": 0.46, "slope": 8.0, **_kw}), _sz, _ph, _tg + " wood floor", axis="u")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("light_oak_herringbone_parquet", (0.58, 0.40, 0.22), (0.46, 0.30, 0.15), (0.18, 0.11, 0.06), "Light oak blocks in classic 45-degree herringbone", "oak light herringbone", {}, (1.2, 1.2)),
    ("walnut_herringbone_parquet", (0.18, 0.09, 0.05), (0.26, 0.14, 0.07), (0.05, 0.03, 0.02), "Walnut herringbone parquet with deep chocolate tones", "walnut dark herringbone", {}, (1.2, 1.2)),
    ("oak_chevron_parquet", (0.52, 0.35, 0.18), (0.42, 0.27, 0.13), (0.16, 0.10, 0.05), "Oak blocks in a Hungarian-point chevron", "oak chevron parisian", {"chevron": True}, (1.2, 1.2)),
    ("smoked_oak_chevron_parquet", (0.20, 0.13, 0.08), (0.28, 0.19, 0.11), (0.05, 0.03, 0.02), "Smoked oak chevron parquet with moody brown tones", "oak smoked chevron moody", {"chevron": True}, (1.2, 1.2)),
    ("teak_herringbone_parquet", (0.36, 0.19, 0.08), (0.44, 0.25, 0.11), (0.10, 0.05, 0.02), "Golden teak herringbone parquet", "teak golden herringbone", {}, (1.2, 1.2)),
    ("fine_mosaic_herringbone_parquet", (0.50, 0.30, 0.14), (0.36, 0.20, 0.09), (0.14, 0.08, 0.04), "Small-block mosaic herringbone parquet", "mosaic small herringbone vintage", {"freq": 6}, (1.0, 1.0)),
    ("white_oiled_chevron_parquet", (0.66, 0.58, 0.46), (0.58, 0.50, 0.38), (0.30, 0.26, 0.20), "White-oiled oak chevron with a pale Nordic finish", "oak white nordic chevron", {"chevron": True, "freq": 6}, (1.0, 1.0)),
    ("cherry_chevron_parquet", (0.42, 0.17, 0.08), (0.34, 0.12, 0.06), (0.10, 0.04, 0.02), "Cherry wood chevron parquet in warm reds", "cherry chevron warm", {"chevron": True}, (1.2, 1.2)),
]:
    _R("parquet", _mid, "herringbone_chevron", C(_c1, _c2, _c3, **{"freq": 4, "is_wood": True, "grout": 0.045, "rough": 0.42, "slope": 8.0, **_kw}), _sz, _ph, _tg + " parquet wood floor")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("cedar_shiplap_siding", (0.46, 0.20, 0.09), (0.38, 0.15, 0.06), (0.08, 0.04, 0.02), "Western red cedar shiplap boards with shadow grooves", "cedar shiplap siding red", {"joints": 1, "groove": 0.06}, (1.8, 0.9)),
    ("white_board_and_batten", (0.72, 0.72, 0.70), (0.68, 0.68, 0.66), (0.40, 0.40, 0.38), "Painted white board-and-batten siding", "white farmhouse painted", {"vertical": True, "joints": 0, "groove": 0.12, "knots": 0.05}, (0.9, 1.8)),
    ("charred_shou_sugi_ban", (0.025, 0.022, 0.02), (0.06, 0.05, 0.045), (0.01, 0.01, 0.01), "Deeply charred yakisugi cedar with alligator char texture", "charred black yakisugi japanese", {"joints": 1, "cycles": 30, "rough": 0.78}, (1.8, 0.9)),
    ("brushed_shou_sugi_ban", (0.06, 0.045, 0.035), (0.20, 0.12, 0.06), (0.01, 0.01, 0.01), "Brushed shou sugi ban exposing warm grain between char", "charred brushed yakisugi japanese", {"joints": 1, "rough": 0.62}, (1.8, 0.9)),
    ("silvered_cedar_cladding", (0.36, 0.35, 0.33), (0.46, 0.44, 0.40), (0.06, 0.06, 0.05), "Weathered silver-gray vertical cedar cladding", "cedar silver weathered coastal", {"vertical": True, "joints": 1}, (0.9, 1.8)),
    ("red_barn_siding", (0.32, 0.04, 0.03), (0.40, 0.07, 0.05), (0.06, 0.01, 0.01), "Faded barn-red painted vertical siding", "red barn painted farm", {"vertical": True, "joints": 1, "knots": 0.3}, (0.9, 1.8)),
    ("larch_vertical_cladding", (0.52, 0.30, 0.14), (0.44, 0.24, 0.10), (0.10, 0.05, 0.02), "Fresh Siberian larch vertical cladding boards", "larch fresh contemporary", {"vertical": True, "joints": 1}, (0.9, 1.8)),
    ("pine_tongue_groove_ceiling", (0.62, 0.42, 0.22), (0.54, 0.35, 0.17), (0.20, 0.12, 0.05), "Knotty pine tongue-and-groove ceiling boards", "pine cabin ceiling knotty", {"planks": 10, "groove": 0.05, "knots": 0.4}, (1.8, 1.0)),
]:
    _R("wood_cladding", _mid, "wood_planks", _planks(_c1, _c2, _c3, **{"planks": 6, "joints": 2, "rough": 0.62, "slope": 9.0, **_kw}), _sz, _ph, _tg + " cladding wood",
       axis=("v" if _kw.get("vertical") else "u"))

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("oak_acoustic_slat_wall", (0.50, 0.33, 0.17), (0.40, 0.25, 0.12), (0.02, 0.02, 0.02), "Oak veneer acoustic slats over black felt", "oak acoustic slat", {}, (0.6, 0.6)),
    ("walnut_slat_panel", (0.16, 0.08, 0.04), (0.23, 0.12, 0.06), (0.03, 0.03, 0.03), "Walnut slat feature panel with dark recesses", "walnut slat feature", {}, (0.6, 0.6)),
    ("horizontal_cedar_fence", (0.44, 0.20, 0.09), (0.36, 0.16, 0.07), (0.04, 0.04, 0.03), "Horizontal cedar fence boards with narrow gaps", "cedar fence horizontal garden", {"vertical": False, "slats": 8, "gap_ratio": 0.12}, (1.2, 1.2)),
    ("teak_louver_screen", (0.36, 0.19, 0.08), (0.28, 0.14, 0.06), (0.01, 0.01, 0.01), "Teak louvered privacy screen with open gaps", "teak louver screen tropical", {"slats": 10, "gap_ratio": 0.35}, (0.8, 0.8)),
    ("white_ash_batten_ceiling", (0.64, 0.54, 0.40), (0.56, 0.46, 0.33), (0.10, 0.10, 0.10), "White ash batten ceiling over dark acoustic fleece", "ash batten ceiling pale", {"vertical": False, "slats": 10, "gap_ratio": 0.30}, (0.8, 0.8)),
    ("bamboo_pole_screen", (0.52, 0.42, 0.20), (0.44, 0.34, 0.14), (0.08, 0.06, 0.03), "Lashed bamboo pole screen", "bamboo screen tropical asian", {"slats": 14, "gap_ratio": 0.08}, (0.7, 0.7)),
    ("white_picket_fence", (0.74, 0.74, 0.72), (0.70, 0.70, 0.68), (0.08, 0.12, 0.05), "Painted white pickets against a green hedge", "white picket fence suburban", {"slats": 8, "gap_ratio": 0.45}, (1.0, 1.0)),
    ("aspen_sauna_slats", (0.66, 0.56, 0.40), (0.60, 0.50, 0.34), (0.12, 0.09, 0.06), "Pale aspen sauna bench and wall slats", "aspen sauna pale nordic", {"vertical": False, "slats": 12, "gap_ratio": 0.18}, (0.8, 0.8)),
]:
    _R("wood_slats", _mid, "wood_slats", C(_c1, _c2, _c3, **{"slats": 12, "gap_ratio": 0.26, "vertical": True, "rough": 0.52, "slope": 9.0, **_kw}), _sz, _ph, _tg + " wood slats",
       axis=("v" if _kw.get("vertical", True) else "u"))

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("furrowed_oak_bark", (0.18, 0.15, 0.12), (0.30, 0.26, 0.21), (0.04, 0.035, 0.03), "Deeply furrowed mature oak bark", "oak bark forest", {"freq": 10}, (0.6, 0.6)),
    ("plated_pine_bark", (0.26, 0.12, 0.06), (0.38, 0.20, 0.10), (0.05, 0.03, 0.02), "Scaly plated pine bark in warm orange-browns", "pine bark plated", {"freq": 8}, (0.6, 0.6)),
    ("fibrous_redwood_bark", (0.30, 0.10, 0.05), (0.42, 0.16, 0.08), (0.06, 0.025, 0.015), "Fibrous rust-red redwood bark ridges", "redwood bark fibrous", {"freq": 16}, (0.8, 0.8)),
    ("white_birch_bark", (0.70, 0.69, 0.65), (0.60, 0.56, 0.50), (0.04, 0.035, 0.03), "Papery white birch bark with dark lenticels", "birch white bark", {"style": "birch"}, (0.5, 0.5)),
    ("cork_oak_bark", (0.36, 0.22, 0.12), (0.48, 0.32, 0.18), (0.10, 0.06, 0.03), "Thick spongy cork oak bark", "cork bark mediterranean", {"freq": 6}, (0.6, 0.6)),
    ("palm_trunk_bark", (0.30, 0.24, 0.16), (0.40, 0.32, 0.22), (0.08, 0.06, 0.04), "Fibrous tropical palm trunk husk", "palm tropical trunk", {"freq": 20}, (0.6, 0.6)),
    ("mossy_beech_bark", (0.30, 0.30, 0.28), (0.38, 0.38, 0.35), (0.08, 0.12, 0.04), "Smooth gray beech bark with moss in shallow fissures", "beech moss bark forest", {"freq": 6, "slope": 9.0}, (0.6, 0.6)),
]:
    _R("bark", _mid, "tree_bark", C(_c1, _c2, _c3, **{"rough": 0.86, "slope": 14.0, **_kw}), _sz, _ph, _tg + " tree")

for _mid, _fam, _p, _sz, _ph, _tg, _kind in [
    ("weathered_driftwood", "wood_planks", _planks((0.46, 0.42, 0.36), (0.58, 0.54, 0.47), (0.20, 0.18, 0.15), planks=3, joints=0, cycles=12, groove=0.02, rough=0.80, slope=10.0, knots=0.3), (1.0, 0.5), "Sun-bleached driftwood with open, eroded grain", "driftwood beach bleached", "wood"),
    ("birch_plywood_sheet", "wood_planks", _planks((0.64, 0.50, 0.32), (0.56, 0.42, 0.25), (0.40, 0.30, 0.18), planks=1, joints=0, cycles=6, groove=0.01, rough=0.58, slope=4.0, knots=0.1), (1.22, 1.22), "Sanded birch plywood sheet with rotary-cut figure", "plywood birch sheet diy", "wood"),
    ("osb_strand_board", "speckled_aggregate", C((0.48, 0.34, 0.17), (0.58, 0.42, 0.22), (0.36, 0.24, 0.11), cells_a=14, cells_b=26, chip_fill=0.70, rough=0.72, slope=6.0), (1.22, 1.22), "Oriented strand board with compressed wood flakes", "osb construction sheathing", "wood"),
    ("mdf_particle_board", "speckled_aggregate", C((0.46, 0.34, 0.20), (0.52, 0.39, 0.24), (0.38, 0.27, 0.15), cells_a=40, cells_b=70, chip_fill=0.50, rough=0.76, slope=3.0), (1.22, 1.22), "Raw particle board with fine compressed wood chips", "particleboard mdf raw", "wood"),
    ("end_grain_butcher_block", "grid_tile", C((0.56, 0.38, 0.20), (0.40, 0.25, 0.12), (0.30, 0.18, 0.08), nx=8, ny=8, grout=0.02, bevel=0.02, rough=0.50, slope=6.0, grout_rmod=0.10), (0.4, 0.4), "End-grain butcher block checkerboard of glued wood blocks", "butcherblock endgrain kitchen", "wood"),
    ("maple_butcher_block_counter", "wood_planks", _planks((0.66, 0.47, 0.28), (0.58, 0.40, 0.22), (0.40, 0.27, 0.15), planks=10, joints=2, groove=0.02, rough=0.42, slope=5.0, knots=0.02), (1.2, 0.4), "Edge-grain maple butcher-block countertop staves", "butcherblock maple kitchen", "wood"),
    ("chinked_log_cabin_wall", "wood_slats", C((0.40, 0.22, 0.10), (0.48, 0.28, 0.13), (0.50, 0.46, 0.38), slats=5, gap_ratio=0.06, vertical=False, rough=0.70, slope=12.0), (1.25, 1.25), "Round log cabin wall with pale mortar chinking", "log cabin rustic chinking", "wood"),
]:
    extra = ("kitchen_countertop", "kitchen_island") if "butcher" in _mid else (("cabin_exterior", "lodge_wall") if "log" in _mid else ())
    roles = ("furniture",) if "butcher" in _mid else (("wall",) if "log" in _mid else ())
    _R("engineered_wood", _mid, _fam, _p, _sz, _ph, _tg + " wood board", kind=_kind, objs=extra, roles=roles)

# ===========================================================================
# 3. STONE, MARBLE & CONCRETE (53 parametric)
# ===========================================================================
for _mid, _c1, _c2, _c3, _ph, _tg, _kw in [
    ("calacatta_gold_marble", (0.80, 0.78, 0.74), (0.74, 0.72, 0.68), (0.40, 0.30, 0.16), "White Calacatta marble with bold gold-gray veining", "white gold calacatta luxury", {"vein_sharp": 3.5}),
    ("statuario_marble", (0.82, 0.82, 0.81), (0.76, 0.76, 0.76), (0.16, 0.17, 0.18), "Bright statuario marble with crisp charcoal veins", "white statuario luxury", {"vein_sharp": 5.0}),
    ("nero_marquina_marble", (0.025, 0.025, 0.028), (0.04, 0.04, 0.045), (0.75, 0.75, 0.74), "Jet-black Nero Marquina marble with bright white veins", "black nero dramatic", {}),
    ("verde_guatemala_marble", (0.02, 0.10, 0.06), (0.04, 0.16, 0.09), (0.55, 0.62, 0.56), "Deep green serpentine marble with pale veining", "green verde serpentine", {}),
    ("rosso_levanto_marble", (0.24, 0.06, 0.05), (0.32, 0.10, 0.07), (0.65, 0.60, 0.55), "Wine-red Rosso Levanto marble with white veins", "red rosso burgundy", {}),
    ("emperador_dark_marble", (0.12, 0.07, 0.04), (0.18, 0.11, 0.06), (0.50, 0.40, 0.28), "Chocolate Emperador marble with tan crackle veins", "brown emperador chocolate", {"vein_freq": 4.0, "vein_sharp": 3.0}),
    ("arabescato_marble", (0.80, 0.80, 0.79), (0.74, 0.74, 0.74), (0.22, 0.22, 0.24), "Arabescato marble with dense swirling gray veins", "white arabescato swirling", {"vein_freq": 4.0, "vein_sharp": 2.5}),
    ("honey_onyx", (0.70, 0.46, 0.18), (0.60, 0.36, 0.12), (0.85, 0.72, 0.48), "Translucent honey onyx with banded amber layers", "onyx honey amber backlit", {"vein_freq": 6.0, "vein_sharp": 2.0, "rough": 0.10}),
    ("green_onyx", (0.36, 0.52, 0.20), (0.24, 0.40, 0.12), (0.78, 0.80, 0.60), "Lime-green onyx with cream bands", "onyx green banded", {"vein_freq": 6.0, "vein_sharp": 2.0, "rough": 0.10}),
    ("charcoal_soapstone", (0.12, 0.13, 0.13), (0.16, 0.17, 0.17), (0.40, 0.42, 0.42), "Honed charcoal soapstone with soft talc veins", "soapstone charcoal honed matte", {"rough": 0.55}),
    ("crema_marfil_marble", (0.72, 0.62, 0.46), (0.66, 0.56, 0.40), (0.50, 0.40, 0.26), "Warm beige Crema Marfil marble with fine veins", "beige crema warm", {}),
    ("rosa_portugal_marble", (0.70, 0.50, 0.44), (0.62, 0.42, 0.36), (0.40, 0.25, 0.22), "Blush-pink Rosa Portugal marble", "pink rosa blush", {}),
]:
    _R("marble", _mid, "veined_marble", C(_c1, _c2, _c3, **{"vein_freq": 3.0, "vein_sharp": 4.5, "angle": 0.55, "rough": 0.16, "slope": 3.0, **_kw}), (1.2, 1.2), _ph, _tg + " marble stone")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("absolute_black_granite", (0.02, 0.02, 0.022), (0.07, 0.07, 0.075), (0.04, 0.04, 0.045), "Polished absolute-black granite with faint crystal sparkle", "black granite polished", {"rough": 0.12}, (1.0, 1.0)),
    ("salt_pepper_granite", (0.40, 0.40, 0.39), (0.04, 0.04, 0.04), (0.70, 0.70, 0.68), "Salt-and-pepper granite with black and white crystals", "gray granite saltpepper", {"cells_a": 30, "cells_b": 60}, (1.0, 1.0)),
    ("kashmir_white_granite", (0.66, 0.64, 0.60), (0.18, 0.12, 0.10), (0.42, 0.40, 0.38), "Kashmir white granite with garnet flecks", "white granite garnet", {"cells_a": 26, "cells_b": 50, "chip_fill": 0.35}, (1.0, 1.0)),
    ("baltic_brown_granite", (0.18, 0.11, 0.07), (0.04, 0.03, 0.03), (0.30, 0.22, 0.14), "Baltic brown granite with large rapakivi feldspar rounds", "brown granite baltic", {"cells_a": 12, "cells_b": 24, "chip_fill": 0.60}, (1.0, 1.0)),
    ("blue_pearl_granite", (0.10, 0.11, 0.14), (0.20, 0.26, 0.36), (0.04, 0.04, 0.05), "Blue pearl larvikite with iridescent blue feldspar", "blue granite larvikite", {"cells_a": 16, "cells_b": 34}, (1.0, 1.0)),
    ("classic_venetian_terrazzo", (0.62, 0.60, 0.56), (0.40, 0.12, 0.08), (0.12, 0.12, 0.12), "Classic Venetian terrazzo with red and black marble chips", "terrazzo venetian classic", {"cells_a": 14, "cells_b": 30, "rough": 0.22}, (1.0, 1.0)),
    ("white_quartz_countertop", (0.78, 0.78, 0.77), (0.66, 0.66, 0.66), (0.55, 0.55, 0.56), "Engineered white quartz with fine gray speckle", "quartz white engineered kitchen", {"cells_a": 40, "cells_b": 80, "chip_fill": 0.25, "rough": 0.08}, (1.0, 1.0)),
    ("epoxy_flake_garage_floor", (0.30, 0.30, 0.31), (0.06, 0.06, 0.06), (0.62, 0.62, 0.60), "Glossy epoxy floor coating with broadcast vinyl flakes", "epoxy garage flake industrial", {"cells_a": 40, "cells_b": 80, "chip_fill": 0.60, "rough": 0.12}, (1.0, 1.0)),
    ("pastel_memphis_terrazzo", (0.78, 0.74, 0.68), (0.62, 0.30, 0.28), (0.20, 0.42, 0.48), "Playful terrazzo with oversized coral and teal chips", "terrazzo pastel memphis playful", {"cells_a": 10, "cells_b": 22, "rough": 0.24}, (1.0, 1.0)),
    ("charcoal_terrazzo", (0.08, 0.08, 0.085), (0.70, 0.70, 0.68), (0.40, 0.22, 0.12), "Charcoal terrazzo with white and rust aggregate", "terrazzo charcoal dark", {"cells_a": 16, "cells_b": 34, "rough": 0.22}, (1.0, 1.0)),
    ("recycled_glass_terrazzo", (0.70, 0.70, 0.67), (0.05, 0.30, 0.20), (0.10, 0.22, 0.45), "Terrazzo with green and blue recycled glass chips", "terrazzo glass recycled eco", {"cells_a": 18, "cells_b": 36, "rough": 0.16}, (1.0, 1.0)),
]:
    objs = ("terrazzo_floor", "airport_concourse") if "terrazzo" in _mid else (("garage_floor", "workshop_floor") if "epoxy" in _mid else ())
    _R("granite_quartz", _mid, "speckled_aggregate", C(_c1, _c2, _c3, **{"rough": 0.18, "slope": 4.0, **_kw}), _sz, _ph, _tg + " stone countertop", objs=objs)

for _mid, _p, _sz, _ph, _tg in [
    ("board_formed_concrete", C((0.32, 0.31, 0.29), (0.40, 0.39, 0.37), (0.10, 0.10, 0.10), style="board_formed", boards=6, rough=0.80, slope=6.0), (1.2, 1.2), "Board-formed concrete with timber grain imprint and tie-rod holes", "boardformed brutalist architectural"),
    ("smooth_architectural_concrete", C((0.42, 0.41, 0.39), (0.48, 0.47, 0.45), (0.20, 0.20, 0.19), grit=0.20, pores=0.15, rough=0.72, slope=4.0), (2.0, 2.0), "Smooth fair-faced architectural concrete with sparse bug holes", "smooth fairfaced minimal"),
    ("polished_concrete_floor", C((0.36, 0.35, 0.33), (0.44, 0.43, 0.41), (0.20, 0.19, 0.18), grit=0.10, pores=0.05, rough=0.22, slope=2.0), (2.0, 2.0), "Diamond-polished concrete floor with a soft satin sheen", "polished floor loft retail"),
    ("raw_pitted_concrete", C((0.30, 0.29, 0.27), (0.38, 0.37, 0.35), (0.12, 0.12, 0.11), grit=0.60, pores=0.60, rough=0.86, slope=6.0), (2.0, 2.0), "Raw cast concrete pitted with air voids", "raw pitted industrial"),
    ("gray_microcement", C((0.46, 0.45, 0.43), (0.56, 0.55, 0.52), (0.30, 0.30, 0.28), grit=0.15, pores=0.05, rough=0.52, slope=3.0), (1.5, 1.5), "Seamless troweled microcement with cloudy tonal swirls", "microcement bathroom seamless"),
    ("dark_brutalist_board_formed", C((0.18, 0.18, 0.17), (0.26, 0.25, 0.24), (0.05, 0.05, 0.05), style="board_formed", boards=8, rough=0.82, slope=6.0), (1.2, 1.2), "Dark board-formed concrete with narrow plank imprints", "boardformed dark brutalist"),
    ("rammed_earth_wall", C((0.48, 0.30, 0.18), (0.58, 0.40, 0.26), (0.34, 0.22, 0.13), style="rammed_earth", rough=0.86, slope=6.0), (2.0, 2.0), "Stratified rammed-earth wall in warm ochre bands", "rammed earth ochre sustainable"),
    ("chalk_rammed_earth", C((0.66, 0.62, 0.54), (0.58, 0.53, 0.45), (0.46, 0.42, 0.36), style="rammed_earth", rough=0.86, slope=6.0), (2.0, 2.0), "Pale chalk rammed earth with soft horizontal strata", "rammed chalk pale"),
    ("acid_stained_concrete", C((0.34, 0.20, 0.12), (0.24, 0.16, 0.10), (0.12, 0.08, 0.05), grit=0.30, pores=0.20, rough=0.40, slope=3.0), (2.0, 2.0), "Acid-stained concrete floor in mottled rust-browns", "stained rust mottled floor"),
]:
    _R("concrete", _mid, "concrete_plaster", _p, _sz, _ph, _tg + " concrete", kind="concrete")
_R("concrete", "exposed_aggregate_concrete", "speckled_aggregate",
   C((0.38, 0.37, 0.35), (0.30, 0.24, 0.18), (0.50, 0.48, 0.44), cells_a=30, cells_b=60, chip_fill=0.65, rough=0.80, slope=8.0),
   (1.5, 1.5), "Exposed-aggregate concrete with washed river pebbles", "exposed aggregate sidewalk pebble concrete", kind="concrete")

for _mid, _p, _sz, _ph, _tg in [
    ("white_lime_plaster", C((0.78, 0.77, 0.73), (0.72, 0.71, 0.67), (0.50, 0.49, 0.46), grit=0.15, pores=0.10, rough=0.74, slope=4.0), (2.0, 2.0), "Hand-troweled white lime plaster with soft cloudy relief", "white lime interior"),
    ("terracotta_tadelakt", C((0.48, 0.22, 0.12), (0.56, 0.28, 0.16), (0.30, 0.14, 0.07), grit=0.08, pores=0.05, rough=0.30, slope=3.0), (1.5, 1.5), "Polished Moroccan tadelakt in burnt terracotta", "tadelakt moroccan terracotta"),
    ("sand_finish_stucco", C((0.62, 0.56, 0.46), (0.68, 0.62, 0.52), (0.40, 0.36, 0.30), grit=0.80, pores=0.30, rough=0.86, slope=6.0), (1.5, 1.5), "Sand-finish exterior stucco in warm beige", "stucco sand exterior"),
    ("spanish_lace_stucco", C((0.72, 0.66, 0.54), (0.66, 0.60, 0.48), (0.46, 0.42, 0.34), grit=0.50, pores=0.50, rough=0.84, slope=6.0), (1.5, 1.5), "Knockdown Spanish-lace stucco with raised splatter islands", "stucco spanish villa"),
    ("orange_peel_drywall", C((0.76, 0.76, 0.74), (0.74, 0.74, 0.72), (0.60, 0.60, 0.58), grit=0.40, pores=0.02, rough=0.82, slope=2.5), (1.2, 1.2), "Painted drywall with fine orange-peel spray texture", "drywall painted interior office"),
    ("charcoal_venetian_plaster", C((0.08, 0.08, 0.085), (0.14, 0.14, 0.145), (0.04, 0.04, 0.04), grit=0.05, pores=0.02, rough=0.25, slope=3.0), (2.0, 2.0), "Burnished charcoal Venetian plaster with deep glossy clouds", "venetian charcoal luxury"),
    ("earthen_clay_plaster", C((0.52, 0.40, 0.28), (0.46, 0.34, 0.23), (0.30, 0.22, 0.14), grit=0.50, pores=0.20, rough=0.84, slope=5.0), (2.0, 2.0), "Natural earthen clay plaster with straw-flecked texture", "clay earthen natural"),
    ("sage_marmorino_plaster", C((0.36, 0.42, 0.32), (0.42, 0.48, 0.38), (0.22, 0.26, 0.20), grit=0.12, pores=0.05, rough=0.35, slope=3.0), (2.0, 2.0), "Sage-green marmorino plaster with silky burnish", "marmorino sage green"),
]:
    _R("plaster_stucco", _mid, "concrete_plaster", _p, _sz, _ph, _tg + " plaster wall", kind="plaster")

for _mid, _fam, _p, _sz, _ph, _tg in [
    ("charcoal_cleft_slate", "rock_cliff", C((0.07, 0.075, 0.08), (0.11, 0.115, 0.12), (0.03, 0.03, 0.03), layers=10, fracture=6, rough=0.62, slope=7.0), (1.2, 1.2), "Natural-cleft charcoal slate with layered ridges", "slate charcoal cleft"),
    ("green_riven_slate", "rock_cliff", C((0.10, 0.13, 0.10), (0.16, 0.19, 0.15), (0.04, 0.05, 0.04), layers=12, fracture=7, rough=0.62, slope=7.0), (1.2, 1.2), "Riven green slate with fine laminations", "slate green riven"),
    ("multicolor_slate_tile", "grid_tile", C((0.10, 0.09, 0.08), (0.24, 0.16, 0.10), (0.18, 0.17, 0.15), nx=3, ny=3, grout=0.03, handmade=0.3, bevel=0.04, rough=0.60, slope=7.0, grout_rmod=0.25), (0.9, 0.9), "Multicolor slate tiles mixing rust and charcoal", "slate tile multicolor rustic"),
    ("silver_travertine", "veined_marble", C((0.56, 0.54, 0.50), (0.48, 0.46, 0.42), (0.30, 0.29, 0.27), vein_freq=8.0, vein_sharp=2.5, angle=0.0, rough=0.50, slope=4.0), (1.2, 1.2), "Silver vein-cut travertine with linear banding", "travertine silver banded"),
    ("honed_fossil_limestone", "concrete_plaster", C((0.66, 0.62, 0.52), (0.60, 0.56, 0.46), (0.40, 0.37, 0.30), grit=0.25, pores=0.35, rough=0.60, slope=4.0), (1.2, 1.2), "Honed limestone with scattered fossil pores", "limestone honed fossil"),
    ("jerusalem_gold_limestone", "concrete_plaster", C((0.66, 0.52, 0.30), (0.70, 0.58, 0.36), (0.46, 0.36, 0.20), grit=0.30, pores=0.40, rough=0.66, slope=4.0), (1.2, 1.2), "Golden Jerusalem limestone with sun-warmed tones", "limestone gold jerusalem"),
    ("red_sandstone_slab", "rock_cliff", C((0.40, 0.16, 0.08), (0.48, 0.22, 0.12), (0.26, 0.10, 0.05), layers=14, fracture=4, rough=0.74, slope=7.0), (1.2, 1.2), "Red sandstone slab with cross-bedded layers", "sandstone red slab"),
    ("white_quartzite_slab", "veined_marble", C((0.74, 0.74, 0.72), (0.66, 0.66, 0.65), (0.42, 0.40, 0.36), vein_freq=2.0, vein_sharp=6.0, rough=0.20, slope=3.0), (1.2, 1.2), "White quartzite slab with glassy gray drifts", "quartzite white kitchen"),
    ("thermal_bluestone_slab", "concrete_plaster", C((0.16, 0.19, 0.22), (0.20, 0.23, 0.26), (0.10, 0.11, 0.12), grit=0.55, pores=0.10, rough=0.74, slope=5.0), (1.2, 1.2), "Thermal-finished bluestone with a flamed grippy face", "bluestone thermal patio"),
    ("honed_basalt_slab", "speckled_aggregate", C((0.06, 0.06, 0.065), (0.10, 0.10, 0.10), (0.03, 0.03, 0.03), cells_a=50, cells_b=90, chip_fill=0.20, rough=0.45, slope=3.0), (1.2, 1.2), "Honed black basalt slab with fine vesicles", "basalt honed black"),
    ("red_porphyry_stone", "speckled_aggregate", C((0.18, 0.06, 0.05), (0.40, 0.30, 0.26), (0.08, 0.03, 0.03), cells_a=24, cells_b=50, chip_fill=0.40, rough=0.40, slope=4.0), (1.2, 1.2), "Imperial red porphyry with pale feldspar crystals", "porphyry red imperial"),
    ("shell_fossil_limestone", "speckled_aggregate", C((0.62, 0.56, 0.44), (0.74, 0.70, 0.60), (0.40, 0.34, 0.24), cells_a=12, cells_b=28, chip_fill=0.35, rough=0.55, slope=5.0), (1.2, 1.2), "Shelly fossil limestone packed with shell fragments", "limestone fossil shell"),
]:
    _R("natural_stone", _mid, _fam, _p, _sz, _ph, _tg + " natural stone", kind="stone")

# ===========================================================================
# 4. TILES & CERAMICS (42 parametric)
# ===========================================================================
for _mid, _c1, _c2, _c3, _ph, _tg, _kw in [
    ("white_gloss_subway_tile", (0.80, 0.80, 0.78), (0.78, 0.78, 0.76), (0.40, 0.40, 0.38), "Classic white gloss 3x6 subway tile with pale gray grout", "white gloss classic", {}),
    ("black_gloss_subway_tile", (0.025, 0.025, 0.028), (0.035, 0.035, 0.04), (0.55, 0.55, 0.53), "Glossy black subway tile with contrasting light grout", "black gloss dramatic", {}),
    ("sage_green_subway_tile", (0.32, 0.42, 0.30), (0.36, 0.46, 0.34), (0.70, 0.70, 0.66), "Soft sage-green glazed subway tile", "sage green kitchen", {}),
    ("navy_subway_tile", (0.03, 0.06, 0.16), (0.05, 0.08, 0.20), (0.65, 0.65, 0.62), "Deep navy glazed subway tile with white grout", "navy blue bold", {}),
    ("vertical_stack_matte_tile", (0.74, 0.73, 0.70), (0.72, 0.71, 0.68), (0.20, 0.20, 0.20), "Matte white tall tiles in vertical stack bond with dark grout", "white matte vertical modern", {"nx": 8, "ny": 2, "stagger": 0.0, "rough": 0.35}),
    ("handmade_cream_subway_tile", (0.74, 0.68, 0.56), (0.70, 0.64, 0.52), (0.52, 0.50, 0.46), "Handmade cream subway tile with wavy glaze", "cream handmade artisan", {"handmade": 0.8}),
    ("elongated_blush_subway_tile", (0.70, 0.44, 0.40), (0.74, 0.48, 0.44), (0.75, 0.74, 0.72), "Long skinny blush-pink subway tile", "pink blush skinny", {"nx": 3, "ny": 12}),
    ("beveled_gray_metro_tile", (0.42, 0.43, 0.44), (0.46, 0.47, 0.48), (0.75, 0.75, 0.73), "Deep-bevel gray metro tile", "gray bevel metro parisian", {"bevel": 0.30}),
]:
    _R("subway_tile", _mid, "grid_tile", C(_c1, _c2, _c3, **{"nx": 4, "ny": 8, "stagger": 0.5, "grout": 0.05, "bevel": 0.14, "rough": 0.12, "slope": 8.0, **_kw}), (0.6, 0.6), _ph, _tg + " subway tile ceramic")

for _mid, _c1, _c2, _ph, _tg, _kw in [
    ("white_zellige_tile", (0.80, 0.79, 0.75), (0.70, 0.69, 0.64), "Glossy white Moroccan zellige with hand-cut edges", "white zellige moroccan", {}),
    ("emerald_zellige_tile", (0.02, 0.18, 0.10), (0.05, 0.28, 0.16), "Emerald zellige with deep pooled glaze", "green emerald zellige", {}),
    ("cobalt_zellige_tile", (0.02, 0.06, 0.30), (0.05, 0.12, 0.42), "Cobalt-blue zellige with kiln variation", "blue cobalt zellige", {}),
    ("terracotta_zellige_tile", (0.45, 0.18, 0.08), (0.55, 0.25, 0.12), "Burnt terracotta zellige with glossy glaze", "terracotta orange zellige", {}),
    ("blush_zellige_tile", (0.72, 0.48, 0.44), (0.66, 0.42, 0.38), "Blush-pink zellige for soft bathroom walls", "pink blush zellige", {}),
    ("mustard_zellige_tile", (0.62, 0.40, 0.06), (0.70, 0.48, 0.10), "Ochre mustard zellige with rich variation", "mustard yellow zellige", {}),
    ("teal_glazed_quarry_tile", (0.06, 0.28, 0.30), (0.10, 0.34, 0.36), "Larger teal glazed handmade quarry tiles", "teal glazed handmade", {"nx": 4, "ny": 4}),
]:
    _R("zellige", _mid, "grid_tile", C(_c1, _c2, (0.62, 0.60, 0.56), **{"nx": 6, "ny": 6, "grout": 0.03, "handmade": 1.0, "bevel": 0.10, "rough": 0.08, "slope": 8.0, **_kw}), (0.6, 0.6), _ph, _tg + " tile ceramic glossy")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw in [
    ("concrete_look_porcelain", (0.40, 0.39, 0.37), (0.46, 0.45, 0.43), (0.30, 0.29, 0.28), "Large-format matte porcelain with a concrete look", "gray matte concrete_look", {"rough": 0.60}),
    ("polished_white_porcelain", (0.82, 0.82, 0.81), (0.80, 0.80, 0.79), (0.70, 0.70, 0.68), "Mirror-polished white porcelain slabs", "white polished glossy", {"rough": 0.06}),
    ("black_porcelain_slab", (0.03, 0.03, 0.03), (0.04, 0.04, 0.04), (0.06, 0.06, 0.06), "Large-format satin black porcelain", "black satin modern", {"rough": 0.20}),
    ("warm_beige_porcelain_floor", (0.62, 0.54, 0.42), (0.66, 0.58, 0.46), (0.52, 0.47, 0.40), "Warm beige porcelain floor tiles", "beige warm floor", {"nx": 3, "ny": 3}),
    ("wood_look_porcelain_plank", (0.46, 0.30, 0.16), (0.38, 0.24, 0.12), (0.20, 0.15, 0.10), "Wood-look porcelain planks in staggered rows", "wood_look plank waterproof", {"nx": 1, "ny": 6, "stagger": 0.5, "rough": 0.50}),
]:
    _R("porcelain", _mid, "grid_tile", C(_c1, _c2, _c3, **{"nx": 2, "ny": 2, "grout": 0.015, "bevel": 0.04, "rough": 0.30, "slope": 5.0, **_kw}), (1.2, 1.2), _ph, _tg + " porcelain tile")

for _mid, _fam, _p, _sz, _ph, _tg, _kind in [
    ("white_hex_mosaic_black_grout", "mosaic_tile", C((0.78, 0.78, 0.76), (0.74, 0.74, 0.72), (0.05, 0.05, 0.05), mode="hex", freq=10, grout=0.10, rough=0.30, slope=8.0), (0.3, 0.3), "Matte white hex mosaic with black grout", "white hex black_grout vintage", None),
    ("large_hex_cement_tile", "mosaic_tile", C((0.46, 0.44, 0.40), (0.52, 0.50, 0.46), (0.38, 0.36, 0.33), mode="hex", freq=4, grout=0.06, rough=0.62, slope=6.0), (0.8, 0.8), "Large hexagonal cement tiles in warm gray", "hex cement large", None),
    ("terracotta_hex_tile", "mosaic_tile", C((0.44, 0.17, 0.07), (0.52, 0.22, 0.10), (0.34, 0.30, 0.25), mode="hex", freq=4, grout=0.08, rough=0.70, slope=7.0), (1.0, 1.0), "Unglazed terracotta hexagon floor tiles", "hex terracotta rustic", None),
    ("green_penny_round_mosaic", "mosaic_tile", C((0.06, 0.30, 0.16), (0.10, 0.38, 0.22), (0.70, 0.70, 0.67), mode="penny", freq=12, grout=0.10, rough=0.12, slope=8.0), (0.24, 0.24), "Glossy green penny-round mosaic", "green penny glossy", None),
    ("white_penny_round_mosaic", "mosaic_tile", C((0.80, 0.80, 0.78), (0.76, 0.76, 0.74), (0.30, 0.30, 0.29), mode="penny", freq=12, grout=0.10, rough=0.14, slope=8.0), (0.24, 0.24), "Classic white penny-round mosaic", "white penny classic", None),
    ("black_penny_round_mosaic", "mosaic_tile", C((0.03, 0.03, 0.03), (0.05, 0.05, 0.05), (0.65, 0.65, 0.62), mode="penny", freq=12, grout=0.10, rough=0.20, slope=8.0), (0.24, 0.24), "Matte black penny-round mosaic", "black penny matte", None),
    ("diamond_marble_mosaic", "mosaic_tile", C((0.72, 0.71, 0.68), (0.50, 0.49, 0.47), (0.40, 0.40, 0.38), mode="diamond", freq=6, grout=0.08, rough=0.20, slope=6.0), (0.4, 0.4), "Honed marble diamond mosaic", "diamond marble honed", "stone"),
    ("blue_glass_pool_mosaic", "grid_tile", C((0.02, 0.20, 0.40), (0.06, 0.34, 0.55), (0.70, 0.70, 0.68), nx=12, ny=12, grout=0.08, bevel=0.10, rough=0.05, slope=6.0), (0.3, 0.3), "Glossy blue glass pool mosaic", "blue pool glass swimming", None),
    ("copper_metallic_mosaic", "grid_tile", C((0.85, 0.45, 0.30), (0.75, 0.38, 0.24), (0.08, 0.07, 0.06), nx=10, ny=10, grout=0.08, bevel=0.10, rough=0.28, slope=6.0, metallic=0.95), (0.3, 0.3), "Metallic copper mosaic chips with dark grout", "copper metallic accent", "copper"),
    ("cobalt_diamond_mosaic", "mosaic_tile", C((0.02, 0.06, 0.30), (0.04, 0.12, 0.45), (0.75, 0.75, 0.72), mode="diamond", freq=8, grout=0.08, rough=0.12, slope=6.0), (0.3, 0.3), "Cobalt diamond mosaic with bright grout", "cobalt diamond blue", None),
    ("kit_kat_finger_tile", "grid_tile", C((0.76, 0.76, 0.74), (0.72, 0.72, 0.70), (0.40, 0.40, 0.38), nx=12, ny=3, grout=0.06, bevel=0.12, rough=0.30, slope=7.0), (0.3, 0.3), "Matte white finger mosaic in stacked rows", "finger kitkat white", None),
]:
    objs = ("pool_lining", "pool_step") if "pool" in _mid else ()
    _R("mosaic", _mid, _fam, _p, _sz, _ph, _tg + " mosaic tile", kind=_kind, objs=objs)

for _mid, _c1, _c2, _c3, _ph, _tg, _kw in [
    ("celadon_crackle_glaze", (0.42, 0.56, 0.46), (0.48, 0.62, 0.52), (0.10, 0.12, 0.10), "Jade celadon glaze with fine crackle network", "celadon jade crackle asian", {"mode": "crackle", "cells": 14, "rough": 0.08}),
    ("ivory_crackle_glaze", (0.74, 0.68, 0.54), (0.70, 0.64, 0.50), (0.26, 0.22, 0.16), "Antique ivory crackle glaze with tea-stained crazing", "ivory antique crackle", {"mode": "crackle", "cells": 14, "rough": 0.10}),
    ("raku_crackle_glaze", (0.62, 0.62, 0.60), (0.52, 0.52, 0.50), (0.02, 0.02, 0.02), "White raku glaze with smoke-blackened crackle", "raku smoke crackle", {"mode": "crackle", "cells": 10, "rough": 0.14}),
    ("oatmeal_speckled_stoneware", (0.62, 0.56, 0.46), (0.56, 0.50, 0.40), (0.10, 0.08, 0.06), "Oatmeal speckled stoneware with iron flecks and throwing lines", "stoneware speckled oatmeal", {"rough": 0.40}),
    ("matte_charcoal_stoneware", (0.08, 0.08, 0.08), (0.12, 0.12, 0.11), (0.40, 0.38, 0.34), "Matte charcoal stoneware with pale speckles", "stoneware charcoal matte", {"rough": 0.60}),
    ("cobalt_enamel_glaze", (0.03, 0.08, 0.38), (0.05, 0.12, 0.48), (0.70, 0.70, 0.68), "Glossy cobalt enamel glaze over stoneware", "cobalt enamel glossy", {"rough": 0.06, "speck_str": 0.20}),
]:
    _R("glaze_ceramic", _mid, "ceramic_glaze", C(_c1, _c2, _c3, **{"mode": "stoneware", "slope": 5.0, **_kw}), (0.3, 0.3), _ph, _tg + " ceramic pottery glaze")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("saltillo_terracotta_tile", (0.46, 0.18, 0.07), (0.54, 0.25, 0.11), (0.42, 0.38, 0.32), "Handmade Mexican Saltillo terracotta tiles", "saltillo mexican rustic", {"nx": 3, "ny": 3, "handmade": 0.9}, (0.9, 0.9)),
    ("cotto_square_tile", (0.52, 0.24, 0.11), (0.58, 0.30, 0.15), (0.50, 0.46, 0.40), "Tuscan cotto square floor tiles", "cotto tuscan italian", {"nx": 4, "ny": 4, "handmade": 0.4}, (1.0, 1.0)),
    ("red_quarry_tile", (0.30, 0.08, 0.04), (0.36, 0.11, 0.05), (0.22, 0.21, 0.20), "Dense red quarry tiles for commercial kitchens", "quarry red commercial", {"nx": 4, "ny": 4, "handmade": 0.1}, (0.6, 0.6)),
    ("ochre_cement_tile", (0.58, 0.42, 0.14), (0.62, 0.46, 0.18), (0.50, 0.46, 0.40), "Pigmented ochre cement tiles with tight joints", "cement ochre encaustic", {"nx": 3, "ny": 3, "grout": 0.012, "bevel": 0.02}, (0.6, 0.6)),
    ("glazed_terracotta_brick_tile", (0.50, 0.22, 0.10), (0.56, 0.28, 0.14), (0.46, 0.42, 0.36), "Glazed terracotta brick-shaped tiles in running bond", "terracotta glazed brick_tile", {"nx": 2, "ny": 8, "stagger": 0.5, "handmade": 0.5, "rough": 0.15}, (0.6, 0.6)),
]:
    _R("terracotta", _mid, "grid_tile", C(_c1, _c2, _c3, **{"grout": 0.04, "bevel": 0.08, "rough": 0.68, "slope": 7.0, "grout_rmod": 0.25, **_kw}), _sz, _ph, _tg + " terracotta tile")

# ===========================================================================
# 5. METALS & INDUSTRIAL (32 parametric)
# ===========================================================================
for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _kind, _sz in [
    ("brushed_stainless_steel", (0.56, 0.57, 0.58), (0.66, 0.67, 0.68), None, "Fine-line brushed stainless steel sheet", "stainless steel appliance", {}, "metal", (0.5, 0.5)),
    ("brushed_aluminum_satin", (0.80, 0.81, 0.82), (0.88, 0.89, 0.90), None, "Satin-brushed aluminum with bright silver sheen", "aluminum silver satin", {"rough": 0.30}, "metal", (0.5, 0.5)),
    ("brushed_satin_brass", (0.78, 0.56, 0.24), (0.86, 0.64, 0.30), None, "Satin-brushed brass for fixtures and trim", "brass gold fixture", {"rough": 0.26}, "brass", (0.5, 0.5)),
    ("brushed_satin_copper", (0.86, 0.46, 0.32), (0.92, 0.54, 0.40), None, "Satin-brushed copper sheet with warm pink sheen", "copper warm", {}, "copper", (0.5, 0.5)),
    ("radial_spun_aluminum", (0.82, 0.83, 0.84), (0.90, 0.91, 0.92), None, "Radially spun aluminum with concentric machining", "aluminum radial spun", {"radial": True}, "metal", (0.3, 0.3)),
    ("brushed_titanium_seam_panel", (0.50, 0.47, 0.43), (0.58, 0.55, 0.50), (0.08, 0.08, 0.08), "Brushed titanium panels with recessed seams", "titanium panel aerospace", {"panel_seam": True}, "metal", (1.0, 1.0)),
    ("anodized_black_aluminum", (0.05, 0.05, 0.055), (0.08, 0.08, 0.085), None, "Bead-brushed black anodized aluminum", "black anodized aluminum gadget", {"metallic": 0.90, "rough": 0.32}, "metal", (0.5, 0.5)),
    ("anodized_blue_aluminum", (0.10, 0.22, 0.55), (0.16, 0.30, 0.65), None, "Brushed blue anodized aluminum", "blue anodized aluminum", {"metallic": 0.90, "rough": 0.30}, "metal", (0.5, 0.5)),
]:
    _R("brushed_metal", _mid, "brushed_metal", C(_c1, _c2, _c3, **{"metallic": 0.95, "rough": 0.28, "slope": 2.0, **_kw}), _sz, _ph, _tg + " brushed metal", kind=_kind,
       axis=("radial" if _kw.get("radial") else "u"))

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _kind, _sz in [
    ("hammered_nickel_silver", (0.62, 0.60, 0.56), (0.72, 0.70, 0.66), None, "Hand-hammered nickel silver with mirror facets", "nickel silver hammered", {"mode": "hammered", "cells": 10, "rough": 0.22}, "metal", (0.4, 0.4)),
    ("hammered_black_iron", (0.14, 0.13, 0.12), (0.22, 0.21, 0.20), None, "Blackened hand-forged iron with hammer marks", "iron black forged", {"mode": "hammered", "cells": 8, "metallic": 0.85, "rough": 0.45}, "metal", (0.4, 0.4)),
    ("spangled_galvanized_steel", (0.58, 0.60, 0.62), (0.70, 0.72, 0.74), None, "Hot-dip galvanized steel with crystalline zinc spangle", "galvanized zinc spangle", {"mode": "galvanized", "cells": 26, "rough": 0.32}, "metal", (0.6, 0.6)),
    ("dull_galvanized_steel", (0.44, 0.45, 0.46), (0.52, 0.53, 0.54), None, "Weathered dull-gray galvanized steel", "galvanized dull industrial", {"mode": "galvanized", "cells": 40, "rough": 0.45}, "metal", (0.6, 0.6)),
    ("corten_weathering_steel", (0.24, 0.09, 0.03), (0.38, 0.15, 0.05), (0.12, 0.05, 0.02), "Corten weathering steel with stable rust patina", "corten rust weathering", {"mode": "corten_patina", "metallic": 0.60, "rough": 0.40}, "metal", (1.0, 1.0)),
    ("heavy_scale_rusted_iron", (0.18, 0.08, 0.03), (0.42, 0.20, 0.07), (0.06, 0.03, 0.02), "Heavily scaled rusted iron with flaking oxide", "rust iron abandoned", {"mode": "corten_patina", "metallic": 0.50, "rough": 0.45}, "metal", (1.0, 1.0)),
    ("blued_raindrop_damascus", (0.20, 0.22, 0.28), (0.55, 0.57, 0.62), None, "Heat-blued damascus steel with flowing etched bands", "damascus blued blade", {"mode": "damascus", "rough": 0.24}, "metal", (0.2, 0.2)),
    ("sand_cast_aluminum", (0.64, 0.65, 0.66), (0.74, 0.75, 0.76), None, "Sand-cast aluminum with pebbled foundry skin", "cast aluminum foundry", {"mode": "cast", "rough": 0.48}, "metal", (0.3, 0.3)),
]:
    _R("specialty_metal", _mid, "specialty_metal", C(_c1, _c2, _c3, **{"metallic": 0.95, "slope": 6.0, **_kw}), _sz, _ph, _tg + " metal", kind=_kind)

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _kind, _sz in [
    ("zinc_standing_seam_roof", (0.40, 0.42, 0.44), (0.48, 0.50, 0.52), (0.20, 0.21, 0.22), "Pre-weathered zinc standing-seam roofing", "zinc standing_seam roof", {"mode": "standing_seam", "freq": 4, "metallic": 0.90, "rough": 0.42}, "metal", (1.6, 1.6)),
    ("copper_standing_seam_roof", (0.84, 0.44, 0.30), (0.90, 0.52, 0.36), (0.30, 0.14, 0.08), "Bright copper standing-seam roof panels", "copper standing_seam roof", {"mode": "standing_seam", "freq": 4}, "copper", (1.6, 1.6)),
    ("corrugated_galvanized_siding", (0.56, 0.58, 0.60), (0.66, 0.68, 0.70), (0.30, 0.30, 0.30), "Corrugated galvanized sheet for barns and sheds", "corrugated galvanized barn", {"mode": "corrugated", "freq": 12}, "metal", (0.912, 0.912)),
    ("rusty_corrugated_roofing", (0.22, 0.10, 0.04), (0.34, 0.16, 0.07), (0.10, 0.05, 0.02), "Rust-eaten corrugated roofing sheet", "corrugated rusty abandoned shanty", {"mode": "corrugated", "freq": 12, "metallic": 0.45, "rough": 0.70}, "metal", (0.912, 0.912)),
    ("fluted_aluminum_facade", (0.78, 0.79, 0.80), (0.86, 0.87, 0.88), (0.40, 0.40, 0.40), "Fluted aluminum facade panels", "aluminum fluted facade", {"mode": "fluted", "freq": 10}, "metal", (0.5, 0.5)),
    ("riveted_aircraft_aluminum", (0.70, 0.71, 0.72), (0.80, 0.81, 0.82), (0.10, 0.10, 0.10), "Riveted aircraft aluminum skin with lap seams", "aircraft riveted aluminum", {"mode": "riveted"}, "metal", (1.0, 1.0)),
    ("riveted_bridge_steel_plate", (0.34, 0.35, 0.36), (0.42, 0.43, 0.44), (0.08, 0.08, 0.08), "Riveted structural steel plate", "steel riveted bridge industrial", {"mode": "riveted", "metallic": 0.85, "rough": 0.50}, "metal", (1.0, 1.0)),
]:
    _R("metal_cladding", _mid, "metal_cladding", C(_c1, _c2, _c3, **{"metallic": 0.95, "rough": 0.38, "slope": 8.0, **_kw}), _sz, _ph, _tg + " metal cladding", kind=_kind,
       axis=("none" if _kw["mode"] == "riveted" else "v"))

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _kind, _sz in [
    ("round_perf_stainless_panel", (0.58, 0.59, 0.60), (0.66, 0.67, 0.68), (0.02, 0.02, 0.02), "Staggered round-hole perforated stainless panel", "perforated stainless round", {"mode": "round_perf", "freq": 16}, "metal", (0.4, 0.4)),
    ("perforated_black_steel_screen", (0.05, 0.05, 0.05), (0.09, 0.09, 0.09), (0.01, 0.01, 0.01), "Powder-coated black perforated steel screen", "perforated black screen", {"mode": "round_perf", "freq": 20, "metallic": 0.60, "rough": 0.40}, "metal", (0.4, 0.4)),
    ("slotted_bronze_grille", (0.50, 0.32, 0.16), (0.60, 0.40, 0.20), (0.02, 0.02, 0.02), "Slotted bronze ventilation grille", "bronze slotted grille", {"mode": "slotted", "freq": 12}, "bronze", (0.4, 0.4)),
    ("woven_wire_insect_screen", (0.30, 0.30, 0.31), (0.40, 0.40, 0.41), (0.01, 0.01, 0.01), "Fine woven wire insect screen", "wire screen window", {"mode": "wire_screen", "freq": 32, "metallic": 0.85}, "metal", (0.1, 0.1)),
    ("brass_cabinet_wire_mesh", (0.78, 0.56, 0.24), (0.86, 0.64, 0.30), (0.02, 0.015, 0.01), "Woven brass mesh for cabinet door inserts", "brass mesh cabinet", {"mode": "wire_screen", "freq": 16}, "brass", (0.15, 0.15)),
]:
    _R("perforated_metal", _mid, "perforated_mesh", C(_c1, _c2, _c3, **{"metallic": 0.95, "rough": 0.32, "slope": 9.0, **_kw}), _sz, _ph, _tg + " metal mesh", kind=_kind)

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _kind, _sz in [
    ("fine_steel_knurl", (0.56, 0.57, 0.58), (0.68, 0.69, 0.70), (0.10, 0.10, 0.10), "Fine diamond knurling on turned steel", "knurl steel grip", {"mode": "knurl", "freq": 20}, "metal", (0.05, 0.05)),
    ("brass_knurl_grip", (0.78, 0.56, 0.24), (0.88, 0.66, 0.32), (0.20, 0.12, 0.04), "Coarse diamond knurl on solid brass", "knurl brass grip", {"mode": "knurl", "freq": 14}, "brass", (0.05, 0.05)),
    ("aluminum_diamond_tread_plate", (0.74, 0.75, 0.76), (0.84, 0.85, 0.86), (0.30, 0.30, 0.30), "Bright aluminum diamond tread plate", "tread aluminum truck", {"mode": "tread", "freq": 8}, "metal", (0.6, 0.6)),
    ("black_steel_tread_plate", (0.10, 0.10, 0.10), (0.18, 0.18, 0.18), (0.04, 0.04, 0.04), "Black steel checker tread plate", "tread steel black industrial", {"mode": "tread", "freq": 12, "rough": 0.50}, "metal", (0.6, 0.6)),
]:
    _R("machined_metal", _mid, "machined_grip", C(_c1, _c2, _c3, **{"metallic": 0.95, "rough": 0.30, "slope": 10.0, **_kw}), _sz, _ph, _tg + " machined metal", kind=_kind)

# ===========================================================================
# 6. FABRICS & UPHOLSTERY (43 parametric)
# ===========================================================================
for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("natural_linen_plain", (0.52, 0.46, 0.36), (0.58, 0.52, 0.42), (0.30, 0.26, 0.20), "Natural flax linen in a slubby plain weave", "linen natural flax", {"mode": "plain", "freq": 32}, (0.06, 0.06)),
    ("white_cotton_canvas", (0.74, 0.72, 0.66), (0.70, 0.68, 0.62), (0.45, 0.43, 0.38), "Heavy white cotton canvas", "canvas white cotton", {"mode": "plain", "freq": 24}, (0.08, 0.08)),
    ("olive_canvas_duck", (0.14, 0.15, 0.07), (0.18, 0.19, 0.09), (0.08, 0.08, 0.04), "Olive cotton duck canvas for bags and tents", "canvas olive duck military", {"mode": "plain", "freq": 20}, (0.08, 0.08)),
    ("raw_selvedge_denim", (0.03, 0.06, 0.16), (0.12, 0.16, 0.26), (0.02, 0.03, 0.06), "Raw indigo selvedge denim in 3x1 twill", "denim indigo raw jeans", {"mode": "twill", "freq": 32}, (0.05, 0.05)),
    ("black_denim_twill", (0.025, 0.025, 0.03), (0.06, 0.06, 0.065), (0.01, 0.01, 0.012), "Washed black denim twill", "denim black twill", {"mode": "twill", "freq": 32}, (0.05, 0.05)),
    ("khaki_chino_twill", (0.42, 0.34, 0.20), (0.48, 0.40, 0.25), (0.24, 0.19, 0.11), "Fine khaki chino cotton twill", "khaki chino twill", {"mode": "twill", "freq": 40}, (0.04, 0.04)),
    ("orange_nylon_ripstop", (0.70, 0.20, 0.02), (0.78, 0.26, 0.04), (0.40, 0.10, 0.01), "Coated orange nylon ripstop", "ripstop orange nylon outdoor", {"mode": "ripstop", "freq": 48, "rough": 0.45}, (0.05, 0.05)),
    ("olive_military_ripstop", (0.10, 0.12, 0.06), (0.14, 0.16, 0.08), (0.05, 0.06, 0.03), "Olive drab military ripstop", "ripstop military olive", {"mode": "ripstop", "freq": 48}, (0.05, 0.05)),
    ("red_tartan_plaid", (0.40, 0.03, 0.03), (0.03, 0.08, 0.03), (0.02, 0.02, 0.04), "Red Scottish tartan with green and navy overchecks", "tartan plaid red scottish", {"mode": "plaid", "freq": 40}, (0.24, 0.24)),
    ("gray_glen_plaid", (0.30, 0.30, 0.30), (0.12, 0.12, 0.12), (0.60, 0.58, 0.55), "Gray glen plaid suiting check", "plaid glen gray suiting", {"mode": "plaid", "freq": 40}, (0.2, 0.2)),
    ("buffalo_check_flannel", (0.40, 0.02, 0.02), (0.02, 0.02, 0.02), (0.25, 0.01, 0.01), "Red and black buffalo check flannel", "flannel buffalo check lumberjack", {"mode": "plaid", "freq": 32}, (0.2, 0.2)),
    ("cream_boucle", (0.72, 0.68, 0.60), (0.66, 0.62, 0.54), (0.40, 0.37, 0.32), "Nubby cream boucle upholstery", "boucle cream nubby", {"mode": "boucle", "freq": 22}, (0.08, 0.08)),
    ("charcoal_boucle", (0.10, 0.10, 0.10), (0.16, 0.16, 0.155), (0.05, 0.05, 0.05), "Charcoal boucle with looped yarns", "boucle charcoal", {"mode": "boucle", "freq": 22}, (0.08, 0.08)),
    ("donegal_tweed_twill", (0.26, 0.22, 0.16), (0.40, 0.34, 0.26), (0.08, 0.06, 0.04), "Flecked Donegal tweed twill", "tweed donegal wool", {"mode": "twill", "freq": 24}, (0.08, 0.08)),
    ("gold_silk_dupioni", (0.55, 0.38, 0.12), (0.65, 0.48, 0.18), (0.30, 0.20, 0.06), "Shimmering gold silk dupioni with slub texture", "silk gold luxury", {"mode": "plain", "freq": 48, "rough": 0.30}, (0.04, 0.04)),
    ("slate_chenille_weave", (0.30, 0.30, 0.32), (0.36, 0.36, 0.38), (0.15, 0.15, 0.16), "Soft slate-blue chenille upholstery weave", "chenille slate upholstery", {"mode": "boucle", "freq": 30}, (0.08, 0.08)),
]:
    _R("woven_textile", _mid, "woven_fabric", C(_c1, _c2, _c3, **{"rough": 0.86, "slope": 6.0, **_kw}), _sz, _ph, _tg + " fabric textile")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz, _extra in [
    ("navy_plush_velvet", (0.02, 0.03, 0.10), (0.05, 0.07, 0.18), None, "Deep navy cotton velvet", "velvet navy plush", {"mode": "velvet"}, (0.2, 0.2), ()),
    ("crushed_burgundy_velvet", (0.16, 0.01, 0.03), (0.30, 0.03, 0.06), None, "Crushed burgundy velvet with mottled nap", "velvet burgundy crushed", {"mode": "velvet", "crush": 1.0}, (0.2, 0.2), ()),
    ("mustard_velvet", (0.50, 0.30, 0.03), (0.62, 0.40, 0.06), None, "Mustard-yellow performance velvet", "velvet mustard", {"mode": "velvet"}, (0.2, 0.2), ()),
    ("wide_wale_rust_corduroy", (0.30, 0.09, 0.03), (0.42, 0.14, 0.05), None, "Wide-wale rust corduroy", "corduroy rust wide_wale", {"mode": "corduroy", "freq": 10}, (0.1, 0.1), ()),
    ("fine_wale_cream_corduroy", (0.66, 0.60, 0.48), (0.72, 0.66, 0.54), None, "Fine-wale cream corduroy", "corduroy cream fine_wale", {"mode": "corduroy", "freq": 20}, (0.1, 0.1), ()),
    ("quilted_puffer_nylon", (0.03, 0.03, 0.04), (0.08, 0.08, 0.09), (0.01, 0.01, 0.01), "Black quilted puffer nylon with stitched baffles", "quilted puffer jacket nylon", {"mode": "quilted", "freq": 14, "rough": 0.35}, (0.3, 0.3), ("puffer_jacket", "sleeping_bag")),
    ("diamond_quilted_mattress", (0.76, 0.76, 0.74), (0.70, 0.70, 0.68), (0.40, 0.40, 0.38), "White diamond-quilted mattress ticking", "quilted mattress white", {"mode": "quilted", "freq": 20}, (0.4, 0.4), ("mattress", "moving_blanket")),
    ("chunky_oatmeal_knit", (0.58, 0.52, 0.42), (0.66, 0.60, 0.50), None, "Chunky oatmeal hand-knit wool", "knit chunky wool oatmeal", {"mode": "knit", "freq": 10}, (0.15, 0.15), ("throw_blanket", "sweater")),
    ("heather_jersey_knit", (0.30, 0.30, 0.31), (0.40, 0.40, 0.41), None, "Fine heather-gray jersey knit", "knit jersey heather tshirt", {"mode": "knit", "freq": 24}, (0.05, 0.05), ("tshirt", "hoodie")),
    ("berber_loop_carpet", (0.50, 0.44, 0.34), (0.58, 0.52, 0.42), None, "Flecked oatmeal berber loop carpet", "carpet berber loop", {"mode": "carpet", "freq": 16}, (0.3, 0.3), ("wall_to_wall_carpet", "office_floor")),
    ("charcoal_commercial_carpet", (0.07, 0.07, 0.075), (0.12, 0.12, 0.125), None, "Charcoal commercial carpet tile", "carpet commercial charcoal office", {"mode": "carpet", "freq": 24}, (0.5, 0.5), ("office_floor", "hotel_corridor_floor")),
    ("camel_suede", (0.36, 0.20, 0.10), (0.42, 0.25, 0.13), None, "Soft camel suede with brushed nap", "suede camel", {"mode": "velvet", "rough": 0.92}, (0.2, 0.2), ("jacket", "boots")),
    ("black_alcantara_microsuede", (0.025, 0.025, 0.025), (0.045, 0.045, 0.045), None, "Black microsuede for steering wheels and car interiors", "alcantara microsuede black automotive", {"mode": "velvet", "rough": 0.92}, (0.2, 0.2), ("steering_wheel", "car_headliner")),
]:
    roles = ("floor",) if "carpet" in _mid else ()
    _R("plush_textile", _mid, "plush_knit", C(_c1, _c2, _c3, **{"rough": 0.90, "slope": 6.0, **_kw}), _sz, _ph, _tg + " fabric textile", objs=_extra, roles=roles,
       axis=("v" if _kw.get("mode") == "corduroy" else None))

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("cognac_pebble_leather", (0.30, 0.12, 0.04), (0.42, 0.20, 0.08), (0.10, 0.04, 0.015), "Cognac pebble-grain leather", "leather cognac pebble", {"cells": 34}, (0.3, 0.3)),
    ("black_nappa_leather", (0.02, 0.02, 0.02), (0.05, 0.05, 0.05), (0.01, 0.01, 0.01), "Supple black nappa leather with fine grain", "leather black nappa", {"cells": 60, "rough": 0.35}, (0.3, 0.3)),
    ("oxblood_full_grain_leather", (0.16, 0.02, 0.02), (0.24, 0.05, 0.04), (0.05, 0.01, 0.01), "Oxblood full-grain leather", "leather oxblood burgundy", {"cells": 40}, (0.3, 0.3)),
    ("distressed_brown_leather", (0.18, 0.09, 0.04), (0.34, 0.20, 0.10), (0.06, 0.03, 0.015), "Distressed brown aviator leather", "leather distressed brown vintage", {"cells": 24, "rough": 0.55}, (0.3, 0.3)),
    ("crocodile_embossed_leather", (0.06, 0.08, 0.04), (0.12, 0.14, 0.08), (0.02, 0.02, 0.01), "Dark green crocodile-embossed leather", "leather croc embossed exotic", {"mode": "croc", "cells": 30}, (0.3, 0.3)),
    ("tan_reptile_scale_leather", (0.50, 0.44, 0.32), (0.62, 0.56, 0.42), (0.08, 0.06, 0.04), "Tan reptile-scale embossed leather", "leather reptile tan snake", {"mode": "croc", "cells": 45}, (0.2, 0.2)),
    ("braided_woven_leather", (0.24, 0.12, 0.05), (0.34, 0.19, 0.09), (0.05, 0.03, 0.015), "Hand-braided woven leather strips", "leather woven braided intrecciato", {"mode": "braided"}, (0.15, 0.15)),
    ("white_saffiano_leather", (0.72, 0.71, 0.68), (0.66, 0.65, 0.62), (0.40, 0.39, 0.37), "White saffiano cross-hatch leather", "leather white saffiano", {"cells": 80, "rough": 0.40}, (0.2, 0.2)),
]:
    _R("leather", _mid, "leather_hide", C(_c1, _c2, _c3, **{"mode": "pebble", "rough": 0.48, "slope": 6.0, **_kw}), _sz, _ph, _tg)

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz, _kind in [
    ("seagrass_basket_weave", (0.46, 0.42, 0.22), (0.54, 0.50, 0.28), (0.10, 0.09, 0.05), "Natural seagrass basket weave", "seagrass basket natural", {"freq": 8}, (0.3, 0.3), "natural"),
    ("dark_wicker_rattan", (0.22, 0.12, 0.05), (0.30, 0.18, 0.08), (0.05, 0.03, 0.015), "Dark-stained wicker rattan", "wicker rattan dark", {"freq": 12}, (0.3, 0.3), "natural"),
    ("white_painted_wicker", (0.72, 0.70, 0.66), (0.66, 0.64, 0.60), (0.10, 0.10, 0.10), "White-painted wicker for porch furniture", "wicker white porch", {"freq": 12}, (0.3, 0.3), "natural"),
    ("water_hyacinth_weave", (0.40, 0.30, 0.16), (0.50, 0.40, 0.22), (0.10, 0.07, 0.04), "Chunky water-hyacinth weave", "hyacinth weave chunky", {"freq": 6}, (0.4, 0.4), "natural"),
    ("washi_rice_paper", (0.80, 0.78, 0.72), (0.76, 0.73, 0.66), (0.60, 0.56, 0.48), "Translucent washi paper with long kozo fibers", "washi paper shoji japanese", {"mode": "paper", "slope": 3.0}, (0.5, 0.5), "paper"),
    ("kraft_paper_sheet", (0.46, 0.32, 0.17), (0.52, 0.38, 0.21), (0.30, 0.20, 0.10), "Brown kraft wrapping paper", "kraft paper brown packaging", {"mode": "paper", "slope": 3.0}, (0.5, 0.5), "paper"),
]:
    _R("natural_fiber", _mid, "natural_weave", C(_c1, _c2, _c3, **{"mode": "wicker", "rough": 0.78, "slope": 8.0, **_kw}), _sz, _ph, _tg, kind=_kind)

# ===========================================================================
# 7. NATURE & TERRAIN (40 parametric)
# ===========================================================================
for _mid, _c1, _c2, _ph, _tg, _kw in [
    ("desert_sand_ripples", (0.62, 0.44, 0.24), (0.70, 0.52, 0.30), "Wind-rippled golden desert sand", "desert dune golden", {"freq": 10}),
    ("white_beach_sand_ripples", (0.76, 0.72, 0.62), (0.80, 0.76, 0.66), "Fine white beach sand with gentle ripples", "beach white tropical", {"freq": 14}),
    ("wet_tidal_sand", (0.28, 0.24, 0.18), (0.36, 0.31, 0.24), "Wet tidal-flat sand with water-cut ripples", "wet tidal beach", {"freq": 8, "rough": 0.35}),
    ("black_volcanic_sand", (0.04, 0.04, 0.04), (0.07, 0.07, 0.07), "Black volcanic sand beach", "black volcanic iceland", {"freq": 12}),
]:
    _R("sand", _mid, "terrain_ground", C(_c1, _c2, (0.2, 0.2, 0.2), **{"mode": "sand_ripples", "rough": 0.88, "slope": 8.0, **_kw}), (1.5, 1.5), _ph, _tg + " sand terrain", axis="u")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz in [
    ("dry_cracked_mud", (0.42, 0.30, 0.18), (0.50, 0.37, 0.23), (0.08, 0.05, 0.03), "Sun-baked cracked mud plates", "mud cracked drought", {"cells": 7}, (1.5, 1.5)),
    ("salt_flat_crust", (0.80, 0.79, 0.76), (0.74, 0.73, 0.70), (0.40, 0.38, 0.36), "Polygonal salt-flat crust", "salt flat white desert", {"cells": 5}, (3.0, 3.0)),
    ("red_clay_desert_crackle", (0.42, 0.16, 0.07), (0.50, 0.22, 0.10), (0.12, 0.05, 0.02), "Red clay desert crackle", "red clay desert outback", {"cells": 10}, (1.5, 1.5)),
    ("gray_silt_mudcrack", (0.30, 0.29, 0.27), (0.36, 0.35, 0.33), (0.08, 0.08, 0.07), "Gray riverbed silt mudcracks", "silt gray riverbed", {"cells": 8}, (1.5, 1.5)),
]:
    _R("mud", _mid, "terrain_ground", C(_c1, _c2, _c3, **{"mode": "cracked_mud", "rough": 0.86, "slope": 11.0, **_kw}), _sz, _ph, _tg + " terrain ground")

for _mid, _c1, _c2, _c3, _ph, _tg in [
    ("pea_gravel_path", (0.42, 0.36, 0.28), (0.56, 0.50, 0.42), (0.12, 0.10, 0.08), "Rounded pea gravel in warm tans", "pea gravel path"),
    ("crushed_granite_gravel", (0.30, 0.30, 0.30), (0.46, 0.45, 0.44), (0.08, 0.08, 0.08), "Angular crushed granite gravel", "crushed granite gray"),
    ("red_lava_rock_gravel", (0.26, 0.07, 0.04), (0.36, 0.12, 0.06), (0.05, 0.02, 0.01), "Porous red lava rock landscaping gravel", "lava red landscaping"),
    ("white_marble_chip_gravel", (0.76, 0.76, 0.74), (0.68, 0.68, 0.66), (0.40, 0.40, 0.38), "White marble chip gravel for zen gardens", "marble white zen"),
]:
    _R("gravel", _mid, "terrain_ground", C(_c1, _c2, _c3, mode="gravel", rough=0.82, slope=13.0), (1.0, 1.0), _ph, _tg + " gravel ground")

for _mid, _c1, _c2, _c3, _ph, _tg in [
    ("frozen_lake_ice", (0.30, 0.46, 0.56), (0.62, 0.72, 0.78), (0.80, 0.84, 0.86), "Clear black lake ice with white fracture planes", "ice frozen lake winter"),
    ("glacier_blue_ice", (0.10, 0.32, 0.50), (0.40, 0.60, 0.72), (0.78, 0.86, 0.90), "Deep glacier-blue compressed ice", "glacier blue arctic"),
    ("hoarfrost_ice_sheet", (0.70, 0.74, 0.78), (0.82, 0.84, 0.86), (0.86, 0.88, 0.90), "Ice sheet blanketed in feathery hoarfrost", "frost hoarfrost white winter"),
]:
    _R("ice", _mid, "terrain_ground", C(_c1, _c2, _c3, mode="ice", rough=0.12, slope=5.0), (3.0, 3.0), _ph, _tg + " ice", kind="glass")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _extra in [
    ("fresh_snow_ground", (0.80, 0.82, 0.85), (0.84, 0.85, 0.86), (0.55, 0.60, 0.68), "Fresh powder snow with soft drifts", "snow white winter", {"cover": "snow", "cells": 32, "rough": 0.70}, ("snowfield", "ski_slope")),
    ("trampled_slush_snow", (0.62, 0.64, 0.66), (0.76, 0.77, 0.79), (0.30, 0.30, 0.30), "Trampled gray slush snow", "snow slush dirty winter", {"cover": "snow", "trample": 0.8, "cells": 24, "rough": 0.50}, ("winter_street", "parking_lot_edge")),
    ("green_lawn_grass", (0.06, 0.16, 0.03), (0.12, 0.26, 0.05), (0.03, 0.05, 0.015), "Lush green lawn grass", "grass green lawn", {"cover": "grass", "cells": 64}, ("park_lawn", "sports_field")),
    ("dry_savanna_grass", (0.42, 0.34, 0.14), (0.54, 0.44, 0.20), (0.18, 0.14, 0.07), "Dry golden savanna grass", "grass dry savanna golden", {"cover": "grass", "cells": 56}, ("prairie", "savanna")),
    ("forest_leaf_litter", (0.22, 0.10, 0.04), (0.34, 0.20, 0.07), (0.05, 0.03, 0.02), "Forest floor of decaying brown leaves", "forest leaves litter brown", {"cover": "leaves", "cells": 18}, ("forest_floor", "woodland_path")),
    ("autumn_maple_leaves", (0.46, 0.10, 0.02), (0.62, 0.30, 0.03), (0.10, 0.04, 0.02), "Freshly fallen red and orange maple leaves", "autumn leaves red orange fall", {"cover": "leaves", "cells": 14}, ("park_path", "forest_floor")),
    ("pine_needle_duff", (0.30, 0.15, 0.06), (0.38, 0.22, 0.10), (0.08, 0.05, 0.03), "Russet pine needle duff", "pine needles forest", {"cover": "needles", "cells": 48}, ("conifer_forest_floor",)),
    ("cushion_moss", (0.08, 0.18, 0.03), (0.16, 0.28, 0.05), (0.02, 0.04, 0.01), "Dense cushion moss", "moss green forest damp", {"cover": "moss", "cells": 40}, ("mossy_rock_top", "forest_floor")),
    ("tundra_reindeer_lichen", (0.56, 0.58, 0.50), (0.46, 0.48, 0.38), (0.12, 0.12, 0.08), "Pale reindeer lichen tundra", "lichen tundra arctic", {"cover": "lichen", "cells": 36}, ("tundra", "alpine_meadow")),
    ("rich_garden_soil", (0.07, 0.045, 0.03), (0.12, 0.08, 0.05), (0.03, 0.02, 0.015), "Dark crumbly garden soil", "soil dark garden earth", {"cover": "soil", "cells": 40}, ("garden_bed", "farm_field")),
    ("dry_dirt_trail", (0.36, 0.26, 0.16), (0.44, 0.33, 0.21), (0.18, 0.12, 0.07), "Compacted dry dirt hiking trail", "dirt trail dry", {"cover": "soil", "pebbles": 0.18, "cells": 32}, ("hiking_trail", "dirt_road")),
    ("red_laterite_soil", (0.40, 0.13, 0.05), (0.48, 0.18, 0.08), (0.16, 0.05, 0.02), "Iron-red laterite soil", "soil red laterite tropical", {"cover": "soil", "cells": 32}, ("dirt_road", "tropical_ground")),
    ("wood_chip_mulch", (0.24, 0.12, 0.05), (0.36, 0.20, 0.09), (0.06, 0.03, 0.015), "Shredded wood-chip garden mulch", "mulch wood chips garden", {"cover": "mulch", "cells": 26}, ("flower_bed", "playground_surface")),
    ("wet_peat_bog_mud", (0.05, 0.04, 0.025), (0.09, 0.07, 0.04), (0.02, 0.015, 0.01), "Wet black peat bog mud", "mud peat bog wet", {"cover": "soil", "pebbles": 0.02, "cells": 40, "rough": 0.30}, ("swamp", "bog")),
]:
    _R("ground_cover", _mid, "terrain_ground", C(_c1, _c2, _c3, **{"mode": "organic", "rough": 0.88, "slope": 10.0, **_kw}), (2.0, 2.0), _ph, _tg + " terrain ground", objs=_extra)

for _mid, _c1, _c2, _c3, _ph, _tg, _kw in [
    ("granite_cliff_face", (0.32, 0.31, 0.30), (0.44, 0.42, 0.40), (0.08, 0.08, 0.08), "Massive jointed granite cliff face", "granite cliff gray", {"layers": 3, "fracture": 6}),
    ("sandstone_canyon_strata", (0.52, 0.24, 0.10), (0.64, 0.36, 0.16), (0.20, 0.08, 0.03), "Banded red sandstone canyon wall", "sandstone canyon red", {"layers": 12, "fracture": 5}),
    ("limestone_karst_cliff", (0.60, 0.58, 0.52), (0.50, 0.48, 0.42), (0.20, 0.19, 0.16), "Weathered gray limestone karst cliff", "limestone karst gray", {"layers": 6, "fracture": 9}),
    ("volcanic_basalt_rock", (0.05, 0.05, 0.055), (0.09, 0.09, 0.095), (0.02, 0.02, 0.02), "Dark columnar volcanic basalt", "basalt volcanic black", {"layers": 2, "fracture": 8}),
    ("cooled_lava_flow", (0.04, 0.035, 0.03), (0.10, 0.06, 0.04), (0.01, 0.01, 0.01), "Cooled glassy lava flow crust", "lava volcanic flow", {"layers": 4, "fracture": 12}),
    ("layered_shale_cliff", (0.14, 0.14, 0.15), (0.22, 0.21, 0.21), (0.05, 0.05, 0.05), "Thinly layered dark shale cliff", "shale layered dark", {"layers": 18, "fracture": 7}),
    ("mossy_boulder_rock", (0.22, 0.22, 0.20), (0.14, 0.20, 0.06), (0.04, 0.06, 0.02), "Rounded boulder rock with moss patches", "boulder moss forest", {"layers": 2, "fracture": 5}),
    ("red_rock_mesa", (0.44, 0.14, 0.06), (0.56, 0.22, 0.10), (0.16, 0.05, 0.02), "Red-rock mesa cliff of the desert southwest", "mesa red desert", {"layers": 8, "fracture": 6}),
    ("white_chalk_cliff", (0.78, 0.77, 0.72), (0.70, 0.69, 0.64), (0.40, 0.38, 0.34), "Bright white chalk sea cliff with flint bands", "chalk white coastal", {"layers": 10, "fracture": 8}),
]:
    _R("rock_cliff", _mid, "rock_cliff", C(_c1, _c2, _c3, **{"rough": 0.80, "slope": 14.0, **_kw}), (4.0, 4.0), _ph, _tg + " rock cliff")

for _mid, _c1, _c2, _c3, _ph, _tg in [
    ("reed_thatch_roof", (0.40, 0.32, 0.16), (0.50, 0.40, 0.22), (0.10, 0.07, 0.04), "Water-reed thatch in stepped courses", "thatch reed cottage"),
    ("palm_frond_thatch", (0.36, 0.30, 0.12), (0.30, 0.26, 0.10), (0.08, 0.06, 0.03), "Dried palm-frond thatch", "thatch palm tropical tiki"),
]:
    _R("thatch", _mid, "natural_weave", C(_c1, _c2, _c3, mode="thatch", rough=0.86, slope=10.0), (1.0, 1.0), _ph, _tg + " roof")

# ===========================================================================
# 8. GLASS & OPTICS (33 parametric)
# ===========================================================================
_CLR1, _CLR2 = (0.80, 0.82, 0.82), (0.84, 0.86, 0.86)
for _mid, _c1, _c2, _ph, _tg, _kw, _sz in [
    ("clear_fluted_glass", _CLR1, _CLR2, "Clear fluted privacy glass", "clear fluted", {}, (0.3, 0.3)),
    ("bronze_fluted_glass", (0.42, 0.30, 0.18), (0.50, 0.36, 0.22), "Bronze-tinted fluted glass", "bronze tinted fluted", {}, (0.3, 0.3)),
    ("smoked_gray_fluted_glass", (0.20, 0.21, 0.22), (0.26, 0.27, 0.28), "Smoked gray fluted glass", "smoked gray fluted", {}, (0.3, 0.3)),
    ("wide_reeded_glass", _CLR1, _CLR2, "Wide-reed architectural glass", "wide reeded", {"freq": 6}, (0.3, 0.3)),
    ("micro_reeded_glass", _CLR1, _CLR2, "Micro-reeded cabinet glass", "micro reeded fine", {"freq": 30}, (0.2, 0.2)),
    ("green_fluted_glass", (0.38, 0.56, 0.44), (0.44, 0.62, 0.50), "Pale green fluted glass", "green fluted", {}, (0.3, 0.3)),
    ("amber_ribbed_glass", (0.66, 0.38, 0.08), (0.72, 0.44, 0.12), "Retro amber ribbed glass", "amber ribbed retro", {"freq": 10}, (0.3, 0.3)),
    ("satin_fluted_glass", (0.76, 0.78, 0.78), (0.80, 0.82, 0.82), "Satin-frosted fluted glass", "frosted satin fluted", {"rough": 0.35}, (0.3, 0.3)),
]:
    _R("fluted_glass", _mid, "glass_optics", C(_c1, _c2, (0.5, 0.5, 0.5), **{"mode": "fluted", "freq": 14, "rough": 0.06, "slope": 8.0, **_kw}), _sz, _ph, _tg + " glass", axis="v")

for _mid, _c1, _c2, _ph, _tg, _kw, _sz in [
    ("clear_wave_glass_block", (0.70, 0.76, 0.78), (0.78, 0.82, 0.84), "Clear wave-pattern glass block", "clear wave", {}, (0.8, 0.8)),
    ("frosted_glass_block", (0.72, 0.74, 0.75), (0.76, 0.78, 0.79), "Frosted-face glass block", "frosted", {"rough": 0.30}, (0.8, 0.8)),
    ("blue_tinted_glass_block", (0.30, 0.48, 0.62), (0.40, 0.56, 0.70), "Blue-tinted glass block", "blue tinted", {}, (0.8, 0.8)),
    ("small_glass_brick_wall", (0.70, 0.76, 0.78), (0.78, 0.82, 0.84), "Small-format glass bricks", "small brick", {"nx": 6, "ny": 6}, (0.9, 0.9)),
    ("rectangular_glass_block", (0.70, 0.76, 0.78), (0.78, 0.82, 0.84), "Rectangular half-height glass blocks", "rectangular", {"ny": 8}, (0.8, 0.8)),
    ("amber_glass_block", (0.64, 0.40, 0.12), (0.70, 0.46, 0.16), "Amber retro glass block", "amber retro", {}, (0.8, 0.8)),
]:
    _R("glass_block", _mid, "glass_optics", C(_c1, _c2, (0.50, 0.50, 0.48), **{"mode": "glass_block", "nx": 4, "ny": 4, "rough": 0.10, "slope": 8.0, **_kw}), _sz, _ph, _tg + " glass block")

for _mid, _c1, _c2, _ph, _tg, _kw in [
    ("pebbled_obscure_glass", (0.78, 0.80, 0.80), (0.82, 0.84, 0.84), "Pebbled obscure bathroom glass", "pebbled obscure", {"freq": 24}),
    ("hammered_cathedral_glass", (0.74, 0.78, 0.78), (0.80, 0.84, 0.84), "Clear hammered cathedral glass", "hammered cathedral", {"freq": 12}),
    ("ruby_cathedral_glass", (0.50, 0.03, 0.05), (0.62, 0.06, 0.08), "Ruby-red cathedral art glass", "ruby red stained church", {"freq": 14}),
    ("cobalt_cathedral_glass", (0.03, 0.08, 0.45), (0.06, 0.14, 0.55), "Cobalt-blue cathedral art glass", "cobalt blue stained church", {"freq": 14}),
    ("aqua_seedy_glass", (0.50, 0.70, 0.62), (0.58, 0.76, 0.68), "Aqua seedy glass with fine bubbles", "aqua seedy antique", {"freq": 40}),
    ("rain_pattern_glass", (0.78, 0.80, 0.80), (0.82, 0.84, 0.84), "Rain-pattern rolled glass", "rain rolled", {"freq": 18, "slope": 4.0}),
]:
    _R("textured_glass", _mid, "glass_optics", C(_c1, _c2, (0.5, 0.5, 0.5), **{"mode": "pebbled", "rough": 0.08, "slope": 7.0, **_kw}), (0.3, 0.3), _ph, _tg + " glass textured")

for _mid, _c1, _ph, _tg, _kw in [
    ("acid_etched_glass", (0.78, 0.80, 0.80), "Uniform acid-etched satin glass", "etched satin", {"slope": 3.0, "rough": 0.42}),
    ("sandblasted_privacy_glass", (0.78, 0.80, 0.80), "Coarse sandblasted privacy glass", "sandblasted privacy", {"slope": 5.0, "rough": 0.55}),
    ("frosted_bathroom_glass", (0.76, 0.80, 0.80), "Frosted bathroom window glass", "frosted bathroom", {"slope": 4.0, "rough": 0.45}),
    ("satin_laminated_glass", (0.82, 0.82, 0.80), "White satin laminated glass", "laminated satin white", {"slope": 1.5, "rough": 0.30}),
    ("frosted_acrylic_panel", (0.80, 0.80, 0.80), "Frosted acrylic diffuser sheet", "acrylic frosted plastic", {"slope": 2.5, "rough": 0.38}),
    ("opal_milk_glass", (0.84, 0.84, 0.82), "Opal milk glass for globe lamps", "opal milk lamp", {"slope": 1.0, "rough": 0.15}),
]:
    _R("frosted_glass", _mid, "glass_optics", C(_c1, tuple(min(0.86, c + 0.02) for c in _c1), (0.5, 0.5, 0.5), **{"mode": "frosted", **_kw}), (0.5, 0.5), _ph, _tg + " glass frosted")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw in [
    ("georgian_wired_glass", (0.78, 0.80, 0.80), (0.80, 0.82, 0.82), (0.25, 0.25, 0.25), "Georgian wired fire-safety glass", "wired georgian safety", {"freq": 12}),
    ("frosted_wired_safety_glass", (0.74, 0.76, 0.76), (0.78, 0.80, 0.80), (0.25, 0.25, 0.25), "Frosted wired safety glass", "wired frosted safety", {"freq": 10, "rough": 0.35}),
    ("green_tinted_wired_glass", (0.40, 0.50, 0.44), (0.44, 0.54, 0.48), (0.15, 0.15, 0.15), "Green-tinted industrial wired glass", "wired green industrial", {"freq": 14}),
]:
    _R("wired_glass", _mid, "glass_optics", C(_c1, _c2, _c3, **{"mode": "wired", "rough": 0.10, "slope": 4.0, **_kw}), (0.3, 0.3), _ph, _tg + " glass")

for _mid, _c1, _ph, _tg in [
    ("clear_float_glazing", (0.82, 0.84, 0.84), "Clear low-iron float glass glazing", "clear float window"),
    ("low_e_blue_green_glazing", (0.36, 0.52, 0.50), "Blue-green low-E coated curtain-wall glazing", "lowe bluegreen curtainwall"),
    ("bronze_tinted_glazing", (0.36, 0.26, 0.16), "Bronze-tinted solar-control glazing", "bronze tinted solar"),
    ("gray_solar_glazing", (0.18, 0.19, 0.20), "Dark gray solar-control glazing", "gray tinted solar privacy"),
]:
    _R("glazing", _mid, "glass_optics", C(_c1, tuple(min(0.86, c + 0.01) for c in _c1), (0.5, 0.5, 0.5), mode="frosted", slope=0.3, rough=0.03), (2.0, 2.0), _ph, _tg + " glass glazing")

# ===========================================================================
# 9. RUBBER, POLYMERS & ROADS (33 parametric)
# ===========================================================================
for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz, _extra in [
    ("fresh_black_asphalt", (0.03, 0.03, 0.03), (0.06, 0.06, 0.06), (0.015, 0.015, 0.015), "Freshly laid jet-black asphalt", "asphalt black fresh road", {}, (1.2, 1.2), ()),
    ("worn_gray_asphalt", (0.12, 0.12, 0.12), (0.20, 0.20, 0.19), (0.05, 0.05, 0.05), "Sun-faded gray asphalt with exposed aggregate", "asphalt gray worn old road", {}, (1.2, 1.2), ()),
    ("red_bike_lane_asphalt", (0.22, 0.05, 0.04), (0.30, 0.08, 0.06), (0.08, 0.02, 0.015), "Red-pigmented bike-lane asphalt", "asphalt red bike_lane", {}, (1.2, 1.2), ("bike_lane", "bus_lane")),
    ("coarse_chip_seal_road", (0.08, 0.08, 0.08), (0.26, 0.25, 0.24), (0.02, 0.02, 0.02), "Coarse chip-seal country road", "chipseal rural road coarse", {"slope": 12.0}, (1.2, 1.2), ("country_road",)),
    ("sealcoat_parking_tarmac", (0.04, 0.04, 0.045), (0.08, 0.08, 0.085), (0.02, 0.02, 0.02), "Sealcoated parking-lot tarmac", "tarmac parking sealcoat", {"rough": 0.70}, (1.2, 1.2), ("parking_lot",)),
    ("rubberized_running_track", (0.30, 0.07, 0.04), (0.36, 0.10, 0.06), (0.10, 0.03, 0.02), "Rubberized EPDM athletics running track", "track running epdm sports", {"rough": 0.80}, (0.6, 0.6), ("running_track", "playground_surface")),
    ("wet_night_asphalt", (0.025, 0.025, 0.03), (0.05, 0.05, 0.055), (0.01, 0.01, 0.01), "Rain-slicked asphalt with glossy wet sheen", "asphalt wet rain night", {"rough": 0.32}, (1.2, 1.2), ("rainy_street",)),
]:
    _R("asphalt", _mid, "rubber_polymer", C(_c1, _c2, _c3, **{"mode": "asphalt", "rough": 0.86, "slope": 9.0, **_kw}), _sz, _ph, _tg, kind="asphalt", objs=_extra)

for _mid, _ph, _tg, _kw, _sz in [
    ("truck_tire_tread", "Heavy truck tire with chevron lugs", "tire truck", {"freq": 6}, (0.3, 0.3)),
    ("offroad_mud_tire_tread", "Aggressive off-road mud tire tread", "tire offroad mud", {"freq": 4}, (0.3, 0.3)),
    ("fine_bicycle_tire_tread", "Fine bicycle tire tread", "tire bicycle fine", {"freq": 16}, (0.1, 0.1)),
    ("tractor_lug_tread", "Deep tractor lug tread", "tire tractor lug farm", {"freq": 3}, (0.5, 0.5)),
]:
    _R("tire_tread", _mid, "rubber_polymer", C((0.03, 0.03, 0.03), (0.05, 0.05, 0.05), (0.01, 0.01, 0.01), **{"mode": "tire_tread", "rough": 0.80, "slope": 11.0, **_kw}), _sz, _ph, _tg + " rubber")

for _mid, _c1, _c2, _ph, _tg, _kw, _sz, _kind in [
    ("glossy_twill_carbon_fiber", (0.02, 0.02, 0.022), (0.07, 0.07, 0.075), "Glossy 2x2 twill carbon fiber", "carbon fiber twill glossy", {"freq": 12}, (0.1, 0.1), "carbon"),
    ("forged_carbon_composite", (0.02, 0.02, 0.02), (0.10, 0.10, 0.10), "Forged carbon with marbled chopped tows", "carbon forged marbled", {"forged": True}, (0.15, 0.15), "carbon"),
    ("kevlar_aramid_weave", (0.62, 0.46, 0.10), (0.70, 0.54, 0.16), "Golden kevlar aramid weave", "kevlar aramid golden", {"freq": 10, "rough": 0.40}, (0.1, 0.1), "carbon"),
    ("blue_carbon_weave", (0.02, 0.03, 0.08), (0.05, 0.08, 0.20), "Blue-tinted carbon weave", "carbon blue", {"freq": 12}, (0.1, 0.1), "carbon"),
    ("fiberglass_weave_cloth", (0.70, 0.70, 0.66), (0.76, 0.76, 0.72), "Woven fiberglass cloth under resin", "fiberglass weave", {"freq": 16, "rough": 0.35}, (0.1, 0.1), "carbon"),
]:
    _R("composite_weave", _mid, "rubber_polymer", C(_c1, _c2, (0.01, 0.01, 0.01), **{"mode": "carbon_weave", "rough": 0.16, "slope": 4.0, **_kw}), _sz, _ph, _tg + " composite", kind=_kind)

for _mid, _c1, _c2, _ph, _tg, _sz in [
    ("open_cell_gray_foam", (0.30, 0.30, 0.30), (0.38, 0.38, 0.38), "Open-cell gray polyurethane foam", "foam gray opencell", (0.1, 0.1)),
    ("white_memory_foam", (0.74, 0.74, 0.70), (0.80, 0.80, 0.76), "White visco-elastic memory foam", "foam memory white", (0.1, 0.1)),
    ("black_eva_foam_mat", (0.04, 0.04, 0.04), (0.07, 0.07, 0.07), "Black closed-cell EVA foam", "foam eva black", (0.2, 0.2)),
]:
    _R("foam", _mid, "rubber_polymer", C(_c1, _c2, (0.05, 0.05, 0.05), mode="foam", rough=0.92, slope=6.0), _sz, _ph, _tg)

for _mid, _c1, _c2, _ph, _tg, _kw in [
    ("matte_black_abs_plastic", (0.03, 0.03, 0.03), (0.045, 0.045, 0.045), "Matte black ABS with mold-tech stipple", "abs black matte plastic", {"rough": 0.55}),
    ("white_appliance_plastic", (0.78, 0.78, 0.76), (0.80, 0.80, 0.78), "Satin white appliance plastic", "plastic white appliance", {"rough": 0.35}),
    ("glossy_red_polymer", (0.50, 0.02, 0.02), (0.56, 0.04, 0.03), "Glossy red molded polymer", "plastic red glossy toy", {"rough": 0.18}),
    ("teal_food_grade_silicone", (0.06, 0.32, 0.30), (0.08, 0.38, 0.36), "Soft-touch teal food-grade silicone", "silicone teal kitchen", {"rough": 0.45}),
    ("textured_dashboard_vinyl", (0.08, 0.08, 0.085), (0.11, 0.11, 0.115), "Grained automotive dashboard vinyl", "vinyl dashboard automotive", {"rough": 0.62, "slope": 5.0}),
    ("white_hdpe_cutting_board", (0.74, 0.74, 0.72), (0.70, 0.70, 0.68), "White HDPE polymer cutting board", "hdpe white kitchen", {"rough": 0.50}),
    ("safety_yellow_polycarbonate", (0.70, 0.50, 0.02), (0.76, 0.56, 0.04), "Safety-yellow polycarbonate housing", "polycarbonate yellow safety", {"rough": 0.30}),
    ("gray_recycled_plastic_lumber", (0.24, 0.24, 0.23), (0.32, 0.32, 0.31), "Gray recycled plastic lumber", "recycled plastic lumber outdoor", {"rough": 0.60}),
]:
    _R("molded_plastic", _mid, "rubber_polymer", C(_c1, _c2, (0.02, 0.02, 0.02), **{"mode": "molded", "slope": 3.0, **_kw}), (0.1, 0.1), _ph, _tg)

_BLK = ((0.04, 0.04, 0.04), (0.06, 0.06, 0.06), (0.02, 0.02, 0.02))
for _mid, _fam, _p, _sz, _ph, _tg in [
    ("coin_stud_rubber_flooring", "mosaic_tile", C(*_BLK, mode="penny", freq=8, grout=0.60, rough=0.72, slope=9.0), (0.5, 0.5), "Raised coin-stud rubber flooring", "stud coin rubber floor"),
    ("ribbed_rubber_runner", "machined_grip", C(*_BLK, mode="flutes", freq=24, rough=0.74, slope=8.0), (0.3, 0.3), "Fine-ribbed rubber runner matting", "ribbed rubber runner"),
    ("diamond_rubber_mat", "machined_grip", C(*_BLK, mode="tread", freq=16, rough=0.74, slope=9.0), (0.4, 0.4), "Diamond-plate rubber floor mat", "diamond rubber mat truck"),
    ("interlocking_gym_tile", "grid_tile", C((0.05, 0.05, 0.05), (0.07, 0.07, 0.07), (0.02, 0.02, 0.02), nx=2, ny=2, grout=0.02, bevel=0.04, rough=0.80, slope=6.0, grout_rmod=0.10), (1.2, 1.2), "Interlocking rubber gym floor tiles", "gym rubber tile fitness"),
    ("epdm_playground_granules", "speckled_aggregate", C((0.08, 0.08, 0.08), (0.30, 0.08, 0.05), (0.08, 0.20, 0.10), cells_a=40, cells_b=80, chip_fill=0.60, rough=0.86, slope=6.0), (0.5, 0.5), "Bonded EPDM rubber granule safety surface", "epdm granules playground"),
    ("anti_fatigue_bubble_mat", "mosaic_tile", C(*_BLK, mode="hex", freq=10, grout=0.30, rough=0.74, slope=8.0), (0.5, 0.5), "Hex-bubble anti-fatigue rubber mat", "antifatigue hex rubber mat"),
]:
    _R("rubber_mat", _mid, _fam, _p, _sz, _ph, _tg, kind="rubber")

# ===========================================================================
# 10. ROOFING, TECH & SCI-FI (36 parametric)
# ===========================================================================
for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz, _kind in [
    ("natural_slate_roof", (0.07, 0.075, 0.08), (0.12, 0.125, 0.13), (0.02, 0.02, 0.02), "Natural blue-black slate roof", "slate roof dark", {"nx": 5, "ny": 8}, (1.5, 1.2), "stone"),
    ("welsh_purple_slate_roof", (0.10, 0.08, 0.11), (0.15, 0.12, 0.15), (0.03, 0.02, 0.03), "Welsh heather-purple slate roof", "slate welsh purple heritage", {"nx": 5, "ny": 8}, (1.5, 1.2), "stone"),
    ("cedar_shake_roof", (0.40, 0.22, 0.10), (0.50, 0.30, 0.14), (0.08, 0.04, 0.02), "Hand-split cedar shake roof", "cedar shake wood roof", {"nx": 6, "ny": 6, "rough": 0.80}, (1.2, 1.2), "wood"),
    ("weathered_gray_shake_roof", (0.28, 0.27, 0.25), (0.38, 0.36, 0.33), (0.06, 0.06, 0.05), "Silvered weathered cedar shakes", "shake weathered gray coastal", {"nx": 6, "ny": 6, "rough": 0.84}, (1.2, 1.2), "wood"),
    ("charcoal_asphalt_shingle", (0.05, 0.05, 0.05), (0.10, 0.10, 0.10), (0.02, 0.02, 0.02), "Charcoal architectural asphalt shingles", "asphalt shingle charcoal suburban", {"nx": 4, "ny": 8, "rough": 0.88}, (1.2, 1.0), "asphalt"),
    ("weathered_wood_asphalt_shingle", (0.16, 0.12, 0.09), (0.24, 0.18, 0.13), (0.04, 0.03, 0.02), "Weathered-wood blend asphalt shingles", "asphalt shingle brown", {"nx": 4, "ny": 8, "rough": 0.88}, (1.2, 1.0), "asphalt"),
    ("spanish_barrel_clay_roof", (0.48, 0.18, 0.07), (0.56, 0.26, 0.12), (0.20, 0.08, 0.03), "Spanish terracotta barrel roof tiles", "barrel clay spanish mediterranean", {"mode": "barrel_clay", "nx": 5, "ny": 6}, (1.25, 1.5), "ceramic"),
    ("aged_mission_barrel_tile", (0.40, 0.20, 0.10), (0.50, 0.30, 0.16), (0.12, 0.08, 0.04), "Aged mission barrel tiles with blotchy patina", "barrel mission aged", {"mode": "barrel_clay", "nx": 5, "ny": 6}, (1.25, 1.5), "ceramic"),
    ("terracotta_fishscale_roof", (0.46, 0.18, 0.08), (0.52, 0.24, 0.11), (0.14, 0.06, 0.03), "Scalloped terracotta fishscale roof tiles", "fishscale scalloped terracotta", {"mode": "scalloped", "nx": 6, "ny": 8}, (1.2, 1.2), "ceramic"),
    ("green_glazed_fishscale_roof", (0.06, 0.22, 0.10), (0.10, 0.30, 0.14), (0.02, 0.06, 0.03), "Glazed green fishscale roof tiles", "fishscale glazed green", {"mode": "scalloped", "nx": 6, "ny": 8, "rough": 0.20}, (1.2, 1.2), "ceramic"),
    ("copper_fishscale_shingles", (0.84, 0.44, 0.30), (0.90, 0.52, 0.36), (0.20, 0.10, 0.06), "Copper fishscale shingles for turrets and domes", "copper fishscale turret dome", {"mode": "scalloped", "nx": 6, "ny": 8, "metallic": 0.90, "rough": 0.30}, (1.2, 1.2), "copper"),
    ("blue_glazed_barrel_tile", (0.04, 0.10, 0.30), (0.06, 0.14, 0.38), (0.02, 0.03, 0.08), "Cobalt glazed barrel tiles of an East Asian temple", "barrel glazed blue temple asian", {"mode": "barrel_clay", "nx": 5, "ny": 6, "rough": 0.20}, (1.25, 1.5), "ceramic"),
]:
    _R("roofing", _mid, "roofing_shingles", C(_c1, _c2, _c3, **{"mode": "asphalt_shingle", "rough": 0.74, "slope": 11.0, **_kw}), _sz, _ph, _tg + " roof", kind=_kind)

for _mid, _c1, _c2, _c3, _ph, _tg in [
    ("blue_pcb_solder_mask", (0.02, 0.05, 0.25), (0.04, 0.10, 0.40), (0.85, 0.70, 0.40), "Blue solder-mask PCB with ENIG gold pads", "pcb blue electronics"),
    ("black_pcb_solder_mask", (0.015, 0.015, 0.018), (0.04, 0.04, 0.045), (0.85, 0.70, 0.40), "Matte black PCB with gold pads", "pcb black electronics"),
    ("red_pcb_hasl", (0.30, 0.02, 0.02), (0.42, 0.04, 0.03), (0.75, 0.76, 0.77), "Red PCB with silver HASL pads", "pcb red electronics"),
    ("purple_pcb_enig", (0.10, 0.02, 0.18), (0.16, 0.04, 0.28), (0.88, 0.72, 0.42), "Purple hobbyist PCB with ENIG pads", "pcb purple electronics"),
]:
    _R("pcb", _mid, "tech_panel", C(_c1, _c2, _c3, mode="pcb", rough=0.22, slope=6.0), (0.1, 0.1), _ph, _tg + " circuit tech")

for _mid, _c1, _c2, _c3, _ph, _tg in [
    ("polycrystalline_solar_array", (0.02, 0.06, 0.22), (0.06, 0.14, 0.36), (0.75, 0.76, 0.78), "Blue polycrystalline solar cells", "solar polycrystalline blue"),
    ("black_monocrystalline_solar", (0.01, 0.012, 0.02), (0.03, 0.035, 0.05), (0.40, 0.40, 0.42), "All-black monocrystalline solar module", "solar monocrystalline black"),
    ("thin_film_solar_panel", (0.04, 0.03, 0.06), (0.07, 0.06, 0.10), (0.20, 0.20, 0.22), "Dark thin-film solar module", "solar thinfilm"),
    ("bifacial_glass_solar_module", (0.02, 0.05, 0.16), (0.04, 0.09, 0.24), (0.80, 0.80, 0.82), "Bifacial glass-glass solar module", "solar bifacial"),
]:
    _R("solar", _mid, "tech_panel", C(_c1, _c2, _c3, mode="solar", rough=0.14, slope=5.0), (0.62, 0.62), _ph, _tg + " energy tech")

for _mid, _c1, _c2, _c3, _ph, _tg, _kw in [
    ("gunmetal_scifi_hull", (0.20, 0.21, 0.23), (0.30, 0.31, 0.33), (0.04, 0.04, 0.045), "Gunmetal starship hull plating with bevel seams", "scifi hull gunmetal spaceship", {"metallic": 0.90}),
    ("white_starship_armor", (0.74, 0.75, 0.76), (0.80, 0.81, 0.82), (0.10, 0.10, 0.11), "Painted white starship armor panels", "scifi armor white clean", {"metallic": 0.0, "rough": 0.35}),
    ("hazard_orange_mech_plate", (0.70, 0.22, 0.02), (0.76, 0.28, 0.04), (0.03, 0.03, 0.03), "Hazard-orange painted mech armor plates", "scifi mech orange industrial", {"metallic": 0.0, "rough": 0.40}),
    ("olive_military_armor_panel", (0.10, 0.12, 0.06), (0.14, 0.16, 0.08), (0.03, 0.03, 0.02), "Olive military vehicle armor panel", "military armor olive vehicle", {"metallic": 0.0, "rough": 0.60}),
    ("scifi_corridor_floor_panel", (0.30, 0.31, 0.33), (0.40, 0.41, 0.43), (0.05, 0.05, 0.05), "Modular sci-fi corridor floor panels", "scifi corridor floor station", {"nx": 4, "ny": 4, "metallic": 0.85}),
    ("titanium_tech_panel", (0.52, 0.49, 0.45), (0.60, 0.57, 0.52), (0.08, 0.08, 0.08), "Brushed titanium tech panels", "titanium tech aerospace", {"nx": 3, "ny": 3, "metallic": 0.95}),
    ("matte_black_stealth_panel", (0.025, 0.025, 0.03), (0.04, 0.04, 0.045), (0.01, 0.01, 0.01), "Matte black stealth composite panels", "stealth black panel", {"nx": 2, "ny": 4, "metallic": 0.20, "rough": 0.55}),
    ("space_station_white_tile", (0.72, 0.72, 0.70), (0.76, 0.76, 0.74), (0.20, 0.20, 0.20), "White space-station interior wall tiles", "space station white interior", {"nx": 4, "ny": 4, "metallic": 0.0, "rough": 0.40}),
    ("gold_thermal_foil_panel", (0.90, 0.68, 0.28), (0.95, 0.78, 0.36), (0.30, 0.20, 0.06), "Gold thermal-foil satellite blanket panels", "satellite gold foil space", {"nx": 3, "ny": 3, "metallic": 0.95, "rough": 0.30}),
    ("cyberpunk_teal_hull", (0.04, 0.30, 0.32), (0.06, 0.40, 0.42), (0.01, 0.02, 0.02), "Teal metallic cyberpunk hull plates", "cyberpunk teal hull", {"metallic": 0.80}),
]:
    _R("scifi_panel", _mid, "tech_panel", C(_c1, _c2, _c3, **{"mode": "scifi_hull", "nx": 2, "ny": 2, "rough": 0.34, "slope": 8.0, **_kw}), (2.0, 2.0), _ph, _tg + " scifi sci-fi futuristic",
       kind=("metal" if _kw.get("metallic", 0.0) > 0.5 else "rubber"))

for _mid, _c1, _c2, _c3, _ph, _tg, _kw, _sz, _kind in [
    ("aluminum_heatsink_fins", (0.70, 0.71, 0.72), (0.80, 0.81, 0.82), (0.10, 0.10, 0.10), "Extruded aluminum heatsink fins", "heatsink aluminum electronics", {"freq": 16}, (0.15, 0.15), "metal"),
    ("black_anodized_heatsink", (0.04, 0.04, 0.045), (0.08, 0.08, 0.085), (0.01, 0.01, 0.01), "Black anodized heatsink fin array", "heatsink black anodized", {"freq": 12, "metallic": 0.85}, (0.15, 0.15), "metal"),
    ("copper_heatpipe_fins", (0.84, 0.44, 0.30), (0.90, 0.52, 0.36), (0.20, 0.10, 0.06), "Dense copper heat-pipe fin stack", "heatsink copper", {"freq": 24}, (0.1, 0.1), "copper"),
]:
    _R("heatsink", _mid, "machined_grip", C(_c1, _c2, _c3, **{"mode": "flutes", "metallic": 0.95, "rough": 0.30, "slope": 10.0, **_kw}), _sz, _ph, _tg + " fins", kind=_kind, axis="v")

for _mid, _c1, _c2, _ph, _tg, _kw in [
    ("charcoal_wedge_acoustic_foam", (0.05, 0.05, 0.055), (0.08, 0.08, 0.085), "Charcoal wedge acoustic foam", "acoustic wedge charcoal studio", {"mode": "flutes", "freq": 8}),
    ("egg_crate_acoustic_foam", (0.20, 0.20, 0.21), (0.26, 0.26, 0.27), "Gray egg-crate acoustic foam", "acoustic eggcrate gray", {"mode": "knurl", "freq": 6}),
    ("blue_pyramid_acoustic_foam", (0.04, 0.10, 0.30), (0.06, 0.14, 0.38), "Blue pyramid acoustic foam panels", "acoustic pyramid blue", {"mode": "knurl", "freq": 8}),
]:
    _R("acoustic_foam", _mid, "machined_grip", C(_c1, _c2, (0.02, 0.02, 0.02), **{"metallic": 0.0, "rough": 0.95, "slope": 9.0, **_kw}), (0.5, 0.5), _ph, _tg + " foam", kind="rubber")


# Border-offset rolls (fraction of tile) for patterns whose lattice lines fall exactly on the wrap border
_ROLLS: dict[str, tuple[float, float]] = {
    "chevron_brick_paver": (0.125, 0.0), "oak_chevron_parquet": (0.125, 0.0), "smoked_oak_chevron_parquet": (0.125, 0.0),
    "white_oiled_chevron_parquet": (0.5 / 6, 0.0), "cherry_chevron_parquet": (0.125, 0.0),
    "teak_louver_screen": (0.075, 0.0), "white_picket_fence": (0.75 / 8, 0.0), "white_ash_batten_ceiling": (0.0, 0.075),
    "bamboo_pole_screen": (0.75 / 14, 0.0), "aspen_sauna_slats": (0.0, 0.0625), "pine_tongue_groove_ceiling": (0.0, 0.065),
    "maple_butcher_block_counter": (0.0, 0.065), "brushed_titanium_seam_panel": (0.25, 0.25),
    "green_tinted_wired_glass": (0.5 / 14, 0.5 / 14), "dry_cracked_mud": (0.31, 0.17), "salt_flat_crust": (0.31, 0.17), "micro_reeded_glass": (0.025, 0.0), "water_hyacinth_weave": (0.5 / 12, 0.5 / 12),
}
_ROLLS.update({
    "round_perf_stainless_panel": (0.75 / 16, 0.75 / 16), "perforated_black_steel_screen": (0.75 / 20, 0.75 / 20),
    "slotted_bronze_grille": (0.75 / 6, 0.75 / 12), "old_town_cobblestone": (0.31, 0.17), "bluestone_flagstone": (0.23, 0.41),
    "charcoal_terrazzo": (0.31, 0.17),
})
for _row in _ROWS:
    if _row["id"] in _ROLLS:
        _row["params"]["roll"] = list(_ROLLS[_row["id"]])
    elif _row["family"] == "wood_planks" and _row["params"].get("planks", 6) >= 2:
        _row["params"]["roll"] = "auto_planks"
    elif _row["family"] == "herringbone_chevron" and not _row["params"].get("chevron"):
        # Put the 45-degree herringbone seams off the wrap border (quarter lattice cell)
        _q = round(0.25 / float(_row["params"].get("freq", 4)), 6)
        _row["params"]["roll"] = [_q, _q]


def _plank_roll(params: dict[str, Any], seed: int) -> list[float]:
    """Border mid-board (away from side grooves) and maximally far from every row's end joints."""
    import numpy as np

    from .atlas_generators import _cell_hash

    planks = float(params.get("planks", 6))
    joints = float(params.get("joints", 2))
    across = round(0.65 / planks, 6)
    along = 0.0
    if joints >= 1.0:
        rows = np.arange(int(planks), dtype=np.float32)
        j_off = _cell_hash(rows, rows * 0.0, planks, 1.0, seed + 41).astype(np.float64)
        cand = np.arange(256) / 256.0
        # joint lines (in the along axis) for row r: uu = (k - j_off[r]) / joints
        d = ((cand[:, None] * joints + j_off[None, :]) % 1.0)
        dist = np.minimum(d, 1.0 - d).min(axis=1)
        along = float(-cand[int(np.argmax(dist))] % 1.0)
    return [round(across, 6), round(along, 6)] if params.get("vertical") else [round(along, 6), round(across, 6)]


# ---------------------------------------------------------------------------
# Flagship (hand-tuned) materials: (id, category, subcategory, size_m, extra objects, tags, anisotropic axis)
# ---------------------------------------------------------------------------
FLAGSHIP_META: list[tuple[str, str, str, tuple[float, float], tuple[str, ...], str, str]] = [
    ("corrugated_cardboard", _WT, "engineered_wood", (0.3, 0.3), ("shipping_box", "packaging"), "cardboard kraft paper", "none"),
    ("knurled_metal_grip", _MI, "machined_metal", (0.05, 0.05), ("flashlight_body", "tool_handle"), "knurl grip steel", "none"),
    ("woven_denim_twill", _FU, "woven_textile", (0.05, 0.05), ("jeans", "jacket"), "denim indigo twill", "none"),
    ("quilted_leather_stitch", _FU, "leather", (0.4, 0.4), ("booth_seat", "handbag"), "leather quilted diamond", "none"),
    ("acoustic_pegboard", _WT, "engineered_wood", (0.6, 0.6), ("workshop_wall", "tool_wall"), "pegboard acoustic workshop", "none"),
    ("sidewalk_expansion_joint", _SM, "concrete", (3.0, 3.0), ("sidewalk", "plaza"), "sidewalk broom concrete", "none"),
    ("subway_metro_tile", _TC, "subway_tile", (0.6, 0.6), ("metro_station_wall",), "subway white metro", "none"),
    ("rubber_tread_stair", _RP, "rubber_mat", (0.5, 0.5), ("stair_tread", "metro_floor"), "rubber coin stud", "none"),
    ("ribbed_corduroy_velvet", _FU, "plush_textile", (0.1, 0.1), ("trousers", "jacket"), "corduroy velvet ribbed", "v"),
    ("waffle_radiator_fins", _MI, "heatsink", (0.5, 0.5), ("radiator", "hvac_unit"), "radiator cast iron fins", "none"),
    ("wood_parquet_herringbone", _WT, "parquet", (1.2, 1.2), (), "oak herringbone parquet", "none"),
    ("carbon_fiber_weave", _RP, "composite_weave", (0.1, 0.1), (), "carbon fiber twill", "none"),
    ("street_storm_grate", _MI, "perforated_metal", (1.0, 1.0), ("storm_drain", "trench_grate"), "cast iron grate drain street", "none"),
    ("diamond_tread_plate", _MI, "machined_metal", (0.6, 0.6), ("truck_bed", "industrial_floor"), "tread diamond checkerplate steel", "none"),
    ("hvac_slotted_vent", _MI, "perforated_metal", (0.6, 0.6), ("air_vent", "return_grille"), "hvac vent louver", "none"),
    ("speaker_mesh_hex", _MI, "perforated_metal", (0.1, 0.1), ("speaker_grille", "laptop_vent"), "speaker mesh hex", "none"),
    ("expanded_metal_mesh", _MI, "perforated_metal", (0.5, 0.5), ("security_screen", "walkway"), "expanded mesh diamond", "none"),
    ("stamped_louver_micro", _MI, "perforated_metal", (0.2, 0.2), ("power_box", "electronics_enclosure"), "louver stamped vent", "none"),
    ("chainlink_wire_mesh", _MI, "perforated_metal", (0.3, 0.3), ("fence_screen", "sieve"), "wire mesh woven galvanized", "none"),
    ("circular_perf_sheet", _MI, "perforated_metal", (0.3, 0.3), ("vent_sheet", "speaker_grille"), "perforated stainless round", "none"),
    ("corrugated_sheet_metal", _MI, "metal_cladding", (0.5, 0.5), ("roller_shutter", "utility_siding"), "corrugated steel trapezoidal", "none"),
    ("ribbed_rubber_mat", _RP, "rubber_mat", (0.3, 0.3), ("escalator_step", "bus_aisle"), "rubber ribbed mat", "none"),
    ("brushed_anodized_plate", _MI, "brushed_metal", (0.5, 0.5), ("elevator_trim",), "brushed aluminum anodized", "none"),
    ("embossed_braille_strip", _MI, "machined_metal", (0.2, 0.2), ("handrail", "elevator_panel"), "braille tactile embossed", "none"),
    ("stamped_vent_slots_stagger", _MI, "perforated_metal", (0.4, 0.4), ("transformer_cabinet", "locker_door"), "vent slots stamped", "none"),
    ("riveted_sheet_seam", _MI, "metal_cladding", (1.0, 1.0), ("bus_body", "train_car", "aircraft_skin"), "riveted lap seam", "none"),
    ("micro_hex_grip", _RP, "molded_plastic", (0.05, 0.05), ("tool_handle", "camera_grip"), "grip hex silicone", "none"),
    ("woven_nylon_strap", _FU, "woven_textile", (0.05, 0.05), ("seatbelt", "backpack_strap"), "nylon webbing strap", "none"),
    ("knurled_straight_flutes", _MI, "machined_metal", (0.05, 0.05), ("control_knob", "pen_barrel"), "knurl straight flutes", "none"),
    ("dimpled_golf_grip", _RP, "molded_plastic", (0.08, 0.08), ("handle_grip", "sports_handle"), "dimpled grip rubber", "none"),
    ("perforated_leather_auto", _FU, "leather", (0.2, 0.2), ("car_seat", "headphone_pad"), "leather perforated automotive", "none"),
    ("stair_nosing_grooves", _MI, "machined_metal", (0.3, 0.3), ("stair_nosing", "step_edge"), "stair nosing aluminum anti_slip", "none"),
    ("heatsink_extrusion_fins", _RT, "heatsink", (0.15, 0.15), (), "heatsink aluminum fins", "none"),
    ("window_blinds_slats", _MI, "perforated_metal", (0.5, 0.5), ("venetian_blind", "window_shade"), "blinds slats venetian", "none"),
    ("acoustic_slat_wood", _WT, "wood_slats", (0.6, 0.6), (), "oak acoustic slat", "v"),
    ("drip_tray_sieve_grate", _MI, "perforated_metal", (0.2, 0.2), ("espresso_drip_tray", "sink_grate"), "stainless grate drip", "none"),
    ("modular_stud_grid", _RP, "molded_plastic", (0.1, 0.1), ("toy_brick", "baseplate"), "stud plastic toy", "none"),
    ("waffle_silicone_pad", _RP, "rubber_mat", (0.2, 0.2), ("coaster", "kitchen_mat"), "silicone waffle", "none"),
    ("keyboard_keycap_grid", _RP, "molded_plastic", (0.2, 0.2), ("keyboard", "control_panel"), "keyboard keycaps plastic", "none"),
    ("solar_cell_grid", _RT, "solar", (0.6, 0.6), (), "solar monocrystalline", "none"),
    ("pcb_solder_mask_traces", _RT, "pcb", (0.1, 0.1), (), "pcb green electronics", "none"),
    ("acoustic_pyramid_foam", _RT, "acoustic_foam", (0.5, 0.5), (), "acoustic pyramid foam", "none"),
    ("terry_velcro_loops", _FU, "plush_textile", (0.05, 0.05), ("towel", "bathrobe"), "terry loops velcro", "none"),
    ("reeded_coin_milling", _MI, "machined_metal", (0.02, 0.02), ("coin_edge", "watch_bezel"), "reeded milling coin", "none"),
    ("fluted_reeded_glass", _GO, "fluted_glass", (0.3, 0.3), (), "fluted reeded clear", "v"),
    ("terrazzo_speckled_stone", _SM, "granite_quartz", (1.0, 1.0), ("terrazzo_floor",), "terrazzo speckled", "none"),
    ("woven_rattan_cane", _FU, "natural_fiber", (0.3, 0.3), ("cafe_chair_seat", "cabinet_door"), "rattan cane webbing", "none"),
    ("corkboard_pin_surface", _FU, "natural_fiber", (0.3, 0.3), ("pinboard", "cork_wall"), "cork board natural", "none"),
    ("hammered_copper_dimple", _MI, "specialty_metal", (0.4, 0.4), ("farmhouse_sink", "range_hood"), "copper hammered", "none"),
    ("cable_knit_wool", _FU, "plush_textile", (0.2, 0.2), ("sweater", "throw_blanket"), "knit wool cable", "none"),
    ("penny_round_mosaic", _TC, "mosaic", (0.24, 0.24), (), "penny teal mosaic", "none"),
    ("hex_mosaic_tile", _TC, "mosaic", (0.3, 0.3), (), "hex mosaic matte", "none"),
    ("fine_asphalt_tarmac", _RP, "asphalt", (1.2, 1.2), ("street",), "asphalt tarmac road", "none"),
    ("venetian_plaster_trowel", _SM, "plaster_stucco", (2.0, 2.0), (), "venetian plaster lime", "none"),
    ("frosted_prism_diffuser", _GO, "frosted_glass", (0.3, 0.3), ("ceiling_light_panel",), "prismatic diffuser acrylic", "none"),
    ("acoustic_kerf_felt", _FU, "plush_textile", (0.6, 0.6), ("acoustic_wall_panel", "office_partition"), "felt acoustic pet", "none"),
    ("patinated_verdigris_bronze", _MI, "specialty_metal", (0.5, 0.5), ("statue", "door_hardware"), "bronze verdigris patina", "none"),
    ("carrara_veined_marble", _SM, "marble", (1.2, 1.2), (), "carrara white marble", "none"),
    ("herringbone_tweed_wool", _FU, "woven_textile", (0.1, 0.1), ("blazer", "coat"), "tweed herringbone wool", "none"),
    ("waffle_cotton_linen", _FU, "woven_textile", (0.1, 0.1), ("towel", "bathrobe"), "waffle cotton spa", "none"),
    ("damascus_folded_steel", _MI, "specialty_metal", (0.2, 0.2), ("knife_blade",), "damascus steel blade", "none"),
    ("mermaid_fishscale_tile", _TC, "mosaic", (0.4, 0.4), ("feature_wall",), "fishscale scallop moroccan", "none"),
    ("burlap_jute_sackcloth", _FU, "woven_textile", (0.1, 0.1), ("sandbag", "coffee_sack"), "burlap jute hessian", "none"),
    ("cast_iron_skillet_pebble", _MI, "specialty_metal", (0.2, 0.2), ("skillet", "cookware"), "cast iron black", "none"),
    ("tufted_emerald_velvet", _FU, "plush_textile", (0.6, 0.6), ("chesterfield_sofa", "headboard"), "velvet tufted emerald", "none"),
    ("woven_bamboo_tatami", _FU, "natural_fiber", (0.9, 0.9), ("tatami_floor", "dojo_floor"), "tatami bamboo japanese", "none"),
    ("travertine_porous_stone", _SM, "natural_stone", (1.2, 1.2), (), "travertine roman", "none"),
    ("gilded_tin_ceiling_tile", _MI, "specialty_metal", (0.6, 0.6), ("pressed_tin_ceiling", "bar_ceiling"), "tin ceiling pressed brass", "none"),
    ("hardops_plate", _MI, "machined_metal", (0.5, 0.5), ("robot_body", "machine_housing", "scifi_door"), "hardops machined gunmetal", "none"),
]
FLAGSHIP_IDS: tuple[str, ...] = tuple(m[0] for m in FLAGSHIP_META)
HARDOPS_DESCRIPTION = "CNC-machined gunmetal HardOps panel with plug stamps, bolt plate, lug boss and weld fillets."
_FLAGSHIP_ROLE_EXTRAS = {
    "acoustic_pegboard": ("wall",), "sidewalk_expansion_joint": ("ground", "road"), "woven_bamboo_tatami": ("floor",),
    "corkboard_pin_surface": ("wall",), "acoustic_kerf_felt": ("wall",), "gilded_tin_ceiling_tile": ("ceiling",),
    "waffle_radiator_fins": ("mechanical",), "window_blinds_slats": ("glass_window",), "keyboard_keycap_grid": ("mechanical",),
}

# Lower low-frequency color drift for textiles (v1.1: removes cloudy "camo" blotches)
_LOW_CHROMA = {"woven_textile": 0.25, "plush_textile": 0.30, "leather": 0.40, "natural_fiber": 0.40}

_ACRONYMS = {"hvac": "HVAC", "pcb": "PCB", "hardops": "HardOps", "cmu": "CMU", "abs": "ABS", "hdpe": "HDPE",
             "epdm": "EPDM", "eva": "EVA", "mdf": "MDF", "osb": "OSB", "enig": "ENIG", "hasl": "HASL", "scifi": "Sci-Fi"}


def _title(mid: str) -> str:
    return " ".join(_ACRONYMS.get(p, p.capitalize()) for p in mid.split("_"))


def _color_tags(rgb_lin: tuple[float, float, float]) -> list[str]:
    srgb = [linear_to_srgb(c) for c in rgb_lin]
    h, s, v = colorsys.rgb_to_hsv(*srgb)
    hd = h * 360.0
    if v < 0.22:
        return ["black", "dark"]
    if s < 0.12:
        return ["white", "light"] if v > 0.82 else (["gray", "light"] if v > 0.62 else ["gray"])
    if (hd < 50 or hd > 340) and v < 0.62 and s > 0.25:
        return ["brown", "warm"]
    if 20 <= hd < 55 and s < 0.35:
        return ["beige", "warm"]
    for lo, hi, name in ((0, 15, "red"), (15, 42, "orange"), (42, 68, "yellow"), (68, 165, "green"), (165, 200, "teal"),
                         (200, 255, "blue"), (255, 295, "purple"), (295, 340, "pink"), (340, 361, "red")):
        if lo <= hd < hi:
            return [name, "warm" if name in ("red", "orange", "yellow", "pink") else "cool"]
    return []


def _finish_tags(rough: float, metallic: float) -> list[str]:
    tags = ["glossy"] if rough < 0.22 else (["satin"] if rough < 0.5 else ["matte"])
    if metallic >= 0.5:
        tags.append("metallic")
    return tags


def _stable_hash(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


_HUES = ("black", "white", "gray", "brown", "beige", "red", "orange", "yellow", "green", "teal", "blue", "purple", "pink")
_NEUTRAL_HUES = {"black", "white", "gray", "brown", "beige"}
_EXTERIOR_TAGS = {"exterior", "garden", "fence", "boardwalk", "barn", "abandoned", "coastal", "road", "driveway", "patio",
                  "pool", "roof", "street", "desert", "forest", "volcanic", "shanty", "rural", "farm"}
_NICHE_TAGS = {"jacket", "tshirt", "puffer", "military", "safety", "toy", "abandoned", "shanty", "playful", "scorched", "soot", "gym", "sports", "basketball", "sackcloth", "hessian"}
_SUB_ENV = {
    **{k: "ext" for k in ("brick", "concrete_block", "ashlar", "stacked_stone", "cobblestone", "paver", "wood_cladding", "bark",
                          "metal_cladding", "sand", "mud", "gravel", "ice", "ground_cover", "rock_cliff", "thatch", "asphalt",
                          "roofing", "solar", "tire_tread")},
    **{k: "int" for k in ("hardwood_floor", "parquet", "granite_quartz", "subway_tile", "zellige", "porcelain", "mosaic",
                          "glaze_ceramic", "woven_textile", "plush_textile", "leather", "natural_fiber", "foam", "molded_plastic",
                          "pcb", "heatsink", "acoustic_foam", "fluted_glass", "frosted_glass", "textured_glass", "brushed_metal", "marble")},
}
_GENERIC_TAGS = {"warm", "cool", "dark", "light", "matte", "satin", "glossy", "metallic", "black", "gray", "white"}


def _assemble() -> list[dict[str, Any]]:
    from . import procedural as gd

    daily = {did: (idx, desc, kind, tint) for idx, (did, desc, kind, tint) in enumerate(gd.DAILY_CATALOG)}
    entries: list[dict[str, Any]] = []

    # --- flagships ---
    for mid, cat, sub, size, extra, tags, axis in FLAGSHIP_META:
        prof = SUBCATEGORIES[sub]
        if mid == "hardops_plate":
            desc0, kind, tint, seed = HARDOPS_DESCRIPTION, "metal", (0.28, 0.31, 0.36), 901
            name = "HardOps Machined Plate"
        else:
            idx, desc0, kind, tint = daily[mid]
            seed = 2000 + idx * 41
            desc0 = desc0.rstrip(".") + "."
            name = _title(mid)
        def_r, def_m = KIND_DEFAULT_META.get(kind, (0.5, 0.0))
        if mid == "hardops_plate":
            def_r, def_m = 0.24, 0.95
        entries.append({
            "id": mid, "name": name, "category": cat, "subcategory": sub, "kind": kind,
            "description": f"{desc0} {prof['sentence']}",
            "generator": {"kind": "flagship", "seed": seed},
            "real_world_size_m": [float(size[0]), float(size[1])],
            "_objs": list(extra), "_roles": list(_FLAGSHIP_ROLE_EXTRAS.get(mid, ())), "_axis": axis,
            "_tags": tags.split(), "_tint": tuple(float(c) for c in tint), "_rough": def_r, "_metal": def_m,
        })

    # --- parametric ---
    for i, row in enumerate(_ROWS):
        prof = SUBCATEGORIES[row["sub"]]
        params = copy.deepcopy(row["params"])
        params.setdefault("rough", prof["rough"])
        if row["sub"] in _LOW_CHROMA:
            params.setdefault("chroma", _LOW_CHROMA[row["sub"]])
        params.setdefault("slope", prof["slope"])
        kind = row["kind"] or prof["kind"]
        metallic = float(params.get("metallic", 0.0))
        if row["family"] == "tech_panel" and params.get("mode") in ("pcb", "solar"):
            metallic = max(metallic, 0.0)
        seed = row["seed"] if row["seed"] is not None else 5000 + i * 37
        if params.get("roll") == "auto_planks":
            params["roll"] = _plank_roll(params, seed)
        entries.append({
            "id": row["id"], "name": row["name"] or _title(row["id"]), "category": prof["category"],
            "subcategory": row["sub"], "kind": kind,
            "description": f"{row['phrase'].rstrip('.')}. {prof['sentence']}",
            "generator": {"kind": "parametric", "family": row["family"], "params": params, "seed": seed},
            "real_world_size_m": [float(row["size"][0]), float(row["size"][1])],
            "_objs": row["objs"], "_roles": row["roles"], "_axis": row["axis"],
            "_tags": row["tags"], "_tint": tuple(float(c) for c in params["c1"]),
            "_rough": float(params["rough"]), "_metal": metallic,
        })

    # --- shared metadata ---
    for e in entries:
        prof = SUBCATEGORIES[e["subcategory"]]
        objs: list[str] = []
        for o in list(e.pop("_objs")) + prof["objects"]:
            if o not in objs:
                objs.append(o)
        e["best_objects"] = objs[:8]
        e["avoid_objects"] = [o for o in prof["avoid"] if o not in e["best_objects"]][:4]
        roles: list[str] = []
        for r in prof["roles"] + list(e.pop("_roles")):
            if r not in roles:
                roles.append(r)
        e["scene_roles"] = roles
        proj = copy.deepcopy(prof["projection"])
        axis = e.pop("_axis")
        if axis:
            proj["anisotropic_axis"] = axis
            proj["requires_tangents"] = bool(proj["requires_tangents"] or axis != "none")
        w, h = e["real_world_size_m"]
        proj["uv_repeat_formula"] = f"repeat_u = surface_width_m / {w:g}; repeat_v = surface_height_m / {h:g}"
        proj["triplanar_scale_formula"] = f"triplanar_scale = 1 / {w:g} (texture tiles per meter along U), 1 / {h:g} along V"
        e["projection"] = proj
        rough, metal = e.pop("_rough"), e.pop("_metal")
        r_lo, r_hi = max(0.03, round(rough - 0.10, 3)), min(1.0, round(rough + 0.28, 3))
        e["shader_setup"] = {
            **SHADER_CONSTANTS,
            "displacement_scale_m": float(prof["disp"]),
            "ior": float(prof["ior"]),
            "is_metallic": bool(metal >= 0.5),
            "roughness_range": [r_lo, r_hi],
            "metallic_range": [0.0, round(min(1.0, metal), 3)] if metal > 0 else [0.0, 0.0],
            "transmission": float(prof["transmission"]),
            "ranges_measured": False,
        }
        wx = dict(prof["weather"])
        if e["kind"] in ("metal",) and e["subcategory"] in ("brushed_metal",) and "copper" in e["id"]:
            wx["rust"] = 0.0
        wx["hypbr_params"] = {
            "wear": wx["wear"], "dirt": wx["dirt"], "wet": wx["moisture"], "moss": wx["moss"],
            "rust": wx["rust"], "dust": round(max(wx["dust"], 0.8 * wx["frost"]), 3),
        }
        e["recommended_weathering"] = wx
        tags: list[str] = []
        for t in (e.pop("_tags") + _color_tags(e.pop("_tint")) + _finish_tags(rough, metal)
                  + [e["subcategory"], e["category"], e["kind"]] + e["scene_roles"]):
            t = t.lower().replace("-", "_")
            if t and t not in tags:
                tags.append(t)
        e["tags"] = tags

    # --- pairing partners (style overlap + color harmony + interior/exterior context) ---
    by_sub: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_sub.setdefault(e["subcategory"], []).append(e)

    def _env(x: dict[str, Any]) -> str:
        t = set(x["tags"])
        if t & _EXTERIOR_TAGS:
            return "ext"
        if t & {"interior", "kitchen", "bathroom", "office", "luxury"}:
            return "int"
        return _SUB_ENV.get(x["subcategory"], "both")

    def _hue(x: dict[str, Any]) -> str | None:
        for t in x["tags"]:
            if t in _HUES:
                return t
        return None

    usage: dict[str, int] = {}
    for e in entries:
        style = set(e["tags"]) - _GENERIC_TAGS - {e["subcategory"], e["category"], e["kind"]} - set(SCENE_ROLES) - set(_HUES)
        env_e, hue_e = _env(e), _hue(e)

        def _score(c: dict[str, Any]) -> float:
            sc = 3.0 * len(style & set(c["tags"]))
            env_c = _env(c)
            if "both" not in (env_e, env_c) and env_c != env_e:
                sc -= 2.5
            hue_c = _hue(c)
            if hue_c in _NEUTRAL_HUES or hue_c is None or (c["shader_setup"]["is_metallic"] and set(c["tags"]) & {"brass", "copper", "gold", "bronze"}):
                sc += 1.0
            elif hue_c == hue_e:
                sc += 1.5
            else:
                sc -= 1.0
            if set(c["tags"]) & _NICHE_TAGS:
                sc -= 2.0
            if c["generator"]["kind"] == "flagship":
                sc += 0.3
            return sc - 0.12 * usage.get(c["id"], 0)

        picks: list[str] = []
        for sub in SUBCATEGORIES[e["subcategory"]]["pairs"]:
            cands = [c for c in by_sub.get(sub, []) if c["id"] != e["id"] and c["id"] not in picks]
            if not cands:
                continue
            cands.sort(key=lambda c: (-_score(c), _stable_hash(e["id"], c["id"])))
            picks.append(cands[0]["id"])
            if len(picks) >= 4:
                break
        if len(picks) < 3:
            same_cat = [c for c in entries if c["category"] == e["category"] and c["subcategory"] != e["subcategory"] and c["id"] not in picks]
            same_cat.sort(key=lambda c: (-_score(c), _stable_hash(e["id"], c["id"])))
            picks.extend(c["id"] for c in same_cat[: 3 - len(picks)])
        e["pairs_well_with"] = picks[:5]
        for pid in e["pairs_well_with"]:
            usage[pid] = usage.get(pid, 0) + 1

    # order: by category, then flagships first within each category
    order = {c["slug"]: i for i, c in enumerate(CATEGORIES)}
    entries.sort(key=lambda e: (order[e["category"]], 0 if e["generator"]["kind"] == "flagship" else 1))
    return entries


@lru_cache(maxsize=1)
def _catalog_cached() -> tuple[dict[str, Any], ...]:
    return tuple(_assemble())


def get_catalog() -> list[dict[str, Any]]:
    """Return a deep copy of all 500 atlas material entries (full metadata contract)."""
    return copy.deepcopy(list(_catalog_cached()))


@lru_cache(maxsize=1)
def _by_id() -> dict[str, dict[str, Any]]:
    return {e["id"]: e for e in _catalog_cached()}


def get_entry(material_id: str) -> dict[str, Any]:
    """Return a deep copy of one atlas entry; raises `KeyError` for unknown IDs."""
    try:
        return copy.deepcopy(_by_id()[material_id])
    except KeyError:
        raise KeyError(f"Unknown material_id '{material_id}'. Use hypbr.list_materials() or hypbr.atlas.search_atlas().") from None


def has_material(material_id: str) -> bool:
    return material_id in _by_id()


def parametric_spec(material_id: str) -> dict[str, Any] | None:
    """Return `{family, params, seed, kind}` for parametric IDs, else None (read-only view)."""
    e = _by_id().get(material_id)
    if e is None or e["generator"]["kind"] != "parametric":
        return None
    g = e["generator"]
    return {"family": g["family"], "params": copy.deepcopy(g["params"]), "seed": g["seed"], "kind": e["kind"]}


REQUIRED_FIELDS = (
    "id", "name", "category", "subcategory", "description", "generator", "real_world_size_m", "best_objects",
    "avoid_objects", "scene_roles", "projection", "shader_setup", "recommended_weathering", "pairs_well_with", "tags",
)


def validate_catalog(entries: list[dict[str, Any]] | None = None) -> list[str]:
    """Return a list of contract violations (empty list means the catalog is valid)."""
    ents = list(_catalog_cached()) if entries is None else entries
    errs: list[str] = []
    ids = [e["id"] for e in ents]
    if len(ents) != 500:
        errs.append(f"expected 500 materials, found {len(ents)}")
    if len(set(ids)) != len(ids):
        errs.append("duplicate ids: " + ", ".join(sorted({i for i in ids if ids.count(i) > 1})))
    idset = set(ids)
    counts: dict[str, int] = {}
    for e in ents:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
        for f in REQUIRED_FIELDS:
            if f not in e or e[f] in (None, "", [], {}):
                errs.append(f"{e['id']}: missing/empty field '{f}'")
        if not (4 <= len(e.get("best_objects", [])) <= 8):
            errs.append(f"{e['id']}: best_objects must have 4-8 items")
        if not (2 <= len(e.get("avoid_objects", [])) <= 4):
            errs.append(f"{e['id']}: avoid_objects must have 2-4 items")
        if set(e.get("best_objects", [])) & set(e.get("avoid_objects", [])):
            errs.append(f"{e['id']}: best/avoid overlap")
        bad_roles = [r for r in e.get("scene_roles", []) if r not in SCENE_ROLES]
        if bad_roles:
            errs.append(f"{e['id']}: invalid scene_roles {bad_roles}")
        pw = e.get("pairs_well_with", [])
        if not (3 <= len(pw) <= 5):
            errs.append(f"{e['id']}: pairs_well_with must have 3-5 items")
        for p in pw:
            if p not in idset or p == e["id"]:
                errs.append(f"{e['id']}: invalid pair '{p}'")
        sz = e.get("real_world_size_m", [])
        if len(sz) != 2 or not all(isinstance(v, float) and 0.005 <= v <= 10.0 for v in sz):
            errs.append(f"{e['id']}: invalid real_world_size_m {sz}")
        pj = e.get("projection", {})
        if pj.get("recommended") not in PROJECTIONS or pj.get("anisotropic_axis") not in ANISOTROPIC_AXES:
            errs.append(f"{e['id']}: invalid projection {pj}")
        ss = e.get("shader_setup", {})
        rr = ss.get("roughness_range", [])
        if len(rr) != 2 or not (0.0 <= rr[0] <= rr[1] <= 1.0):
            errs.append(f"{e['id']}: invalid roughness_range {rr}")
        wx = e.get("recommended_weathering", {})
        for k in ("wear", "dirt", "moisture", "moss", "rust", "frost"):
            if not (0.0 <= float(wx.get(k, -1)) <= 1.0):
                errs.append(f"{e['id']}: invalid weathering channel {k}")
        g = e.get("generator", {})
        if g.get("kind") not in ("flagship", "parametric"):
            errs.append(f"{e['id']}: invalid generator kind")
        if g.get("kind") == "parametric" and not {"family", "params", "seed"} <= set(g):
            errs.append(f"{e['id']}: incomplete parametric generator")
    for c in CATEGORIES:
        if counts.get(c["slug"], 0) != c["count"]:
            errs.append(f"category {c['slug']}: expected {c['count']}, found {counts.get(c['slug'], 0)}")
    return errs
