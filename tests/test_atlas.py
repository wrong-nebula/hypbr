"""Tests for the 500-material agent-ready hypbr PBR Atlas."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import hypbr
from hypbr import atlas, atlas_catalog as ac
from hypbr import procedural as gd
from hypbr.synthesize import list_materials, synthesize_hardops_plate, synthesize_material
from hypbr.verify import verify_material

DATA = Path(__file__).parent / "data"
CATALOG = ac.get_catalog()
BY_ID = {e["id"]: e for e in CATALOG}


# ---------------------------------------------------------------------------
# Catalog contract
# ---------------------------------------------------------------------------
def test_exactly_500_unique_materials():
    ids = [e["id"] for e in CATALOG]
    assert len(ids) == 500
    assert len(set(ids)) == 500
    assert all(i == i.lower() and " " not in i for i in ids)


def test_category_counts_match_spec():
    counts = {}
    for e in CATALOG:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
    assert counts == {c["slug"]: c["count"] for c in ac.CATEGORIES}
    assert sum(c["count"] for c in ac.CATEGORIES) == 500


def test_validate_catalog_has_no_violations():
    assert ac.validate_catalog() == []


def test_all_pairs_reference_existing_materials():
    for e in CATALOG:
        assert 3 <= len(e["pairs_well_with"]) <= 5, e["id"]
        for p in e["pairs_well_with"]:
            assert p in BY_ID and p != e["id"], (e["id"], p)


def test_metadata_fields_valid_and_non_empty():
    for e in CATALOG:
        for f in ac.REQUIRED_FIELDS:
            assert e.get(f) not in (None, "", [], {}), (e["id"], f)
        assert 4 <= len(e["best_objects"]) <= 8
        assert 2 <= len(e["avoid_objects"]) <= 4
        assert set(e["scene_roles"]) <= set(ac.SCENE_ROLES)
        w, h = e["real_world_size_m"]
        assert 0.005 <= w <= 10 and 0.005 <= h <= 10
        pj = e["projection"]
        assert pj["recommended"] in ac.PROJECTIONS
        assert pj["anisotropic_axis"] in ac.ANISOTROPIC_AXES
        assert isinstance(pj["triplanar_fallback"], bool) and isinstance(pj["requires_tangents"], bool)
        assert pj["tiling_note"] and "repeat_u" in pj["uv_repeat_formula"]
        ss = e["shader_setup"]
        assert ss["normal_space"] == "OpenGL (+Y)"
        assert ss["orm_packing"] == "R=AO, G=Roughness, B=Metallic"
        assert ss["aux_packing"] == "R=Height, G=Curvature, B=Cavity"
        assert ss["albedo_colorspace"] == "sRGB" and ss["data_colorspace"] == "Linear"
        assert 1.0 <= ss["ior"] <= 2.5 and ss["displacement_scale_m"] >= 0.0
        lo, hi = ss["roughness_range"]
        assert 0.0 <= lo <= hi <= 1.0
        wx = e["recommended_weathering"]
        for k in ("wear", "dirt", "moisture", "moss", "rust", "frost"):
            assert 0.0 <= wx[k] <= 1.0
        assert wx["preset"] and wx["note"]
        assert len(e["description"]) > 30 and e["description"].count(".") <= 3
        assert e["generator"]["kind"] in ("flagship", "parametric")


def test_flagships_and_parametric_split():
    flag = [e for e in CATALOG if e["generator"]["kind"] == "flagship"]
    par = [e for e in CATALOG if e["generator"]["kind"] == "parametric"]
    assert len(flag) == 69 and len(par) == 431
    legacy = {d for d, *_ in gd.DAILY_CATALOG} | {"hardops_plate"}
    assert {e["id"] for e in flag} == legacy


def test_parametric_param_sets_are_distinct():
    sigs = {json.dumps([e["generator"]["family"], e["generator"]["params"]], sort_keys=True)
            for e in CATALOG if e["generator"]["kind"] == "parametric"}
    assert len(sigs) == 431


def test_list_materials_backward_compatible():
    mats = list_materials()
    assert len(mats) == 500
    legacy = [d for d, *_ in gd.DAILY_CATALOG]
    assert [m["id"] for m in mats[: len(legacy)]] == legacy
    kinds = {d: k for d, _desc, k, _t in gd.DAILY_CATALOG}
    for m in mats[: len(legacy)]:
        assert m["category"] == kinds[m["id"]]
        assert {"id", "description", "category"} <= set(m)


def test_version_is_1_1_0():
    assert hypbr.__version__ == "1.1.0"
    assert ac.ATLAS_VERSION == "1.1.0"


def test_detail_helpers_are_periodic():
    from hypbr import atlas_detail as ad

    f = ad.fbm(96, 96, 7, (4, 48, 160))
    shifted = ad.warp(f, np.full_like(f, 1.0), np.full_like(f, 0.0))  # warp by exactly one tile
    assert np.allclose(shifted, f, atol=1e-4)
    sc = ad.ellipse_scatter(64, 64, 6, 3)
    assert sc["mask"].any() and (sc["dist"] <= 1.0).all()


@pytest.mark.parametrize("cover", ["grass", "moss", "lichen", "soil", "leaves", "mulch", "needles", "snow"])
def test_ground_cover_modes_seamless(tmp_path, cover):
    mid = next(e["id"] for e in CATALOG if e["generator"].get("params", {}).get("cover") == cover)
    d = synthesize_material(mid, tmp_path / mid, size=128)
    assert verify_material(d)["passed"], mid


# ---------------------------------------------------------------------------
# Legacy synthesis is unchanged
# ---------------------------------------------------------------------------
LEGACY_STATS = json.loads((DATA / "legacy_map_stats.json").read_text())


@pytest.mark.parametrize("mid", sorted(LEGACY_STATS["materials"]))
def test_legacy_materials_synthesize_identically(tmp_path, mid):
    size = LEGACY_STATS["size"]
    if mid == "hardops_plate":
        d = synthesize_hardops_plate(tmp_path / mid, size=size)
    else:
        d = synthesize_material(mid, tmp_path / mid, size=size, calibrated=False)
    for name, (mean_ref, std_ref) in LEGACY_STATS["materials"][mid].items():
        a = np.asarray(Image.open(d / f"{name}.png"), dtype=np.float64)
        assert abs(a.mean() - mean_ref) < 0.75, (mid, name, a.mean(), mean_ref)
        assert abs(a.std() - std_ref) < 0.75, (mid, name, a.std(), std_ref)


def test_calibration_table_valid():
    from hypbr import calibration as cal

    cal.clear_cache()
    table = cal._table()
    assert len(table) >= 350
    for mid, c in table.items():
        assert mid in BY_ID, mid
        assert len(c["band_gains"]) == cal.N_BANDS and all(0.45 <= g <= 2.6 for g in c["band_gains"])
        assert len(c["height_gains"]) == cal.N_BANDS and all(0.5 <= g <= 2.0 for g in c["height_gains"])
        assert 0.85 <= c["sat_gain"] <= 1.2 and abs(c["rough_shift"]) <= 0.15 and 0.6 <= c["rough_std_gain"] <= 1.8
        assert c["n_refs"] >= 2 and c["refs"] and c["targets"]["alb_bands"]


@pytest.mark.parametrize("mid", ["classic_red_brick", "woven_denim_twill", "green_lawn_grass", "subway_metro_tile", "fresh_black_asphalt"])
def test_calibrated_synthesis_seamless_and_different(tmp_path, mid):
    a = synthesize_material(mid, tmp_path / "cal", size=128)
    b = synthesize_material(mid, tmp_path / "raw", size=128, calibrated=False)
    assert verify_material(a)["passed"]
    ia = np.asarray(Image.open(a / "albedo_delit.png"), dtype=np.float64)
    ib = np.asarray(Image.open(b / "albedo_delit.png"), dtype=np.float64)
    assert np.abs(ia - ib).mean() > 0.2


def test_equalizer_is_periodic():
    from hypbr import calibration as cal

    rng = np.random.default_rng(0)
    alb = (0.2 + 0.3 * rng.random((64, 64, 3))).astype(np.float32)
    out = cal.equalize_albedo(alb, [2.0] * cal.N_BANDS, 1.1)
    out_rolled = cal.equalize_albedo(np.roll(alb, (7, 13), axis=(0, 1)), [2.0] * cal.N_BANDS, 1.1)
    assert np.allclose(np.roll(out, (7, 13), axis=(0, 1)), out_rolled, atol=1e-4)


def test_hardops_dispatch_through_synthesize_material(tmp_path):
    d = synthesize_material("hardops_plate", tmp_path / "hp", size=64)
    assert (d / "orm.png").is_file() and (d / "normal.png").is_file()


# ---------------------------------------------------------------------------
# Parametric families are seamless & non-degenerate
# ---------------------------------------------------------------------------
def _family_representatives():
    seen = {}
    for e in CATALOG:
        g = e["generator"]
        if g["kind"] != "parametric":
            continue
        key = (g["family"], g["params"].get("mode", g["params"].get("style", "")))
        seen.setdefault(key, e["id"])
    return sorted(seen.values())


FAMILY_REPS = _family_representatives()


def test_every_parametric_family_is_covered():
    fams = {BY_ID[i]["generator"]["family"] for i in FAMILY_REPS}
    assert len(fams) == 28


@pytest.mark.parametrize("mid", FAMILY_REPS)
def test_parametric_family_seamless_and_finite(tmp_path, mid):
    d = synthesize_material(mid, tmp_path / mid, size=128)
    rep = verify_material(d)
    assert rep["passed"], (mid, rep["errors"])
    for name in ("albedo_delit", "normal", "orm", "height"):
        a = np.asarray(Image.open(d / f"{name}.png"), dtype=np.float64)
        assert np.isfinite(a).all()
        assert a.std() > 0.05 or name == "orm", (mid, name)
    alb = np.asarray(Image.open(d / "albedo_delit.png"), dtype=np.float64)
    assert alb.std() > 0.5, mid


def test_parametric_weathering_runs(tmp_path):
    d = synthesize_material("classic_red_brick", tmp_path / "b", size=64, weather_preset="aged")
    assert (d / "weather_mask.png").is_file()
    assert verify_material(d)["passed"]


# ---------------------------------------------------------------------------
# Query API accuracy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query,check",
    [
        ("brick wall", lambda e: e["subcategory"] == "brick"),
        ("kitchen counter", lambda e: "kitchen_countertop" in e["best_objects"]),
        ("oak floor", lambda e: "oak" in e["tags"] and "floor" in e["scene_roles"]),
        ("asphalt road", lambda e: e["subcategory"] == "asphalt"),
        ("sofa", lambda e: "sofa" in e["best_objects"]),
        ("sci-fi corridor", lambda e: e["subcategory"] == "scifi_panel"),
    ],
)
def test_search_accuracy(query, check):
    res = atlas.search_atlas(query, limit=5)
    assert len(res) == 5
    for r in res:
        assert check(BY_ID[r["id"]]), (query, r["id"])


def test_search_filters():
    res = atlas.search_atlas(category="metals_industrial", limit=100)
    assert len(res) == 58
    met = atlas.search_atlas("panel", metallic=True, limit=50)
    assert met and all(BY_ID[r["id"]]["shader_setup"]["is_metallic"] or r["metallic"] for r in met)
    floors = atlas.search_atlas(scene_role="floor", max_roughness=0.5, limit=500)
    assert floors and all("floor" in r["scene_roles"] and r["roughness"] <= 0.5 for r in floors)


@pytest.mark.parametrize(
    "obj,check",
    [
        ("brick wall", lambda e: e["subcategory"] == "brick"),
        ("kitchen counter", lambda e: "kitchen_countertop" in e["best_objects"]),
        ("oak floor", lambda e: "oak" in e["tags"]),
        ("asphalt road", lambda e: e["subcategory"] == "asphalt"),
        ("sofa", lambda e: "sofa" in e["best_objects"]),
        ("sci-fi corridor", lambda e: e["subcategory"] == "scifi_panel"),
    ],
)
def test_recommend_for_object(obj, check):
    rec = atlas.recommend_for_object(obj, limit=3)
    assert len(rec["matches"]) == 3
    for m in rec["matches"]:
        e = BY_ID[m["id"]]
        assert check(e), (obj, m["id"])
        w, h = e["real_world_size_m"]
        sw, sh = m["surface_size_m"]
        assert m["uv_repeat"] == [round(sw / w, 3), round(sh / h, 3)]
        assert set(m["weathering"]["hypbr_params"]) == {"wear", "dirt", "wet", "moss", "rust", "dust"}


def test_recommend_environment_adjusts_weathering():
    interior = atlas.recommend_for_object("sofa", environment="interior", limit=1)["matches"][0]["weathering"]
    assert interior["moisture"] == 0.0 and interior["moss"] == 0.0
    coastal = atlas.recommend_for_object("corrugated roof", environment="coastal", limit=1)["matches"][0]
    base = BY_ID[coastal["id"]]["recommended_weathering"]
    assert coastal["weathering"]["moisture"] >= base["moisture"]


def test_material_spec_card():
    spec = atlas.get_material_spec("calacatta_gold_marble", base_url="https://example.test")
    assert spec["maps"]["orm"] == "https://example.test/assets/mats/calacatta_gold_marble/orm.webp"
    assert set(spec["maps"]) == {"albedo_delit", "normal", "orm", "aux", "preview"}
    assert "MeshPhysicalMaterial" in spec["snippets"]["threejs"]
    assert "ShaderNodeNormalMap" in spec["snippets"]["blender_bpy"]
    compile(spec["snippets"]["blender_bpy"], "<bpy>", "exec")
    assert spec["snippets"]["gltf"]["normalTexture"]["uri"].endswith("normal.webp")
    with pytest.raises(KeyError):
        atlas.get_material_spec("does_not_exist")


def test_get_atlas_manifest():
    m = atlas.get_atlas("https://example.test/")
    assert m["material_count"] == 500 and m["version"] == "1.1.0"
    assert m["endpoints"]["llms_txt"] == "https://example.test/llms.txt"
    assert len(m["categories"]) == 10


def test_export_atlas_files(tmp_path):
    info = atlas.export_atlas_files(tmp_path, base_url="https://example.test")
    assert info["materials"] == 500
    for f in ("llms.txt", "agent.md", "atlas.json", "atlas/index.json"):
        assert (tmp_path / f).is_file(), f
    idx = json.loads((tmp_path / "atlas" / "index.json").read_text())
    assert idx["count"] == 500 and len(idx["materials"]) == 500
    assert len(list((tmp_path / "atlas" / "mats").glob("*.json"))) == 500
    assert len(list((tmp_path / "atlas" / "categories").glob("*.json"))) == 10
    shard_total = sum(json.loads(p.read_text())["count"] for p in (tmp_path / "atlas" / "categories").glob("*.json"))
    assert shard_total == 500
    llms = (tmp_path / "llms.txt").read_text()
    assert llms.startswith("# hypbr PBR Material Atlas") and "https://example.test/atlas/index.json" in llms
    card = json.loads((tmp_path / "atlas" / "mats" / "classic_red_brick.json").read_text())
    assert card["maps"]["albedo_delit"].startswith("https://example.test/assets/mats/classic_red_brick/")
    assert (tmp_path / "atlas" / "index.json").stat().st_size < 600_000


def test_playground_entries():
    rows = atlas.playground_entries()
    assert len(rows) == 500
    groups = {r["group"] for r in rows}
    assert groups == {c["name"] for c in ac.CATEGORIES}
    assert all(r["bundle"]["orm"].endswith("/orm.webp") for r in rows)


def test_cli_atlas_commands(capsys, tmp_path):
    from hypbr.cli import main

    assert main(["atlas", "search", "brick", "wall", "--limit", "3"]) == 0
    assert "brick" in capsys.readouterr().out
    assert main(["atlas", "info", "classic_red_brick"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "classic_red_brick"
    assert main(["atlas", "recommend", "kitchen", "counter", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["matches"]
