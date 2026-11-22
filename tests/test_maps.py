"""In-memory `MaterialMaps` API: identical bytes to the directory API, round-trips, chaining."""
from __future__ import annotations

import filecmp
from pathlib import Path

import numpy as np
import pytest

from hypbr import (
    SUPPORTED_TARGETS,
    MaterialMaps,
    blend_materials,
    blend_materials_maps,
    export_material,
    synthesize_material,
    synthesize_material_maps,
    verify_material,
    weather_material,
    weather_material_maps,
)

SIZE = 96


def _same_tree(a: Path, b: Path) -> None:
    fa = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert fa == fb
    diff = [str(r) for r in fa if not filecmp.cmp(a / r, b / r, shallow=False)]
    assert not diff, diff


@pytest.fixture(scope="module")
def brick() -> MaterialMaps:
    return synthesize_material_maps("classic_red_brick", size=SIZE)


@pytest.fixture(scope="module")
def plate() -> MaterialMaps:
    return synthesize_material_maps("hardops_plate", size=SIZE)


def test_maps_types_and_metadata(brick: MaterialMaps) -> None:
    assert brick.shape == (SIZE, SIZE) and brick.size == SIZE
    assert brick.material_id == "classic_red_brick" and brick.metadata["calibrated"] is True
    for name in ("albedo", "normal", "height", "roughness", "metallic", "ao", "bent_normal",
                 "curvature", "cavity", "edge_wear", "anisotropy", "specular", "thickness"):
        arr = getattr(brick, name)
        assert arr.dtype == np.float32, name
        assert np.isfinite(arr).all(), name
    assert brick.albedo.shape == (SIZE, SIZE, 3) and brick.anisotropy.shape == (SIZE, SIZE, 3)
    assert np.allclose(np.linalg.norm(brick.normal, axis=-1), 1.0, atol=1e-4)
    assert brick.preview.dtype == np.uint8 and brick.weather_mask is None
    assert brick.orm.shape == (SIZE, SIZE, 3)
    assert "orm.png" in brick.file_names() and "weather_mask.png" not in brick.file_names()


@pytest.mark.parametrize("mid,kw", [
    ("classic_red_brick", {}),
    ("classic_red_brick", {"calibrated": False, "wear": 0.6}),
    ("hardops_plate", {}),
    ("hardops_plate", {"weather_preset": "aged"}),
    ("woven_denim_twill", {"weather_preset": "rain"}),
])
def test_save_matches_synthesize_material(tmp_path: Path, mid: str, kw: dict) -> None:
    a = synthesize_material(mid, tmp_path / "dir" / mid, size=SIZE, **kw)
    b = synthesize_material_maps(mid, size=SIZE, **kw).save(tmp_path / "mem" / mid)
    _same_tree(a, b)


def test_load_save_round_trip(tmp_path: Path, brick: MaterialMaps, plate: MaterialMaps) -> None:
    for m in (brick, plate, weather_material_maps(brick, preset="mossy")):
        d = m.save(tmp_path / f"src_{m.material_id}")
        loaded = MaterialMaps.load(d)
        assert loaded.material_id == d.name
        _same_tree(d, loaded.save(tmp_path / f"rt_{m.material_id}"))
    assert "source_512" in MaterialMaps.load(tmp_path / "src_hardops_plate").extras


def test_blend_weather_export_accept_maps(tmp_path: Path, brick: MaterialMaps, plate: MaterialMaps) -> None:
    da = brick.save(tmp_path / "a" / "classic_red_brick")
    db = plate.save(tmp_path / "b" / "hardops_plate")
    for mode in ("height", "cavity"):
        _same_tree(
            blend_materials(da, db, tmp_path / f"bd_{mode}", mode=mode, factor=0.4),
            blend_materials(brick, plate, tmp_path / f"bm_{mode}", mode=mode, factor=0.4),
        )
    _same_tree(
        weather_material(da, tmp_path / "wd", preset="coastal", seed=7),
        weather_material(brick, tmp_path / "wm", preset="coastal", seed=7),
    )
    for t in SUPPORTED_TARGETS:
        _same_tree(export_material(da, tmp_path / f"ed_{t}", t), export_material(brick, tmp_path / f"em_{t}", t))


def test_in_memory_chain_verifies(tmp_path: Path, brick: MaterialMaps) -> None:
    moss = synthesize_material_maps("mossy_cobblestone_path", size=SIZE)
    blended = blend_materials_maps(brick, moss, factor=0.35)
    assert blended.blend_mask is not None and blended.metadata["blend"]["mode"] == "height"
    aged = weather_material_maps(blended, preset="abandoned", seed=3)
    assert aged.weather_mask.shape == (SIZE, SIZE, 4)
    rep = verify_material(aged)
    assert rep["passed"], rep["errors"]
    assert verify_material(aged.save(tmp_path / "aged"))["checks"] == rep["checks"]


def test_maps_validation_and_encode_errors(brick: MaterialMaps) -> None:
    with pytest.raises(ValueError, match="shape"):
        MaterialMaps(albedo=brick.albedo, normal=brick.normal, height=brick.height[:8],
                     roughness=brick.roughness, metallic=brick.metallic, ao=brick.ao)
    with pytest.raises(KeyError):
        brick.encode("weather_mask")
    with pytest.raises(KeyError):
        brick.encode("not_a_map.png")
    cp = brick.copy()
    cp.roughness[:] = 0.25
    assert int(cp.encode("roughness.png")[0, 0]) == 64 and not np.array_equal(cp.roughness, brick.roughness)
