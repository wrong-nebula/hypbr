"""Tests for optional multi-mode procedural & neural PBR weathering (`hypbr weather`)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hypbr import (
    WEATHER_PRESETS,
    cli,
    resolve_weather_params,
    synthesize_material,
    verify_material,
    weather_material,
)


def test_weather_presets_pass_verify_and_write_packed_mask(tmp_path: Path) -> None:
    """Every built-in weathering preset yields a seamless, physically verified material + RGBA8 weather_mask."""
    base_dir = synthesize_material("travertine_porous_stone", tmp_path / "base", size=128)
    assert verify_material(base_dir)["passed"] is True

    for preset in sorted(WEATHER_PRESETS):
        out_dir = weather_material(base_dir, tmp_path / f"w_{preset}", preset=preset, seed=101)
        report = verify_material(out_dir)
        assert report["passed"] is True, f"Preset {preset} failed verify: {report}"
        wm_path = out_dir / "weather_mask.png"
        assert wm_path.exists()
        wm = np.asarray(Image.open(wm_path), dtype=np.uint8)
        assert wm.shape == (128, 128, 4)


def test_synthesize_material_optional_weathering_flags(tmp_path: Path) -> None:
    """`synthesize_material` supports optional multi-mode weathering while keeping default fresh mode unchanged."""
    fresh = synthesize_material("travertine_porous_stone", tmp_path / "fresh", size=128)
    assert not (fresh / "weather_mask.png").exists()

    rainy = synthesize_material("travertine_porous_stone", tmp_path / "rainy", size=128, wet=0.85, moss=0.65)
    assert (rainy / "weather_mask.png").exists()
    report = verify_material(rainy)
    assert report["passed"] is True, report

    r_fresh = np.asarray(Image.open(fresh / "roughness.png"), dtype=np.float32) / 255.0
    r_rainy = np.asarray(Image.open(rainy / "roughness.png"), dtype=np.float32) / 255.0
    wm = np.asarray(Image.open(rainy / "weather_mask.png"), dtype=np.float32) / 255.0
    puddle = wm[..., 2] > 0.55
    assert puddle.mean() > 0.05, f"Expected standing puddles in wet stone, got {puddle.mean():.4f}"
    assert r_rainy[puddle].mean() < 0.20, f"Puddles should be glossy, got {r_rainy[puddle].mean():.3f}"
    assert r_rainy[puddle].mean() < r_fresh[puddle].mean() - 0.35

    # Puddle normals should flatten strongly toward +Z
    n_rainy = np.asarray(Image.open(rainy / "normal.png"), dtype=np.float32) / 127.5 - 1.0
    assert n_rainy[..., 2][puddle].mean() > 0.96


def test_physical_material_archetype_responses(tmp_path: Path) -> None:
    """Moss turns crevices organic green/matte; rust oxidizes steel orange & strips metallic; copper gets verdigris."""
    # 1. Mossy stone
    stone_moss = synthesize_material("travertine_porous_stone", tmp_path / "stone_moss", size=128, moss=0.90)
    assert verify_material(stone_moss)["passed"] is True
    alb = np.asarray(Image.open(stone_moss / "albedo_delit.png"), dtype=np.float32) / 255.0
    rough = np.asarray(Image.open(stone_moss / "roughness.png"), dtype=np.float32) / 255.0
    wm = np.asarray(Image.open(stone_moss / "weather_mask.png"), dtype=np.float32) / 255.0
    moss_px = wm[..., 3] > 0.55
    assert moss_px.mean() > 0.04
    assert alb[..., 1][moss_px].mean() > alb[..., 0][moss_px].mean() + 0.04  # green > red
    assert alb[..., 1][moss_px].mean() > alb[..., 2][moss_px].mean() + 0.08  # green > blue
    assert rough[moss_px].mean() > 0.80

    # 2. Rusty steel
    steel_fresh = synthesize_material("diamond_tread_plate", tmp_path / "steel_fresh", size=128)
    steel_rust = synthesize_material("diamond_tread_plate", tmp_path / "steel_rust", size=128, rust=0.90)
    assert verify_material(steel_rust)["passed"] is True
    m_fresh = np.asarray(Image.open(steel_fresh / "metallic.png"), dtype=np.float32) / 255.0
    m_rust = np.asarray(Image.open(steel_rust / "metallic.png"), dtype=np.float32) / 255.0
    alb_rust = np.asarray(Image.open(steel_rust / "albedo_delit.png"), dtype=np.float32) / 255.0
    wm_rust = np.asarray(Image.open(steel_rust / "weather_mask.png"), dtype=np.float32) / 255.0
    rust_px = wm_rust[..., 1] > 0.50
    assert rust_px.mean() > 0.05
    assert m_rust[rust_px].mean() < m_fresh[rust_px].mean() - 0.50
    assert alb_rust[..., 0][rust_px].mean() > alb_rust[..., 2][rust_px].mean() + 0.08

    # 3. Copper verdigris
    cu_rust = synthesize_material("hammered_copper_dimple", tmp_path / "cu_rust", size=128, rust=0.90)
    assert verify_material(cu_rust)["passed"] is True
    alb_cu = np.asarray(Image.open(cu_rust / "albedo_delit.png"), dtype=np.float32) / 255.0
    wm_cu = np.asarray(Image.open(cu_rust / "weather_mask.png"), dtype=np.float32) / 255.0
    cu_px = wm_cu[..., 1] > 0.55
    assert cu_px.mean() > 0.05
    assert alb_cu[..., 1][cu_px].mean() > alb_cu[..., 0][cu_px].mean() + 0.05  # verdigris green-teal > red


def test_edge_wear_has_no_chalky_global_veil(tmp_path: Path) -> None:
    """Smoothstepped edge wear modulates convex ridges without washing out flat faces."""
    base = synthesize_material("subway_metro_tile", tmp_path / "tile_base", size=256)
    worn = weather_material(base, tmp_path / "tile_wear", wear=0.80, seed=7)
    assert verify_material(worn)["passed"] is True

    alb_b = np.asarray(Image.open(base / "albedo_delit.png"), dtype=np.float32) / 255.0
    alb_w = np.asarray(Image.open(worn / "albedo_delit.png"), dtype=np.float32) / 255.0
    wm = np.asarray(Image.open(worn / "weather_mask.png"), dtype=np.float32) / 255.0
    flat_px = wm[..., 0] < 0.05
    assert flat_px.mean() > 0.40
    # Flat faces remain virtually untouched (zero global desaturation haze)
    assert float(np.abs(alb_w[flat_px] - alb_b[flat_px]).mean()) < 0.015


def test_weathering_resolution_invariance_256_vs_512(tmp_path: Path) -> None:
    """Weathering maps stay consistent across 256px and 512px synthesis."""
    d256 = synthesize_material("travertine_porous_stone", tmp_path / "c256", size=256, weather_preset="abandoned")
    d512 = synthesize_material("travertine_porous_stone", tmp_path / "c512", size=512, weather_preset="abandoned")
    assert verify_material(d256)["passed"] is True
    assert verify_material(d512)["passed"] is True

    for fname in ("roughness.png", "albedo_delit.png", "weather_mask.png"):
        v256 = np.asarray(Image.open(d256 / fname), dtype=np.float32).mean() / 255.0
        v512 = np.asarray(Image.open(d512 / fname), dtype=np.float32).mean() / 255.0
        assert abs(v256 - v512) < 0.045, f"{fname} drifted across resolutions: 256={v256:.4f} vs 512={v512:.4f}"


def test_cli_synth_and_weather_commands(tmp_path: Path) -> None:
    """Both `hypbr synth --weather-preset` and `hypbr weather` run cleanly via CLI."""
    rc1 = cli.main([
        "synth", "sidewalk_expansion_joint",
        "--out", str(tmp_path / "s1"),
        "--size", "128",
        "--weather-preset", "rain",
    ])
    assert rc1 == 0
    mat_dir = tmp_path / "s1" / "sidewalk_expansion_joint"
    assert verify_material(mat_dir)["passed"] is True
    assert (mat_dir / "weather_mask.png").exists()

    out_w = tmp_path / "s2_weathered"
    rc2 = cli.main([
        "weather", str(mat_dir),
        "--out", str(out_w),
        "--preset", "mossy",
        "--rust", "0.4",
    ])
    assert rc2 == 0
    assert verify_material(out_w)["passed"] is True
    assert (out_w / "weather_mask.png").exists()


def test_weather_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown weathering preset"):
        resolve_weather_params("nonexistent_preset")

    base = synthesize_material("travertine_porous_stone", tmp_path / "b", size=64)
    with pytest.raises(ValueError, match="Unknown weathering backend"):
        weather_material(base, tmp_path / "bad_be", wet=0.5, backend="magic")

    with pytest.raises(FileNotFoundError, match="patch directory not found"):
        weather_material(base, tmp_path / "no_dir", wet=0.5, backend="neural", patch_dir=tmp_path / "missing")
