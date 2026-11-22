"""Tests for `hypbr.verify.verify_material` seamlessness and physical PBR QA checks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from hypbr import synthesize_material, verify_material


def test_verify_passes_on_synthesized_material(tmp_path: Path) -> None:
    out = synthesize_material("patinated_verdigris_bronze", tmp_path / "bronze", size=128)
    report = verify_material(out)
    assert report["passed"] is True, report
    assert report["checks"]["orm_max_channel_diff"] == 0
    assert report["checks"]["normal"]["positive_z_fraction"] == 1.0
    assert report["checks"]["seamless"]["height"]["seam_ratio"] < 2.0


def test_verify_catches_broken_seam_and_bad_orm(tmp_path: Path) -> None:
    out = synthesize_material("damascus_folded_steel", tmp_path / "broken", size=64)

    # Inject a hard non-seamless linear gradient into height.png
    grad = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    Image.fromarray(grad, mode="L").save(out / "height.png")

    # Corrupt ORM channel
    orm = np.asarray(Image.open(out / "orm.png")).copy()
    orm[..., 1] = 255 - orm[..., 1]
    Image.fromarray(orm, mode="RGB").save(out / "orm.png")

    report = verify_material(out)
    assert report["passed"] is False
    assert any("height wrap seam" in e for e in report["errors"])
    assert any("orm.png channels diverge" in e for e in report["errors"])


MAPS = ("albedo_delit.png", "normal.png", "height.png", "height16.png", "ao.png", "roughness.png", "metallic.png", "orm.png")


def test_verify_catches_cropped_wrap_seam(tmp_path: Path) -> None:
    """A 128px window of a 160px tile no longer wraps: verify must flag it."""
    big = synthesize_material("calacatta_gold_marble", tmp_path / "big", size=160)
    crop = tmp_path / "crop"
    crop.mkdir()
    for f in MAPS:
        Image.open(big / f).crop((11, 17, 139, 145)).save(crop / f)
    report = verify_material(crop)
    assert report["passed"] is False
    assert any("wrap seam" in e for e in report["errors"])


def test_verify_catches_subtle_albedo_seam_on_smooth_material(tmp_path: Path) -> None:
    """A 6-level left-to-right ramp only breaks the wrap (invisible inside the tile). The old
    absolute 0.03 gate let this through; the per-line outlier test catches it."""
    out = synthesize_material("frosted_bathroom_glass", tmp_path / "ramp", size=128)
    a = np.asarray(Image.open(out / "albedo_delit.png").convert("RGB")).astype(np.float32)
    a = np.clip(a + np.linspace(0.0, 6.0, 128)[None, :, None], 0, 255).round().astype(np.uint8)
    Image.fromarray(a).save(out / "albedo_delit.png")
    report = verify_material(out)
    assert report["passed"] is False
    assert any("albedo wrap seam" in e and "h:" in e for e in report["errors"])


def test_verify_ignores_sub_level_jumps_on_flat_maps(tmp_path: Path) -> None:
    """Near-flat maps have tiny interior gradients; a 1-level border step must not count as a seam."""
    out = synthesize_material("smooth_architectural_concrete", tmp_path / "flat", size=64)
    flat = np.full((64, 64), 128, np.uint8)
    flat[:, :32] = 129
    Image.fromarray(flat, mode="L").save(out / "height.png")
    report = verify_material(out)
    assert not any("height wrap seam" in e for e in report["errors"]), report["errors"]
