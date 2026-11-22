"""Smoke tests for hypbr procedural synthesis and node graph registry."""
from pathlib import Path

import pytest
from PIL import Image

from hypbr import PRESETS, list_materials, synthesize_material

EXPECTED_MAPS = {
    "albedo_delit.png",
    "anisotropy.png",
    "ao.png",
    "bent_normal.png",
    "cavity.png",
    "curvature.png",
    "edge_wear.png",
    "height.png",
    "height16.png",
    "metallic.png",
    "normal.png",
    "orm.png",
    "preview.png",
    "roughness.png",
    "specular.png",
    "thickness.png",
}


def test_catalog_and_presets_present() -> None:
    mats = list_materials()
    assert len(mats) > 0
    assert "v13-default" in PRESETS


def test_synthesize_single_material_seamless(tmp_path: Path) -> None:
    out_dir = synthesize_material("damascus_folded_steel", tmp_path / "damascus", size=128)
    produced = {p.name for p in out_dir.glob("*.png")}
    assert EXPECTED_MAPS.issubset(produced)

    h16 = Image.open(out_dir / "height16.png")
    assert h16.size == (128, 128)
    assert h16.mode in ("I;16", "I")

    orm = Image.open(out_dir / "orm.png")
    assert orm.size == (128, 128)
    assert orm.mode == "RGB"


def test_gpu_cpu_parity_hbao_and_blur() -> None:
    import numpy as np
    torch = pytest.importorskip("torch", reason="GPU/CPU parity needs torch")
    from hypbr import depthpro_pbr as dp
    from hypbr import normal_graph as ng

    h = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    h = np.sin(h * 12.0).astype(np.float32)
    g_blur = dp.gaussian_blur_np(h, sigma=3.0, circular=True)
    g_hbao = ng.node_horizon_bent_normal_ao({"circular": True}, {"height": h})

    old_dev = dp.DEVICE
    try:
        dp.DEVICE = torch.device("cpu")
        c_blur = dp.gaussian_blur_np(h, sigma=3.0, circular=True)
        c_hbao = ng.node_horizon_bent_normal_ao({"circular": True}, {"height": h})
    finally:
        dp.DEVICE = old_dev

    assert np.allclose(g_blur, c_blur, atol=1e-4)
    assert np.allclose(g_hbao["ao"], c_hbao["ao"], atol=1e-4)
    assert np.allclose(g_hbao["bent_normal"], c_hbao["bent_normal"], atol=1e-4)


def test_gpu_cpu_parity_periodic_fbm_and_preview() -> None:
    import numpy as np
    torch = pytest.importorskip("torch", reason="GPU/CPU parity needs torch")
    from hypbr import depthpro_pbr as dp
    from hypbr import procedural as gd
    from hypbr.synthesize import render_studio_preview

    g_fbm = gd._periodic_fbm(512, 512, 1234, (4, 8, 16, 32))
    alb = np.full((512, 512, 3), 0.55, dtype=np.float32)
    N = np.zeros((512, 512, 3), dtype=np.float32)
    N[..., 2] = 1.0
    rgh = np.full((512, 512), 0.32, dtype=np.float32)
    met = np.full((512, 512), 0.85, dtype=np.float32)
    ao = np.full((512, 512), 0.90, dtype=np.float32)
    g_prev = render_studio_preview(alb, N, rgh, met, ao)

    old_dev = dp.DEVICE
    try:
        dp.DEVICE = torch.device("cpu")
        c_fbm = gd._periodic_fbm(512, 512, 1234, (4, 8, 16, 32))
        c_prev = render_studio_preview(alb, N, rgh, met, ao)
    finally:
        dp.DEVICE = old_dev

    assert np.allclose(g_fbm, c_fbm, atol=1e-3)
    assert int(np.max(np.abs(g_prev.astype(int) - c_prev.astype(int)))) <= 1
