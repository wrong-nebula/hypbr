"""`hypbr photo`'s default macro-relief backend: Depth Anything V2 Small + the shipped 5 MB relief patch.

Offline: the model is built from the shipped config with random weights, so no Hugging Face download happens here.
"""
from __future__ import annotations

import numpy as np
import pytest

from hypbr import depthpro_pbr as dp
from hypbr.cli import build_parser


def test_photo_defaults_to_da2_and_rejects_unknown_models():
    args = build_parser().parse_args(["photo", "surface.jpg"])
    assert args.depth_model == "da2"
    assert build_parser().parse_args(["photo", "s.jpg", "--depth-model", "depthpro"]).depth_model == "depthpro"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["photo", "s.jpg", "--depth-model", "midas"])
    with pytest.raises(ValueError, match="depth_model"):
        dp.macro_relief_maps({}, depth_model="midas")


def test_patch_ships_as_package_data():
    from hypbr import relief_da2 as rd

    assert rd.PATCH_PATH.exists() and rd.CONFIG_PATH.exists()
    assert rd.PATCH_PATH.stat().st_size < 8 * 2**20  # the patch only, never the 99 MB base model
    assert (rd.PKG_DIR / "weights" / "NOTICE.md").exists()


torch_only = pytest.mark.skipif(
    not __import__("hypbr._backend", fromlist=["has_torch"]).has_torch(), reason="needs torch"
)


@pytest.fixture(scope="module")
def patched_model():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    from hypbr import relief_da2 as rd

    model, n = rd.load_da2_relief(pretrained=False)
    return model, n


@torch_only
def test_patch_fits_da2_small_exactly(patched_model):
    from safetensors.torch import load_file

    from hypbr import relief_da2 as rd

    model, n = patched_model
    patch = load_file(str(rd.PATCH_PATH))
    assert n == 2_728_513 and n == sum(v.numel() for v in patch.values())
    assert {k.split(".")[0] for k in patch} == {"neck", "head"}  # the DINOv2 backbone is never touched
    sd = model.state_dict()
    k = "head.conv3.weight"
    assert np.allclose(sd[k].float().cpu().numpy(), patch[k].float().numpy())


@torch_only
def test_patch_refuses_a_mismatched_model(patched_model, tmp_path):
    import torch
    from safetensors.torch import save_file

    from hypbr import relief_da2 as rd

    model, _ = patched_model
    bad = tmp_path / "bad.safetensors"
    save_file({"head.conv3.weight": torch.zeros(2, 32, 1, 1)}, str(bad))
    with pytest.raises(ValueError, match="does not fit"):
        rd.apply_relief_patch(model, bad)


@torch_only
@pytest.mark.parametrize("tileable,rot4", [(False, False), (False, True), (True, False)])
def test_infer_relief_returns_finite_height_at_input_size(patched_model, tileable, rot4):
    from hypbr import relief_da2 as rd

    model, _ = patched_model
    rgb = (np.random.default_rng(0).random((96, 96, 3)) * 255).astype(np.uint8)
    h = rd.infer_relief(model, rgb, tileable=tileable, rot4=rot4, res=126)
    assert h.shape == (96, 96) and h.dtype == np.float32 and np.isfinite(h).all() and h.std() > 0


def test_relief_normal_is_calibrated_unit_length_and_seamless():
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    bumps = np.sin(xx * 2 * np.pi / 32) * np.cos(yy * 2 * np.pi / 32)
    flat = dp.relief_normal(np.zeros((64, 64), np.float32))
    assert np.allclose(flat[..., 2], 1.0)
    n = dp.relief_normal(bumps)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-5)
    slope = lambda m: float((np.hypot(m[..., 0], m[..., 1]) / m[..., 2])[8:-8, 8:-8].mean())
    # the relief's units barely matter: 10x steeper relief gives 10**RELIEF_SLOPE_B = 2.3x the slope...
    assert slope(dp.relief_normal(bumps * 10)) / slope(n) == pytest.approx(10 ** dp.RELIEF_SLOPE_B, rel=1e-3)
    # ...and amplitude scales the slope linearly
    assert slope(dp.relief_normal(bumps, amplitude=2.0)) / slope(dp.relief_normal(bumps, amplitude=1.0)) == pytest.approx(2.0, rel=1e-3)
    # circular mode wraps: the first column continues the last one
    c = dp.relief_normal(bumps, circular=True)
    assert np.abs(c[:, 0] - c[:, -1]).mean() < 3 * np.abs(c[:, 1] - c[:, 0]).mean() + 1e-3


def test_without_deepbump_da2_relief_feeds_the_neural_normals(monkeypatch):
    from hypbr import normal_graph as ng

    rgb = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype(np.uint8)
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    relief = np.sin(xx / 5.0) + np.cos(yy / 7.0)
    monkeypatch.setattr(dp, "_run_deepbump_batch", lambda rgbs: (None, False))
    monkeypatch.setattr(ng, "_run_wsp_roughness_batch", lambda rgbs: None)
    monkeypatch.setattr(dp, "macro_relief_maps", lambda rgbs, **kw: {k: relief for k in rgbs})
    da2 = ng.extract_neural_sources({"a": rgb}, depth_model="da2")["a"]
    assert np.allclose(da2["neural_n"], dp.relief_normal(relief))
    dpro = ng.extract_neural_sources({"a": rgb}, depth_model="depthpro")["a"]
    assert not np.allclose(dpro["neural_n"], da2["neural_n"])  # Depth Pro keeps the luminance micro-relief stand-in
