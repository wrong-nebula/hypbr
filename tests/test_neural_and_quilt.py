"""Tests for hypbr's new neural modules (`micro_detail`, `svbrdf_da2`, `relief_da3`) and `seam_quilt`."""
from __future__ import annotations

import numpy as np
import pytest

from hypbr.cli import build_parser
from hypbr.seam_quilt import make_seamless_quilt


def test_seam_quilt_reduces_border_discontinuity_and_syncs_pbr_maps():
    rng = np.random.default_rng(42)
    # Non-tileable linear gradient + fine texture -> large wrap jump at border!
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    base = np.clip((xx / 128.0) * 0.7 + (yy / 128.0) * 0.25 + rng.normal(0, 0.04, (128, 128)), 0.0, 1.0)
    rgb = np.stack([base, base * 0.85, base * 0.7], axis=-1)
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    rough_u8 = (np.clip(base * 0.8, 0, 1) * 255.0).astype(np.uint8)
    height_f32 = base.astype(np.float32)

    q_rgb, q_maps = make_seamless_quilt(rgb_u8, {"roughness": rough_u8, "height": height_f32})
    assert q_rgb.shape == (128, 128, 3) and q_rgb.dtype == np.uint8
    assert q_maps["roughness"].shape == (128, 128) and q_maps["roughness"].dtype == np.uint8
    assert q_maps["height"].shape == (128, 128) and q_maps["height"].dtype == np.float32

    # Wrap seam jump should drop dramatically compared to the unquilted input!
    orig_jump = float(np.abs(rgb_u8[:, 0].astype(float) - rgb_u8[:, -1].astype(float)).mean())
    quilt_jump = float(np.abs(q_rgb[:, 0].astype(float) - q_rgb[:, -1].astype(float)).mean())
    assert quilt_jump < orig_jump * 0.35


def test_da3mono_cli_choice():
    args = build_parser().parse_args(["photo", "surface.jpg", "--depth-model", "da3mono"])
    assert args.depth_model == "da3mono"


torch_only = pytest.mark.skipif(
    not __import__("hypbr._backend", fromlist=["has_torch"]).has_torch(), reason="needs torch"
)


@torch_only
def test_microunet_dh_1k_ships_and_refines_normals():
    from hypbr import micro_detail as md

    assert md.WEIGHTS_F16_PATH.exists() and md.WEIGHTS_META_PATH.exists()
    assert md.WEIGHTS_F16_PATH.stat().st_size < 300 * 1024
    model = md.load_microunet_dh(device="cpu")
    rgb = (np.random.default_rng(7).random((64, 64, 3)) * 255).astype(np.uint8)
    n0 = np.zeros((64, 64, 3), np.float32)
    n0[..., 2] = 1.0
    h0 = np.zeros((64, 64), np.float32)
    n_out, dh = md.refine_micro_detail(model, rgb, n0, h0, tileable=True)
    assert n_out.shape == (64, 64, 3) and np.isfinite(n_out).all()
    assert np.allclose(np.linalg.norm(n_out, axis=-1), 1.0, atol=1e-4)
    assert dh.shape == (64, 64) and np.isfinite(dh).all()


@torch_only
def test_svbrdf_v2_ships_and_predicts_maps():
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    from hypbr import relief_da2 as rd, svbrdf_da2 as sv

    assert sv.SVBRDF_PATCH_PATH.exists()
    assert sv.SVBRDF_PATCH_PATH.stat().st_size < 4 * 2**20
    da2, _ = rd.load_da2_relief(pretrained=False)
    dec = sv.load_svbrdf_decoder(device="cpu")
    rgb = (np.random.default_rng(9).random((64, 64, 3)) * 255).astype(np.uint8)
    out = sv.infer_svbrdf(da2, dec, rgb, res=126)
    assert out["albedo"].shape == (64, 64, 3) and out["albedo"].dtype == np.uint8
    assert out["roughness"].shape == (64, 64) and out["roughness"].dtype == np.uint8
    assert out["metalness"].shape == (64, 64) and out["metalness"].dtype == np.uint8
