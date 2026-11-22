"""Tests for zero-training analytic WeightCode kernel surgery & 4-quadrant seamless blending."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="weightcode operates on torch modules (pip install hypbr[neural])")
nn = torch.nn

from hypbr.weightcode import (
    BASIS_3X3,
    apply_depthpro_weightcode,
    spectral_filter_3x3_kernel,
    svd_energy_shrink,
    tileable_4quad_blend,
)


def test_3x3_basis_is_orthonormal_and_identity() -> None:
    gram = BASIS_3X3 @ BASIS_3X3.T
    assert torch.allclose(gram, torch.eye(9), atol=1e-5)

    w = torch.randn(8, 4, 3, 3)
    w_id = spectral_filter_3x3_kernel(w, dc_gain=1.0, ramp_gain=1.0, detail_gain=1.0)
    assert torch.allclose(w, w_id, atol=1e-5)


def test_spectral_surgery_suppresses_ramp_and_boosts_laplacian() -> None:
    # Channel 0: pure linear ramp (perspective tilt), Channel 1: pure 2D Laplacian (fine detail)
    ramp = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    lap = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    w = torch.stack([ramp, lap], dim=0).unsqueeze(1)  # (2, 1, 3, 3)

    w_out = spectral_filter_3x3_kernel(w, dc_gain=1.0, ramp_gain=0.25, detail_gain=1.5)
    assert torch.allclose(w_out[0, 0], 0.25 * ramp, atol=1e-5)
    assert torch.allclose(w_out[1, 0], 1.50 * lap, atol=1e-5)


def test_svd_energy_shrink_and_dpt_weightcode() -> None:
    mat = torch.randn(16, 16, 1, 1)
    shrunk = svd_energy_shrink(mat, energy_keep=0.80, tail_scale=0.2)
    assert shrunk.shape == mat.shape
    assert torch.norm(shrunk) < torch.norm(mat)

    class _DummyFusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.resnet1 = nn.Module()
            self.resnet1.residual = nn.Sequential(
                nn.Identity(), nn.Identity(), nn.Identity(), nn.Conv2d(4, 4, 3, padding=1)
            )

    class _DummyDPT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.convs = nn.ModuleList([nn.Conv2d(4, 4, 3, padding=1) for _ in range(4)])
            self.decoder.fusions = nn.ModuleList([_DummyFusion(), _DummyFusion()])
            self.head = nn.Conv2d(4, 1, 3, padding=1)

    model = _DummyDPT()
    report = apply_depthpro_weightcode(model, circular_pad=True)
    assert report.tensors_edited == 8  # 4 convs + 2 fusions * (weight + bias)
    assert report.circular_conv_layers == 7  # 4 convs + 2 fusion convs + 1 head
    assert model.head.padding_mode == "circular"


def test_tileable_4quad_blend_cancels_border_discontinuity() -> None:
    # Periodic signal + artificial non-periodic border corruption injected by a simulated ViT
    H = W = 64
    y = np.linspace(0, 2 * np.pi, H, endpoint=False)
    x = np.linspace(0, 2 * np.pi, W, endpoint=False)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rgb = np.clip((0.5 + 0.4 * np.sin(2 * xx) * np.cos(2 * yy)) * 255, 0, 255).astype(np.uint8)
    rgb3 = np.stack([rgb, rgb, rgb], axis=-1)

    def _biased_infer(img: np.ndarray) -> np.ndarray:
        base = img[..., 0].astype(np.float32) / 255.0
        # Inject strong border artifacts on rows 0,-1 and cols 0,-1
        corrupted = base.copy()
        corrupted[0, :] += 1.0
        corrupted[-1, :] -= 1.0
        corrupted[:, 0] += 1.0
        corrupted[:, -1] -= 1.0
        return corrupted

    blended = tileable_4quad_blend(rgb3, _biased_infer)
    # Border wrap jump on raw biased pass is ~2.0; 4-quad raised cosine cancels border seams cleanly
    seam_v = float(np.mean(np.abs(blended[0, :] - blended[-1, :])))
    seam_h = float(np.mean(np.abs(blended[:, 0] - blended[:, -1])))
    assert seam_v < 0.08
    assert seam_h < 0.08
