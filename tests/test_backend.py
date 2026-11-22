"""Optional-torch backend: NumPy fallbacks match the torch kernels; hypbr runs without torch."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hypbr import _backend as bk

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def numpy_only(monkeypatch):
    """Run the backend as if torch were not installed."""
    monkeypatch.setattr(bk, "_TORCH", None)
    monkeypatch.setattr(bk, "_TORCH_TRIED", True)


def _both(fn, *args, **kwargs):
    """Return (torch result, numpy result) for a backend op."""
    pytest.importorskip("torch")
    if bk.get_torch() is None:
        pytest.skip("torch disabled by HYPBR_TORCH=0")
    ref = fn(*args, **kwargs)
    saved = (bk._TORCH, bk._TORCH_TRIED)
    bk._TORCH, bk._TORCH_TRIED = None, True
    try:
        alt = fn(*args, **kwargs)
    finally:
        bk._TORCH, bk._TORCH_TRIED = saved
    return ref, alt


RNG = np.random.default_rng(7)
IMG = RNG.standard_normal((37, 53)).astype(np.float32)


@pytest.mark.parametrize("mode", ["reflect", "circular", "replicate"])
def test_pad_and_correlate(mode: str) -> None:
    a, b = _both(bk.pad2d, IMG, 3, 2, 1, 4, mode)
    assert np.array_equal(a, b)
    k = RNG.standard_normal((3, 3)).astype(np.float32)
    a, b = _both(bk.correlate3x3, IMG, k, mode)
    assert np.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("circular", [True, False])
@pytest.mark.parametrize("rad", [1, 4, 9])
def test_separable_conv(circular: bool, rad: int) -> None:
    ker = RNG.random(2 * rad + 1).astype(np.float32)
    a, b = _both(bk.separable_conv, IMG, ker, circular)
    assert np.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("mode", ["reflect", "circular"])
def test_max_filter_exact(mode: str) -> None:
    a, b = _both(bk.max_filter, IMG, 3, mode)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize("shape", [(5, 7), (64, 50), (200, 260)])
def test_resize(mode: str, shape: tuple[int, int]) -> None:
    a, b = _both(bk.resize, IMG, shape[0], shape[1], mode)
    assert np.allclose(a, b, atol=2e-5)


@pytest.mark.parametrize("padding", ["border", "reflection"])
def test_grid_sample_bilinear(padding: str) -> None:
    gy, gx = np.meshgrid(np.linspace(-1.7, 2.3, 41), np.linspace(-2.9, 1.4, 45), indexing="ij")
    gx = (gx + 0.05 * np.sin(7 * gy)).astype(np.float32)
    gy = gy.astype(np.float32)
    a, b = _both(bk.grid_sample, IMG, gx, gy, "bilinear", padding, True)
    assert np.allclose(a, b, atol=1e-5)
    img3 = np.stack([IMG, -IMG, IMG * 2])
    a, b = _both(bk.grid_sample, img3, gx, gy, "bilinear", padding, True)
    assert a.shape == (3, 41, 45) and np.allclose(a, b, atol=1e-5)


def test_grid_sample_bicubic_border() -> None:
    gy, gx = np.meshgrid(np.linspace(-1.2, 1.2, 33), np.linspace(-0.98, 1.05, 31), indexing="ij")
    a, b = _both(bk.grid_sample, IMG, gx.astype(np.float32), gy.astype(np.float32), "bicubic", "border", False)
    assert np.allclose(a, b, atol=1e-4)


def test_high_octave_fbm_matches() -> None:
    from hypbr import procedural as gd

    g = RNG.standard_normal((300, 300)).astype(np.float32)
    a, b = _both(gd._fbm_octave_grid_sample, g, 96, 80, 300)
    assert np.allclose(a, b, atol=1e-3)  # float32 lattice coords ~300 -> ~1e-4 rounding


def test_numpy_blur_paths(numpy_only) -> None:
    from hypbr import depthpro_pbr as dp

    x = RNG.standard_normal((96, 96)).astype(np.float32)
    for sigma in (0.8, 1.6, 3.0, 7.0):  # direct taps and FFT branches
        rad = int(np.ceil(3 * sigma))
        xs = np.arange(-rad, rad + 1, dtype=np.float32)
        ker = np.exp(-0.5 * (xs / sigma) ** 2)
        ker /= ker.sum()
        for circ in (True, False):
            assert np.allclose(dp.gaussian_blur_np(x, sigma, circular=circ), dp._gaussian_blur_fft(x, ker, circ), atol=1e-5)


def test_require_torch_message(numpy_only) -> None:
    assert not bk.has_torch()
    assert bk.resolve_device().type == "cpu"
    with pytest.raises(ImportError, match=r"hypbr\[neural\]"):
        bk.require_torch("Neural photo mode")


_SUBPROC = r"""
import json, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import hypbr
from hypbr import synthesize_material
out = {"torch_loaded": "torch" in sys.modules, "has_torch": hypbr.has_torch(), "maps": {}}
d = Path(tempfile.mkdtemp())
for mid in sys.argv[1:]:
    m = synthesize_material(mid, d / mid, size=64)
    out["maps"][mid] = {p.name: np.asarray(Image.open(p)).astype(np.int64).tolist() for p in sorted(m.glob("*.png"))}
out["torch_loaded"] = out["torch_loaded"] or ("torch" in sys.modules)
print(json.dumps(out))
"""


def _run(env_extra: dict, mids: list[str]) -> dict:
    env = {**os.environ, "HYPBR_DEVICE": "cpu", "PYTHONPATH": str(ROOT), **env_extra}
    r = subprocess.run([sys.executable, "-c", _SUBPROC, *mids], env=env, capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_synth_without_torch_matches_torch_output() -> None:
    mids = ["classic_red_brick", "hardops_plate", "mossy_cobblestone_path"]
    no_torch = _run({"HYPBR_TORCH": "0"}, mids)
    assert no_torch["torch_loaded"] is False and no_torch["has_torch"] is False
    try:
        import torch  # noqa: F401
    except ImportError:
        return  # torch-free environment: importing + synthesizing without torch is the check
    with_torch = _run({"HYPBR_TORCH": "1"}, mids)
    for mid in mids:
        for name, arr in with_torch["maps"][mid].items():
            a, b = np.asarray(arr), np.asarray(no_torch["maps"][mid][name])
            assert a.shape == b.shape, (mid, name)
            assert np.abs(a - b).max() <= (257 if name == "height16.png" else 1), (mid, name)
