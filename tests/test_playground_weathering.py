"""The playground's in-browser weathering (playground/static_engine.js) must respond like a dial too.

Runs the pure `weatherBase` function in Node on three baked playground materials and applies the same
rules as tests/test_weathering_response.py. Skipped when Node is not installed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PLAY = ROOT / "playground"
SCRIPT = ROOT / "tests" / "js" / "weather_response.mjs"
CHANNELS = ("wear", "dirt", "wet", "moss", "rust", "dust")
LEVELS = (0.005, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0)
MATERIALS = ("diamond_tread_plate", "classic_red_brick", "white_oak_plank_floor")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")


def _rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), np.uint8)


def _curves(mid: str, tmp: Path) -> dict[str, list[float]]:
    rows = {m["id"]: m for m in json.loads((PLAY / "assets" / "catalog.json").read_text())["materials"]}
    entry = {k: rows[mid][k] for k in ("id", "kind", "archetype", "default_metallic", "default_seed")}
    d = PLAY / "assets" / "mats" / mid
    alb, nrm, orm, aux = (_rgba(d / f"{n}.webp") for n in ("albedo_delit", "normal", "orm", "aux"))
    planes = {"albedo": alb, "normal": nrm, "ao": orm[..., 0], "rough": orm[..., 1], "metal": orm[..., 2],
              "height": aux[..., 0], "curv": aux[..., 1], "cav": aux[..., 2]}
    for name, arr in planes.items():
        (tmp / f"{name}.bin").write_bytes(np.ascontiguousarray(arr).tobytes())
    spec = tmp / "case.json"
    spec.write_text(json.dumps({"size": int(alb.shape[0]), "entry": entry, "channels": CHANNELS, "levels": LEVELS}))
    out = subprocess.run(["node", "--no-warnings", str(SCRIPT), str(spec)], capture_output=True, text=True, check=True, timeout=300)
    data = json.loads(out.stdout)
    from hypbr.weathering import FADE_IN_KNEES

    assert data["knees"] == FADE_IN_KNEES, "playground FADE_IN_KNEES drifted from hypbr.weathering"
    return data["curves"]


@pytest.mark.parametrize("mid", MATERIALS)
def test_playground_channels_respond_like_dials(mid, tmp_path):
    curves = _curves(mid, tmp_path)
    bad = []
    for ch in CHANNELS:
        d = curves[ch]
        full = d[-1]
        if full <= 0:
            bad.append(f"{ch}: no effect")
            continue
        rel = [x / full for x in d]
        shown = ", ".join(f"{lv}:{r:.2f}" for lv, r in zip(LEVELS, rel))
        if rel[0] > 0.04:
            bad.append(f"{ch} jumps on at a trace ({shown})")
        if not (rel[2] > 5e-4 and rel[4] >= 0.01):
            bad.append(f"{ch} has a dead zone ({shown})")
        if any(b < a - 0.03 for a, b in zip(rel, rel[1:])):
            bad.append(f"{ch} is not monotone ({shown})")
    assert not bad, f"{mid}: " + "; ".join(bad)
