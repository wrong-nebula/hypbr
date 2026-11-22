"""Every weathering channel must respond like a dial, not a switch.

Measured as the mean change in albedo + roughness against an unweathered pass, relative to the change at
full intensity: a trace of a channel does almost nothing, small values already do something, and more
never does less. (Before the fade-in knees, `dirt=0.01` gave ~50% of its full effect on brick and wood,
`wet=0.01` ~57% on metal, and `wear` did nothing at all below ~0.25.)
"""
from __future__ import annotations

import numpy as np
import pytest

from hypbr.synthesize import synthesize_material_maps
from hypbr.weathering import FADE_IN_KNEES, WEATHER_PRESETS, weather_material_maps

CHANNELS = ("wear", "dirt", "wet", "moss", "rust", "dust")
LEVELS = (0.005, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0)
MATERIALS = {"diamond_tread_plate": "metal", "classic_red_brick": "brick", "white_oak_plank_floor": "wood"}
_cache: dict = {}


def _fresh(mid: str):
    if mid not in _cache:
        _cache[mid] = synthesize_material_maps(mid, size=96, preview=False)
    return _cache[mid]


def _effect(mid: str, kind: str, **kw) -> float:
    w = weather_material_maps(_fresh(mid), material_kind=kind, seed=7, preview=False, **kw)
    z = _cache.setdefault((mid, "zero"), weather_material_maps(_fresh(mid), material_kind=kind, seed=7, preview=False, wear=0.0))
    alb = np.abs(np.asarray(w.albedo, np.float64) - np.asarray(z.albedo, np.float64)).mean() / max(float(np.asarray(z.albedo).max()), 1.0)
    rgh = np.abs(np.asarray(w.roughness, np.float64) - np.asarray(z.roughness, np.float64)).mean() / max(float(np.asarray(z.roughness).max()), 1.0)
    return float(alb + rgh)


LOW_RELIEF_WEAR = pytest.mark.xfail(
    strict=True,
    reason="Known: edge wear is thresholded twice, so low-relief materials (oak planks, p99 convexity ~0.37) "
    "show no wear below ~0.5. Fixing it changes every preset's wear look; tracked separately.",
)
CASES = [
    pytest.param(ch, mid, marks=LOW_RELIEF_WEAR) if (ch, mid) == ("wear", "white_oak_plank_floor") else (ch, mid)
    for ch in CHANNELS for mid in sorted(MATERIALS)
]


@pytest.mark.parametrize("channel,mid", CASES)
def test_channel_response_is_a_dial(channel, mid):
    kind = MATERIALS[mid]
    d = [_effect(mid, kind, **{channel: lv}) for lv in LEVELS]
    full = d[-1]
    assert full > 0, f"{channel} has no effect on {mid}"
    rel = [x / full for x in d]
    shown = ", ".join(f"{lv}:{r:.2f}" for lv, r in zip(LEVELS, rel))
    assert rel[0] <= 0.04, f"{channel} on {mid} jumps on at a trace ({shown})"
    # Some channels start slowly by design (dust on metal only settles in crevices), but none may be dead.
    assert rel[2] > 5e-4 and rel[4] >= 0.01, f"{channel} on {mid} has a dead zone ({shown})"
    for a, b in zip(rel, rel[1:]):
        assert b >= a - 0.03, f"{channel} on {mid} is not monotone ({shown})"


def test_presets_sit_at_or_above_the_knees():
    """Presets were tuned on the original curves, which the fade-in keeps from each knee upward.
    `wear` is the exception: below its knee the old curve did nothing at all."""
    for name, p in WEATHER_PRESETS.items():
        for ch, val in p.items():
            if ch == "wear" or val <= 0.0:
                continue
            if ch == "rust" and name == "aged":
                continue  # deliberately lighter: rust used to switch on at full strength
            assert val >= FADE_IN_KNEES[ch] - 1e-9, f"preset {name} has {ch}={val} below its knee {FADE_IN_KNEES[ch]}"
