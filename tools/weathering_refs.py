#!/usr/bin/env python3
"""Compare hypbr weathering against real CC0 scans of weathered surfaces.

For each weathering type, the pixels that *are* the weathering are compared in CIE Lab:
  - in each scan, pixels classified by colour (rust: warm, saturated orange-brown; moss: green);
  - in hypbr output, the pixels where that channel's own mask is above 0.6, on matching materials.
Prints per-scan and per-material L/a/b/chroma/lightness spread and layout (coverage, largest connected
patch as a share of the covered area, patch count, all at 512px) plus group means, and writes a board of
scans beside hypbr renders.

usage: python tools/weathering_refs.py <scan_root> <board.png>
  <scan_root> holds <id>.npz files ("albedo", uint8 sRGB) in any sub-folder: the reference set
  `hypbr atlas calibrate --refs` uses, plus anything imported with tools/import_refs.py. Scans named in
  GROUPS are always compared; any other scan whose id or tags mention rust/corrosion or moss/lichen is
  added to that group automatically.

The rust and moss tones in hypbr/weathering.py were fitted with this tool (mean colour distance to the
scans: rust dE 19 -> 5, moss dE 17 -> 9).
"""
import os, sys, numpy as np
os.environ["HYPBR_DEVICE"] = "cpu"
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout without installing
from hypbr.color import linear_to_srgb_u8, srgb_u8_to_linear
from hypbr.synthesize import synthesize_material_maps
from hypbr.weathering import weather_material_maps

if len(sys.argv) != 3:
    sys.exit(__doc__)
D = Path(sys.argv[1]); OUT = Path(sys.argv[2])

def lab(lin):
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = lin.reshape(-1, 3) @ M.T / np.array([0.9505, 1.0, 1.089], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], -1)

_INDEX = None


def _index():
    """Every <id>.npz under the scan root (any sub-folder, including tools/import_refs.py output)."""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        for f in sorted(D.rglob("*.npz")):
            _INDEX.setdefault(f.stem, f)
    return _INDEX


def scan(name):
    f = _index().get(name)
    if f is None:
        raise FileNotFoundError(name)
    return np.load(f)["albedo"]


AUTO = {"rust": r"(?<![a-z])(rust|rusty|rusted|corroded|corrosion|oxidi[sz]ed)(?![a-z])", "moss": r"(?<![a-z])(moss|mossy|lichen)(?![a-z])"}


def auto_refs(grp, listed):
    """Scans not listed by name whose id or tags name this kind of weathering (e.g. imported Megascans)."""
    import json, re
    pat = re.compile(AUTO[grp], re.I)
    out = []
    for rid, f in _index().items():
        if rid in listed:
            continue
        tags = ""
        try:
            tags = " ".join(json.loads(str(np.load(f)["meta"])).get("tags") or [])
        except Exception:  # noqa: BLE001 - scans without metadata
            pass
        if pat.search(rid) or pat.search(tags):
            out.append(rid)
    return out

def feature(lab_px, kind):
    L, a, b = lab_px[:, 0], lab_px[:, 1], lab_px[:, 2]
    if kind == "rust":   # orange-brown oxide: warm and saturated
        return (a > 6) & (b > 14) & (L > 12)
    if kind == "moss":   # green: negative a*, positive b*
        return (a < -4) & (b > 4)

def patches(mask, min_px=12):
    """Connected 4-neighbour patches of a boolean mask (NumPy min-label propagation, no SciPy).
    Returns patch areas (>= min_px pixels), largest first."""
    h, w = mask.shape
    lab_ = np.where(mask, np.arange(1, h * w + 1, dtype=np.int64).reshape(h, w), 0)
    big = np.int64(h * w + 10)
    while True:
        cur = np.where(mask, lab_, big)
        nb = cur.copy()
        nb[1:, :] = np.minimum(nb[1:, :], cur[:-1, :]); nb[:-1, :] = np.minimum(nb[:-1, :], cur[1:, :])
        nb[:, 1:] = np.minimum(nb[:, 1:], cur[:, :-1]); nb[:, :-1] = np.minimum(nb[:, :-1], cur[:, 1:])
        new = np.where(mask, nb, 0)
        if np.array_equal(new, lab_):
            break
        lab_ = new
    _, counts = np.unique(lab_[mask], return_counts=True)
    counts = np.sort(counts)[::-1]
    return counts[counts >= min_px]

def layout(mask):
    """coverage, largest patch share of the covered area, patch count."""
    cov = float(mask.mean())
    areas = patches(mask)
    if cov < 1e-4 or not len(areas):
        return {"cov": cov, "largest": 0.0, "count": 0}
    return {"cov": cov, "largest": float(areas[0] / mask.sum()), "count": int(len(areas))}

def stats(lab_px, sel):
    x = lab_px[sel]
    if len(x) < 50:
        return None
    return {"n": len(x), "L": x[:, 0].mean(), "a": x[:, 1].mean(), "b": x[:, 2].mean(), "C": np.hypot(x[:, 1], x[:, 2]).mean(), "sdL": x[:, 0].std()}

GROUPS = {
    "rust": (["rust_coarse_01", "rusted_shutter", "rusty_metal_02", "rusty_metal_grid", "rusty_metal_sheet", "rusty_metal_shutter", "rusty_painted_metal", "worn_corrugated_iron"],
             [("diamond_tread_plate", "metal", {"rust": 0.9}), ("black_steel_tread_plate", "metal", {"rust": 0.9}), ("hardops_plate", "metal", {"rust": 0.9})]),
    "moss": (["brick_moss_001", "mossy_brick_floor", "mossy_stone_wall", "moss_wood"],
             [("classic_red_brick", "brick", {"moss": 0.8}), ("fieldstone_drystack_wall", "stone", {"moss": 0.8}), ("weathered_driftwood", "wood", {"moss": 0.8})]),
}
board = []
for grp, (refs, ours) in GROUPS.items():
    refs = list(refs) + auto_refs(grp, set(refs))
    print(f"== {grp}")
    agg = {"ref": [], "ours": []}
    for r in refs:
        try:
            a = scan(r)
        except FileNotFoundError:
            continue
        a = a[:512, :512]
        lp = lab(srgb_u8_to_linear(a)); sel = feature(lp, grp); st = stats(lp, sel)
        if st:
            st.update(layout(sel.reshape(a.shape[:2])))
            agg["ref"].append(st)
            print(f"   ref  {r:26s} L {st['L']:5.1f} a {st['a']:5.1f} b {st['b']:5.1f} C {st['C']:5.1f} sdL {st['sdL']:4.1f} | cover {st['cov']:.2f} largest {st['largest']:.2f} patches {st['count']}")
        board.append((grp, "scan: " + r, a))
    for mid, kind, kw in ours:
        try:
            m = synthesize_material_maps(mid, size=512, preview=False)
        except Exception as e:
            print("   skip", mid, e); continue
        w = weather_material_maps(m, material_kind=kind, seed=7, preview=False, **kw)
        alb_lin = np.asarray(w.albedo, np.float32)  # MaterialMaps.albedo is linear light
        alb = linear_to_srgb_u8(alb_lin)
        mask = np.asarray(w.weather_mask).astype(np.float32); mask = mask / 255 if mask.max() > 1.5 else mask
        ch = mask[..., 1] if grp == "rust" else mask[..., 3]
        lp = lab(alb_lin); st = stats(lp, ch.reshape(-1) > 0.6)
        if st:
            st.update(layout(ch > 0.5))
            agg["ours"].append(st)
            print(f"   ours {mid:26s} L {st['L']:5.1f} a {st['a']:5.1f} b {st['b']:5.1f} C {st['C']:5.1f} sdL {st['sdL']:4.1f} | cover {st['cov']:.2f} largest {st['largest']:.2f} patches {st['count']}")
        else:
            print(f"   ours {mid:26s} (no pixels above 0.6) | cover {float((ch > 0.5).mean()):.3f}")
        board.append((grp, "hypbr: " + mid, alb))
    for side in ("ref", "ours"):
        v = agg[side]
        if v: print(f"   {side.upper():4s} mean: L {np.mean([s['L'] for s in v]):5.1f} a {np.mean([s['a'] for s in v]):5.1f} b {np.mean([s['b'] for s in v]):5.1f} C {np.mean([s['C'] for s in v]):5.1f} sdL {np.mean([s['sdL'] for s in v]):4.1f} | cover {np.mean([s['cov'] for s in v]):.2f} largest {np.median([s['largest'] for s in v]):.2f} (median) patches {np.median([s['count'] for s in v]):.0f} (median)")
W = 150; cols = 6
rows = (len(board) + cols - 1) // cols
img = Image.new("RGB", (W * cols, rows * (W + 14)), (24, 24, 24)); d = ImageDraw.Draw(img)
for i, (g, lab_, a) in enumerate(board):
    x, y = (i % cols) * W, (i // cols) * (W + 14)
    img.paste(Image.fromarray(a[:256, :256]).convert("RGB").resize((W, W), Image.LANCZOS), (x, y + 14))
    d.text((x + 3, y + 1), lab_[:26], fill=(230, 230, 230) if lab_.startswith("scan") else (255, 170, 90))
img.save(OUT)
