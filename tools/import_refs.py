#!/usr/bin/env python3
"""Import reference PBR scans into the 512px .npz format the calibration and weathering tools read.

Two sources:

  folder    A local folder of downloaded materials (e.g. Megascans surfaces from Fab, or any PBR set).
            Every sub-folder or .zip holding an albedo/base-colour map is one material; zips are read
            in place, never extracted.
            python tools/import_refs.py folder ~/Downloads/megascans --source megascans --out ~/hypbr-refs

  matsynth  The MatSynth dataset on Hugging Face (gvecchio/MatSynth), read shard by shard with only the
            needed columns (needs `pip install pyarrow`). CC0 items only unless --allow-cc-by.
            python tools/import_refs.py matsynth --out ~/hypbr-refs --limit 200

Each material becomes <out>/<source>/<id>.npz with
  albedo (512x512x3 uint8 sRGB), normal (512x512x3 uint8, OpenGL +Y), height (512x512 float16, 0..1),
  roughness / metallic / ao (512x512 uint8, when present) and meta (JSON: id, source, license, size_m, ...),
and one line in <out>/manifest.jsonl recording where it came from and under which license. Whole tiles
are resized to 512 (as the existing references are), so a scan's physical size is kept.

Rules this tool enforces:
  - It refuses to write inside a git working tree (imported scans must never be committed or published).
  - It stops before the disk runs low (--min-free-gb).
  - It never deletes your downloads unless --delete-originals is given.
  - Normal maps in DirectX convention are detected from their height map and flipped to OpenGL.

`hypbr atlas calibrate --refs <out>/<source> ...` and `tools/weathering_refs.py <out> ...` read the result.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from PIL import Image

SIZE = 512
MEGASCANS_LICENSE = "Fab Standard License; AI training permitted (Quixel, 2026-11-05); no standalone redistribution"
MATSYNTH_REPO = "datasets/gvecchio/MatSynth"
MATSYNTH_COLUMNS = ["name", "category", "metadata", "basecolor", "normal", "height", "roughness", "metallic"]
MATSYNTH_CATEGORIES = ("ceramic", "concrete", "fabric", "ground", "leather", "marble", "metal", "misc", "plaster",
                       "plastic", "stone", "terracotta", "wood")

# Map roles by filename keyword, most specific first. Matched against the lower-cased stem.
ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "albedo": ("basecolor", "base_color", "albedo", "diffuse", "_color", "_col_", "_col"),
    "normal": ("normal",),
    "height": ("displacement", "height", "_disp"),
    "roughness": ("roughness", "_rough"),
    "gloss": ("glossiness", "gloss"),
    "ao": ("ambientocclusion", "ambient_occlusion", "occlusion", "_ao"),
    "metallic": ("metalness", "metallic", "_metal"),
}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")


# ---------------------------------------------------------------------------- decoding helpers

def _open(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im.load()
    return im


def _square(im: Image.Image) -> Image.Image:
    w, h = im.size
    if w == h:
        return im
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def decode_rgb(data: bytes) -> np.ndarray:
    im = _square(_open(data).convert("RGB"))
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def decode_gray(data: bytes) -> np.ndarray:
    """Single channel in [0, 1] (16-bit aware), area-resized to SIZE."""
    im = _square(_open(data))
    arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    scale = 65535.0 if (arr.dtype in (np.uint16, np.int32) or arr.max() > 255) else 255.0
    f = Image.fromarray(arr.astype(np.float32) / scale, mode="F")
    if f.size != (SIZE, SIZE):
        f = f.resize((SIZE, SIZE), Image.BOX)
    return np.clip(np.asarray(f, dtype=np.float32), 0.0, 1.0)


def height_normal_consistency(h: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """(x, y) correlation between the height map's slope and an OpenGL normal map; y < 0 means DirectX."""
    if float(h.std()) < 0.01:
        return float("nan"), float("nan")
    gx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    gy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    sl = slice(8, -8)
    cx = float(np.corrcoef(-gx[sl, sl].ravel(), n[sl, sl, 0].ravel())[0, 1])
    cy = float(np.corrcoef(gy[sl, sl].ravel(), n[sl, sl, 1].ravel())[0, 1])
    return cx, cy


def to_gl_normal(n_u8: np.ndarray, height: np.ndarray | None, assume: str) -> tuple[np.ndarray, str, float]:
    """Return an OpenGL normal map, the detected/assumed source convention, and the height consistency."""
    n = n_u8.astype(np.float32) / 127.5 - 1.0
    if height is not None:
        cx, cy = height_normal_consistency(height, n)
        if np.isfinite(cy) and abs(cy) > 0.05:
            conv = "dx" if cy < 0 else "gl"
            if conv == "dx":
                n_u8 = n_u8.copy()
                n_u8[..., 1] = 255 - n_u8[..., 1]
            return n_u8, f"{conv} (detected)", round(0.5 * (cx + abs(cy)), 3)
    if assume == "dx":
        n_u8 = n_u8.copy()
        n_u8[..., 1] = 255 - n_u8[..., 1]
    return n_u8, f"{assume} (assumed)", float("nan")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80] or "material"


# ---------------------------------------------------------------------------- output

@dataclass
class Material:
    id: str
    source: str
    license: str
    maps: dict[str, bytes]          # role -> encoded image bytes
    meta: dict[str, Any]
    origin: str                     # path or URL it came from
    normal_hint: str = "auto"       # "gl", "dx" or "auto"


def write_material(mat: Material, out_root: Path, default_normal: str = "gl") -> dict[str, Any] | None:
    """Decode, normalize and save one material. Returns its manifest record, or None when unusable."""
    if "albedo" not in mat.maps or "normal" not in mat.maps:
        return None
    arrays: dict[str, np.ndarray] = {"albedo": decode_rgb(mat.maps["albedo"])}
    height = decode_gray(mat.maps["height"]) if "height" in mat.maps else None
    hint = mat.normal_hint if mat.normal_hint in ("gl", "dx") else default_normal
    normal, conv, consistency = to_gl_normal(decode_rgb(mat.maps["normal"]), height, hint)
    arrays["normal"] = normal
    if height is not None:
        arrays["height"] = height.astype(np.float16)
    if "roughness" in mat.maps:
        arrays["roughness"] = np.clip(decode_gray(mat.maps["roughness"]) * 255 + 0.5, 0, 255).astype(np.uint8)
    elif "gloss" in mat.maps:
        arrays["roughness"] = np.clip((1.0 - decode_gray(mat.maps["gloss"])) * 255 + 0.5, 0, 255).astype(np.uint8)
    for role in ("metallic", "ao"):
        if role in mat.maps:
            arrays[role] = np.clip(decode_gray(mat.maps[role]) * 255 + 0.5, 0, 255).astype(np.uint8)
    meta = {"id": mat.id, "source": mat.source, "license": mat.license, "normal_convention": conv,
            "consistency": consistency if np.isfinite(consistency) else None, **mat.meta}
    dest = out_root / mat.source / f"{mat.id}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, meta=np.array(json.dumps(meta)), **arrays)
    record = {**meta, "file": str(dest), "origin": mat.origin, "maps": sorted(arrays),
              "redistributable": False, "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(out_root / "manifest.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


# ---------------------------------------------------------------------------- folder source

def classify(name: str) -> tuple[str | None, str]:
    """(role, normal convention hint) for an image filename."""
    stem = Path(name).stem.lower()
    low = "_" + stem + "_"
    for role, pats in ROLE_PATTERNS.items():
        if any(p in low for p in pats):
            if role == "albedo" and ("normal" in low or "ao" in low and "_ao_" in low):
                continue
            hint = "auto"
            if role == "normal":
                if re.search(r"(normal_?dx|_dx_|directx)", low):
                    hint = "dx"
                elif re.search(r"(normal_?gl|_gl_|opengl)", low):
                    hint = "gl"
            return role, hint
    return None, "auto"


def _pick(files: Iterable[tuple[str, Callable[[], bytes]]]) -> tuple[dict[str, Callable[[], bytes]], str]:
    chosen: dict[str, tuple[int, Callable[[], bytes]]] = {}
    hint = "auto"
    for name, reader in files:
        if not name.lower().endswith(IMAGE_EXT) or "preview" in name.lower() or "thumb" in name.lower():
            continue
        role, h = classify(name)
        if role is None:
            continue
        # prefer the largest-resolution file per role (resolution tokens like 2K/4K/8K in the name)
        m = re.search(r"(\d)k", Path(name).stem.lower())
        rank = int(m.group(1)) if m else 0
        if role not in chosen or rank > chosen[role][0]:
            chosen[role] = (rank, reader)
            if role == "normal":
                hint = h
    return {r: rd for r, (_, rd) in chosen.items()}, hint


def iter_folder(root: Path, source: str, license_: str) -> Iterator[tuple[Material, Path]]:
    """Yield (material, original path) for each sub-folder or zip under root holding albedo + normal."""
    seen: set[Path] = set()
    for zp in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(zp) as zf:
            files = [(i.filename, (lambda n=i.filename: zf.read(n))) for i in zf.infolist() if not i.is_dir()]
            readers, hint = _pick(files)
            if "albedo" in readers and "normal" in readers:
                maps = {r: rd() for r, rd in readers.items()}
                yield Material(f"{source[:4]}_{slug(zp.stem)}", source, license_, maps, {"name": zp.stem}, str(zp), hint), zp
        seen.add(zp)
    for d in sorted({p.parent for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT}):
        files = [(p.name, (lambda p=p: p.read_bytes())) for p in sorted(d.iterdir()) if p.is_file()]
        readers, hint = _pick(files)
        if "albedo" in readers and "normal" in readers:
            maps = {r: rd() for r, rd in readers.items()}
            yield Material(f"{source[:4]}_{slug(d.name)}", source, license_, maps, {"name": d.name}, str(d), hint), d


# ---------------------------------------------------------------------------- MatSynth source

def matsynth_keep(meta: dict[str, Any], allow_cc_by: bool, skip_sources: tuple[str, ...]) -> tuple[bool, str]:
    lic = str(meta.get("license") or "")
    if lic.upper() != "CC0" and not (allow_cc_by and lic.upper().startswith("CC-BY")):
        return False, f"license {lic or '?'}"
    src = str(meta.get("source") or "").lower()
    for s in skip_sources:
        if s and s in src:
            return False, f"source {src}"
    return True, ""


def matsynth_material(row: dict[str, Any]) -> Material:
    meta = row.get("metadata") or {}
    cat = row.get("category")
    cat_name = MATSYNTH_CATEGORIES[cat] if isinstance(cat, int) and 0 <= cat < len(MATSYNTH_CATEGORIES) else str(meta.get("category") or cat or "misc")
    maps = {}
    for col, role in (("basecolor", "albedo"), ("normal", "normal"), ("height", "height"), ("roughness", "roughness"), ("metallic", "metallic")):
        cell = row.get(col)
        data = cell.get("bytes") if isinstance(cell, dict) else cell
        if data:
            maps[role] = data
    name = str(row.get("name") or meta.get("name") or "material")
    size = meta.get("physical_size")
    keep = {"name": name, "category": cat_name, "tags": list(meta.get("tags") or []), "method": meta.get("method"),
            "upstream_source": meta.get("source"), "authors": list(meta.get("authors") or []), "link": meta.get("link")}
    if isinstance(size, (int, float)) and size > 0:
        keep["size_m"] = float(size)
    return Material(f"msyn_{slug(cat_name)}_{slug(name)}", "matsynth", str(meta.get("license") or "CC0"), maps, keep,
                    str(meta.get("link") or f"hf://{MATSYNTH_REPO}"), "gl")  # card: "We adopt the OpenGL standard"


def iter_matsynth(split: str, allow_cc_by: bool, skip_sources: tuple[str, ...], log: Callable[[str], None]) -> Iterator[dict[str, Any]]:
    """Stream MatSynth rows shard by shard, reading only MATSYNTH_COLUMNS (row-group sized, never whole shards)."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
    except ImportError as e:  # pragma: no cover - depends on optional extras
        raise SystemExit(f"matsynth needs pyarrow and huggingface_hub ({e}); pip install pyarrow huggingface_hub")
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"{MATSYNTH_REPO}/data/{split}-*.parquet"))
    log(f"{len(shards)} {split} shards")
    for shard in shards:
        with fs.open(shard, "rb") as fh:
            pf = pq.ParquetFile(fh)
            for g in range(pf.num_row_groups):
                metas = pf.read_row_group(g, columns=["metadata"]).column("metadata").to_pylist()
                if not any(matsynth_keep(m or {}, allow_cc_by, skip_sources)[0] for m in metas):
                    continue
                table = pf.read_row_group(g, columns=MATSYNTH_COLUMNS)
                for row in table.to_pylist():
                    ok, why = matsynth_keep(row.get("metadata") or {}, allow_cc_by, skip_sources)
                    if ok:
                        yield row
                    else:
                        log(f"skip {row.get('name')}: {why}")


# ---------------------------------------------------------------------------- CLI

def inside_git_tree(path: Path) -> Path | None:
    p = path.resolve()
    for cand in (p, *p.parents):
        if (cand / ".git").exists():
            return cand
    return None


def free_gb(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def run(args: argparse.Namespace, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    out = Path(args.out).expanduser()
    repo = inside_git_tree(out)
    if repo is not None and not args.allow_in_repo:
        raise SystemExit(f"refusing to write inside git working tree {repo}: imported scans must not be committed "
                         "(choose another --out, or pass --allow-in-repo if that folder is git-ignored)")
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    records: list[dict[str, Any]] = []
    if args.cmd == "folder":
        items = ((m, orig) for m, orig in iter_folder(Path(args.root).expanduser(), args.source, args.license))
    else:
        skip = tuple(s.strip().lower() for s in args.skip_sources.split(",") if s.strip())
        items = ((matsynth_material(r), None) for r in iter_matsynth(args.split, args.allow_cc_by, skip, log))
    for mat, orig in items:
        if args.limit and done >= args.limit:
            break
        if free_gb(out) < args.min_free_gb:
            log(f"stopping: less than {args.min_free_gb} GB free")
            break
        if (out / mat.source / f"{mat.id}.npz").exists() and not args.overwrite:
            log(f"have {mat.id}")
            continue
        rec = write_material(mat, out, args.default_normal)
        if rec is None:
            log(f"skip {mat.id}: needs an albedo and a normal map")
            continue
        records.append(rec)
        done += 1
        log(f"[{done}] {mat.id}  normal {rec['normal_convention']}  maps {','.join(rec['maps'])}")
        if orig is not None and getattr(args, "delete_originals", False):
            (shutil.rmtree if orig.is_dir() else Path.unlink)(orig)
            log(f"    deleted original {orig}")
    log(f"imported {done} into {out}")
    return records


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("folder", "matsynth"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="Reference root (outside any git repo)")
        p.add_argument("--limit", type=int, default=0, help="Import at most N materials (0 = all)")
        p.add_argument("--min-free-gb", type=float, default=2.0, help="Stop before free disk drops below this")
        p.add_argument("--overwrite", action="store_true", help="Re-import materials that already exist")
        p.add_argument("--default-normal", choices=("gl", "dx"), default="gl",
                       help="Convention to assume when a normal map can't be checked against a height map")
        p.add_argument("--allow-in-repo", action="store_true", help=argparse.SUPPRESS)
        if name == "folder":
            p.add_argument("root", help="Folder of downloaded materials (sub-folders and/or zips)")
            p.add_argument("--source", default="megascans", help="Source name, also the output sub-folder")
            p.add_argument("--license", default=None, help="License string recorded per item (default for megascans)")
            p.add_argument("--delete-originals", action="store_true",
                           help="Delete each source folder/zip after it was imported (permanent)")
        else:
            p.add_argument("--split", default="train", choices=("train", "test"))
            p.add_argument("--allow-cc-by", action="store_true", help="Also import CC-BY items (attribution recorded)")
            p.add_argument("--skip-sources", default="polyhaven,ambientcg,deschaintre",
                           help="Upstream sources to skip: already in the reference set, or license not verified")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "folder" and args.license is None:
        if args.source != "megascans":
            raise SystemExit("--license is required for sources other than megascans")
        args.license = MEGASCANS_LICENSE
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
