"""In-memory PBR material maps (`MaterialMaps`) and the directory <-> memory bridge.

`synthesize_material_maps()`, `blend_materials_maps()` and `weather_material_maps()` return a
`MaterialMaps`; `.save(dir)` writes the exact same PNG files the directory-based functions
write (byte-identical), and `MaterialMaps.load(dir)` reads any hypbr material directory back.
`blend_materials`, `weather_material` and `export_material` accept either form.

Conventions (all arrays `float32`, `(H, W)` or `(H, W, C)`, values in `[0, 1]` unless noted):
  - `albedo` is *linear-light* RGB (encoded to sRGB on save).
  - `normal` / `bent_normal` are tangent-space unit vectors in `[-1, 1]`, OpenGL (+Y up).
  - `height` is normalized relief (saved as 8-bit `height.png` and 16-bit `height16.png`).
  - `curvature` is signed around 0.5 (0.5 = flat, > 0.5 convex), `edge_wear` the convex mask.
  - `anisotropy` stores the glTF flow map as saved: `R = cos*0.5+0.5, G = sin*0.5+0.5, B = strength`.
  - `preview` is the studio render as 8-bit sRGB (`uint8`, display only).
  - `orm.png` is packed from `ao`, `roughness`, `metallic` on save.
"""
from __future__ import annotations

import copy as _copy
import io
import json
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from .color import linear_to_srgb_u8, srgb_u8_to_linear

METADATA_FILE = "hypbr_material.json"

# Scalar / vector maps in save order: (field, file stem, kind)
_MAP_FILES: tuple[tuple[str, str, str], ...] = (
    ("albedo", "albedo_delit", "srgb"),
    ("preview", "preview", "raw"),
    ("height", "height", "u8"),
    ("height", "height16", "u16"),
    ("normal", "normal", "normal"),
    ("bent_normal", "bent_normal", "normal"),
    ("ao", "ao", "u8"),
    ("roughness", "roughness", "u8"),
    ("metallic", "metallic", "u8"),
    ("orm", "orm", "orm"),
    ("curvature", "curvature", "u8"),
    ("cavity", "cavity", "u8"),
    ("edge_wear", "edge_wear", "u8"),
    ("anisotropy", "anisotropy", "u8"),
    ("specular", "specular", "u8"),
    ("thickness", "thickness", "u8"),
    ("weather_mask", "weather_mask", "u8"),
    ("blend_mask", "blend_mask", "u8"),
)
_STEM_TO_SPEC = {stem: (fld, kind) for fld, stem, kind in _MAP_FILES}


def to_u8(x: np.ndarray) -> np.ndarray:
    """`[0, 1]` float -> rounded 8-bit (the exact quantizer every hypbr map uses)."""
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_u16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 65535.0 + 0.5, 0, 65535).astype(np.uint16)


def unit_normals(N: np.ndarray) -> np.ndarray:
    """Renormalize tangent-space normals exactly as `depthpro_pbr.normal_to_u8` does before encoding."""
    return N / (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-8)


def normal_to_u8(N_unit: np.ndarray) -> np.ndarray:
    """Encode (already unit) normals to 8-bit RGB; `normal_to_u8(unit_normals(N)) == dp.normal_to_u8(N)`."""
    return np.clip((N_unit * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _stem(name: str) -> str:
    return name[:-4] if name.endswith(".png") else name


@dataclass
class MaterialMaps:
    """A complete seamless PBR material held in memory (see module docstring for conventions)."""

    albedo: np.ndarray
    normal: np.ndarray
    height: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    ao: np.ndarray
    bent_normal: np.ndarray | None = None
    curvature: np.ndarray | None = None
    cavity: np.ndarray | None = None
    edge_wear: np.ndarray | None = None
    anisotropy: np.ndarray | None = None
    specular: np.ndarray | None = None
    thickness: np.ndarray | None = None
    preview: np.ndarray | None = None
    weather_mask: np.ndarray | None = None
    blend_mask: np.ndarray | None = None
    #: Additional pre-encoded images written verbatim as `<name>.png` (e.g. `source_512`).
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    #: Free-form provenance: `material_id`, `size`, `seed`, `calibrated`, weathering params, ...
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        hw = self.albedo.shape[:2]
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray) and v.shape[:2] != hw:
                raise ValueError(f"MaterialMaps.{f.name} has shape {v.shape}, expected {hw} like albedo")

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, int]:
        """`(H, W)` in pixels."""
        return tuple(self.albedo.shape[:2])  # type: ignore[return-value]

    @property
    def size(self) -> int:
        """Square side length (`H` for non-square maps)."""
        return int(self.albedo.shape[0])

    @property
    def material_id(self) -> str | None:
        return self.metadata.get("material_id")

    @property
    def orm(self) -> np.ndarray:
        """Packed glTF occlusion/roughness/metallic as float `(H, W, 3)` (8-bit quantized)."""
        return self.encode("orm").astype(np.float32) / 255.0

    def copy(self) -> "MaterialMaps":
        return _copy.deepcopy(self)

    # ------------------------------------------------------------- encoding
    def file_names(self) -> list[str]:
        """PNG file names `.save()` writes for this material, in order."""
        names = [f"{stem}.png" for fld, stem, _k in _MAP_FILES if fld == "orm" or getattr(self, fld) is not None]
        if self.extras.get("source_512") is not None:  # keep historic position right after albedo
            names.insert(1, "source_512.png")
        names += [f"{k}.png" for k in self.extras if k != "source_512"]
        return names

    def has(self, name: str) -> bool:
        return f"{_stem(name)}.png" in self.file_names()

    def encode(self, name: str) -> np.ndarray:
        """Exact 8/16-bit image array for a map file (e.g. `"normal"` or `"height16.png"`)."""
        stem = _stem(name)
        if stem in self.extras:
            return np.asarray(self.extras[stem])
        if stem not in _STEM_TO_SPEC:
            raise KeyError(f"Unknown map {name!r}; known: {', '.join(self.file_names())}")
        fld, kind = _STEM_TO_SPEC[stem]
        if kind == "orm":
            return np.stack([to_u8(self.ao), to_u8(self.roughness), to_u8(self.metallic)], axis=-1)
        val = getattr(self, fld)
        if val is None:
            raise KeyError(f"MaterialMaps.{fld} is not set")
        if kind == "srgb":
            return linear_to_srgb_u8(val)
        if kind == "raw":
            return np.asarray(val, dtype=np.uint8)
        if kind == "u16":
            return to_u16(val)
        if kind == "normal":
            return normal_to_u8(val)
        return to_u8(val)

    def encoded(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield `(file_name, array)` for every file `.save()` writes."""
        done: dict[str, np.ndarray] = {}
        for name in self.file_names():
            if name == "orm.png" and {"ao.png", "roughness.png", "metallic.png"} <= done.keys():
                arr = np.stack([done["ao.png"], done["roughness.png"], done["metallic.png"]], axis=-1)
            else:
                arr = self.encode(name)
            done[name] = arr
            yield name, arr

    def png_bytes(self, name: str, compress_level: int = 4) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(self.encode(name)).save(buf, format="PNG", compress_level=compress_level)
        return buf.getvalue()

    # ------------------------------------------------------------ disk I/O
    def save(self, out_dir: Path | str, *, compress_level: int = 4, write_metadata: bool = False) -> Path:
        """Write the PNG map set (identical to `synthesize_material`'s output) and return `out_dir`.

        `write_metadata=True` also writes `hypbr_material.json` with `metadata`.
        """
        from .synthesize import save_maps_parallel

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        save_maps_parallel([(d / name, arr) for name, arr in self.encoded()], compress_level=compress_level)
        if write_metadata:
            (d / METADATA_FILE).write_text(json.dumps(self.metadata, indent=2, sort_keys=True, default=str))
        return d

    @classmethod
    def load(cls, src_dir: Path | str) -> "MaterialMaps":
        """Read a hypbr material directory (any version) into memory.

        Missing core maps fall back to neutral values (roughness 0.5, metallic 0, ao 1, flat normal,
        mid height). Unknown PNGs are kept verbatim in `extras`. Decoded maps re-encode to the
        same pixels, so `MaterialMaps.load(d).save(d2)` reproduces every map file of `d`.
        """
        d = Path(src_dir)
        alb_p = d / "albedo_delit.png"
        if not alb_p.exists():
            raise FileNotFoundError(f"No albedo_delit.png in {d}")
        with Image.open(alb_p) as im:
            albedo_u8 = np.asarray(im.convert("RGB"), dtype=np.uint8)
        H, W = albedo_u8.shape[:2]

        def read(stem: str, mode: str | None = None) -> np.ndarray | None:
            p = d / f"{stem}.png"
            if not p.exists():
                return None
            with Image.open(p) as im:
                if mode is not None and im.mode != mode:
                    im = im.convert(mode)
                arr = np.asarray(im)
            if arr.shape[:2] != (H, W):
                raise ValueError(f"{p.name} is {arr.shape[:2]}, albedo is {(H, W)}; resize maps first")
            return arr

        def gray(stem: str, default: float | None) -> np.ndarray | None:
            a = read(stem, "L")
            if a is None:
                return None if default is None else np.full((H, W), default, dtype=np.float32)
            return a.astype(np.float32) / 255.0

        def vec(stem: str, mode: str = "RGB") -> np.ndarray | None:
            a = read(stem, mode)
            return None if a is None else a.astype(np.float32) / 255.0

        def nrm(stem: str) -> np.ndarray | None:
            a = read(stem, "RGB")
            return None if a is None else (a.astype(np.float32) / 127.5 - 1.0).astype(np.float32)

        h16 = read("height16")
        if h16 is not None:
            height = (h16.astype(np.float32) / (65535.0 if h16.dtype != np.uint8 else 255.0)).astype(np.float32)
        else:
            height = gray("height", 0.5)
        normal = nrm("normal")
        if normal is None:
            normal = np.zeros((H, W, 3), dtype=np.float32)
            normal[..., 2] = 1.0
        prev = read("preview", "RGB")

        known = {f"{stem}.png" for _f, stem, _k in _MAP_FILES} | {METADATA_FILE}
        extras = {}
        for p in sorted(d.glob("*.png")):
            if p.name not in known:
                with Image.open(p) as im:
                    extras[p.stem] = np.asarray(im)
        meta: dict[str, Any] = {}
        mp = d / METADATA_FILE
        if mp.exists():
            meta = json.loads(mp.read_text())
        meta.setdefault("material_id", d.name)
        meta.setdefault("source_dir", str(d))

        return cls(
            albedo=srgb_u8_to_linear(albedo_u8),
            normal=normal,
            height=height,
            roughness=gray("roughness", 0.5),
            metallic=gray("metallic", 0.0),
            ao=gray("ao", 1.0),
            bent_normal=nrm("bent_normal"),
            curvature=gray("curvature", None),
            cavity=gray("cavity", None),
            edge_wear=gray("edge_wear", None),
            anisotropy=vec("anisotropy"),
            specular=gray("specular", None),
            thickness=gray("thickness", None),
            preview=None if prev is None else np.asarray(prev, dtype=np.uint8),
            weather_mask=vec("weather_mask", "RGBA"),
            blend_mask=gray("blend_mask", None),
            extras=extras,
            metadata=meta,
        )

    def __repr__(self) -> str:
        have = [f.name for f in fields(self) if isinstance(getattr(self, f.name), np.ndarray)]
        return f"MaterialMaps({self.material_id or '?'}, {self.shape[1]}x{self.shape[0]}, maps={have})"



class MapSource:
    """Read-only view of a material given as a directory **or** a `MaterialMaps`.

    Images are served exactly as they would be read back from the saved PNGs, so functions that
    consume a `MapSource` produce identical results for `maps` and `maps.save(dir)`.
    """

    def __init__(self, src: "Path | str | MaterialMaps | MapSource"):
        if isinstance(src, MapSource):
            self.dir, self.maps, self.name = src.dir, src.maps, src.name
            self._cache = src._cache
            return
        self._cache: dict[str, np.ndarray] = {}
        if isinstance(src, MaterialMaps):
            self.dir = None
            self.maps = src
            self.name = str(src.metadata.get("material_id") or "material")
        else:
            self.dir = Path(src)
            self.maps = None
            self.name = self.dir.name

    def __str__(self) -> str:
        return str(self.dir) if self.dir is not None else repr(self.maps)

    def exists(self, fname: str) -> bool:
        if self.dir is not None:
            return (self.dir / fname).exists()
        return self.maps.has(fname)

    def array(self, fname: str) -> np.ndarray:
        if fname not in self._cache:
            self._cache[fname] = self.maps.encode(fname)
        return self._cache[fname]

    def open(self, fname: str) -> Image.Image:
        """PIL image of a map file (raises `FileNotFoundError` if absent)."""
        if self.dir is not None:
            return Image.open(self.dir / fname)
        if not self.maps.has(fname):
            raise FileNotFoundError(f"{fname} not present in {self.maps!r}")
        return Image.fromarray(self.array(fname))

    def copy_to(self, fname: str, dst: Path) -> bool:
        """Copy a map file to `dst` (byte-identical to a saved directory); False if absent."""
        if not self.exists(fname):
            return False
        if self.dir is not None:
            shutil.copy2(self.dir / fname, dst)
        else:
            Image.fromarray(self.array(fname)).save(dst, compress_level=4)
        return True

    def read_bytes(self, fname: str) -> bytes:
        if self.dir is not None:
            return (self.dir / fname).read_bytes()
        if not self.maps.has(fname):
            raise FileNotFoundError(f"{fname} not present in {self.maps!r}")
        return self.maps.png_bytes(fname)


__all__ = ["MapSource", "MaterialMaps", "normal_to_u8", "to_u16", "to_u8", "unit_normals"]
