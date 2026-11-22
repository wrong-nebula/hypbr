#!/usr/bin/env python3
"""CPU-only HTTP server and disk-cached synthesis backend for the hypbr PBR Playground.

STRICT HARDWARE CONSTRAINT:
  Forces HYPBR_DEVICE=cpu BEFORE importing hypbr/torch so background MPS GPU jobs
  are never touched or contended with.
"""
from __future__ import annotations

import os

# CRITICAL: Must be set before importing torch or hypbr!
os.environ["HYPBR_DEVICE"] = "cpu"

import argparse
import hashlib
import io
import json
import mimetypes
import sys
import threading
import time
import urllib.parse
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PLAYGROUND_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLAYGROUND_DIR.parent
# Works both from `<workspace>/playground` (repo at `<workspace>/hypbr`) and `<repo>/playground`
for _cand in (REPO_ROOT, REPO_ROOT / "hypbr"):
    if (_cand / "hypbr" / "synthesize.py").is_file():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

import hypbr  # noqa: E402
from hypbr import depthpro_pbr as dp  # noqa: E402
from hypbr import procedural as gd  # noqa: E402
from hypbr.verify import verify_material  # noqa: E402
from hypbr.weathering import (  # noqa: E402
    WEATHER_ARCHETYPES,
    WEATHER_PRESETS,
    infer_material_kind,
    resolve_weather_params,
    weather_material,
)

# Double-check CPU isolation at startup
assert dp.DEVICE.type == "cpu", f"Expected CPU device, got {dp.DEVICE}"

CACHE_DIR = PLAYGROUND_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SIZES = (256, 512)
DEFAULT_SIZE = 512
WEATHER_CHANNELS = ("wear", "dirt", "wet", "moss", "rust", "dust")

MAP_KEYS = (
    ("albedo_delit", "albedo_delit.png"),
    ("normal", "normal.png"),
    ("roughness", "roughness.png"),
    ("metallic", "metallic.png"),
    ("ao", "ao.png"),
    ("height", "height.png"),
    ("height16", "height16.png"),
    ("curvature", "curvature.png"),
    ("cavity", "cavity.png"),
    ("edge_wear", "edge_wear.png"),
    ("bent_normal", "bent_normal.png"),
    ("orm", "orm.png"),
    ("anisotropy", "anisotropy.png"),
    ("specular", "specular.png"),
    ("thickness", "thickness.png"),
    ("weather_mask", "weather_mask.png"),
    ("preview", "preview.png"),
)

KIND_TO_GROUP: dict[str, str] = {
    "metal": "Metal",
    "copper": "Metal",
    "bronze": "Metal",
    "brass": "Metal",
    "stone": "Stone & Concrete",
    "concrete": "Stone & Concrete",
    "asphalt": "Stone & Concrete",
    "plaster": "Stone & Concrete",
    "wood": "Wood & Natural",
    "natural": "Wood & Natural",
    "paper": "Wood & Natural",
    "ceramic": "Ceramic & Glass",
    "glass": "Ceramic & Glass",
    "fabric": "Fabric & Leather",
    "leather": "Fabric & Leather",
    "rubber": "Rubber & Polymer",
    "carbon": "Rubber & Polymer",
}

from hypbr.atlas_catalog import CATEGORIES as ATLAS_CATEGORIES  # noqa: E402

GROUP_ORDER = ["All"] + [c["name"] for c in ATLAS_CATEGORIES]

# Approximate per-kind defaults for catalog browsing before synthesis
KIND_DEFAULT_META: dict[str, tuple[float, float]] = {
    "metal": (0.28, 0.95),
    "copper": (0.26, 0.96),
    "bronze": (0.34, 0.85),
    "brass": (0.24, 0.95),
    "stone": (0.55, 0.0),
    "concrete": (0.78, 0.0),
    "asphalt": (0.82, 0.0),
    "plaster": (0.64, 0.0),
    "wood": (0.52, 0.0),
    "natural": (0.74, 0.0),
    "paper": (0.84, 0.0),
    "ceramic": (0.22, 0.0),
    "glass": (0.12, 0.0),
    "fabric": (0.86, 0.0),
    "leather": (0.42, 0.0),
    "rubber": (0.68, 0.0),
    "carbon": (0.20, 0.42),
}


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    vals = [max(0, min(255, int(round(c * 255.0)))) for c in rgb]
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _format_title(mat_id: str) -> str:
    acronyms = {"hvac": "HVAC", "pcb": "PCB", "hardops": "HardOps"}
    parts = mat_id.split("_")
    return " ".join(acronyms.get(p, p.capitalize()) for p in parts)


def _build_catalog() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """All 500 atlas materials (lean playground rows shared with the static baker)."""
    from hypbr.atlas import playground_entries

    items = playground_entries(asset_prefix=None)
    return items, {e["id"]: e for e in items}


CATALOG_LIST, CATALOG_BY_ID = _build_catalog()


# Per-key lock registry so concurrent requests for the same hash don't duplicate CPU work
_KEY_LOCKS_GUARD = threading.Lock()
_KEY_LOCKS: dict[str, threading.Lock] = {}


def _get_key_lock(key: str) -> threading.Lock:
    with _KEY_LOCKS_GUARD:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _KEY_LOCKS[key] = lock
        return lock


def _canonical_weather(
    weather_preset: str | None = None,
    *,
    wear: float | None = None,
    dirt: float | None = None,
    wet: float | None = None,
    moss: float | None = None,
    rust: float | None = None,
    dust: float | None = None,
) -> dict[str, float]:
    preset_clean = None
    if weather_preset is not None:
        p = str(weather_preset).strip().lower()
        if p and p != "fresh":
            preset_clean = p
    raw = resolve_weather_params(
        preset_clean,
        wear=wear,
        dirt=dirt,
        wet=wet,
        moss=moss,
        rust=rust,
        dust=dust,
    )
    return {k: round(float(raw[k]), 3) for k in WEATHER_CHANNELS}


def _synth_cache_key(mat_id: str, size: int, seed: int, weather: dict[str, float]) -> str:
    payload = {
        "v": 2,
        "type": "material",
        "id": mat_id,
        "size": size,
        "seed": seed,
        "weather": weather,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _blend_cache_key(
    id_a: str,
    id_b: str,
    size: int,
    seed_a: int,
    seed_b: int,
    factor: float,
    contrast: float,
    mode: str,
) -> str:
    payload = {
        "v": 2,
        "type": "blend",
        "id_a": id_a,
        "id_b": id_b,
        "size": size,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "factor": round(factor, 3),
        "contrast": round(contrast, 3),
        "mode": mode,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "b_" + hashlib.sha256(blob).hexdigest()[:16]


def _collect_map_urls(cache_key: str, out_dir: Path) -> dict[str, str | None]:
    maps: dict[str, str | None] = {}
    for key, fname in MAP_KEYS:
        if (out_dir / fname).is_file():
            maps[key] = f"/cache/{cache_key}/{fname}"
        else:
            maps[key] = None
    return maps


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _summarize_verify(report: dict[str, Any]) -> dict[str, Any]:
    checks = report.get("checks", {})
    seamless = checks.get("seamless", {})
    seam_ratios = {
        k: seamless[k]["seam_ratio"]
        for k in ("height", "normal", "albedo")
        if k in seamless and "seam_ratio" in seamless[k]
    }
    max_seam_ratio = max(seam_ratios.values()) if seam_ratios else None
    nrm = checks.get("normal", {})
    pbr = checks.get("pbr_ranges", {})
    return {
        "passed": bool(report.get("passed", False)),
        "errors": list(report.get("errors", [])),
        "warnings": list(report.get("warnings", [])),
        "resolution": checks.get("resolution"),
        "max_seam_ratio": round(max_seam_ratio, 3) if max_seam_ratio is not None else None,
        "seam_ratios": seam_ratios,
        "normal_unit_error": nrm.get("unit_length_error"),
        "normal_pos_z": nrm.get("positive_z_fraction"),
        "normal_curl_ratio": nrm.get("curl_residual_ratio"),
        "orm_max_diff": checks.get("orm_max_channel_diff"),
        "pbr_ranges": pbr,
    }


def synthesize_cached(
    mat_id: str,
    *,
    size: int = DEFAULT_SIZE,
    seed: int | None = None,
    weather_preset: str | None = None,
    wear: float | None = None,
    dirt: float | None = None,
    wet: float | None = None,
    moss: float | None = None,
    rust: float | None = None,
    dust: float | None = None,
) -> dict[str, Any]:
    if mat_id not in CATALOG_BY_ID:
        raise KeyError(f"Unknown material ID '{mat_id}'")
    if size not in ALLOWED_SIZES:
        raise ValueError(f"Invalid size {size}. Allowed: {ALLOWED_SIZES}")

    entry = CATALOG_BY_ID[mat_id]
    eff_seed = int(entry["default_seed"] if seed is None else seed)
    weather = _canonical_weather(
        weather_preset,
        wear=wear,
        dirt=dirt,
        wet=wet,
        moss=moss,
        rust=rust,
        dust=dust,
    )
    has_any_weather = any(v > 1e-4 for v in weather.values())
    cache_key = _synth_cache_key(mat_id, size, eff_seed, weather)
    out_dir = CACHE_DIR / cache_key
    meta_path = out_dir / "meta.json"

    lock = _get_key_lock(cache_key)
    with lock:
        if meta_path.is_file() and (out_dir / "albedo_delit.png").is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["cached"] = True
            meta["maps"] = _collect_map_urls(cache_key, out_dir)
            return meta

        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        if mat_id == "hardops_plate":
            if has_any_weather:
                # Synthesize/reuse fresh HardOps plate first, then apply physical weathering
                fresh_key = _synth_cache_key(
                    mat_id, size, eff_seed, {k: 0.0 for k in WEATHER_CHANNELS}
                )
                fresh_dir = CACHE_DIR / fresh_key
                if not (fresh_dir / "meta.json").is_file():
                    synthesize_cached(mat_id, size=size, seed=eff_seed)
                weather_material(
                    fresh_dir,
                    out_dir,
                    wear=weather["wear"],
                    dirt=weather["dirt"],
                    wet=weather["wet"],
                    moss=weather["moss"],
                    rust=weather["rust"],
                    dust=weather["dust"],
                    material_kind="metal",
                    seed=eff_seed,
                )
            else:
                hypbr.synthesize_hardops_plate(out_dir, size=size)
        else:
            has_multi = any(weather[k] > 1e-4 for k in ("dirt", "wet", "moss", "rust", "dust"))
            if has_multi:
                hypbr.synthesize_material(
                    mat_id,
                    out_dir,
                    size=size,
                    seed=eff_seed,
                    wear=weather["wear"],
                    weather_preset=None,
                    dirt=weather["dirt"],
                    wet=weather["wet"],
                    moss=weather["moss"],
                    rust=weather["rust"],
                    dust=weather["dust"],
                )
            else:
                hypbr.synthesize_material(
                    mat_id,
                    out_dir,
                    size=size,
                    seed=eff_seed,
                    wear=weather["wear"],
                )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        qa_raw = verify_material(out_dir)
        qa = _summarize_verify(qa_raw)

        result = {
            "cache_key": cache_key,
            "id": mat_id,
            "name": entry["name"],
            "kind": entry["kind"],
            "group": entry["group"],
            "description": entry["description"],
            "size": size,
            "seed": eff_seed,
            "weather": weather,
            "is_weathered": has_any_weather,
            "elapsed_ms": elapsed_ms,
            "cached": False,
            "cache_dir": str(out_dir),
            "maps": _collect_map_urls(cache_key, out_dir),
            "verify": qa,
            "download_url": f"/api/download/{cache_key}.zip",
        }
        _write_json_atomic(meta_path, result)
        return result


def blend_cached(
    id_a: str,
    id_b: str,
    *,
    size: int = DEFAULT_SIZE,
    seed_a: int | None = None,
    seed_b: int | None = None,
    factor: float = 0.5,
    contrast: float = 0.08,
    mode: str = "height",
) -> dict[str, Any]:
    if id_a not in CATALOG_BY_ID:
        raise KeyError(f"Unknown material_a ID '{id_a}'")
    if id_b not in CATALOG_BY_ID:
        raise KeyError(f"Unknown material_b ID '{id_b}'")
    if size not in ALLOWED_SIZES:
        raise ValueError(f"Invalid size {size}. Allowed: {ALLOWED_SIZES}")
    if mode not in ("height", "cavity", "linear"):
        raise ValueError(f"Invalid blend mode '{mode}'. Allowed: height, cavity, linear")

    eff_seed_a = int(CATALOG_BY_ID[id_a]["default_seed"] if seed_a is None else seed_a)
    eff_seed_b = int(CATALOG_BY_ID[id_b]["default_seed"] if seed_b is None else seed_b)
    f = float(max(0.0, min(1.0, factor)))
    c = float(max(0.01, min(0.4, contrast)))

    cache_key = _blend_cache_key(id_a, id_b, size, eff_seed_a, eff_seed_b, f, c, mode)
    out_dir = CACHE_DIR / cache_key
    meta_path = out_dir / "meta.json"

    lock = _get_key_lock(cache_key)
    with lock:
        if meta_path.is_file() and (out_dir / "albedo_delit.png").is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["cached"] = True
            meta["maps"] = _collect_map_urls(cache_key, out_dir)
            return meta

        t0 = time.perf_counter()
        res_a = synthesize_cached(id_a, size=size, seed=eff_seed_a)
        res_b = synthesize_cached(id_b, size=size, seed=eff_seed_b)

        out_dir.mkdir(parents=True, exist_ok=True)
        hypbr.blend_materials(
            Path(res_a["cache_dir"]),
            Path(res_b["cache_dir"]),
            out_dir,
            factor=f,
            contrast=c,
            mode=mode,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        qa = _summarize_verify(verify_material(out_dir))

        result = {
            "cache_key": cache_key,
            "id": f"{id_a} + {id_b}",
            "name": f"{CATALOG_BY_ID[id_a]['name']} × {CATALOG_BY_ID[id_b]['name']}",
            "kind": "blend",
            "group": "Blend",
            "description": f"{mode.capitalize()} blend ({int(round(f * 100))}% B, contrast {c:.2f})",
            "size": size,
            "seed": eff_seed_a,
            "blend": {
                "id_a": id_a,
                "id_b": id_b,
                "factor": round(f, 3),
                "contrast": round(c, 3),
                "mode": mode,
            },
            "weather": {k: 0.0 for k in WEATHER_CHANNELS},
            "is_weathered": False,
            "elapsed_ms": elapsed_ms,
            "cached": False,
            "cache_dir": str(out_dir),
            "maps": _collect_map_urls(cache_key, out_dir),
            "verify": qa,
            "download_url": f"/api/download/{cache_key}.zip",
        }
        _write_json_atomic(meta_path, result)
        return result


def get_catalog_payload() -> dict[str, Any]:
    fresh_weather = {k: 0.0 for k in WEATHER_CHANNELS}
    materials = []
    cached_count = 0

    for entry in CATALOG_LIST:
        key_256 = _synth_cache_key(entry["id"], 256, entry["default_seed"], fresh_weather)
        key_512 = _synth_cache_key(entry["id"], 512, entry["default_seed"], fresh_weather)
        has_256 = (CACHE_DIR / key_256 / "meta.json").is_file()
        has_512 = (CACHE_DIR / key_512 / "meta.json").is_file()
        preview_url = None
        if has_512 and (CACHE_DIR / key_512 / "preview.png").is_file():
            preview_url = f"/cache/{key_512}/preview.png"
        elif has_256 and (CACHE_DIR / key_256 / "preview.png").is_file():
            preview_url = f"/cache/{key_256}/preview.png"
        elif (PLAYGROUND_DIR / "assets" / "mats" / entry["id"] / "preview.webp").is_file():
            preview_url = f"/assets/mats/{entry['id']}/preview.webp"

        if has_256 or has_512:
            cached_count += 1

        materials.append({
            **entry,
            "cached": has_256 or has_512,
            "cached_sizes": [s for s, ok in ((256, has_256), (512, has_512)) if ok],
            "preview_url": preview_url,
        })

    presets = {"fresh": {k: 0.0 for k in WEATHER_CHANNELS}}
    for name, vals in WEATHER_PRESETS.items():
        presets[name] = {k: float(vals.get(k, 0.0)) for k in WEATHER_CHANNELS}

    return {
        "version": getattr(hypbr, "__version__", "0.5.0"),
        "device": dp.DEVICE.type,
        "count": len(materials),
        "cached_count": cached_count,
        "groups": GROUP_ORDER,
        "group_counts": {g: sum(1 for m in materials if m["group"] == g) for g in GROUP_ORDER[1:]},
        "materials": materials,
        "weather_presets": presets,
        "weather_archetypes": list(WEATHER_ARCHETYPES),
        "allowed_sizes": list(ALLOWED_SIZES),
        "default_size": DEFAULT_SIZE,
    }


def _parse_optional_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def _parse_optional_int(val: Any) -> int | None:
    if val is None or val == "":
        return None
    return int(val)


def _build_zip_bytes(cache_key: str) -> tuple[bytes, str]:
    safe_key = Path(cache_key).name
    target_dir = (CACHE_DIR / safe_key).resolve()
    if CACHE_DIR.resolve() not in target_dir.parents or not target_dir.is_dir():
        raise FileNotFoundError(f"Cache entry '{cache_key}' not found")

    meta_path = target_dir / "meta.json"
    zip_stem = safe_key
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mat_slug = str(meta.get("id", safe_key)).replace(" ", "_").replace("+", "x")
        zip_stem = f"hypbr_{mat_slug}_{meta.get('size', 256)}px_{safe_key[:8]}"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for fp in sorted(target_dir.iterdir()):
            if fp.is_file() and fp.suffix.lower() in (".png", ".json"):
                zf.write(fp, arcname=f"{zip_stem}/{fp.name}")
    return buf.getvalue(), f"{zip_stem}.zip"


class PlaygroundHandler(BaseHTTPRequestHandler):
    server_version = "hypbr-playground/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Keep console output crisp and readable
        msg = format % args
        if not msg.startswith('"GET /cache/'):
            sys.stderr.write(f"[playground] {self.address_string()} {msg}\n")

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: int = HTTPStatus.OK,
        cache_control: str = "no-cache",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("Access-Control-Allow-Origin", "*")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status=status)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > 256_000:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _serve_file(self, fp: Path, cache_control: str) -> None:
        if not fp.is_file():
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        mime, _ = mimetypes.guess_type(str(fp))
        if fp.suffix == ".js":
            mime = "application/javascript; charset=utf-8"
        elif fp.suffix == ".css":
            mime = "text/css; charset=utf-8"
        elif fp.suffix == ".html":
            mime = "text/html; charset=utf-8"
        elif fp.suffix == ".png":
            mime = "image/png"
        elif fp.suffix == ".webp":
            mime = "image/webp"
        elif fp.suffix == ".json":
            mime = "application/json; charset=utf-8"
        elif fp.suffix == ".txt":
            mime = "text/plain; charset=utf-8"
        elif fp.suffix == ".md":
            mime = "text/markdown; charset=utf-8"
        else:
            mime = mime or "application/octet-stream"
        self._send_bytes(fp.read_bytes(), mime, cache_control=cache_control)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path

        try:
            if path in ("/", "/index.html"):
                self._serve_file(PLAYGROUND_DIR / "index.html", "no-cache")
                return

            if path in ("/styles.css", "/app.js", "/static_engine.js"):
                self._serve_file(PLAYGROUND_DIR / path.lstrip("/"), "no-cache")
                return

            # Agent atlas + baked static assets (CORS-enabled like the Vercel deployment)
            if path in ("/llms.txt", "/agent.md", "/atlas.json") or path.startswith(("/atlas/", "/assets/")):
                rel = path.lstrip("/")
                target = (PLAYGROUND_DIR / rel).resolve()
                if PLAYGROUND_DIR.resolve() not in target.parents:
                    self._send_json({"error": "Forbidden path"}, status=HTTPStatus.FORBIDDEN)
                    return
                if path == "/atlas.json" or path.startswith("/atlas/") or path in ("/llms.txt", "/agent.md"):
                    if not target.is_file():
                        self._send_json({"error": "Atlas files not exported yet: run `hypbr atlas export playground/` or bake_static.py"},
                                        status=HTTPStatus.NOT_FOUND)
                        return
                self._serve_file(target, "public, max-age=300")
                return

            if path == "/api/health":
                self._send_json({"ok": True, "device": dp.DEVICE.type, "materials": len(CATALOG_LIST)})
                return

            if path == "/api/catalog":
                self._send_json(get_catalog_payload())
                return

            if path == "/api/material":
                qs = urllib.parse.parse_qs(parsed.query)
                def _q(name: str) -> str | None:
                    vals = qs.get(name)
                    return vals[0] if vals else None

                mat_id = _q("id")
                if not mat_id:
                    self._send_json({"error": "Missing required parameter 'id'"}, status=HTTPStatus.BAD_REQUEST)
                    return
                size = int(_q("size") or DEFAULT_SIZE)
                res = synthesize_cached(
                    mat_id,
                    size=size,
                    seed=_parse_optional_int(_q("seed")),
                    weather_preset=_q("weather_preset"),
                    wear=_parse_optional_float(_q("wear")),
                    dirt=_parse_optional_float(_q("dirt")),
                    wet=_parse_optional_float(_q("wet")),
                    moss=_parse_optional_float(_q("moss")),
                    rust=_parse_optional_float(_q("rust")),
                    dust=_parse_optional_float(_q("dust")),
                )
                self._send_json(res)
                return

            if path.startswith("/api/download/") and path.endswith(".zip"):
                key = path[len("/api/download/") : -len(".zip")]
                zip_bytes, filename = _build_zip_bytes(key)
                self._send_bytes(
                    zip_bytes,
                    "application/zip",
                    cache_control="private, max-age=3600",
                    extra_headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                )
                return

            if path.startswith("/cache/"):
                rel = path[len("/cache/") :]
                target = (CACHE_DIR / rel).resolve()
                if CACHE_DIR.resolve() not in target.parents:
                    self._send_json({"error": "Forbidden path"}, status=HTTPStatus.FORBIDDEN)
                    return
                self._serve_file(target, "public, max-age=31536000, immutable")
                return

            self._send_json({"error": f"Unknown route: {path}"}, status=HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"Internal error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path

        try:
            if path == "/api/synthesize":
                data = self._read_json_body()
                mat_id = data.get("id")
                if not mat_id:
                    self._send_json({"error": "Missing required field 'id'"}, status=HTTPStatus.BAD_REQUEST)
                    return
                size = int(data.get("size", DEFAULT_SIZE))
                res = synthesize_cached(
                    str(mat_id),
                    size=size,
                    seed=_parse_optional_int(data.get("seed")),
                    weather_preset=data.get("weather_preset"),
                    wear=_parse_optional_float(data.get("wear")),
                    dirt=_parse_optional_float(data.get("dirt")),
                    wet=_parse_optional_float(data.get("wet")),
                    moss=_parse_optional_float(data.get("moss")),
                    rust=_parse_optional_float(data.get("rust")),
                    dust=_parse_optional_float(data.get("dust")),
                )
                self._send_json(res)
                return

            if path == "/api/blend":
                data = self._read_json_body()
                id_a = data.get("id_a")
                id_b = data.get("id_b")
                if not id_a or not id_b:
                    self._send_json(
                        {"error": "Missing required fields 'id_a' and 'id_b'"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                res = blend_cached(
                    str(id_a),
                    str(id_b),
                    size=int(data.get("size", DEFAULT_SIZE)),
                    seed_a=_parse_optional_int(data.get("seed_a")),
                    seed_b=_parse_optional_int(data.get("seed_b")),
                    factor=float(data.get("factor", 0.5)),
                    contrast=float(data.get("contrast", 0.08)),
                    mode=str(data.get("mode", "height")),
                )
                self._send_json(res)
                return

            self._send_json({"error": f"Unknown POST route: {path}"}, status=HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"Internal error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def _prewarm_worker(count: int, size: int) -> None:
    targets = CATALOG_LIST[: max(0, min(count, len(CATALOG_LIST)))]
    t0 = time.perf_counter()
    for i, item in enumerate(targets, 1):
        try:
            res = synthesize_cached(item["id"], size=size)
            tag = "cached" if res["cached"] else f"{res['elapsed_ms']}ms"
            sys.stderr.write(f"[prewarm {i:02d}/{len(targets):02d}] {item['id']} ({size}px) -> {tag}\n")
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[prewarm] failed on {item['id']}: {exc}\n")
    sys.stderr.write(f"[prewarm] finished {len(targets)} materials in {time.perf_counter() - t0:.1f}s\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="hypbr Interactive Three.js PBR Material Playground (CPU-only)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8766, help="Port number (default: 8766)")
    parser.add_argument(
        "--prewarm",
        type=int,
        default=0,
        help="Optional number of catalog materials to prewarm in background",
    )
    parser.add_argument(
        "--prewarm-size",
        type=int,
        default=DEFAULT_SIZE,
        choices=ALLOWED_SIZES,
        help="Resolution for --prewarm (default: 512)",
    )
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), PlaygroundHandler)
    httpd.daemon_threads = True

    url = f"http://{args.host}:{args.port}"
    print(f"hypbr PBR Material Playground running at {url}")
    print(f"  Device:   {dp.DEVICE} (strict CPU-only)")
    print(f"  Catalog:  {len(CATALOG_LIST)} procedural materials ({len(WEATHER_PRESETS)} weather presets)")
    print(f"  Cache:    {CACHE_DIR}")
    print("  Press Ctrl+C to stop.")

    if args.prewarm > 0:
        t = threading.Thread(target=_prewarm_worker, args=(args.prewarm, args.prewarm_size), daemon=True)
        t.start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down playground server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
