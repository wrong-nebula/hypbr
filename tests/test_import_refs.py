"""tools/import_refs.py: local folders/zips and MatSynth rows -> 512px reference .npz + manifest."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hypbr import synthesize_material

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("import_refs", ROOT / "tools" / "import_refs.py")
imp = importlib.util.module_from_spec(_spec)
sys.modules["import_refs"] = imp  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(imp)


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def source_maps(tmp_path_factory):
    d = synthesize_material("classic_red_brick", tmp_path_factory.mktemp("src") / "brick", size=128)
    return {
        "albedo": np.asarray(Image.open(d / "albedo_delit.png").convert("RGB")),
        "normal": np.asarray(Image.open(d / "normal.png").convert("RGB")),
        "height16": np.asarray(Image.open(d / "height16.png")),
        "rough": np.asarray(Image.open(d / "roughness.png").convert("L")),
    }


def _megascans_folder(root: Path, m: dict) -> Path:
    """One asset folder with a DirectX normal (Megascans-style names) and one zip with GL normal + gloss."""
    a = root / "Rusty_Metal_Sheet_vbqhdeal"
    a.mkdir(parents=True)
    dx = m["normal"].copy()
    dx[..., 1] = 255 - dx[..., 1]
    Image.fromarray(m["albedo"]).save(a / "vbqhdeal_2K_Albedo.jpg", quality=95)
    Image.fromarray(dx).save(a / "vbqhdeal_2K_Normal.jpg", quality=95)
    Image.fromarray(m["height16"]).save(a / "vbqhdeal_2K_Displacement.png")
    Image.fromarray(m["rough"]).save(a / "vbqhdeal_2K_Roughness.jpg", quality=95)
    Image.fromarray(m["albedo"]).save(a / "vbqhdeal_Preview.png")
    with zipfile.ZipFile(root / "Mossy_Stone_Wall_tcmoabc.zip", "w") as zf:
        zf.writestr("tcmoabc_4K_BaseColor.png", _png(m["albedo"]))
        zf.writestr("tcmoabc_4K_NormalGL.png", _png(m["normal"]))
        zf.writestr("tcmoabc_4K_Height.png", _png(m["height16"]))
        zf.writestr("tcmoabc_4K_Gloss.png", _png(255 - m["rough"]))
    return root


def test_folder_import_detects_dx_normals_reads_zips_and_records_provenance(tmp_path, source_maps):
    src = _megascans_folder(tmp_path / "downloads", source_maps)
    out = tmp_path / "refs"
    assert imp.main(["folder", str(src), "--out", str(out)]) == 0

    a = np.load(out / "megascans" / "mega_rusty_metal_sheet_vbqhdeal.npz")
    b = np.load(out / "megascans" / "mega_mossy_stone_wall_tcmoabc.npz")
    for z in (a, b):
        assert z["albedo"].shape == (512, 512, 3) and z["albedo"].dtype == np.uint8
        assert z["normal"].shape == (512, 512, 3) and z["height"].dtype == np.float16
        assert z["roughness"].shape == (512, 512)

    ref_gl = np.asarray(Image.fromarray(source_maps["normal"]).resize((512, 512), Image.LANCZOS)).astype(float)
    meta_a = json.loads(str(a["meta"]))
    assert meta_a["normal_convention"] == "dx (detected)"  # flipped back to OpenGL...
    assert np.abs(a["normal"][..., 1].astype(float) - ref_gl[..., 1]).mean() < 6.0  # ...matching the GL original
    assert json.loads(str(b["meta"]))["normal_convention"].startswith("gl")
    # gloss was inverted into roughness
    assert abs(float(b["roughness"].mean()) - float(a["roughness"].mean())) < 6.0

    lines = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    assert all(r["redistributable"] is False and "AI training permitted" in r["license"] for r in lines)
    assert {Path(r["origin"]).name for r in lines} == {"Rusty_Metal_Sheet_vbqhdeal", "Mossy_Stone_Wall_tcmoabc.zip"}

    # re-running skips what is already imported
    assert imp.run(imp.build_parser().parse_args(["folder", str(src), "--out", str(out)]), log=lambda _m: None) == []


def test_refuses_to_write_inside_a_git_repo(tmp_path, source_maps):
    src = _megascans_folder(tmp_path / "downloads", source_maps)
    with pytest.raises(SystemExit, match="git working tree"):
        imp.main(["folder", str(src), "--out", str(ROOT / "scratch_refs")])
    assert not (ROOT / "scratch_refs").exists()


def test_other_sources_need_an_explicit_license(tmp_path, source_maps):
    src = _megascans_folder(tmp_path / "downloads", source_maps)
    with pytest.raises(SystemExit, match="--license"):
        imp.main(["folder", str(src), "--out", str(tmp_path / "refs"), "--source", "mystuff"])


def test_matsynth_rows_filters_and_mapping(tmp_path, source_maps):
    keep = imp.matsynth_keep
    skip = ("polyhaven", "textures_heaven", "deschaintre")
    scan = {"method": "photogrammetry"}
    assert keep({"license": "CC0", "source": "cgbookcase", **scan}, False, skip)[0]
    assert not keep({"license": "CC-BY", "source": "julio", **scan}, False, skip)[0]
    assert keep({"license": "CC-BY", "source": "julio", **scan}, True, skip)[0]
    assert not keep({"license": "CC BY-NC-ND 2.0", "source": "x", **scan}, True, skip)[0]  # never NC/ND
    assert not keep({"license": "CC0", "source": "deschaintre_2020", **scan}, False, skip)[0]
    assert not keep({"license": "CC0", "source": "PolyHaven", **scan}, False, skip)[0]
    assert not keep({"license": "CC0", "source": "textures_heaven", **scan}, False, skip)[0]  # Poly Haven's old name
    assert not keep({"license": "CC0", "source": "matsynth", "method": "blend"}, False, skip)[0]
    assert not keep({"license": "CC0", "source": "cgbookcase", "method": "procedural"}, False, skip)[0]
    assert keep({"license": "CC0", "source": "cgbookcase", "method": "procedural"}, False, skip, ("all",))[0]
    assert keep({"license": "CC0", "source": "ambientcg", "method": "photogrammetry"}, False, skip)[0]  # 129 new scans
    assert keep({"license": "CC0", "source": "sharetextures", "method": "scan"}, False, skip)[0]
    assert not keep({"license": "CC0", "source": "ambientcg", "method": "approximation"}, False, skip)[0]

    row = {
        "name": "Bricks075", "category": 11,  # terracotta
        "metadata": {"license": "CC0", "source": "cgbookcase", "physical_size": 150, "tags": ["brick", "red"],
                     "method": "photogrammetry", "link": "https://example.invalid/bricks075"},
        "basecolor": {"bytes": _png(source_maps["albedo"])}, "normal": {"bytes": _png(source_maps["normal"])},
        "height": {"bytes": _png(source_maps["height16"])}, "roughness": {"bytes": _png(source_maps["rough"])},
        "metallic": None,
    }
    mat = imp.matsynth_material(row)
    assert mat.id == "msyn_terracotta_bricks075" and mat.license == "CC0"
    rec = imp.write_material(mat, tmp_path / "refs")
    assert rec["size_m"] == 1.5 and rec["category"] == "terracotta" and "metallic" not in rec["maps"]  # 150 cm
    for bad in (0, 0.5, 1e6, "3m", None):  # implausible or missing edge lengths are dropped, not guessed
        row["metadata"]["physical_size"] = bad
        assert "size_m" not in imp.matsynth_material(row).meta


def test_calibration_uses_imported_roughness_and_physical_size(tmp_path, source_maps):
    from hypbr.calibration import reference_stats

    row = {"name": "Bricks075", "category": 11,
           "metadata": {"license": "CC0", "source": "cgbookcase", "physical_size": 150},
           "basecolor": {"bytes": _png(source_maps["albedo"])}, "normal": {"bytes": _png(source_maps["normal"])},
           "roughness": {"bytes": _png(source_maps["rough"])}}
    imp.write_material(imp.matsynth_material(row), tmp_path / "refs")
    st = reference_stats([tmp_path / "refs" / "matsynth"], [])["msyn_terracotta_bricks075"]
    assert st["size_m"] == 1.5                      # from metadata, not the name-based 2.0 default
    assert abs(st["rough"] - source_maps["rough"].mean() / 255.0) < 0.02
