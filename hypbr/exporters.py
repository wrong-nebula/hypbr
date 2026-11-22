"""Engine & standard PBR I/O exporters for hypbr.

Supports zero-dependency export from any synthesized 15-map hypbr directory to:
  - unity       (Unity URP: MetallicSmoothness RGBA, OpenGL +Y normal, AO, Height)
  - unity-hdrp  (Unity HDRP: MaskMap RGBA [M, AO, Detail, Smoothness], OpenGL +Y normal)
  - godot       (Godot 4.x StandardMaterial3D .tres + ORM + OpenGL +Y normal + POM height)
  - unreal      (Unreal Engine 5: DirectX -Y normal/bent-normal, ORM, 16-bit height, UE5 manifest)
  - blender     (Blender 4.x Principled BSDF import_blender.py + Non-Color/sRGB textures)
  - threejs     (Three.js ES module MeshPhysicalMaterial factory + textures)
  - gltf        (glTF 2.0 material.gltf with pbrMetallicRoughness + ORM + normalTexture)
  - vulkan      (Vulkan / WebGPU manifest with VK_FORMAT specs, ORM, RG normal, 16-bit height)
  - materialx   (MaterialX 1.38 / OpenUSD standard_surface .mtlx document)
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .maps import MapSource, MaterialMaps  # noqa: F401  (MaterialMaps: annotations)

SUPPORTED_TARGETS: tuple[str, ...] = (
    "unity",
    "unity-hdrp",
    "godot",
    "unreal",
    "blender",
    "threejs",
    "gltf",
    "glb",
    "vulkan",
    "materialx",
    "usd",
)


def _load_gray(src: MapSource, fname: str, default: int = 128, size: tuple[int, int] | None = None) -> np.ndarray:
    if src.exists(fname):
        im = src.open(fname).convert("L")
        if size and im.size != size:
            im = im.resize(size, Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)
    w, h = size or (512, 512)
    return np.full((h, w), default, dtype=np.uint8)


def _load_rgb(src: MapSource, fname: str, default: tuple[int, int, int] = (128, 128, 255),
              size: tuple[int, int] | None = None) -> np.ndarray:
    if src.exists(fname):
        im = src.open(fname).convert("RGB")
        if size and im.size != size:
            im = im.resize(size, Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)
    w, h = size or (512, 512)
    arr = np.empty((h, w, 3), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = default
    return arr


def _copy_if_exists(src: MapSource, fname: str, dst: Path) -> bool:
    return src.copy_to(fname, dst)


def _flip_normal_green(rgb: np.ndarray) -> np.ndarray:
    """Convert OpenGL (+Y) tangent-space normal to DirectX (-Y)."""
    out = rgb.copy()
    out[..., 1] = 255 - out[..., 1]
    return out


def _detect_size(src_dir: MapSource) -> tuple[int, int]:
    for name in ("albedo_delit.png", "normal.png", "height.png"):
        if src_dir.exists(name):
            with src_dir.open(name) as im:
                return im.size
    raise FileNotFoundError(f"No PBR maps found in {src_dir}")


def _export_unity(src_dir: MapSource, out_dir: Path, name: str, hdrp: bool = False) -> Path:
    size = _detect_size(src_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(src_dir, "albedo_delit.png", out_dir / "BaseMap.png")
    _copy_if_exists(src_dir, "normal.png", out_dir / "NormalMap.png")
    _copy_if_exists(src_dir, "height.png", out_dir / "HeightMap.png")
    _copy_if_exists(src_dir, "height16.png", out_dir / "HeightMap16.png")

    metallic = _load_gray(src_dir, "metallic.png", default=0, size=size)
    roughness = _load_gray(src_dir, "roughness.png", default=128, size=size)
    smoothness = (255 - roughness).astype(np.uint8)
    ao = _load_gray(src_dir, "ao.png", default=255, size=size)

    if hdrp:
        detail = _load_gray(src_dir, "curvature.png", default=128, size=size)
        mask_map = np.stack([metallic, ao, detail, smoothness], axis=-1)
        Image.fromarray(mask_map, mode="RGBA").save(out_dir / "MaskMap.png")
        manifest: dict[str, Any] = {
            "engine": "unity-hdrp",
            "material": name,
            "normal_convention": "OpenGL (+Y)",
            "textures": {
                "BaseColorMap": {"file": "BaseMap.png", "sRGB": True},
                "NormalMap": {"file": "NormalMap.png", "sRGB": False, "textureType": "NormalMap"},
                "MaskMap": {
                    "file": "MaskMap.png",
                    "sRGB": False,
                    "channels": {"R": "Metallic", "G": "AO", "B": "DetailMask", "A": "Smoothness (1-Roughness)"},
                },
                "HeightMap": {"file": "HeightMap.png", "sRGB": False},
            },
        }
    else:
        zeros = np.zeros_like(metallic)
        met_smooth = np.stack([metallic, zeros, zeros, smoothness], axis=-1)
        Image.fromarray(met_smooth, mode="RGBA").save(out_dir / "MetallicSmoothness.png")
        Image.fromarray(ao, mode="L").save(out_dir / "OcclusionMap.png")
        manifest = {
            "engine": "unity-urp",
            "material": name,
            "normal_convention": "OpenGL (+Y)",
            "textures": {
                "BaseMap": {"file": "BaseMap.png", "sRGB": True},
                "NormalMap": {"file": "NormalMap.png", "sRGB": False, "textureType": "NormalMap"},
                "MetallicGlossMap": {
                    "file": "MetallicSmoothness.png",
                    "sRGB": False,
                    "channels": {"R": "Metallic", "A": "Smoothness (1-Roughness)"},
                },
                "OcclusionMap": {"file": "OcclusionMap.png", "sRGB": False},
                "ParallaxMap": {"file": "HeightMap.png", "sRGB": False},
            },
        }

    (out_dir / "unity_material.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def _export_godot(src_dir: MapSource, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(src_dir, "albedo_delit.png", out_dir / "albedo.png")
    _copy_if_exists(src_dir, "orm.png", out_dir / "orm.png")
    _copy_if_exists(src_dir, "normal.png", out_dir / "normal.png")
    _copy_if_exists(src_dir, "height.png", out_dir / "height.png")
    _copy_if_exists(src_dir, "anisotropy.png", out_dir / "anisotropy.png")

    tres = f"""[gd_resource type="StandardMaterial3D" load_steps=5 format=3]

[ext_resource type="Texture2D" path="albedo.png" id="1_albedo"]
[ext_resource type="Texture2D" path="orm.png" id="2_orm"]
[ext_resource type="Texture2D" path="normal.png" id="3_normal"]
[ext_resource type="Texture2D" path="height.png" id="4_height"]

[resource]
resource_name = "{name}"
albedo_texture = ExtResource("1_albedo")
metallic = 1.0
metallic_texture = ExtResource("2_orm")
metallic_texture_channel = 2
roughness_texture = ExtResource("2_orm")
roughness_texture_channel = 1
normal_enabled = true
normal_texture = ExtResource("3_normal")
ao_enabled = true
ao_texture = ExtResource("2_orm")
ao_texture_channel = 0
heightmap_enabled = true
heightmap_scale = 2.0
heightmap_deep_parallax = true
heightmap_min_layers = 8
heightmap_max_layers = 32
heightmap_texture = ExtResource("4_height")
"""
    (out_dir / "material.tres").write_text(tres)
    return out_dir


def _export_unreal(src_dir: MapSource, out_dir: Path, name: str) -> Path:
    size = _detect_size(src_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(src_dir, "albedo_delit.png", out_dir / "T_BaseColor.png")
    _copy_if_exists(src_dir, "orm.png", out_dir / "T_ORM.png")
    _copy_if_exists(src_dir, "height16.png", out_dir / "T_Height16.png")
    _copy_if_exists(src_dir, "specular.png", out_dir / "T_Specular.png")

    normal_gl = _load_rgb(src_dir, "normal.png", default=(128, 128, 255), size=size)
    Image.fromarray(_flip_normal_green(normal_gl), mode="RGB").save(out_dir / "T_Normal_DX.png")

    if src_dir.exists("bent_normal.png"):
        bent_gl = _load_rgb(src_dir, "bent_normal.png", default=(128, 128, 255), size=size)
        Image.fromarray(_flip_normal_green(bent_gl), mode="RGB").save(out_dir / "T_BentNormal_DX.png")

    manifest = {
        "engine": "unreal-engine-5",
        "material": name,
        "normal_convention": "DirectX (-Y)",
        "textures": {
            "BaseColor": {"file": "T_BaseColor.png", "sRGB": True, "compression": "TC_Default"},
            "ORM": {
                "file": "T_ORM.png",
                "sRGB": False,
                "compression": "TC_Masks",
                "channels": {"R": "AmbientOcclusion", "G": "Roughness", "B": "Metallic"},
            },
            "Normal": {"file": "T_Normal_DX.png", "sRGB": False, "compression": "TC_Normalmap"},
            "BentNormal": {"file": "T_BentNormal_DX.png", "sRGB": False, "compression": "TC_Normalmap"},
            "Height": {"file": "T_Height16.png", "sRGB": False, "compression": "TC_Grayscale"},
        },
    }
    (out_dir / "ue5_material.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def _export_blender(src_dir: MapSource, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in (
        ("albedo_delit.png", "albedo.png"),
        ("normal.png", "normal.png"),
        ("roughness.png", "roughness.png"),
        ("metallic.png", "metallic.png"),
        ("ao.png", "ao.png"),
        ("height16.png", "height16.png"),
        ("specular.png", "specular.png"),
    ):
        _copy_if_exists(src_dir, src_name, out_dir / dst_name)

    script = f'''"""Auto-generated Blender 4.x Principled BSDF importer for `{name}`."""
from pathlib import Path
import bpy


def create_material(mat_name: str = "{name}") -> bpy.types.Material:
    tex_dir = Path(__file__).resolve().parent
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (260, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def load_tex(filename: str, colorspace: str, loc: tuple[int, int]):
        p = tex_dir / filename
        if not p.exists():
            return None
        node = nt.nodes.new("ShaderNodeTexImage")
        node.location = loc
        node.image = bpy.data.images.load(str(p), check_existing=True)
        node.image.colorspace_settings.name = colorspace
        return node

    tex_alb = load_tex("albedo.png", "sRGB", (-420, 320))
    tex_ao = load_tex("ao.png", "Non-Color", (-420, 80))
    if tex_alb and tex_ao:
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 0.65
        mix.location = (-80, 240)
        nt.links.new(tex_alb.outputs["Color"], mix.inputs["A"])
        nt.links.new(tex_ao.outputs["Color"], mix.inputs["B"])
        nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    elif tex_alb:
        nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])

    tex_met = load_tex("metallic.png", "Non-Color", (-420, -160))
    if tex_met:
        nt.links.new(tex_met.outputs["Color"], bsdf.inputs["Metallic"])

    tex_rgh = load_tex("roughness.png", "Non-Color", (-420, -400))
    if tex_rgh:
        nt.links.new(tex_rgh.outputs["Color"], bsdf.inputs["Roughness"])

    tex_nrm = load_tex("normal.png", "Non-Color", (-420, -640))
    if tex_nrm:
        nrm = nt.nodes.new("ShaderNodeNormalMap")
        nrm.location = (-80, -640)
        nt.links.new(tex_nrm.outputs["Color"], nrm.inputs["Color"])
        nt.links.new(nrm.outputs["Normal"], bsdf.inputs["Normal"])

    tex_h = load_tex("height16.png", "Non-Color", (-420, -880))
    if tex_h:
        disp = nt.nodes.new("ShaderNodeDisplacement")
        disp.location = (260, -480)
        disp.inputs["Scale"].default_value = 0.04
        nt.links.new(tex_h.outputs["Color"], disp.inputs["Height"])
        nt.links.new(disp.outputs["Displacement"], out.inputs["Displacement"])

    return mat


if __name__ == "__main__":
    create_material()
'''
    (out_dir / "import_blender.py").write_text(script)
    return out_dir


def _export_threejs(src_dir: MapSource, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in (
        ("albedo_delit.png", "color.png"),
        ("normal.png", "normal.png"),
        ("roughness.png", "roughness.png"),
        ("metallic.png", "metalness.png"),
        ("ao.png", "ao.png"),
        ("height.png", "displacement.png"),
        ("orm.png", "orm.png"),
    ):
        _copy_if_exists(src_dir, src_name, out_dir / dst_name)

    js = f"""// Auto-generated Three.js MeshPhysicalMaterial factory for `{name}`
import * as THREE from "three";

export function createMaterial(
  loader = new THREE.TextureLoader(),
  basePath = "./",
  {{ repeat = 1, displacementScale = 0.02 }} = {{}},
) {{
  const load = (file, isSRGB = false) => {{
    const tex = loader.load(`${{basePath}}${{file}}`);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(repeat, repeat);
    tex.colorSpace = isSRGB ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    tex.anisotropy = 8;
    return tex;
  }};

  return new THREE.MeshPhysicalMaterial({{
    name: "{name}",
    map: load("color.png", true),
    normalMap: load("normal.png", false),
    normalMapType: THREE.TangentSpaceNormalMap,
    roughnessMap: load("roughness.png", false),
    metalnessMap: load("metalness.png", false),
    aoMap: load("ao.png", false),
    displacementMap: load("displacement.png", false),
    displacementScale,
    roughness: 1.0,
    metalness: 1.0,
  }});
}}
"""
    (out_dir / "material.js").write_text(js)
    return out_dir


def _export_gltf(src_dir: MapSource, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(src_dir, "albedo_delit.png", out_dir / "baseColor.png")
    _copy_if_exists(src_dir, "orm.png", out_dir / "orm.png")
    _copy_if_exists(src_dir, "normal.png", out_dir / "normal.png")

    gltf = {
        "asset": {"version": "2.0", "generator": "hypbr"},
        "samplers": [
            {
                "magFilter": 9729,  # LINEAR
                "minFilter": 9987,  # LINEAR_MIPMAP_LINEAR
                "wrapS": 10497,     # REPEAT
                "wrapT": 10497,     # REPEAT
            }
        ],
        "images": [
            {"uri": "baseColor.png", "mimeType": "image/png"},
            {"uri": "orm.png", "mimeType": "image/png"},
            {"uri": "normal.png", "mimeType": "image/png"},
        ],
        "textures": [
            {"sampler": 0, "source": 0},
            {"sampler": 0, "source": 1},
            {"sampler": 0, "source": 2},
        ],
        "materials": [
            {
                "name": name,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                    "metallicFactor": 1.0,
                    "roughnessFactor": 1.0,
                },
                "occlusionTexture": {"index": 1, "strength": 1.0},
                "normalTexture": {"index": 2, "scale": 1.0},
            }
        ],
    }
    (out_dir / "material.gltf").write_text(json.dumps(gltf, indent=2) + "\n")
    return out_dir


def _export_vulkan(src_dir: MapSource, out_dir: Path, name: str) -> Path:
    size = _detect_size(src_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(src_dir, "albedo_delit.png", out_dir / "albedo_srgb.png")
    _copy_if_exists(src_dir, "orm.png", out_dir / "orm_unorm.png")
    _copy_if_exists(src_dir, "normal.png", out_dir / "normal_rgb.png")
    _copy_if_exists(src_dir, "height16.png", out_dir / "height_r16.png")
    _copy_if_exists(src_dir, "bent_normal.png", out_dir / "bent_normal.png")

    # 2-channel RG normal (BC5 / VK_FORMAT_R8G8_UNORM ready; reconstruct Z = sqrt(1 - x^2 - y^2))
    nrm = _load_rgb(src_dir, "normal.png", default=(128, 128, 255), size=size)
    rg = np.zeros_like(nrm)
    rg[..., 0] = nrm[..., 0]
    rg[..., 1] = nrm[..., 1]
    Image.fromarray(rg, mode="RGB").save(out_dir / "normal_rg_bc5_src.png")

    manifest = {
        "api": "vulkan",
        "material": name,
        "resolution": list(size),
        "address_mode": "VK_SAMPLER_ADDRESS_MODE_REPEAT",
        "filter": "VK_FILTER_LINEAR",
        "mipmap_mode": "VK_SAMPLER_MIPMAP_MODE_LINEAR",
        "normal_convention": "OpenGL (+Y)",
        "bindings": [
            {
                "binding": 0,
                "name": "albedo",
                "file": "albedo_srgb.png",
                "vk_format": "VK_FORMAT_R8G8B8A8_SRGB",
                "ktx2_cmd": "ktx create --format R8G8B8A8_SRGB --generate-mipmap --encode uastc albedo_srgb.png albedo.ktx2",
            },
            {
                "binding": 1,
                "name": "orm",
                "file": "orm_unorm.png",
                "vk_format": "VK_FORMAT_R8G8B8A8_UNORM",
                "channels": {"R": "AO", "G": "Roughness", "B": "Metallic"},
                "ktx2_cmd": "ktx create --format R8G8B8A8_UNORM --generate-mipmap --encode uastc orm_unorm.png orm.ktx2",
            },
            {
                "binding": 2,
                "name": "normal",
                "file": "normal_rg_bc5_src.png",
                "vk_format": "VK_FORMAT_R8G8_UNORM",
                "reconstruct_z": "sqrt(max(0.0, 1.0 - dot(xy, xy)))",
                "ktx2_cmd": "ktx create --format R8G8_UNORM --generate-mipmap --encode uastc --normal-mode normal_rg_bc5_src.png normal.ktx2",
            },
            {
                "binding": 3,
                "name": "height",
                "file": "height_r16.png",
                "vk_format": "VK_FORMAT_R16_UNORM",
            },
        ],
    }
    (out_dir / "vulkan_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def _export_materialx(src_dir: MapSource, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("albedo_delit.png", "normal.png", "roughness.png", "metallic.png", "ao.png", "height16.png"):
        _copy_if_exists(src_dir, fname, out_dir / fname)

    mtlx = f"""<?xml version="1.0"?>
<materialx version="1.38">
  <nodegraph name="NG_{name}">
    <image name="img_albedo" type="color3">
      <input name="file" type="filename" value="albedo_delit.png" colorspace="srgb_texture" />
    </image>
    <image name="img_roughness" type="float">
      <input name="file" type="filename" value="roughness.png" colorspace="lin_rec709" />
    </image>
    <image name="img_metallic" type="float">
      <input name="file" type="filename" value="metallic.png" colorspace="lin_rec709" />
    </image>
    <image name="img_normal" type="vector3">
      <input name="file" type="filename" value="normal.png" colorspace="lin_rec709" />
    </image>
    <normalmap name="nrm_map" type="vector3">
      <input name="in" type="vector3" nodename="img_normal" />
    </normalmap>
    <image name="img_height" type="float">
      <input name="file" type="filename" value="height16.png" colorspace="lin_rec709" />
    </image>
    <output name="out_albedo" type="color3" nodename="img_albedo" />
    <output name="out_roughness" type="float" nodename="img_roughness" />
    <output name="out_metallic" type="float" nodename="img_metallic" />
    <output name="out_normal" type="vector3" nodename="nrm_map" />
    <output name="out_height" type="float" nodename="img_height" />
  </nodegraph>

  <standard_surface name="SR_{name}" type="surfaceshader">
    <input name="base_color" type="color3" nodegraph="NG_{name}" output="out_albedo" />
    <input name="specular_roughness" type="float" nodegraph="NG_{name}" output="out_roughness" />
    <input name="metalness" type="float" nodegraph="NG_{name}" output="out_metallic" />
    <input name="normal" type="vector3" nodegraph="NG_{name}" output="out_normal" />
  </standard_surface>

  <displacement name="DISP_{name}" type="displacementshader">
    <input name="displacement" type="float" nodegraph="NG_{name}" output="out_height" />
    <input name="scale" type="float" value="0.02" />
  </displacement>

  <surfacematerial name="{name}" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_{name}" />
    <input name="displacementshader" type="displacementshader" nodename="DISP_{name}" />
  </surfacematerial>
</materialx>
"""
    (out_dir / "material.mtlx").write_text(mtlx)
    return out_dir


def _build_uv_sphere(rings: int = 32, sectors: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate unit UV sphere geometry (positions, normals, tangents, uvs, indices) for GLB preview."""
    verts = []
    norms = []
    tangs = []
    uvs = []
    for r in range(rings + 1):
        v = r / rings
        phi = v * np.pi
        sp, cp = np.sin(phi), np.cos(phi)
        for s in range(sectors + 1):
            u = s / sectors
            theta = u * 2.0 * np.pi
            st, ct = np.sin(theta), np.cos(theta)
            nx, ny, nz = sp * ct, cp, sp * st
            verts.append((nx, ny, nz))
            norms.append((nx, ny, nz))
            tangs.append((-st, 0.0, ct, 1.0))
            uvs.append((u * 2.0, v * 2.0))

    indices = []
    stride = sectors + 1
    for r in range(rings):
        for s in range(sectors):
            a = r * stride + s
            b = a + stride
            indices.extend((a, b, a + 1, a + 1, b, b + 1))

    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(norms, dtype=np.float32),
        np.asarray(tangs, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        np.asarray(indices, dtype=np.uint16),
    )


def _export_glb(src_dir: MapSource, out_dir: Path, name: str) -> Path:
    """Export a single self-contained binary glTF 2.0 (.glb) preview sphere with embedded PBR maps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pos, nrm, tan, uv, idx = _build_uv_sphere()

    bin_chunks: list[bytes] = []
    buffer_views: list[dict[str, Any]] = []
    offset = 0

    def add_view(raw: bytes, target: int | None = None) -> int:
        nonlocal offset
        pad = (-len(raw)) % 4
        view: dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        bin_chunks.append(raw + (b"\x00" * pad))
        offset += len(raw) + pad
        buffer_views.append(view)
        return len(buffer_views) - 1

    bv_pos = add_view(pos.tobytes(), target=34962)
    bv_nrm = add_view(nrm.tobytes(), target=34962)
    bv_tan = add_view(tan.tobytes(), target=34962)
    bv_uv = add_view(uv.tobytes(), target=34962)
    bv_idx = add_view(idx.tobytes(), target=34963)

    img_views: list[int] = []
    for fname in ("albedo_delit.png", "orm.png", "normal.png"):
        if not src_dir.exists(fname):
            raise FileNotFoundError(f"Required map missing for GLB export: {src_dir}/{fname}")
        img_views.append(add_view(src_dir.read_bytes(fname)))

    bin_data = b"".join(bin_chunks)
    gltf_doc: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "hypbr"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [
            {
                "name": f"{name}_sphere",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TANGENT": 2, "TEXCOORD_0": 3},
                        "indices": 4,
                        "material": 0,
                    }
                ],
            }
        ],
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": buffer_views,
        "accessors": [
            {
                "bufferView": bv_pos,
                "componentType": 5126,
                "count": len(pos),
                "type": "VEC3",
                "min": pos.min(axis=0).tolist(),
                "max": pos.max(axis=0).tolist(),
            },
            {"bufferView": bv_nrm, "componentType": 5126, "count": len(nrm), "type": "VEC3"},
            {"bufferView": bv_tan, "componentType": 5126, "count": len(tan), "type": "VEC4"},
            {"bufferView": bv_uv, "componentType": 5126, "count": len(uv), "type": "VEC2"},
            {"bufferView": bv_idx, "componentType": 5123, "count": len(idx), "type": "SCALAR"},
        ],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "images": [
            {"bufferView": img_views[0], "mimeType": "image/png", "name": "baseColor"},
            {"bufferView": img_views[1], "mimeType": "image/png", "name": "orm"},
            {"bufferView": img_views[2], "mimeType": "image/png", "name": "normal"},
        ],
        "textures": [
            {"sampler": 0, "source": 0},
            {"sampler": 0, "source": 1},
            {"sampler": 0, "source": 2},
        ],
        "materials": [
            {
                "name": name,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                    "metallicFactor": 1.0,
                    "roughnessFactor": 1.0,
                },
                "occlusionTexture": {"index": 1, "strength": 1.0},
                "normalTexture": {"index": 2, "scale": 1.0},
            }
        ],
    }

    json_bytes = json.dumps(gltf_doc, separators=(",", ":")).encode("utf-8")
    json_pad = (-len(json_bytes)) % 4
    json_bytes += b" " * json_pad

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    glb_path = out_dir / f"{name}.glb"
    with glb_path.open("wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total_len))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_data), 0x004E4942))
        f.write(bin_data)
    return out_dir


def _export_usd(src_dir: MapSource, out_dir: Path, name: str) -> Path:
    """Export an OpenUSD ASCII (.usda) UsdPreviewSurface material (RealityKit / Omniverse / Houdini)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("albedo_delit.png", "normal.png", "roughness.png", "metallic.png", "ao.png", "height16.png"):
        _copy_if_exists(src_dir, fname, out_dir / fname)

    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    if not safe or safe[0].isdigit():
        safe = f"M_{safe}"

    usda = f"""#usda 1.0
(
    defaultPrim = "{safe}_Root"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "{safe}_Root"
{{
    def Material "{safe}"
    {{
        token outputs:surface.connect = </{safe}_Root/{safe}/PreviewSurface.outputs:surface>

        def Shader "PreviewSurface"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </{safe}_Root/{safe}/TexAlbedo.outputs:rgb>
            float inputs:metallic.connect = </{safe}_Root/{safe}/TexMetallic.outputs:r>
            float inputs:roughness.connect = </{safe}_Root/{safe}/TexRoughness.outputs:r>
            float inputs:occlusion.connect = </{safe}_Root/{safe}/TexAO.outputs:r>
            normal3f inputs:normal.connect = </{safe}_Root/{safe}/TexNormal.outputs:rgb>
            float inputs:ior = 1.5
            token outputs:surface
        }}

        def Shader "PrimvarReader"
        {{
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }}

        def Shader "TexAlbedo"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @./albedo_delit.png@
            token inputs:sourceColorSpace = "sRGB"
            token inputs:wrapS = "repeat"
            token inputs:wrapT = "repeat"
            float2 inputs:st.connect = </{safe}_Root/{safe}/PrimvarReader.outputs:result>
            color3f outputs:rgb
        }}

        def Shader "TexRoughness"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @./roughness.png@
            token inputs:sourceColorSpace = "raw"
            token inputs:wrapS = "repeat"
            token inputs:wrapT = "repeat"
            float2 inputs:st.connect = </{safe}_Root/{safe}/PrimvarReader.outputs:result>
            float outputs:r
        }}

        def Shader "TexMetallic"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @./metallic.png@
            token inputs:sourceColorSpace = "raw"
            token inputs:wrapS = "repeat"
            token inputs:wrapT = "repeat"
            float2 inputs:st.connect = </{safe}_Root/{safe}/PrimvarReader.outputs:result>
            float outputs:r
        }}

        def Shader "TexAO"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @./ao.png@
            token inputs:sourceColorSpace = "raw"
            token inputs:wrapS = "repeat"
            token inputs:wrapT = "repeat"
            float2 inputs:st.connect = </{safe}_Root/{safe}/PrimvarReader.outputs:result>
            float outputs:r
        }}

        def Shader "TexNormal"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @./normal.png@
            token inputs:sourceColorSpace = "raw"
            float4 inputs:scale = (2, 2, 2, 1)
            float4 inputs:bias = (-1, -1, -1, 0)
            token inputs:wrapS = "repeat"
            token inputs:wrapT = "repeat"
            float2 inputs:st.connect = </{safe}_Root/{safe}/PrimvarReader.outputs:result>
            normal3f outputs:rgb
        }}
    }}
}}
"""
    (out_dir / "material.usda").write_text(usda)
    return out_dir


def export_material(
    src_dir: "Path | str | MaterialMaps",
    out_dir: Path | str,
    target: str,
    name: str | None = None,
) -> Path:
    """Export a synthesized hypbr material directory to a target engine/format.

    Args:
        src_dir: Directory containing hypbr PBR maps (`albedo_delit.png`, `normal.png`, etc.),
            or an in-memory `MaterialMaps` (maps are encoded straight into the target layout).
        out_dir: Destination directory for the engine-formatted output.
        target: One of `SUPPORTED_TARGETS` (e.g. "unity", "unity-hdrp", "godot",
            "unreal", "blender", "threejs", "gltf", "glb", "vulkan", "materialx", "usd").
        name: Optional material name (defaults to `src_dir.name` / `MaterialMaps.material_id`).

    Returns:
        Path to `out_dir`.
    """
    src = MapSource(src_dir)
    dst = Path(out_dir)
    mat_name = name or src.name
    t = target.lower().strip()

    if t == "unity":
        return _export_unity(src, dst, mat_name, hdrp=False)
    if t == "unity-hdrp":
        return _export_unity(src, dst, mat_name, hdrp=True)
    if t == "godot":
        return _export_godot(src, dst, mat_name)
    if t == "unreal":
        return _export_unreal(src, dst, mat_name)
    if t == "blender":
        return _export_blender(src, dst, mat_name)
    if t == "threejs":
        return _export_threejs(src, dst, mat_name)
    if t == "gltf":
        return _export_gltf(src, dst, mat_name)
    if t == "glb":
        return _export_glb(src, dst, mat_name)
    if t == "vulkan":
        return _export_vulkan(src, dst, mat_name)
    if t == "materialx":
        return _export_materialx(src, dst, mat_name)
    if t == "usd":
        return _export_usd(src, dst, mat_name)

    raise ValueError(f"Unsupported export target {target!r}. Choose one of: {', '.join(SUPPORTED_TARGETS)}")
