"""hypbr — Hybrid Neural + Exact Periodic Procedural PBR Material Synthesizer."""
from __future__ import annotations

from . import atlas
from ._backend import has_torch
from .atlas import export_atlas_files, get_atlas, get_material_spec, recommend_for_object, search_atlas
from .blend import blend_materials, blend_materials_maps, compute_height_blend_mask
from .exporters import SUPPORTED_TARGETS, export_material
from .maps import MaterialMaps
from .normal_graph import PRESETS, run_graph
from .preview import render_preview_html, serve_preview
from .synthesize import (
    list_materials,
    synthesize_all,
    synthesize_hardops_plate,
    synthesize_hardops_plate_maps,
    synthesize_material,
    synthesize_material_maps,
)
from .verify import verify_material
from .weathering import (
    WEATHER_ARCHETYPES,
    WEATHER_PRESETS,
    apply_weathering_maps,
    compute_weathering_masks,
    infer_material_kind,
    predict_neural_weathering_masks,
    resolve_weather_params,
    weather_material,
    weather_material_maps,
)

# `weightcode` operates on torch modules; resolve its names lazily so `import hypbr` (and the
# whole procedural pipeline) works without PyTorch installed.
_LAZY_TORCH_EXPORTS = {
    "apply_depthpro_weightcode": "weightcode",
    "spectral_filter_3x3_kernel": "weightcode",
    "tileable_4quad_blend": "weightcode",
}


def __getattr__(name: str):
    if name in _LAZY_TORCH_EXPORTS:
        import importlib

        from ._backend import require_torch

        require_torch(f"hypbr.{name}")
        return getattr(importlib.import_module(f".{_LAZY_TORCH_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module 'hypbr' has no attribute {name!r}")

__version__ = "1.1.0"

__all__ = [
    "__version__",
    "MaterialMaps",
    "atlas",
    "export_atlas_files",
    "get_atlas",
    "get_material_spec",
    "has_torch",
    "recommend_for_object",
    "search_atlas",
    "PRESETS",
    "SUPPORTED_TARGETS",
    "WEATHER_ARCHETYPES",
    "WEATHER_PRESETS",
    "apply_depthpro_weightcode",
    "apply_weathering_maps",
    "blend_materials",
    "blend_materials_maps",
    "compute_height_blend_mask",
    "compute_weathering_masks",
    "export_material",
    "infer_material_kind",
    "list_materials",
    "predict_neural_weathering_masks",
    "render_preview_html",
    "resolve_weather_params",
    "run_graph",
    "serve_preview",
    "spectral_filter_3x3_kernel",
    "synthesize_all",
    "synthesize_hardops_plate",
    "synthesize_hardops_plate_maps",
    "synthesize_material",
    "synthesize_material_maps",
    "tileable_4quad_blend",
    "verify_material",
    "weather_material",
    "weather_material_maps",
]


def _torch_installed() -> bool:
    import importlib.util

    from ._backend import torch_disabled_by_env

    return not torch_disabled_by_env() and importlib.util.find_spec("torch") is not None


if not _torch_installed():  # keep `from hypbr import *` working without torch
    __all__ = [n for n in __all__ if n not in _LAZY_TORCH_EXPORTS]

