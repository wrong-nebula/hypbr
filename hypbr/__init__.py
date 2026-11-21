"""hypbr — Hybrid Neural + Exact Periodic Procedural PBR Material Synthesizer."""
from __future__ import annotations

from .blend import blend_materials, compute_height_blend_mask
from .exporters import SUPPORTED_TARGETS, export_material
from .normal_graph import PRESETS, run_graph
from .preview import render_preview_html, serve_preview
from .synthesize import (
    list_materials,
    synthesize_all,
    synthesize_hardops_plate,
    synthesize_material,
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
)
from .weightcode import (
    apply_depthpro_weightcode,
    spectral_filter_3x3_kernel,
    tileable_4quad_blend,
)

__version__ = "0.6.0"

__all__ = [
    "__version__",
    "PRESETS",
    "SUPPORTED_TARGETS",
    "WEATHER_ARCHETYPES",
    "WEATHER_PRESETS",
    "apply_depthpro_weightcode",
    "apply_weathering_maps",
    "blend_materials",
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
    "synthesize_material",
    "tileable_4quad_blend",
    "verify_material",
    "weather_material",
]
