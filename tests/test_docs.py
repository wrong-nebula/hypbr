"""Keep llms.txt (the file agents read) in sync with the real API and CLI."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import hypbr
from hypbr.cli import build_parser

LLMS = (Path(__file__).resolve().parents[1] / "llms.txt").read_text(encoding="utf-8")

DOCUMENTED = [
    "synthesize_material", "synthesize_all", "synthesize_hardops_plate", "list_materials",
    "search_atlas", "recommend_for_object", "get_material_spec", "get_atlas", "export_atlas_files",
    "weather_material", "resolve_weather_params", "infer_material_kind",
    "blend_materials", "export_material", "verify_material", "render_preview_html", "serve_preview", "run_graph",
]


def _doc_params(name: str) -> list[str] | None:
    m = re.search(rf"`{name}\(([^`]*?)\)", LLMS)
    assert m, f"{name} is not documented in llms.txt"
    inner = m.group(1).strip()
    if inner in ("", "..."):
        return [] if inner == "" else None
    out = []
    for part in inner.split(","):
        part = part.strip()
        if part in ("*", "...", ""):
            continue
        out.append(part.split("=")[0].strip())
    return out


@pytest.mark.parametrize("name", DOCUMENTED)
def test_llms_txt_signatures_match_code(name):
    doc = _doc_params(name)
    if doc is None:
        return
    real = list(inspect.signature(getattr(hypbr, name)).parameters)
    assert doc == real, f"llms.txt documents {name}({', '.join(doc)}) but the code is {name}({', '.join(real)})"


def _subparsers(parser):
    for action in parser._actions:
        if action.__class__.__name__ == "_SubParsersAction":
            return action.choices
    return {}


def test_llms_txt_cli_flags_exist():
    root = build_parser()
    checked = 0
    for cmd in re.findall(r"`(hypbr [^`]+)`", LLMS):
        tokens = cmd.split()[1:]
        parser, i = root, 0
        while i < len(tokens) and tokens[i] in _subparsers(parser):
            parser = _subparsers(parser)[tokens[i]]
            i += 1
        if parser is root:
            continue
        options = {s for a in parser._actions for s in a.option_strings}
        for flag in re.findall(r"--[a-z][a-z0-9-]*", " ".join(tokens[i:])):
            assert flag in options, f"`{cmd}` documents {flag}, which `{parser.prog}` does not accept"
            checked += 1
    assert checked >= 30
