"""Every text file TempoBench reads or writes names its encoding.

Without one, Python uses the locale's code page, which on Windows is a legacy
one: `report` crashed writing its first non-ASCII character there, and a
runs.jsonl holding a non-ASCII grid value would be misread.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "tembench"
TEXT_CALLS = {"open", "read_text", "write_text"}


def _mode(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    # open(path, mode) for the builtin; Path.open(mode) for the method.
    positional = call.args[1:] if isinstance(call.func, ast.Name) else call.args
    if positional and isinstance(positional[0], ast.Constant):
        return str(positional[0].value)
    return "r"


def _unencoded(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name not in TEXT_CALLS:
            continue
        if name == "open" and "b" in _mode(node):
            continue
        if not any(kw.arg == "encoding" for kw in node.keywords):
            found.append(f"{path.relative_to(SRC)}:{node.lineno}")
    return found


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_text_io_names_its_encoding(path: Path):
    assert _unencoded(path) == []
