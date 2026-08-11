"""The release workflow's binary locator, tested against real Nuitka layouts.

The v0.1.0 release failed because release.yml assumed Nuitka wrote
`dist/tembench.dist/tembench.exe`. Nuitka names the output directory after the
entry module, so the file was at `dist/tembench_entry.dist/tembench.exe`. That
step only ran on a tag, so the assumption was never exercised until the release
itself. These tests exercise it on every commit instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / ".github" / "scripts"


def _load(name: str):
    """Import a workflow script the way CI runs it, straight from .github/."""
    spec = importlib.util.spec_from_file_location(f"_{name}", SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF fake")
    return path


def test_finds_the_windows_standalone_exe(tmp_path: Path):
    """The exact layout that broke v0.1.0: dist dir named after the entry module."""
    find = _load("find_binary")
    dist = tmp_path / "dist"
    wanted = _touch(dist / "tembench_entry.dist" / "tembench.exe")
    # Nuitka drops the package data beside the executable; it must not confuse
    # the search.
    _touch(dist / "tembench_entry.dist" / "tembench" / "reporting" / "assets" / "report.css")
    _touch(dist / "tembench_entry.dist" / "python312.dll")
    assert find.find_binary(dist, windows=True) == wanted


def test_prefers_the_self_contained_onefile_binary(tmp_path: Path):
    """Onefile leaves its intermediate standalone tree behind.

    Both copies are called tembench.bin. Only the shallow one runs on a machine
    that does not have the build directory, so that is the one to ship.
    """
    find = _load("find_binary")
    dist = tmp_path / "dist"
    onefile = _touch(dist / "tembench.bin")
    _touch(dist / "tembench_entry.dist" / "tembench.bin")
    assert find.find_binary(dist, windows=False) == onefile


def test_reports_the_tree_instead_of_a_bare_failure(tmp_path: Path, capsys):
    find = _load("find_binary")
    dist = tmp_path / "dist"
    _touch(dist / "tembench_entry.dist" / "something_else")
    with pytest.raises(SystemExit) as exc:
        find.find_binary(dist, windows=False)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "something_else" in out, "a failure must show what was actually built"


def test_windows_packaging_zips_the_whole_standalone_directory(tmp_path, monkeypatch):
    """The Windows exe cannot run alone: the DLLs beside it have to ship too."""
    import zipfile

    pkg = _load("package_binary")
    monkeypatch.setenv("RUNNER_OS", "Windows")
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "dist" / "tembench_entry.dist" / "tembench.exe")
    _touch(tmp_path / "dist" / "tembench_entry.dist" / "python312.dll")
    _touch(tmp_path / "dist" / "tembench_entry.dist" / "tembench" / "reporting" / "assets" / "report.css")

    assert pkg.main() == 0
    archive = tmp_path / "release" / "tembench-windows.zip"
    assert archive.is_file()
    names = zipfile.ZipFile(archive).namelist()
    assert any(n.endswith("tembench.exe") for n in names)
    assert any(n.endswith("python312.dll") for n in names), "DLLs must ship"
    assert any(n.endswith("report.css") for n in names), "packaged assets must ship"


@pytest.mark.parametrize("runner_os, expected", [("Linux", "tembench-linux"), ("macOS", "tembench-macos")])
def test_posix_packaging_emits_the_named_release_asset(tmp_path, monkeypatch, runner_os, expected):
    pkg = _load("package_binary")
    monkeypatch.setenv("RUNNER_OS", runner_os)
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "dist" / "tembench.bin")

    assert pkg.main() == 0
    asset = tmp_path / "release" / expected
    assert asset.is_file()
    # The upload step globs release/tembench-*; a differently named file would
    # be silently dropped from the release.
    assert asset.name.startswith("tembench-")
