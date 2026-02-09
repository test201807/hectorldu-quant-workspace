"""Smoke tests: compile all .py files and import every src/ module.

Run from repo root:
    pytest tests/test_imports.py -v
    python tests/test_imports.py          # also works standalone

No notebooks executed, no data downloaded, no API keys required.
"""
from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

try:
    import pytest

    _parametrize = pytest.mark.parametrize
except ModuleNotFoundError:  # standalone runner doesn't need pytest
    pytest = None  # type: ignore[assignment]

    def _parametrize(*_a, **_kw):  # type: ignore[misc]  # noqa: E303
        """No-op decorator when pytest is absent."""
        return lambda fn: fn

# ── Resolve repo root ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECTS = REPO_ROOT / "projects"
SHARED = REPO_ROOT / "shared"

VENV_DIRS = {".venv", "venv", "venv1", "btc_env", "env", "__pycache__"}


# ── Helper ─────────────────────────────────────────────────────────
def _collect_py_files() -> list[Path]:
    """All .py files under projects/ and shared/, excluding venvs."""
    files: list[Path] = []
    for root_dir in (PROJECTS, SHARED):
        if not root_dir.exists():
            continue
        for p in root_dir.rglob("*.py"):
            if not any(part in VENV_DIRS for part in p.parts):
                files.append(p)
    return sorted(files)


def _collect_src_modules() -> list[tuple[str, Path, str]]:
    """Return (project_name, src_dir, dotted_module) for every src/ module."""
    modules: list[tuple[str, Path, str]] = []
    for proj_dir in sorted(PROJECTS.iterdir()):
        src_dir = proj_dir / "src"
        if not src_dir.is_dir():
            continue
        for py_file in sorted(src_dir.rglob("*.py")):
            if any(part in VENV_DIRS for part in py_file.parts):
                continue
            rel = py_file.relative_to(src_dir)
            # Convert path to dotted module name
            if rel.name == "__init__.py":
                parts = list(rel.parent.parts)
            else:
                parts = list(rel.with_suffix("").parts)
            if not parts:
                continue
            dotted = ".".join(parts)
            modules.append((proj_dir.name, src_dir, dotted))
    return modules


# ── 1. Syntax check ───────────────────────────────────────────────
PY_FILES = _collect_py_files()


@_parametrize(
    "py_file",
    PY_FILES,
    ids=[str(f.relative_to(REPO_ROOT)) for f in PY_FILES],
)
def test_syntax(py_file: Path) -> None:
    """Every .py file must compile without syntax errors."""
    py_compile.compile(str(py_file), doraise=True)


# ── 2. Import every src/ module ───────────────────────────────────
SRC_MODULES = _collect_src_modules()


@_parametrize(
    "project,src_dir,module",
    SRC_MODULES,
    ids=[f"{proj}::{mod}" for proj, _, mod in SRC_MODULES],
)
def test_import(project: str, src_dir: Path, module: str) -> None:
    """Every src/ module must import without error."""
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    try:
        importlib.import_module(module)
    finally:
        if src_str in sys.path:
            sys.path.remove(src_str)


# ── Standalone runner ──────────────────────────────────────────────
if __name__ == "__main__":
    failures = 0

    print(f"=== Syntax check ({len(PY_FILES)} files) ===")
    for f in PY_FILES:
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as exc:
            print(f"  FAIL  {f.relative_to(REPO_ROOT)}: {exc}")
            failures += 1
    if failures == 0:
        print(f"  PASS  All {len(PY_FILES)} files compile OK")

    print(f"\n=== Import smoke tests ({len(SRC_MODULES)} modules) ===")
    for proj, src_dir, mod in SRC_MODULES:
        src_str = str(src_dir)
        sys.path.insert(0, src_str)
        try:
            importlib.import_module(mod)
            print(f"  PASS  {proj}::{mod}")
        except Exception as exc:
            print(f"  FAIL  {proj}::{mod}: {exc}")
            failures += 1
        finally:
            sys.path.remove(src_str)

    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {failures} failure(s)")
    raise SystemExit(1 if failures else 0)
