"""Static path validator for MT5_Data_Extraction notebooks.

Scans all .ipynb files under PROJECT_ROOT (as JSON), extracts path-like
strings, and checks that:
  1. No hardcoded absolute paths outside PROJECT_ROOT
  2. Canonical paths (from path_contract) are resolvable
  3. All notebooks import path_contract from shared/contracts/

Usage:
    python tools/validate_paths_static.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve PROJECT_ROOT (same logic as path_contract)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent        # tools/
PROJECT_ROOT = SCRIPT_DIR.parent                     # MT5_Data_Extraction/

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "paths_audit_report.md"

# Patterns that look like paths
PATH_PATTERNS = [
    re.compile(r'[A-Z]:\\[^\s"\']{3,}', re.IGNORECASE),       # Windows absolute (min 3 chars after :\)
    re.compile(r'(?:Path\(|/)\S*(?:\.parquet|\.csv|\.json|\.zip|\.pkl)', re.IGNORECASE),
    re.compile(r'["\'](/[^"\']+)["\']'),                        # Unix absolute in strings
]

# False-positive exclusions (f-string escapes like "s:\n", "n:\n{var}")
_FALSE_POSITIVE = re.compile(r'^[a-zA-Z]:\\[\\nt{]', re.IGNORECASE)

# Patterns indicating path_contract usage
CONTRACT_IMPORT_PATTERN = re.compile(r'import\s+path_contract')
CONTRACT_SHARED_PATTERN = re.compile(r'shared.*contracts.*path_contract')

# Known safe references (not actual paths)
SAFE_PATTERNS = [
    "C:\\Users",           # Windows user paths in tracebacks etc.
    "C:\\Program Files",   # System paths
    "C:\\Windows",
]


def find_notebooks(root: Path) -> list[Path]:
    """Find all .ipynb files, excluding venv and _trash_review."""
    excluded = {"venv1", "_trash_review", ".git", "__pycache__", ".pytest_cache"}
    notebooks = []
    for p in root.rglob("*.ipynb"):
        if any(part in excluded for part in p.parts):
            continue
        notebooks.append(p)
    return sorted(notebooks)


def extract_paths_from_cell(source: str) -> list[str]:
    """Extract path-like strings from a cell's source code."""
    found = []
    for pattern in PATH_PATTERNS:
        for m in pattern.finditer(source):
            text = m.group(0)
            if _FALSE_POSITIVE.match(text):
                continue
            found.append(text)
    return found


def is_hardcoded_external(path_str: str, project_root: Path) -> bool:
    """Check if a path string points outside PROJECT_ROOT."""
    # Only check Windows absolute paths
    if not re.match(r'^[A-Z]:\\', path_str, re.IGNORECASE):
        return False
    # Ignore safe system paths
    for safe in SAFE_PATTERNS:
        if path_str.startswith(safe):
            return False
    # Check if it's under project root
    try:
        pr_str = str(project_root).lower().replace("/", "\\")
        path_lower = path_str.lower().replace("/", "\\")
        if path_lower.startswith(pr_str):
            return False
    except Exception:
        pass
    return True


def check_contract_import(cells: list[dict]) -> bool:
    """Check if notebook imports path_contract from shared/contracts/."""
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if CONTRACT_IMPORT_PATTERN.search(src):
            if CONTRACT_SHARED_PATTERN.search(src):
                return True
            # Old-style import (directly from root)
            return False
    return False


def check_canonical_dirs() -> dict[str, bool]:
    """Check if canonical directories exist or are creatable."""
    dirs = {
        "data/": PROJECT_ROOT / "data",
        "data/bulk_data/m5_raw/": PROJECT_ROOT / "data" / "bulk_data" / "m5_raw",
        "data/historical_data/m5_clean/": PROJECT_ROOT / "data" / "historical_data" / "m5_clean",
        "data/metadata/": PROJECT_ROOT / "data" / "metadata",
        "data/processed_data/": PROJECT_ROOT / "data" / "processed_data",
        "outputs/": PROJECT_ROOT / "outputs",
        "outputs/er_filter_5m/": PROJECT_ROOT / "outputs" / "er_filter_5m",
        "outputs/trend_v2/": PROJECT_ROOT / "outputs" / "trend_v2",
        "outputs/range_v1/": PROJECT_ROOT / "outputs" / "range_v1",
        "config/": PROJECT_ROOT / "config",
        "shared/contracts/": PROJECT_ROOT / "shared" / "contracts",
        "shared/contracts/path_contract.py": PROJECT_ROOT / "shared" / "contracts" / "path_contract.py",
    }
    return {name: p.exists() for name, p in dirs.items()}


def audit_notebook(nb_path: Path) -> dict:
    """Audit a single notebook for path issues."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    result = {
        "path": str(nb_path.relative_to(PROJECT_ROOT)),
        "n_cells": len(cells),
        "imports_contract": check_contract_import(cells),
        "uses_shared_contracts": False,
        "hardcoded_external": [],
        "all_paths": [],
        "status": "OK",
    }

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))

        if CONTRACT_SHARED_PATTERN.search(src):
            result["uses_shared_contracts"] = True

        paths = extract_paths_from_cell(src)
        for p in paths:
            entry = {"cell": i, "path": p}
            result["all_paths"].append(entry)
            if is_hardcoded_external(p, PROJECT_ROOT):
                result["hardcoded_external"].append(entry)

    # Determine status
    issues = []
    if not result["imports_contract"]:
        issues.append("NO_CONTRACT_IMPORT")
    if not result["uses_shared_contracts"]:
        issues.append("NOT_SHARED_CONTRACTS")
    if result["hardcoded_external"]:
        issues.append("HARDCODED_EXTERNAL")

    result["status"] = "ISSUES: " + ", ".join(issues) if issues else "OK"
    return result


def generate_report(results: list[dict], canonical: dict[str, bool]) -> str:
    """Generate markdown audit report."""
    lines = [
        "# Paths Audit Report",
        "",
        f"**Project Root**: `{PROJECT_ROOT}`",
        f"**Generated by**: `tools/validate_paths_static.py`",
        "",
        "---",
        "",
        "## Canonical Directories",
        "",
        "| Path | Exists |",
        "|------|--------|",
    ]
    for name, exists in canonical.items():
        icon = "OK" if exists else "MISSING"
        lines.append(f"| `{name}` | {icon} |")

    lines += [
        "",
        "---",
        "",
        "## Notebook Audit",
        "",
    ]

    for r in results:
        status_icon = "OK" if r["status"] == "OK" else "ISSUE"
        lines += [
            f"### {r['path']}",
            "",
            f"- **Status**: {status_icon} {r['status']}",
            f"- **Cells**: {r['n_cells']}",
            f"- **Imports path_contract**: {'Yes' if r['imports_contract'] else 'No'}",
            f"- **Uses shared/contracts**: {'Yes' if r['uses_shared_contracts'] else 'No'}",
            f"- **Paths found**: {len(r['all_paths'])}",
            f"- **Hardcoded external**: {len(r['hardcoded_external'])}",
        ]
        if r["hardcoded_external"]:
            lines.append("")
            lines.append("**External hardcoded paths:**")
            for entry in r["hardcoded_external"]:
                lines.append(f"  - Cell {entry['cell']}: `{entry['path']}`")
        lines.append("")

    # Summary
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "OK")
    lines += [
        "---",
        "",
        "## Summary",
        "",
        f"- **Total notebooks**: {total}",
        f"- **OK**: {ok}",
        f"- **With issues**: {total - ok}",
        "",
    ]

    return "\n".join(lines)


def main() -> int:
    print(f"[validate_paths_static] PROJECT_ROOT = {PROJECT_ROOT}")
    print()

    # Find notebooks
    notebooks = find_notebooks(PROJECT_ROOT)
    print(f"Found {len(notebooks)} notebooks:")
    for nb in notebooks:
        print(f"  {nb.relative_to(PROJECT_ROOT)}")
    print()

    # Check canonical dirs
    canonical = check_canonical_dirs()
    print("Canonical directories:")
    for name, exists in canonical.items():
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name}")
    print()

    # Audit each notebook
    results = []
    for nb in notebooks:
        r = audit_notebook(nb)
        results.append(r)
        icon = "OK" if r["status"] == "OK" else "!!"
        print(f"  [{icon}] {r['path']} — {r['status']}")
    print()

    # Generate report
    report = generate_report(results, canonical)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to: {REPORT_PATH}")

    # Exit code
    issues = sum(1 for r in results if r["status"] != "OK")
    missing = sum(1 for v in canonical.values() if not v)
    if issues or missing:
        print(f"\n{issues} notebook(s) with issues, {missing} canonical dir(s) missing.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
