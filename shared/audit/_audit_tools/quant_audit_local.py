import os, re, json, csv, argparse, datetime, traceback

EXCLUDE_DIRS = {
  ".git",".venv","venv","__pycache__","node_modules",
  ".mypy_cache",".pytest_cache",".ruff_cache",".idea",".vscode",
  "site-packages","dist","build",".tox",".cache"
}

CODE_EXT = {".py",".ps1",".psm1",".sql",".js",".ts",".tsx",".jsx",".r",".mql5",".mq5",".mqh",".ipynb"}
CFG_EXT  = {".json",".toml",".yml",".yaml",".ini",".cfg",".conf"}
DOC_EXT  = {".md",".txt",".rst"}
DATA_EXT = {".csv",".parquet",".arrow",".feather",".pkl",".pickle",".h5",".hdf5",".db"}
ARC_EXT  = {".zip",".7z",".rar",".gz",".bz2",".tar"}
BIN_EXT  = {".dll",".exe",".pyd",".so",".bin",".lib"}

CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def kind_from_ext(ext: str) -> str:
    ext = (ext or "").lower()
    if ext in CODE_EXT: return "code"
    if ext in CFG_EXT:  return "config"
    if ext in DOC_EXT:  return "docs"
    if ext in DATA_EXT: return "data"
    if ext in ARC_EXT:  return "archive"
    if ext in BIN_EXT:  return "binary"
    return "other"

def safe_text(s: str) -> str:
    if s is None: return ""
    s = s.replace("\uFFFD","")
    s = CTRL_RE.sub("", s)
    return s

def rel_depth(root: str, full: str) -> int:
    rel = os.path.relpath(full, root)
    if rel == ".": return 0
    return rel.count(os.sep) + 1

def top_folder(rel: str) -> str:
    rel = rel.replace("/", os.sep)
    parts = rel.split(os.sep)
    return parts[0] if parts and parts[0] not in ("",".") else "."

PY_IMPORT_RE = re.compile(r"(?m)^\s*(from\s+\S+\s+import\s+.+|import\s+.+)\s*$")
PY_DEF_RE    = re.compile(r"(?m)^\s*(def|class)\s+([A-Za-z_]\w*)\s*\(")
MD_HEAD_RE   = re.compile(r"(?m)^\s*#{1,6}\s+.+$")

def py_outline(text: str, max_imports=80, max_defs=120):
    imports = []
    defs = []
    for m in PY_IMPORT_RE.finditer(text):
        imports.append(m.group(0).strip())
        if len(imports) >= max_imports: break
    for m in PY_DEF_RE.finditer(text):
        defs.append(f"{m.group(1)}:{m.group(2)}")
        if len(defs) >= max_defs: break
    return {"imports": imports, "defs": defs}

def md_headings(text: str, max_heads=80):
    heads = []
    for m in MD_HEAD_RE.finditer(text):
        heads.append(m.group(0).strip())
        if len(heads) >= max_heads: break
    return heads

def ipynb_extract(path: str, max_cells=120, max_chars=20000):
    # Parse completo (limite por tamaño en main). Si falla, fallback a texto crudo.
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        out = []
        i = 0
        for c in cells:
            if i >= max_cells: break
            ctype = c.get("cell_type","?")
            src = c.get("source","")
            if isinstance(src, list): src = "".join(src)
            if not src: 
                i += 1
                continue
            out.append(f"### cell[{i}] ({ctype})\n{src}\n")
            i += 1
            if sum(len(x) for x in out) >= max_chars: break
        text = "".join(out)
        return text[:max_chars]
    except Exception:
        try:
            with open(path, "rb") as f:
                raw = f.read(max_chars*4)
            return raw.decode("utf-8", errors="replace")[:max_chars]
        except Exception:
            return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Quant")
    ap.add_argument("--out",  default=r"C:\Quant\_audit_out")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--max-files", type=int, default=50000)
    ap.add_argument("--max-text-kb", type=int, default=1024)
    ap.add_argument("--max-ipynb-kb", type=int, default=10240)
    ap.add_argument("--max-chars", type=int, default=20000)
    ap.add_argument("--max-cells", type=int, default=120)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out  = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    manifest_path = os.path.join(out, "manifest.jsonl")
    digests_path  = os.path.join(out, "digests.jsonl")
    catalog_path  = os.path.join(out, "catalog.csv")
    summary_path  = os.path.join(out, "local_summary.md")
    errors_path   = os.path.join(out, "errors.log")

    for p in (manifest_path, digests_path, catalog_path, summary_path, errors_path):
        try:
            if os.path.exists(p): os.remove(p)
        except Exception:
            pass

    def log_err(msg):
        try:
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            with open(errors_path, "a", encoding="utf-8") as ef:
                ef.write(f"{ts}\t{msg}\n")
        except Exception:
            pass

    ext_stats = {}
    folder_stats = {}
    biggest = []

    def guess_category(rel, ext):
        s = rel.lower().replace("\\","/")
        if "mt5" in s or "data_extraction" in s or "data_engine" in s: return "data_engine"
        if "er_filter" in s or "/filters" in s or "filter_" in s: return "filters"
        if "strategylab" in s or "/strateg" in s: return "strategies"
        if ext == ".ipynb" or "/notebook" in s or "/research" in s: return "research"
        if "/utils" in s or "/lib" in s or "helpers" in s: return "utils"
        if "/docs" in s or ext in (".md",".rst",".txt"): return "docs"
        if "/outputs" in s or "/output" in s: return "outputs"
        if "/metadata" in s: return "metadata"
        return "other"

    file_counter = 0

    with open(manifest_path, "w", encoding="utf-8") as mw, \
         open(digests_path,  "w", encoding="utf-8") as dw, \
         open(catalog_path,  "w", encoding="utf-8", newline="") as cw:

        catw = csv.DictWriter(cw, fieldnames=[
            "path","rel","top_folder","ext","kind","size_kb","last_write",
            "guess_category"
        ])
        catw.writeheader()

        for dirpath, dirnames, filenames in os.walk(root):
            # prune excludes
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

            for fn in filenames:
                full = os.path.join(dirpath, fn)
                try:
                    if rel_depth(root, full) > args.max_depth:
                        continue
                    st = os.stat(full)
                except Exception:
                    continue

                file_counter += 1
                if file_counter > args.max_files:
                    break

                rel = os.path.relpath(full, root)
                ext = os.path.splitext(fn)[1].lower()
                kind = kind_from_ext(ext)
                size = int(st.st_size)
                size_kb = round(size / 1024.0, 1)
                lw = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
                tf = top_folder(rel)

                # manifest
                mrec = {
                    "path": full, "rel": rel,
                    "ext": ext if ext else "(noext)",
                    "kind": kind,
                    "size_bytes": size,
                    "size_kb": size_kb,
                    "last_write": lw
                }
                mw.write(json.dumps(mrec, ensure_ascii=False) + "\n")

                # stats
                ext_key = mrec["ext"]
                ext_stats.setdefault(ext_key, {"ext": ext_key, "count": 0, "bytes": 0})
                ext_stats[ext_key]["count"] += 1
                ext_stats[ext_key]["bytes"] += size

                folder_stats.setdefault(tf, {"folder": tf, "count": 0, "bytes": 0})
                folder_stats[tf]["count"] += 1
                folder_stats[tf]["bytes"] += size

                biggest.append({"path": full, "kb": size_kb})

                # catalog row
                catw.writerow({
                    "path": full, "rel": rel, "top_folder": tf,
                    "ext": ext if ext else "(noext)",
                    "kind": kind,
                    "size_kb": size_kb,
                    "last_write": lw,
                    "guess_category": guess_category(rel, ext)
                })

                # digests (solo texto razonable)
                is_text = (kind in ("code","config","docs")) or (ext == ".ipynb")
                if not is_text:
                    continue

                # size guards
                if ext == ".ipynb":
                    if size_kb > args.max_ipynb_kb:
                        continue
                else:
                    if size_kb > args.max_text_kb:
                        continue

                try:
                    outline = None
                    headings = None

                    if ext == ".ipynb":
                        preview = ipynb_extract(full, max_cells=args.max_cells, max_chars=args.max_chars)
                    else:
                        with open(full, "rb") as f:
                            raw = f.read(args.max_chars * 4)
                        preview = raw.decode("utf-8", errors="replace")[:args.max_chars]

                    preview = safe_text(preview)

                    if ext == ".py":
                        outline = py_outline(preview)
                    elif ext == ".md":
                        headings = md_headings(preview)

                    drec = {
                        "path": full, "rel": rel,
                        "kind": kind,
                        "ext": ext if ext else "(noext)",
                        "size_kb": size_kb,
                        "last_write": lw,
                        "headings": headings,
                        "outline": outline,
                        "preview": preview
                    }
                    dw.write(json.dumps(drec, ensure_ascii=False) + "\n")
                except Exception as e:
                    log_err(f"Digest fail: {full} :: {e}")

            if file_counter > args.max_files:
                break

    # summary
    ext_top = sorted(ext_stats.values(), key=lambda x: x["bytes"], reverse=True)[:30]
    folder_top = sorted(folder_stats.values(), key=lambda x: x["bytes"], reverse=True)[:50]
    big_top = sorted(biggest, key=lambda x: x["kb"], reverse=True)[:40]

    total_bytes = sum(x["bytes"] for x in ext_stats.values())

    lines = []
    lines.append("# Quant Audit — Resumen Local")
    lines.append("")
    lines.append(f"ROOT: {root}")
    lines.append(f"Generado: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Archivos considerados: {min(file_counter, args.max_files)}")
    lines.append(f"Tamaño total (MB): {round(total_bytes/1024/1024, 2)}")
    lines.append("")
    lines.append("## Top extensiones (por bytes)")
    for x in ext_top:
        lines.append(f"- {x['ext']} :: count={x['count']} :: MB={round(x['bytes']/1024/1024,2)}")
    lines.append("")
    lines.append("## Top carpetas (primer nivel) por bytes")
    for x in folder_top:
        lines.append(f"- {x['folder']} :: count={x['count']} :: MB={round(x['bytes']/1024/1024,2)}")
    lines.append("")
    lines.append("## 40 archivos más pesados (KB)")
    for x in big_top:
        lines.append(f"- {x['kb']} KB :: {x['path']}")
    lines.append("")
    lines.append("## Salidas")
    lines.append(f"- manifest: {manifest_path}")
    lines.append(f"- digests : {digests_path}")
    lines.append(f"- catalog : {catalog_path}")
    lines.append(f"- errors  : {errors_path}")

    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("\n".join(lines) + "\n")

    # final prints
    def count_lines(p):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    print("OK")
    print("manifest lines:", count_lines(manifest_path))
    print("digests  lines:", count_lines(digests_path))
    print("catalog  path :", catalog_path)
    print("summary  path :", summary_path)

if __name__ == "__main__":
    main()
