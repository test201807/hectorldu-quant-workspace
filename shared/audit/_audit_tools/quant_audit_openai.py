import argparse
import json
import os
import time
import urllib.error
import urllib.request


def sanitize(s: str) -> str:
    if s is None: return ""
    # remove control chars except \t \n \r
    out = []
    for ch in s:
        o = ord(ch)
        if o in (9,10,13) or o >= 32:
            out.append(ch)
    return "".join(out).replace("\uFFFD","")

def extract_output_text(resp: dict) -> str:
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c.get("text", "")
    return ""

def post_responses(api_key: str, model: str, prompt: str) -> dict:
    url = "https://api.openai.com/v1/responses"
    body = {"model": model, "input": prompt}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode("utf-8", errors="replace")
    return json.loads(raw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Quant\_audit_out")
    ap.add_argument("--model", default="gpt-5.2-codex")
    ap.add_argument("--max-files", type=int, default=3000)
    ap.add_argument("--batch-max-chars", type=int, default=75000)
    ap.add_argument("--preview-chars", type=int, default=800)
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY","").strip()
    if not api_key:
        raise SystemExit("Falta OPENAI_API_KEY en variables de entorno.")

    out = os.path.abspath(args.out)
    digests_path = os.path.join(out, "digests.jsonl")
    if not os.path.exists(digests_path):
        raise SystemExit("No existe digests.jsonl. Corre primero quant_audit_local.py")

    batches_dir = os.path.join(out, "openai_batches")
    os.makedirs(batches_dir, exist_ok=True)

    # read digests -> compact lines
    compact_lines = []
    with open(digests_path, encoding="utf-8", errors="ignore") as f:
        for i, ln in enumerate(f):
            if i >= args.max_files:
                break
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            compact = {
                "path": d.get("path",""),
                "rel": d.get("rel",""),
                "ext": d.get("ext",""),
                "kind": d.get("kind",""),
                "headings": d.get("headings", None),
                "outline": d.get("outline", None),
                "preview": (d.get("preview","") or "")[:args.preview_chars],
            }
            compact_lines.append(json.dumps(compact, ensure_ascii=False))

    # batch by chars
    batches = []
    cur = []
    cur_chars = 0
    for ln in compact_lines:
        ln2 = sanitize(ln)
        l = len(ln2) + 1
        if (cur_chars + l) > args.batch_max_chars and cur:
            batches.append(cur)
            cur = []
            cur_chars = 0
        cur.append(ln2)
        cur_chars += l
    if cur:
        batches.append(cur)

    instructions = """Eres un auditor técnico para repos cuant/trading en Windows.
Te mando una lista de "digests" por archivo (path, rel, ext, kind, outline/headings y preview).
Tareas:
1) Para CADA archivo: resumen 1-2 líneas de qué hace/qué contiene; clasifica en: data_engine | filters | research | strategies | utils | docs | config | env | legacy | data | other.
2) Indica "move_hint": a qué carpeta superior debería ir (sin romper contratos; usa solo carpetas top-level razonables).
3) Señala red_flags por archivo si aplica (paths hardcodeados, outputs fuera de contrato, duplicados, varios data_roots, etc).

Responde SOLO JSON (sin markdown), con forma:
{
  "files": [
    {
      "path": "...",
      "category": "...",
      "summary": "...",
      "move_hint": "...",
      "red_flags": ["..."]
    }
  ]
}
"""

    all_files = []
    for bi, b in enumerate(batches):
        payload = "\n".join(b)
        prompt = instructions + "\n\nDIGESTS:\n" + payload

        print(f"Batch {bi+1}/{len(batches)} ...")
        try:
            resp = post_responses(api_key, args.model, prompt)
            raw_path = os.path.join(batches_dir, f"batch_{bi:03d}.json")
            with open(raw_path, "w", encoding="utf-8") as rf:
                json.dump(resp, rf, ensure_ascii=False, indent=2)

            out_text = extract_output_text(resp)
            if not out_text:
                raise RuntimeError("No output_text en respuesta.")

            obj = json.loads(out_text)
            all_files.extend(obj.get("files", []))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            print("HTTPError:", e.code, body[:500])
        except Exception as e:
            print("Error:", str(e)[:500])

        time.sleep(0.25)

    final_path = os.path.join(out, "openai_files_map.json")
    with open(final_path, "w", encoding="utf-8") as ff:
        json.dump({"files": all_files}, ff, ensure_ascii=False, indent=2)

    print("OK ->", final_path)

if __name__ == "__main__":
    main()
