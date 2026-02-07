from __future__ import annotations
import csv, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_HEADERS = ["cell","status","utc_ts","elapsed_s","event","wf_id","seed_run","message"]

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def start_cell_log(log_csv: Path, cell: str, message: str = "") -> float:
    _ensure_parent(log_csv)
    t0 = time.perf_counter()
    if not log_csv.exists():
        with log_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_HEADERS)
    with log_csv.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([cell,"START",_utc_now_iso(),"", "", "", "", message])
    print(f"[{cell}] START :: {message}")
    return t0

def end_cell_log(log_csv: Path, t0: float, cell: str, message: str = "") -> None:
    elapsed = time.perf_counter() - t0
    with log_csv.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([cell,"END",_utc_now_iso(),f"{elapsed:.3f}","", "", "", message])
    print(f"[{cell}] END   :: elapsed_s={elapsed:.3f} :: {message}")

def write_event(log_csv: Path, cell: str, event: str, message: str = "", wf_id: Optional[int]=None, seed_run: Optional[int]=None) -> None:
    with log_csv.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([cell,"INFO",_utc_now_iso(),"", event, wf_id if wf_id is not None else "", seed_run if seed_run is not None else "", message])
    print(f"[{cell}] INFO  :: {event} :: {message}")
