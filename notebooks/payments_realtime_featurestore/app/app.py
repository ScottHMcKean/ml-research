"""Payments real-time feature-store demo — Databricks App (FastAPI).

A cost-effective control plane for the demo:

* **Generator** — a background loop that synthesizes payment events and inserts them
  into the raw Unity Catalog table (a cheap stand-in for a Kafka/Kinesis source).
* **Scorer** (`/score`) — Stage-0 deterministic rule pre-filter (with an optional direct
  Lakebase counter lookup) followed by the LightGBM serving endpoint, which auto-joins
  online features. Per-request latency is recorded.
* **Backfill** (`/backfill`) — triggers the daily/monthly cache job (durable work runs as
  a Job, never in the app process).
* **Dashboard** (`/`) + **metrics** (`/metrics`) — live throughput and p50/p90/p99 latency.

Auth uses the app's injected service principal (`WorkspaceClient()` reads
DATABRICKS_CLIENT_ID/SECRET/HOST). Resource keys are injected as env vars by app.yaml.
"""
from __future__ import annotations

import os
import time
import math
import uuid
import random
import threading
import datetime as dt
from collections import deque
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient

# ---------------------------------------------------------------------------- config
CATALOG = os.getenv("CATALOG", "shm_skunkworks_catalog")
SCHEMA = os.getenv("SCHEMA", "payments")
RAW_EVENTS = f"{CATALOG}.{SCHEMA}.raw_events"
SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT", "payments-scoring")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
BACKFILL_JOB_ID = os.getenv("BACKFILL_JOB_ID")

w = WorkspaceClient()

# ----------------------------------------------------------------------- in-memory state
_latencies: deque[float] = deque(maxlen=5000)   # recent per-request ms
_events_generated = 0
_events_scored = 0
_blocked_by_rules = 0
_gen_thread: Optional[threading.Thread] = None
_gen_stop = threading.Event()
_lock = threading.Lock()

CHANNELS = ["ch_1", "ch_2", "ch_3", "ch_4"]
ITYPES = ["t1", "t2", "t3"]
RTYPES = ["std", "pre", "comp"]
CATS = ["A", "B", "C", "D", "E"]


# ------------------------------------------------------------------- synthetic events
def synth_event() -> dict:
    amount = round(abs(random.lognormvariate(3.5, 1.2)), 2)
    return {
        "event_id": f"EVT_{uuid.uuid4().hex[:16]}",
        "instrument_id": f"INS_{random.randint(0, 49_999):06d}",
        "account_id": f"ACC_{random.randint(0, 4_999):05d}",
        "bin_prefix": str(random.randint(400000, 559999)),
        "category_code": random.choice(CATS),
        "amount": amount,
        "channel": random.choices(CHANNELS, weights=[40, 30, 20, 10])[0],
        "instrument_type": random.choice(ITYPES),
        "request_type": random.choices(RTYPES, weights=[60, 30, 10])[0],
        "flag_a": 1 if random.random() < 0.08 else 0,
    }


def insert_events(events: list[dict]) -> None:
    """Insert a batch of events into the raw UC table via the SQL warehouse."""
    if not WAREHOUSE_ID:
        return
    now = dt.datetime.utcnow().isoformat()
    rows = ",\n".join(
        "(" + ", ".join([
            f"'{e['event_id']}'", f"timestamp'{now}'", f"date'{now[:10]}'",
            f"'{e['instrument_id']}'", f"'{e['account_id']}'", f"'{e['bin_prefix']}'",
            f"'{e['category_code']}'", str(e["amount"]), f"'{e['channel']}'",
            f"'{e['instrument_type']}'", f"'{e['request_type']}'", str(e["flag_a"]),
            "'pass'", "0",
        ]) + ")"
        for e in events
    )
    stmt = (
        f"INSERT INTO {RAW_EVENTS} (event_id, event_ts, event_date, instrument_id, "
        f"account_id, bin_prefix, category_code, amount, channel, instrument_type, "
        f"request_type, flag_a, outcome, blocked) VALUES {rows}"
    )
    w.statement_execution.execute_statement(warehouse_id=WAREHOUSE_ID, statement=stmt, wait_timeout="30s")


def _generator_loop(rate_per_sec: int, batch: int):
    global _events_generated
    while not _gen_stop.is_set():
        events = [synth_event() for _ in range(batch)]
        try:
            insert_events(events)
            with _lock:
                _events_generated += len(events)
        except Exception as exc:  # keep the loop alive on transient errors
            print(f"generator insert failed: {exc}")
        time.sleep(max(0.1, batch / max(1, rate_per_sec)))


# --------------------------------------------------------------------- Stage-0 rules
def stage0_rules(rec: dict) -> Optional[str]:
    """Cheap deterministic pre-filter. Returns a rule name if it fires, else None."""
    if rec["amount"] > 5000 and rec.get("flag_a"):
        return "r_high_amount_flagged"
    if rec.get("channel") == "ch_4" and rec["amount"] > 2000:
        return "r_risky_channel_high_amount"
    return None


# --------------------------------------------------------------------------- FastAPI
app = FastAPI(title="Payments Real-Time Feature Store Demo")


class ScoreRequest(BaseModel):
    instrument_id: Optional[str] = None
    account_id: Optional[str] = None
    category_code: Optional[str] = None
    amount: Optional[float] = None


@app.post("/score")
def score(req: ScoreRequest):
    global _events_scored, _blocked_by_rules
    # Fill any field the caller omitted from a synthetic event, so empty or partial
    # request bodies still produce a complete, scorable record (avoids None in the rules).
    rec = {**synth_event(), **{k: v for k, v in req.model_dump().items() if v is not None}}
    rec = {k: rec[k] for k in ("instrument_id", "account_id", "category_code", "amount")}

    t0 = time.perf_counter()
    # Stage 0: deterministic pre-filter (short-circuits before the model).
    fired = stage0_rules(rec)
    if fired:
        with _lock:
            _events_scored += 1
            _blocked_by_rules += 1
            _latencies.append((time.perf_counter() - t0) * 1000)
        return {"decision": "blocked", "stage": "rules", "rule": fired}

    # Stage 1: LightGBM with automatic online feature lookup.
    payload = {**rec, "event_ts": dt.datetime.utcnow().isoformat()}
    resp = w.serving_endpoints.query(name=SERVING_ENDPOINT, dataframe_records=[payload])
    elapsed = (time.perf_counter() - t0) * 1000
    # The fe.log_model LightGBM classifier serves the predicted class (0/1), not a
    # calibrated probability, so report it as the model's output rather than a score.
    output = float(resp.predictions[0]) if resp.predictions else 0.0
    with _lock:
        _events_scored += 1
        _latencies.append(elapsed)
    return {"decision": "blocked" if output >= 0.5 else "pass", "stage": "model",
            "model_output": round(output, 4), "latency_ms": round(elapsed, 1)}


@app.post("/generate")
def generate(rate_per_sec: int = 20, batch: int = 20, stop: bool = False):
    global _gen_thread
    if stop:
        _gen_stop.set()
        return {"status": "stopping generator"}
    if _gen_thread and _gen_thread.is_alive():
        return {"status": "generator already running"}
    _gen_stop.clear()
    _gen_thread = threading.Thread(target=_generator_loop, args=(rate_per_sec, batch), daemon=True)
    _gen_thread.start()
    return {"status": "generator started", "rate_per_sec": rate_per_sec, "batch": batch}


@app.post("/backfill")
def backfill(grain: str = "both"):
    if not BACKFILL_JOB_ID:
        return JSONResponse({"error": "BACKFILL_JOB_ID not configured"}, status_code=400)
    run = w.jobs.run_now(job_id=int(BACKFILL_JOB_ID), notebook_params={"grain": grain})
    return {"status": "backfill triggered", "run_id": run.run_id, "grain": grain}


@app.get("/metrics")
def metrics():
    with _lock:
        lat = sorted(_latencies)
        n = len(lat)
        gen, scored, blocked = _events_generated, _events_scored, _blocked_by_rules

    def pct(p):
        # Nearest-rank percentile: ceil(p/100 * n) - 1, clamped to [0, n-1].
        if not n:
            return None
        return round(lat[max(0, min(n - 1, math.ceil(p / 100 * n) - 1))], 1)

    return {
        "events_generated": gen,
        "events_scored": scored,
        "blocked_by_rules": blocked,
        "latency_ms": {"p50": pct(50), "p90": pct(90), "p99": pct(99), "samples": n},
        "generator_running": bool(_gen_thread and _gen_thread.is_alive()),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
<!doctype html><html><head><title>Payments Feature Store Demo</title>
<style>
 body{font-family:system-ui,sans-serif;margin:2rem;background:#0b0e14;color:#e6e6e6}
 h1{color:#ff6b6b} button{padding:.5rem 1rem;margin:.25rem;border:0;border-radius:6px;
 background:#1f6feb;color:#fff;cursor:pointer} .card{background:#161b22;padding:1rem;
 border-radius:8px;margin:.5rem 0;display:inline-block;min-width:160px}
 .big{font-size:2rem;font-weight:700} pre{background:#161b22;padding:1rem;border-radius:8px}
</style></head><body>
<h1>Payments Real-Time Feature Store</h1>
<p>Generator → raw table → feature store (Lakebase online) → LightGBM serving with automatic feature lookup.</p>
<div>
 <button onclick="call('/generate?rate_per_sec=20&batch=20','POST')">Start generator</button>
 <button onclick="call('/generate?stop=true','POST')">Stop generator</button>
 <button onclick="call('/score','POST')">Score one event</button>
 <button onclick="call('/backfill?grain=both','POST')">Backfill caches</button>
</div>
<div id="cards"></div>
<pre id="log"></pre>
<script>
async function call(url,method){
  const r=await fetch(url,{method:method||'GET',headers:{'Content-Type':'application/json'},
    body:method==='POST'?'{}':undefined});
  document.getElementById('log').textContent=JSON.stringify(await r.json(),null,2);
}
async function refresh(){
  const m=await (await fetch('/metrics')).json();
  document.getElementById('cards').innerHTML=
   card('Generated',m.events_generated)+card('Scored',m.events_scored)+
   card('Blocked (rules)',m.blocked_by_rules)+card('p50 ms',m.latency_ms.p50)+
   card('p99 ms',m.latency_ms.p99)+card('Generator',m.generator_running?'on':'off');
}
function card(t,v){return `<div class="card">${t}<div class="big">${v??'-'}</div></div>`}
setInterval(refresh,2000); refresh();
</script></body></html>
"""


@app.get("/health")
def health():
    return {"status": "ok", "endpoint": SERVING_ENDPOINT}
