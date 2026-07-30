"""Payments real-time feature-store demo — Databricks App (FastAPI).

An instrumented walk of the real-time scoring architecture. The app drives the loop and times
each component from `docs/architecture/03_latency_path`:

    1. READ         consume a transaction from the Kafka topic
    2. INFERENCE     score via the serving endpoint — LightGBM with **automatic online feature
                     lookup** through the Feature Engineering API (the model was logged with
                     `fe.log_model`, so the endpoint joins the online features itself)
    3. WRITE BACK    produce the decision to the Kafka results topic

The dashboard renders one latency gauge per stage (p50/p99), so the architecture diagram and
the running system line up one-to-one. Feature reads happen **inside** the serving endpoint via
the Feature Engineering API — the app never talks to the online store directly.

Auth: the app's injected service principal (`WorkspaceClient()` reads
DATABRICKS_CLIENT_ID/SECRET/HOST).

Env (app.yaml):
  CATALOG / SCHEMA / SERVING_ENDPOINT   demo identifiers
  REDPANDA_BROKER                       "localhost:9092" (in-app broker; blank disables Kafka)
"""
from __future__ import annotations

import os
import json
import time
import uuid
import random
import shutil
import atexit
import subprocess
import threading
import datetime as dt
from collections import deque
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient

# ---------------------------------------------------------------------------- config
CATALOG = os.getenv("CATALOG", "shm_catalog")
SCHEMA = os.getenv("SCHEMA", "payments")
SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT", "payments-scoring")

# In-app Redpanda broker (Kafka wire protocol). Blank disables the Kafka stages.
REDPANDA_BROKER = os.getenv("REDPANDA_BROKER", "localhost:9092").strip()
TOPIC_IN = os.getenv("KAFKA_TOPIC", "transactions")
TOPIC_OUT = os.getenv("KAFKA_TOPIC_OUT", "decisions")

# External-broker egress probe (optional). All values come from env/secrets only — never from
# the request, never echoed back. Set these to test whether this app can reach a managed
# broker (e.g. Confluent Cloud = PLAIN, Redpanda Cloud = SCRAM-SHA-256).
EXT_KAFKA_BOOTSTRAP = os.getenv("EXT_KAFKA_BOOTSTRAP", "").strip()
EXT_KAFKA_MECHANISM = os.getenv("EXT_KAFKA_MECHANISM", "PLAIN").strip()  # PLAIN|SCRAM-SHA-256|SCRAM-SHA-512
EXT_KAFKA_USERNAME = os.getenv("EXT_KAFKA_USERNAME", "").strip()
EXT_KAFKA_PASSWORD = os.getenv("EXT_KAFKA_PASSWORD", "").strip()

STAGES = ["read", "inference", "write_back"]

w = WorkspaceClient()

# ----------------------------------------------------------------------- in-memory state
_samples: dict[str, deque] = {s: deque(maxlen=2000) for s in STAGES + ["total"]}
_counts = {"scored": 0, "blocked": 0}
_lock = threading.Lock()
_pipe_thread: Optional[threading.Thread] = None
_pipe_stop = threading.Event()

CATS = ["A", "B", "C", "D", "E"]


def synth_event() -> dict:
    return {
        "event_id": f"EVT_{uuid.uuid4().hex[:16]}",
        "instrument_id": f"INS_{random.randint(0, 49_999):06d}",
        "account_id": f"ACC_{random.randint(0, 4_999):05d}",
        "category_code": random.choice(CATS),
        "amount": round(abs(random.lognormvariate(3.5, 1.2)), 2),
    }


# --------------------------------------------------------------------- Redpanda broker
_broker_proc: Optional[subprocess.Popen] = None
_broker_status = "not-started"


def start_broker() -> str:
    """Start an in-process Redpanda broker in dev mode. Returns a status string.

    Redpanda is a single static binary. If it isn't on PATH (e.g. the Databricks Apps
    container ships no Kafka binary) the Kafka stages fall back to an in-process queue and the
    rest of the app still works.
    """
    global _broker_proc
    if not REDPANDA_BROKER:
        return "disabled"
    if _broker_proc and _broker_proc.poll() is None:
        return "running"
    rpk = shutil.which("redpanda") or shutil.which("rpk")
    if not rpk:
        return "binary-not-found"
    data_dir = "/tmp/redpanda"
    os.makedirs(data_dir, exist_ok=True)
    cmd = [
        shutil.which("redpanda") or rpk, "start",
        "--overprovisioned", "--smp", "1", "--memory", "512M",
        "--reserve-memory", "0M", "--node-id", "0", "--check=false",
        "--kafka-addr", f"PLAINTEXT://{REDPANDA_BROKER}",
        "--advertise-kafka-addr", f"PLAINTEXT://{REDPANDA_BROKER}",
    ]
    _broker_proc = subprocess.Popen(cmd, cwd=data_dir,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(stop_broker)
    time.sleep(5)  # let the broker bind the port before clients connect
    return "running" if _broker_proc.poll() is None else "failed"


def stop_broker() -> None:
    global _broker_proc
    if _broker_proc and _broker_proc.poll() is None:
        _broker_proc.terminate()


# --------------------------------------------------------------------- Kafka clients
# Two interchangeable backends behind the same tiny produce/poll shape:
#   * real Kafka (confluent-kafka) when a Redpanda/Kafka broker is actually reachable, or
#   * an in-process queue when it is not (the Databricks Apps container has no Kafka binary).
# The stub preserves the READ / WRITE-BACK stages and their timing; only the wire protocol
# differs. The 09_kafka_io notebook covers a real broker over the network.
import queue as _queue

_stub_topics: dict[str, "_queue.Queue"] = {}


def _use_real_kafka() -> bool:
    return _broker_status == "running"


class _StubProducer:
    def produce(self, topic, key=None, value=None):
        _stub_topics.setdefault(topic, _queue.Queue()).put(value)

    def flush(self, timeout=0):
        return 0


class _StubMessage:
    def __init__(self, value):
        self._v = value

    def value(self):
        return self._v

    def error(self):
        return None


class _StubConsumer:
    def __init__(self, topic):
        self._q = _stub_topics.setdefault(topic, _queue.Queue())

    def poll(self, timeout=1.0):
        try:
            return _StubMessage(self._q.get(timeout=timeout))
        except _queue.Empty:
            return None

    def close(self):
        pass


_producer = None


def get_producer():
    global _producer
    if not REDPANDA_BROKER:
        return None
    if _producer is None:
        if _use_real_kafka():
            from confluent_kafka import Producer
            _producer = Producer({"bootstrap.servers": REDPANDA_BROKER})
        else:
            _producer = _StubProducer()
    return _producer


def make_consumer():
    if not _use_real_kafka():
        return _StubConsumer(TOPIC_IN)
    from confluent_kafka import Consumer
    c = Consumer({
        "bootstrap.servers": REDPANDA_BROKER,
        "group.id": "payments-app",
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })
    c.subscribe([TOPIC_IN])
    return c


# --------------------------------------------------------------------------- the pipeline
def _record(stage_ms: dict, total_ms: float, blocked: bool) -> None:
    with _lock:
        for s, v in stage_ms.items():
            _samples[s].append(v)
        _samples["total"].append(total_ms)
        _counts["scored"] += 1
        if blocked:
            _counts["blocked"] += 1


def score_one(consumer, producer) -> Optional[dict]:
    """Run one transaction through the three stages, timing each. Returns a decision dict."""
    t0 = time.perf_counter()

    # 1 · READ — pull the next transaction off the inbound topic.
    msg = consumer.poll(1.0)
    if msg is None or msg.error():
        return None
    txn = json.loads(msg.value())
    t_read = time.perf_counter()

    # 2 · INFERENCE — the serving endpoint scores it, looking up online features itself via
    # the Feature Engineering API (fe.log_model wired automatic feature lookup at serve time).
    payload = {"instrument_id": txn["instrument_id"], "account_id": txn["account_id"],
               "category_code": txn["category_code"], "amount": float(txn["amount"]),
               "event_ts": dt.datetime.utcnow().isoformat()}
    resp = w.serving_endpoints.query(name=SERVING_ENDPOINT, dataframe_records=[payload])
    output = float(resp.predictions[0]) if resp.predictions else 0.0
    blocked = output >= 0.5
    t_inf = time.perf_counter()

    # 3 · WRITE BACK — produce the decision to the outbound topic.
    decision = {"event_id": txn["event_id"], "instrument_id": txn["instrument_id"],
                "decision": "blocked" if blocked else "pass",
                "model_output": round(output, 4), "scored_at": dt.datetime.utcnow().isoformat()}
    producer.produce(TOPIC_OUT, key=txn["instrument_id"], value=json.dumps(decision))
    producer.flush(5)
    t_out = time.perf_counter()

    ms = lambda a, b: (b - a) * 1000.0
    stage_ms = {"read": ms(t0, t_read), "inference": ms(t_read, t_inf),
                "write_back": ms(t_inf, t_out)}
    _record(stage_ms, ms(t0, t_out), blocked)
    return {**decision, "stage_ms": {k: round(v, 1) for k, v in stage_ms.items()}}


def _pipeline_loop(rate_per_sec: int):
    """Produce synthetic transactions and score them, end to end, until stopped."""
    producer = get_producer()
    consumer = make_consumer()
    interval = 1.0 / max(1, rate_per_sec)
    try:
        while not _pipe_stop.is_set():
            producer.produce(TOPIC_IN, key="k", value=json.dumps(synth_event()))
            producer.flush(5)
            try:
                score_one(consumer, producer)
            except Exception as exc:  # keep the loop alive on transient errors
                print(f"pipeline error: {exc}")
            time.sleep(interval)
    finally:
        consumer.close()


# --------------------------------------------------------------------------- FastAPI
app = FastAPI(title="Payments Real-Time Feature Store — Latency Walk")


@app.on_event("startup")
def _startup():
    global _broker_status
    _broker_status = start_broker()
    print("broker:", _broker_status)


class ScoreRequest(BaseModel):
    instrument_id: Optional[str] = None
    account_id: Optional[str] = None
    category_code: Optional[str] = None
    amount: Optional[float] = None


@app.post("/score")
def score(req: ScoreRequest):
    """Score a single transaction through the three stages and return the per-stage timings."""
    if not REDPANDA_BROKER:
        return JSONResponse({"error": "Kafka disabled (REDPANDA_BROKER unset)"}, status_code=400)
    producer = get_producer()
    consumer = make_consumer()
    try:
        event = {**synth_event(), **{k: v for k, v in req.model_dump().items() if v is not None}}
        producer.produce(TOPIC_IN, key="k", value=json.dumps(event))
        producer.flush(5)
        for _ in range(5):  # small retry so the just-produced message is available to poll
            out = score_one(consumer, producer)
            if out:
                return out
        return JSONResponse({"error": "no message scored"}, status_code=504)
    finally:
        consumer.close()


@app.post("/generate")
def generate(rate_per_sec: int = 10, stop: bool = False):
    global _pipe_thread
    if stop:
        _pipe_stop.set()
        return {"status": "stopping"}
    if _pipe_thread and _pipe_thread.is_alive():
        return {"status": "already running"}
    _pipe_stop.clear()
    _pipe_thread = threading.Thread(target=_pipeline_loop, args=(rate_per_sec,), daemon=True)
    _pipe_thread.start()
    return {"status": "started", "rate_per_sec": rate_per_sec}


def _pct(vals, p):
    if not vals:
        return None
    import math
    s = sorted(vals)
    return round(s[max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))], 1)


@app.get("/metrics")
def metrics():
    with _lock:
        snap = {s: list(v) for s, v in _samples.items()}
        scored, blocked = _counts["scored"], _counts["blocked"]
    stages = {s: {"p50": _pct(snap[s], 50), "p99": _pct(snap[s], 99), "n": len(snap[s])}
              for s in STAGES}
    return {
        "stages": stages,
        "total": {"p50": _pct(snap["total"], 50), "p99": _pct(snap["total"], 99)},
        "scored": scored, "blocked": blocked,
        "running": bool(_pipe_thread and _pipe_thread.is_alive()),
    }


@app.get("/health")
def health():
    return {"status": "ok",
            "kafka_backend": "kafka" if _use_real_kafka() else "in-process queue",
            "broker_status": _broker_status,
            "endpoint": SERVING_ENDPOINT}


@app.get("/kafka_probe")
def kafka_probe():
    """Bare egress diagnostic: can this app reach the external broker configured via env?

    Reads broker/creds from env only; never accepts them from the request and never returns
    the address, credentials, or raw error text. Response is deliberately minimal.
    """
    if not EXT_KAFKA_BOOTSTRAP:
        return {"configured": False, "reachable": False, "detail": "EXT_KAFKA_BOOTSTRAP unset"}
    conf = {"bootstrap.servers": EXT_KAFKA_BOOTSTRAP, "socket.timeout.ms": 8000}
    if EXT_KAFKA_USERNAME and EXT_KAFKA_PASSWORD:
        conf.update({"security.protocol": "SASL_SSL", "sasl.mechanism": EXT_KAFKA_MECHANISM,
                     "sasl.username": EXT_KAFKA_USERNAME, "sasl.password": EXT_KAFKA_PASSWORD})
    try:
        from confluent_kafka.admin import AdminClient
        md = AdminClient(conf).list_topics(timeout=8)
        return {"configured": True, "reachable": True,
                "broker_count": len(md.brokers), "mechanism": EXT_KAFKA_MECHANISM}
    except Exception as exc:
        return {"configured": True, "reachable": False, "mechanism": EXT_KAFKA_MECHANISM,
                "error_type": type(exc).__name__}


# --------------------------------------------------------------------------- dashboard
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return _DASHBOARD_HTML


_DASHBOARD_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Payments Latency Walk</title>
<style>
 :root{--surface:#1a1a19;--card:#232320;--ink:#ffffff;--muted:#898781;
  --s1:#3987e5;--s2:#199e70;--s3:#9085e9;--good:#0ca30c;--crit:#d03b3b}
 *{box-sizing:border-box} body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  margin:0;padding:2rem;background:#0d0d0d;color:var(--ink)}
 h1{font-size:1.4rem;margin:0 0 .25rem} p.sub{color:var(--muted);margin:.2rem 0 1.2rem}
 button{padding:.55rem 1rem;margin:.25rem .4rem .25rem 0;border:0;border-radius:8px;
  background:#2a2a28;color:#fff;cursor:pointer;font-size:.9rem}
 button.primary{background:var(--s1)} button:hover{filter:brightness(1.15)}
 .flow{display:flex;gap:.6rem;flex-wrap:wrap;margin:1.2rem 0}
 .gauge{background:var(--card);border:1px solid rgba(255,255,255,.08);border-radius:12px;
  padding:1rem 1.1rem;flex:1;min-width:190px}
 .gauge .name{font-size:.85rem;color:var(--muted);margin-bottom:.5rem}
 .gauge .p50{font-size:2rem;font-weight:700;font-variant-numeric:tabular-nums}
 .gauge .unit{font-size:.9rem;color:var(--muted);font-weight:400}
 .gauge .p99{font-size:.8rem;color:var(--muted);margin-top:.15rem;font-variant-numeric:tabular-nums}
 .gauge .desc{font-size:.72rem;color:var(--muted);margin-top:.5rem;line-height:1.35}
 .bar{height:6px;border-radius:3px;background:#333;margin:.6rem 0;overflow:hidden}
 .bar > i{display:block;height:100%;border-radius:3px}
 .arrow{align-self:center;color:var(--muted);font-size:1.3rem}
 .totals{display:flex;gap:1.2rem;margin:.4rem 0 1rem;color:var(--muted);font-size:.9rem}
 .totals b{color:var(--ink);font-variant-numeric:tabular-nums}
 .story{background:var(--card);border:1px solid rgba(255,255,255,.08);border-radius:12px;
  padding:1rem 1.2rem;margin:1rem 0;font-size:.9rem;line-height:1.5;color:#c3c2b7}
 .story b{color:var(--ink)} .story code{color:#e6e6e6;background:#0d0d0d;padding:0 .3rem;
  border-radius:4px;font-size:.82rem}
 pre{background:var(--card);padding:1rem;border-radius:10px;color:#c3c2b7;font-size:.8rem;
  max-height:180px;overflow:auto}
 .hint{font-size:.78rem;color:var(--muted);margin:.3rem 0 0}
</style></head><body>
<h1>Payments Real-Time Feature Store — Latency Walk</h1>
<p class="sub">A single payment authorization, walked through the real-time scoring
 architecture — with each component's latency measured live. <span id="backend"></span></p>

<div class="story">
 <b>What this shows.</b> Every transaction takes the same three-stage path from the
 architecture diagram, and the app times each hop so you can see <b>where the milliseconds
 go</b>:
 <br>①&nbsp;<b>Read</b> — pull the transaction off the Kafka topic (the event bus).
 &nbsp;→&nbsp; ②&nbsp;<b>Inference</b> — call the model serving endpoint; it looks up the
 online features itself through the <b>Feature Engineering API</b> (automatic feature lookup)
 and returns block/pass.
 &nbsp;→&nbsp; ③&nbsp;<b>Write back</b> — publish the decision to the outbound Kafka topic.
 <br><br>The feature store read is <b>inside</b> stage ② — the app never touches the online
 store directly; the endpoint joins features via the Feature Engineering API, exactly as a
 production scorer would.
 <br><br><b>Reading the gauges.</b> The big number is <b>p50</b> (typical latency); the small
 line is <b>p99</b> (tail). The first request is slower — the serving endpoint cold-starts —
 so watch the numbers settle after a few seconds of stream. <span id="transport-note"></span>
</div>

<div>
 <button class="primary" onclick="call('/generate?rate_per_sec=10','POST')">Start stream</button>
 <button onclick="call('/generate?stop=true','POST')">Stop</button>
 <button onclick="call('/score','POST')">Score one</button>
</div>
<p class="hint"><b>Start stream</b> runs a continuous flow of transactions; <b>Score one</b>
 walks a single transaction and prints its per-stage timings below.</p>

<div class="totals">
 <div>end-to-end p50 <b id="tp50">–</b> ms</div>
 <div>p99 <b id="tp99">–</b> ms</div>
 <div>scored <b id="scored">0</b></div>
 <div>blocked <b id="blocked">0</b></div>
 <div>stream <b id="run">off</b></div>
</div>
<div class="flow" id="flow"></div>
<pre id="log">Click "Score one" to walk a single transaction through the three stages — the response
shows the per-stage latency breakdown.</pre>
<script>
// [key, label, color, description]
const STAGES=[
 ["read","1 · Read","var(--s1)","Consume the transaction from the Kafka topic (the event bus)."],
 ["inference","2 · Inference","var(--s2)","Serving endpoint scores it, auto-joining online features via the Feature Engineering API."],
 ["write_back","3 · Write back","var(--s3)","Publish the block/pass decision to the outbound Kafka topic."],
];
const flow=document.getElementById('flow');
STAGES.forEach(([k,label,color,desc],i)=>{
 if(i>0){const a=document.createElement('div');a.className='arrow';a.textContent='→';flow.appendChild(a);}
 const g=document.createElement('div');g.className='gauge';g.id='g_'+k;
 g.innerHTML=`<div class="name">${label}</div>
  <div class="p50"><span id="p50_${k}">–</span><span class="unit"> ms</span></div>
  <div class="p99">p99 <span id="p99_${k}">–</span> ms</div>
  <div class="bar"><i id="bar_${k}" style="width:0%;background:${color}"></i></div>
  <div class="desc">${desc}</div>`;
 flow.appendChild(g);
});
async function call(url,method){
 const r=await fetch(url,{method:method||'GET',headers:{'Content-Type':'application/json'},
   body:method==='POST'?'{}':undefined});
 document.getElementById('log').textContent=JSON.stringify(await r.json(),null,2);
}
async function refresh(){
 const m=await (await fetch('/metrics')).json();
 const max=Math.max(1,...STAGES.map(([k])=>m.stages[k].p50||0));
 STAGES.forEach(([k])=>{
  const st=m.stages[k];
  document.getElementById('p50_'+k).textContent=st.p50??'–';
  document.getElementById('p99_'+k).textContent=st.p99??'–';
  document.getElementById('bar_'+k).style.width=((st.p50||0)/max*100)+'%';
 });
 document.getElementById('tp50').textContent=m.total.p50??'–';
 document.getElementById('tp99').textContent=m.total.p99??'–';
 document.getElementById('scored').textContent=m.scored;
 document.getElementById('blocked').textContent=m.blocked;
 document.getElementById('run').textContent=m.running?'on':'off';
}
fetch('/health').then(r=>r.json()).then(h=>{
 document.getElementById('backend').textContent='· transport: '+h.kafka_backend;
 const real = h.kafka_backend && h.kafka_backend.indexOf('kafka')>=0;
 document.getElementById('transport-note').innerHTML = real
  ? '<br><br><b>Transport:</b> a real Kafka broker is handling the Read/Write stages.'
  : '<br><br><b>Transport:</b> the Read/Write stages use an <b>in-process queue</b> — the '+
    'Databricks Apps container ships no Kafka binary. Stage ② (inference + feature lookup) is '+
    'always real. A real broker over the network is exercised by <code>09_kafka_io</code>.';
});
setInterval(refresh,1500); refresh();
</script></body></html>
"""
