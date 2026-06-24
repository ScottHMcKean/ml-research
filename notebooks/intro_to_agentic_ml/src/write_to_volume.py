# Databricks notebook source
# write_to_volume.py — STREAMING JOB: brewery telemetry producer
# -----------------------------------------------------------------------------
# Run as the bundle's streaming job:  databricks bundle run ml_workshop_streaming
#
# Emits Sparkplug-B-style JSON micro-batches of brewery telemetry into the UC
# landing Volume, so Lab 2's Auto Loader stream has new files to pick up. Reuses
# the shared engine (brewery_generator) for the value model, so the live trickle is
# statistically identical to the batch history — including a LIVE injected
# chiller-drift anomaly window the lab's gold detector is meant to catch.
#
# Sparkplug-B mental model (simplified): an edge node publishes a DDATA payload
# containing a timestamp + an array of metrics. We emit one JSON object per
# reading (name = tag_id, alias = asset_id) — the shape an MQTT->ADLS bridge would
# land. Auto Loader infers this schema directly.
# -----------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %run ./brewery_generator

# COMMAND ----------

import json
import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Resolve VOLUME_PATH if this module is run standalone (lab already has it via 00-setup).
try:
    VOLUME_PATH
except NameError:
    VOLUME_PATH = "/Volumes/ml_workshop/brewery/landing"

INCOMING = f"{VOLUME_PATH}/incoming"

try:
    dbutils.fs.mkdirs(INCOMING)
except Exception:
    pass


def _readings_for_window(window_start, window_end, live_anomaly=False):
    """Build a long-form DataFrame of (ts, tag_id, asset_id, value, quality) for
    every tag across a short window at the 5-min cadence."""
    idx = build_timestamp_index(window_start, window_end)
    if len(idx) == 0:
        return pd.DataFrame(columns=["reading_ts", "tag_id", "asset_id", "value", "quality_code"])
    ambient = _az_ambient_f(idx)
    frames = []
    for tag in _TAGS:
        vals, qual = generate_tag_series(tag, idx, ambient_f=ambient)
        if live_anomaly and tag["tag_id"] == "GLY.GLY01.RETURN_TEMP":
            # Inject a visible chiller-drift ramp across THIS live window so the
            # lab's detector has a fresh anomaly that isn't in the batch labels.
            ramp = np.linspace(0, 12.0, len(idx))
            vals = vals + ramp
        frames.append(pd.DataFrame({
            "reading_ts": idx.to_pydatetime(),
            "tag_id": tag["tag_id"], "asset_id": tag["asset_id"],
            "value": np.round(vals, 4), "quality_code": qual,
        }))
    return pd.concat(frames, ignore_index=True)


def _to_sparkplug_records(pdf):
    """One JSON record per reading, Sparkplug-B DDATA-metric shaped."""
    recs = []
    for r in pdf.itertuples():
        recs.append({
            "timestamp": int(r.reading_ts.timestamp() * 1000),  # epoch ms
            "edge_node": "plant-01",
            "name": r.tag_id,          # Sparkplug metric name == historian tag
            "alias": r.asset_id,
            "value": float(r.value),
            "quality": r.quality_code,  # OPC quality folded into the payload
        })
    return recs


def emit_batch(window_minutes=60, batch_label=None, live_anomaly=False, anchor=None):
    """Generate one JSON file covering the last `window_minutes` and drop it on
    the landing volume. Returns (path, n_records)."""
    anchor = anchor or datetime(2026, 6, 22, 0, 0)  # deterministic 'now' = end of history
    start = anchor - timedelta(minutes=window_minutes)
    pdf = _readings_for_window(start, anchor, live_anomaly=live_anomaly)
    recs = _to_sparkplug_records(pdf)
    label = batch_label or anchor.strftime("%Y%m%dT%H%M%S")
    path = f"{INCOMING}/batch_{label}.json"
    payload = "\n".join(json.dumps(r) for r in recs)
    dbutils.fs.put(path, payload, overwrite=True)
    return path, len(recs)


def trickle(n_batches=6, window_minutes=30, interval_seconds=10, live_anomaly_from=2):
    """Emit a sequence of micro-batches walking forward in time, so an active
    Auto Loader stream sees new files arrive. `live_anomaly_from` = batch index
    at which the chiller-drift ramp turns on."""
    anchor = datetime(2026, 6, 22, 0, 0)
    total = 0
    for b in range(n_batches):
        win_start = anchor + timedelta(minutes=window_minutes * b)
        win_end = win_start + timedelta(minutes=window_minutes)
        pdf = _readings_for_window(win_start, win_end, live_anomaly=(b >= live_anomaly_from))
        recs = _to_sparkplug_records(pdf)
        path = f"{INCOMING}/batch_{b:03d}.json"
        dbutils.fs.put(path, "\n".join(json.dumps(r) for r in recs), overwrite=True)
        total += len(recs)
        print(f"  batch {b}: {len(recs)} records -> {path}"
              f"{'  [LIVE ANOMALY]' if b >= live_anomaly_from else ''}")
        if b < n_batches - 1:
            _time.sleep(interval_seconds)
    print(f"Trickle complete: {total:,} records across {n_batches} files in {INCOMING}")
    return total


def stream_forever(window_minutes=5, interval_seconds=20, anomaly_after_batches=5):
    """Continuously emit micro-batches anchored to REAL wall-clock time — so
    readings are always recent (never in the future). The chiller-drift anomaly
    turns on after `anomaly_after_batches` batches. Each iteration emits one
    JSON file covering (now - window_minutes, now], then sleeps."""
    b, total = 0, 0
    print(f"Streaming OT files to {INCOMING} every {interval_seconds}s "
          f"(anomaly starts at batch {anomaly_after_batches}). Cancel the run to stop.")
    while True:
        win_end = datetime.utcnow()
        win_start = win_end - timedelta(minutes=window_minutes)
        live = b >= anomaly_after_batches
        pdf = _readings_for_window(win_start, win_end, live_anomaly=live)
        recs = _to_sparkplug_records(pdf)
        path = f"{INCOMING}/batch_{b:05d}.json"
        dbutils.fs.put(path, "\n".join(json.dumps(r) for r in recs), overwrite=True)
        total += len(recs)
        print(f"  batch {b}: {len(recs)} records -> {path}"
              f"{'  [LIVE ANOMALY]' if live else ''}  (total {total:,})")
        b += 1
        _time.sleep(interval_seconds)


# COMMAND ----------

# DBTITLE 1,Stream continuously — this is what the streaming job runs (cancel to stop)
# Drops a new Sparkplug-B file every 20s; the chiller-drift anomaly turns on at batch 5.
# Runs until the job is cancelled (or the job's timeout is hit).
stream_forever(window_minutes=5, interval_seconds=20, anomaly_after_batches=5)
