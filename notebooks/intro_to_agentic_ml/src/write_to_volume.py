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

# DBTITLE 1,Stateful random-walk streaming engine
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


# =============================================================================
# STATEFUL RANDOM-WALK ENGINE
# Each tag carries a "current value" that evolves via mean-reverting random walk
# (Ornstein-Uhlenbeck process). This produces realistic, varying telemetry where
# consecutive readings are correlated but not identical — like real sensors.
# =============================================================================

def _init_tag_state(tags):
    """Initialize per-tag state with midpoint values."""
    state = {}
    for tag in tags:
        mid = (tag["normal_low"] + tag["normal_high"]) / 2.0
        state[tag["tag_id"]] = {
            "value": mid,
            "asset_id": tag["asset_id"],
            "lo": tag["normal_low"],
            "hi": tag["normal_high"],
            "band": tag["normal_high"] - tag["normal_low"],
            "noise": tag["noise_frac"],
            "drift": tag["drift_frac"],
            "mid": mid,
        }
    return state


def _step_random_walk(state, rng, anomaly_tags=None):
    """Advance every tag by one timestep using an OU random walk.
    Returns a list of (tag_id, asset_id, value, quality) tuples."""
    readings = []
    anomaly_tags = anomaly_tags or set()
    for tag_id, s in state.items():
        band = s["band"]
        # Mean-reversion strength (pulls back to midpoint)
        theta = 0.15
        # Volatility per step
        sigma = band * s["noise"] * 2.5
        # OU step: dx = theta*(mu - x)*dt + sigma*dW
        dx = theta * (s["mid"] - s["value"]) + sigma * rng.normal()
        # Additional slow drift component
        dx += band * s["drift"] * 0.3 * rng.normal()
        new_val = s["value"] + dx

        # Anomaly injection: persistent upward bias for chiller fault
        if tag_id in anomaly_tags:
            new_val += band * 0.4 * rng.uniform(0.8, 1.2)

        # Soft clip to physical bounds
        phys_lo = s["lo"] - band * 0.5
        phys_hi = s["hi"] + band * 2.5
        new_val = np.clip(new_val, phys_lo, phys_hi)
        s["value"] = new_val

        # Quality code (mostly Good, rare Uncertain/Bad)
        q = rng.random()
        quality = "Good" if q > 0.004 else ("Uncertain" if q > 0.001 else "Bad")
        readings.append((tag_id, s["asset_id"], round(new_val, 4), quality))
    return readings


def _to_sparkplug_records_rw(readings, ts):
    """Convert readings list to Sparkplug-B JSON records."""
    epoch_ms = int(ts.timestamp() * 1000)
    return [{
        "timestamp": epoch_ms,
        "edge_node": "plant-01",
        "name": tag_id,
        "alias": asset_id,
        "value": value,
        "quality": quality,
    } for tag_id, asset_id, value, quality in readings]


def backfill_and_stream(backfill_hours=24, anomaly_start_hours=18,
                        interval_seconds=20, anomaly_after_batches=5):
    """Two-phase operation:
    Phase 1 — BACKFILL: generate `backfill_hours` of history at 5-min cadence
              (e.g. 24h = 288 steps). The chiller anomaly ramps up in the last
              `backfill_hours - anomaly_start_hours` hours of backfill so the
              pipeline's gold layer has something to detect immediately.
    Phase 2 — STREAM:  continue with real-time random walk every
              `interval_seconds`, same state (no discontinuity).
    """
    rng = np.random.default_rng(42)
    state = _init_tag_state(_TAGS)
    anomaly_tag = "GLY.GLY01.RETURN_TEMP"
    total, b = 0, 0

    # --- Phase 1: Backfill ---
    n_backfill = int(backfill_hours * 60 / CADENCE_MIN)  # steps at 5-min cadence
    anomaly_start_step = int(anomaly_start_hours * 60 / CADENCE_MIN)
    now = datetime.utcnow()
    backfill_start = now - timedelta(hours=backfill_hours)

    print(f"=== PHASE 1: BACKFILL ({backfill_hours}h = {n_backfill} steps) ===")
    print(f"  Range: {backfill_start.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Chiller anomaly starts at step {anomaly_start_step} ({anomaly_start_hours}h in)")

    for step in range(n_backfill):
        ts = backfill_start + timedelta(minutes=step * CADENCE_MIN)
        inject_anomaly = step >= anomaly_start_step
        anomaly_tags = {anomaly_tag} if inject_anomaly else set()
        readings = _step_random_walk(state, rng, anomaly_tags=anomaly_tags)
        recs = _to_sparkplug_records_rw(readings, ts)

        path = f"{INCOMING}/batch_{b:05d}.json"
        payload = "\n".join(json.dumps(r) for r in recs)
        dbutils.fs.put(path, payload, overwrite=True)
        total += len(recs)
        b += 1

        if step % 50 == 0 or step == n_backfill - 1:
            chiller_val = state[anomaly_tag]["value"]
            flag = " [DRIFT]" if inject_anomaly else ""
            print(f"  step {step}/{n_backfill}: {ts.strftime('%H:%M')} "
                  f"| GLY.RETURN={chiller_val:.1f}°F{flag}")

    print(f"\n✓ Backfill complete: {total:,} records in {b} files")
    print(f"  Chiller drifted to {state[anomaly_tag]['value']:.1f}°F")

    # --- Phase 2: Stream forever ---
    print(f"\n=== PHASE 2: LIVE STREAMING (every {interval_seconds}s) ===")
    print(f"  Continuing from same state — no discontinuity")
    print(f"  Cancel the run to stop.\n")

    while True:
        ts = datetime.utcnow()
        # Keep anomaly going in live phase
        anomaly_tags = {anomaly_tag}
        readings = _step_random_walk(state, rng, anomaly_tags=anomaly_tags)
        recs = _to_sparkplug_records_rw(readings, ts)

        path = f"{INCOMING}/batch_{b:05d}.json"
        payload = "\n".join(json.dumps(r) for r in recs)
        dbutils.fs.put(path, payload, overwrite=True)
        total += len(recs)

        chiller_val = state[anomaly_tag]["value"]
        print(f"  batch {b}: {len(recs)} tags @ {ts.strftime('%H:%M:%S')} "
              f"| GLY.RETURN={chiller_val:.1f}°F [DRIFT]  (total {total:,})")
        b += 1
        _time.sleep(interval_seconds)


# COMMAND ----------

# DBTITLE 1,Backfill 24h then stream continuously (cancel to stop)
# Phase 1: backfill 24h of history (chiller anomaly starts at hour 18)
# Phase 2: stream live every 20s (anomaly persists)
# Cancel the run to stop.
backfill_and_stream(backfill_hours=24, anomaly_start_hours=18, interval_seconds=20)
