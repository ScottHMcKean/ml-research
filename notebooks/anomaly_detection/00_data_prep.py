# Databricks notebook source

# MAGIC %md
# MAGIC # Rare-Failure Anomaly Detection — Data Preparation
# MAGIC Generates a **physics-inspired, multi-sensor rotating-equipment (industrial pump)**
# MAGIC dataset with a handful of **rare failure events** and writes it to
# MAGIC `shm_skunkworks_catalog.anomaly_detection.pump_sensors`.
# MAGIC
# MAGIC ## The problem this demo is about
# MAGIC In real predictive-maintenance settings you have **years of mostly-normal sensor
# MAGIC data and only one or two confirmed failures**. That is far too few to train a
# MAGIC supervised classifier. The winning pattern is **unsupervised anomaly detection**
# MAGIC (learn what "normal" looks like, flag deviations) **augmented with the couple of
# MAGIC observed failures** — used not for training but to *calibrate the decision
# MAGIC threshold*. That is exactly what the `01_pca` and `02_autoencoder` notebooks do.
# MAGIC
# MAGIC ## What we generate
# MAGIC A single pump line sampled hourly for ~2.3 years (20,000 readings) across **12
# MAGIC correlated sensors**. Normal operation follows a simplified pump/motor physics
# MAGIC model driven by a varying load (duty cycle). We then inject a small number of
# MAGIC short **failure episodes** across five realistic failure modes. The overall
# MAGIC failure rate is ~0.5%, and **only 2 of those failure rows are "observed"**
# MAGIC (`is_labeled = 1`) — the few-shot signal the models are allowed to see.
# MAGIC
# MAGIC > `is_failure` / `failure_mode` are the **hidden ground truth**, used *only* for
# MAGIC > final evaluation. In production you would not have them. `is_labeled` is the
# MAGIC > only supervision the detectors are permitted to use.
# MAGIC
# MAGIC ### Swapping in a real dataset
# MAGIC To run this on real data, replace this notebook with a loader for a public
# MAGIC predictive-maintenance dataset with rare failures and multiple sensors, e.g.
# MAGIC **MetroPT-3** (metro train air-compressor failures) or the **NASA IMS bearing**
# MAGIC run-to-failure set. Keep the same output schema (sensor columns + `is_failure`,
# MAGIC `failure_mode`, `is_labeled`) and the downstream notebooks work unchanged.

# COMMAND ----------

import numpy as np
import pandas as pd

CATALOG = "shm_skunkworks_catalog"
SCHEMA = "anomaly_detection"
TABLE = f"{CATALOG}.{SCHEMA}.pump_sensors"

N_ROWS = 20_000
N_LABELED = 2          # how many failure rows are "observed" (the few-shot signal)
SEED = 42
rng = np.random.default_rng(SEED)

# The 12 sensors the detectors see. Kept in one place — the PCA and AutoEncoder
# notebooks import this exact list.
SENSORS = [
    "vibration_rms",        # mm/s, overall vibration energy
    "vibration_peak",       # mm/s, peak / crest
    "bearing_temp",         # deg C
    "motor_temp",           # deg C
    "motor_current",        # A
    "motor_voltage",        # V
    "flow_rate",            # m3/h
    "discharge_pressure",   # bar
    "suction_pressure",     # bar
    "rpm",                  # rev/min
    "oil_particle_count",   # ISO particle count proxy
    "acoustic_db",          # dB
]

# Catalog is assumed to exist (FEVM metastores don't grant CREATE CATALOG). We only
# create the schema, which the catalog owner is allowed to do.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Normal operation
# MAGIC A varying **load factor** (duty cycle, 0.6–1.0 with a slow drift + daily cycle)
# MAGIC drives every sensor through a simplified pump/motor model. Sensors are correlated
# MAGIC *through the shared load* — which is precisely the structure PCA and an
# MAGIC autoencoder learn to compress.

# COMMAND ----------

t = np.arange(N_ROWS)

# Load: slow random-walk drift + daily cycle + noise, clipped to a realistic band.
daily = 0.06 * np.sin(2 * np.pi * t / 24.0)
drift = np.cumsum(rng.normal(0, 0.004, N_ROWS))
drift = 0.15 * (drift - drift.mean()) / (drift.std() + 1e-9)
load = np.clip(0.82 + daily + drift, 0.6, 1.0)

def n(scale):
    return rng.normal(0, scale, N_ROWS)

df = pd.DataFrame({
    "rpm":                1488 + 6 * load + n(1.2),
    "flow_rate":          40 + 95 * load + n(2.0),
})
df["discharge_pressure"] = 9.2 - 0.018 * df["flow_rate"] + n(0.08)          # pump curve
df["suction_pressure"]   = 1.25 + 0.05 * load + n(0.03)
df["motor_current"]      = 18 + 34 * load + n(0.8)
df["motor_voltage"]      = 415 + n(1.5)
df["motor_temp"]         = 32 + 0.55 * df["motor_current"] + n(1.0)
df["bearing_temp"]       = 40 + 14 * load + n(1.2)
df["vibration_rms"]      = 1.6 + 0.5 * load + n(0.10)
df["vibration_peak"]     = df["vibration_rms"] * (1.5 + n(0.03)) + n(0.05)
df["oil_particle_count"] = rng.poisson(45 + 15 * load)
df["acoustic_db"]        = 70 + 5 * load + n(0.6)

df["is_failure"] = 0
df["failure_mode"] = "normal"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inject rare failure episodes
# MAGIC Five realistic modes, each a short episode (a few consecutive hours) that pushes
# MAGIC a *specific subset* of sensors off their normal manifold with a ramping severity.
# MAGIC This is what makes the problem multivariate: no single sensor threshold catches
# MAGIC every mode, but the *joint* deviation does.

# COMMAND ----------

def episode(mode, start, length):
    """Apply a ramping fault signature to rows [start, start+length)."""
    idx = np.arange(start, start + length)
    sev = np.linspace(0.3, 1.0, length)          # severity ramps up over the episode
    d = df.loc[idx]

    if mode == "bearing_wear":
        df.loc[idx, "bearing_temp"]       = d["bearing_temp"] + 22 * sev
        df.loc[idx, "vibration_rms"]      = d["vibration_rms"] + 2.8 * sev
        df.loc[idx, "vibration_peak"]     = d["vibration_peak"] + 6.0 * sev
        df.loc[idx, "oil_particle_count"] = d["oil_particle_count"] + (260 * sev).astype(int)
        df.loc[idx, "acoustic_db"]        = d["acoustic_db"] + 8 * sev
    elif mode == "cavitation":
        df.loc[idx, "suction_pressure"]   = d["suction_pressure"] - 0.7 * sev
        df.loc[idx, "flow_rate"]          = d["flow_rate"] - 18 * sev
        df.loc[idx, "discharge_pressure"] = d["discharge_pressure"] - 1.2 * sev + rng.normal(0, 0.25, length)
        df.loc[idx, "vibration_rms"]      = d["vibration_rms"] + 1.6 * sev
        df.loc[idx, "acoustic_db"]        = d["acoustic_db"] + 12 * sev
    elif mode == "impeller_imbalance":
        df.loc[idx, "vibration_rms"]      = d["vibration_rms"] + 3.4 * sev
        df.loc[idx, "vibration_peak"]     = d["vibration_peak"] + 8.5 * sev
        df.loc[idx, "discharge_pressure"] = d["discharge_pressure"] + rng.normal(0, 0.4, length) * sev
        df.loc[idx, "acoustic_db"]        = d["acoustic_db"] + 5 * sev
    elif mode == "seal_leak":
        df.loc[idx, "suction_pressure"]   = d["suction_pressure"] - 0.4 * sev
        df.loc[idx, "discharge_pressure"] = d["discharge_pressure"] - 0.9 * sev
        df.loc[idx, "flow_rate"]          = d["flow_rate"] - 12 * sev
        df.loc[idx, "oil_particle_count"] = d["oil_particle_count"] + (140 * sev).astype(int)
    elif mode == "motor_winding":
        df.loc[idx, "motor_current"]      = d["motor_current"] + 16 * sev
        df.loc[idx, "motor_temp"]         = d["motor_temp"] + 28 * sev
        df.loc[idx, "motor_voltage"]      = d["motor_voltage"] - 8 * sev
        df.loc[idx, "vibration_rms"]      = d["vibration_rms"] + 0.9 * sev

    df.loc[idx, "is_failure"] = 1
    df.loc[idx, "failure_mode"] = mode


# Episodes are placed well apart across the 20k-hour window. ~0.5% failure rate.
EPISODES = [
    ("bearing_wear",       2_400, 14),
    ("cavitation",         5_100, 10),
    ("impeller_imbalance", 8_300, 12),
    ("seal_leak",         11_700, 16),
    ("motor_winding",     14_900, 10),
    ("bearing_wear",      17_600, 13),
    ("cavitation",        19_100,  9),
]
for mode, start, length in EPISODES:
    episode(mode, start, length)

# Re-derive vibration_peak floor so it never falls below rms after injection
df["vibration_peak"] = np.maximum(df["vibration_peak"], df["vibration_rms"] * 1.05)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reveal only a couple of observed failures (the few-shot signal)
# MAGIC Of the ~80 failure rows, we mark just **`N_LABELED` = 2** as "observed" — the
# MAGIC peak-severity row from two *different* failure modes, mimicking a
# MAGIC maintenance team that has confirmed a small handful of past failures. Everything
# MAGIC else is unlabeled as far as the detectors are concerned.

# COMMAND ----------

df["is_labeled"] = 0

# Pick the peak-severity row of two distinct modes so the labels are informative.
# Severity ramps 0.3 -> 1.0 across an episode, so the *last* row of an episode is the
# peak. A mode can appear in more than one episode, so we take the last row of its
# first contiguous block (where reading_id is consecutive).
labeled_rows = []
for mode in ["bearing_wear", "cavitation"]:
    mode_idx = df.index[df["failure_mode"] == mode].to_numpy()
    first_break = np.where(np.diff(mode_idx) > 1)[0]     # end of the first episode
    end_of_first = mode_idx[first_break[0]] if len(first_break) else mode_idx[-1]
    labeled_rows.append(int(end_of_first))               # peak-severity row
labeled_rows = labeled_rows[:N_LABELED]
df.loc[labeled_rows, "is_labeled"] = 1

ts0 = pd.Timestamp("2024-01-01")
df.insert(0, "reading_id", t)
df.insert(1, "event_time", ts0 + pd.to_timedelta(t, unit="h"))
df.insert(2, "unit_id", "PUMP-A1")

print(f"rows={len(df)}  failures={int(df.is_failure.sum())} "
      f"({df.is_failure.mean()*100:.2f}%)  labeled={int(df.is_labeled.sum())}")
print("failure mode counts:\n", df.loc[df.is_failure == 1, "failure_mode"].value_counts())
print("labeled rows:", labeled_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write to Unity Catalog

# COMMAND ----------

sdf = spark.createDataFrame(df)
sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE)

print(f"Wrote {sdf.count()} rows to {TABLE}")
display(spark.sql(f"SELECT * FROM {TABLE} WHERE is_failure = 1 ORDER BY reading_id LIMIT 20"))
