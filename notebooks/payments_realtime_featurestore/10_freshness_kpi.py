# Databricks notebook source

# MAGIC %md
# MAGIC # 10 · Feature freshness KPI (P10/P50/P90/P99)
# MAGIC
# MAGIC Validates that the online feature store stays **fresh** under a sub-second event rate,
# MAGIC measured **entirely through the Feature Engineering API** (`fe.write_table` +
# MAGIC `fe.publish_table`) — the same path the rest of the demo and the serving endpoint use.
# MAGIC No direct Lakebase/Postgres access.
# MAGIC
# MAGIC ```
# MAGIC freshness = (moment fe.publish_table makes the feature online-readable)
# MAGIC             − (event_ts of the newest event folded into it)
# MAGIC ```
# MAGIC
# MAGIC and reports the P10 / P50 / P90 / P99 of that freshness across many samples.
# MAGIC
# MAGIC ## What the number tells you (important)
# MAGIC `fe.publish_table(TRIGGERED)` provisions and runs a sync each call, which adds **seconds
# MAGIC of fixed overhead per publish** — on the FEVM at 200 rows/s, 1-s tick, freshness measured
# MAGIC **P50 ≈ 11.6 s** (only a handful of publishes complete in 60 s). That is the honest cost
# MAGIC of the **triggered-publish API path**, and it is the right number for a *batch/triggered*
# MAGIC cadence. For genuinely low-latency freshness (sub-second to low-seconds), use a
# MAGIC **continuous** publish/stream so the sync is always-on rather than spun up per tick — the
# MAGIC same batch-vs-continuous tradeoff called out for the 30-second counter path in `07`. This
# MAGIC harness makes that cost measurable: point it at your candidate publish mode and read the
# MAGIC percentiles.
# MAGIC
# MAGIC ## Pipeline (pure Feature Engineering API)
# MAGIC ```
# MAGIC sub-second batch generator → Delta raw_stream → per-key running aggregation
# MAGIC   → fe.write_table (offline feature table) → fe.publish_table TRIGGERED (online sync)
# MAGIC   → freshness = publish-return time − max(event_ts)
# MAGIC ```
# MAGIC
# MAGIC - **Generator** — a driver-side loop appends a **sub-second micro-batch** of events to a
# MAGIC   Delta table every `tick_seconds`, each stamped with `event_ts` at generation. (The
# MAGIC   Spark **rate source** is the native generator, but a *background* `writeStream`
# MAGIC   alongside a driver polling loop is restricted on serverless Spark Connect, so this uses
# MAGIC   a batch-append generator — the same "sub-second batches into Delta, as a job" shape.
# MAGIC   `dbldatagen` wraps the rate source the same way for load tests.)
# MAGIC - **Aggregate → feature store** — the loop computes per-key running counts and calls
# MAGIC   `fe.write_table(mode="merge")` then `fe.publish_table(publish_mode="TRIGGERED")`.
# MAGIC   `publish_table` blocks until the online sync completes, so its return marks the moment
# MAGIC   the feature is online-readable — that is the freshness clock, via the API alone.
# MAGIC - Every FE call stays on the driver (serverless Spark Connect forbids them inside a
# MAGIC   streaming `foreachBatch` worker — see `07`).

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering numpy
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

dbutils.widgets.text("rows_per_second", "200", "Event rate (rows/sec)")
dbutils.widgets.text("duration_seconds", "120", "How long to run the harness (seconds)")
dbutils.widgets.text("tick_seconds", "1", "Micro-batch / publish interval (seconds)")
ROWS_PER_SECOND = int(dbutils.widgets.get("rows_per_second"))
DURATION = int(dbutils.widgets.get("duration_seconds"))
TICK = float(dbutils.widgets.get("tick_seconds"))

RAW_STREAM = f"{CATALOG}.{SCHEMA}.freshness_raw_stream"
FT_FRESHNESS = f"{CATALOG}.{SCHEMA}.ft_freshness"

import time
import datetime as dt
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType, TimestampType
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the feature table + online store publication
# MAGIC A single-key running-count feature, published to the same Lakebase-backed online store
# MAGIC the rest of the demo uses — created through the Feature Engineering API.

# COMMAND ----------

SCHEMA_RAW = StructType([
    StructField("key", LongType()), StructField("value", LongType()),
    StructField("event_ts", TimestampType()),
])
NUM_KEYS = 50
BATCH_ROWS = max(1, int(ROWS_PER_SECOND * TICK))

spark.sql(f"DROP TABLE IF EXISTS {RAW_STREAM}")
spark.sql(f"CREATE TABLE {RAW_STREAM} (key BIGINT, value BIGINT, event_ts TIMESTAMP) USING delta")

# Seed the feature table over the full key universe so create_table has a schema + all keys.
seed = spark.createDataFrame(
    [(k, 0, dt.datetime.utcnow()) for k in range(NUM_KEYS)],
    StructType([StructField("key", LongType()), StructField("cnt", LongType()),
                StructField("feature_ts", TimestampType())]),
)
spark.sql(f"DROP TABLE IF EXISTS {FT_FRESHNESS}")
fe.create_table(
    name=FT_FRESHNESS,
    primary_keys=["key"],
    df=seed,
    description="Freshness-KPI running counter (per key), materialized via the FE API.",
)
spark.sql(f"ALTER TABLE {FT_FRESHNESS} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

store = fe.get_online_store(name=ONLINE_STORE)
FT_ONLINE = f"{FT_FRESHNESS}_online"
fe.publish_table(online_store=store, source_table_name=FT_FRESHNESS,
                 online_table_name=FT_ONLINE, publish_mode="TRIGGERED")
print(f"Feature table + online publication ready: {FT_FRESHNESS} -> {FT_ONLINE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate + aggregate + publish, sampling freshness each tick

# COMMAND ----------

def generate_batch(start_value: int) -> tuple:
    """Append BATCH_ROWS rows to Delta, each stamped now(); returns (rows, next_value)."""
    now = dt.datetime.utcnow()
    rows = [(int((start_value + i) % NUM_KEYS), int(start_value + i), now)
            for i in range(BATCH_ROWS)]
    spark.createDataFrame(rows, SCHEMA_RAW).write.mode("append").saveAsTable(RAW_STREAM)
    return rows, start_value + BATCH_ROWS


freshness_ms = []   # publish-return time − max(event_ts)
running = {}
next_value = 0

deadline = time.time() + DURATION
while time.time() < deadline:
    tick_start = time.time()

    # 1 · generate a sub-second micro-batch into Delta (event_ts = generation time).
    rows, next_value = generate_batch(next_value)
    max_event_ts = max(r[2] for r in rows)

    # 2 · aggregate: per-key running counts over this batch.
    for k, _v, _ts in rows:
        running[k] = running.get(k, 0) + 1
    feat = spark.createDataFrame(
        [(int(k), int(c), max_event_ts) for k, c in running.items()],
        StructType([StructField("key", LongType()), StructField("cnt", LongType()),
                    StructField("feature_ts", TimestampType())]),
    )

    # 3 · materialize + publish through the FE API; publish_table blocks until online-readable.
    fe.write_table(name=FT_FRESHNESS, df=feat, mode="merge")
    fe.publish_table(online_store=store, source_table_name=FT_FRESHNESS,
                     online_table_name=FT_ONLINE, publish_mode="TRIGGERED")
    online_ready = dt.datetime.utcnow()

    freshness_ms.append((online_ready - max_event_ts).total_seconds() * 1000.0)
    time.sleep(max(0.0, TICK - (time.time() - tick_start)))

print(f"Collected {len(freshness_ms)} freshness samples.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Freshness KPI — P10 / P50 / P90 / P99

# COMMAND ----------

def pctls(vals):
    a = np.array(vals)
    return {p: round(float(np.percentile(a, p)), 1) for p in (10, 50, 90, 99)}


kpi = pctls(freshness_ms)
print("Feature freshness (ms) — event_ts -> online-readable via fe.publish_table:")
for p, v in kpi.items():
    print(f"  P{p}: {v} ms")

kpi_row = spark.createDataFrame([{
    "run_ts": dt.datetime.utcnow(),
    "rows_per_second": ROWS_PER_SECOND,
    "tick_seconds": TICK,
    "samples": len(freshness_ms),
    "p10": kpi[10], "p50": kpi[50], "p90": kpi[90], "p99": kpi[99],
}])
kpi_row.write.mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.freshness_kpi")
print(f"\nAppended KPI row to {CATALOG}.{SCHEMA}.freshness_kpi")

dbutils.notebook.exit(str({"samples": len(freshness_ms), "rows_per_second": ROWS_PER_SECOND,
                           "tick_seconds": TICK, "freshness_ms": kpi}))
