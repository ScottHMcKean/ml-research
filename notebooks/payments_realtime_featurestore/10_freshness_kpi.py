# Databricks notebook source

# MAGIC %md
# MAGIC # 10 · Feature freshness KPI (P10/P50/P90/P99)
# MAGIC
# MAGIC Validates that the online feature store stays **fresh** under a sub-second event rate,
# MAGIC end to end. It measures the clock that matters for real-time serving:
# MAGIC
# MAGIC ```
# MAGIC freshness = (moment the aggregated feature is READABLE in Lakebase) − (event_ts of the
# MAGIC             newest event folded into it)
# MAGIC ```
# MAGIC
# MAGIC and reports the P10 / P50 / P90 / P99 of that freshness across many samples.
# MAGIC
# MAGIC ## Pipeline
# MAGIC ```
# MAGIC sub-second batch generator → Delta raw_stream → running per-key aggregation
# MAGIC   → upsert to Lakebase online table → read back to timestamp visibility → freshness
# MAGIC ```
# MAGIC
# MAGIC - **Generator** — a driver-side loop that appends a **sub-second micro-batch** of rows to
# MAGIC   a Delta table every `tick_seconds`, each row stamped with an `event_ts` at generation.
# MAGIC   (The Spark **rate source** is the native generator, but a *background* `writeStream`
# MAGIC   running alongside a driver polling loop is restricted on serverless Spark Connect, so
# MAGIC   this uses a batch-append generator — the same "sub-second batches into Delta, as a
# MAGIC   job" shape, without a background streaming query. `dbldatagen` wraps the rate source
# MAGIC   the same way for load tests.)
# MAGIC - **Aggregation → online** — the same loop reads the newest events, computes a per-key
# MAGIC   running count, and upserts each key into the Lakebase online table. Every Feature
# MAGIC   Engineering / Lakebase call stays on the driver (serverless Spark Connect forbids them
# MAGIC   inside a streaming `foreachBatch` worker — see `07`).
# MAGIC - **KPI** — freshness per sample = online-write time − max(event_ts) in that batch; we
# MAGIC   also read the row back from Lakebase to include the read-visibility hop.

# COMMAND ----------

# MAGIC %pip install --quiet psycopg[binary] numpy
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

dbutils.widgets.text("rows_per_second", "200", "Rate-source event rate (rows/sec)")
dbutils.widgets.text("duration_seconds", "120", "How long to run the harness (seconds)")
dbutils.widgets.text("tick_seconds", "1", "Aggregation micro-batch interval (seconds)")
ROWS_PER_SECOND = int(dbutils.widgets.get("rows_per_second"))
DURATION = int(dbutils.widgets.get("duration_seconds"))
TICK = float(dbutils.widgets.get("tick_seconds"))

RAW_STREAM = f"{CATALOG}.{SCHEMA}.freshness_raw_stream"

import time, uuid
import numpy as np
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lakebase (online) connection
# MAGIC Connects to the online-store Postgres and owns a small feature table for the KPI. Uses
# MAGIC the injected `PGHOST` when present (Apps) or the SDK/REST here in a notebook; the OAuth
# MAGIC token is the password.

# COMMAND ----------

import psycopg
from databricks.sdk import WorkspaceClient

wc = WorkspaceClient()
LB_USER = wc.current_user.me().user_name


def _lb_host() -> str:
    # Older serverless/Apps SDKs lack wc.database; hit the REST API directly.
    try:
        return wc.database.get_database_instance(name=ONLINE_STORE).read_write_dns
    except AttributeError:
        return wc.api_client.do("GET", f"/api/2.0/database/instances/{ONLINE_STORE}")["read_write_dns"]


def _lb_token() -> str:
    try:
        return wc.database.generate_database_credential(
            request_id=str(uuid.uuid4()), instance_names=[ONLINE_STORE]).token
    except AttributeError:
        return wc.api_client.do("POST", "/api/2.0/database/credentials",
                                body={"request_id": str(uuid.uuid4()),
                                      "instance_names": [ONLINE_STORE]})["token"]


LB_HOST = _lb_host()


def lb_connect():
    conn = psycopg.connect(host=LB_HOST, port=5432, dbname="databricks_postgres",
                           user=LB_USER, password=_lb_token(), sslmode="require", autocommit=True)
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS freshness_demo")
        cur.execute("CREATE TABLE IF NOT EXISTS freshness_demo.feature ("
                    "key bigint PRIMARY KEY, cnt bigint, "
                    "event_ts timestamptz, online_ts timestamptz)")
    return conn


pg = lb_connect()
print(f"Lakebase online store ready: {LB_HOST}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate + aggregate + sample freshness, each tick
# MAGIC One driver-side loop does all three, `tick_seconds` apart:
# MAGIC 1. **Generate** — append a sub-second micro-batch of `rows_per_second * tick_seconds`
# MAGIC    rows to the Delta table, each stamped with `event_ts = now()` at generation and
# MAGIC    bucketed to a small key space so the aggregation has repeat keys.
# MAGIC 2. **Aggregate** — read the just-written batch, add to per-key running counts.
# MAGIC 3. **Publish + sample** — upsert each key into Lakebase (timing the online write), read
# MAGIC    one key back, and record freshness = online time − max(event_ts) in the batch.

# COMMAND ----------

import datetime as dt
from pyspark.sql.types import StructType, StructField, LongType, TimestampType

spark.sql(f"DROP TABLE IF EXISTS {RAW_STREAM}")
spark.sql(f"CREATE TABLE {RAW_STREAM} (key BIGINT, value BIGINT, event_ts TIMESTAMP) USING delta")

SCHEMA_RAW = StructType([
    StructField("key", LongType()), StructField("value", LongType()),
    StructField("event_ts", TimestampType()),
])
BATCH_ROWS = max(1, int(ROWS_PER_SECOND * TICK))
NUM_KEYS = 50


def generate_batch(start_value: int) -> tuple:
    """Append BATCH_ROWS rows to Delta, each stamped now(); returns (rows, max_value)."""
    now = dt.datetime.utcnow()
    rows = [(int((start_value + i) % NUM_KEYS), int(start_value + i), now)
            for i in range(BATCH_ROWS)]
    spark.createDataFrame(rows, SCHEMA_RAW).write.mode("append").saveAsTable(RAW_STREAM)
    return rows, start_value + BATCH_ROWS


freshness_write_ms = []   # online write time - event_ts
freshness_read_ms = []    # read-back visible time - event_ts
running = {}              # key -> cumulative count
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
    keys_in_batch = sorted({k for k, _v, _ts in rows})

    # 3 · publish to Lakebase + sample freshness.
    with pg.cursor() as cur:
        for k in keys_in_batch:
            cur.execute(
                "INSERT INTO freshness_demo.feature (key, cnt, event_ts, online_ts) "
                "VALUES (%s,%s,%s, now()) "
                "ON CONFLICT (key) DO UPDATE SET cnt=EXCLUDED.cnt, "
                "event_ts=EXCLUDED.event_ts, online_ts=now()",
                (int(k), int(running[k]), max_event_ts),
            )
        cur.execute("SELECT online_ts FROM freshness_demo.feature WHERE key=%s",
                    (int(keys_in_batch[0]),))
        online_ts = cur.fetchone()[0]
    read_now = dt.datetime.now(dt.timezone.utc)
    ev = max_event_ts.replace(tzinfo=dt.timezone.utc)   # event_ts is naive UTC
    freshness_write_ms.append((online_ts - ev).total_seconds() * 1000.0)
    freshness_read_ms.append((read_now - ev).total_seconds() * 1000.0)

    time.sleep(max(0.0, TICK - (time.time() - tick_start)))

print(f"Collected {len(freshness_write_ms)} freshness samples.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Freshness KPI — P10 / P50 / P90 / P99

# COMMAND ----------

def pctls(vals):
    a = np.array(vals)
    return {p: round(float(np.percentile(a, p)), 1) for p in (10, 50, 90, 99)}


write_kpi = pctls(freshness_write_ms)
read_kpi = pctls(freshness_read_ms)

print("Feature freshness (ms) — event_ts -> online write:")
for p, v in write_kpi.items():
    print(f"  P{p}: {v} ms")
print("\nFeature freshness (ms) — event_ts -> online read-visible (end-to-end):")
for p, v in read_kpi.items():
    print(f"  P{p}: {v} ms")

# Persist the KPI so it can be tracked over time / across configs.
import datetime as dt
kpi_row = spark.createDataFrame([{
    "run_ts": dt.datetime.utcnow(),
    "rows_per_second": ROWS_PER_SECOND,
    "tick_seconds": TICK,
    "samples": len(freshness_write_ms),
    "write_p10": write_kpi[10], "write_p50": write_kpi[50],
    "write_p90": write_kpi[90], "write_p99": write_kpi[99],
    "read_p10": read_kpi[10], "read_p50": read_kpi[50],
    "read_p90": read_kpi[90], "read_p99": read_kpi[99],
}])
kpi_row.write.mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.freshness_kpi")
print(f"\nAppended KPI row to {CATALOG}.{SCHEMA}.freshness_kpi")

pg.close()

# Surface the KPI in the run output too (P50 tracks the tick interval; lower `tick_seconds`
# to tighten freshness, raise `rows_per_second` to stress it).
summary = {"samples": len(freshness_write_ms), "rows_per_second": ROWS_PER_SECOND,
           "tick_seconds": TICK, "write_ms": write_kpi, "read_ms": read_kpi}
dbutils.notebook.exit(str(summary))
