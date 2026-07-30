# Databricks notebook source

# MAGIC %md
# MAGIC # 07 · Counter refresh (hot path)
# MAGIC
# MAGIC Keeps the per-instrument 1h sliding-window counters fresh in the Lakebase online store.
# MAGIC Each refresh recomputes the last-hour counters over the **full instrument universe** —
# MAGIC so instruments idle in the last hour are reset to 0 rather than keeping stale values —
# MAGIC merges them into the counter feature table, and re-publishes to Lakebase.
# MAGIC
# MAGIC One `mode` widget selects the cadence:
# MAGIC
# MAGIC | `mode` | Cadence | Freshness | Deploy as | Use when |
# MAGIC |--------|---------|-----------|-----------|----------|
# MAGIC | `batch` | one refresh per run | driven by the job schedule (e.g. 5-min cron) | scheduled job | minutes of staleness are acceptable (cheapest) |
# MAGIC | `loop`  | driver-side loop every `trigger_seconds` | ~30–60 s | continuous / always-running job | burst-detection features needing ~30 s freshness |
# MAGIC
# MAGIC ## Why a driver loop, not streaming `foreachBatch`
# MAGIC The recompute resets idle instruments to 0 by recomputing over the full universe, and
# MAGIC `fe.write_table` / `fe.publish_table` are **driver-side** APIs that need the notebook's
# MAGIC auth. On serverless (Spark Connect) a streaming `foreachBatch` closure runs in an
# MAGIC isolated worker with no Spark session and no default credentials, so the Feature
# MAGIC Engineering client cannot run there. A plain batch recompute (for `batch`) or a driver
# MAGIC loop (for `loop`) keeps every FE call on the driver where auth works.
# MAGIC
# MAGIC A truly incremental streaming aggregation belongs with the Kafka source: the stream
# MAGIC writes a Delta feature table and Lakebase **`CONTINUOUS`** publish auto-syncs it, with no
# MAGIC FE client in the hot loop. See `09_kafka_io`. Real-Time Mode (a ~5 ms floor, dedicated
# MAGIC compute slots) is only needed when a *feature* or the *per-transaction decision* must be
# MAGIC sub-second — not for a 30-second counter.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

dbutils.widgets.dropdown("mode", "batch", ["batch", "loop"], "batch = one refresh (cron); loop = refresh every trigger_seconds")
dbutils.widgets.text("trigger_seconds", "30", "loop mode: seconds between refreshes")
dbutils.widgets.text("max_ticks", "0", "loop mode: number of refreshes before stopping (0 = run forever)")

MODE = dbutils.widgets.get("mode")
TRIGGER_SECONDS = int(dbutils.widgets.get("trigger_seconds"))
MAX_TICKS = int(dbutils.widgets.get("max_ticks"))

import time
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
# get_online_store returns None (does not raise) when the store is missing; fail fast with a
# clear message rather than a confusing NoneType error inside the refresh.
store = fe.get_online_store(name=ONLINE_STORE)
if store is None:
    raise RuntimeError(f"Online store '{ONLINE_STORE}' not found — run 03_online_store first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## One refresh: recompute 1h counters over the full universe & publish
# MAGIC Recomputing over the full instrument universe (not just recent rows) resets idle
# MAGIC instruments to 0 rather than leaving stale values.

# COMMAND ----------

def refresh_once(tick: int) -> None:
    raw = spark.read.table(RAW_EVENTS)
    cutoff = raw.agg(F.max("event_ts").alias("m")).first()["m"]
    if cutoff is None:
        print(f"[tick {tick}] no events yet; skipping.")
        return
    recent = raw.filter(F.col("event_ts") >= F.lit(cutoff) - F.expr("INTERVAL 1 HOUR"))

    agg = (
        recent.groupBy("instrument_id").agg(
            F.count("*").alias("inst_txn_cnt"),
            F.sum(F.when(F.col("outcome") == "hard_fail", 1).otherwise(0)).alias("inst_fail_cnt"),
            F.countDistinct("account_id").alias("inst_distinct_accounts"),
            F.round(F.avg("amount"), 2).alias("inst_avg_amount"),
            F.max("amount").alias("inst_max_amount"),
        )
        .withColumn("inst_fail_ratio", F.round(F.col("inst_fail_cnt") / F.col("inst_txn_cnt"), 4))
    )
    # Reset instruments with no recent activity to 0 by recomputing over the full universe.
    universe = raw.select("instrument_id").distinct()
    counters = (
        universe.join(agg, "instrument_id", "left")
        .na.fill(0)
        .withColumn("computed_at", F.current_timestamp())
    )

    fe.write_table(name=FT_COUNTERS_1H, df=counters, mode="merge")
    fe.publish_table(
        online_store=store,
        source_table_name=FT_COUNTERS_1H,
        online_table_name=f"{FT_COUNTERS_1H}_online",
        publish_mode="TRIGGERED",
    )
    print(f"[tick {tick}] refreshed + published 1h counters (idle reset to 0).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run
# MAGIC `batch` does a single refresh and returns — schedule the job (e.g. every 5 minutes) for
# MAGIC the cadence. `loop` refreshes every `trigger_seconds` on the driver; `max_ticks=0` runs
# MAGIC forever (deploy as a continuous job), any positive value stops after that many ticks
# MAGIC (`max_ticks=1` is handy for a single CI refresh). The loop sleeps for the remainder of
# MAGIC each interval so the cadence stays ~`trigger_seconds` regardless of recompute time.

# COMMAND ----------

if MODE == "batch":
    refresh_once(1)
else:
    tick = 0
    while True:
        tick += 1
        started = time.time()
        refresh_once(tick)
        if MAX_TICKS and tick >= MAX_TICKS:
            print(f"Completed {tick} refresh tick(s).")
            break
        elapsed = time.time() - started
        time.sleep(max(0.0, TRIGGER_SECONDS - elapsed))
