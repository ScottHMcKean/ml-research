# Databricks notebook source

# MAGIC %md
# MAGIC # 07b · Counter refresh (30-second hot path)
# MAGIC
# MAGIC The **30-second-freshness** answer. `07_streaming_counters` refreshes the hot 1h
# MAGIC counters with a plain batch recompute on a **5-minute** cron — simple, cheap, correct
# MAGIC when minutes of staleness are fine. This notebook is the alternative for
# MAGIC **burst-detection features that must reflect events within ~30 seconds**: the same
# MAGIC full-universe recompute, driven on a **30-second driver-side clock** and re-published to
# MAGIC the Lakebase online store each tick.
# MAGIC
# MAGIC ## Why a driver-side loop, not `foreachBatch`
# MAGIC The recompute resets *idle* instruments to 0 by recomputing over the full instrument
# MAGIC universe, and `fe.write_table` / `fe.publish_table` are **driver-side** APIs that need
# MAGIC the notebook's auth context. On serverless (Spark Connect) a streaming `foreachBatch`
# MAGIC closure runs in an **isolated worker process** with no Spark session and no default
# MAGIC credentials, so the Feature Engineering client cannot run there
# MAGIC (`STREAMING_CONNECT_SERIALIZATION_ERROR`, then `cannot configure default credentials`).
# MAGIC A short driver-side loop keeps every FE call on the driver where auth works — and, as
# MAGIC `07`'s own header notes, a periodic recompute is simpler than a rate-stream clock for the
# MAGIC same "refresh every N seconds" behavior.
# MAGIC
# MAGIC A true **incremental streaming aggregation** belongs with the Kafka source: the stream
# MAGIC writes to a Delta feature table and Lakebase **`CONTINUOUS`** publish auto-syncs it
# MAGIC (no FE client in the hot loop). See `09_kafka_io`.
# MAGIC
# MAGIC | Path | Cadence | Freshness | Compute | When |
# MAGIC |------|---------|-----------|---------|------|
# MAGIC | `07` batch | 5-min cron | ~5 min | serverless job | minutes OK, cheapest |
# MAGIC | `07b` loop (this) | 30-s driver loop | ~30–60 s | serverless job | 30-s burst features |
# MAGIC | `09` stream → CONTINUOUS | micro-batch | ~10–20 s | serverless stream | incremental agg from Kafka |
# MAGIC | RTM (see `09`) | real-time mode | sub-second | dedicated slots | per-event decisions |
# MAGIC
# MAGIC ## Why not Real-Time Mode for 30 seconds
# MAGIC The requirement is **30-second feature freshness**, not sub-second per-event latency, so
# MAGIC RTM (a ~5 ms floor, dedicated compute slots, real cost) is the wrong altitude. Reach for
# MAGIC RTM only when a *feature* or the *per-transaction decision* must land sub-second.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

dbutils.widgets.text("trigger_seconds", "30", "Refresh interval (seconds)")
dbutils.widgets.text("max_ticks", "1", "How many refreshes to run (1 = single tick for CI; 0 = run forever)")
TRIGGER_SECONDS = int(dbutils.widgets.get("trigger_seconds"))
MAX_TICKS = int(dbutils.widgets.get("max_ticks"))

import time
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
# get_online_store returns None (does not raise) when the store is missing; fail fast with a
# clear message rather than a confusing NoneType error inside the refresh loop.
store = fe.get_online_store(name=ONLINE_STORE)
if store is None:
    raise RuntimeError(f"Online store '{ONLINE_STORE}' not found — run 03_online_store first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## One refresh: recompute 1h counters over the full universe & publish
# MAGIC Identical logic to `07`, factored into a function we call on a timer. Recomputing over
# MAGIC the full instrument universe (not just recent rows) resets idle instruments to 0 rather
# MAGIC than leaving stale values.

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
# MAGIC ## Run on a 30-second clock
# MAGIC `max_ticks=1` runs a single refresh and stops — ideal for CI / bundle runs.
# MAGIC `max_ticks=0` loops forever (every `trigger_seconds`); deploy that as a
# MAGIC **continuous** job for a live 30-second hot path. The loop sleeps for the remainder of
# MAGIC each interval so the cadence stays ~30 s regardless of recompute time.

# COMMAND ----------

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
