# Databricks notebook source

# MAGIC %md
# MAGIC # 07b · Counter refresh (real-time streaming path)
# MAGIC
# MAGIC The **30-second-freshness** answer. `07_streaming_counters` refreshes the hot 1h
# MAGIC counters with a plain batch recompute on a 5-minute cron — simple, cheap, and correct
# MAGIC when minutes of staleness are acceptable. This notebook is the alternative for
# MAGIC **burst-detection features that must reflect events within ~30 seconds**: a continuous
# MAGIC Structured Streaming job with `trigger(processingTime="30 seconds")` that recomputes
# MAGIC the rolling-window counters every micro-batch and publishes them to the Lakebase
# MAGIC online store.
# MAGIC
# MAGIC ## Why micro-batch (and not Real-Time Mode) for 30 seconds
# MAGIC The requirement is **30-second feature freshness**, not sub-second per-event latency.
# MAGIC A `processingTime="30 seconds"` micro-batch trigger meets that directly, on standard
# MAGIC (GA) Structured Streaming, with **no always-on classic cluster required** (serverless
# MAGIC streaming works). Reach for **Spark Real-Time Mode (RTM)** only when a *feature* or the
# MAGIC *per-transaction decision* must land in sub-second time — RTM delivers a ~5 ms floor
# MAGIC (typical p99 ~300 ms) but demands dedicated compute slots and real cost. For a 30 s
# MAGIC counter that is the wrong altitude. See the freshness-tier table in the README.
# MAGIC
# MAGIC | Path | Trigger | Freshness | Compute | When |
# MAGIC |------|---------|-----------|---------|------|
# MAGIC | `07` batch | 5-min cron | ~5 min | serverless job | minutes OK, cheapest |
# MAGIC | `07b` micro-batch (this) | `processingTime=30s` | ~30–60 s | serverless streaming | 30 s burst features |
# MAGIC | RTM (see `09_kafka_io`) | real-time mode | sub-second | dedicated slots | per-event decisions |
# MAGIC
# MAGIC ## Freshness vs. window
# MAGIC The 1h **window** (how far back a counter looks) is independent of the 30 s **trigger**
# MAGIC (how often it recomputes). We keep the 1h rolling window and recompute it every 30 s so
# MAGIC a burst shows up in the counter within one trigger interval.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

dbutils.widgets.text("trigger_seconds", "30", "Micro-batch trigger interval (seconds)")
dbutils.widgets.dropdown("run_mode", "once", ["once", "continuous"], "once = single refresh (CI), continuous = keep running")
TRIGGER_SECONDS = int(dbutils.widgets.get("trigger_seconds"))
RUN_MODE = dbutils.widgets.get("run_mode")

from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
store = fe.get_online_store(name=ONLINE_STORE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stream the raw events
# MAGIC `raw_events` is a Delta table, so we read it as a stream. In production this source is
# MAGIC the payment-authorization stream itself (Kafka — see `09_kafka_io`); pointing the same
# MAGIC aggregation at a Delta table keeps this notebook self-contained and testable.

# COMMAND ----------

events = spark.readStream.table(RAW_EVENTS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-micro-batch: recompute 1h counters over the full universe & publish
# MAGIC We can't hold an unbounded streaming aggregation and reset idle instruments to 0 at the
# MAGIC same time, and `fe.write_table` / `fe.publish_table` are batch APIs. So we use
# MAGIC `foreachBatch`: each 30 s micro-batch hands us a *batch* DataFrame, and we recompute the
# MAGIC last-hour counters over the **full instrument universe** (idle instruments reset to 0),
# MAGIC merge into the feature table, and re-publish to Lakebase — the same logic as the batch
# MAGIC job in `07`, driven by a streaming clock instead of a cron.
# MAGIC
# MAGIC > **Serverless note:** on serverless, Spark Connect forbids referencing the outer
# MAGIC > `spark` session or the FE client from inside a `foreachBatch` closure that runs on
# MAGIC > executors. Here the closure runs on the **driver** (Structured Streaming invokes
# MAGIC > `foreachBatch` driver-side), so using `spark`, `fe`, and `store` is fine. Recompute
# MAGIC > from the full table (not just `micro_df`) so idle instruments are reset to 0.

# COMMAND ----------

def refresh_counters(micro_df, epoch_id: int) -> None:
    # micro_df carries only the new rows for this trigger; we ignore its contents and
    # recompute the authoritative 1h counters from the full table so idle instruments
    # (no activity in the last hour) are correctly reset to 0 rather than left stale.
    if micro_df.isEmpty():
        return

    raw = spark.read.table(RAW_EVENTS)
    cutoff = raw.agg(F.max("event_ts").alias("m")).first()["m"]
    if cutoff is None:
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
    print(f"[epoch {epoch_id}] refreshed + published 1h counters (idle reset to 0).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start the stream
# MAGIC `run_mode=once` runs a single micro-batch and stops — ideal for CI / bundle runs.
# MAGIC `run_mode=continuous` keeps refreshing every `trigger_seconds`; deploy it as an
# MAGIC always-running job (or a serverless continuous job) for a live 30 s hot path.

# COMMAND ----------

checkpoint = f"{CHECKPOINT_ROOT}/counters_rt"

writer = (
    events.writeStream
    .foreachBatch(refresh_counters)
    .option("checkpointLocation", checkpoint)
    .queryName("counters_rt")
)

if RUN_MODE == "continuous":
    q = writer.trigger(processingTime=f"{TRIGGER_SECONDS} seconds").start()
    print(f"Streaming counters every {TRIGGER_SECONDS}s (continuous). Query id: {q.id}")
    q.awaitTermination()
else:
    # availableNow drains whatever is currently in the source, runs foreachBatch once, stops.
    q = writer.trigger(availableNow=True).start()
    q.awaitTermination()
    print("Single refresh complete (run_mode=once).")
