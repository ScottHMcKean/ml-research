# Databricks notebook source

# MAGIC %md
# MAGIC # 07 · Streaming counter refresh (hot path)
# MAGIC
# MAGIC Keeps the per-instrument 1h sliding-window counters fresh. A `rate` stream acts as a
# MAGIC clock; on each micro-batch we recompute the **last-hour** counters from `raw_events`
# MAGIC and merge them into the counter feature table, so the online store always serves
# MAGIC current values on the hot path.
# MAGIC
# MAGIC Run this as the **continuous** `payments_counters_streaming` job (paused by default).
# MAGIC With the counter table published `CONTINUOUS`, online values track these writes;
# MAGIC with `TRIGGERED`, pair this with a periodic `fe.publish_table` refresh.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

dbutils.widgets.text("refresh_seconds", "30", "Seconds between counter refreshes")
REFRESH_SECONDS = int(dbutils.widgets.get("refresh_seconds"))

from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# COMMAND ----------

def recompute_counters(batch_df, batch_id):
    """Recompute last-1h per-instrument counters from raw_events and merge them in."""
    raw = spark.read.table(RAW_EVENTS)
    cutoff = raw.agg(F.max("event_ts").alias("m")).first()["m"]
    if cutoff is None:
        return
    recent = raw.filter(F.col("event_ts") >= F.lit(cutoff) - F.expr("INTERVAL 1 HOUR"))
    counters = (
        recent.groupBy("instrument_id").agg(
            F.count("*").alias("inst_txn_cnt"),
            F.sum(F.when(F.col("outcome") == "hard_fail", 1).otherwise(0)).alias("inst_fail_cnt"),
            F.countDistinct("account_id").alias("inst_distinct_accounts"),
            F.round(F.avg("amount"), 2).alias("inst_avg_amount"),
            F.max("amount").alias("inst_max_amount"),
        )
        .withColumn("inst_fail_ratio", F.round(F.col("inst_fail_cnt") / F.col("inst_txn_cnt"), 4))
        .withColumn("computed_at", F.current_timestamp())
    )
    # Upsert into the feature table (mode='merge' keys on the primary key).
    fe.write_table(name=FT_COUNTERS_1H, df=counters, mode="merge")
    print(f"[batch {batch_id}] refreshed counters for {counters.count():,} instruments")

# COMMAND ----------

clock = (
    spark.readStream.format("rate")
    .option("rowsPerSecond", 1)
    .load()
    .where(F.col("value") % REFRESH_SECONDS == 0)  # fire roughly every REFRESH_SECONDS
)

(
    clock.writeStream
    .foreachBatch(recompute_counters)
    .option("checkpointLocation", f"{CHECKPOINT_ROOT}/counters_refresh")
    .trigger(processingTime=f"{REFRESH_SECONDS} seconds")
    .start()
    .awaitTermination()
)
