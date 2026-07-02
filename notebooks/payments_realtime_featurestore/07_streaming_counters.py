# Databricks notebook source

# MAGIC %md
# MAGIC # 07 · Counter refresh (hot path)
# MAGIC
# MAGIC Keeps the per-instrument 1h sliding-window counters fresh. Run on a short schedule
# MAGIC (the `payments_counters_refresh` job, every few minutes): each run recomputes the
# MAGIC last-hour counters over the **full instrument universe** — so instruments idle in
# MAGIC the last hour are reset to 0 rather than keeping stale values — merges them into the
# MAGIC counter feature table, and re-publishes to the Lakebase online store.
# MAGIC
# MAGIC This is a plain batch recompute (no `foreachBatch`): serverless runs on Spark
# MAGIC Connect, where a streaming `foreachBatch` cannot reference the outer Spark session or
# MAGIC the Feature Engineering client, and a periodic batch is simpler than a rate-stream
# MAGIC clock for the same "refresh every N minutes" behavior.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
raw = spark.read.table(RAW_EVENTS)

# COMMAND ----------

# Last-hour window, relative to the most recent event.
cutoff = raw.agg(F.max("event_ts").alias("m")).first()["m"]
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
print(f"Refreshed 1h counters for {counters.count():,} instruments (idle reset to 0).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Re-publish to the online store

# COMMAND ----------

store = fe.get_online_store(name=ONLINE_STORE)
fe.publish_table(
    online_store=store,
    source_table_name=FT_COUNTERS_1H,
    online_table_name=f"{FT_COUNTERS_1H}_online",
    publish_mode="TRIGGERED",
)
print("Re-published counters to the online store.")
