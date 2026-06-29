# Databricks notebook source

# MAGIC %md
# MAGIC # 08 · Backfill the daily & monthly caches
# MAGIC
# MAGIC Recomputes the per-account **daily** and **monthly** aggregate caches from
# MAGIC `raw_events` and merges them into the feature tables, then refreshes the online
# MAGIC store. This is the scheduled job behind the **"cache daily & monthly values and feed
# MAGIC forward"** requirement — run it daily (and monthly) on a cron, or on demand from the
# MAGIC App's `/backfill` button.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

dbutils.widgets.dropdown("grain", "both", ["daily", "monthly", "both"], "Which cache to refresh")
GRAIN = dbutils.widgets.get("grain")

from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
raw = spark.read.table(RAW_EVENTS)

# COMMAND ----------

if GRAIN in ("daily", "both"):
    daily = (
        raw.withColumn("d", F.to_date("event_ts"))
        .groupBy("account_id", "d").agg(
            F.count("*").alias("acct_daily_cnt"),
            F.round(F.avg("amount"), 4).alias("acct_daily_avg_amount"),
            F.round(F.avg(F.col("blocked").cast("double")), 4).alias("acct_daily_block_rate"),
            F.sum(F.when(F.col("outcome") == "soft_fail", 1).otherwise(0)).alias("acct_daily_softfail_cnt"),
        )
        .withColumn("feature_ts", F.expr("timestamp(d) + INTERVAL 1 DAY"))
        .drop("d")
    )
    fe.write_table(name=FT_ACCOUNT_DAILY, df=daily, mode="merge")
    print(f"Daily cache refreshed: {daily.count():,} account-days")

# COMMAND ----------

if GRAIN in ("monthly", "both"):
    monthly = (
        raw.withColumn("m", F.trunc("event_ts", "MM"))
        .groupBy("account_id", "m").agg(
            F.count("*").alias("acct_monthly_cnt"),
            F.round(F.avg("amount"), 4).alias("acct_monthly_avg_amount"),
            F.round(F.avg(F.col("blocked").cast("double")), 4).alias("acct_monthly_block_rate"),
        )
        .withColumn("feature_ts", F.add_months(F.col("m"), 1).cast("timestamp"))
        .drop("m")
    )
    fe.write_table(name=FT_ACCOUNT_MONTHLY, df=monthly, mode="merge")
    print(f"Monthly cache refreshed: {monthly.count():,} account-months")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refresh the online store
# MAGIC Re-trigger publishing so the merged values land in the Lakebase online store.

# COMMAND ----------

store = fe.get_online_store(name=ONLINE_STORE)
refresh = []
if GRAIN in ("daily", "both"):
    refresh.append(FT_ACCOUNT_DAILY)
if GRAIN in ("monthly", "both"):
    refresh.append(FT_ACCOUNT_MONTHLY)

for src in refresh:
    fe.publish_table(
        online_store=store,
        source_table_name=src,
        online_table_name=f"{src}_online",
        publish_mode="TRIGGERED",
    )
    print(f"Re-published {src}")
