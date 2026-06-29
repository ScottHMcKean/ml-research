# Databricks notebook source

# MAGIC %md
# MAGIC # 02 · Feature Engineering (Feature Engineering API)
# MAGIC
# MAGIC Declares the demo's features with the **Databricks Feature Engineering API**
# MAGIC (`FeatureEngineeringClient` in Unity Catalog — *not* the legacy `FeatureStoreClient`):
# MAGIC
# MAGIC | Table | Key | Timeseries | Pattern |
# MAGIC |-------|-----|-----------|---------|
# MAGIC | `ft_instrument_counters_1h` | `instrument_id` | `computed_at` | hot 1h sliding-window counters (kept fresh by `07_streaming_counters`) |
# MAGIC | `ft_account_daily` | `account_id` | `feature_ts` | **daily** aggregate cache, fed forward, point-in-time correct |
# MAGIC | `ft_account_monthly` | `account_id` | `feature_ts` | **monthly** aggregate cache, fed forward, point-in-time correct |
# MAGIC | `ft_account_profile` | `account_id` | — | slow-changing enrichment dimension |
# MAGIC
# MAGIC Plus **on-demand** request-time features as Unity Catalog Python UDFs, computed
# MAGIC in-request at both training and serving time via `FeatureFunction`.
# MAGIC
# MAGIC The **"cache daily & monthly values and feed forward"** requirement is handled by
# MAGIC the timeseries column: a feature row becomes effective at the *end* of its period,
# MAGIC so an event looks up the most recent *completed* day/month and carries it forward
# MAGIC until the next refresh. `08_backfill_cache` recomputes these on a schedule.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
raw = spark.read.table(RAW_EVENTS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reset (idempotent re-runs)
# MAGIC `fe.create_table` errors if the table already exists, so drop the feature tables up
# MAGIC front. Only safe before the tables are published to the online store.

# COMMAND ----------

for t in (FT_COUNTERS_1H, FT_ACCOUNT_DAILY, FT_ACCOUNT_MONTHLY, FT_ACCOUNT_PROFILE):
    spark.sql(f"DROP TABLE IF EXISTS {t}")
    print(f"dropped (if existed): {t}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## On-demand request-time features (UC Python UDFs)
# MAGIC Scalar UDFs that take only fields present on the request, so the serving caller
# MAGIC passes nothing but keys + raw request inputs.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {ODF_REQUEST}_log_amount(amount DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON
COMMENT 'On-demand: log1p of the transaction amount.'
AS $$
import math
return float(math.log1p(amount or 0.0))
$$
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {ODF_REQUEST}_is_night(event_ts TIMESTAMP)
RETURNS INT
LANGUAGE PYTHON
COMMENT 'On-demand: 1 if the event hour is overnight (>=23 or <6).'
AS $$
if event_ts is None:
    return 0
h = event_ts.hour
return 1 if (h >= 23 or h < 6) else 0
$$
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {ODF_REQUEST}_high_amount(amount DOUBLE)
RETURNS INT
LANGUAGE PYTHON
COMMENT 'On-demand: 1 if the amount is in the high-value band (>500).'
AS $$
return 1 if (amount or 0.0) > 500.0 else 0
$$
""")

# Ordinal-encode the request category so every model feature is numeric (the serving
# endpoint replays raw assembled columns straight into LightGBM — no string categoricals).
spark.sql(f"""
CREATE OR REPLACE FUNCTION {ODF_REQUEST}_category_ordinal(category_code STRING)
RETURNS INT
LANGUAGE PYTHON
COMMENT 'On-demand: ordinal encoding of category_code (A..E -> 0..4).'
AS $$
return {{'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}}.get(category_code, -1)
$$
""")

print("Created on-demand UDFs:")
for suffix in ("log_amount", "is_night", "high_amount", "category_ordinal"):
    print(f"  {ODF_REQUEST}_{suffix}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sliding-window counters (hot, per instrument)
# MAGIC A **latest-value** snapshot per instrument (no timeseries column) — at serving time
# MAGIC the online store always returns the *current* counter, which is exactly the hot-path
# MAGIC behavior we want. We **seed** from full history so every instrument has a row for
# MAGIC training; in production `07_streaming_counters` overwrites these with rolling 1h
# MAGIC values every few seconds. Missing keys yield nulls, which LightGBM handles natively
# MAGIC (the seed covers every instrument in the data, so nulls are rare).

# COMMAND ----------

counters = (
    raw.groupBy("instrument_id").agg(
        F.count("*").alias("inst_txn_cnt"),
        F.sum(F.when(F.col("outcome") == "hard_fail", 1).otherwise(0)).alias("inst_fail_cnt"),
        F.countDistinct("account_id").alias("inst_distinct_accounts"),
        F.avg("amount").alias("inst_avg_amount"),
        F.max("amount").alias("inst_max_amount"),
    )
    .withColumn("inst_fail_ratio", F.round(F.col("inst_fail_cnt") / F.col("inst_txn_cnt"), 4))
    .withColumn("inst_avg_amount", F.round("inst_avg_amount", 2))
    .withColumn("computed_at", F.current_timestamp())
)

fe.create_table(
    name=FT_COUNTERS_1H,
    primary_keys=["instrument_id"],
    df=counters,
    description="Per-instrument activity counters; seeded from history, kept fresh (1h rolling) by the streaming job.",
)
print(f"{FT_COUNTERS_1H}: {counters.count():,} instruments")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Daily aggregate cache (per account, fed forward)
# MAGIC One row per account per day. `feature_ts` is set to the **start of the next day** so
# MAGIC a point-in-time lookup returns the most recent *completed* day and feeds it forward.

# COMMAND ----------

daily = (
    raw.withColumn("d", F.to_date("event_ts"))
    .groupBy("account_id", "d").agg(
        F.count("*").alias("acct_daily_cnt"),
        F.avg("amount").alias("acct_daily_avg_amount"),
        F.avg(F.col("blocked").cast("double")).alias("acct_daily_block_rate"),
        F.sum(F.when(F.col("outcome") == "soft_fail", 1).otherwise(0)).alias("acct_daily_softfail_cnt"),
    )
    # Effective at the start of the following day -> feed-forward / point-in-time correct.
    .withColumn("feature_ts", F.expr("timestamp(d) + INTERVAL 1 DAY"))
    .drop("d")
)
for c in ["acct_daily_avg_amount", "acct_daily_block_rate"]:
    daily = daily.withColumn(c, F.round(c, 4))

fe.create_table(
    name=FT_ACCOUNT_DAILY,
    # Time-series feature tables require the timeseries column to be part of the primary key.
    primary_keys=["account_id", "feature_ts"],
    timeseries_column="feature_ts",
    df=daily,
    description="Daily per-account aggregate cache; fed forward via point-in-time timeseries.",
)
print(f"{FT_ACCOUNT_DAILY}: {daily.count():,} account-days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monthly aggregate cache (per account, fed forward)

# COMMAND ----------

monthly = (
    raw.withColumn("m", F.trunc("event_ts", "MM"))
    .groupBy("account_id", "m").agg(
        F.count("*").alias("acct_monthly_cnt"),
        F.avg("amount").alias("acct_monthly_avg_amount"),
        F.avg(F.col("blocked").cast("double")).alias("acct_monthly_block_rate"),
    )
    # Effective at the start of the following month.
    .withColumn("feature_ts", F.add_months(F.col("m"), 1).cast("timestamp"))
    .drop("m")
)
for c in ["acct_monthly_avg_amount", "acct_monthly_block_rate"]:
    monthly = monthly.withColumn(c, F.round(c, 4))

fe.create_table(
    name=FT_ACCOUNT_MONTHLY,
    primary_keys=["account_id", "feature_ts"],
    timeseries_column="feature_ts",
    df=monthly,
    description="Monthly per-account aggregate cache; fed forward via point-in-time timeseries.",
)
print(f"{FT_ACCOUNT_MONTHLY}: {monthly.count():,} account-months")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Account profile (enrichment dimension)

# COMMAND ----------

# Numeric-encode enrichment so every model feature is numeric at serving time.
profile = (
    spark.read.table(f"{CATALOG}.{SCHEMA}.account_profile_raw")
    .withColumn("is_premium", (F.col("account_tier") == "premium").cast("int"))
    .withColumn("home_region_code", F.regexp_replace("home_region", "r", "").cast("int"))
    .select("account_id", "is_premium", "account_age_days", "home_region_code", "historical_decline_rate")
)

fe.create_table(
    name=FT_ACCOUNT_PROFILE,
    primary_keys=["account_id"],
    df=profile,
    description="Slow-changing per-account enrichment features (numeric).",
)
print(f"{FT_ACCOUNT_PROFILE}: {profile.count():,} accounts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable Change Data Feed
# MAGIC `TRIGGERED`/`CONTINUOUS` online publishing reads the table's Change Data Feed, so
# MAGIC enable it on every feature table.

# COMMAND ----------

for t in (FT_COUNTERS_1H, FT_ACCOUNT_DAILY, FT_ACCOUNT_MONTHLY, FT_ACCOUNT_PROFILE):
    spark.sql(f"ALTER TABLE {t} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"CDF enabled: {t}")
