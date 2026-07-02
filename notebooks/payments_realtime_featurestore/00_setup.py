# Databricks notebook source

# MAGIC %md
# MAGIC # Payments Real-Time Feature Store — Setup
# MAGIC
# MAGIC Shared constants and idempotent Unity Catalog setup for the payments real-time
# MAGIC feature-store + serving demo. Every other notebook starts with `%run ./00_setup`
# MAGIC so the catalog/schema, table names, model name, online-store name, and serving
# MAGIC endpoint name are defined in exactly one place.
# MAGIC
# MAGIC The demo turns a real-time scoring **architecture sample** into a running,
# MAGIC benchmarkable system:
# MAGIC `seed events → declare features → publish online (Lakebase) → train LightGBM →
# MAGIC serve with automatic feature lookup → benchmark latency`.
# MAGIC
# MAGIC All data is **synthetic** and all names are **generic** — this is a pattern
# MAGIC reference, not tied to any company.

# COMMAND ----------

dbutils.widgets.text("catalog", "shm_skunkworks_catalog", "Unity Catalog catalog")
dbutils.widgets.text("schema", "payments", "Schema for the demo artifacts")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

# COMMAND ----------

# Raw event stream (one row per payment authorization request).
RAW_EVENTS = f"{CATALOG}.{SCHEMA}.raw_events"

# Feature tables (declared with the Feature Engineering API in 02_feature_engineering).
#   counters_1h : hot, fast-changing 1h sliding-window counters keyed by instrument.
#   account_daily / account_monthly : "cache and feed forward" aggregates keyed by account,
#                                     point-in-time correct via a timeseries column.
#   account_profile : slow-changing enrichment / reference dimension keyed by account.
FT_COUNTERS_1H = f"{CATALOG}.{SCHEMA}.ft_instrument_counters_1h"
FT_ACCOUNT_DAILY = f"{CATALOG}.{SCHEMA}.ft_account_daily"
FT_ACCOUNT_MONTHLY = f"{CATALOG}.{SCHEMA}.ft_account_monthly"
FT_ACCOUNT_PROFILE = f"{CATALOG}.{SCHEMA}.ft_account_profile"

# On-demand (request-time) feature function — computed in-request at training and serving.
ODF_REQUEST = f"{CATALOG}.{SCHEMA}.odf_request_transforms"

# Labelled training spine and benchmark results.
TRAIN_SPINE = f"{CATALOG}.{SCHEMA}.training_spine"
BENCHMARK_RESULTS = f"{CATALOG}.{SCHEMA}.benchmark_results"

# Model (Unity Catalog), online store, and serving endpoint.
MODEL_NAME = f"{CATALOG}.{SCHEMA}.payments_lgbm"
ONLINE_STORE = "payments-online-store"
SERVING_ENDPOINT = "payments-scoring"

# Entity keys + label used throughout.
KEYS = ["instrument_id", "account_id", "bin_prefix", "category_code"]
LABEL = "blocked"

# COMMAND ----------

# The catalog is expected to already exist (e.g. the workspace's skunkworks catalog);
# we only manage the schema + volume inside it.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.checkpoints")

CHECKPOINT_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints"

# COMMAND ----------

print("Payments feature-store demo configuration")
print(f"  catalog.schema : {CATALOG}.{SCHEMA}")
print(f"  raw events     : {RAW_EVENTS}")
print(f"  feature tables : {FT_COUNTERS_1H}")
print(f"                   {FT_ACCOUNT_DAILY}")
print(f"                   {FT_ACCOUNT_MONTHLY}")
print(f"                   {FT_ACCOUNT_PROFILE}")
print(f"  on-demand udf  : {ODF_REQUEST}")
print(f"  model          : {MODEL_NAME}")
print(f"  online store   : {ONLINE_STORE}")
print(f"  endpoint       : {SERVING_ENDPOINT}")
