# Databricks notebook source

# MAGIC %md
# MAGIC # 03 · Online Feature Store (Lakebase)
# MAGIC
# MAGIC Publishes the feature tables to a **Lakebase-backed Online Feature Store** for
# MAGIC low-latency point lookups at serving time. This is the current path —
# MAGIC `fe.create_online_store` + `fe.publish_table` — and replaces the legacy
# MAGIC `OnlineTableSpec` / online tables (no longer supported) and the external Redis
# MAGIC hot-cache in the original architecture sample.
# MAGIC
# MAGIC | Feature table | Online publish mode | Why |
# MAGIC |---------------|---------------------|-----|
# MAGIC | counters (1h) | `TRIGGERED` (use `CONTINUOUS` in prod) | hot, changes every second |
# MAGIC | account daily | `TRIGGERED` | refreshed once per day by the backfill job |
# MAGIC | account monthly | `TRIGGERED` | refreshed once per month |
# MAGIC | account profile | `TRIGGERED` | slow-changing reference data |

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

dbutils.widgets.text("capacity", "CU_1", "Online store capacity (CU_1/2/4/8)")
CAPACITY = dbutils.widgets.get("capacity")

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create (or reuse) the online store
# MAGIC Backed by a Lakebase (Postgres) instance that Databricks manages for us.

# COMMAND ----------

import time

# get_online_store returns None (it does not raise) when the store does not exist.
store = fe.get_online_store(name=ONLINE_STORE)
if store is None:
    store = fe.create_online_store(name=ONLINE_STORE, capacity=CAPACITY)
    print(f"Creating online store: {ONLINE_STORE} ({CAPACITY}) — provisioning Lakebase...")
else:
    print(f"Reusing existing online store: {ONLINE_STORE}")

# Wait until the store is AVAILABLE before publishing (provisioning can take several minutes).
deadline = time.time() + 1800
while True:
    store = fe.get_online_store(name=ONLINE_STORE)
    state = getattr(store, "state", None)
    state = getattr(state, "value", state)
    print(f"  online store state: {state}")
    if str(state) == "AVAILABLE":
        break
    if time.time() > deadline:
        raise TimeoutError(f"Online store not AVAILABLE after 30 min (state={state})")
    time.sleep(30)

print(f"Online store ready: {store}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Publish each feature table

# COMMAND ----------

PUBLISH = [
    (FT_COUNTERS_1H, "TRIGGERED"),
    (FT_ACCOUNT_DAILY, "TRIGGERED"),
    (FT_ACCOUNT_MONTHLY, "TRIGGERED"),
    (FT_ACCOUNT_PROFILE, "TRIGGERED"),
]

for source, mode in PUBLISH:
    online_name = f"{source}_online"
    fe.publish_table(
        online_store=store,
        source_table_name=source,
        online_table_name=online_name,
        publish_mode=mode,
    )
    print(f"Published {source}  ->  {online_name}  [{mode}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify a point lookup
# MAGIC Confirm the published tables serve a known key with low latency.

# COMMAND ----------

sample_account = spark.read.table(FT_ACCOUNT_PROFILE).limit(1).first()["account_id"]
print(f"Sample account: {sample_account}")
print("Online store is ready. Feature lookups will be exercised by the serving endpoint in 05.")
