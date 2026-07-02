# Databricks notebook source

# MAGIC %md
# MAGIC # 05 · Model Serving (automatic feature lookup)
# MAGIC
# MAGIC Deploys the registered LightGBM model to a Model Serving endpoint. Because the model
# MAGIC was logged with `fe.log_model(training_set=...)`, the endpoint **automatically** looks
# MAGIC up online features (from the Lakebase online store) and computes the on-demand UDFs
# MAGIC in-request. Callers send only **keys + raw request inputs**:
# MAGIC `instrument_id, account_id, category_code, amount, event_ts`.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow.tracking import MlflowClient

w = WorkspaceClient()
champion = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(MODEL_NAME, "champion")
print(f"Serving {MODEL_NAME} v{champion.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or update the endpoint

# COMMAND ----------

served = ServedEntityInput(
    entity_name=MODEL_NAME,
    entity_version=champion.version,
    workload_size="Small",
    scale_to_zero_enabled=True,
)
config = EndpointCoreConfigInput(name=SERVING_ENDPOINT, served_entities=[served])

existing = [e.name for e in w.serving_endpoints.list()]
if SERVING_ENDPOINT in existing:
    w.serving_endpoints.update_config_and_wait(name=SERVING_ENDPOINT, served_entities=[served])
    print(f"Updated endpoint {SERVING_ENDPOINT}")
else:
    w.serving_endpoints.create_and_wait(name=SERVING_ENDPOINT, config=config)
    print(f"Created endpoint {SERVING_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test — keys + request inputs only
# MAGIC The endpoint joins counters/daily/monthly/profile features and computes the
# MAGIC on-demand features before scoring.

# COMMAND ----------

import datetime as dt

sample = spark.read.table(RAW_EVENTS).select(
    "instrument_id", "account_id", "category_code", "amount",
).limit(3).toPandas()

records = [
    {
        "instrument_id": r.instrument_id,
        "account_id": r.account_id,
        "category_code": r.category_code,
        "amount": float(r.amount),
        "event_ts": dt.datetime.utcnow().isoformat(),
    }
    for r in sample.itertuples()
]

resp = w.serving_endpoints.query(name=SERVING_ENDPOINT, dataframe_records=records)
print("Predictions:", resp.predictions)
