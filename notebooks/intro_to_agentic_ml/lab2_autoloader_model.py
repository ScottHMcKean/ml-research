# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md-sandbox
# MAGIC <div style="background:#1B3139; color:white; padding:24px; border-radius:8px;">
# MAGIC   <h1 style="margin:0; color:white;">Lab 2B — Train an anomaly model & score the live stream</h1>
# MAGIC   <p style="margin:6px 0 0 0; color:#9EB7BE; font-size:16px;">
# MAGIC     Persona: <strong style="color:white;">ML Engineer</strong>
# MAGIC     &nbsp;·&nbsp; Builds on Lab 2A's medallion pipeline
# MAGIC   </p>
# MAGIC </div>
# MAGIC
# MAGIC Lab 2A detected anomalies with **SQL rules** (threshold + rolling z-score). Here we **train a
# MAGIC simple ML model** (Isolation Forest) on the **historical** readings (`fact_sensor_readings`),
# MAGIC **log + register it to MLflow / Unity Catalog**, then use it to run **inference on the live
# MAGIC streamed data** the Auto Loader pipeline is producing (`live_silver_readings`) — the same
# MAGIC running job, now scored by a registered model.
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #FF3620; padding:15px; margin:15px 0;">
# MAGIC   <strong>The pattern:</strong> train offline on history → register to UC → load the model and
# MAGIC   score the live pipeline output. The model travels with its signature, so the same artifact can
# MAGIC   later back a Model Serving endpoint (last section).
# MAGIC </div>

# COMMAND ----------

# MAGIC %run ./src/00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Engineer features from the historical readings
# MAGIC Two tag-normalised, row-wise features so a *single* model works across all ~60 tags:
# MAGIC - **`band_pos`** — where the value sits in its normal band `(value - low) / (high - low)`; an
# MAGIC   injected fault pushes this well outside `[0, 1]`.
# MAGIC - **`roll_z`** — rolling z-score vs the tag's own recent mean (catches *drift* before it leaves
# MAGIC   the band), mirroring Lab 2A's z-score tier.

# COMMAND ----------

# DBTITLE 1,Step 1: feature engineering (shared by train + inference)
import pyspark.sql.functions as F
from pyspark.sql.window import Window

_w = Window.partitionBy("tag_id").orderBy("reading_ts").rowsBetween(-12, -1)

def add_features(df):
    """Add band_pos + roll_z. df must carry value, normal_low, normal_high, reading_ts, tag_id."""
    band = (F.col("normal_high") - F.col("normal_low"))
    return (df
        .withColumn("band_pos", (F.col("value") - F.col("normal_low")) / band)
        .withColumn("_rmean", F.avg("value").over(_w))
        .withColumn("_rstd",  F.stddev("value").over(_w))
        .withColumn("roll_z", F.when(F.col("_rstd") > 0,
                                     (F.col("value") - F.col("_rmean")) / F.col("_rstd"))
                              .otherwise(F.lit(0.0)))
        .drop("_rmean", "_rstd"))

FEATURES = ["band_pos", "roll_z"]

hist = (spark.table(BREW_FACT_READINGS)
        .join(spark.table(BREW_DIM_TAG).select("tag_id", "normal_low", "normal_high"), "tag_id"))
hist_feat = add_features(hist).select("reading_ts", "tag_id", *FEATURES).dropna(subset=FEATURES)
print(f"Historical feature rows: {hist_feat.count():,}")
display(hist_feat.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Train Isolation Forest and log + register to Unity Catalog
# MAGIC Isolation Forest is an unsupervised outlier detector — perfect when most history is "normal"
# MAGIC and anomalies are rare. We log it with `mlflow.sklearn`, attach a signature, and register it as
# MAGIC a **UC model** so it's versioned and governed (and servable later).

# COMMAND ----------

# DBTITLE 1,Step 2: train + log + register (UC)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import IsolationForest

mlflow.set_registry_uri("databricks-uc")
MODEL_NAME = brew_t("brewery_anomaly_iforest")   # ml_workshop.brewery.brewery_anomaly_iforest

# Train on a sample of history (mostly-normal) — IForest learns the normal envelope.
train_pdf = hist_feat.select(*FEATURES).sample(False, 0.25, seed=42).toPandas()
print(f"Training rows: {len(train_pdf):,}")

with mlflow.start_run(run_name="brewery_anomaly_iforest") as run:
    model = IsolationForest(n_estimators=150, contamination=0.02, random_state=42)
    model.fit(train_pdf[FEATURES])
    preds = model.predict(train_pdf[FEATURES])            # -1 = anomaly, 1 = normal
    signature = infer_signature(train_pdf[FEATURES], preds)
    mlflow.log_params({"n_estimators": 150, "contamination": 0.02, "features": ",".join(FEATURES)})
    mlflow.log_metric("train_anomaly_rate", float((preds == -1).mean()))
    info = mlflow.sklearn.log_model(
        model, artifact_path="model", signature=signature,
        input_example=train_pdf[FEATURES].head(3),
        registered_model_name=MODEL_NAME)
    run_id = run.info.run_id

# Promote the version we just registered to the 'champion' alias for a stable load URI.
client = MlflowClient(registry_uri="databricks-uc")
ver = max(int(mv.version) for mv in client.search_model_versions(f"name='{MODEL_NAME}'"))
client.set_registered_model_alias(MODEL_NAME, "champion", ver)
print(f"Registered {MODEL_NAME} v{ver} (alias: champion)  ·  run {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Inference on the LIVE streamed data
# MAGIC Load the registered model and score `live_silver_readings` — the table the **running** Auto
# MAGIC Loader pipeline is filling. We build the *same* features, apply the model as a Spark UDF, and
# MAGIC write the model-flagged anomalies to a gold table.

# COMMAND ----------

# DBTITLE 1,Step 3: score the live pipeline output with the registered model
LIVE_SILVER = brew_t("live_silver_readings")
BREW_GOLD_MODEL = brew_t("gold_model_anomalies")

live_feat = add_features(spark.table(LIVE_SILVER)).dropna(subset=FEATURES)

predict_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{MODEL_NAME}@champion", result_type="double")
scored = live_feat.withColumn("anomaly_pred", predict_udf(F.struct(*[F.col(c) for c in FEATURES])))

model_anoms = (scored.where("anomaly_pred = -1")
               .select("reading_ts", "tag_id", "asset_id", "area", "value",
                       F.round("band_pos", 3).alias("band_pos"),
                       F.round("roll_z", 2).alias("roll_z"), "anomaly_pred"))
(model_anoms.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(BREW_GOLD_MODEL))

n_live = spark.table(LIVE_SILVER).count()
n_anom = spark.table(BREW_GOLD_MODEL).count()
print(f"Scored {n_live:,} live readings → {n_anom:,} model-flagged anomalies → {BREW_GOLD_MODEL}")
display(spark.table(BREW_GOLD_MODEL).orderBy(F.desc("reading_ts")).limit(20))

# COMMAND ----------

# DBTITLE 1,Step 3b: did the model catch the live chiller-drift fault?
display(spark.sql(f"""
  SELECT tag_id, COUNT(*) AS model_flags, ROUND(MAX(value),1) AS max_value
  FROM {BREW_GOLD_MODEL}
  WHERE tag_id = 'GLY.GLY01.RETURN_TEMP'
  GROUP BY tag_id
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 (optional) — Serve the model on a Model Serving endpoint
# MAGIC The registered UC model can back a real-time endpoint for online inference. This is gated off by
# MAGIC default (endpoint provisioning takes a few minutes); flip `DEPLOY_ENDPOINT = True` to create it,
# MAGIC then query it with feature rows.

# COMMAND ----------

# DBTITLE 1,Step 4: deploy + query a serving endpoint (optional)
DEPLOY_ENDPOINT = False  # set True to provision a scale-to-zero serving endpoint

if DEPLOY_ENDPOINT:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (EndpointCoreConfigInput, ServedEntityInput)
    w = WorkspaceClient()
    ep_name = "brewery-anomaly-iforest"
    w.serving_endpoints.create_and_wait(
        name=ep_name,
        config=EndpointCoreConfigInput(served_entities=[ServedEntityInput(
            entity_name=MODEL_NAME, entity_version=str(ver),
            workload_size="Small", scale_to_zero_enabled=True)]))
    resp = w.serving_endpoints.query(ep_name, dataframe_records=[
        {"band_pos": 0.5, "roll_z": 0.1},     # normal
        {"band_pos": 2.4, "roll_z": 6.0},     # anomalous
    ])
    print("endpoint predictions:", resp.predictions)
else:
    print("Serving endpoint skipped (DEPLOY_ENDPOINT=False). Model is registered in UC and ready to serve.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Wrap-up
# MAGIC You trained an **Isolation Forest** on the historical readings, **logged + registered it to Unity
# MAGIC Catalog** (versioned, alias `champion`), then ran **inference on the live pipeline output**,
# MAGIC writing `gold_model_anomalies`. Same artifact can back a **Model Serving endpoint** (Step 4) for
# MAGIC real-time scoring — the ML upgrade to Lab 2A's SQL rules.
