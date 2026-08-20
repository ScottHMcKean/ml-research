# Databricks notebook source

# MAGIC %md
# MAGIC # ML Lineage — 02 · Batch Inference (both paths)
# MAGIC
# MAGIC We now score both registered models and write two prediction tables:
# MAGIC
# MAGIC | Path | Inference API | Output table |
# MAGIC |------|---------------|--------------|
# MAGIC | **Plain** | `mlflow.pyfunc.spark_udf` over the source table | `pred_plain` |
# MAGIC | **Feature Engineering** | `fe.score_batch` (auto-joins features by key) | `pred_fe` |
# MAGIC
# MAGIC For the plain path we also demonstrate the **recommended workaround**: stamp the model
# MAGIC name / version / run-id into the prediction table so the model→table relationship is
# MAGIC recoverable even though UC lineage won't draw the edge for you.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "shm_catalog")
dbutils.widgets.text("schema", "ml")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.lineage_source"

MODEL_PLAIN = f"{CATALOG}.{SCHEMA}.model_plain"
MODEL_FE = f"{CATALOG}.{SCHEMA}.model_fe"
PLAIN_URI = f"models:/{MODEL_PLAIN}@champion"
FE_URI = f"models:/{MODEL_FE}@champion"

PRED_PLAIN = f"{CATALOG}.{SCHEMA}.pred_plain"
PRED_FE = f"{CATALOG}.{SCHEMA}.pred_fe"

KEY = "customer_id"
PLAIN_FEATURES = ["tenure", "monthly_charges", "num_products"]

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plain path — `load_model` + pandas UDF
# MAGIC Load the champion and score the source table in a distributed pandas UDF. This is the
# MAGIC most common batch-scoring pattern — and the one the customer reported: the resulting
# MAGIC table has **no** lineage edge back to the model.
# MAGIC
# MAGIC > On **classic** ML compute you'd typically use the one-liner
# MAGIC > `mlflow.pyfunc.spark_udf(spark, PLAIN_URI)`. On **Serverless** that helper currently
# MAGIC > errors parsing the runtime version, so we use `load_model` inside a `pandas_udf`
# MAGIC > (equivalent for lineage — Spark still reads the source table and writes the output).

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

plain_version = client.get_model_version_by_alias(MODEL_PLAIN, "champion")

# Load on the driver; the closure ships the (small) model to executors.
plain_model = mlflow.sklearn.load_model(PLAIN_URI)


@pandas_udf(DoubleType())
def plain_predict(*cols: pd.Series) -> pd.Series:
    X = pd.concat(cols, axis=1)
    X.columns = PLAIN_FEATURES
    return pd.Series(plain_model.predict_proba(X)[:, 1])


scored_plain = (
    spark.read.table(SOURCE_TABLE)
    .withColumn("prediction", plain_predict(*[F.col(c) for c in PLAIN_FEATURES]))
    # ---- Workaround: stamp model provenance into the prediction table ----
    .withColumn("model_name", F.lit(MODEL_PLAIN))
    .withColumn("model_version", F.lit(plain_version.version))
    .withColumn("model_run_id", F.lit(plain_version.run_id))
    .withColumn("scored_at", F.current_timestamp())
)

(
    scored_plain.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PRED_PLAIN)
)
print(f"Wrote {PRED_PLAIN} (model_plain v{plain_version.version})")
display(spark.read.table(PRED_PLAIN).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering path — `fe.score_batch`
# MAGIC We pass only the **keys** (and any pass-through columns). `score_batch` reads the model's
# MAGIC packaged feature metadata, looks the features up from `lineage_features` by key, and
# MAGIC scores. The feature table→model dependency is what shows up in Catalog Explorer's model
# MAGIC lineage — but, again, the *output* table below gets no model edge.
# MAGIC
# MAGIC > `fe.score_batch` uses `spark_udf` under the hood, so like the plain path it needs
# MAGIC > **classic ML compute** on this workspace (Serverless hits the version-parsing bug).
# MAGIC > Because `score_batch` physically reads the feature table, that table shows up as an
# MAGIC > upstream source of the prediction table in §2 of the lineage notebook.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

inference_spine = spark.read.table(SOURCE_TABLE).select(KEY, "region")

scored_fe = fe.score_batch(model_uri=FE_URI, df=inference_spine)

fe_version = client.get_model_version_by_alias(MODEL_FE, "champion")
scored_fe = (
    scored_fe
    .withColumn("model_name", F.lit(MODEL_FE))
    .withColumn("model_version", F.lit(fe_version.version))
    .withColumn("model_run_id", F.lit(fe_version.run_id))
    .withColumn("scored_at", F.current_timestamp())
)

(
    scored_fe.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PRED_FE)
)
print(f"Wrote {PRED_FE} (model_fe v{fe_version.version})")
display(spark.read.table(PRED_FE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next
# MAGIC Continue to **`03_lineage_investigation.py`** to inspect what UC lineage actually captured.
