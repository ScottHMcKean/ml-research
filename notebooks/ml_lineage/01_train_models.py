# Databricks notebook source

# MAGIC %md
# MAGIC # ML Lineage — 01 · Train & Register Two Models
# MAGIC
# MAGIC We register **two** models to Unity Catalog so the lineage investigation can compare them:
# MAGIC
# MAGIC | Model | How it's trained | Model lineage UC captures |
# MAGIC |-------|------------------|---------------------------|
# MAGIC | **`model_plain`** | features read inline from `lineage_source`; logged with `mlflow.log_input` | training table → model (via `log_input`) |
# MAGIC | **`model_fe`** | `FeatureEngineeringClient` training set + `fe.log_model` | feature table → model (automatic) |
# MAGIC
# MAGIC Neither approach, as we'll prove in notebook `03`, produces a **model → prediction-table** edge.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "shm_catalog")
dbutils.widgets.text("schema", "ml")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.lineage_source"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.lineage_features"

MODEL_PLAIN = f"{CATALOG}.{SCHEMA}.model_plain"
MODEL_FE = f"{CATALOG}.{SCHEMA}.model_fe"

LABEL = "churned"
KEY = "customer_id"
PLAIN_FEATURES = ["tenure", "monthly_charges", "num_products"]

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model A — plain model (no Feature Engineering)
# MAGIC Standard MLflow flow: pull training data to pandas, fit a scikit-learn classifier,
# MAGIC and log it. We attach the training data with **`mlflow.log_input`** using a Delta
# MAGIC dataset — this is what gives UC the *training* **table → model** lineage. (Note this is
# MAGIC lineage into the model at **train** time; it says nothing about the model's *outputs*.)

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

pdf = spark.read.table(SOURCE_TABLE).select(KEY, *PLAIN_FEATURES, LABEL).toPandas()
X_train, X_test, y_train, y_test = train_test_split(
    pdf[PLAIN_FEATURES], pdf[LABEL], test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="model_plain") as run:
    # Delta dataset -> UC training lineage (table -> model).
    train_dataset = mlflow.data.load_delta(table_name=SOURCE_TABLE, name="lineage_source")
    mlflow.log_input(train_dataset, context="training")

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    mlflow.log_metric("test_auc", auc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_PLAIN,
        input_example=X_train.head(3),
    )
    plain_run_id = run.info.run_id

print(f"model_plain test AUC = {auc:.3f}")

# Alias the version we just registered as @champion.
plain_version = max(
    client.search_model_versions(f"name='{MODEL_PLAIN}'"), key=lambda v: int(v.version)
)
client.set_registered_model_alias(MODEL_PLAIN, "champion", plain_version.version)
print(f"{MODEL_PLAIN} v{plain_version.version} -> @champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model B — Feature Engineering model
# MAGIC Here we build a **training set** with `FeatureLookup`s against the governed feature
# MAGIC table, train on it, and register with **`fe.log_model`**. Because the model is logged
# MAGIC *with* its feature metadata, UC records **feature table → model** lineage automatically,
# MAGIC and `fe.score_batch` (notebook `02`) will re-join those features by key at inference.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

FE_FEATURES = ["tenure", "monthly_charges", "num_products", "total_charges"]

# Spine = keys + label from the source table; features are looked up from the feature table.
spine_df = spark.read.table(SOURCE_TABLE).select(KEY, LABEL)

training_set = fe.create_training_set(
    df=spine_df,
    feature_lookups=[
        FeatureLookup(table_name=FEATURE_TABLE, lookup_key=KEY, feature_names=FE_FEATURES),
    ],
    label=LABEL,
    exclude_columns=[KEY],
)

training_pdf = training_set.load_df().toPandas()
Xf = training_pdf[FE_FEATURES]
yf = training_pdf[LABEL]
Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="model_fe") as run:
    fe_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    fe_model.fit(Xf_train, yf_train)
    fe_auc = roc_auc_score(yf_test, fe_model.predict_proba(Xf_test)[:, 1])
    mlflow.log_metric("test_auc", fe_auc)

    fe.log_model(
        model=fe_model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=MODEL_FE,
    )
    fe_run_id = run.info.run_id

print(f"model_fe test AUC = {fe_auc:.3f}")

fe_version = max(client.search_model_versions(f"name='{MODEL_FE}'"), key=lambda v: int(v.version))
client.set_registered_model_alias(MODEL_FE, "champion", fe_version.version)
print(f"{MODEL_FE} v{fe_version.version} -> @champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next
# MAGIC Continue to **`02_batch_inference.py`** to score both models and write prediction tables.
