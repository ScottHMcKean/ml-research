# Databricks notebook source

# MAGIC %md
# MAGIC # ML Lineage — 00 · Setup & Feature Table
# MAGIC
# MAGIC This walkthrough answers a concrete customer question:
# MAGIC
# MAGIC > *When a prediction table is produced by an ML model, does Unity Catalog lineage
# MAGIC > show the relationship between the prediction table and the model that produced it?*
# MAGIC
# MAGIC We build the assets end-to-end, score a model **with** and **without** the Feature
# MAGIC Engineering engine, and then (in notebook `03`) inspect exactly what UC lineage
# MAGIC captures — via SQL on the `system.access` lineage tables and via the lineage REST API.
# MAGIC
# MAGIC This first notebook creates:
# MAGIC - a raw **source table** `lineage_source` (the inference input + labels), and
# MAGIC - a governed **UC feature table** `lineage_features` (used by the Feature Engineering path).

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

N_ROWS = 20_000

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the schema
# MAGIC The **catalog must already exist** — point the `catalog` widget at one you can write to.
# MAGIC (On workspaces with UC *Default Storage*, new catalogs must be created from the Catalog
# MAGIC Explorer UI or with an explicit `MANAGED LOCATION`, so we don't create it here.) The
# MAGIC schema is created idempotently.

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Using {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate a synthetic dataset
# MAGIC A simple binary-classification problem (think "will this customer churn?"). Every row
# MAGIC has a stable **`customer_id`** primary key — that key is what the Feature Engineering
# MAGIC path uses to look features up at scoring time.

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

rng = np.random.default_rng(42)

tenure = rng.integers(1, 72, N_ROWS)                       # months as a customer
monthly_charges = rng.uniform(20, 120, N_ROWS).round(2)    # $/month
num_products = rng.integers(1, 6, N_ROWS)                  # products held
region = rng.choice(["north", "south", "east", "west"], N_ROWS)

# A learnable signal: short tenure + high charges + few products => higher churn risk.
logit = (
    -0.05 * tenure
    + 0.03 * monthly_charges
    - 0.40 * num_products
    + rng.normal(0, 1.0, N_ROWS)
)
prob = 1 / (1 + np.exp(-logit))
label = (prob > 0.5).astype(int)

pdf = pd.DataFrame(
    {
        "customer_id": np.arange(1, N_ROWS + 1, dtype="int64"),
        "region": region,
        "tenure": tenure.astype("int32"),
        "monthly_charges": monthly_charges,
        "num_products": num_products.astype("int32"),
        "churned": label.astype("int32"),
    }
)
print(pdf["churned"].value_counts(normalize=True).round(3).to_dict())
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write the source table
# MAGIC This is a plain managed Delta table in Unity Catalog. It serves two purposes:
# MAGIC - the **plain model** reads its features directly from here, and
# MAGIC - the **Feature Engineering path** uses it as the inference *spine* (keys + label),
# MAGIC   joining engineered features in from the feature table at score time.

# COMMAND ----------

(
    spark.createDataFrame(pdf)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SOURCE_TABLE)
)
print(f"Wrote {SOURCE_TABLE}")
display(spark.read.table(SOURCE_TABLE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a governed UC feature table
# MAGIC We use the **Feature Engineering client** (`create_table`) so `lineage_features` is a
# MAGIC first-class UC feature table with a declared primary key. Registering the model against
# MAGIC this table (notebook `01`) is what produces **feature-table → model** lineage — the one
# MAGIC kind of model lineage UC *does* capture automatically.
# MAGIC
# MAGIC `total_charges` is an engineered column (tenure × monthly spend) that exists **only** in
# MAGIC the feature table, so the FE model genuinely depends on this table at inference time.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

features_pdf = pdf[["customer_id", "tenure", "monthly_charges", "num_products"]].copy()
features_pdf["total_charges"] = (features_pdf["tenure"] * features_pdf["monthly_charges"]).round(2)
features_sdf = spark.createDataFrame(features_pdf)

# Recreate cleanly so the notebook is idempotent.
spark.sql(f"DROP TABLE IF EXISTS {FEATURE_TABLE}")

fe.create_table(
    name=FEATURE_TABLE,
    primary_keys=["customer_id"],
    df=features_sdf,
    description="Engineered customer features for the ML lineage walkthrough.",
)
print(f"Created feature table {FEATURE_TABLE}")
display(spark.read.table(FEATURE_TABLE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next
# MAGIC Continue to **`01_train_models.py`** to train and register the two UC models
# MAGIC (plain + Feature Engineering).
