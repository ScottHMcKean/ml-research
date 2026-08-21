# Databricks notebook source

# MAGIC %md
# MAGIC # Serverless GPU + UC Volumes — 00 · Set up Volume assets
# MAGIC
# MAGIC Creates a Unity Catalog **Volume** and drops two kinds of assets into it so the later
# MAGIC notebooks have something to load:
# MAGIC - a **dataset** (`train.csv` + a small Parquet directory), and
# MAGIC - a **model artifact** (a pickled scikit-learn model).
# MAGIC
# MAGIC Run this on **regular Serverless** (or any UC-enabled compute) — writing to a Volume via
# MAGIC the local path works fine here. The interesting behavior shows up in `01` on Serverless GPU.

# COMMAND ----------

dbutils.widgets.text("catalog", "shm_catalog")
dbutils.widgets.text("schema", "ml")
dbutils.widgets.text("volume", "gpu_volume_demo")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
DATA_DIR = f"{VOLUME_ROOT}/data"
MODEL_DIR = f"{VOLUME_ROOT}/models"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the schema and Volume
# MAGIC The **catalog must already exist** and be writable. The schema and a **managed Volume**
# MAGIC are created idempotently.

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"Volume ready at {VOLUME_ROOT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write a sample dataset
# MAGIC A tiny binary-classification table. We persist it two ways so `01`/`02` can exercise both
# MAGIC a single-file read (`train.csv`) and a directory read (Parquet).

# COMMAND ----------

import os
import numpy as np
import pandas as pd

os.makedirs(DATA_DIR, exist_ok=True)   # local-path write works on regular Serverless

rng = np.random.default_rng(42)
n = 5_000
pdf = pd.DataFrame(
    {
        "tenure": rng.integers(1, 72, n),
        "monthly_charges": rng.uniform(20, 120, n).round(2),
        "num_products": rng.integers(1, 6, n),
    }
)
logit = -0.05 * pdf["tenure"] + 0.03 * pdf["monthly_charges"] - 0.40 * pdf["num_products"]
pdf["churned"] = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)

pdf.to_csv(f"{DATA_DIR}/train.csv", index=False)
print(f"Wrote {DATA_DIR}/train.csv  ({len(pdf)} rows)")

# Parquet directory (multiple part files) via Spark
(
    spark.createDataFrame(pdf)
    .repartition(3)
    .write.mode("overwrite")
    .parquet(f"{DATA_DIR}/train_parquet")
)
print(f"Wrote {DATA_DIR}/train_parquet/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and drop a model artifact into the Volume
# MAGIC A minimal scikit-learn model, pickled to a raw file in the Volume. This mimics the
# MAGIC customer's "load a model from `/Volumes/...`" case (as opposed to loading a registered
# MAGIC MLflow model, which we cover in `02`).

# COMMAND ----------

import pickle
from sklearn.linear_model import LogisticRegression

os.makedirs(MODEL_DIR, exist_ok=True)

X = pdf[["tenure", "monthly_charges", "num_products"]].values
y = pdf["churned"].values
model = LogisticRegression(max_iter=1000).fit(X, y)

with open(f"{MODEL_DIR}/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"Wrote {MODEL_DIR}/churn_model.pkl  (train acc {model.score(X, y):.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confirm the layout
# MAGIC `LIST` is governed SQL and works on every compute — a good sanity check.

# COMMAND ----------

display(spark.sql(f"LIST '{VOLUME_ROOT}'"))
display(spark.sql(f"LIST '{DATA_DIR}'"))
display(spark.sql(f"LIST '{MODEL_DIR}'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next
# MAGIC Switch to **Serverless GPU** compute and run **`01_reproduce_gpu_access.py`** to see the
# MAGIC local-filesystem access behavior the customer reported.
