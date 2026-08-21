# Databricks notebook source

# MAGIC %md
# MAGIC # Serverless GPU + UC Volumes — 01 · Reproduce the reported behavior
# MAGIC
# MAGIC **Run this notebook on Serverless GPU compute.**
# MAGIC
# MAGIC We reproduce exactly what the customer described:
# MAGIC - local Python (`os.listdir`, `open`, `pandas.read_csv`) against `/Volumes/...` **fails**,
# MAGIC - while **Spark** and **SQL** read the same files **fine**.
# MAGIC
# MAGIC The takeaway: Serverless GPU does not expose the `/Volumes` **POSIX FUSE mount** to local
# MAGIC libraries. UC-governed access paths (Spark, SQL) are unaffected. This is expected.

# COMMAND ----------

dbutils.widgets.text("catalog", "shm_catalog")
dbutils.widgets.text("schema", "ml")
dbutils.widgets.text("volume", "gpu_volume_demo")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
DATA_DIR = f"{VOLUME_ROOT}/data"
CSV_PATH = f"{DATA_DIR}/train.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confirm we are actually on a GPU
# MAGIC Sanity check so the reproduction is unambiguous.

# COMMAND ----------

import subprocess
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout)
except Exception as e:
    print("nvidia-smi not available:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ❌ Local filesystem access — expected to FAIL on Serverless GPU
# MAGIC Each cell is wrapped so the notebook keeps running and prints what actually happened.

# COMMAND ----------

import os

# 1) os.listdir — customer reports this returns nothing / errors
try:
    entries = os.listdir(DATA_DIR)
    print(f"os.listdir({DATA_DIR}) -> {entries!r}"
          f"{'   <-- EMPTY (no FUSE mount)' if not entries else ''}")
except Exception as e:
    print(f"os.listdir FAILED: {type(e).__name__}: {e}")

# 2) os.path.exists — the path looks absent to local Python
print(f"os.path.exists({CSV_PATH}) -> {os.path.exists(CSV_PATH)}")

# COMMAND ----------

# 3) builtin open() — expected FileNotFoundError
try:
    with open(CSV_PATH) as f:
        print(f.readline())
except Exception as e:
    print(f"open() FAILED: {type(e).__name__}: {e}")

# COMMAND ----------

# 4) pandas.read_csv on the /Volumes path — expected to fail
import pandas as pd
try:
    df = pd.read_csv(CSV_PATH)
    print(df.head())
except Exception as e:
    print(f"pd.read_csv('{CSV_PATH}') FAILED: {type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Governed access — expected to WORK on Serverless GPU
# MAGIC Same files, reached through Unity Catalog instead of a POSIX mount.

# COMMAND ----------

# Spark reads the CSV fine
sdf = spark.read.option("header", True).option("inferSchema", True).csv(CSV_PATH)
print(f"Spark read {sdf.count()} rows from {CSV_PATH}")
display(sdf.limit(5))

# COMMAND ----------

# SQL LIST + read_files also work
display(spark.sql(f"LIST '{DATA_DIR}'"))
display(spark.sql(f"SELECT * FROM read_files('{CSV_PATH}', format => 'csv', header => true) LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verdict
# MAGIC If the local cells above failed while the Spark/SQL cells succeeded, you have reproduced
# MAGIC the reported behavior. It is **expected** on Serverless GPU: no `/Volumes` FUSE mount for
# MAGIC local libraries.
# MAGIC
# MAGIC Continue to **`02_recommended_loading.py`** for the patterns that *do* work — including how
# MAGIC to hand a real local path to libraries that demand one.
