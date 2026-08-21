# Databricks notebook source

# MAGIC %md
# MAGIC # Serverless GPU + UC Volumes — 02 · Recommended loading patterns
# MAGIC
# MAGIC **Run this notebook on Serverless GPU compute.**
# MAGIC
# MAGIC Four patterns that work on Serverless GPU, mapped to the customer's questions:
# MAGIC 1. **List files** in a Volume — SDK Files API (replaces `os.listdir`).
# MAGIC 2. **Load a dataset** — SDK download into memory (replaces `pandas.read_csv` on `/Volumes`).
# MAGIC 3. **Libraries that need a local path** — download to `/tmp` first, then pass the path.
# MAGIC 4. **Load a model** — MLflow load-by-URI (preferred) and the raw-artifact fallback.
# MAGIC 5. **Training data at scale** — `UCVolumeDataset` + Serverless GPU `DataLoader`.

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
CSV_PATH = f"{DATA_DIR}/train.csv"
MODEL_PATH = f"{MODEL_DIR}/churn_model.pkl"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · List files in a Volume (replaces `os.listdir`)
# MAGIC The Databricks SDK Files API reads through UC governance, so it works on Serverless GPU.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

for entry in w.files.list_directory_contents(f"{DATA_DIR}/"):
    print(f"{'DIR ' if entry.is_directory else 'FILE'}  "
          f"{entry.path}  ({entry.file_size} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Load a dataset into memory (replaces `pandas.read_csv` on `/Volumes`)
# MAGIC Download the bytes and hand them to pandas via a buffer — no local path required.

# COMMAND ----------

import io
import pandas as pd

resp = w.files.download(CSV_PATH)
df = pd.read_csv(io.BytesIO(resp.contents.read()))
print(f"Loaded {len(df)} rows via the Files API")
display(df.head())

# For tabular data you can also just use Spark and convert:
#   df = spark.read.option("header", True).csv(CSV_PATH).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Libraries that require a LOCAL filesystem path
# MAGIC When a library will only take a path string (not bytes), stage the file to a writable
# MAGIC local scratch directory, then pass that path. This is the general fix for the customer's
# MAGIC "libraries cannot find the file" case.
# MAGIC
# MAGIC **Use `tempfile.gettempdir()` (`/tmp`), not `/local_disk0`** — verified that `/local_disk0`
# MAGIC is *not* writable on serverless (raises `PermissionError`). `/tmp` works on both serverless
# MAGIC and serverless GPU.

# COMMAND ----------

from pathlib import Path
import tempfile
import pickle

LOCAL_SCRATCH = Path(tempfile.gettempdir())          # /tmp — writable on serverless & GPU
local_model = LOCAL_SCRATCH / "churn_model.pkl"
with local_model.open("wb") as f:
    f.write(w.files.download(MODEL_PATH).contents.read())
print(f"Staged model to {local_model}  ({local_model.stat().st_size} bytes)")

# The library now gets a real local path it can open directly
with local_model.open("rb") as f:
    model = pickle.load(f)

import numpy as np
sample = np.array([[12, 95.0, 2]])
print("prediction:", model.predict(sample), "  proba:", model.predict_proba(sample).round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Helper: download a whole Volume directory to local disk
# MAGIC Some libraries expect a directory (e.g. a HF checkpoint folder). List + download each file.

# COMMAND ----------

import os

def download_volume_dir(w: WorkspaceClient, volume_dir: str, local_dir: str) -> str:
    """Copy every file under a Volume directory to local disk; returns local_dir."""
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    for entry in w.files.list_directory_contents(volume_dir.rstrip("/") + "/"):
        if entry.is_directory:
            sub = Path(entry.path).name
            download_volume_dir(w, entry.path, str(Path(local_dir) / sub))
        else:
            dst = Path(local_dir) / Path(entry.path).name
            with dst.open("wb") as f:
                f.write(w.files.download(entry.path).contents.read())
    return local_dir

staged = download_volume_dir(w, MODEL_DIR, str(LOCAL_SCRATCH / "models"))
print("Staged directory contents:", os.listdir(staged))   # os.listdir on LOCAL disk works fine

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Loading a MODEL — prefer MLflow load-by-URI
# MAGIC If the model is registered in Unity Catalog, don't touch the Volume path at all — MLflow
# MAGIC stages the artifacts to local disk for you and works on Serverless GPU.
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC model = mlflow.pyfunc.load_model(f"models:/{CATALOG}.{SCHEMA}.churn_model@champion")
# MAGIC ```
# MAGIC
# MAGIC Only use the raw-artifact download from step 3 when the model is a plain file sitting in a
# MAGIC Volume rather than a registered model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Training data at scale — `UCVolumeDataset`
# MAGIC For image/audio/text training sets, this is the first-class Serverless GPU pattern. It
# MAGIC streams files from the Volume, **caches to local disk on first access**, and **partitions
# MAGIC files across `torch.distributed` ranks and DataLoader workers**. Build the dataset inside
# MAGIC the `@distributed` run and keep `num_workers` identical across ranks.

# COMMAND ----------

# NOTE: requires image files at f"{VOLUME_ROOT}/images". Shown as the canonical pattern.
try:
    from serverless_gpu.data import UCVolumeDataset, DataLoader
    from torch.utils.data import IterableDataset
    from PIL import Image
    import torchvision.transforms.functional as TF

    class ImageDataset(IterableDataset):
        def __init__(self, path_dataset: UCVolumeDataset):
            self._path_dataset = path_dataset
        def __iter__(self):
            for local_path in self._path_dataset:      # a LOCAL cached path, not /Volumes
                img = Image.open(local_path).convert("RGB")
                yield TF.to_tensor(img)

    path_dataset = UCVolumeDataset(f"{VOLUME_ROOT}/images")
    loader = DataLoader(ImageDataset(path_dataset), batch_size=32, pin_memory=True)
    print("UCVolumeDataset + DataLoader constructed:", loader)
except ImportError as e:
    print("serverless_gpu not importable here (run on Serverless GPU AI Runtime):", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC | Customer question | Recommended API on Serverless GPU |
# MAGIC |---|---|
# MAGIC | Load a dataset | SDK `w.files.download` → pandas, or Spark `.toPandas()`; `UCVolumeDataset` for large training sets |
# MAGIC | Use `UCVolumeDataset` for governed file access | `serverless_gpu.data.UCVolumeDataset` + `DataLoader` (auto local cache + rank/worker partitioning) |
# MAGIC | Libraries needing a local path | Download to `/tmp` (`tempfile.gettempdir()`) first, then pass that path |
# MAGIC | List files in a Volume | `w.files.list_directory_contents(...)` or SQL `LIST` |
# MAGIC | Load a model | MLflow `load_model("models:/...")`; raw-file fallback = download to local disk |
