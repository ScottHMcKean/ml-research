# Databricks notebook source

# MAGIC %md
# MAGIC # Rare-Failure Anomaly Detection — Compare PCA vs AutoEncoder
# MAGIC Loads both registered detectors from Unity Catalog, scores the full sensor table,
# MAGIC and compares them on the **hidden ground truth** with threshold-free ranking
# MAGIC metrics (ROC-AUC, PR-AUC) and per-failure-mode recall. Writes a batch-scored
# MAGIC table with the winner and logs the comparison to MLflow.

# COMMAND ----------

# MAGIC %pip install -q torch scikit-learn matplotlib mlflow
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

CATALOG = "shm_skunkworks_catalog"
SCHEMA = "anomaly_detection"
TABLE = f"{CATALOG}.{SCHEMA}.pump_sensors"
SCORED_TABLE = f"{CATALOG}.{SCHEMA}.pump_scored"
EXPERIMENT = "/Shared/anomaly_detection"

SENSORS = [
    "vibration_rms", "vibration_peak", "bearing_temp", "motor_temp",
    "motor_current", "motor_voltage", "flow_rate", "discharge_pressure",
    "suction_pressure", "rpm", "oil_particle_count", "acoustic_db",
]

mlflow.set_experiment(EXPERIMENT)
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

pdf = spark.table(TABLE).orderBy("reading_id").toPandas()
y = pdf["is_failure"].to_numpy()
held = pdf["is_labeled"].to_numpy() == 0        # evaluate on held-out rows

models = {
    "pca": mlflow.pyfunc.load_model(f"models:/{CATALOG}.{SCHEMA}.pca_anomaly@champion"),
    "autoencoder": mlflow.pyfunc.load_model(f"models:/{CATALOG}.{SCHEMA}.autoencoder_anomaly@champion"),
}

preds = {name: m.predict(pdf[SENSORS]) for name, m in models.items()}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ranking metrics + per-mode recall

# COMMAND ----------

rows = []
for name, out in preds.items():
    s = out["anomaly_score"].to_numpy()
    rows.append({
        "model": name,
        "roc_auc": roc_auc_score(y[held], s[held]),
        "pr_auc": average_precision_score(y[held], s[held]),
        "recall_at_flag": out["is_anomaly"].to_numpy()[held & (y == 1)].mean(),
        "alert_rate": out["is_anomaly"].to_numpy()[held].mean(),
    })
summary = pd.DataFrame(rows).set_index("model")
print(summary.round(3))

# Per-failure-mode recall at each model's calibrated threshold.
mode_recall = {}
for name, out in preds.items():
    flag = out["is_anomaly"].to_numpy()
    mode_recall[name] = (
        pd.DataFrame({"mode": pdf["failure_mode"], "flag": flag})
        .query("mode != 'normal'").groupby("mode")["flag"].mean()
    )
mode_recall = pd.DataFrame(mode_recall).round(2)
print("\nper-mode recall at calibrated threshold:\n", mode_recall)

champion = summary["pr_auc"].idxmax()
print(f"\nChampion by PR-AUC: {champion}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PR curves + write batch-scored table

# COMMAND ----------

fig, ax = plt.subplots(figsize=(7, 5))
for name, out in preds.items():
    p, r, _ = precision_recall_curve(y[held], out["anomaly_score"].to_numpy()[held])
    ax.plot(r, p, label=f"{name} (PR-AUC={summary.loc[name,'pr_auc']:.3f})")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("PCA vs AutoEncoder — precision-recall"); ax.legend()
fig.tight_layout()

scored = pdf[["reading_id", "event_time", "unit_id", "is_failure", "failure_mode"]].copy()
for name, out in preds.items():
    scored[f"{name}_score"] = out["anomaly_score"].to_numpy()
    scored[f"{name}_flag"] = out["is_anomaly"].to_numpy()

spark.createDataFrame(scored).write.mode("overwrite").option(
    "overwriteSchema", "true").saveAsTable(SCORED_TABLE)

with mlflow.start_run(run_name="compare_pca_vs_autoencoder"):
    for name in summary.index:
        for col in summary.columns:
            mlflow.log_metric(f"{name}_{col}", float(summary.loc[name, col]))
    mlflow.log_param("champion", champion)
    mlflow.log_figure(fig, "precision_recall.png")

print(f"Wrote {SCORED_TABLE}. Champion = {champion}.")
display(spark.sql(f"SELECT * FROM {SCORED_TABLE} WHERE is_failure = 1 ORDER BY reading_id"))
