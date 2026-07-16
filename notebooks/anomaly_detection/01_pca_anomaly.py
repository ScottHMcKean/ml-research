# Databricks notebook source

# MAGIC %md
# MAGIC # Rare-Failure Anomaly Detection — PCA Reconstruction Error
# MAGIC **Approach #1.** Learn the normal operating manifold with **PCA**, then score each
# MAGIC reading by how badly it is **reconstructed** from the top principal components.
# MAGIC Normal readings live near the low-dimensional manifold (small error); a failure
# MAGIC pushes sensors off it (large error).
# MAGIC
# MAGIC This is **unsupervised** — PCA never sees a label. The **two observed failures**
# MAGIC (`is_labeled = 1`) are used for one thing only: to **calibrate the decision
# MAGIC threshold**. We show this beats the usual "flag the top 1%" percentile rule.
# MAGIC
# MAGIC Everything is tracked in **MLflow** and the calibrated detector is registered to
# MAGIC Unity Catalog as a pyfunc.

# COMMAND ----------

# MAGIC %pip install -q scikit-learn matplotlib mlflow
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
)
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

CATALOG = "shm_skunkworks_catalog"
SCHEMA = "anomaly_detection"
TABLE = f"{CATALOG}.{SCHEMA}.pump_sensors"
EXPERIMENT = "/Shared/anomaly_detection"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.pca_anomaly"

SENSORS = [
    "vibration_rms", "vibration_peak", "bearing_temp", "motor_temp",
    "motor_current", "motor_voltage", "flow_rate", "discharge_pressure",
    "suction_pressure", "rpm", "oil_particle_count", "acoustic_db",
]
N_COMPONENTS = 5   # top-k principal components that define "normal"

mlflow.set_experiment(EXPERIMENT)
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data
# MAGIC `X` = sensor matrix. We fit the manifold on the **unlabeled** rows only (we exclude
# MAGIC the two known failures so they don't pollute "normal"). `is_failure` is pulled out
# MAGIC purely for the final held-out evaluation — the model never trains on it.

# COMMAND ----------

pdf = spark.table(TABLE).orderBy("reading_id").toPandas()
X = pdf[SENSORS].to_numpy(dtype=float)
is_failure = pdf["is_failure"].to_numpy()
is_labeled = pdf["is_labeled"].to_numpy() == 1

fit_mask = ~is_labeled                        # unsupervised fit on everything we haven't confirmed bad
labeled_idx = np.where(is_labeled)[0]
print(f"rows={len(pdf)}  fit_rows={fit_mask.sum()}  labeled_anomalies={is_labeled.sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit PCA and compute reconstruction error
# MAGIC Standardize (sensors are on wildly different scales), keep the top
# MAGIC `N_COMPONENTS`, and reconstruct. The **reconstruction error** — squared distance
# MAGIC between a reading and its projection — is the anomaly score.

# COMMAND ----------

scaler = StandardScaler().fit(X[fit_mask])
Xs_all = scaler.transform(X)

pca = PCA(n_components=N_COMPONENTS, random_state=42).fit(Xs_all[fit_mask])
var_explained = float(pca.explained_variance_ratio_.sum())

def recon_error(Xs):
    recon = pca.inverse_transform(pca.transform(Xs))
    return np.mean((Xs - recon) ** 2, axis=1)

scores = recon_error(Xs_all)
print(f"kept {N_COMPONENTS} components, variance explained = {var_explained:.3f}")
print(f"labeled-anomaly scores: {scores[labeled_idx]}")
print(f"normal score p50/p99: {np.percentile(scores, 50):.3f} / {np.percentile(scores, 99):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Threshold: unsupervised percentile vs. few-shot calibrated
# MAGIC The naive rule flags the top 1% of scores — but 1% is arbitrary and usually way
# MAGIC too many alerts. Instead we **use the two observed failures**: set the threshold
# MAGIC just below the *smaller* of their scores (with a small safety margin). This is the
# MAGIC "augment the unsupervised model with a couple of examples" idea — two labels turn
# MAGIC an arbitrary knob into a grounded one.

# COMMAND ----------

# Baseline: pure-unsupervised percentile threshold.
thr_percentile = float(np.percentile(scores, 99.0))

# Few-shot: threshold anchored to the observed failures. Sit just under the lowest
# labeled-anomaly score so both are caught, backed off 10% for margin.
labeled_scores = scores[labeled_idx]
thr_fewshot = float(labeled_scores.min() * 0.90)

def evaluate(threshold):
    pred = (scores >= threshold).astype(int)
    # Evaluate on held-out rows only (exclude the 2 we were given as labels).
    m = ~is_labeled
    return {
        "precision": precision_score(is_failure[m], pred[m], zero_division=0),
        "recall": recall_score(is_failure[m], pred[m], zero_division=0),
        "f1": f1_score(is_failure[m], pred[m], zero_division=0),
        "alert_rate": float(pred[m].mean()),
    }

eval_percentile = evaluate(thr_percentile)
eval_fewshot = evaluate(thr_fewshot)
# Ranking metrics are threshold-free (computed on held-out rows).
held = ~is_labeled
roc_auc = float(roc_auc_score(is_failure[held], scores[held]))
pr_auc = float(average_precision_score(is_failure[held], scores[held]))

print("percentile(99%) threshold:", round(thr_percentile, 3), eval_percentile)
print("few-shot     threshold:", round(thr_fewshot, 3), eval_fewshot)
print(f"ROC-AUC={roc_auc:.3f}  PR-AUC={pr_auc:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log to MLflow and register the calibrated detector to Unity Catalog
# MAGIC The pyfunc bundles the scaler + PCA + calibrated threshold and returns an anomaly
# MAGIC score and a 0/1 flag for each incoming reading.

# COMMAND ----------

class PCAAnomalyDetector(mlflow.pyfunc.PythonModel):
    """Scaler + PCA reconstruction-error detector with a calibrated threshold."""

    def __init__(self, scaler, pca, threshold, sensors):
        self.scaler, self.pca, self.threshold, self.sensors = scaler, pca, threshold, sensors

    def predict(self, context, model_input):
        X = model_input[self.sensors].to_numpy(dtype=float)
        Xs = self.scaler.transform(X)
        recon = self.pca.inverse_transform(self.pca.transform(Xs))
        score = np.mean((Xs - recon) ** 2, axis=1)
        return pd.DataFrame({
            "anomaly_score": score,
            "is_anomaly": (score >= self.threshold).astype(int),
        })


# Diagnostic plot: score distribution with both thresholds + the labeled anomalies.
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(np.log10(scores[is_failure == 0] + 1e-6), bins=60, alpha=0.6, label="normal")
ax.hist(np.log10(scores[is_failure == 1] + 1e-6), bins=30, alpha=0.8, label="failure (hidden truth)")
ax.axvline(np.log10(thr_fewshot), color="green", ls="--", label="few-shot threshold")
ax.axvline(np.log10(thr_percentile), color="red", ls=":", label="99% threshold")
for s in labeled_scores:
    ax.axvline(np.log10(s), color="black", lw=0.8, alpha=0.5)
ax.set_xlabel("log10 reconstruction error"); ax.set_ylabel("count")
ax.set_title("PCA anomaly score — normal vs failure"); ax.legend()
fig.tight_layout()

input_example = pdf[SENSORS].head(3)
detector = PCAAnomalyDetector(scaler, pca, thr_fewshot, SENSORS)
signature = infer_signature(input_example, detector.predict(None, input_example))

with mlflow.start_run(run_name="pca_reconstruction") as run:
    mlflow.log_params({
        "method": "pca_reconstruction",
        "n_components": N_COMPONENTS,
        "n_sensors": len(SENSORS),
        "threshold_strategy": "few_shot_calibrated",
        "n_labeled": int(is_labeled.sum()),
    })
    mlflow.log_metrics({
        "variance_explained": var_explained,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "fewshot_precision": eval_fewshot["precision"],
        "fewshot_recall": eval_fewshot["recall"],
        "fewshot_f1": eval_fewshot["f1"],
        "fewshot_alert_rate": eval_fewshot["alert_rate"],
        "percentile_precision": eval_percentile["precision"],
        "percentile_recall": eval_percentile["recall"],
        "threshold_fewshot": thr_fewshot,
        "threshold_percentile": thr_percentile,
    })
    mlflow.log_figure(fig, "score_distribution.png")
    logged = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=detector,
        signature=signature,
        input_example=input_example,
        registered_model_name=UC_MODEL_NAME,
    )
    run_id = run.info.run_id

version = logged.registered_model_version   # UC registry doesn't support order_by search
MlflowClient().set_registered_model_alias(UC_MODEL_NAME, "champion", version)
print(f"Registered {UC_MODEL_NAME} v{version} @champion  (run {run_id})")
