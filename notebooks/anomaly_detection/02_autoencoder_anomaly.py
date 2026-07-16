# Databricks notebook source

# MAGIC %md
# MAGIC # Rare-Failure Anomaly Detection — AutoEncoder Reconstruction Error
# MAGIC **Approach #2.** Same idea as the PCA notebook — score by reconstruction error —
# MAGIC but the manifold is learned by an **undercomplete autoencoder** (a small
# MAGIC PyTorch MLP) instead of a linear projection. The AE captures *non-linear* sensor
# MAGIC interactions PCA can't, which matters for modes like cavitation and imbalance.
# MAGIC
# MAGIC ### How the two observed failures are used (few-shot augmentation)
# MAGIC The AE trains unsupervised on the unlabeled data, **plus a margin penalty** that
# MAGIC forces the two labeled failures to reconstruct *badly*. Without it, a flexible AE
# MAGIC can accidentally learn to reconstruct anomalies too (they're in the training mix),
# MAGIC blunting the score. Two labels are enough to keep the anomalies off the manifold.
# MAGIC The same two labels then **calibrate the threshold**, exactly as in the PCA notebook.
# MAGIC
# MAGIC Tracked in **MLflow**; registered to Unity Catalog as a pyfunc.

# COMMAND ----------

# MAGIC %pip install -q torch scikit-learn matplotlib mlflow
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
)
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

CATALOG = "shm_skunkworks_catalog"
SCHEMA = "anomaly_detection"
TABLE = f"{CATALOG}.{SCHEMA}.pump_sensors"
EXPERIMENT = "/Shared/anomaly_detection"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.autoencoder_anomaly"

SENSORS = [
    "vibration_rms", "vibration_peak", "bearing_temp", "motor_temp",
    "motor_current", "motor_voltage", "flow_rate", "discharge_pressure",
    "suction_pressure", "rpm", "oil_particle_count", "acoustic_db",
]
BOTTLENECK = 3       # latent dimension (< 12 → undercomplete)
EPOCHS = 120
BATCH = 512
LR = 1e-3
FEWSHOT_MARGIN = 4.0     # labeled anomalies must reconstruct at least this badly (std units)
FEWSHOT_LAMBDA = 1.0     # weight of the few-shot margin penalty

torch.manual_seed(42)
np.random.seed(42)
mlflow.set_experiment(EXPERIMENT)
mlflow.set_registry_uri("databricks-uc")

# This task runs on a serverless GPU (A10) node — train on CUDA when available.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__}  device={DEVICE}  "
      f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data
# MAGIC Fit the manifold on the unlabeled rows; hold the two labeled anomalies aside for
# MAGIC the margin penalty. `is_failure` is used only for the final held-out evaluation.

# COMMAND ----------

pdf = spark.table(TABLE).orderBy("reading_id").toPandas()
X = pdf[SENSORS].to_numpy(dtype=float)
is_failure = pdf["is_failure"].to_numpy()
is_labeled = pdf["is_labeled"].to_numpy() == 1

fit_mask = ~is_labeled
labeled_idx = np.where(is_labeled)[0]

scaler = StandardScaler().fit(X[fit_mask])
Xs_all = scaler.transform(X).astype(np.float32)
X_train = torch.tensor(Xs_all[fit_mask]).to(DEVICE)
X_anom = torch.tensor(Xs_all[labeled_idx]).to(DEVICE)     # the 2 observed failures
print(f"train_rows={len(X_train)}  labeled_anomalies={len(X_anom)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define and train the autoencoder
# MAGIC Architecture `12 → 8 → 3 → 8 → 12`. The loss is mean reconstruction error on the
# MAGIC unlabeled batch **plus** `λ · relu(margin − error)` on the labeled anomalies —
# MAGIC a hinge that keeps their reconstruction error above `margin`.

# COMMAND ----------

class AutoEncoder(nn.Module):
    def __init__(self, n_in, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 8), nn.ReLU(),
            nn.Linear(8, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 8), nn.ReLU(),
            nn.Linear(8, n_in),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def row_error(net, x):
    return ((net(x) - x) ** 2).mean(dim=1)


net = AutoEncoder(len(SENSORS), BOTTLENECK).to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=LR)
n = len(X_train)
loss_hist = []

for epoch in range(EPOCHS):
    net.train()
    perm = torch.randperm(n)
    epoch_loss = 0.0
    for i in range(0, n, BATCH):
        xb = X_train[perm[i:i + BATCH]]
        opt.zero_grad()
        recon_loss = row_error(net, xb).mean()
        # Few-shot margin: penalize anomalies that reconstruct *too well*.
        anom_err = row_error(net, X_anom)
        margin_loss = torch.relu(FEWSHOT_MARGIN - anom_err).mean()
        loss = recon_loss + FEWSHOT_LAMBDA * margin_loss
        loss.backward()
        opt.step()
        epoch_loss += recon_loss.item() * len(xb)
    loss_hist.append(epoch_loss / n)
    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch+1:3d}  recon_mse={loss_hist[-1]:.4f}  anom_err={anom_err.mean().item():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score, calibrate threshold, evaluate

# COMMAND ----------

net.eval()
with torch.no_grad():
    scores = row_error(net, torch.tensor(Xs_all).to(DEVICE)).cpu().numpy()

labeled_scores = scores[labeled_idx]
thr_fewshot = float(labeled_scores.min() * 0.90)          # few-shot calibrated threshold
thr_percentile = float(np.percentile(scores, 99.0))       # naive baseline

def evaluate(threshold):
    pred = (scores >= threshold).astype(int)
    m = ~is_labeled
    return {
        "precision": precision_score(is_failure[m], pred[m], zero_division=0),
        "recall": recall_score(is_failure[m], pred[m], zero_division=0),
        "f1": f1_score(is_failure[m], pred[m], zero_division=0),
        "alert_rate": float(pred[m].mean()),
    }

held = ~is_labeled
eval_fewshot = evaluate(thr_fewshot)
roc_auc = float(roc_auc_score(is_failure[held], scores[held]))
pr_auc = float(average_precision_score(is_failure[held], scores[held]))
print("few-shot threshold:", round(thr_fewshot, 3), eval_fewshot)
print(f"ROC-AUC={roc_auc:.3f}  PR-AUC={pr_auc:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log to MLflow and register to Unity Catalog

# COMMAND ----------

class AEAnomalyDetector(mlflow.pyfunc.PythonModel):
    """Scaler + trained autoencoder with a calibrated threshold."""

    def __init__(self, net, scaler, threshold, sensors):
        self.net = net.eval()
        self.scaler, self.threshold, self.sensors = scaler, threshold, sensors

    def predict(self, context, model_input):
        X = model_input[self.sensors].to_numpy(dtype=np.float32)
        Xs = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            xt = torch.tensor(Xs)
            score = ((self.net(xt) - xt) ** 2).mean(dim=1).numpy()
        return pd.DataFrame({
            "anomaly_score": score,
            "is_anomaly": (score >= self.threshold).astype(int),
        })


fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(loss_hist); axes[0].set_title("training reconstruction MSE")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE")
axes[1].hist(np.log10(scores[is_failure == 0] + 1e-6), bins=60, alpha=0.6, label="normal")
axes[1].hist(np.log10(scores[is_failure == 1] + 1e-6), bins=30, alpha=0.8, label="failure (hidden truth)")
axes[1].axvline(np.log10(thr_fewshot), color="green", ls="--", label="few-shot threshold")
axes[1].set_xlabel("log10 reconstruction error"); axes[1].set_title("AE anomaly score"); axes[1].legend()
fig.tight_layout()

input_example = pdf[SENSORS].head(3)
detector = AEAnomalyDetector(net.cpu(), scaler, thr_fewshot, SENSORS)
signature = infer_signature(input_example, detector.predict(None, input_example))

with mlflow.start_run(run_name="autoencoder_reconstruction") as run:
    mlflow.log_params({
        "method": "autoencoder_reconstruction",
        "bottleneck": BOTTLENECK, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "fewshot_margin": FEWSHOT_MARGIN, "fewshot_lambda": FEWSHOT_LAMBDA,
        "n_sensors": len(SENSORS), "n_labeled": int(is_labeled.sum()),
        "threshold_strategy": "few_shot_calibrated",
        "device": str(DEVICE),
    })
    mlflow.log_metrics({
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "fewshot_precision": eval_fewshot["precision"],
        "fewshot_recall": eval_fewshot["recall"],
        "fewshot_f1": eval_fewshot["f1"],
        "fewshot_alert_rate": eval_fewshot["alert_rate"],
        "threshold_fewshot": thr_fewshot,
        "final_train_mse": loss_hist[-1],
    })
    mlflow.log_figure(fig, "training_and_scores.png")
    logged = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=detector,
        signature=signature,
        input_example=input_example,
        pip_requirements=["torch", "scikit-learn", "pandas", "numpy"],
        registered_model_name=UC_MODEL_NAME,
    )
    run_id = run.info.run_id

version = logged.registered_model_version   # UC registry doesn't support order_by search
MlflowClient().set_registered_model_alias(UC_MODEL_NAME, "champion", version)
print(f"Registered {UC_MODEL_NAME} v{version} @champion  (run {run_id})")
