# Rare-Failure Anomaly Detection — PCA + AutoEncoder (few-shot)

Unsupervised anomaly detection for **rotating equipment** where you have years of
mostly-normal multi-sensor data and **only one or two confirmed failures**. Too few
failures to train a supervised classifier — so we learn what *normal* looks like and
flag deviations, then **use the couple of observed failures to calibrate the decision
threshold** (and, for the autoencoder, to keep anomalies off the learned manifold).

```
data  →  PCA detector  ┐
                        ├─→  compare  →  champion + batch-scored table
data  →  AE  detector  ┘
```

Everything is synthetic and self-contained — no external data — so it runs anywhere
with zero setup.

## The notebooks

| Notebook | Approach | What it does |
|---|---|---|
| `00_data_prep.py` | **Data** | Generates 20K hourly readings of a physics-inspired **12-sensor** industrial pump (vibration, bearing/motor temp, current/voltage, flow, pressures, rpm, oil particles, acoustics). Injects **7 rare failure episodes** across 5 modes (bearing wear, cavitation, impeller imbalance, seal leak, motor winding) at a ~0.5% rate, and marks **only 2 rows as observed** (`is_labeled`). Writes `shm_skunkworks_catalog.anomaly_detection.pump_sensors`. |
| `01_pca_anomaly.py` | **PCA reconstruction error** | Fits `StandardScaler` + PCA (top-5 components) on the unlabeled data. Anomaly score = reconstruction error. Threshold is **calibrated from the 2 observed failures** (vs. a naive top-1% percentile rule, logged side-by-side). Registers a pyfunc to UC as `pca_anomaly@champion`. |
| `02_autoencoder_anomaly.py` | **AutoEncoder reconstruction error** | Trains an undercomplete PyTorch AE (`12→8→3→8→12`) on a **serverless GPU (A10)**. Unsupervised reconstruction loss **plus a few-shot margin penalty** that forces the 2 observed failures to reconstruct badly (so a flexible AE can't learn to reproduce anomalies). Same few-shot threshold calibration. Registers `autoencoder_anomaly@champion`. |
| `03_compare.py` | **Compare** | Loads both champions, scores the full table, compares on hidden ground truth (ROC-AUC, PR-AUC, **per-failure-mode recall**), plots PR curves, and writes `pump_scored`. |

**The through-line:** unsupervised detectors, augmented with just two labels used two
ways — (1) to calibrate the alert threshold, and (2) to shape the AE's manifold. The
hidden `is_failure` / `failure_mode` columns are used **only for evaluation** — never
for training.

## Verified live (FEVM `fe-vm-shm-skunkworks`, serverless)

Full pipeline ran green as a serverless job (AutoEncoder task on a **GPU A10** —
`device: cuda`). Evaluated on the held-out hidden ground truth (the 2 labeled rows
excluded):

| Model | ROC-AUC | PR-AUC | Precision @ few-shot thr | Recall | Alert rate |
|---|---|---|---|---|---|
| PCA (5 comp, 86% var) | 0.980 | 0.814 | 0.11 | 0.89 | 3.3% |
| **AutoEncoder (12→8→3→8→12)** | **1.000** | **0.997** | **1.00** | 0.77 | **0.3%** |

The AutoEncoder is the champion by PR-AUC: the non-linear manifold + the few-shot margin
penalty give it near-perfect ranking and a **0.3% alert rate at 100% precision** — the
difference between a usable and an unusable predictive-maintenance alarm. Both detectors
registered to UC `@champion`; `pump_scored` written by `03`.

## Run it

**Interactively (easiest):** open `00 → 03` in order and Run All on **serverless**.
Each notebook `%pip install`s its own deps and restarts Python.

**As a job (Asset Bundle, serverless):** the pipeline is defined in
`resources/anomaly_detection_jobs.yml` (`anomaly_detection_pipeline`) and is fully
serverless — no cloud-specific compute — so the job itself is portable to any workspace.

```bash
databricks bundle deploy -t skunkworks
databricks bundle run  anomaly_detection_pipeline -t skunkworks
```

> Note: `bundle deploy` deploys **every** resource in this repo, and a few of the other
> jobs (fraud, distributed-xgboost) are pinned to Azure node types, so a full deploy
> fails on an AWS workspace at *those* jobs. On AWS FEVM, run the notebooks
> interactively (above), or copy this one resource into a standalone bundle. The
> pipeline was verified end-to-end on FEVM `fe-vm-shm-skunkworks` via a serverless job
> over the imported notebooks.

## Config

All notebooks share these constants (edit at the top of each):

- Catalog / schema: `shm_skunkworks_catalog.anomaly_detection`
- Table: `pump_sensors` (+ `pump_scored` from `03`)
- Experiment: `/Shared/anomaly_detection`
- UC models: `pca_anomaly`, `autoencoder_anomaly` (alias `@champion`)

## Swapping in a real dataset

Replace `00_data_prep.py` with a loader for a public predictive-maintenance dataset that
has rare failures and multiple sensors — e.g. **MetroPT-3** (metro air-compressor
failures) or the **NASA IMS bearing** run-to-failure set. Keep the output schema (sensor
columns + `is_failure`, `failure_mode`, `is_labeled`) and `01`–`03` work unchanged.

## Cleanup

```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
spark.sql("DROP TABLE IF EXISTS shm_skunkworks_catalog.anomaly_detection.pump_sensors")
spark.sql("DROP TABLE IF EXISTS shm_skunkworks_catalog.anomaly_detection.pump_scored")
```
