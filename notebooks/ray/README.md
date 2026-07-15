# Ray on Databricks — ML

Distributed **machine learning** with Ray on Databricks: a primer, hyperparameter
tuning, distributed deep-learning training, and a Spark-vs-Ray compute benchmark. All
notebooks are ML (classical + deep learning) — no LLM/GenAI content lives here.

Every notebook sets up Ray-on-Spark, derives paths from the current user, and shuts the
cluster down at the end, so they run with no manual path edits.

## Notebooks

| # | Notebook | What it does | Compute |
|---|---|---|---|
| 01 | `01_getting_started.py` | Ray primer — Monte Carlo π with `@ray.remote`, one task then fanned out. | Serverless / any |
| 02 | `02_tune_xgboost.ipynb` | Ray Tune hyperparameter search over a GPU XGBoost classifier (breast-cancer). | GPU ML |
| 03 | `03_train_pytorch.py` | Ray Train + PyTorch Lightning distributed image classification on the flowers Delta dataset; logs to MLflow. | GPU ML |
| 04 | `04_train_tensorflow.py` | Ray Train + Keras `MultiWorkerMirroredStrategy` distributed MNIST; checkpoints and reloads. | CPU ML (GPU optional) |
| 05 | `05_spark_vs_ray_base.py` | Spark vs Ray on a heterogeneous CPU→GPU→CPU NLP pipeline (biomedical NER + sentiment over 10k docs). The straightforward shape. | GPU ML |
| 06 | `06_spark_vs_ray_optimized.py` | Same pipeline, tuned — `mapInPandas` on Spark, off-GPU unpacking on Ray, larger batches. Pairs with 05. | GPU ML |

## Spark vs Ray as a job

Notebooks 05 and 06 are also defined as bundle jobs in `resources/ray_jobs.yml`
(`spark_vs_ray_base`, `spark_vs_ray_optimized`) on a single-node GPU cluster:

```bash
databricks bundle deploy -t dev
databricks bundle run spark_vs_ray_base -t dev
```

The Spark-vs-Ray notebooks write to `shm.ml.spark_vs_ray_*`; edit the catalog/schema
constants at the top of each if `shm.ml` doesn't exist in your workspace.
