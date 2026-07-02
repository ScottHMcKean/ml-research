# Databricks notebook source

# MAGIC %md
# MAGIC # 04 · Train & Register LightGBM
# MAGIC
# MAGIC Builds a training set from the feature tables with **point-in-time correct** lookups
# MAGIC (daily/monthly caches via `timestamp_lookup_key`) plus **on-demand** request-time
# MAGIC features, trains a **LightGBM** classifier on the `blocked` label, and logs it with
# MAGIC `fe.log_model(...)` so the feature metadata travels with the model. At serving time
# MAGIC the endpoint then joins online features and runs the on-demand UDFs automatically —
# MAGIC callers pass only keys + raw request inputs.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering lightgbm
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import mlflow
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from databricks.feature_engineering import (
    FeatureEngineeringClient,
    FeatureLookup,
    FeatureFunction,
)

mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spine + feature lookups
# MAGIC The spine carries the entity keys, the event timestamp (for point-in-time joins),
# MAGIC the on-demand inputs (`amount`, `event_ts`), and the label.

# COMMAND ----------

spine = spark.read.table(RAW_EVENTS).select(
    "instrument_id", "account_id", "bin_prefix", "category_code",
    "event_ts", "amount", LABEL,
)

feature_lookups = [
    # Hot counters: latest value (no timestamp). Missing keys yield nulls, which LightGBM
    # handles natively (the seed covers every instrument in the data, so nulls are rare).
    FeatureLookup(
        table_name=FT_COUNTERS_1H,
        lookup_key="instrument_id",
        feature_names=[
            "inst_txn_cnt", "inst_fail_cnt", "inst_fail_ratio",
            "inst_distinct_accounts", "inst_avg_amount", "inst_max_amount",
        ],
    ),
    # Daily cache: point-in-time correct, fed forward.
    FeatureLookup(
        table_name=FT_ACCOUNT_DAILY,
        lookup_key="account_id",
        timestamp_lookup_key="event_ts",
        feature_names=[
            "acct_daily_cnt", "acct_daily_avg_amount",
            "acct_daily_block_rate", "acct_daily_softfail_cnt",
        ],
    ),
    # Monthly cache: point-in-time correct, fed forward.
    FeatureLookup(
        table_name=FT_ACCOUNT_MONTHLY,
        lookup_key="account_id",
        timestamp_lookup_key="event_ts",
        feature_names=["acct_monthly_cnt", "acct_monthly_avg_amount", "acct_monthly_block_rate"],
    ),
    # Enrichment dimension (numeric).
    FeatureLookup(
        table_name=FT_ACCOUNT_PROFILE,
        lookup_key="account_id",
        feature_names=["is_premium", "account_age_days", "home_region_code", "historical_decline_rate"],
    ),
    # On-demand request-time features.
    FeatureFunction(udf_name=f"{ODF_REQUEST}_log_amount", input_bindings={"amount": "amount"}, output_name="odf_log_amount"),
    FeatureFunction(udf_name=f"{ODF_REQUEST}_is_night", input_bindings={"event_ts": "event_ts"}, output_name="odf_is_night"),
    FeatureFunction(udf_name=f"{ODF_REQUEST}_high_amount", input_bindings={"amount": "amount"}, output_name="odf_high_amount"),
    FeatureFunction(udf_name=f"{ODF_REQUEST}_category_ordinal", input_bindings={"category_code": "category_code"}, output_name="odf_category_ordinal"),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the training set

# COMMAND ----------

training_set = fe.create_training_set(
    df=spine,
    feature_lookups=feature_lookups,
    label=LABEL,
    # Drop identifiers/timestamps and the raw category string (its ordinal is the feature).
    # Every remaining model feature is numeric, so the serving endpoint can replay the raw
    # assembled columns straight into LightGBM with no encoding step.
    exclude_columns=["instrument_id", "account_id", "bin_prefix", "event_ts", "category_code"],
)

train_pdf = training_set.load_df().toPandas()
print(f"Training set: {train_pdf.shape[0]:,} rows x {train_pdf.shape[1]} cols")

X = train_pdf.drop(columns=[LABEL])
y = train_pdf[LABEL].astype(int)

# Time-agnostic random split is fine here; data is already point-in-time correct.
split = int(len(train_pdf) * 0.8)
X_train, X_valid = X.iloc[:split], X.iloc[split:]
y_train, y_valid = y.iloc[:split], y.iloc[split:]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train LightGBM

# COMMAND ----------

mlflow.set_experiment(f"/Shared/{SCHEMA}_lgbm")

with mlflow.start_run(run_name="lgbm_blocked") as run:
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 300,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, proba)
    ap = average_precision_score(y_valid, proba)
    mlflow.log_params(params)
    mlflow.log_metrics({"val_auc": auc, "val_ap": ap})
    print(f"Validation AUC={auc:.4f}  AP={ap:.4f}")

    # Log with feature metadata + register to UC, inside the same run.
    # fe.log_model embeds the FeatureLookup/FeatureFunction metadata so inference
    # joins features automatically.
    fe.log_model(
        model=model,
        artifact_path="payments_lgbm",
        flavor=mlflow.lightgbm,
        training_set=training_set,
        registered_model_name=MODEL_NAME,
    )

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
latest = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
client.set_registered_model_alias(MODEL_NAME, "champion", latest)
print(f"Registered {MODEL_NAME} v{latest} and set alias 'champion'.")
