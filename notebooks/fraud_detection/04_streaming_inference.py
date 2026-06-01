# Databricks notebook source

# MAGIC %md
# MAGIC # Fraud Detection — Structured Streaming Inference (Live Data)
# MAGIC The third **inference** pattern, and the headline of the demo: scoring a **live
# MAGIC stream** of transactions in near real time.
# MAGIC
# MAGIC We use Spark's built-in `rate` source as a transaction generator — it emits new
# MAGIC rows every second with no external dependencies — synthesize the same features
# MAGIC the model was trained on, score each transaction with the champion model, and
# MAGIC stream the results (plus fraud alerts) into `shm.ml.fraud_scored_stream`.
# MAGIC
# MAGIC Swap the `rate` source for Kafka / Auto Loader / Zerobus and this exact pipeline
# MAGIC scores real production traffic.

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

CATALOG = "shm"
SCHEMA = "ml"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_xgboost"
MODEL_URI = f"models:/{UC_MODEL_NAME}@champion"
SCORED_TABLE = f"{CATALOG}.{SCHEMA}.fraud_scored_stream"
CHECKPOINT = f"/tmp/fraud_detection/checkpoints/{SCORED_TABLE}"

FEATURES = [
    "amount", "amount_to_avg_ratio", "distance_from_home_km", "num_tx_last_hour",
    "hour_of_day", "merchant_risk", "is_foreign", "card_present", "account_age_days",
]
ROWS_PER_SECOND = 2      # ~one new transaction every 0.5s — "live" feel
ALERT_THRESHOLD = 0.50
RUN_SECONDS = 90         # demo runs for a bounded window, then stops cleanly.
                         # Set to None to stream continuously until you stop it.

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build a live transaction stream
# MAGIC The `rate` source gives us `timestamp` and a monotonically increasing `value`.
# MAGIC We turn each row into a realistic synthetic transaction with the model's features.

# COMMAND ----------

raw = (
    spark.readStream.format("rate")
    .option("rowsPerSecond", ROWS_PER_SECOND)
    .load()
)

transactions = (
    raw
    .withColumn("transaction_id", F.col("value"))
    .withColumn("event_time", F.col("timestamp"))
    .withColumn("customer_id", (F.col("value") % F.lit(20000)).cast("int"))
    .withColumn("amount", F.round(F.exp(F.lit(3.5) + F.lit(1.1) * F.randn()), 2))
    .withColumn("amount_to_avg_ratio", F.round(F.exp(F.lit(0.6) * F.randn()), 3))
    .withColumn("distance_from_home_km", F.round(-F.lit(12.0) * F.log(F.rand() + F.lit(1e-9)), 1))
    .withColumn("num_tx_last_hour", F.floor(F.rand() * F.lit(6)).cast("int"))
    .withColumn("hour_of_day", F.floor(F.rand() * F.lit(24)).cast("int"))
    .withColumn("merchant_risk", F.round(F.rand(), 3))
    .withColumn("is_foreign", (F.rand() < F.lit(0.06)).cast("int"))
    .withColumn("card_present", (F.rand() < F.lit(0.72)).cast("int"))
    .withColumn("account_age_days", (F.floor(F.rand() * F.lit(3620)) + F.lit(30)).cast("int"))
    .select("transaction_id", "event_time", "customer_id", *FEATURES)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score the stream with the champion model
# MAGIC The same `spark_udf` used in batch — one model, every inference pattern.

# COMMAND ----------

score_udf = mlflow.pyfunc.spark_udf(spark, MODEL_URI, result_type="double")

scored = (
    transactions
    .withColumn("fraud_probability", score_udf(*[F.col(c) for c in FEATURES]))
    .withColumn("fraud_alert", (F.col("fraud_probability") >= F.lit(ALERT_THRESHOLD)).cast("int"))
    .withColumn("scored_at", F.current_timestamp())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write the scored stream to Delta
# MAGIC A checkpoint makes the stream exactly-once and restartable.

# COMMAND ----------

query = (
    scored.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT)
    .trigger(processingTime="2 seconds")
    .toTable(SCORED_TABLE)
)
print(f"Streaming live transactions into {SCORED_TABLE} (query id: {query.id})")

# Bounded demo run: stream for RUN_SECONDS, then stop the query cleanly.
# In an interactive notebook, comment this out to keep the stream running while
# you re-run the monitoring cells below.
if RUN_SECONDS:
    query.awaitTermination(RUN_SECONDS)
    query.stop()
    print(f"Stopped after {RUN_SECONDS}s.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Watch the alerts arrive
# MAGIC Freshly scored transactions and the high-risk alerts flagged from the stream.
# MAGIC (If running continuously, re-run these cells to see new rows arrive.)

# COMMAND ----------

display(spark.sql(f"""
    SELECT
      COUNT(*)                                   AS total_scored,
      SUM(fraud_alert)                           AS alerts,
      ROUND(AVG(fraud_probability), 4)           AS avg_score,
      MAX(scored_at)                             AS latest_score_time
    FROM {SCORED_TABLE}
"""))

# COMMAND ----------

display(spark.sql(f"""
    SELECT transaction_id, customer_id, amount, distance_from_home_km,
           is_foreign, card_present, fraud_probability, scored_at
    FROM {SCORED_TABLE}
    WHERE fraud_alert = 1
    ORDER BY scored_at DESC
    LIMIT 15
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop the stream when you're done
# MAGIC ```python
# MAGIC for q in spark.streams.active:
# MAGIC     q.stop()
# MAGIC ```
