# Databricks notebook source

# MAGIC %md
# MAGIC # 09 · Kafka integration (consume transactions → score → produce results)
# MAGIC
# MAGIC The Databricks App in `app/` uses an **API** pattern: it synthesizes events and inserts
# MAGIC them into the raw table as a cost-effective stand-in for a real bus. This notebook shows
# MAGIC the **Kafka-to-Kafka** pattern instead — the shape a production payments deployment
# MAGIC uses:
# MAGIC
# MAGIC ```
# MAGIC Kafka (transactions) → readStream → assemble features → score → writeStream → Kafka (decisions)
# MAGIC ```
# MAGIC
# MAGIC It is written to be **read as a reference**: the Kafka source/sink blocks are guarded by
# MAGIC a `KAFKA_BOOTSTRAP` widget and are inert until you point them at a real broker, so the
# MAGIC notebook imports and validates without a cluster attached to Kafka. The scoring step
# MAGIC reuses the same serving endpoint (`05_serving`) with automatic online-feature lookup.
# MAGIC
# MAGIC ## Two ways to build the feature side
# MAGIC 1. **This notebook** — you own the Structured Streaming aggregation. Write it to a Delta
# MAGIC    feature table and let Lakebase **`CONTINUOUS`** publish auto-sync it to the online
# MAGIC    store (works today, fully GA, maximum control). Keep the Feature Engineering client
# MAGIC    **out** of `foreachBatch`: on serverless the closure runs in an isolated worker with
# MAGIC    no auth, so publishing belongs to a driver-side `fe.publish_table(..., CONTINUOUS)`
# MAGIC    or the synced-table pipeline — not inside the stream. (`07` handles the same constraint.)
# MAGIC 2. **Streaming Declarative Features** (Kafka → serverless SDP → Lakebase, Public Preview
# MAGIC    mid-2026) — you declare `create_feature(...)` over the Kafka source and Databricks
# MAGIC    runs the pipeline. p95 freshness <0.5 s. See the preview block near the bottom and the
# MAGIC    caveats in the README (JSON only at launch; Count/Avg/Sum/StddevPop aggs; rolling
# MAGIC    windows ≤ 1 week).

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

dbutils.widgets.text("kafka_bootstrap", "", "Kafka bootstrap servers (blank = inert / reference only)")
dbutils.widgets.text("topic_in", "transactions", "Source topic (transactions in)")
dbutils.widgets.text("topic_out", "decisions", "Sink topic (decisions out)")
dbutils.widgets.dropdown("auth", "uc_service_credential", ["uc_service_credential", "mtls", "sasl_scram", "msk_iam"], "Kafka auth mechanism")

KAFKA_BOOTSTRAP = dbutils.widgets.get("kafka_bootstrap").strip()
TOPIC_IN = dbutils.widgets.get("topic_in")
TOPIC_OUT = dbutils.widgets.get("topic_out")
AUTH = dbutils.widgets.get("auth")

import datetime as dt
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Kafka auth options
# MAGIC Pick the mechanism your bus uses. **UC service credentials** are the cleanest — no
# MAGIC secrets in the notebook, and they work on serverless and classic. Others are shown as
# MAGIC commented recipes. Store any keystore/password material in **UC secrets**, never inline.

# COMMAND ----------

def kafka_auth_options(auth: str) -> dict:
    """Return the reader/writer options for the chosen Kafka auth mechanism."""
    if auth == "uc_service_credential":
        # Credential-less: reference a Unity Catalog service credential granted access to the bus.
        return {"kafka.security.protocol": "SASL_SSL",
                "kafka.sasl.mechanism": "OAUTHBEARER",
                # databricks.service.credential names a UC service credential.
                "databricks.service.credential": "payments_kafka_cred"}
    if auth == "mtls":
        # Keystore/truststore staged on a UC volume; passwords pulled from a secret scope.
        return {"kafka.security.protocol": "SSL",
                "kafka.ssl.keystore.location": f"/Volumes/{CATALOG}/{SCHEMA}/kafka/keystore.jks",
                "kafka.ssl.truststore.location": f"/Volumes/{CATALOG}/{SCHEMA}/kafka/truststore.jks",
                "kafka.ssl.keystore.password": dbutils.secrets.get("payments_kafka", "keystore_pw"),
                "kafka.ssl.truststore.password": dbutils.secrets.get("payments_kafka", "truststore_pw")}
    if auth == "sasl_scram":
        pw = dbutils.secrets.get("payments_kafka", "scram_pw")
        return {"kafka.security.protocol": "SASL_SSL",
                "kafka.sasl.mechanism": "SCRAM-SHA-512",
                "kafka.sasl.jaas.config":
                    "kafkashaded.org.apache.kafka.common.security.scram.ScramLoginModule required "
                    f'username="payments" password="{pw}";'}
    if auth == "msk_iam":
        return {"kafka.security.protocol": "SASL_SSL",
                "kafka.sasl.mechanism": "AWS_MSK_IAM",
                "kafka.sasl.jaas.config": "shadedmskiam.software.amazon.msk.auth.iam.IAMLoginModule required;",
                "kafka.sasl.client.callback.handler.class":
                    "shadedmskiam.software.amazon.msk.auth.iam.IAMClientCallbackHandler"}
    raise ValueError(f"unknown auth: {auth}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transaction message schema
# MAGIC The bus carries JSON payment-authorization events; we parse the `value` bytes with an
# MAGIC explicit schema (matches the keys the serving endpoint needs). For Avro/Proto buses use
# MAGIC `from_avro` / `from_protobuf` with a schema-registry config instead of `from_json`.

# COMMAND ----------

txn_schema = StructType([
    StructField("event_id", StringType()),
    StructField("instrument_id", StringType()),
    StructField("account_id", StringType()),
    StructField("category_code", StringType()),
    StructField("amount", DoubleType()),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Consume transactions from Kafka
# MAGIC `readStream.format("kafka")` with the bootstrap servers, source topic, and auth options.
# MAGIC When `kafka_bootstrap` is blank this cell is skipped so the notebook stays runnable as a
# MAGIC reference; set the widget to a real broker to activate it.

# COMMAND ----------

if KAFKA_BOOTSTRAP:
    transactions = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC_IN)
        .option("startingOffsets", "latest")
        .options(**kafka_auth_options(AUTH))
        .load()
        # value is bytes -> string -> parsed JSON columns.
        .select(F.from_json(F.col("value").cast("string"), txn_schema).alias("t"))
        .select("t.*")
        .withColumn("event_ts", F.current_timestamp())
    )
    print(f"Reading from kafka topic '{TOPIC_IN}' on {KAFKA_BOOTSTRAP} (auth={AUTH}).")
else:
    print("kafka_bootstrap is blank — Kafka source inert (reference mode). "
          "Set the widget to a broker to activate.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Score each micro-batch and 3 · produce decisions back to Kafka
# MAGIC `foreachBatch` gives us a batch DataFrame per trigger. We query the serving endpoint
# MAGIC (which auto-joins the online features and computes the on-demand UDFs), build a decision
# MAGIC record, and write it to the outbound topic. Model scoring per micro-batch keeps the bus
# MAGIC integration decoupled from the model version.
# MAGIC
# MAGIC For the **strictest per-transaction SLAs**, run this stream in **Spark Real-Time Mode**
# MAGIC (`trigger(realTime=...)`, GA) instead of a `processingTime` trigger — RTM drives
# MAGIC end-to-end Kafka-to-Kafka latency toward a ~5 ms floor. Use the `processingTime` trigger
# MAGIC below when ~seconds of decision latency is acceptable (cheaper, no dedicated slots).
# MAGIC
# MAGIC > **Compute / auth note (same constraint as `07`):** a `foreachBatch` closure runs in an
# MAGIC > isolated worker. On serverless (Spark Connect) that worker has **no default
# MAGIC > credentials**, so a bare `WorkspaceClient()` fails with *"cannot configure default
# MAGIC > credentials."* Run this scoring stream on **classic compute** (where the worker
# MAGIC > inherits notebook auth), or, as below, construct the client with an **explicit host +
# MAGIC > token** (from a secret scope) so it authenticates inside the worker regardless of
# MAGIC > compute type.

# COMMAND ----------

from databricks.sdk import WorkspaceClient


def _worker_workspace_client() -> WorkspaceClient:
    """Auth that survives an isolated foreachBatch worker (see the compute/auth note above).

    On classic compute a bare WorkspaceClient() inherits notebook auth. On serverless the
    worker has no default credentials, so build the client from an explicit host + token
    stored in a secret scope. Falls back to the default constructor when the secret is absent
    (e.g. classic compute), so this works in both places without config churn.
    """
    try:
        host = dbutils.secrets.get("payments_kafka", "workspace_host")
        token = dbutils.secrets.get("payments_kafka", "workspace_token")
        return WorkspaceClient(host=host, token=token)
    except Exception:
        return WorkspaceClient()


def score_and_produce(batch_df, epoch_id: int) -> None:
    rows = batch_df.select("event_id", "instrument_id", "account_id", "category_code", "amount").collect()
    if not rows:
        return
    # Serverless (Spark Connect): get the session from the batch DF and build the client here —
    # the foreachBatch worker has no captured spark session or notebook auth.
    spark = batch_df.sparkSession
    w = _worker_workspace_client()
    records = [{
        "instrument_id": r.instrument_id, "account_id": r.account_id,
        "category_code": r.category_code, "amount": float(r.amount or 0.0),
        "event_ts": dt.datetime.utcnow().isoformat(),
    } for r in rows]

    resp = w.serving_endpoints.query(name=SERVING_ENDPOINT, dataframe_records=records)
    preds = resp.predictions or []

    decisions = [{
        "event_id": rows[i].event_id,  # carry the inbound event id through as the correlation key
        "account_id": rows[i].account_id,
        "decision": "blocked" if float(preds[i]) >= 0.5 else "pass",
        "model_output": float(preds[i]),
        "scored_at": dt.datetime.utcnow().isoformat(),
    } for i in range(min(len(rows), len(preds)))]

    # Produce back to Kafka: the sink expects a `value` (and optional `key`) column.
    out = (
        spark.createDataFrame(decisions)
        .select(
            F.col("account_id").alias("key"),
            F.to_json(F.struct("event_id", "account_id", "decision", "model_output", "scored_at")).alias("value"),
        )
    )
    (
        out.write.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("topic", TOPIC_OUT)
        .options(**kafka_auth_options(AUTH))
        .save()
    )
    print(f"[epoch {epoch_id}] scored {len(records)} txns -> produced {len(decisions)} decisions to '{TOPIC_OUT}'.")

# COMMAND ----------

if KAFKA_BOOTSTRAP:
    q = (
        transactions.writeStream
        .foreachBatch(score_and_produce)
        .option("checkpointLocation", f"{CHECKPOINT_ROOT}/kafka_io")
        # Swap for .trigger(realTime="5 seconds") to run in Real-Time Mode for sub-second decisions.
        .trigger(processingTime="5 seconds")
        .queryName("kafka_score_produce")
        .start()
    )
    print(f"Kafka-to-Kafka scoring stream started. Query id: {q.id}")
    q.awaitTermination()
else:
    print("Reference mode — scoring/produce stream not started.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview alternative — Streaming Declarative Features (Kafka → serverless SDP → Lakebase)
# MAGIC Instead of hand-writing the aggregation above, you can **declare** streaming features
# MAGIC directly over the Kafka source and let Databricks run the managed pipeline. This is the
# MAGIC `create_feature` / declarative path (Public Preview mid-2026). Shown as a reference —
# MAGIC API surface firms up as it approaches GA; validate against the current SDK before use.
# MAGIC
# MAGIC ```python
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient
# MAGIC fe = FeatureEngineeringClient()
# MAGIC
# MAGIC # Declare a Kafka-sourced streaming feature with a rolling window aggregation.
# MAGIC # Supported aggregations at launch: Count / Avg / Sum / StddevPop. Rolling windows ≤ 1 week.
# MAGIC # Serialization: JSON only at launch (Avro/Proto + schema registry post-Q2).
# MAGIC fe.create_feature(
# MAGIC     source={"format": "kafka",
# MAGIC             "options": {"kafka.bootstrap.servers": KAFKA_BOOTSTRAP, "subscribe": TOPIC_IN,
# MAGIC                         "databricks.service.credential": "payments_kafka_cred"}},
# MAGIC     entity_keys=["instrument_id"],
# MAGIC     timestamp_key="event_ts",
# MAGIC     features=[{"name": "inst_txn_cnt_1h", "agg": "count", "window": "1 hour"}],
# MAGIC     online_store=ONLINE_STORE,   # publishes to Lakebase automatically
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### On-prem / PCI networking
# MAGIC When the broker is on-prem or PCI-scoped, the network path and its round-trip latency
# MAGIC often gate the architecture more than the feature technology does:
# MAGIC - **Serverless → on-prem broker:** Network Connectivity Config (NCC) + PrivateLink.
# MAGIC - **Classic compute → on-prem broker:** VPC peering / Private Service Connect.
# MAGIC - Prefer **mTLS**; keep all secrets in **UC secret scopes**.
# MAGIC - Measure the cloud-to-broker round-trip early. If it exceeds the scoring SLA,
# MAGIC   inference belongs on-prem regardless of the feature technology.
