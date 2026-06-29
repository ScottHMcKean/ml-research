# Databricks notebook source

# MAGIC %md
# MAGIC # 06 · Benchmark online serving latency
# MAGIC
# MAGIC Profiles the end-to-end scoring path (online feature lookup + on-demand compute +
# MAGIC LightGBM inference) under concurrency and reports **p50 / p90 / p99** latency and
# MAGIC throughput, then compares against the architecture sample's latency targets
# MAGIC (Feature Store read p99 ~8ms, model inference p99 ~5ms, end-to-end p99 < 50ms).
# MAGIC Results are written to `benchmark_results` for tracking over time.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

dbutils.widgets.text("n_requests", "1000", "Number of scoring requests")
dbutils.widgets.text("concurrency", "8", "Concurrent workers")
N_REQUESTS = int(dbutils.widgets.get("n_requests"))
CONCURRENCY = int(dbutils.widgets.get("concurrency"))

# COMMAND ----------

import time
import datetime as dt
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

# Pre-build a pool of realistic request payloads from real keys.
pool = spark.read.table(RAW_EVENTS).select(
    "instrument_id", "account_id", "category_code", "amount",
).limit(5000).toPandas()


def make_record(i):
    r = pool.iloc[i % len(pool)]
    return {
        "instrument_id": r.instrument_id,
        "account_id": r.account_id,
        "category_code": r.category_code,
        "amount": float(r.amount),
        "event_ts": dt.datetime.utcnow().isoformat(),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Warm up, then measure under concurrency

# COMMAND ----------

# Warm up to spin the endpoint up from scale-to-zero (excluded from the measurement).
for i in range(10):
    client.predict(endpoint=SERVING_ENDPOINT, inputs={"dataframe_records": [make_record(i)]})


def timed_call(i):
    rec = make_record(i)
    t0 = time.perf_counter()
    client.predict(endpoint=SERVING_ENDPOINT, inputs={"dataframe_records": [rec]})
    return (time.perf_counter() - t0) * 1000.0  # ms


wall_start = time.perf_counter()
with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
    latencies = list(ex.map(timed_call, range(N_REQUESTS)))
wall_s = time.perf_counter() - wall_start

lat = np.array(latencies)
p50, p90, p99 = np.percentile(lat, [50, 90, 99])
throughput = N_REQUESTS / wall_s

print(f"Requests   : {N_REQUESTS:,}  (concurrency {CONCURRENCY})")
print(f"p50 / p90 / p99 : {p50:.1f} / {p90:.1f} / {p99:.1f} ms")
print(f"mean / max      : {lat.mean():.1f} / {lat.max():.1f} ms")
print(f"throughput      : {throughput:.1f} req/s")
print(f"p99 < 50ms target: {'PASS' if p99 < 50 else 'review (cold cache / scale-to-zero / region)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist the result

# COMMAND ----------

result = pd.DataFrame([{
    "run_ts": dt.datetime.utcnow(),
    "endpoint": SERVING_ENDPOINT,
    "n_requests": N_REQUESTS,
    "concurrency": CONCURRENCY,
    "p50_ms": round(float(p50), 2),
    "p90_ms": round(float(p90), 2),
    "p99_ms": round(float(p99), 2),
    "mean_ms": round(float(lat.mean()), 2),
    "throughput_rps": round(float(throughput), 2),
}])

spark.createDataFrame(result).write.mode("append").option("mergeSchema", "true").saveAsTable(BENCHMARK_RESULTS)
display(spark.read.table(BENCHMARK_RESULTS).orderBy("run_ts", ascending=False))
