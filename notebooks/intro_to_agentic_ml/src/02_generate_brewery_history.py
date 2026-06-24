# Databricks notebook source
# 02_generate_brewery_history.py
# -----------------------------------------------------------------------------
# Builds the brewery OT historian tables for Lab 2 (Auto Loader).
# Single regional brewery. Run ONCE (interactively or as a job).
#
# Output (brewery schema):
#   dim_asset             ISA-95 equipment hierarchy (~22 rows)
#   dim_tag               ~60 historian tags w/ units + alarm thresholds
#   fact_sensor_readings  narrow tag-based readings, 5-min cadence (~15.6M rows)
#   fact_anomaly_labels   ground-truth anomaly windows (one row per affected tag)
#
# Comments use plain `#` (not `# MAGIC %md`) so this is safe to run as a
# serverless jobs-submit task — see memory feedback_fevm_serverless_magic_md_cells_skip.
# Verify side-effects via the invariant checks at the bottom.
# -----------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %run ./brewery_generator

# COMMAND ----------

# Shared telemetry engine loaded above: asset/tag catalog, anomaly schedule, value model.

import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql import types as T

# History window — short, demo-sized window ending at 'now'. ~52 days keeps the
# dataset small (~0.9M rows) so the gen job runs fast, while still spanning the
# 2026-05-19 chiller-fault anomaly used for detection/eval and model training.
HIST_START = datetime(2026, 5, 1, 0, 0)
HIST_END   = datetime(2026, 6, 22, 0, 0)

DT_INDEX = build_timestamp_index(HIST_START, HIST_END)   # 5-min cadence
N_STEPS  = len(DT_INDEX)
N_TAGS   = len(_TAGS)
print(f"Timestamps: {N_STEPS:,} @ {CADENCE_MIN}-min  ·  tags: {N_TAGS}  ·  "
      f"projected rows: {N_STEPS * N_TAGS:,}")

# Pre-compute the AZ ambient curve once (shared across all ambient-coupled tags).
AMBIENT_F = _az_ambient_f(DT_INDEX)

# COMMAND ----------

# DBTITLE 1,dim_asset + dim_tag (small dimensions, written from pandas)
asset_pdf = pd.DataFrame(asset_rows())
tag_pdf   = pd.DataFrame(tag_rows())

(spark.createDataFrame(asset_pdf)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(BREW_DIM_ASSET))
(spark.createDataFrame(tag_pdf)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(BREW_DIM_TAG))
print(f"Wrote {BREW_DIM_ASSET} ({len(asset_pdf)} rows), {BREW_DIM_TAG} ({len(tag_pdf)} tags)")

# COMMAND ----------

# DBTITLE 1,fact_sensor_readings — generate per tag, union into Delta (bounds driver memory)
# Strategy: each tag is ~260k points. Generating all 60 in one pandas frame would
# be ~15.6M rows in the driver. Instead we generate per-tag, build a Spark DF per
# tag, and APPEND — so the driver never holds more than one tag's series at once.

READING_SCHEMA = T.StructType([
    T.StructField("reading_ts",   T.TimestampType(), False),
    T.StructField("tag_id",       T.StringType(),    False),
    T.StructField("asset_id",     T.StringType(),    False),
    T.StructField("value",        T.DoubleType(),    False),
    T.StructField("quality_code", T.StringType(),    False),
])

# Materialise the timestamp column once as python datetimes (reused per tag).
TS_PY = DT_INDEX.to_pydatetime()

# Fresh table — drop then append tag-by-tag.
spark.sql(f"DROP TABLE IF EXISTS {BREW_FACT_READINGS}")

batch_dfs = []
BATCH_TAGS = 6   # union a handful of tags before writing to amortise write overhead

def _flush(dfs, first):
    if not dfs:
        return
    big = dfs[0]
    for d in dfs[1:]:
        big = big.unionByName(d)
    (big.write.mode("overwrite" if first else "append")
        .option("mergeSchema", "false")
        .saveAsTable(BREW_FACT_READINGS))

first_write = True
for i, tag in enumerate(_TAGS):
    vals, qual = generate_tag_series(tag, DT_INDEX, ambient_f=AMBIENT_F)
    pdf = pd.DataFrame({
        "reading_ts": TS_PY,
        "tag_id": tag["tag_id"],
        "asset_id": tag["asset_id"],
        "value": vals.astype("float64"),
        "quality_code": qual,
    })
    batch_dfs.append(spark.createDataFrame(pdf, schema=READING_SCHEMA))
    if len(batch_dfs) >= BATCH_TAGS or i == N_TAGS - 1:
        _flush(batch_dfs, first_write)
        first_write = False
        batch_dfs = []
        print(f"  ...{i + 1}/{N_TAGS} tags written")

print(f"Wrote {BREW_FACT_READINGS}")

# COMMAND ----------

# DBTITLE 1,fact_anomaly_labels — ground-truth windows for Lab 2 evaluation
labels_pdf = pd.DataFrame(anomaly_label_rows())
# Keep only labels whose window overlaps the (shortened) history span, so Lab 2's
# recall eval scores against anomalies that actually have data in fact_sensor_readings.
labels_pdf = labels_pdf[(labels_pdf["start_ts"] < HIST_END) & (labels_pdf["end_ts"] > HIST_START)].reset_index(drop=True)
print(f"In-window anomaly labels: {len(labels_pdf)} (of {len(anomaly_label_rows())} total)")
LABEL_SCHEMA = T.StructType([
    T.StructField("asset_id",     T.StringType(),    False),
    T.StructField("tag_id",       T.StringType(),    False),
    T.StructField("start_ts",     T.TimestampType(), False),
    T.StructField("end_ts",       T.TimestampType(), False),
    T.StructField("anomaly_type", T.StringType(),    False),
    T.StructField("severity",     T.StringType(),    False),
])
(spark.createDataFrame(labels_pdf, schema=LABEL_SCHEMA)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(BREW_FACT_ANOMALY_LABELS))
print(f"Wrote {BREW_FACT_ANOMALY_LABELS} ({len(labels_pdf)} label rows, "
      f"{labels_pdf['anomaly_type'].nunique()} anomaly types)")

# COMMAND ----------

# DBTITLE 1,Invariant checks — prove the data is realistic before building labs
import pyspark.sql.functions as F

checks = []

# 1. Row count is in the expected ballpark (~0.9M for the ~52-day demo window)
n_readings = spark.table(BREW_FACT_READINGS).count()
checks.append(("row_count ~0.9M", 600_000 <= n_readings <= 1_200_000, f"{n_readings:,}"))

# 2. Time span matches the requested window
span = spark.table(BREW_FACT_READINGS).agg(
    F.min("reading_ts").alias("lo"), F.max("reading_ts").alias("hi")).first()
checks.append(("span starts 2026-05-01", str(span["lo"]).startswith("2026-05-01"), str(span["lo"])))
checks.append(("span ends <= 2026-06-22", str(span["hi"]) <= "2026-06-22 00:00:00", str(span["hi"])))

# 3. Quality-code mix: mostly Good but a realistic minority non-Good
qmix = {r["quality_code"]: r["c"] for r in
        spark.table(BREW_FACT_READINGS).groupBy("quality_code").agg(F.count("*").alias("c")).collect()}
good_frac = qmix.get("Good", 0) / max(n_readings, 1)
checks.append(("Good fraction 0.94-0.999", 0.94 <= good_frac <= 0.999, f"{good_frac:.4f} · {qmix}"))

# 4. Every tag in dim_tag produced readings
n_tags_with_data = spark.table(BREW_FACT_READINGS).select("tag_id").distinct().count()
checks.append(("all tags have data", n_tags_with_data == N_TAGS, f"{n_tags_with_data}/{N_TAGS}"))

# 5. Anomaly windows actually breach: the 2026-05-19 chiller fault drives glycol
#    return temp above its crit threshold (this is the in-window demo anomaly).
gly_crit = TAG_BY_ID["GLY.GLY01.RETURN_TEMP"]["crit_threshold"]
gly_max = (spark.table(BREW_FACT_READINGS)
           .where("tag_id = 'GLY.GLY01.RETURN_TEMP'")
           .where("reading_ts between '2026-05-19 12:00' and '2026-05-20 02:00'")
           .agg(F.max("value").alias("m")).first()["m"])
checks.append(("chiller fault breaches crit", gly_max is not None and gly_max > gly_crit,
               f"max={gly_max} crit={gly_crit}"))

# 6. Labels reference real tags
orphan = (spark.table(BREW_FACT_ANOMALY_LABELS).select("tag_id").distinct()
          .join(spark.table(BREW_DIM_TAG).select("tag_id"), "tag_id", "left_anti").count())
checks.append(("no orphan label tags", orphan == 0, f"orphans={orphan}"))

print("\n=== INVARIANT CHECKS ===")
all_ok = True
for name, ok, detail in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:32s} {detail}")
    all_ok = all_ok and ok
assert all_ok, "One or more brewery-data invariants failed — inspect output above."
print("\nAll brewery-history invariants passed.")

