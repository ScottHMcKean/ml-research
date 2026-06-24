# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md-sandbox
# MAGIC <div style="background:#1B3139; color:white; padding:24px; border-radius:8px;">
# MAGIC   <h1 style="margin:0; color:white;">Lab 2A — Brewery OT Anomaly Detection (Auto Loader)</h1>
# MAGIC   <p style="margin:6px 0 0 0; color:#9EB7BE; font-size:16px;">
# MAGIC     Persona: <strong style="color:white;">Data Engineer</strong>
# MAGIC     &nbsp;·&nbsp; Estimated time: 60 min &nbsp;·&nbsp; Agenda slot: Brewery OT sensor lab
# MAGIC   </p>
# MAGIC </div>
# MAGIC
# MAGIC **The setting:** a **regional brewery** streams sensor telemetry
# MAGIC from its brewhouse, fermentation cellar, glycol/refrigeration, canning line, CIP, and
# MAGIC utilities — ~60 tags at a 5-minute cadence. We'll build a **medallion pipeline** that lands
# MAGIC the raw stream, enriches it with ISA-95 context + alarm thresholds, and **detects anomalies**
# MAGIC (a creeping glycol-chiller fault) — then score our detector against labeled ground truth.
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #FF3620; padding:15px; margin:15px 0;">
# MAGIC   <strong>Why this schema is realistic.</strong> Real plant historians (PI, Canary, Aveva)
# MAGIC   store telemetry <em>narrow</em> — one row per tag per timestamp — not a wide
# MAGIC   column-per-sensor table. Tags follow an <strong>ISA-95</strong> hierarchy
# MAGIC   (enterprise → site → area → work&nbsp;center → equipment) and carry an
# MAGIC   <strong>OPC quality code</strong> (Good / Uncertain / Bad / Substituted). We model all of that.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Genie Code throughout:</strong> press <kbd>Cmd</kbd>+<kbd>I</kbd> in the code
# MAGIC   cell, paste the prompt, let Genie write it. Reference solutions are included.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#FF3621; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   Tested on Serverless v5
# MAGIC </div>

# COMMAND ----------

# MAGIC %run ./src/00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Review the asset & tag dimensions
# MAGIC The batch history (`fact_sensor_readings`) and the ISA-95 dimensions were built once by the
# MAGIC **data-generation job** (`02_generate_brewery_history.py`). Let's see what we're working with.

# COMMAND ----------

# DBTITLE 1,Step 1: ISA-95 hierarchy + tag thresholds
print("Assets by area:")
display(spark.sql(f"SELECT area, work_center, COUNT(*) assets FROM {BREW_DIM_ASSET} GROUP BY area, work_center ORDER BY area"))
print("Sample tags with alarm thresholds:")
display(spark.sql(f"""
  SELECT tag_id, metric, unit, normal_low, normal_high, warn_threshold, crit_threshold
  FROM {BREW_DIM_TAG} WHERE asset_id IN ('FV-003','GLY-01','FILL-01','SEAM-01') ORDER BY tag_id
"""))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 2 — A streaming job is dropping Sparkplug-B JSON on the Volume
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>💡 What this simulates:</strong> in production an MQTT broker (Sparkplug B) →
# MAGIC   cloud-storage bridge drops JSON files into a landing zone. We don't have the broker here, so the
# MAGIC   bundle's <strong>streaming job</strong> (<code>write_to_volume.py</code>, on a small classic cluster)
# MAGIC   emits the <em>same JSON shape</em> into the UC Volume — a new file every ~20s — including a
# MAGIC   <strong>live chiller-drift anomaly</strong> that turns on a few batches in. <strong>Start it now and
# MAGIC   leave it running</strong> through the lab:
# MAGIC   <blockquote style="border-left:3px solid #1B5161; margin:8px 0; padding:4px 12px; color:#1B3139;">
# MAGIC   databricks bundle run ml_workshop_streaming --no-wait
# MAGIC   </blockquote>
# MAGIC   Because it runs continuously, each time you re-run the Auto Loader step below it picks up the
# MAGIC   <em>new</em> files that have landed since. Cancel the job run when you finish Lab 2.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 2: Confirm the streaming job is landing files (it's continuous — counts grow)
try:
    incoming = dbutils.fs.ls(f"{VOLUME_PATH}/incoming")
except Exception:
    incoming = []
if not incoming:
    raise RuntimeError(
        f"No files in {VOLUME_PATH}/incoming yet. Start the streaming job and give it a moment:\n"
        "  databricks bundle run ml_workshop_streaming --no-wait")
print(f"{len(incoming)} Sparkplug-B JSON micro-batches landed so far in {VOLUME_PATH}/incoming "
      "(this keeps growing while the job runs)")
display(spark.sql(f"LIST '{VOLUME_PATH}/incoming/'"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 3 — Bronze: Auto Loader ingestion
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #1B5161; padding:15px; margin:15px 0;">
# MAGIC   <strong>Auto Loader 101:</strong>
# MAGIC   <ul style="margin:6px 0 0 0;">
# MAGIC     <li><code>cloudFiles</code> incrementally reads <em>new</em> files — no bucket-listing races.</li>
# MAGIC     <li><code>schemaLocation</code> persists the inferred schema between runs.</li>
# MAGIC     <li><code>rescuedDataColumn</code> captures fields that don't match — your schema-drift safety net.</li>
# MAGIC   </ul>
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Genie Code prompt:</strong>
# MAGIC   <blockquote style="border-left:3px solid #1B5161; margin:8px 0; padding:4px 12px; color:#1B3139;">
# MAGIC   Write a Spark Structured Streaming read using Auto Loader (format cloudFiles, json) from
# MAGIC   the path in the variable INCOMING, with schema inference, a schemaLocation under
# MAGIC   VOLUME_PATH/_schemas, and a rescued data column. Add an ingest timestamp column, then
# MAGIC   writeStream with availableNow trigger and a checkpoint under VOLUME_PATH/_checkpoints
# MAGIC   into the table named by BREW_BRONZE. Await termination.
# MAGIC   </blockquote>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 3: Bronze streaming ingest (availableNow = batch-style drain)
from pyspark.sql import functions as F

INCOMING        = f"{VOLUME_PATH}/incoming"
bronze_ckpt     = f"{VOLUME_PATH}/_checkpoints/bronze_readings"
bronze_schema   = f"{VOLUME_PATH}/_schemas/bronze_readings"

bronze_stream = (spark.readStream.format("cloudFiles")
                 .option("cloudFiles.format", "json")
                 .option("cloudFiles.schemaLocation", bronze_schema)
                 .option("cloudFiles.inferColumnTypes", "true")
                 .option("rescuedDataColumn", "_rescued_data")
                 .load(INCOMING)
                 .withColumn("_ingest_ts", F.current_timestamp())
                 .withColumn("_source_file", F.col("_metadata.file_path")))

(bronze_stream.writeStream
 .option("checkpointLocation", bronze_ckpt)
 .trigger(availableNow=True)
 .toTable(BREW_BRONZE))

# availableNow streams finish on their own; wait for the query to complete
for q in spark.streams.active:
    q.awaitTermination()

print(f"Bronze rows: {spark.table(BREW_BRONZE).count():,}")
display(spark.table(BREW_BRONZE).limit(5))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 4 — Silver: enrich with ISA-95 context + thresholds
# MAGIC
# MAGIC The bronze payload has Sparkplug fields (`name`, `alias`, `value`, `quality`, `timestamp`).
# MAGIC Silver casts the epoch-ms timestamp, renames to our historian vocabulary, and **joins
# MAGIC `dim_tag`** so every reading carries its normal band + warn/crit thresholds and **`dim_asset`**
# MAGIC for area / work-center context. This is the table the anomaly logic runs on.
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Genie Code prompt:</strong>
# MAGIC   <blockquote style="border-left:3px solid #1B5161; margin:8px 0; padding:4px 12px; color:#1B3139;">
# MAGIC   From BREW_BRONZE, build a silver DataFrame: convert the epoch-millis `timestamp` column to
# MAGIC   a TIMESTAMP reading_ts, rename `name` to tag_id and `alias` to asset_id, keep value and
# MAGIC   quality as quality_code, join BREW_DIM_TAG on tag_id to bring in unit/normal_low/normal_high/
# MAGIC   warn_threshold/crit_threshold and BREW_DIM_ASSET on asset_id for area/work_center, then
# MAGIC   write it to the table named by BREW_SILVER.
# MAGIC   </blockquote>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 4: Silver transform + write
silver = (spark.table(BREW_BRONZE)
          .select(
              (F.col("timestamp") / 1000).cast("timestamp").alias("reading_ts"),
              F.col("name").alias("tag_id"),
              F.col("alias").alias("asset_id"),
              F.col("value").cast("double").alias("value"),
              F.col("quality").alias("quality_code"))
          .join(spark.table(BREW_DIM_TAG).select(
                    "tag_id", "metric", "unit", "normal_low", "normal_high",
                    "warn_threshold", "crit_threshold"), "tag_id", "left")
          .join(spark.table(BREW_DIM_ASSET).select("asset_id", "area", "work_center", "asset_type"),
                "asset_id", "left"))

silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(BREW_SILVER)
print(f"Silver rows: {spark.table(BREW_SILVER).count():,}")
display(spark.table(BREW_SILVER).where("asset_id='GLY-01'").orderBy("reading_ts").limit(8))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 5 — Gold: detect anomalies
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #FF3620; padding:15px; margin:15px 0;">
# MAGIC   <strong>Two complementary detectors:</strong>
# MAGIC   <ol style="margin:6px 0 0 0;">
# MAGIC     <li><strong>Threshold breach</strong> — value crosses the tag's <code>warn</code>/<code>crit</code>
# MAGIC         line. Cheap, explainable, and exactly what a plant alarm system does.</li>
# MAGIC     <li><strong>Rolling z-score</strong> — value deviates &gt; 3σ from its own recent mean.
# MAGIC         Catches <em>drift</em> (the chiller creeping up) <em>before</em> it trips the hard limit.</li>
# MAGIC   </ol>
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Genie Code prompt:</strong>
# MAGIC   <blockquote style="border-left:3px solid #1B5161; margin:8px 0; padding:4px 12px; color:#1B3139;">
# MAGIC   From BREW_SILVER, for each tag_id compute a rolling mean and standard deviation over the
# MAGIC   preceding 12 readings using a window ordered by reading_ts, then a z-score. Flag a row as
# MAGIC   an anomaly when value exceeds crit_threshold OR the absolute z-score is above 3. Write the
# MAGIC   flagged rows to BREW_GOLD_ANOMALIES.
# MAGIC   </blockquote>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 5: Gold anomaly detection (threshold + rolling z-score)
from pyspark.sql.window import Window

w = Window.partitionBy("tag_id").orderBy("reading_ts").rowsBetween(-12, -1)
scored = (spark.table(BREW_SILVER)
          .withColumn("roll_mean", F.avg("value").over(w))
          .withColumn("roll_std",  F.stddev("value").over(w))
          .withColumn("zscore", F.when(F.col("roll_std") > 0,
                                        (F.col("value") - F.col("roll_mean")) / F.col("roll_std"))
                                 .otherwise(F.lit(0.0))))

anomalies = (scored
             .withColumn("breach_crit", F.col("value") > F.col("crit_threshold"))
             .withColumn("zscore_anom", F.abs(F.col("zscore")) > 3.0)
             .where("breach_crit OR zscore_anom")
             .withColumn("detect_reason",
                         F.when(F.col("breach_crit"), F.lit("crit_threshold"))
                          .otherwise(F.lit("zscore>3")))
             .select("reading_ts", "tag_id", "asset_id", "area", "value",
                     "crit_threshold", F.round("zscore", 2).alias("zscore"), "detect_reason"))

anomalies.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(BREW_GOLD_ANOMALIES)
print(f"Detected anomaly readings: {spark.table(BREW_GOLD_ANOMALIES).count():,}")
display(spark.table(BREW_GOLD_ANOMALIES).orderBy("reading_ts").limit(20))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 6 — Score the detector against labeled ground truth
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #1B5161; padding:15px; margin:15px 0;">
# MAGIC   <strong>Range-based recall.</strong> Anomalies are <em>windows</em>, not points
# MAGIC   (<code>fact_anomaly_labels</code> has start/end per affected tag). The right question
# MAGIC   isn't "did we flag every single reading" but <strong>"did we catch each labeled
# MAGIC   episode at all?"</strong> We compute, per labeled window, whether ≥1 detection fell inside it.
# MAGIC </div>
# MAGIC
# MAGIC > Note: this runs over the **batch history** windows (`fact_anomaly_labels`). The live
# MAGIC > trickle's chiller drift is an *additional* fresh anomaly the z-score should also surface.

# COMMAND ----------

# DBTITLE 1,Step 6: Per-window recall vs fact_anomaly_labels
# Score against the batch history's labeled windows. Run the SAME detector over the
# full batch readings so we can evaluate recall on known episodes.
w_hist = Window.partitionBy("tag_id").orderBy("reading_ts").rowsBetween(-12, -1)
hist_scored = (spark.table(BREW_FACT_READINGS)
               .join(spark.table(BREW_DIM_TAG).select("tag_id", "crit_threshold"), "tag_id")
               .withColumn("roll_mean", F.avg("value").over(w_hist))
               .withColumn("roll_std",  F.stddev("value").over(w_hist))
               .withColumn("zscore", F.when(F.col("roll_std") > 0,
                                            (F.col("value") - F.col("roll_mean")) / F.col("roll_std")).otherwise(0.0))
               .where("(value > crit_threshold) OR (abs(zscore) > 3)")
               .select("tag_id", "reading_ts"))
hist_scored.createOrReplaceTempView("hist_detections")

recall = spark.sql(f"""
  WITH hits AS (
    SELECT l.anomaly_type, l.severity, l.tag_id, l.start_ts, l.end_ts,
           COUNT(d.reading_ts) AS detections_in_window
    FROM {BREW_FACT_ANOMALY_LABELS} l
    LEFT JOIN hist_detections d
      ON d.tag_id = l.tag_id AND d.reading_ts BETWEEN l.start_ts AND l.end_ts
    GROUP BY l.anomaly_type, l.severity, l.tag_id, l.start_ts, l.end_ts
  )
  SELECT anomaly_type,
         COUNT(*)                                    AS windows,
         SUM(CASE WHEN detections_in_window>0 THEN 1 ELSE 0 END) AS caught,
         ROUND(AVG(CASE WHEN detections_in_window>0 THEN 1.0 ELSE 0.0 END), 2) AS window_recall
  FROM hits GROUP BY anomaly_type ORDER BY window_recall DESC
""")
display(recall)
overall = spark.sql("""
  SELECT ROUND(AVG(caught_flag),2) AS overall_window_recall FROM (
    SELECT CASE WHEN COUNT(d.reading_ts)>0 THEN 1.0 ELSE 0.0 END AS caught_flag
    FROM """ + BREW_FACT_ANOMALY_LABELS + """ l
    LEFT JOIN hist_detections d ON d.tag_id=l.tag_id AND d.reading_ts BETWEEN l.start_ts AND l.end_ts
    GROUP BY l.tag_id, l.start_ts, l.end_ts)
""")
display(overall)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 7 — Visualize the chiller-fault window
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   Plot glycol return temperature around a labeled chiller-drift episode. You should see the
# MAGIC   value <strong>creep above its normal band</strong> well before it would trip a hard alarm —
# MAGIC   which is exactly the early-warning the z-score detector buys you.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 7: Glycol return temp around a chiller-drift window
chiller = spark.sql(f"""
  SELECT reading_ts, value
  FROM {BREW_FACT_READINGS}
  WHERE tag_id='GLY.GLY01.RETURN_TEMP'
    AND reading_ts BETWEEN '2025-06-27 06:00' AND '2025-06-28 06:00'
  ORDER BY reading_ts
""")
display(chiller)   # render as a line chart: reading_ts (x) vs value (y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Wrap-up
# MAGIC You built a **medallion OT pipeline**: Sparkplug-B JSON → **Auto Loader** bronze → ISA-95-
# MAGIC enriched silver → **anomaly gold** (threshold + rolling z-score) → **range-based recall**
# MAGIC against labeled windows. The z-score tier catches *drift* before hard alarms fire — the
# MAGIC difference between a planned fix and a lost batch.
