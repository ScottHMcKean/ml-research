# Databricks notebook source

# MAGIC %md
# MAGIC # ML Lineage — 03 · Lineage Investigation (the proof)
# MAGIC
# MAGIC This is the notebook that answers the customer's question. We inspect what Unity Catalog
# MAGIC lineage captured for the two prediction tables, using two independent sources of truth:
# MAGIC
# MAGIC 1. **SQL on the `system.access` lineage tables** — `table_lineage` and `column_lineage`
# MAGIC    (the queryable, auditable source the customer asked for).
# MAGIC 2. **The lineage-tracking REST API** — near-real-time, backs the Catalog Explorer UI.
# MAGIC
# MAGIC What we expect to find, and will confirm below:
# MAGIC
# MAGIC | Relationship | Captured by UC? |
# MAGIC |--------------|-----------------|
# MAGIC | source table → prediction table | ✅ yes (table→table) |
# MAGIC | feature table → prediction table (FE path only) | ✅ yes (`score_batch` reads it) |
# MAGIC | **model → prediction table** | ❌ **no** — the customer's gap |
# MAGIC | feature table → model | ✅ yes, but only in the model's own lineage (not the pred table) |
# MAGIC | training table → model | ✅ yes (via `mlflow.log_input`) |
# MAGIC
# MAGIC > ⚠️ **`system.access` lineage has ingestion latency** (typically a few minutes, can be
# MAGIC > longer). If the SQL cells come back empty right after running `02`, wait and re-run —
# MAGIC > the REST API section reflects lineage almost immediately.

# COMMAND ----------

dbutils.widgets.text("catalog", "shm_catalog")
dbutils.widgets.text("schema", "ml")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.lineage_source"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.lineage_features"
PRED_PLAIN = f"{CATALOG}.{SCHEMA}.pred_plain"
PRED_FE = f"{CATALOG}.{SCHEMA}.pred_fe"
MODEL_PLAIN = f"{CATALOG}.{SCHEMA}.model_plain"
MODEL_FE = f"{CATALOG}.{SCHEMA}.model_fe"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Table lineage via SQL — plain prediction table
# MAGIC Everything upstream of `pred_plain`. Note the columns available: `source_table_full_name`,
# MAGIC `entity_type`, `entity_id`. **There is no "model" column** — the schema can only express
# MAGIC table/path sources and the entity (notebook/job/pipeline) that ran the operation.

# COMMAND ----------

display(spark.sql(f"""
    SELECT source_table_full_name,
           source_type,
           entity_type,          -- NOTEBOOK / JOB / PIPELINE ... never MODEL
           entity_id,
           created_by,
           MAX(event_time) AS last_seen
    FROM system.access.table_lineage
    WHERE target_table_full_name = '{PRED_PLAIN}'
    GROUP BY ALL
    ORDER BY last_seen DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Table lineage via SQL — Feature Engineering prediction table
# MAGIC The FE path gives **richer table lineage**: `score_batch` reads the feature table, so
# MAGIC `lineage_features` shows up as an upstream source of `pred_fe`. Still no model node, but
# MAGIC the feature dependency is at least visible here.

# COMMAND ----------

display(spark.sql(f"""
    SELECT source_table_full_name,
           source_type,
           entity_type,
           entity_id,
           MAX(event_time) AS last_seen
    FROM system.access.table_lineage
    WHERE target_table_full_name = '{PRED_FE}'
    GROUP BY ALL
    ORDER BY last_seen DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Column lineage via SQL
# MAGIC Column-level lineage for the `prediction` column. Useful to confirm that even at the
# MAGIC column grain, the model is not represented — the prediction column simply has no tracked
# MAGIC upstream (it was produced by an opaque UDF / `score_batch`), while pass-through columns do.

# COMMAND ----------

display(spark.sql(f"""
    SELECT target_column_name,
           source_table_full_name,
           source_column_name,
           entity_type
    FROM system.access.column_lineage
    WHERE target_table_full_name = '{PRED_PLAIN}'
    ORDER BY target_column_name
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. The proof: no MODEL entity anywhere
# MAGIC Sweep every lineage row that touches our two prediction tables and list the distinct
# MAGIC `source_type` / `entity_type` values. If UC tracked model→table lineage, a model entity
# MAGIC would appear here. It does not.

# COMMAND ----------

display(spark.sql(f"""
    SELECT 'source_type' AS field, source_type AS value, COUNT(*) AS n
    FROM system.access.table_lineage
    WHERE target_table_full_name IN ('{PRED_PLAIN}', '{PRED_FE}')
    GROUP BY source_type
    UNION ALL
    SELECT 'entity_type' AS field, entity_type AS value, COUNT(*) AS n
    FROM system.access.table_lineage
    WHERE target_table_full_name IN ('{PRED_PLAIN}', '{PRED_FE}')
    GROUP BY entity_type
    ORDER BY field, n DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cross-check with the lineage-tracking REST API
# MAGIC The REST API backs Catalog Explorer's **Lineage** tab and updates almost immediately
# MAGIC (no `system.access` ingestion lag). We list the **upstream** entities it returns for each
# MAGIC prediction table. Each entry carries a `tableInfo` and `notebookInfos`/`jobInfos` — but
# MAGIC **never a `modelInfo`**. There is simply no model on the upstream side of a prediction
# MAGIC table. (Observed: `pred_plain` ← `lineage_source`; `pred_fe` ← `lineage_source` +
# MAGIC `lineage_features`.)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()


def table_lineage(table_name: str) -> dict:
    return w.api_client.do(
        "GET",
        "/api/2.0/lineage-tracking/table-lineage",
        query={"table_name": table_name, "include_entity_lineage": True},
    )


for tbl in (PRED_PLAIN, PRED_FE):
    upstreams = table_lineage(tbl).get("upstreams", []) or []
    keys = sorted({k for u in upstreams for k in u.keys()})
    print(f"\n=== upstream of {tbl} ===")
    print("  entry keys seen:", keys, "  <-- note: no 'modelInfo'")
    for u in upstreams:
        if "tableInfo" in u:
            ti = u["tableInfo"]
            print(f"  TABLE  {ti.get('catalog_name')}.{ti.get('schema_name')}.{ti.get('name')}")
        if "modelInfo" in u:
            print(f"  MODEL  {u['modelInfo'].get('model_name')}   <-- would appear here if tracked")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model lineage that UC *does* track — on the INPUT side
# MAGIC UC absolutely represents model nodes in lineage — just pointing **into** the model, not
# MAGIC out to its predictions. Look at the **downstream** of the input tables: a `modelInfo`
# MAGIC node appears there. Both paths produce it:
# MAGIC - `lineage_source` → `model_plain` (from `mlflow.log_input` at training time), and
# MAGIC - `lineage_features` → `model_fe` (from `fe.log_model`).
# MAGIC
# MAGIC This is the asymmetry in one screen: **input table → model** is tracked; **model →
# MAGIC output table** is not.

# COMMAND ----------

for tbl in (SOURCE_TABLE, FEATURE_TABLE):
    downstreams = table_lineage(tbl).get("downstreams", []) or []
    print(f"\n=== downstream of {tbl} ===")
    for d in downstreams:
        if "tableInfo" in d:
            ti = d["tableInfo"]
            print(f"  TABLE  {ti.get('name')}")
        if "modelInfo" in d:
            print(f"  MODEL  {d['modelInfo'].get('model_name')}   <-- model IS tracked as a downstream of its input table")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6b. The FE model also records its feature dependency on the model version itself
# MAGIC `fe.log_model` writes `model_version_dependencies` onto the registered model version —
# MAGIC queryable straight from the models API. (Plain `mlflow.log_input` does **not** populate
# MAGIC this field; it only produces the table-lineage node shown in §6.)

# COMMAND ----------

for model_name in (MODEL_FE, MODEL_PLAIN):
    mv = w.api_client.do(
        "GET", f"/api/2.1/unity-catalog/models/{model_name}/versions/1"
    )
    print(f"{model_name} v1 model_version_dependencies:",
          json.dumps(mv.get("model_version_dependencies")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. The recommended workaround — reconstruct model → table in SQL
# MAGIC Because we stamped model provenance into the prediction tables in notebook `02`, the
# MAGIC model→table relationship is fully recoverable with a plain SQL query. This is the
# MAGIC pattern to recommend to the customer today: **carry the model name / version / run-id as
# MAGIC columns** (and optionally log an output snapshot artifact to the MLflow run).

# COMMAND ----------

display(spark.sql(f"""
    SELECT model_name, model_version, model_run_id, COUNT(*) AS n_rows, MAX(scored_at) AS scored_at
    FROM {PRED_PLAIN}
    GROUP BY model_name, model_version, model_run_id
    UNION ALL
    SELECT model_name, model_version, model_run_id, COUNT(*), MAX(scored_at)
    FROM {PRED_FE}
    GROUP BY model_name, model_version, model_run_id
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - **Table→table lineage is captured** for both prediction tables (§1, §2). The FE path
# MAGIC   additionally exposes the feature table as an upstream source (§2).
# MAGIC - **No model entity appears** in `system.access.table_lineage` / `column_lineage`, nor in
# MAGIC   the lineage REST API response, for either prediction table (§3–§5). Model→prediction-table
# MAGIC   lineage is **not tracked** — this is an expected limitation, not a bug.
# MAGIC - **Model lineage that UC *does* track** points *into* the model (feature/training table →
# MAGIC   model, §6), never toward the model's output tables.
# MAGIC - **Workaround** (§7): stamp `model_name` / `model_version` / `model_run_id` into the
# MAGIC   prediction table to make the relationship queryable and auditable today.
# MAGIC
# MAGIC See `README.md` for the full customer-facing write-up and recommendations.
