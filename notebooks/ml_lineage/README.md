# ML Lineage in Unity Catalog — can a prediction table trace back to its model?

A hands-on walkthrough that answers a real customer question raised during UAT:

> The prediction table is successfully generated from the ML model, but Unity Catalog
> lineage does not show the relationship between the prediction table and the ML model
> used to generate the predictions. Is this a known limitation or a bug? Is there a
> workaround or an expected fix timeline?

## Short answer

**This is an expected limitation of Unity Catalog lineage today — not a bug.**

Unity Catalog lineage is fundamentally **data lineage** (table→table, column→column) plus
the **entity** that ran the operation (notebook / job / pipeline / dashboard / query). An
MLflow model applied inside a batch-scoring job — whether via `mlflow.pyfunc.spark_udf`,
`mlflow.pyfunc.load_model`, or `FeatureEngineeringClient.score_batch` — is **not** recorded
as an upstream node of the output table. There is no MLflow "log output table" API that
mirrors `mlflow.log_input`, so nothing links the prediction table back to the model version.

Model lineage *does* exist in UC, but it only points **into** the model
(training/feature table → model), never **out** to the model's prediction tables.

## What UC lineage captures vs. what it doesn't

| Relationship | Tracked? | Where you see it |
|---|---|---|
| source table → prediction table | ✅ yes | `system.access.table_lineage`, Catalog Explorer, REST API |
| feature table → prediction table (FE `score_batch` path) | ✅ yes | same — `score_batch` reads the feature table |
| training / input table → model | ✅ yes (via `mlflow.log_input`) | table-lineage API (model node **downstream** of the table); model Lineage tab |
| feature table → model | ✅ yes (via `fe.log_model`) | same, **plus** `model_version_dependencies` on the model version |
| **model → prediction table** | ❌ **no** | — (this is the customer's gap) |
| model → serving inference table (online) | ✅ yes | AI Gateway / serving lineage (different feature) |

**The core asymmetry:** UC lineage represents model nodes only on the **input** side of a
model (input/feature table → model). A prediction table's **upstream** list contains tables
and notebooks/jobs — never a model. So you can trace *into* a model, but not *out* to what
it produced.

Key nuance: the **Feature Engineering path is strictly better for lineage**. Because
`score_batch` physically reads the feature table, that dependency shows up as an upstream
source of the prediction table — and the feature table→model edge is captured on the model
side. It still does **not** draw a model→prediction-table edge, but it leaves the richest
trail.

## Empirical results (verified on Databricks, Aug 2026)

Ran the four notebooks end-to-end (models trained + scored on a UC-enabled cluster) and
queried the lineage-tracking REST API directly:

```
upstream of ...pred_plain :  TABLE lineage_source            (+ notebook)   → no modelInfo
upstream of ...pred_fe    :  TABLE lineage_source,
                             TABLE lineage_features           (+ notebook)   → no modelInfo

downstream of lineage_source   :  TABLE pred_plain, TABLE pred_fe, MODEL model_plain
downstream of lineage_features :  TABLE pred_fe,               MODEL model_fe

model_fe    v1 model_version_dependencies = {"dependencies":[{"table":{"table_full_name":"...lineage_features"}}]}
model_plain v1 model_version_dependencies = null   (log_input still yields the table-lineage MODEL node above)
```

Reading of the evidence: every prediction-table **upstream** entry carried only `tableInfo`
and `notebookInfos` — **no `modelInfo`** in either the plain or the Feature Engineering path.
Yet a `MODEL` node *does* appear **downstream of the input tables**, confirming UC tracks
model lineage on the input side only. This is a platform behavior, not a bug in the workload.

And the same result from the **SQL** side (`system.access.table_lineage`), once it propagated
(~9 min after the run):

```sql
SELECT source_table_full_name, source_type, entity_type
FROM system.access.table_lineage
WHERE target_table_full_name = 'procurement_demo.ml_lineage.pred_plain';
--  procurement_demo.ml_lineage.lineage_source | TABLE | NOTEBOOK      (pred_fe also lists lineage_features)
```

- Rows in `system.access.table_lineage` referencing either model as a source or target: **0**.
- Distinct `entity_type` values for our prediction tables: **only `NOTEBOOK`**.
- The table's **schema has no model column** — just `source_table_*`, `target_table_*`, and
  `entity_type` (NOTEBOOK/JOB/PIPELINE/DASHBOARD). It structurally cannot express a
  model→table edge.

## Recommended workarounds (available today)

1. **Stamp model provenance into the prediction table** *(primary recommendation)* — write
   `model_name`, `model_version`, and `model_run_id` as columns alongside the predictions.
   The model↔table relationship becomes a trivial SQL query and is fully auditable. This
   walkthrough does exactly this in `02_batch_inference.py` and queries it back in `03`.
2. **Log an output snapshot/manifest artifact to the MLflow run** (`mlflow.log_artifact`)
   so the run itself references what it produced.
3. **For online/served models**, use Model Serving inference tables — those carry their own
   model-associated lineage (a separate mechanism from batch table lineage).

## How to query lineage (SQL first, UI second)

**SQL (the auditable path the customer asked for):**
```sql
-- Everything upstream of the prediction table
SELECT source_table_full_name, source_type, entity_type, entity_id, created_by
FROM system.access.table_lineage
WHERE target_table_full_name = 'shm_catalog.ml.pred_plain';
```
The schema has no "model" column — only table/path sources and the entity (notebook/job)
that ran the write. Sweeping distinct `source_type`/`entity_type` values confirms no model
entity ever appears. (`system.access` lineage has a few-minutes ingestion latency.)

**REST API (near-real-time, backs Catalog Explorer):**
```bash
databricks api GET "/api/2.0/lineage-tracking/table-lineage?table_name=shm_catalog.ml.pred_plain&include_entity_lineage=true"
```
The response returns `tableInfos` and `notebookInfos`/`jobInfos`, but there is no
`modelInfos` for an upstream model of a table.

**UI (second-best):** Catalog Explorer → the prediction table → **Lineage** tab shows the
source table and the notebook/job — but no model. The model→feature lineage lives on the
**model's** own Lineage tab instead.

## Fix timeline

There is no committed GA timeline for model→output-table lineage as of this writing. The
right move is to (a) confirm with the account team / file a feature request referencing this
gap, and (b) adopt the provenance-column workaround, which fully satisfies the audit /
traceability requirement in the meantime.

## The notebooks

Run in order on a workspace with Unity Catalog. They target `shm_catalog.ml` — change the
`CATALOG`/`SCHEMA` params at the top of each notebook to your own location.

| Notebook | What it does |
|---|---|
| `00_setup_and_features.py` | Creates the schema, a synthetic **source table**, and a governed **UC feature table**. |
| `01_train_models.py` | Trains & registers **two** UC models: a plain model (`mlflow.log_input`) and a Feature Engineering model (`fe.log_model`). |
| `02_batch_inference.py` | Scores both — `spark_udf` → `pred_plain`, `fe.score_batch` → `pred_fe` — and stamps model provenance columns. |
| `03_lineage_investigation.py` | The proof: SQL on `system.access.{table,column}_lineage`, the lineage REST API, feature→model lineage, and the workaround reconstruction. |
