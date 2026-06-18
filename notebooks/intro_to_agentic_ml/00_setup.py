# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop setup — shared config (`%run` from every notebook)
# MAGIC
# MAGIC This notebook defines the workspace-wide variables — catalog, schemas, volume,
# MAGIC warehouse, and the **table-name contract** — so the labs themselves stay short.
# MAGIC Every lab notebook starts with `%run ./00_setup`.
# MAGIC
# MAGIC **What it does**
# MAGIC 1. Reads config from widgets (change `catalog` to land in a different sandbox).
# MAGIC 2. Creates the catalog + `sales` / `brewery` schemas + the `landing` volume (idempotent).
# MAGIC 3. Sets session-default catalog so SQL can use short names.
# MAGIC 4. Defines fully-qualified table-name constants (`SALES_*`, `BREW_*`) used across labs.
# MAGIC
# MAGIC **Idempotent.** Safe to re-run.

# COMMAND ----------

# DBTITLE 1,Widgets — change the catalog name here to use your own sandbox
from databricks.sdk.runtime import dbutils, spark
import re

# Single shared catalog by default. For multi-attendee delivery where each person
# needs isolation, set the `catalog` widget (or --var catalog=...) to e.g.
# <username>_ml_workshop. The DATA GENERATION notebooks populate the shared
# read-only tables; lab outputs use CREATE OR REPLACE so re-runs stay clean.
# Default = the catalog the FEVM pre-provisions for this workspace
# (`ml_workshop`). On an FEVM the deploying user is workspace admin but
# NOT metastore admin, so they cannot CREATE CATALOG — but they OWN the
# pre-provisioned one with ALL PRIVILEGES. The create-catalog step below is
# therefore guarded.
dbutils.widgets.text("catalog",           "ml_workshop", "1. Catalog")
dbutils.widgets.text("sales_schema",      "sales",        "2. Sales schema")
dbutils.widgets.text("brewery_schema",    "brewery",      "3. Brewery schema")
dbutils.widgets.text("volume",            "landing",      "4. Brewery landing volume")

CATALOG    = dbutils.widgets.get("catalog").strip()           or "ml_workshop"
SALES      = dbutils.widgets.get("sales_schema").strip()      or "sales"
BREWERY    = dbutils.widgets.get("brewery_schema").strip()    or "brewery"
VOLUME     = dbutils.widgets.get("volume").strip()            or "landing"

for label, val in [("catalog", CATALOG), ("sales_schema", SALES),
                   ("brewery_schema", BREWERY), ("volume", VOLUME)]:
    if not re.match(r"^[a-z][a-z0-9_]*$", val):
        raise ValueError(f"{label}={val!r} must be lowercase alphanumeric + underscore, starting with a letter")

print("Workshop config:")
print(f"  catalog        : {CATALOG}")
print(f"  sales schema   : {SALES}")
print(f"  brewery schema : {BREWERY}")
print(f"  brewery volume : {VOLUME}")

# COMMAND ----------

# DBTITLE 1,Create catalog + schemas + volume (idempotent)
# CREATE CATALOG needs metastore-level privilege the FEVM user lacks. If the
# catalog already exists (the pre-provisioned case), skip; only attempt to
# create when it's genuinely absent (e.g. a self-hosted workspace where the
# user IS metastore admin).
_existing = {r["catalog"] for r in spark.sql("SHOW CATALOGS").collect()}
if CATALOG not in _existing:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    print(f"Created catalog {CATALOG}")
else:
    print(f"Using existing catalog {CATALOG} (no CREATE needed)")
for sch in (SALES, BREWERY):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{sch}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{BREWERY}.{VOLUME}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SALES}")
print(f"Ready: {CATALOG}  ·  schemas [{SALES}, {BREWERY}]  ·  /Volumes/{CATALOG}/{BREWERY}/{VOLUME}")

# COMMAND ----------

# DBTITLE 1,Table-name contract — every notebook references these constants
def sales_t(name: str) -> str:
    """Fully-qualified name in the sales schema."""
    return f"{CATALOG}.{SALES}.{name}"

def brew_t(name: str) -> str:
    """Fully-qualified name in the brewery schema."""
    return f"{CATALOG}.{BREWERY}.{name}"

# --- sales schema (Lab 1 recommender + Lab 3 forecasting) ---
SALES_DIM_PRODUCT      = sales_t("dim_product")
SALES_DIM_ACCOUNT      = sales_t("dim_account")
SALES_DIM_DISTRIBUTOR  = sales_t("dim_distributor")
SALES_DIM_DATE         = sales_t("dim_date")
SALES_FACT_DEPLETIONS  = sales_t("fact_depletions")
SALES_FACT_ASSORTMENT  = sales_t("fact_assortment")
SALES_RECOMMENDATIONS  = sales_t("account_sku_recommendations")  # Lab 1 output
SALES_DEMAND_FORECAST  = sales_t("demand_forecast")              # Lab 3 output

# --- brewery schema (Lab 2) ---
BREW_DIM_ASSET            = brew_t("dim_asset")
BREW_DIM_TAG              = brew_t("dim_tag")
BREW_FACT_READINGS        = brew_t("fact_sensor_readings")        # batch history (~15.6M rows)
BREW_FACT_ANOMALY_LABELS  = brew_t("fact_anomaly_labels")         # ground-truth windows
BREW_BRONZE               = brew_t("bronze_readings")             # Auto Loader target
BREW_SILVER               = brew_t("silver_readings")             # enriched w/ thresholds
BREW_GOLD_ANOMALIES       = brew_t("gold_anomalies")             # detected anomalies

VOLUME_PATH = f"/Volumes/{CATALOG}/{BREWERY}/{VOLUME}"

print("Defined table constants: SALES_*, BREW_*")
print(f"  VOLUME_PATH = {VOLUME_PATH}")

# COMMAND ----------

# DBTITLE 1,Resolve a serverless SQL warehouse (best-effort, for Genie/dashboards)
try:
    from databricks.sdk import WorkspaceClient
    _w = WorkspaceClient()
    _cands = [w for w in _w.warehouses.list() if w.state and w.state.value in ("RUNNING", "STARTING", "STOPPED")]
    _serverless = [w for w in _cands if getattr(w, "enable_serverless_compute", False)]
    DEFAULT_WAREHOUSE_ID = (_serverless or _cands)[0].id if _cands else ""
except Exception as exc:
    DEFAULT_WAREHOUSE_ID = ""
    print(f"Could not auto-discover a warehouse: {exc}")
print(f"  default warehouse_id : {DEFAULT_WAREHOUSE_ID or '(none found)'}")
