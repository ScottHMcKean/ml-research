# Databricks notebook source

# MAGIC %md
# MAGIC # Gorda Ridge — Process
# MAGIC
# MAGIC Turns the bronze landings into a single analytics-ready table per
# MAGIC sensor stream:
# MAGIC
# MAGIC | Silver table   | Built from                                       |
# MAGIC | -------------- | ------------------------------------------------ |
# MAGIC | `silver_xrf`   | `bronze_xrf_geochem` ∪ `bronze_xrf_soil` + locs  |
# MAGIC | `silver_mscl`  | `bronze_mscl` + locs                             |
# MAGIC
# MAGIC Adds:
# MAGIC * a `site` column (`GC01-Y` → `GC01`) so we can roll up across the
# MAGIC   X/Y/Z section splits;
# MAGIC * `log_{El}` columns (log1p) since elemental ppm spans several orders
# MAGIC   of magnitude;
# MAGIC * lat / lon / water depth from the locations table.

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "ml"
SCHEMA = "gordaridge"

XRF_ELEMENTS = ("Mn", "Fe", "Cu", "Pb", "S", "Ca")

# COMMAND ----------

def site_from_core(col="core_id"):
    return F.regexp_replace(F.col(col), r"-[XYZ]$", "")


def write(df, name):
    (df.write.mode("overwrite").option("overwriteSchema", "true")
       .saveAsTable(f"{CATALOG}.{SCHEMA}.{name}"))
    n = spark.table(f"{CATALOG}.{SCHEMA}.{name}").count()
    print(f"wrote {CATALOG}.{SCHEMA}.{name}: {n:,} rows")


locations = spark.table(f"{CATALOG}.{SCHEMA}.bronze_locations")

# COMMAND ----------

# MAGIC %md ## silver_xrf

# COMMAND ----------

xrf = (
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_xrf_geochem")
    .unionByName(spark.table(f"{CATALOG}.{SCHEMA}.bronze_xrf_soil"))
)
for el in XRF_ELEMENTS:
    xrf = xrf.withColumn(
        f"log_{el}", F.log1p(F.greatest(F.col(f"{el}_ppm"), F.lit(0.0)))
    )
silver_xrf = (
    xrf.withColumn("site", site_from_core())
    .join(locations, F.col("site") == locations["core_id"], "left")
    .drop(locations["core_id"])
)
write(silver_xrf, "silver_xrf")

# COMMAND ----------

# MAGIC %md ## silver_mscl

# COMMAND ----------

silver_mscl = (
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_mscl")
    .withColumn("site", site_from_core())
    .join(locations, F.col("site") == locations["core_id"], "left")
    .drop(locations["core_id"])
)
write(silver_mscl, "silver_mscl")

# COMMAND ----------

# MAGIC %md ## Sanity checks

# COMMAND ----------

display(spark.sql(f"""
  SELECT mode, COUNT(*) AS n_rows, COUNT(DISTINCT core_id) AS n_cores
  FROM {CATALOG}.{SCHEMA}.silver_xrf
  GROUP BY mode
  ORDER BY mode
"""))

# COMMAND ----------

display(spark.sql(f"""
  SELECT site, mode, COUNT(*) AS n_samples,
         AVG(latitude) AS lat, AVG(longitude) AS lon,
         AVG(water_depth_m) AS water_depth
  FROM {CATALOG}.{SCHEMA}.silver_xrf
  GROUP BY site, mode
  ORDER BY site, mode
"""))
