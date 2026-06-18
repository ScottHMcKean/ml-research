# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="background:#1B3139; color:white; padding:24px; border-radius:8px;">
# MAGIC   <h1 style="margin:0; color:white;">Lab 1 — B2B Store → SKU Recommender</h1>
# MAGIC   <p style="margin:6px 0 0 0; color:#9EB7BE; font-size:16px;">
# MAGIC     Persona: <strong style="color:white;">Data Scientist / Analyst</strong>
# MAGIC     &nbsp;·&nbsp; Estimated time: 60 min &nbsp;·&nbsp; Agenda slot: AI/ML on Databricks
# MAGIC   </p>
# MAGIC </div>
# MAGIC
# MAGIC **The business question:** *Which SKUs should each retail account add next?*
# MAGIC A beverage company's distributor reps walk into hundreds of stores. Today they guess at the
# MAGIC "next best SKU" to pitch. We'll learn it from **depletion history** — what each store
# MAGIC actually sells — and recommend high-affinity SKUs the store **doesn't yet carry**.
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #FF3620; padding:15px; margin:15px 0;">
# MAGIC   <strong>The technique — collaborative filtering (ALS).</strong> Stores that sell similar
# MAGIC   products have similar taste. <strong>Alternating Least Squares</strong> (Spark MLlib)
# MAGIC   factorizes the store × SKU "how much did they sell" matrix into latent taste vectors,
# MAGIC   then predicts affinity for SKUs a store has never carried. We use the
# MAGIC   <strong>implicit-feedback</strong> variant: cases sold = a confidence signal, not a star rating.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 How this lab works — Genie Code.</strong> At each step you'll see a
# MAGIC   <em>prompt to paste</em>. Press <kbd>Cmd</kbd>+<kbd>I</kbd> (Mac) / <kbd>Ctrl</kbd>+<kbd>I</kbd>
# MAGIC   in the cell below the instructions, paste the prompt, and let Genie write the code. The
# MAGIC   reference solution is also provided so you're never stuck — but try Genie first.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#FFE0E0; border-left:5px solid #C0392B; padding:15px; margin:15px 0;">
# MAGIC   <strong>⚠️ Compute requirement — this notebook needs a CLASSIC single-user cluster.</strong>
# MAGIC   Spark MLlib's <code>ALS.recommendForAllUsers()</code> relies on Spark higher-order array
# MAGIC   functions that <em>Unity-Catalog Serverless does not permit</em> (you'll hit
# MAGIC   <code>UC_COMMAND_NOT_SUPPORTED</code>). Attach this notebook to a CLASSIC single-user
# MAGIC   cluster (DBR ML). On a serverless-only workspace, the equivalent result is reachable with
# MAGIC   the <code>implicit</code> library (scipy sparse ALS) — but this notebook is the canonical
# MAGIC   Spark-MLlib reference for classic-cluster / production environments.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0 — Shared setup
# MAGIC Every lab notebook starts with the same `%run`: it sets the catalog / schemas and defines
# MAGIC the table-name constants (`SALES_*`) the steps below reference.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0.1 — Prerequisite check
# MAGIC Confirm the sales data exists (built once by `01_generate_sales.py`).

# COMMAND ----------

# DBTITLE 1,Confirm the source tables are present
for tbl in (SALES_FACT_DEPLETIONS, SALES_FACT_ASSORTMENT, SALES_DIM_PRODUCT, SALES_DIM_ACCOUNT):
    n = spark.table(tbl).count()
    print(f"  {tbl:55s} {n:>10,} rows")
print("\nIf any are missing, run 01_generate_sales.py first.")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 1 — Explore the depletion data
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Genie Code prompt</strong> &nbsp;(paste in the cell below with <kbd>Cmd</kbd>+<kbd>I</kbd>):
# MAGIC   <blockquote style="border-left:3px solid #1B5161; margin:8px 0; padding:4px 12px; color:#1B3139;">
# MAGIC   Using the table identified by the Python variable SALES_FACT_DEPLETIONS joined to
# MAGIC   SALES_DIM_PRODUCT on sku_id, write Spark SQL via spark.sql() that returns total cases
# MAGIC   and net_revenue by brand, sorted by cases descending, and display() it.
# MAGIC   </blockquote>
# MAGIC   <strong>Expected:</strong> Thirsty Otter on top, then Lazy Llama / Fancy Flamingo, with the
# MAGIC   newer brands (Hydro Hippo, Moose Juice) much smaller.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 1: Volume by brand  (← try Genie here, reference below)
display(spark.sql(f"""
  SELECT p.brand,
         SUM(f.cases)        AS total_cases,
         ROUND(SUM(f.net_revenue),0) AS total_revenue,
         COUNT(DISTINCT f.account_id) AS accounts_selling
  FROM {SALES_FACT_DEPLETIONS} f
  JOIN {SALES_DIM_PRODUCT}     p USING (sku_id)
  GROUP BY p.brand
  ORDER BY total_cases DESC
"""))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 2 — Build the implicit-feedback matrix
# MAGIC
# MAGIC ALS needs three numeric columns: a **user id**, an **item id**, and a **rating**
# MAGIC (here, a confidence weight). Our ids are strings (`ACC-00042`, `WC-BCH-12`), so we map
# MAGIC them to integers first. The "rating" is total cases the store has sold of that SKU —
# MAGIC log-scaled so a 500-case grocery chain doesn't completely drown out a 20-case bar.
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Genie Code prompt:</strong>
# MAGIC   <blockquote style="border-left:3px solid #1B5161; margin:8px 0; padding:4px 12px; color:#1B3139;">
# MAGIC   From the table in SALES_FACT_DEPLETIONS, aggregate total cases per (account_id, sku_id).
# MAGIC   Use StringIndexer to create integer columns account_idx and sku_idx. Add a column
# MAGIC   confidence = log(1 + total_cases). Return a Spark DataFrame called ratings.
# MAGIC   </blockquote>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 2: Aggregate + index into a ratings matrix
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer

pairs = (spark.table(SALES_FACT_DEPLETIONS)
         .groupBy("account_id", "sku_id")
         .agg(F.sum("cases").alias("total_cases"))
         .where("total_cases > 0"))

acc_indexer = StringIndexer(inputCol="account_id", outputCol="account_idx", handleInvalid="skip")
sku_indexer = StringIndexer(inputCol="sku_id",     outputCol="sku_idx",     handleInvalid="skip")
acc_model = acc_indexer.fit(pairs)
sku_model = sku_indexer.fit(pairs)

ratings = (sku_model.transform(acc_model.transform(pairs))
           .withColumn("confidence", F.log(F.lit(1.0) + F.col("total_cases")))
           .withColumn("account_idx", F.col("account_idx").cast("int"))
           .withColumn("sku_idx",     F.col("sku_idx").cast("int")))

print(f"ratings rows: {ratings.count():,}  ·  "
      f"accounts: {ratings.select('account_idx').distinct().count():,}  ·  "
      f"skus: {ratings.select('sku_idx').distinct().count()}")
display(ratings.limit(5))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 3 — Train ALS (implicit feedback)
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #1B5161; padding:15px; margin:15px 0;">
# MAGIC   <strong>The knobs that matter:</strong>
# MAGIC   <ul style="margin:6px 0 0 0;">
# MAGIC     <li><code>implicitPrefs=True</code> — treat the rating as confidence, not an explicit score.</li>
# MAGIC     <li><code>rank</code> — size of each latent taste vector (10–20 is plenty here).</li>
# MAGIC     <li><code>regParam</code> — guards against overfitting the heavy hitters.</li>
# MAGIC     <li><code>alpha</code> — how strongly "they sold a lot" implies preference.</li>
# MAGIC     <li><code>coldStartStrategy="drop"</code> — don't emit NaN predictions for unseen pairs.</li>
# MAGIC   </ul>
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Genie Code prompt:</strong>
# MAGIC   <blockquote style="border-left:3px solid #1B5161; margin:8px 0; padding:4px 12px; color:#1B3139;">
# MAGIC   Train a Spark MLlib ALS recommender on the DataFrame `ratings` with userCol account_idx,
# MAGIC   itemCol sku_idx, ratingCol confidence, implicitPrefs True, rank 15, regParam 0.08,
# MAGIC   alpha 25, coldStartStrategy drop, seed 42. Fit it and call the model als_model.
# MAGIC   </blockquote>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 3: Fit the ALS model
from pyspark.ml.recommendation import ALS

als = ALS(userCol="account_idx", itemCol="sku_idx", ratingCol="confidence",
          implicitPrefs=True, rank=15, regParam=0.08, alpha=25.0,
          coldStartStrategy="drop", nonnegative=True, seed=42)
als_model = als.fit(ratings)
print("ALS trained. rank =", als_model.rank)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 4 — Generate Top-N and filter to the GAP
# MAGIC
# MAGIC The model will happily recommend SKUs a store already carries — useless to a rep. The
# MAGIC value is the **gap**: high-affinity SKUs the store does **not** currently stock
# MAGIC (`fact_assortment` = what each account carries today). We generate the top 10 per account,
# MAGIC then anti-join the assortment to keep only genuinely new recommendations.
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Genie Code prompt:</strong>
# MAGIC   <blockquote style="border-left:3px solid #1B5161; margin:8px 0; padding:4px 12px; color:#1B3139;">
# MAGIC   Use als_model.recommendForAllUsers(10) to get the top 10 sku_idx per account_idx,
# MAGIC   explode the recommendations array into rows with columns sku_idx and rating, and
# MAGIC   show the result.
# MAGIC   </blockquote>
# MAGIC   We then map the integer indices back to real ids and remove already-carried SKUs.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 4: Top-N recs, mapped back to ids, gap-filtered
from pyspark.ml.feature import IndexToString

# 10 raw recs per account
raw = (als_model.recommendForAllUsers(10)
       .select("account_idx", F.explode("recommendations").alias("rec"))
       .select("account_idx",
               F.col("rec.sku_idx").alias("sku_idx"),
               F.col("rec.rating").alias("affinity")))

# map integer indices back to account_id / sku_id
acc_labels = IndexToString(inputCol="account_idx", outputCol="account_id", labels=acc_model.labelsArray[0])
sku_labels = IndexToString(inputCol="sku_idx",     outputCol="sku_id",     labels=sku_model.labelsArray[0])
recs_named = sku_labels.transform(acc_labels.transform(raw))

# GAP filter: drop SKUs the account already carries (in fact_assortment)
gap_recs = (recs_named.alias("r")
            .join(spark.table(SALES_FACT_ASSORTMENT).alias("a"),
                  on=["account_id", "sku_id"], how="left_anti")
            .join(spark.table(SALES_DIM_PRODUCT).select("sku_id", "brand", "flavor", "pack_config", "category"),
                  on="sku_id", how="left"))

# rank the gap recs within each account
from pyspark.sql.window import Window
w = Window.partitionBy("account_id").orderBy(F.col("affinity").desc())
gap_ranked = (gap_recs.withColumn("rec_rank", F.row_number().over(w))
              .where("rec_rank <= 5")
              .select("account_id", "rec_rank", "sku_id", "brand", "flavor",
                      "pack_config", "category", F.round("affinity", 4).alias("affinity")))
display(gap_ranked.orderBy("account_id", "rec_rank").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Persist the recommendations
# MAGIC Write the gap recommendations to a managed Delta table reps / a Genie space can query.

# COMMAND ----------

# DBTITLE 1,Step 5: Write account_sku_recommendations
(gap_ranked.write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(SALES_RECOMMENDATIONS))
print(f"Wrote {SALES_RECOMMENDATIONS}: {spark.table(SALES_RECOMMENDATIONS).count():,} gap recommendations")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 6 — Sanity-check a recommendation
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #FF3620; padding:15px; margin:15px 0;">
# MAGIC   <strong>Does it make sense?</strong> Pick a liquor account and look at what it already
# MAGIC   sells vs what we recommend. A good rec is on-profile (a liquor store gets a Fancy Flamingo
# MAGIC   or Old Grumpy Bear SKU it lacks, not a hydration drink) and genuinely absent from its assortment.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 6: Inspect one account's profile vs its recommendations
sample_acct = (spark.table(SALES_RECOMMENDATIONS)
               .join(spark.table(SALES_DIM_ACCOUNT).where("banner='liquor'").select("account_id"),
                     "account_id")
               .select("account_id").limit(1).collect()[0]["account_id"])
print(f"Sample account: {sample_acct}")
print("\n— Currently carries (top by trailing cases):")
display(spark.sql(f"""
  SELECT a.sku_id, p.brand, p.flavor, p.category, a.trailing_26w_cases
  FROM {SALES_FACT_ASSORTMENT} a JOIN {SALES_DIM_PRODUCT} p USING (sku_id)
  WHERE a.account_id = '{sample_acct}' ORDER BY a.trailing_26w_cases DESC LIMIT 8
"""))
print("\n— We recommend it ADD:")
display(spark.sql(f"""
  SELECT rec_rank, sku_id, brand, flavor, category, affinity
  FROM {SALES_RECOMMENDATIONS} WHERE account_id = '{sample_acct}' ORDER BY rec_rank
"""))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 7 (stretch) — Wrap as a UC function for Genie
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   Registering the recommendations behind a <strong>UC function</strong> means a business
# MAGIC   user can ask a Genie space <em>"what should account ACC-00042 add?"</em> and Genie will
# MAGIC   discover and call it — no SQL required.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 7: recommend_skus(account_id, n) UC function
spark.sql(f"""
CREATE OR REPLACE FUNCTION {SALES}.recommend_skus(in_account_id STRING, n INT)
RETURNS TABLE(rec_rank INT, sku_id STRING, brand STRING, flavor STRING, category STRING, affinity DOUBLE)
COMMENT 'Top-N next-best SKUs a retail account should add (gap recommendations from the ALS model).'
RETURN
  SELECT rec_rank, sku_id, brand, flavor, category, affinity
  FROM {SALES_RECOMMENDATIONS}
  WHERE account_id = in_account_id AND rec_rank <= n
  ORDER BY rec_rank
""")
print(f"Created UC function {SALES}.recommend_skus(account_id, n)")
display(spark.sql(f"SELECT * FROM {SALES}.recommend_skus('{sample_acct}', 3)"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Appendix — Non-ML alternative (market-basket co-occurrence)
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #1B5161; padding:15px; margin:15px 0;">
# MAGIC   Not comfortable with ALS? The same "stores like yours also carry…" intuition can be
# MAGIC   expressed as a pure-SQL co-occurrence query: for SKUs a store lacks, count how many of
# MAGIC   its <em>peer</em> stores (that overlap on what it does carry) sell them. Simpler,
# MAGIC   explainable, and a useful baseline to compare ALS against.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Appendix: item-item co-occurrence baseline (SQL)
display(spark.sql(f"""
WITH carried AS (
  SELECT account_id, sku_id FROM {SALES_FACT_ASSORTMENT}
),
-- SKUs that co-occur in the same store, with co-occurrence counts
cooc AS (
  SELECT a.sku_id AS has_sku, b.sku_id AS also_sku, COUNT(*) AS stores
  FROM carried a JOIN carried b ON a.account_id = b.account_id AND a.sku_id <> b.sku_id
  GROUP BY a.sku_id, b.sku_id
)
SELECT has_sku, also_sku, stores
FROM cooc ORDER BY stores DESC LIMIT 15
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Wrap-up
# MAGIC You built a **collaborative-filtering recommender** end-to-end: implicit-feedback matrix →
# MAGIC ALS → top-N → **gap filter** → a Delta table → a **UC function** a Genie space can call.
# MAGIC The same pattern powers next-best-SKU, cross-sell, and assortment-optimization across CPG.
