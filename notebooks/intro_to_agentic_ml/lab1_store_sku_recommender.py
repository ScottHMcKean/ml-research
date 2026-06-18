# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="background:#1B3139; color:white; padding:24px; border-radius:8px;">
# MAGIC   <h1 style="margin:0; color:white;">Lab 1 — Store → SKU Recommender (propensity model)</h1>
# MAGIC   <p style="margin:6px 0 0 0; color:#9EB7BE; font-size:16px;">
# MAGIC     Persona: <strong style="color:white;">Data Scientist / Analyst</strong>
# MAGIC     &nbsp;·&nbsp; Estimated time: 60 min &nbsp;·&nbsp; Agenda slot: AI/ML on Databricks
# MAGIC   </p>
# MAGIC </div>
# MAGIC
# MAGIC **The business question:** *Which products should each store stock next?* Reps visit hundreds of
# MAGIC stores and guess what to pitch. We'll turn that into a model.
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #FF3620; padding:15px; margin:15px 0;">
# MAGIC   <strong>How we'll frame it — a "propensity" model.</strong> Instead of anything exotic, we ask a
# MAGIC   plain <strong>yes/no question</strong>: <em>given a store and a product, is this the kind of store
# MAGIC   that carries this kind of product?</em> That's just <strong>classification</strong> — the most
# MAGIC   common ML task in industry. We train it on what stores carry <em>today</em>, then point it at the
# MAGIC   products a store <em>doesn't</em> carry and recommend the ones it's most likely to want.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>The workflow you'll learn (the standard ML loop on Databricks):</strong>
# MAGIC   build features → train with <code>mlflow.autolog()</code> → <strong>register the model to Unity
# MAGIC   Catalog</strong> → load it back and <strong>test</strong> it. Pure Python (pandas + scikit-learn)
# MAGIC   — runs on serverless, no Spark cluster needed.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Genie Code:</strong> press <kbd>Cmd</kbd>+<kbd>I</kbd> in a cell, paste the prompt, let
# MAGIC   it write the code. Reference solutions are included so you're never stuck.
# MAGIC </div>

# COMMAND ----------

# MAGIC %run ./src/00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0 — Confirm the data is there
# MAGIC The sales tables were built once by the data-generation job (`src/01_generate_sales.py`).

# COMMAND ----------

# DBTITLE 1,Step 0: Prerequisite check
for tbl in (SALES_DIM_ACCOUNT, SALES_DIM_PRODUCT, SALES_FACT_ASSORTMENT, SALES_FACT_DEPLETIONS):
    print(f"  {tbl:55s} {spark.table(tbl).count():>10,} rows")
print("\nIf any are missing, run the data-generation job first.")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 1 — Understand the data
# MAGIC
# MAGIC Two tables drive this lab:
# MAGIC - **`dim_account`** — one row per store, with attributes: `banner` (grocery, convenience, liquor…),
# MAGIC   `channel`, `region`, `size_tier`.
# MAGIC - **`dim_product`** — one row per SKU, with `brand`, `category`, `base_price`, `pack_units`, etc.
# MAGIC - **`fact_assortment`** — which (store, SKU) pairs a store **currently carries**. This is our
# MAGIC   "ground truth": the products a store stocks today.

# COMMAND ----------

# DBTITLE 1,Step 1: How many products does a typical store carry?
display(spark.sql(f"""
  SELECT a.banner,
         COUNT(DISTINCT a.account_id)        AS stores,
         ROUND(AVG(carried.n_skus))          AS avg_skus_carried
  FROM {SALES_DIM_ACCOUNT} a
  JOIN (SELECT account_id, COUNT(*) AS n_skus FROM {SALES_FACT_ASSORTMENT} GROUP BY account_id) carried
    USING (account_id)
  GROUP BY a.banner ORDER BY avg_skus_carried DESC
"""))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 2 — Build the training table (features + label)
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #1B5161; padding:15px; margin:15px 0;">
# MAGIC   We make <strong>one row for every (store, product) combination</strong>. Each row gets:
# MAGIC   <ul style="margin:6px 0 0 0;">
# MAGIC     <li><strong>features</strong> — the store's attributes + the product's attributes, and</li>
# MAGIC     <li><strong>a label</strong> — <code>1</code> if the store currently carries that product,
# MAGIC         <code>0</code> if not.</li>
# MAGIC   </ul>
# MAGIC   The model learns which <em>kinds</em> of stores carry which <em>kinds</em> of products. ~800
# MAGIC   stores × ~50 products ≈ 40k rows — small enough for plain pandas.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Genie Code prompt:</strong>
# MAGIC   <blockquote style="border-left:3px solid #1B5161; margin:8px 0; padding:4px 12px; color:#1B3139;">
# MAGIC   Cross join the accounts and products pandas DataFrames into one row per (account_id, sku_id),
# MAGIC   then add a column "carries" = 1 if that pair is in the assortment DataFrame, else 0.
# MAGIC   </blockquote>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 2: One row per (store, product) with a carries label
import pandas as pd

accounts = spark.table(SALES_DIM_ACCOUNT).select(
    "account_id", "banner", "channel", "region", "size_tier").toPandas()
products = spark.table(SALES_DIM_PRODUCT).select(
    "sku_id", "brand", "category", "pack_units", "abv", "base_price", "popularity").toPandas()
assortment = spark.table(SALES_FACT_ASSORTMENT).select("account_id", "sku_id").toPandas()

# Every store paired with every product (pandas cross join).
pairs = accounts.merge(products, how="cross")

# Label: does the store currently carry this product?
carried = assortment.assign(carries=1)
data = pairs.merge(carried, on=["account_id", "sku_id"], how="left")
data["carries"] = data["carries"].fillna(0).astype(int)

print(f"rows: {len(data):,}   carried (label=1): {data['carries'].mean():.1%}")
display(spark.createDataFrame(data.head(8)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Split into train and test
# MAGIC Hold out 20% of the rows so we can measure the model on pairs it didn't train on. We `stratify`
# MAGIC on the label so both splits keep the same carried/not-carried balance.

# COMMAND ----------

# DBTITLE 1,Step 3: Train/test split
from sklearn.model_selection import train_test_split

CAT_COLS = ["banner", "channel", "region", "size_tier", "brand", "category"]
NUM_COLS = ["pack_units", "abv", "base_price", "popularity"]
FEATURES = CAT_COLS + NUM_COLS

X = data[FEATURES]
y = data["carries"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"train: {len(X_train):,}   test: {len(X_test):,}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 4 — Train, track with `mlflow.autolog()`, and register to Unity Catalog
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #FF3620; padding:15px; margin:15px 0;">
# MAGIC   We use a <strong>scikit-learn Pipeline</strong>: one-hot encode the text columns, then a
# MAGIC   <strong>RandomForestClassifier</strong> (a solid, no-tuning-needed default for tabular data).
# MAGIC   <ul style="margin:6px 0 0 0;">
# MAGIC     <li><code>mlflow.sklearn.autolog()</code> records the parameters and training metrics
# MAGIC         <em>automatically</em> — no logging code.</li>
# MAGIC     <li>We log the trained pipeline with a <strong>signature</strong> and register it to
# MAGIC         <strong>Unity Catalog</strong> as a governed, versioned model.</li>
# MAGIC   </ul>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 4: Pipeline + autolog + register to UC
import mlflow
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Register models to Unity Catalog (3-level name: catalog.schema.model).
mlflow.set_registry_uri("databricks-uc")
me = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{me}/lab1_recommender")

# autolog params + metrics; we log the model ourselves (with a signature) so we can register it.
mlflow.sklearn.autolog(log_models=False)

model = Pipeline([
    ("prep", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
        remainder="passthrough")),
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=12,
                                  random_state=42, n_jobs=-1)),
])

UC_MODEL = f"{CATALOG}.{SALES}.store_sku_recommender"

with mlflow.start_run(run_name="store_sku_recommender") as run:
    model.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    mlflow.log_metric("test_auc", float(test_auc))
    print(f"Test AUC: {test_auc:.3f}  (1.0 = perfect, 0.5 = coin flip)")

    sig = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
    info = mlflow.sklearn.log_model(
        model, artifact_path="model", signature=sig,
        input_example=X_train.head(3), registered_model_name=UC_MODEL)

print(f"Registered {UC_MODEL}  version {info.registered_model_version}")

# Give this version a friendly alias so we can load it by name.
mlflow.MlflowClient().set_registered_model_alias(UC_MODEL, "champion", info.registered_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — What did the model learn?
# MAGIC A quick look at feature importance — which inputs drive the "will carry it" decision. (Expect
# MAGIC product `category`/`popularity` and store `banner` near the top — exactly how a category manager
# MAGIC thinks.)

# COMMAND ----------

# DBTITLE 1,Step 5: Feature importance
import numpy as np
feat_names = model.named_steps["prep"].get_feature_names_out()
imp = pd.DataFrame({"feature": feat_names,
                    "importance": model.named_steps["rf"].feature_importances_})
display(spark.createDataFrame(imp.sort_values("importance", ascending=False).head(15)))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 6 — Test the registered model: recommend for one store
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #1B5161; padding:15px; margin:15px 0;">
# MAGIC   The real test: <strong>load the model back from Unity Catalog</strong> (as any downstream job or
# MAGIC   app would) and use it. We score every product a chosen store <em>doesn't</em> carry and surface the
# MAGIC   ones it's most likely to want — the rep's "next best SKUs."
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 6: Load from UC and recommend the top 5 SKUs for one store
loaded = mlflow.sklearn.load_model(f"models:/{UC_MODEL}@champion")

sample_account = "ACC-00042"
acct = accounts.loc[accounts.account_id == sample_account].iloc[0]
already_carries = set(assortment.loc[assortment.account_id == sample_account, "sku_id"])

# Candidates = products this store does NOT carry yet, with the store's attributes attached.
candidates = products.loc[~products.sku_id.isin(already_carries)].copy()
for c in ["banner", "channel", "region", "size_tier"]:
    candidates[c] = acct[c]

candidates["score"] = loaded.predict_proba(candidates[FEATURES])[:, 1]
top5 = candidates.sort_values("score", ascending=False).head(5)

print(f"Store {sample_account} ({acct['banner']}, {acct['region']}) — recommended next SKUs:")
display(spark.createDataFrame(
    top5[["sku_id", "brand", "category", "base_price", "score"]].round({"score": 3})))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Wrap-up
# MAGIC You built a recommender as a **plain classification model** in Python, tracked it with
# MAGIC **`mlflow.autolog()`**, **registered it to Unity Catalog** as a governed, versioned model, and
# MAGIC **loaded it back to make recommendations** — the exact loop you'll reuse for almost any ML project
# MAGIC on Databricks. From here, the same `@champion` model could be served behind a REST endpoint or
# MAGIC called from a Genie space, with no change to how it was built.
