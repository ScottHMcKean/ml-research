# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md-sandbox
# MAGIC <div style="background:#1B3139; color:white; padding:24px; border-radius:8px;">
# MAGIC   <h1 style="margin:0; color:white;">Lab 3 — Demand Forecasting, the Agentic Way</h1>
# MAGIC   <p style="margin:6px 0 0 0; color:#9EB7BE; font-size:16px;">
# MAGIC     Persona: <strong style="color:white;">Analytics Engineer → Data Scientist</strong>
# MAGIC     &nbsp;·&nbsp; Estimated time: 75 min &nbsp;·&nbsp; Agenda slot: Forecasting
# MAGIC   </p>
# MAGIC </div>
# MAGIC
# MAGIC **One workflow, three levels of effort — and you build it by *driving an agent*, not typing.**
# MAGIC We forecast monthly depletions for **Thirsty Otter × West**, adding accuracy only where it pays:
# MAGIC
# MAGIC 1. **`ai_forecast()`** — a built-in SQL function. No training, no Python. Minutes to a governed forecast.
# MAGIC 2. **statsmodels (Holt-Winters)** — a classical seasonal model, captured by `mlflow.autolog`.
# MAGIC 3. **LightGBM** — a gradient-boosted model that learns from lags + calendar, also under `mlflow.autolog`.
# MAGIC
# MAGIC Then we wrap the result as a **governed UC function** a business user reaches through Genie.
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:16px; margin:14px 0;">
# MAGIC   <strong>🤖 Three agentic tools you'll use.</strong>
# MAGIC   <ul style="margin:8px 0 0 0;">
# MAGIC     <li><strong>Genie Code (<kbd>Cmd</kbd>+<kbd>I</kbd>)</strong> — paste the prompt in a code cell; the
# MAGIC         in-notebook assistant writes the step.</li>
# MAGIC     <li><strong>Databricks MCP</strong> — ask an agent (e.g. Claude Code) to act on the workspace for
# MAGIC         you: run SQL, register the function, schedule a job.</li>
# MAGIC     <li><strong>Claude Code skills</strong> — short instruction files that teach the agent to do
# MAGIC         one thing well. <strong>Load <code>time-series-forecasting.skill.md</code> (in this folder)
# MAGIC         before you start</strong> so Genie Code follows forecasting best practices when it iterates
# MAGIC         on the <code>ai_forecast</code> steps below.</li>
# MAGIC   </ul>
# MAGIC   Each step has one <strong>🤖 Agentic move</strong> — the single best way to let the agent do it.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#FF3621; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   Tested on Serverless v5
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC These two libraries aren't in the serverless base environment, so install them first.
# MAGIC (On an ML-runtime cluster they're already present — this is a quick no-op.)

# COMMAND ----------

# MAGIC %pip install -q lightgbm statsmodels

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./src/00_setup

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 1 — Aggregate depletions to monthly demand
# MAGIC We roll weekly depletions up to a monthly value per brand × region — the shape every forecaster wants.
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Agentic move:</strong> before modelling, ask your agent to run the Databricks MCP tool
# MAGIC   <code>get_table_stats_and_schema</code> on <code>{catalog}.sales.fact_depletions</code> so you know
# MAGIC   the columns and date range you're working with.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 1: Build the monthly demand history
spark.sql(f"""
CREATE OR REPLACE TABLE {SALES}.demand_monthly AS
SELECT p.brand,
       a.region,
       date_trunc('MONTH', f.week)        AS ds,
       CAST(SUM(f.cases) AS DOUBLE)        AS y
FROM {SALES_FACT_DEPLETIONS} f
JOIN {SALES_DIM_PRODUCT}  p USING (sku_id)
JOIN {SALES_DIM_ACCOUNT}  a USING (account_id)
GROUP BY p.brand, a.region, date_trunc('MONTH', f.week)
HAVING SUM(f.cases) > 0
""")
display(spark.sql(f"""
  SELECT brand, region, COUNT(*) months, ROUND(AVG(y)) avg_cases
  FROM {SALES}.demand_monthly GROUP BY brand, region ORDER BY avg_cases DESC LIMIT 10
"""))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 2 — Level 1: forecast with `ai_forecast()`
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #FF3620; padding:15px; margin:15px 0;">
# MAGIC   <strong>The fastest credible forecast on Databricks.</strong> <code>ai_forecast()</code> is a built-in
# MAGIC   SQL function: give it a history table, a time column, a value column, and a horizon — it returns the
# MAGIC   forecast plus prediction intervals. <strong>No training loop, no Python.</strong> Univariate and
# MAGIC   perfect for a fast, governed baseline.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Agentic move — write it with Genie Code.</strong> In the cell below press
# MAGIC   <kbd>Cmd</kbd>+<kbd>I</kbd> and paste:
# MAGIC   <blockquote style="border-left:3px solid #1B5161; margin:8px 0; padding:4px 12px; color:#1B3139;">
# MAGIC   Write a SQL query using ai_forecast() that forecasts 9 months ahead from
# MAGIC   {catalog}.sales.demand_monthly, with time_col ds, value_col y, grouped by brand and region,
# MAGIC   returning the forecast and prediction interval. Filter the history to dates before 2026-06-01.
# MAGIC   </blockquote>
# MAGIC   The reference cell below has a graceful fallback if <code>ai_forecast</code> is gated.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 2: Run ai_forecast (with graceful fallback)
def run_ai_forecast():
    spark.sql(f"""
    CREATE OR REPLACE TABLE {SALES_DEMAND_FORECAST} AS
    SELECT brand, region,
           CAST(ds AS DATE)   AS forecast_month,
           y_forecast         AS forecast_cases,
           y_lower            AS forecast_p10,
           y_upper            AS forecast_p90,
           'ai_forecast'      AS method,
           current_timestamp() AS scored_ts
    FROM ai_forecast(
      TABLE(
        SELECT brand, region, ds, y
        FROM {SALES}.demand_monthly
        WHERE ds < DATE'2026-06-01'
      ),
      horizon   => '2027-02-01',
      time_col  => 'ds',
      value_col => 'y',
      group_col => array('brand','region')
    )
    """)
    return "ai_forecast"

def run_seasonal_naive_fallback():
    # Last-year-same-month value as the forecast; ±20% band. Honest, simple baseline.
    spark.sql(f"""
    CREATE OR REPLACE TABLE {SALES_DEMAND_FORECAST} AS
    WITH last_year AS (
      SELECT brand, region, month(ds) AS mo, AVG(y) AS y_ly
      FROM {SALES}.demand_monthly
      WHERE ds >= add_months(DATE'2026-06-01', -12) AND ds < DATE'2026-06-01'
      GROUP BY brand, region, month(ds)
    ),
    horizon AS (
      SELECT explode(sequence(DATE'2026-06-01', DATE'2027-02-01', INTERVAL 1 MONTH)) AS forecast_month
    )
    SELECT ly.brand, ly.region, h.forecast_month,
           ly.y_ly                 AS forecast_cases,
           ly.y_ly * 0.8           AS forecast_p10,
           ly.y_ly * 1.2           AS forecast_p90,
           'seasonal_naive_fallback' AS method,
           current_timestamp()     AS scored_ts
    FROM horizon h JOIN last_year ly ON month(h.forecast_month) = ly.mo
    """)
    return "seasonal_naive_fallback"

try:
    method = run_ai_forecast()
    print("✅ ai_forecast succeeded.")
except Exception as e:
    print(f"⚠️  ai_forecast unavailable ({str(e)[:160]}...)\n   Falling back to seasonal-naïve.")
    method = run_seasonal_naive_fallback()

print(f"Method used: {method}  ·  rows: {spark.table(SALES_DEMAND_FORECAST).count():,}")
display(spark.table(SALES_DEMAND_FORECAST).orderBy("brand", "region", "forecast_month").limit(15))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 3 — Wrap it as a governed UC function + Genie
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   A <strong>UC function</strong> turns the forecast table into a callable a business user, a Genie space,
# MAGIC   or an agent can invoke by name — governed permissions, no SQL templating. Add it to a Genie space and a
# MAGIC   planner just asks: <em>"What's the forecast for Thirsty Otter in the West next quarter?"</em>
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #1B5161; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Agentic move:</strong> ask your agent to load the <code>databricks-genie</code> skill,
# MAGIC   create a Genie space over <code>{catalog}.sales</code> that includes <code>forecast_demand</code>, and
# MAGIC   ask the question above through the MCP <code>ask_genie</code> tool.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 3: forecast_demand(brand, region, horizon) UC function
spark.sql(f"""
CREATE OR REPLACE FUNCTION {SALES}.forecast_demand(
  brand_filter  STRING DEFAULT NULL,
  region_filter STRING DEFAULT NULL,
  horizon_months INT   DEFAULT 9
)
RETURNS TABLE(brand STRING, region STRING, forecast_month DATE,
              forecast_cases DOUBLE, forecast_p10 DOUBLE, forecast_p90 DOUBLE)
COMMENT 'Monthly depletion forecast (cases) by brand and region, with P10/P90 interval.'
RETURN
  SELECT brand, region, forecast_month, forecast_cases, forecast_p10, forecast_p90
  FROM {SALES_DEMAND_FORECAST}
  WHERE (brand_filter  IS NULL OR brand  = brand_filter)
    AND (region_filter IS NULL OR region = region_filter)
    AND forecast_month < add_months(
          (SELECT MIN(forecast_month) FROM {SALES_DEMAND_FORECAST}), horizon_months)
  ORDER BY brand, region, forecast_month
""")
print(f"Created {SALES}.forecast_demand(brand, region, horizon_months)")

display(spark.sql(f"""
  SELECT * FROM {SALES}.forecast_demand(
    brand_filter => 'Thirsty Otter', region_filter => 'West', horizon_months => 6)
"""))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 4 — Level 2: a classical model with `mlflow.autolog` (statsmodels)
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #FF3620; padding:15px; margin:15px 0;">
# MAGIC   <code>ai_forecast()</code> is fast but a black box. When you want to <em>tune</em> a model and
# MAGIC   <em>track</em> what you tried, you write a training loop — and <strong>MLflow autolog</strong> captures
# MAGIC   the params, metrics, and the model artifact <em>automatically</em>, with zero logging code. We use a
# MAGIC   <strong>Holt-Winters</strong> exponential-smoothing model — the classic seasonal baseline.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Agentic move:</strong> ask your agent to load the <code>mlflow-onboarding</code> skill to
# MAGIC   scaffold the experiment and confirm autolog is on before you train.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 4: Holt-Winters with mlflow.autolog (flavor-specific, serverless-safe)
import pandas as pd, numpy as np, mlflow
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

# Pull one well-populated series to a pandas Series indexed by month.
series_pdf = spark.sql(f"""
  SELECT ds, y FROM {SALES}.demand_monthly
  WHERE brand='Thirsty Otter' AND region='West' ORDER BY ds
""").toPandas()
series_pdf["ds"] = pd.to_datetime(series_pdf["ds"])
s = series_pdf.set_index("ds")["y"].asfreq("MS")

HOLDOUT = 6
train, test = s.iloc[:-HOLDOUT], s.iloc[-HOLDOUT:]

# Point the experiment at your workspace folder. Flavor-specific autolog avoids the
# Spark model-registry config read that serverless blocks for global mlflow.autolog().
me = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{me}/lab3_forecast")
mlflow.statsmodels.autolog()

with mlflow.start_run(run_name="holt_winters"):
    hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
    hw_pred = hw.forecast(HOLDOUT)
    hw_mape = mean_absolute_percentage_error(test, hw_pred)
    mlflow.log_metric("holdout_mape", float(hw_mape))   # the one metric autolog can't know
    print(f"Holt-Winters holdout MAPE: {hw_mape:.3f}  (autolog logged params + model)")

display(spark.createDataFrame(
    pd.DataFrame({"month": test.index, "actual": test.values, "holt_winters": hw_pred.values})))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 5 — Level 3: a gradient-boosted model with `mlflow.autolog` (LightGBM)
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #FF3620; padding:15px; margin:15px 0;">
# MAGIC   The third model class. <strong>LightGBM</strong> learns from engineered <strong>lags</strong> and
# MAGIC   <strong>calendar</strong> features (and can fold in price/promo/weather that a univariate model can't
# MAGIC   see). <code>mlflow.lightgbm.autolog()</code> captures the booster and params automatically — same
# MAGIC   one-line pattern, different model class.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Agentic move:</strong> generate the feature-engineering with Genie Code —
# MAGIC   <em>"add lag-1, lag-2, lag-12 and a month feature to this monthly series, then train a LightGBM
# MAGIC   regressor and predict the 6-month holdout."</em>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 5: LightGBM with mlflow.lightgbm.autolog
import lightgbm as lgb

# Lag + calendar features on the same monthly series.
feat = series_pdf.copy()
for lag in (1, 2, 3, 12):
    feat[f"lag_{lag}"] = feat["y"].shift(lag)
feat["month"] = feat["ds"].dt.month
feat = feat.dropna().reset_index(drop=True)
FEATURES = [c for c in feat.columns if c.startswith("lag_")] + ["month"]

f_train, f_test = feat.iloc[:-HOLDOUT], feat.iloc[-HOLDOUT:]

from mlflow.models import infer_signature

# Register models to Unity Catalog. autolog logs params/metrics; we log + register the
# model ourselves (with a signature) so we can govern, version, and reload it.
mlflow.set_registry_uri("databricks-uc")
mlflow.lightgbm.autolog(log_models=False)
UC_FORECAST_MODEL = f"{CATALOG}.{SALES}.demand_lightgbm"

with mlflow.start_run(run_name="lightgbm"):
    dtrain = lgb.Dataset(f_train[FEATURES], label=f_train["y"])
    params = dict(objective="regression", learning_rate=0.05, num_leaves=15,
                  min_data_in_leaf=3, verbosity=-1)
    booster = lgb.train(params, dtrain, num_boost_round=200)
    lgb_pred = booster.predict(f_test[FEATURES])
    lgb_mape = mean_absolute_percentage_error(f_test["y"], lgb_pred)
    mlflow.log_metric("holdout_mape", float(lgb_mape))

    sig = infer_signature(f_train[FEATURES], booster.predict(f_train[FEATURES]))
    lgb_info = mlflow.lightgbm.log_model(
        booster, artifact_path="model", signature=sig,
        input_example=f_train[FEATURES].head(3), registered_model_name=UC_FORECAST_MODEL)
    print(f"LightGBM holdout MAPE: {lgb_mape:.3f}  ·  "
          f"registered {UC_FORECAST_MODEL} v{lgb_info.registered_model_version}")

mlflow.MlflowClient().set_registered_model_alias(
    UC_FORECAST_MODEL, "champion", lgb_info.registered_model_version)

display(spark.createDataFrame(
    pd.DataFrame({"month": f_test["ds"], "actual": f_test["y"].values, "lightgbm": lgb_pred})))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 6 — Compare the three, then productionize the winner
# MAGIC
# MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
# MAGIC   All three were scored on the <strong>same 6-month holdout</strong>. Autolog already put the two models
# MAGIC   side-by-side in the MLflow experiment; here we add <code>ai_forecast</code>'s number so they're
# MAGIC   comparable. Whichever wins re-populates <code>sales.demand_forecast</code> — so the
# MAGIC   <strong>same UC function + Genie space</strong> from Step 3 now serve the better forecast, and the
# MAGIC   business user's question never changes.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #1B5161; padding:12px; margin:10px 0;">
# MAGIC   <strong>🤖 Agentic move — hand the last mile to your agent.</strong> Ask it to load the
# MAGIC   <code>databricks-jobs</code> skill and create a monthly Lakeflow Job (MCP <code>manage_jobs</code>) that
# MAGIC   re-trains and re-points <code>sales.demand_forecast</code> with the winning model.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 6: ai_forecast on the same holdout + leaderboard
af_mape = None
try:
    af = spark.sql(f"""
      SELECT CAST(ds AS DATE) AS ds, y_forecast
      FROM ai_forecast(
        TABLE(SELECT ds, y FROM {SALES}.demand_monthly
              WHERE brand='Thirsty Otter' AND region='West' AND ds < '{test.index.min().date()}'),
        horizon => '{test.index.max().date()}', time_col => 'ds', value_col => 'y')
    """).toPandas()
    af["ds"] = pd.to_datetime(af["ds"])
    af = af.set_index("ds").reindex(test.index).dropna()
    if len(af):
        af_mape = mean_absolute_percentage_error(test.loc[af.index], af["y_forecast"])
except Exception as e:
    print(f"⚠️  ai_forecast holdout skipped ({str(e)[:120]})")

board = pd.DataFrame([
    {"model": "ai_forecast", "class": "native SQL",    "holdout_mape": af_mape},
    {"model": "holt_winters","class": "statsmodels",   "holdout_mape": float(hw_mape)},
    {"model": "lightgbm",    "class": "gradient boost", "holdout_mape": float(lgb_mape)},
]).sort_values("holdout_mape", na_position="last").reset_index(drop=True)
print("Leaderboard — lower MAPE is better (Thirsty Otter × West, 6-month holdout):")
display(spark.createDataFrame(board))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 7 — Register → test: load the champion model from Unity Catalog
# MAGIC
# MAGIC <div style="background:#F1F1F1; border-left:5px solid #1B5161; padding:15px; margin:15px 0;">
# MAGIC   Same loop as Lab 1: the model is now a <strong>governed, versioned object in Unity Catalog</strong>
# MAGIC   (registered in Step 5 under the <code>@champion</code> alias). We <strong>load it back</strong> — as
# MAGIC   any downstream job would — and confirm it reproduces the forecast. From here it can be served behind
# MAGIC   an endpoint or refreshed by a scheduled job.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Step 7: Load the registered model and confirm it reproduces the forecast
loaded = mlflow.lightgbm.load_model(f"models:/{UC_FORECAST_MODEL}@champion")
reloaded_pred = loaded.predict(f_test[FEATURES])
print(f"Loaded {UC_FORECAST_MODEL}@champion — predictions match the in-notebook model:",
      bool(np.allclose(reloaded_pred, lgb_pred)))
display(spark.createDataFrame(
    pd.DataFrame({"month": f_test["ds"], "actual": f_test["y"].values,
                  "reloaded_lightgbm": reloaded_pred})))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Wrap-up
# MAGIC One forecasting workflow at **three levels of effort** — `ai_forecast()` for speed, Holt-Winters and
# MAGIC LightGBM for accuracy — with **`mlflow.autolog` capturing every run for free**. The winning model is
# MAGIC **registered to Unity Catalog and loaded back to test it** (the same autolog → register → test loop as
# MAGIC Lab 1), and the governed `ai_forecast` UC function means whichever model wins, the business user's Genie
# MAGIC question stays the same. You drove it through **Genie Code, Databricks MCP, and Claude Code skills**.
