---
name: time-series-forecasting
description: >
  Teaches an in-notebook assistant (Genie Code) how to do time series demand
  forecasting on Databricks — starting from the built-in ai_forecast() SQL
  function and graduating to tracked statsmodels / LightGBM models. Load this
  before iterating on Lab 3's ai_forecast workflow so generated code follows
  forecasting best practices instead of generic guesses.
---

# Time series forecasting on Databricks

You help a user build and iterate on demand forecasts. Forecasting has rules that
ordinary "write me some SQL/pandas" prompts get wrong. Follow the guidance below.

## The golden rule: never let the future leak into the past

Time is ordered. When you split data for evaluation, **always split by time** — train
on the earliest rows, test on the most recent ones. Never shuffle, never random-split.
A shuffled split lets a model peek at the future and reports a score that won't hold up
in production. Default to holding out the **last N periods** (e.g. last 6 months) as the test set.

## Step 1 — Get the data into forecasting shape

A forecast needs one value per time step per series:

- **One row per (group, time step).** For demand: `(brand, region, month)` with a numeric `y`.
- **A regular grain with no gaps.** Monthly means every month present; fill missing periods
  with 0 (or interpolate) rather than skipping them. `ai_forecast` and most models assume
  evenly spaced timestamps.
- **A clean time column** as a real `DATE`/`TIMESTAMP`, not a string.
- **Cast numerics to DOUBLE.** Spark `DECIMAL` round-trips to pandas `object` and breaks
  statsmodels / LightGBM.

When asked to aggregate, roll up with `date_trunc('MONTH', ...)` (or week/day) and `SUM`/`AVG`
grouped by the series keys.

## Step 2 — Start with `ai_forecast()` (the fast path)

`ai_forecast()` is a built-in SQL table function. It trains a univariate model per group and
returns the forecast plus a prediction interval. No Python, no model registry. Reach for it first.

```sql
SELECT brand, region, ds, y_forecast, y_lower, y_upper
FROM ai_forecast(
  TABLE(SELECT brand, region, ds, y FROM sales.demand_monthly WHERE ds < DATE'2026-06-01'),
  horizon   => '2027-02-01',          -- forecast UP TO this date (or an integer # of steps)
  time_col  => 'ds',
  value_col => 'y',
  group_col => array('brand','region') -- one independent forecast per group
)
```

Key arguments and how to iterate on them:

| Argument | What it does | How users ask to change it |
|---|---|---|
| `horizon` | How far ahead — a target date or an integer number of steps | "forecast 12 months instead of 9" |
| `group_col` | One forecast per distinct group | "also break it down by region" → add to the array |
| `prediction_interval_width` | Interval coverage (default 0.95) | "give me an 80% band" → set `0.8` |
| `frequency` | Step size if it can't be inferred | "the data is weekly" → set `'week'` |

Iterating in plain English is the point: *"widen the interval to 90%"*, *"only forecast the top 5
brands by volume"*, *"forecast weekly instead of monthly"*. Translate each to one argument or one
`WHERE`/`GROUP BY` change — don't rewrite the whole query.

### `ai_forecast` gotchas
- It is **univariate**: it sees only the history of `y`. It cannot use price, promotion, or weather.
  If the user needs those drivers, that's the signal to graduate to a real model (Step 4).
- Irregular or gappy timestamps give poor results or errors — fix the grain first (Step 1).
- Very short or brand-new series (a SKU launched last month) have nothing to learn from; expect a
  flat/naïve forecast and say so.
- It may be gated on some preview workspaces. Always wrap it so a failure falls back to a simple
  **seasonal-naïve** baseline (last-year-same-period value) rather than hard-stopping.

## Step 3 — Always sanity-check against a naïve baseline

Before trusting any forecast, compare it to **seasonal-naïve** (this period = same period last
year). On seasonal data this is shockingly hard to beat. If a fancy model can't beat naïve, prefer
naïve. Report error with **MAPE** for one series, or **WMAPE** (volume-weighted) across many series
so big sellers count more. Lower is better.

## Step 4 — Graduate to a tracked model when accuracy pays

When the user needs to tune, use exogenous drivers, or track experiments, write a training loop and
turn on **MLflow autolog** — it records params, metrics, and the model automatically. On serverless,
use **flavor-specific** autolog (`mlflow.statsmodels.autolog()`, `mlflow.lightgbm.autolog()`), not
the global `mlflow.autolog()`, which tries to read a Spark model-registry config that serverless blocks.

Two model classes that fit most demand problems:

- **statsmodels Holt-Winters / SARIMAX** — classical, interpretable seasonality. Good when you have
  one well-behaved series with clear annual structure.
  ```python
  from statsmodels.tsa.holtwinters import ExponentialSmoothing
  model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
  pred = model.forecast(horizon)
  ```
- **LightGBM with lag + calendar features** — learns from `lag_1/lag_2/lag_12`, rolling means, and
  `month`/`week`, and can fold in exogenous columns (price, promo, temperature) that `ai_forecast`
  can't. Strong when those drivers move demand.
  ```python
  for lag in (1, 2, 3, 12): df[f"lag_{lag}"] = df["y"].shift(lag)
  df["month"] = df["ds"].dt.month
  ```

Score every model on the **same time-based holdout** and put them on one small leaderboard (MAPE).
The right model depends on the series — speed (`ai_forecast`), interpretability (Holt-Winters), or
signal-rich accuracy (LightGBM).

## Step 5 — Keep the production surface stable

Whatever model wins, have it re-populate the same forecast table (e.g. `sales.demand_forecast`) that a
UC function and Genie space read. The business user's question never changes; only the accuracy behind
it improves. When asked to productionize, suggest a scheduled job that re-trains and re-points that table.

## Prompt patterns to prefer
- "Aggregate `fact_depletions` to monthly `y` per brand × region, no gaps, `y` as DOUBLE."
- "Forecast 9 months with `ai_forecast`, 90% interval, grouped by brand and region; fall back to
  seasonal-naïve if it errors."
- "Hold out the last 6 months, fit Holt-Winters with autolog, report holdout MAPE."
- "Compare ai_forecast, Holt-Winters, and LightGBM on the same holdout; show a MAPE leaderboard."
