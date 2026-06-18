# Machine learning on Databricks — a hands-on workshop

This workshop is for people who are comfortable with Python and SQL but are new to doing
machine learning on Databricks. By the end you'll have built three working projects and you'll
understand the shape of an ML workflow here: get data in, build something with it, and put the
result somewhere a non-technical colleague can actually use.

You don't need prior ML experience. Each step explains *why* you're doing it, and you'll generate
most of the code by describing what you want to an assistant rather than typing it from scratch.

## The setting

You're on the data team at a beverage company. (Everything here is made up — the data is synthetic
and the brands are deliberately silly: **Thirsty Otter**, **Lazy Llama**, **Fancy Flamingo**,
**Moose Juice**, **Old Grumpy Bear**, and **Hydro Hippo**.) The company sells through retail accounts
and runs its own brewery. That gives us two very different kinds of data — sales records and factory
sensor readings — and three real questions to answer.

## What you'll build, and what each lab teaches

**Lab 1 — Which products should each store stock next?**
Sales reps visit hundreds of stores and guess what to pitch. You'll build a recommender by asking a
simple **yes/no question** — *is this the kind of store that carries this kind of product?* — which is
just **classification**, the most common ML task. You'll learn the standard Databricks ML loop:
build features → train with `mlflow.autolog()` → **register the model to Unity Catalog** → load it back
and test it. Pure Python (pandas + scikit-learn), so it runs on serverless.
→ `lab1_store_sku_recommender.py`

**Lab 2 — Is a piece of brewery equipment about to fail?**
The brewery streams sensor readings — temperatures, pressures, vibration — every few minutes. You'll
build a pipeline that ingests that stream as it lands, cleans and enriches it, and flags readings that
look wrong *before* they trip a hard alarm. This is the unglamorous but essential half of ML: reliably
getting messy, continuous data into a usable table. You'll learn the **bronze → silver → gold**
("medallion") pattern and how **Auto Loader** turns streaming ingestion into a few lines of code.
→ `lab2_brewery_autoloader.py`

**Lab 3 — How much of each product will we sell next quarter?**
You'll forecast demand three ways, from least to most effort: a one-line built-in function
(`ai_forecast()`), then two models you train yourself and track with **MLflow**. You'll see how to
compare them fairly and how the winner can quietly improve the answer a business planner gets when they
ask, in plain English, "what's the forecast for Thirsty Otter in the West?"
→ `lab3_ai_forecast.py`

The labs build on shared data, so run them in order the first time.

## How the data gets there

Two background jobs (defined in `databricks.yml`, run through the asset bundle) create the data the
labs read. You run them once before you start:

- **Data generation job** builds the synthetic sales and brewery-history tables.
- **Streaming job** drips sensor readings into a storage volume, simulating the live feed Lab 2 picks up.

A shared setup file (`src/00_setup.py`) creates the catalog, schemas, and volume, and defines the table
names every notebook uses — so the labs themselves stay short.

## The "agentic" part

You'll lean on three assistants that write and run code for you. If these are new, here's the short
version:

- **Genie Code** is an assistant *inside the notebook*. Press <kbd>Cmd</kbd>+<kbd>I</kbd> in a cell,
  describe the step in words, and it writes the code. You refine it by talking to it
  ("now group by region too") instead of editing by hand.
- **Databricks MCP** lets an outside agent (like Claude Code) *act on your workspace* — run a query,
  create a function, schedule a job — when you ask it to.
- **Skills** are short instruction files that teach an assistant how to do a specific thing well. This
  repo ships one: [`time-series-forecasting.skill.md`](time-series-forecasting.skill.md). Load it before
  Lab 3 so Genie Code follows forecasting best practices (proper train/test splits, naïve baselines,
  when to leave `ai_forecast` for a real model) instead of guessing.

Each lab marks the spots where an assistant does the work.

## Getting started

```bash
# 1. Connect to your workspace
databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile ml-workshop
export DATABRICKS_CONFIG_PROFILE=ml-workshop

# 2. Deploy the bundle (catalog, schemas, jobs, notebooks)
databricks bundle validate
databricks bundle deploy

# 3. Build the data (run once)
databricks bundle run ml_workshop_data_generation

# 4. For Lab 2, start the streaming job and leave it running (it streams continuously
#    on a small classic cluster). Cancel the run when you finish Lab 2.
databricks bundle run ml_workshop_streaming --no-wait
```

Then open each lab notebook in the workspace and run it top to bottom, using Genie Code
(<kbd>Cmd</kbd>+<kbd>I</kbd>) to generate each step from the prompt provided.

A note on compute: all three lab notebooks run on **standard serverless** — select **environment
version 5** in the notebook's serverless panel. Lab 3 `%pip install`s `lightgbm` and `statsmodels`
(they aren't in the serverless base); Lab 1 (scikit-learn) and Lab 2 (Auto Loader) need nothing extra.
Only the background *streaming job* uses a small classic cluster — it runs a continuous producer loop,
which isn't a serverless-notebook fit. (GPU serverless / the `databricks_ai_v5` base environment isn't
needed here — these are CPU workloads.)

## When you're done

Tear down in two steps:

```bash
# 1. Remove the deployed jobs and notebooks. Deleting the streaming job also
#    cancels its continuous run, so nothing keeps billing.
databricks bundle destroy --auto-approve
```

The synthetic data is *not* bundle-managed — the notebooks create the catalog when they run — so
drop it yourself. In a Databricks SQL editor or notebook on the workspace:

```sql
-- 2. Drop the catalog and all its data.
DROP CATALOG IF EXISTS ml_workshop CASCADE;
```

If you started the streaming job and want to stop it *without* a full teardown, just cancel its run
(`databricks jobs list-runs --job-id <id>` to find it, then `databricks jobs cancel-run <run-id>`).

## What's in the folder

```
lab1_store_sku_recommender.py     Lab 1
lab2_brewery_autoloader.py        Lab 2
lab3_ai_forecast.py               Lab 3
time-series-forecasting.skill.md  forecasting skill to load before Lab 3
databricks.yml                    asset bundle: variables + the two jobs
README.md                         you are here
src/                              supporting source code (run by jobs, %run by labs)
  00_setup.py                      shared config: catalog, schemas, volume, table names
  01_generate_sales.py             data-gen job: sales & depletions (Labs 1 and 3)
  02_generate_brewery_history.py   data-gen job: brewery sensor history (Lab 2)
  brewery_generator.py             shared engine that synthesizes the sensor readings
  write_to_volume.py               streaming job: drips sensor JSON onto the volume (Lab 2 source)
```
