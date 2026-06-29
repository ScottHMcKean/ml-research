# Payments — Real-Time Feature Store, Serving & Benchmark

A **real, deployable, benchmarkable** version of a real-time payment-authorization scoring
system on Databricks. It turns an architecture *sample pack* (schema shapes + latency
targets, no running infra) into a working pipeline:

```
seed events → declare features → publish online (Lakebase) → train LightGBM
            → serve with automatic feature lookup → benchmark latency
```

A **Databricks App** acts as a cost-effective synthetic event generator, the scoring
front-end (Stage-0 rules + model), the cache-backfill trigger, and a live latency
dashboard. **All data is synthetic and all names are generic** — this is a pattern
reference, not tied to any company.

## Architecture

![Solution](docs/architecture/01_solution.svg)

Diagrams are authored in [**D2**](https://d2lang.com) — the `.d2` sources are the
version-controllable, editable artifacts; rendered `.svg`s are committed alongside. See
[`docs/architecture/`](docs/architecture/): `01_solution`, `02_feature_topology`
(feature topology & cache feed-forward), and `03_latency_path` (request-time sequence).
Re-render with `d2 <file>.d2 <file>.svg`.

## What maps to what

| Goal | Where |
|------|-------|
| Seed a table | `01_seed_data.py` → `shm_skunkworks_catalog.payments.raw_events` |
| Declare features (Feature Engineering API) | `02_feature_engineering.py` — `FeatureEngineeringClient`, `FeatureLookup`, `FeatureFunction` |
| Online serving (Lakebase) | `03_online_store.py` — `fe.create_online_store` + `fe.publish_table` |
| LightGBM (section-4 pipeline) | `04_train_register.py` — training set + `fe.log_model` |
| Serve + auto feature lookup | `05_serving.py` |
| Profile online performance | `06_benchmark.py` — p50/p90/p99, throughput |
| Cache daily/monthly + feed forward | `02` (initial) + `08_backfill_cache.py` (scheduled) |
| Hot 1h counters | `07_streaming_counters.py` |
| Cost-effective generator + backfill + dashboard | `app/` (FastAPI Databricks App) |

### Caching & "feed forward"
Daily and monthly aggregates are stored in feature tables with a **timeseries column** set
to the *end* of each period. A point-in-time lookup (`timestamp_lookup_key=event_ts`) returns
the most recent **completed** period and carries it forward until the next scheduled refresh —
correct for both training (no leakage) and serving. `08_backfill_cache` recomputes and
re-publishes these on a cron (or on demand from the App's `/backfill`).

### Current vs legacy APIs
Uses `databricks-feature-engineering` → `FeatureEngineeringClient` and the **Lakebase Online
Feature Store** (`create_online_store` / `publish_table`). It deliberately avoids the legacy
`FeatureStoreClient` and `OnlineTableSpec`/online tables (no longer supported), and replaces
the original sample's external Redis hot-cache with Lakebase.

## Deploy & run

Prereqs: Databricks CLI ≥ 0.265.0 (you have a newer one), a workspace with **Lakebase
(Autoscaling)** and **Databricks Apps** in-region, and DBR 16.4 LTS ML+ / serverless. The
bundle is part of the repo root `databricks.yml` (target `dev`, profile `DEFAULT`).

```bash
# 1. Validate
databricks bundle validate -t dev

# 2. Set the App's warehouse id (used to insert generated events), then deploy
databricks bundle deploy -t dev --var="payments_warehouse_id=<warehouse-id>"

# 3. Run the full pipeline (seed → features → online → train → serve → benchmark)
databricks bundle run payments_setup_and_train -t dev

# 4. (optional) start the hot-counter stream / schedule the cache backfill
databricks bundle run payments_counters_streaming -t dev   # continuous, paused by default
databricks bundle run payments_backfill_cache -t dev
```

The App (`payments-feature-store-demo`) deploys with the bundle; open it, click **Start
generator**, **Score one event**, and **Backfill caches**, and watch p50/p99 update live.

## Benchmark vs sample targets

`06_benchmark.py` writes results to `shm_skunkworks_catalog.payments.benchmark_results`. Compare against the
original sample's per-component targets:

| Component | Sample target |
|-----------|---------------|
| Feature Store (online) read | p50 ~3ms · p99 ~8ms |
| Model inference | p50 ~2ms · p99 ~5ms |
| **End-to-end scoring** | **p99 < 50ms** |

(Real numbers depend on workload size, scale-to-zero warm-up, and region; the benchmark
warms the endpoint before measuring.)

## Notebooks
`00_setup` (shared constants) · `01_seed_data` · `02_feature_engineering` · `03_online_store`
· `04_train_register` · `05_serving` · `06_benchmark` · `07_streaming_counters` ·
`08_backfill_cache`.
