# Weibull Survival Analysis — Beating the "Runs Forever" Illusion

Fits a **Weibull life distribution** to a canonical published reliability dataset and
shows why early-life reliability estimates are dangerous — and how to fix them.

## The through-line

Early in an asset's life almost everything is still running (right-censored) and you've
seen only a handful of failures. A plain **maximum-likelihood** Weibull fit on that data
is optimistically biased: the characteristic life **α** runs high and the survival curve
promises life the equipment doesn't have. A **Bayesian / bounded** fit puts a
weakly-informative prior on the parameters, so when data is thin it reports a wide
credible interval instead of a confident-but-wrong point estimate — then contracts onto
the truth as failures accrue.

## The dataset

**Lieblein & Zelen (1956)** deep-groove ball-bearing fatigue life (23 units, all run to
failure, millions of revolutions) — the textbook Weibull dataset. All failures known, so
it doubles as ground truth. We simulate the early-life view by *censoring* at an
inspection time and re-fitting.

## What it shows (verified numbers)

| Snapshot | Failures seen | Estimate | Verdict |
|---|---|---|---|
| Ground truth (all 23) | 23 | β≈2.1, α≈82, median life ≈69 | the target |
| Naive MLE @ t=28 | 1 of 23 | α≈110, median life ≈94 | overstates life ~36% |
| Bayesian @ t=28 | 1 of 23 | α mean 134, **90% CI [63, 259]** | honest: huge interval, truth inside |
| Bayesian @ t=60 | 11 | α≈76, CI [61, 100] | tightening onto truth |

## Notebook

| File | What it does |
|---|---|
| `weibull_survival.py` | Right-censored Weibull log-likelihood; MLE vs. grid-Bayesian fit with priors on (β, α); a convergence sweep over inspection times; and two plots — α-estimate vs. time with shrinking credible band, and early-life survival curves (illusion vs. honest fit). Self-contained; only `scipy` + `matplotlib`. |

## Swapping in your own data

Replace the `LIFETIMES` array with your run-to-failure / suspension records (durations +
a `censored` flag). For fleet data, read `(asset_id, age, failed)` from Unity Catalog into
pandas, build `times`/`censored`, and reuse `fit_bayes`. For a full posterior with
covariates, graduate to PyMC or `lifelines` — the intuition here carries over.
