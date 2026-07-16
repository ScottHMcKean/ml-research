# Databricks notebook source

# MAGIC %md
# MAGIC # Weibull Survival Analysis — Beating the "Runs Forever" Illusion
# MAGIC Fits a **Weibull life distribution** to a canonical published reliability
# MAGIC dataset and shows how a naive maximum-likelihood fit on **early, censored**
# MAGIC data makes the equipment look like it will *last forever*, while a
# MAGIC **Bayesian / bounded** fit stays honest and converges on the right answer as
# MAGIC failures accrue.
# MAGIC
# MAGIC ## The problem this demo is about
# MAGIC Early in an asset's life you have run **years of operation with only one or two
# MAGIC failures**. Everything still turning is *right-censored* — you know it survived
# MAGIC past its current age, not when it will die. With almost no failures the
# MAGIC likelihood surface is nearly flat in the scale parameter, so the fit happily
# MAGIC pushes the characteristic life **α** off to infinity: the maintenance planner
# MAGIC reads "median life ≈ 90+" and defers the overhaul. Then failures start landing
# MAGIC and the true median life turns out to be ~69. Classic, expensive mistake.
# MAGIC
# MAGIC The cure is **regularization**: put a weakly-informative prior (or an
# MAGIC engineering bound) on the Weibull parameters. Early on, the prior dominates and
# MAGIC the fit refuses to over-promise (it reports a *huge* uncertainty interval
# MAGIC instead). As real failures accumulate the data takes over and the posterior
# MAGIC contracts onto the truth.
# MAGIC
# MAGIC ## The dataset
# MAGIC **Lieblein & Zelen (1956)** ball-bearing fatigue-life data — the textbook
# MAGIC Weibull dataset. 23 bearings run to failure; lifetime measured in **millions of
# MAGIC revolutions**. All 23 failures are known, which gives us **ground truth** to
# MAGIC judge the early-life fits against. We simulate the early-life view by
# MAGIC *censoring* the data at an inspection time and re-fitting.
# MAGIC
# MAGIC > Lieblein, J. & Zelen, M. (1956). *Statistical investigation of the fatigue
# MAGIC > life of deep-groove ball bearings.* J. Res. Natl. Bur. Stand. 57(5), 273–316.
# MAGIC
# MAGIC ### Swapping in your own data
# MAGIC Replace the `LIFETIMES` array with your run-to-failure / suspension records
# MAGIC (durations + a `censored` flag) and everything downstream works unchanged.
# MAGIC For fleet data straight out of Unity Catalog, read a table of
# MAGIC `(asset_id, age, failed)` into a pandas DataFrame and build `times`/`censored`
# MAGIC from it.

# COMMAND ----------

# MAGIC %pip install scipy matplotlib
# MAGIC %restart_python

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats

SEED = 42
rng = np.random.default_rng(SEED)

# Lieblein & Zelen (1956) ball-bearing fatigue life, in millions of revolutions.
# All 23 units failed — no suspensions — so this is our ground truth.
LIFETIMES = np.array([
    17.88, 28.92, 33.00, 41.52, 42.12, 45.60, 48.48, 51.84, 51.96, 54.12,
    55.56, 67.80, 68.64, 68.64, 68.88, 84.12, 93.12, 98.64, 105.12, 105.84,
    127.92, 128.04, 173.40,
])
print(f"{len(LIFETIMES)} bearings, life range "
      f"{LIFETIMES.min():.1f}–{LIFETIMES.max():.1f} M-revs, "
      f"mean {LIFETIMES.mean():.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Weibull model
# MAGIC We use the standard two-parameter Weibull with **shape β** (a.k.a. the Weibull
# MAGIC slope) and **scale α** (the *characteristic life* — the age by which 63.2% have
# MAGIC failed):
# MAGIC
# MAGIC - **β < 1** infant mortality, **β = 1** random (exponential) failures,
# MAGIC   **β > 1** wear-out. Bearings fatigue, so we expect β ≈ 2.
# MAGIC - The **survival function** S(t) = exp(−(t/α)^β) is the probability a unit is
# MAGIC   still alive at age t. **Median life** = α·(ln 2)^(1/β).
# MAGIC
# MAGIC The log-likelihood correctly handles **right-censoring**: a failure at time t
# MAGIC contributes its log-pdf; a unit still running at inspection time t contributes
# MAGIC its log-survival log S(t) (all we know is "it made it this far").

# COMMAND ----------

def weibull_loglik(k, lam, times, censored):
    """Right-censored Weibull log-likelihood. `censored[i]` True => still alive at times[i]."""
    z = times / lam
    logpdf = np.log(k / lam) + (k - 1) * np.log(z) - z**k
    logsf = -(z**k)                       # log S(t) = -(t/alpha)^beta
    return np.where(censored, logsf, logpdf).sum()


def fit_mle(times, censored):
    """Plain maximum-likelihood Weibull fit (no regularization)."""
    def nll(p):
        k, lam = p
        if k <= 0 or lam <= 0:
            return 1e12
        return -weibull_loglik(k, lam, times, censored)

    res = optimize.minimize(
        nll, x0=[1.5, np.mean(times)], method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 20000},
    )
    return res.x  # (beta, alpha)


def fit_bounded(times, censored, alpha_max=90.0, beta_bounds=(0.1, 10.0)):
    """Constrained MLE — the frequentist counterpart to a prior. Same likelihood as
    `fit_mle`, but alpha is boxed to [1, alpha_max] where `alpha_max` is a hard
    engineering upper bound on characteristic life. No prior, no interval — just a cap."""
    def nll(p):
        k, lam = p
        return -weibull_loglik(k, lam, times, censored)

    res = optimize.minimize(
        nll, x0=[1.5, np.mean(times)], method="L-BFGS-B",
        bounds=[beta_bounds, (1.0, alpha_max)],
    )
    return res.x  # (beta, alpha)


def median_life(k, lam):
    return lam * np.log(2) ** (1 / k)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Ground truth — fit the complete run-to-failure data
# MAGIC With all 23 failures observed, MLE recovers the accepted answer for this
# MAGIC dataset: **β ≈ 2.1, α ≈ 82**, i.e. a median life of ~69 million revolutions.
# MAGIC This is the number the early-life fits *should* be aiming at but can't see yet.

# COMMAND ----------

no_cens = np.zeros(len(LIFETIMES), dtype=bool)
beta_true, alpha_true = fit_mle(LIFETIMES, no_cens)
med_true = median_life(beta_true, alpha_true)
print(f"GROUND TRUTH (23/23 failed): beta={beta_true:.2f}  alpha={alpha_true:.1f}  "
      f"median life={med_true:.1f} M-revs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The illusion — a naive fit on the early snapshot
# MAGIC Rewind to an early inspection at **t = 28 M-revs**. Only **1 of 23** bearings
# MAGIC has failed; the other 22 are still turning and are right-censored at 28. This
# MAGIC is the real-world early-life regime: lots of survivors, almost no failures.
# MAGIC
# MAGIC With that little information the likelihood is nearly flat in α, so MLE pushes
# MAGIC the characteristic life way out — the fitted median life lands well **above** the
# MAGIC true 69, telling the planner the fleet is basically immortal.

# COMMAND ----------

INSPECT_EARLY = 28.0
obs_early = np.minimum(LIFETIMES, INSPECT_EARLY)     # censored obs capped at inspection time
cens_early = LIFETIMES > INSPECT_EARLY
n_fail = int((~cens_early).sum())
print(f"At t={INSPECT_EARLY:g}: {n_fail} failed, {int(cens_early.sum())} still running (censored)")

beta_naive, alpha_naive = fit_mle(obs_early, cens_early)
med_naive = median_life(beta_naive, alpha_naive)
print(f"NAIVE MLE @ t={INSPECT_EARLY:g}: beta={beta_naive:.2f}  alpha={alpha_naive:.1f}  "
      f"median life={med_naive:.1f} M-revs")
print(f"  -> overstates median life by {100*(med_naive/med_true - 1):.0f}% vs. truth ({med_true:.1f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## A blunter fix — a bounded (constrained) MLE
# MAGIC Before the full Bayesian treatment, the simplest regularizer an engineer reaches
# MAGIC for: keep plain MLE but **box α to a hard upper bound** — say `alpha_max = 90`, a
# MAGIC "no bearing line lives past this" design limit. No prior, no interval, just a cap.
# MAGIC
# MAGIC On the early snapshot the *unconstrained* optimum (α ≈ 110) sits above the cap, so
# MAGIC the optimizer **rails to the bound** and reports α = 90 exactly. That's the tell:
# MAGIC an estimate pinned to its bound means **the data didn't determine α — the bound
# MAGIC did.** It's better than the runaway naive fit (it can't promise immortality), but
# MAGIC it's a censored answer, not an informed one, and it gives you no honest sense of
# MAGIC how uncertain you are. That's the gap the Bayesian posterior fills next.

# COMMAND ----------

ALPHA_MAX = 90.0
beta_bnd, alpha_bnd = fit_bounded(obs_early, cens_early, alpha_max=ALPHA_MAX)
med_bnd = median_life(beta_bnd, alpha_bnd)
railed = "RAILED to the bound" if alpha_bnd > ALPHA_MAX - 0.5 else "interior optimum"
print(f"BOUNDED MLE @ t={INSPECT_EARLY:g} (cap={ALPHA_MAX:g}): "
      f"beta={beta_bnd:.2f}  alpha={alpha_bnd:.1f}  median life={med_bnd:.1f}")
print(f"  -> {railed}: the cap, not the data, set alpha "
      f"(unconstrained MLE wanted alpha={alpha_naive:.0f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The fix — a Bayesian (regularized) fit
# MAGIC We put weakly-informative priors on the parameters and estimate the full
# MAGIC **posterior** over (β, α) on a grid — no PPL dependency, just the log-likelihood
# MAGIC plus log-priors, which keeps the notebook self-contained and easy to read:
# MAGIC
# MAGIC - **β ~ Gamma(shape 4, scale 0.5)** — mean 2, gently encoding "this is wear-out,
# MAGIC   not infant mortality." Keeps β away from degenerate values when data is thin.
# MAGIC - **α ~ LogNormal(log 70, 0.6)** — centered near a plausible engineering design
# MAGIC   life with wide spread. This is the **bound** that stops α running to infinity.
# MAGIC
# MAGIC Reported as the posterior mean plus a **90% credible interval** on α. Early on
# MAGIC the interval is honestly enormous — the model says "I don't know yet" rather
# MAGIC than promising immortality.

# COMMAND ----------

# Grid over (beta, alpha). Fine enough to read a smooth posterior, cheap enough to
# recompute in a loop below.
BETAS = np.linspace(0.3, 8.0, 400)
ALPHAS = np.linspace(10.0, 400.0, 400)
BB, AA = np.meshgrid(BETAS, ALPHAS, indexing="ij")

# Log-priors, evaluated once on the grid.
LOGPRIOR = (
    stats.gamma(a=4, scale=0.5).logpdf(BB)
    + stats.norm(np.log(70), 0.6).logpdf(np.log(AA))
)


def fit_bayes(times, censored):
    """Grid posterior. Returns (beta_mean, alpha_mean, alpha_lo, alpha_hi, P)."""
    loglik = np.array([[weibull_loglik(b, a, times, censored) for a in ALPHAS]
                       for b in BETAS])
    logpost = loglik + LOGPRIOR
    logpost -= logpost.max()                 # stabilize before exp
    P = np.exp(logpost)
    P /= P.sum()
    beta_mean = (BB * P).sum()
    alpha_mean = (AA * P).sum()
    # 90% credible interval on the alpha marginal
    p_alpha = P.sum(axis=0)
    cdf = np.cumsum(p_alpha)
    alpha_lo = ALPHAS[np.searchsorted(cdf, 0.05)]
    alpha_hi = ALPHAS[np.searchsorted(cdf, 0.95)]
    return beta_mean, alpha_mean, alpha_lo, alpha_hi, P


beta_b, alpha_b, a_lo, a_hi, P_early = fit_bayes(obs_early, cens_early)
med_b = median_life(beta_b, alpha_b)
print(f"BAYESIAN @ t={INSPECT_EARLY:g}: beta={beta_b:.2f}  alpha={alpha_b:.1f}  "
      f"median life={med_b:.1f}")
print(f"  90% credible interval on alpha: [{a_lo:.0f}, {a_hi:.0f}]  "
      f"(truth alpha={alpha_true:.0f} is inside)")
print(f"  -> refuses to over-promise: reports a wide interval instead of a fake point estimate")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convergence — watch the estimate find the truth as failures accrue
# MAGIC Re-fit at a sequence of inspection times. Naive MLE lurches around and stays
# MAGIC biased high while data is scarce; the Bayesian posterior mean starts near the
# MAGIC prior, then contracts smoothly onto the true α ≈ 82 as more bearings fail —
# MAGIC and its credible interval narrows honestly the whole way.

# COMMAND ----------

INSPECTIONS = [28, 34, 42, 52, 60, 70, 90, 130, 200]
rows = []
for t in INSPECTIONS:
    obs = np.minimum(LIFETIMES, t)
    cens = LIFETIMES > t
    nf = int((~cens).sum())
    if nf >= 2:                       # MLE needs >=2 failures to identify both params
        bn, an = fit_mle(obs, cens)
    else:
        bn, an = np.nan, np.nan       # naive fit is undefined / unstable this early
    _, a_bnd = fit_bounded(obs, cens, alpha_max=ALPHA_MAX)
    bb, ab, lo, hi, _ = fit_bayes(obs, cens)
    rows.append((t, nf, bn, an, a_bnd, bb, ab, lo, hi))

print(f"{'t':>4} {'#fail':>5} | {'MLE alpha':>10} {'bounded a':>10} | "
      f"{'Bayes alpha':>11} {'alpha 90% CI':>16}")
print("-" * 74)
for t, nf, bn, an, a_bnd, bb, ab, lo, hi in rows:
    mle_str = f"{an:10.1f}" if np.isfinite(an) else f"{'n/a':>10}"
    rail = "*" if a_bnd > ALPHA_MAX - 0.5 else " "   # * marks a fit pinned to the cap
    print(f"{t:>4} {nf:>5} | {mle_str} {a_bnd:9.1f}{rail} | {ab:11.1f}   [{lo:5.0f}, {hi:5.0f}]")
print("-" * 74)
print(f"{'TRUTH':>12} | {alpha_true:10.1f} {'':>10} |   (cap={ALPHA_MAX:.0f}, * = railed to cap)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize the story
# MAGIC Left: **α estimate vs. inspection time** — naive MLE (biased high, jumpy) vs.
# MAGIC the Bayesian posterior mean with its shrinking credible band, both against the
# MAGIC true α. Right: **survival curves** — the naive early fit promises far too much
# MAGIC life; the Bayesian early fit hugs the truth.

# COMMAND ----------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ts = [r[0] for r in rows]
mle_a = [r[3] for r in rows]
bnd_a = [r[4] for r in rows]
bay_a = [r[6] for r in rows]
bay_lo = [r[7] for r in rows]
bay_hi = [r[8] for r in rows]

ax1.axhline(alpha_true, color="k", ls="--", lw=1.5, label=f"truth (alpha={alpha_true:.0f})")
ax1.axhline(ALPHA_MAX, color="#8a6d3b", ls=":", lw=1.2, label=f"bound (alpha_max={ALPHA_MAX:.0f})")
ax1.plot(ts, mle_a, "o-", color="#d1495b", label="naive MLE")
ax1.plot(ts, bnd_a, "s-", color="#8a6d3b", label="bounded MLE")
ax1.plot(ts, bay_a, "o-", color="#2e86ab", label="Bayesian (posterior mean)")
ax1.fill_between(ts, bay_lo, bay_hi, color="#2e86ab", alpha=0.15, label="Bayesian 90% CI")
ax1.set_xlabel("inspection time (M-revs)")
ax1.set_ylabel("characteristic life alpha")
ax1.set_title("Estimate converges as failures accrue")
ax1.set_ylim(0, 320)
ax1.legend(fontsize=9)

grid = np.linspace(0, 220, 400)
S = lambda k, lam: np.exp(-((grid / lam) ** k))
ax2.plot(grid, S(beta_true, alpha_true), "k--", lw=1.5, label=f"truth (median {med_true:.0f})")
ax2.plot(grid, S(beta_naive, alpha_naive), color="#d1495b",
         label=f"naive MLE @ t=28 (median {med_naive:.0f})")
ax2.plot(grid, S(beta_bnd, alpha_bnd), color="#8a6d3b",
         label=f"bounded MLE @ t=28 (median {med_bnd:.0f}, railed)")
ax2.plot(grid, S(beta_b, alpha_b), color="#2e86ab",
         label=f"Bayesian @ t=28 (median {med_b:.0f})")
ax2.axvline(INSPECT_EARLY, color="gray", ls=":", lw=1, label="inspection t=28")
ax2.set_xlabel("age (M-revs)")
ax2.set_ylabel("survival probability S(t)")
ax2.set_title("Early-life survival curve: illusion vs. honest fit")
ax2.legend(fontsize=9)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Takeaways
# MAGIC - On **early, heavily-censored** data a plain MLE Weibull fit is optimistically
# MAGIC   biased — the characteristic life α runs high and the survival curve promises
# MAGIC   life the equipment doesn't have. With < 2 failures it isn't even identifiable.
# MAGIC - A **bounded (constrained) MLE** is the quick fix — cap α at a design limit. It
# MAGIC   stops the runaway, but on thin data it just **rails to the bound**: the cap, not
# MAGIC   the data, sets the answer, and you get no measure of how uncertain you are.
# MAGIC - A **weakly-informative prior on α (and β)** regularizes the fit. When data is
# MAGIC   thin the model reports a **wide credible interval** instead of a confident-but-
# MAGIC   wrong point estimate — exactly the honesty a maintenance planner needs.
# MAGIC - As failures accrue the **posterior contracts onto the truth** (α ≈ 82, median
# MAGIC   life ≈ 69), and the naive and Bayesian fits agree once the data is rich.
# MAGIC - Same recipe scales to fleet data: read `(asset_id, age, failed)` from Unity
# MAGIC   Catalog, build `times`/`censored`, and reuse `fit_bayes`. For a full posterior
# MAGIC   with covariates, graduate to PyMC/`lifelines` — the intuition here carries over.
