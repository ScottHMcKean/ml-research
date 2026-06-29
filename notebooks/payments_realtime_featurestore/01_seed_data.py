# Databricks notebook source

# MAGIC %md
# MAGIC # 01 · Seed Data
# MAGIC
# MAGIC Generates a **synthetic payment-authorization event stream** plus a slow-changing
# MAGIC **account profile** reference dimension, and writes both to Unity Catalog.
# MAGIC
# MAGIC This is the raw input the rest of the demo builds on. The schema is an anonymized
# MAGIC version of a real-time scoring sample pack: one row per authorization request, with
# MAGIC entity keys (`instrument_id`, `account_id`, `bin_prefix`, `category_code`), request
# MAGIC fields (amount, channel, instrument type), an `outcome`, and a learnable `blocked`
# MAGIC label. Everything is synthetic and generic.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import numpy as np
import pandas as pd

dbutils.widgets.text("n_events", "500000", "Number of raw events to generate")
dbutils.widgets.text("history_days", "60", "Days of history to spread events over")

N_EVENTS = int(dbutils.widgets.get("n_events"))
HISTORY_DAYS = int(dbutils.widgets.get("history_days"))
N_ACCOUNTS = 5_000
N_INSTRUMENTS = 50_000
SEED = 42
rng = np.random.default_rng(SEED)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Account profile (reference dimension)
# MAGIC Slow-changing attributes per account. Published to the online store as enrichment
# MAGIC features and joined at scoring time.

# COMMAND ----------

acct_ids = np.array([f"ACC_{i:05d}" for i in range(N_ACCOUNTS)])
account_profile = pd.DataFrame({
    "account_id": acct_ids,
    "account_tier": rng.choice(["standard", "premium"], N_ACCOUNTS, p=[0.85, 0.15]),
    "account_age_days": rng.integers(30, 3650, size=N_ACCOUNTS).astype(int),
    "home_region": rng.choice(["r1", "r2", "r3", "r4"], N_ACCOUNTS, p=[0.4, 0.3, 0.2, 0.1]),
    # Each account carries a latent base risk that drives its historical decline rate.
    "base_risk": np.round(rng.beta(2.0, 12.0, size=N_ACCOUNTS), 4),
})
account_profile["historical_decline_rate"] = np.round(
    np.clip(account_profile["base_risk"] + rng.normal(0, 0.01, N_ACCOUNTS), 0, 1), 4
)
account_profile["enriched_at"] = pd.Timestamp.utcnow().tz_localize(None)
print(f"Accounts: {len(account_profile):,}")
account_profile.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw authorization events
# MAGIC Each event references an account and a payment instrument. Amount is log-normal;
# MAGIC the `blocked` label is a noisy function of amount, velocity, time-of-day, channel,
# MAGIC and the account's latent risk — strong enough to learn (~0.9 AUC) but not trivial.

# COMMAND ----------

now = pd.Timestamp.utcnow().tz_localize(None)
event_ts = now - pd.to_timedelta(rng.integers(0, HISTORY_DAYS * 24 * 3600, size=N_EVENTS), unit="s")

# Map each event to an account, and each account to a small pool of instruments.
acct_idx = rng.integers(0, N_ACCOUNTS, size=N_EVENTS)
instrument_id = np.array([f"INS_{i:06d}" for i in rng.integers(0, N_INSTRUMENTS, size=N_EVENTS)])
account_id = acct_ids[acct_idx]
base_risk = account_profile["base_risk"].to_numpy()[acct_idx]

amount = np.round(np.abs(rng.lognormal(3.5, 1.2, N_EVENTS)), 2)
channel = rng.choice(["ch_1", "ch_2", "ch_3", "ch_4"], N_EVENTS, p=[0.40, 0.30, 0.20, 0.10])
instrument_type = rng.choice(["t1", "t2", "t3"], N_EVENTS, p=[0.45, 0.40, 0.15])
request_type = rng.choice(["std", "pre", "comp"], N_EVENTS, p=[0.60, 0.30, 0.10])
category_code = rng.choice(["A", "B", "C", "D", "E"], N_EVENTS)
bin_prefix = np.array([f"{n}" for n in rng.integers(400000, 559999, size=N_EVENTS)])
flag_a = rng.binomial(1, 0.08, size=N_EVENTS)
hour_of_day = pd.Series(event_ts).dt.hour.to_numpy()
night = ((hour_of_day < 6) | (hour_of_day >= 23)).astype(float)

# Risk signal -> blocked label.
signal = (
    0.55 * np.log1p(amount / 100.0)
    + 3.0 * base_risk
    + 0.9 * night
    + 0.6 * (channel == "ch_4").astype(float)
    + 1.1 * flag_a
    + 0.4 * (request_type == "comp").astype(float)
)
noise = rng.normal(0, 0.5 * signal.std(), size=N_EVENTS)
score = signal + noise
threshold = np.quantile(score, 1 - 0.12)  # block the riskiest ~12%
blocked = (score >= threshold).astype(int)

# Realistic outcome: blocked events hard-fail; a few unblocked soft-fail.
outcome = np.where(
    blocked == 1, "hard_fail",
    np.where(rng.random(N_EVENTS) < 0.10, "soft_fail", "pass"),
)

raw = pd.DataFrame({
    "event_id": [f"EVT_{i:010d}" for i in range(N_EVENTS)],
    "event_ts": event_ts,
    "event_date": pd.Series(event_ts).dt.date.astype("datetime64[ns]"),
    "instrument_id": instrument_id,
    "account_id": account_id,
    "bin_prefix": bin_prefix,
    "category_code": category_code,
    "amount": amount,
    "channel": channel,
    "instrument_type": instrument_type,
    "request_type": request_type,
    "flag_a": flag_a.astype(int),
    "outcome": outcome,
    "blocked": blocked.astype(int),
})
print(f"Events: {len(raw):,}   block rate: {raw['blocked'].mean():.1%}")
raw.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Unity Catalog

# COMMAND ----------

(
    spark.createDataFrame(account_profile.drop(columns=["base_risk"]))
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.account_profile_raw")
)

(
    spark.createDataFrame(raw)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .partitionBy("event_date")
    .saveAsTable(RAW_EVENTS)
)

print(f"Wrote {len(account_profile):,} accounts -> {CATALOG}.{SCHEMA}.account_profile_raw")
print(f"Wrote {len(raw):,} events     -> {RAW_EVENTS}")
display(spark.sql(f"SELECT blocked, COUNT(*) n FROM {RAW_EVENTS} GROUP BY blocked ORDER BY blocked"))
