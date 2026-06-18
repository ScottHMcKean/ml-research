# Databricks notebook source
# 01_generate_sales.py
# -----------------------------------------------------------------------------
# Synthetic SALES / DEPLETIONS data for a multi-brand beverage company workshop.
# Powers Lab 1 (store -> SKU recommender) and Lab 3 (demand forecasting).
#
# Design notes (why the data looks like it does):
#   * Beverage three-tier distribution vocabulary: distributors "deplete" product
#     to retail accounts (sell-through). fact_depletions is the weekly grain.
#   * The demand model embeds REAL drivers so Lab 3's ML models can genuinely
#     beat a seasonal-naive baseline: annual seasonality (summer seltzer peak),
#     a per-brand trend, promo lift, price elasticity, temperature, and holidays.
#   * Each account has a latent "preference profile" over category/pack so the
#     account x SKU matrix has structure for ALS (Lab 1) to exploit.
#
# Invariants enforced:
#   * No depletions before a SKU's launch_date (Moose Juice launches
#     Apr 2026 -> naturally sparse -> exercises intermittent-demand methods).
#   * fact_assortment (what an account currently carries) is a SUBSET of the
#     SKUs it has depleted recently -> Lab 1 recommends into the GAP.
#   * net_revenue = cases * avg_price (per-case wholesale), units = cases * pack_units.
#
# Coverage: 2024-01-01 .. 2026-06-22 (~130 ISO weeks).
# Sizing: ~800 accounts, ~50 SKUs, ~2M depletion rows (printed at the end).
#
# NOTE: written with plain '#' comments (no '# MAGIC %md') so it runs correctly
# as a serverless jobs-submit notebook task. Run it once to build the data.
# -----------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import date, timedelta

SEED = 42
np.random.seed(SEED)

START_DATE = date(2024, 1, 1)
END_DATE   = date(2026, 6, 22)
N_WEEKS    = ((END_DATE - START_DATE).days // 7) + 1     # ~130
WEEK_DATES = [START_DATE + timedelta(weeks=w) for w in range(N_WEEKS)]
print(f"Weeks: {N_WEEKS}  ({WEEK_DATES[0]} .. {WEEK_DATES[-1]})")

def week_index_for(d: date) -> int:
    """Earliest week index whose week-start is >= d (used for launch/onboard)."""
    return max(0, (d - START_DATE).days // 7)

# COMMAND ----------

# dim_product -- explicit SKU catalog.
# category in {hard_seltzer, fmb, rtd_cocktail, spirit_rtd, whisky, hydration}
# base_price = per-case wholesale ($). pack_units = eaches per case.
# popularity = relative depletion weight. trend_pct_yr = annual volume trend.
# season_amp = amplitude of summer peak. temp_coef = sensitivity to warm weather.
products = []
def add(sku_id, brand, category, flavor, pack, pack_units, abv, base_price,
        launch, popularity, trend_pct_yr, season_amp, temp_coef, is_seasonal=False):
    products.append(dict(sku_id=sku_id, brand=brand, category=category, flavor=flavor,
        pack_config=pack, pack_units=pack_units, abv=abv, base_price=base_price,
        launch_date=launch, popularity=popularity, trend_pct_yr=trend_pct_yr,
        season_amp=season_amp, temp_coef=temp_coef, is_seasonal=is_seasonal))

L0 = date(2024, 1, 1)   # established before the window
# ---- Thirsty Otter (hard seltzer) -- flagship, high volume, maturing (slight decline) ----
add("WC-VAR1-12",  "Thirsty Otter", "hard_seltzer", "Variety Pack No.1", "12x12oz", 12, 5.0, 15.50, L0, 100, -0.06, 0.45, 0.35)
add("WC-VAR3-12",  "Thirsty Otter", "hard_seltzer", "Variety Pack No.3", "12x12oz", 12, 5.0, 15.50, L0,  82, -0.04, 0.45, 0.35)
add("WC-BCH-12",   "Thirsty Otter", "hard_seltzer", "Black Cherry",      "12x12oz", 12, 5.0, 15.50, L0,  78, -0.05, 0.45, 0.35)
add("WC-MNG-12",   "Thirsty Otter", "hard_seltzer", "Mango",             "12x12oz", 12, 5.0, 15.50, L0,  70, -0.03, 0.45, 0.35)
add("WC-WML-12",   "Thirsty Otter", "hard_seltzer", "Watermelon",        "12x12oz", 12, 5.0, 15.50, L0,  58, -0.02, 0.50, 0.40, True)
add("WC-LIM-12",   "Thirsty Otter", "hard_seltzer", "Natural Lime",      "12x12oz", 12, 5.0, 15.50, L0,  52, -0.05, 0.45, 0.35)
add("WC-RSP-12",   "Thirsty Otter", "hard_seltzer", "Raspberry",         "12x12oz", 12, 5.0, 15.50, L0,  44, -0.05, 0.45, 0.35)
add("WC-GRP-12",   "Thirsty Otter", "hard_seltzer", "Ruby Grapefruit",   "12x12oz", 12, 5.0, 15.50, L0,  40, -0.04, 0.45, 0.35)
add("WC-PIN-12",   "Thirsty Otter", "hard_seltzer", "Pineapple",         "12x12oz", 12, 5.0, 15.50, L0,  36, -0.02, 0.48, 0.38, True)
add("WC-BCH-192",  "Thirsty Otter", "hard_seltzer", "Black Cherry",      "1x19.2oz",1, 5.0,  2.65, L0,  62, -0.03, 0.40, 0.40)
add("WC-MNG-192",  "Thirsty Otter", "hard_seltzer", "Mango",             "1x19.2oz",1, 5.0,  2.65, L0,  50, -0.02, 0.40, 0.40)
add("WC-VAR-24",   "Thirsty Otter", "hard_seltzer", "Variety Pack No.1", "24x12oz", 24, 5.0, 28.00, L0,  66, -0.04, 0.45, 0.35)
add("WC-SRG-VAR",  "Thirsty Otter", "hard_seltzer", "Surge Variety",     "12x12oz", 12, 8.0, 17.50, date(2024,3,4), 38, 0.08, 0.42, 0.30)
add("WC-SRG-BB192","Thirsty Otter", "hard_seltzer", "Surge Blackberry",  "1x19.2oz",1, 8.0,  3.10, date(2024,3,4), 30, 0.10, 0.40, 0.32)
add("WC-0PCT-VAR", "Thirsty Otter", "hard_seltzer", "0% Alc Variety",    "12x12oz", 12, 0.0, 16.00, date(2024,4,1), 18, 0.20, 0.35, 0.30)
add("WC-VS-VAR",   "Thirsty Otter", "spirit_rtd",   "Vodka Soda Variety","8x12oz",  8, 4.5, 17.00, date(2024,6,3), 28, 0.18, 0.45, 0.35)
# ---- Lazy Llama (FMB) -- stable, broad ----
add("MK-LEM-12",   "Lazy Llama", "fmb", "Lemonade",          "12x12oz", 12, 5.0, 14.75, L0, 74,  0.01, 0.40, 0.30)
add("MK-VAR-12",   "Lazy Llama", "fmb", "Variety",           "12x12oz", 12, 5.0, 14.75, L0, 60,  0.02, 0.40, 0.30)
add("MK-BCH-12",   "Lazy Llama", "fmb", "Black Cherry",      "12x12oz", 12, 5.0, 14.75, L0, 46,  0.01, 0.40, 0.30)
add("MK-STR-12",   "Lazy Llama", "fmb", "Strawberry",        "12x12oz", 12, 5.0, 14.75, L0, 40,  0.00, 0.42, 0.32)
add("MK-MNG-12",   "Lazy Llama", "fmb", "Mango",             "12x12oz", 12, 5.0, 14.75, L0, 38,  0.03, 0.44, 0.34, True)
add("MK-LEM-24",   "Lazy Llama", "fmb", "Lemonade",          "24x12oz", 24, 5.0, 27.00, L0, 44,  0.01, 0.40, 0.30)
add("MK-LEM-16",   "Lazy Llama", "fmb", "Lemonade",          "1x16oz",  1, 5.0,  2.30, L0, 48,  0.00, 0.38, 0.32)
add("MK-FRZ-VAR",  "Lazy Llama", "fmb", "Freeze Variety",    "10x10oz", 10, 5.0, 15.25, date(2024,5,1), 26, 0.05, 0.55, 0.45, True)
# ---- Fancy Flamingo (RTD cocktail) -- growth ----
add("CJ-MRG-12",   "Fancy Flamingo", "rtd_cocktail", "Margarita",          "12x12oz", 12, 5.8, 17.25, L0, 52, 0.09, 0.42, 0.34)
add("CJ-MRG-24",   "Fancy Flamingo", "rtd_cocktail", "Margarita",          "24x12oz", 24, 5.8, 31.50, L0, 30, 0.10, 0.42, 0.34)
add("CJ-MUL-12",   "Fancy Flamingo", "rtd_cocktail", "Moscow Mule",        "12x12oz", 12, 5.8, 17.25, L0, 28, 0.08, 0.40, 0.30)
add("CJ-PAL-12",   "Fancy Flamingo", "rtd_cocktail", "Paloma",             "12x12oz", 12, 5.8, 17.25, date(2024,3,1), 24, 0.12, 0.44, 0.36)
add("CJ-CUC-12",   "Fancy Flamingo", "rtd_cocktail", "Cucumber Margarita", "12x12oz", 12, 5.8, 17.25, date(2024,4,15), 22, 0.14, 0.44, 0.36)
add("CJ-MRG-192",  "Fancy Flamingo", "rtd_cocktail", "Margarita",          "1x19.2oz",1, 5.8,  3.05, L0, 26, 0.08, 0.40, 0.34)
# ---- Moose Juice (spirit RTD) -- LAUNCHES Apr 2026 -> sparse / intermittent ----
FLD = date(2026, 4, 6)
add("FLD-TRD-6",   "Moose Juice", "spirit_rtd", "Traditional", "6x12oz",  6, 5.3, 11.50, FLD, 20, 0.0, 0.40, 0.30)
add("FLD-VAR-8",   "Moose Juice", "spirit_rtd", "Variety",     "8x12oz",  8, 5.3, 14.50, FLD, 18, 0.0, 0.40, 0.30)
add("FLD-STR-6",   "Moose Juice", "spirit_rtd", "Strong",      "6x12oz",  6, 8.5, 12.50, FLD, 12, 0.0, 0.38, 0.28)
add("FLD-ZRO-6",   "Moose Juice", "spirit_rtd", "Zero",        "6x12oz",  6, 5.0, 11.50, FLD, 10, 0.0, 0.36, 0.28)
add("FLD-TRD-12",  "Moose Juice", "spirit_rtd", "Traditional", "12x12oz", 12, 5.3, 21.00, FLD, 14, 0.0, 0.40, 0.30)
# ---- Old Grumpy Bear (whisky) -- low volume, winter-skewed (negative season_amp) ----
add("BF-TO-750",   "Old Grumpy Bear", "whisky", "Triple Oak 7yr", "1x750ml",  1, 42.5, 22.00, L0, 10, 0.04, -0.30, -0.20)
add("BF-TO-175",   "Old Grumpy Bear", "whisky", "Triple Oak 7yr", "1x1.75L",  1, 42.5, 41.00, L0,  6, 0.04, -0.30, -0.20)
# ---- Hydro Hippo (hydration, non-alc) -- new-ish, growth, summer ----
MAS = date(2024, 9, 1)
add("MAS-LIM-12",  "Hydro Hippo", "hydration", "Limon",    "12x16.9oz", 12, 0.0, 13.00, MAS, 16, 0.25, 0.50, 0.45, True)
add("MAS-BER-12",  "Hydro Hippo", "hydration", "Berry",    "12x16.9oz", 12, 0.0, 13.00, MAS, 13, 0.25, 0.50, 0.45, True)
add("MAS-TRO-12",  "Hydro Hippo", "hydration", "Tropical", "12x16.9oz", 12, 0.0, 13.00, MAS, 11, 0.28, 0.52, 0.47, True)

dim_product = pd.DataFrame(products)
dim_product["launch_week"] = dim_product["launch_date"].apply(week_index_for)
# Snap launch_date to the Monday (week_start) of its launch week. Depletions are
# weekly buckets keyed by week_start, so a mid-week launch_date would otherwise
# sit AFTER its own launch-week Monday and trip the no-pre-launch invariant.
dim_product["launch_date"] = dim_product["launch_week"].apply(lambda w: WEEK_DATES[int(w)])
print(f"dim_product: {len(dim_product)} SKUs across {dim_product['brand'].nunique()} brands")

# COMMAND ----------

# dim_distributor -- three-tier beverage distributors, region-assigned.
REGIONS = ["West", "Southwest", "Midwest", "Northeast", "Southeast", "Canada"]
distributors = [
    ("DST-001", "Reyes Beverage Group - West",   "West"),
    ("DST-002", "Reyes Beverage Group - SoCal",   "West"),
    ("DST-003", "Hensley Beverage",               "Southwest"),
    ("DST-004", "Crescent Crown Distributing",    "Southwest"),
    ("DST-005", "Breakthru Beverage - IL",        "Midwest"),
    ("DST-006", "Lee Distributors",               "Midwest"),
    ("DST-007", "Sheehan Family Companies",       "Northeast"),
    ("DST-008", "Manhattan Beer Distributors",    "Northeast"),
    ("DST-009", "Eagle Rock Distributing",        "Southeast"),
    ("DST-010", "Gold Coast Beverage",            "Southeast"),
    ("DST-011", "Southern Glazer's W&S - National","West"),
    ("DST-012", "RNDC - South",                   "Southeast"),
    ("DST-013", "Great Lakes Wine & Spirits",     "Midwest"),
    ("DST-014", "Columbia Distributing",          "West"),
    ("DST-015", "Andrew Peller Import",           "Canada"),
    ("DST-016", "Sleeman Distribution",           "Canada"),
]
dim_distributor = pd.DataFrame(distributors, columns=["distributor_id", "name", "region"])
print(f"dim_distributor: {len(dim_distributor)}")

# COMMAND ----------

# dim_account -- ~800 retail accounts with a latent preference profile.
# banner drives channel + which categories/packs the account skews toward.
N_ACCOUNTS = 800
# region population weights (US-heavy, ~10% Canada)
REGION_W = {"West":0.22, "Southwest":0.16, "Midwest":0.20, "Northeast":0.18, "Southeast":0.14, "Canada":0.10}
# banner -> (channel, weight, size bias, category-affinity profile)
# category-affinity: relative propensity to carry each category (latent ALS structure)
BANNERS = {
    "grocery":     dict(channel="off_premise", w=0.22, cats=dict(hard_seltzer=1.0, fmb=1.0, rtd_cocktail=0.8, spirit_rtd=0.5, whisky=0.2, hydration=0.9)),
    "convenience": dict(channel="off_premise", w=0.26, cats=dict(hard_seltzer=1.0, fmb=0.9, rtd_cocktail=0.6, spirit_rtd=0.4, whisky=0.1, hydration=1.0)),
    "liquor":      dict(channel="off_premise", w=0.20, cats=dict(hard_seltzer=0.9, fmb=0.8, rtd_cocktail=1.0, spirit_rtd=1.0, whisky=1.0, hydration=0.2)),
    "club":        dict(channel="off_premise", w=0.08, cats=dict(hard_seltzer=1.0, fmb=0.9, rtd_cocktail=0.7, spirit_rtd=0.3, whisky=0.5, hydration=0.6)),
    "big_box":     dict(channel="off_premise", w=0.12, cats=dict(hard_seltzer=1.0, fmb=1.0, rtd_cocktail=0.8, spirit_rtd=0.4, whisky=0.4, hydration=0.8)),
    "bar_resto":   dict(channel="on_premise",  w=0.12, cats=dict(hard_seltzer=0.8, fmb=0.7, rtd_cocktail=1.0, spirit_rtd=0.9, whisky=0.9, hydration=0.3)),
}
# pack-size affinity by banner: club skews 24-packs, convenience skews singles, etc.
PACK_AFFINITY = {
    "grocery":     {"12x12oz":1.0, "24x12oz":0.7, "1x19.2oz":0.5, "1x16oz":0.5, "8x12oz":0.8, "6x12oz":0.7, "10x10oz":0.8, "12x16.9oz":0.9, "1x750ml":0.6, "1x1.75L":0.4},
    "convenience": {"12x12oz":0.6, "24x12oz":0.2, "1x19.2oz":1.0, "1x16oz":1.0, "8x12oz":0.5, "6x12oz":0.6, "10x10oz":0.4, "12x16.9oz":1.0, "1x750ml":0.3, "1x1.75L":0.1},
    "liquor":      {"12x12oz":1.0, "24x12oz":0.6, "1x19.2oz":0.6, "1x16oz":0.4, "8x12oz":0.9, "6x12oz":0.9, "10x10oz":0.7, "12x16.9oz":0.3, "1x750ml":1.0, "1x1.75L":1.0},
    "club":        {"12x12oz":0.7, "24x12oz":1.0, "1x19.2oz":0.2, "1x16oz":0.2, "8x12oz":0.6, "6x12oz":0.4, "10x10oz":0.6, "12x16.9oz":0.8, "1x750ml":0.7, "1x1.75L":0.9},
    "big_box":     {"12x12oz":1.0, "24x12oz":0.9, "1x19.2oz":0.5, "1x16oz":0.5, "8x12oz":0.8, "6x12oz":0.7, "10x10oz":0.7, "12x16.9oz":0.9, "1x750ml":0.6, "1x1.75L":0.6},
    "bar_resto":   {"12x12oz":0.8, "24x12oz":0.7, "1x19.2oz":0.4, "1x16oz":0.6, "8x12oz":0.7, "6x12oz":0.7, "10x10oz":0.5, "12x16.9oz":0.3, "1x750ml":0.9, "1x1.75L":0.8},
}
SIZE_TIERS = {"A": 2.2, "B": 1.0, "C": 0.45}   # volume multiplier

CITIES = {  # region -> list of (city, state, lat, lon)
    "West":      [("Los Angeles","CA",34.05,-118.24),("San Francisco","CA",37.77,-122.42),("Seattle","WA",47.61,-122.33),("Portland","OR",45.52,-122.68),("Denver","CO",39.74,-104.99)],
    "Southwest": [("Phoenix","AZ",33.45,-112.07),("Las Vegas","NV",36.17,-115.14),("Tucson","AZ",32.22,-110.97),("Albuquerque","NM",35.08,-106.65),("Dallas","TX",32.78,-96.80)],
    "Midwest":   [("Chicago","IL",41.88,-87.63),("Detroit","MI",42.33,-83.05),("Minneapolis","MN",44.98,-93.27),("Columbus","OH",39.96,-83.00),("St. Louis","MO",38.63,-90.20)],
    "Northeast": [("New York","NY",40.71,-74.01),("Boston","MA",42.36,-71.06),("Philadelphia","PA",39.95,-75.17),("Pittsburgh","PA",40.44,-79.996),("Newark","NJ",40.74,-74.17)],
    "Southeast": [("Atlanta","GA",33.75,-84.39),("Miami","FL",25.76,-80.19),("Charlotte","NC",35.23,-80.84),("Nashville","TN",36.16,-86.78),("Tampa","FL",27.95,-82.46)],
    "Canada":    [("Toronto","ON",43.65,-79.38),("Vancouver","BC",49.28,-123.12),("Calgary","AB",51.04,-114.07),("Montreal","QC",45.50,-73.57),("Ottawa","ON",45.42,-75.69)],
}
CHAINS = {
    "grocery":["Kroger","Safeway","Publix","Albertsons","Hy-Vee","Loblaws"],
    "convenience":["7-Eleven","Circle K","QuikTrip","Wawa","Sheetz","Couche-Tard"],
    "liquor":["Total Wine","BevMo","Spec's","ABC Fine Wine","LCBO","Liquor Depot"],
    "club":["Costco","Sam's Club","BJ's"],
    "big_box":["Walmart","Target","Meijer"],
    "bar_resto":["Buffalo Wild Wings","Applebee's","Local Tavern","Hotel Bar","Sports Grill"],
}

rng = np.random.default_rng(SEED)
banner_names = list(BANNERS.keys())
banner_w = np.array([BANNERS[b]["w"] for b in banner_names]); banner_w /= banner_w.sum()
region_names = list(REGION_W.keys())
region_w = np.array([REGION_W[r] for r in region_names]); region_w /= region_w.sum()

accts = []
for i in range(N_ACCOUNTS):
    banner = rng.choice(banner_names, p=banner_w)
    region = rng.choice(region_names, p=region_w)
    city, state, lat, lon = CITIES[region][rng.integers(len(CITIES[region]))]
    tier = rng.choice(["A","B","C"], p=[0.18,0.5,0.32])
    chain = CHAINS[banner][rng.integers(len(CHAINS[banner]))]
    # onboard: most accounts active from the start; ~15% ramp in during the window
    onboard_week = 0 if rng.random() < 0.85 else int(rng.integers(0, N_WEEKS-20))
    accts.append(dict(
        account_id=f"ACC-{i+1:05d}", account_name=f"{chain} #{rng.integers(100,9999)}",
        banner=banner, chain=chain, channel=BANNERS[banner]["channel"],
        country="Canada" if region=="Canada" else "USA",
        region=region, state_prov=state, city=city,
        latitude=round(float(lat+rng.normal(0,0.05)),4), longitude=round(float(lon+rng.normal(0,0.05)),4),
        size_tier=tier, onboard_week=onboard_week,
    ))
dim_account = pd.DataFrame(accts)
print(f"dim_account: {len(dim_account)}  banners={dim_account['banner'].value_counts().to_dict()}")

# COMMAND ----------

# Assign each account a region distributor + a CARRIED SKU set (its assortment),
# plus per-(account,sku) base demand parameters. This small table (~tens of
# thousands of rows) drives the Spark fact generation in the next cell.
reg_to_dists = {r: dim_distributor[dim_distributor.region==r].distributor_id.tolist() for r in region_names}
# fallback for any region with no distributor
all_dists = dim_distributor.distributor_id.tolist()

prod = dim_product.to_dict("records")
series_rows = []
for a in dim_account.itertuples():
    banner = a.banner
    cat_aff = BANNERS[banner]["cats"]
    pack_aff = PACK_AFFINITY[banner]
    dists = reg_to_dists.get(a.region) or all_dists
    dist_id = dists[rng.integers(len(dists))]
    size_mult = SIZE_TIERS[a.size_tier]
    # number of SKUs this account carries scales with size tier
    n_carry = int(np.clip(rng.normal({"A":34,"B":24,"C":15}[a.size_tier], 5), 8, len(prod)))
    # score each SKU for this account: category affinity * pack affinity * popularity * noise
    scores = []
    for p in prod:
        s = cat_aff.get(p["category"],0.3) * pack_aff.get(p["pack_config"],0.5) * (p["popularity"]/100.0)
        s *= (0.6 + 0.8*rng.random())   # idiosyncratic taste
        scores.append(s)
    order = np.argsort(scores)[::-1][:n_carry]
    for idx in order:
        p = prod[idx]
        # base weekly cases: popularity * size * affinity, scaled small for singles (pack_units=1)
        aff = cat_aff.get(p["category"],0.3) * pack_aff.get(p["pack_config"],0.5)
        base = (p["popularity"]/100.0) * size_mult * aff * 9.0
        if p["pack_units"] == 1:
            base *= 2.2   # singles move more "cases" (cases here = display units)
        base *= (0.7 + 0.6*rng.random())
        start_week = max(int(p["launch_week"]), int(a.onboard_week))
        if start_week > N_WEEKS - 2:
            # Launched/onboarded essentially at the end of the window — there is no
            # room for history without generating rows BEFORE the SKU's launch_date
            # (which would break the no-pre-launch invariant). Skip this series.
            # launch_week is a hard floor; never clamp start_week below it.
            continue
        series_rows.append(dict(
            account_id=a.account_id, sku_id=p["sku_id"], distributor_id=dist_id,
            region=a.region, category=p["category"], base_price=float(p["base_price"]),
            pack_units=int(p["pack_units"]), base_cases=float(round(base,3)),
            trend_pct_yr=float(p["trend_pct_yr"]), season_amp=float(p["season_amp"]),
            temp_coef=float(p["temp_coef"]), elasticity=float(round(rng.uniform(1.1,1.8),3)),
            promo_prob=float(round(rng.uniform(0.06,0.18),3)), start_week=int(start_week),
        ))
series_pdf = pd.DataFrame(series_rows)
print(f"account x SKU carried series: {len(series_pdf):,}")

# COMMAND ----------

# Build the weekly fact in SPARK (scales to millions of rows).
# Strategy: register the series-params + a weeks dimension, cross-join with a
# week filter, then compute the demand model with Spark SQL functions.
from pyspark.sql import functions as F

# Arrow speeds up pandas->Spark conversion; it's ON by default on serverless and
# the conf is locked there, so set it best-effort (no-op / skip on serverless).
try:
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
except Exception:
    pass
series_sdf = spark.createDataFrame(series_pdf)
series_sdf.createOrReplaceTempView("series_params")

# weeks dimension with week_start date + day-of-year (for seasonality/temp)
weeks_pdf = pd.DataFrame({
    "week": list(range(N_WEEKS)),
    "week_start": WEEK_DATES,
})
weeks_pdf["doy"] = pd.to_datetime(weeks_pdf["week_start"]).dt.dayofyear
weeks_sdf = spark.createDataFrame(weeks_pdf)
weeks_sdf.createOrReplaceTempView("weeks_dim")

# region temperature offsets (deg F): annual mean + summer amplitude per region
region_temp = {
    "West":      (62, 18), "Southwest": (75, 22), "Midwest": (52, 26),
    "Northeast": (54, 24), "Southeast": (70, 18), "Canada":  (46, 26),
}
rt_pdf = pd.DataFrame([(r, m, a) for r,(m,a) in region_temp.items()],
                      columns=["region","temp_mean","temp_amp"])
spark.createDataFrame(rt_pdf).createOrReplaceTempView("region_temp")

# Holiday lift weeks (US summer + key holidays): doy ranges -> multiplier
fact = spark.sql(f"""
WITH grid AS (
  SELECT s.*, w.week, w.week_start, w.doy,
         rt.temp_mean, rt.temp_amp
  FROM series_params s
  JOIN weeks_dim w   ON w.week >= s.start_week
  JOIN region_temp rt ON rt.region = s.region
),
modeled AS (
  SELECT
    account_id, sku_id, distributor_id, region, category, pack_units,
    week, week_start,
    base_price,
    -- temperature: seasonal sinusoid peaking ~ day 200 (mid-July)
    temp_mean + temp_amp * cos(2*pi()*(doy-200)/365.0)
      + ( sqrt(-2*ln(greatest(1e-12, CAST(conv(substr(md5(concat(account_id,sku_id,week)),1,8),16,10) AS DOUBLE)/4294967296.0)))
          * cos(2*pi()* CAST(conv(substr(md5(concat(account_id,sku_id,week)),9,8),16,10) AS DOUBLE)/4294967296.0) ) * 2.5 AS avg_temp_f,
    -- annual seasonality multiplier (summer peak for seltzer, winter for whisky via negative amp)
    (1 + season_amp * cos(2*pi()*(doy-200)/365.0)) AS season_mult,
    -- linear trend over the ~2.5yr window
    (1 + trend_pct_yr * (week/52.0)) AS trend_mult,
    -- promo: deterministic uniform in [0,1) from a per-row hash (no rand() seed)
    CASE WHEN (CAST(conv(substr(md5(concat(sku_id,account_id,week,'p')),1,8),16,10) AS DOUBLE)/4294967296.0) < promo_prob
         THEN 1 ELSE 0 END AS promo_flag,
    base_cases, elasticity, temp_coef, season_amp
  FROM grid
),
priced AS (
  SELECT *,
    -- promo discounts the case price 12-25%
    round(base_price * (CASE WHEN promo_flag=1 THEN 0.80 ELSE 1.0 END), 2) AS avg_price,
    -- holiday lift around Memorial Day / July 4 / Labor Day / Thanksgiving / Xmas
    CASE
      WHEN dayofyear(week_start) BETWEEN 144 AND 152 THEN 1.25  -- Memorial Day
      WHEN dayofyear(week_start) BETWEEN 181 AND 190 THEN 1.35  -- July 4
      WHEN dayofyear(week_start) BETWEEN 240 AND 250 THEN 1.20  -- Labor Day
      WHEN dayofyear(week_start) BETWEEN 326 AND 334 THEN 1.15  -- Thanksgiving
      WHEN dayofyear(week_start) BETWEEN 354 AND 366 THEN 1.18  -- Christmas/NY
      ELSE 1.0 END AS holiday_mult
  FROM modeled
),
rated AS (
  SELECT *,
    GREATEST(0.05,
      base_cases
      * GREATEST(0.05, season_mult)
      * trend_mult
      * (CASE WHEN promo_flag=1 THEN 1.45 ELSE 1.0 END)              -- promo volume lift
      * pow( (avg_price/base_price), -elasticity )                   -- price elasticity
      * (1 + temp_coef * ((avg_temp_f-65)/30.0))                     -- warm-weather pull
      * holiday_mult
    ) AS rate
  FROM priced
),
counted AS (
  SELECT *,
    -- multiplicative lognormal-ish noise (deterministic Box-Muller from a per-row hash) then round to integer cases
    CAST( round( rate * exp(
      ( sqrt(-2*ln(greatest(1e-12, CAST(conv(substr(md5(concat(account_id,sku_id,week,'n')),1,8),16,10) AS DOUBLE)/4294967296.0)))
        * cos(2*pi()* CAST(conv(substr(md5(concat(account_id,sku_id,week,'n')),9,8),16,10) AS DOUBLE)/4294967296.0) ) * 0.35
    ) ) AS INT) AS cases
  FROM rated
)
SELECT
  account_id, sku_id, distributor_id,
  CAST(week_start AS DATE) AS week,
  cases,
  CAST(cases * pack_units AS INT) AS units,
  round(cases * avg_price, 2) AS net_revenue,
  avg_price,
  promo_flag,
  round(avg_temp_f, 1) AS avg_temp_f
FROM counted
WHERE cases > 0
""")

fact.createOrReplaceTempView("fact_depletions_tmp")
n_fact = fact.count()
print(f"fact_depletions rows: {n_fact:,}")

# COMMAND ----------

# Persist dimensions + fact as managed Delta tables.
sp_product = spark.createDataFrame(dim_product.drop(columns=["launch_week"]))
sp_product = sp_product.withColumn("launch_date", F.col("launch_date").cast("date"))
sp_product.write.mode("overwrite").option("overwriteSchema","true").saveAsTable(SALES_DIM_PRODUCT)

spark.createDataFrame(dim_distributor).write.mode("overwrite").option("overwriteSchema","true").saveAsTable(SALES_DIM_DISTRIBUTOR)
spark.createDataFrame(dim_account.drop(columns=["onboard_week"])).write.mode("overwrite").option("overwriteSchema","true").saveAsTable(SALES_DIM_ACCOUNT)

# dim_date helper (weekly + calendar attributes for forecasting features).
# Drop the integer week index first so week_start can be renamed to `week`
# (the date key) without colliding with the existing integer `week` column.
dim_date = weeks_pdf.drop(columns=["week"]).copy()
dim_date["year"] = pd.to_datetime(dim_date["week_start"]).dt.year
dim_date["month"] = pd.to_datetime(dim_date["week_start"]).dt.month
dim_date["iso_week"] = pd.to_datetime(dim_date["week_start"]).dt.isocalendar().week.astype(int)
dim_date["quarter"] = pd.to_datetime(dim_date["week_start"]).dt.quarter
spark.createDataFrame(dim_date.rename(columns={"week_start":"week"})).write.mode("overwrite").option("overwriteSchema","true").saveAsTable(SALES_DIM_DATE)

spark.table("fact_depletions_tmp").write.mode("overwrite").option("overwriteSchema","true").saveAsTable(SALES_FACT_DEPLETIONS)
print("Wrote dims + fact_depletions.")

# COMMAND ----------

# fact_assortment = the SKUs each account CURRENTLY carries (depleted in the
# last 26 weeks). This is a SUBSET of all depleted SKUs and defines the GAP that
# Lab 1's recommender fills (recommend high-affinity SKUs NOT in the assortment).
spark.sql(f"""
CREATE OR REPLACE TABLE {SALES_FACT_ASSORTMENT} AS
SELECT account_id, sku_id,
       SUM(cases)  AS trailing_26w_cases,
       MAX(week)   AS last_depletion_week
FROM {SALES_FACT_DEPLETIONS}
WHERE week >= date_sub((SELECT MAX(week) FROM {SALES_FACT_DEPLETIONS}), 182)
GROUP BY account_id, sku_id
HAVING SUM(cases) >= 3
""")
n_assort = spark.table(SALES_FACT_ASSORTMENT).count()
print(f"fact_assortment rows: {n_assort:,}")

# COMMAND ----------

# Verification — invariants must hold.
print("=== INVARIANT CHECKS ===")
# 1. No depletions before launch
bad_launch = spark.sql(f"""
  SELECT COUNT(*) AS violations
  FROM {SALES_FACT_DEPLETIONS} f JOIN {SALES_DIM_PRODUCT} p USING (sku_id)
  WHERE f.week < p.launch_date
""").collect()[0]["violations"]
print(f"1. depletions before launch_date (expect 0): {bad_launch}")

# 2. Assortment is subset of depleted SKUs
orphan_assort = spark.sql(f"""
  SELECT COUNT(*) AS orphans FROM {SALES_FACT_ASSORTMENT} a
  LEFT ANTI JOIN (SELECT DISTINCT account_id, sku_id FROM {SALES_FACT_DEPLETIONS}) d
  USING (account_id, sku_id)
""").collect()[0]["orphans"]
print(f"2. assortment rows w/o any depletion (expect 0): {orphan_assort}")

# 3. Seasonality present: summer vs winter seltzer volume
season = spark.sql(f"""
  SELECT CASE WHEN month(week) IN (6,7,8) THEN 'summer'
              WHEN month(week) IN (12,1,2) THEN 'winter' ELSE 'other' END AS seas,
         SUM(cases) AS cases
  FROM {SALES_FACT_DEPLETIONS} f JOIN {SALES_DIM_PRODUCT} p USING (sku_id)
  WHERE p.category='hard_seltzer' GROUP BY 1
""").toPandas().set_index("seas")["cases"].to_dict()
print(f"3. seltzer seasonality summer/winter ratio (expect >1): "
      f"{season.get('summer',0)/max(1,season.get('winter',1)):.2f}")

# 4. Moose Juice only in 2026
fld_range = spark.sql(f"""
  SELECT MIN(week) AS first_wk, MAX(week) AS last_wk
  FROM {SALES_FACT_DEPLETIONS} WHERE sku_id LIKE 'FLD-%'
""").collect()[0]
print(f"4. Moose Juice depletion range (expect 2026): {fld_range['first_wk']} .. {fld_range['last_wk']}")

# 5. Promo price < base price
promo_chk = spark.sql(f"""
  SELECT AVG(CASE WHEN promo_flag=1 THEN avg_price END) AS promo_p,
         AVG(CASE WHEN promo_flag=0 THEN avg_price END) AS reg_p
  FROM {SALES_FACT_DEPLETIONS}
""").collect()[0]
print(f"5. avg promo price {promo_chk['promo_p']:.2f} < regular {promo_chk['reg_p']:.2f} (expect yes)")

print("\nDate span:")
spark.sql(f"SELECT MIN(week) lo, MAX(week) hi, COUNT(*) rows FROM {SALES_FACT_DEPLETIONS}").show()
