# Databricks notebook source
# brewery_generator.py  — SHARED telemetry engine (loaded via %run)
# -----------------------------------------------------------------------------
# Defines the ISA-95 asset/tag catalog for a regional brewery and the
# physics-ish value model used to synthesize realistic OT historian telemetry.
#
# Loaded by:
#   * 02_generate_brewery_history.py  (full 2.5yr batch history, data-gen job)
#   * write_to_volume.py              (Lab 2 live micro-batches, streaming job)
#
# Design choices (defensible to a plant engineer):
#   * ISA-95 hierarchy: enterprise > site > area > work_center > equipment.
#   * NARROW tag-based historian schema (one row per tag per timestamp) — exactly
#     how PI / Canary / Aveva historians store data, NOT a wide column-per-sensor table.
#   * OPC-style quality codes (Good / Uncertain / Bad / Substituted).
#   * ~60 representative tags across 6 areas. Ranges/units grounded in brewing OT
#     references (ferment temp/jacket/head pressure/gravity, glycol supply/return/ΔT,
#     ISO-10816 vibration mm/s, canning fill volume mL & line speed CPM, CIP
#     conductivity, utilities).
#   * 5-minute analytics cadence. (Real historians keep 1-sec raw; downstream
#     analytics stores rollups — this is that rollup tier.)
#   * Value model = bounded RANDOM WALK around the band midpoint (the dominant
#     dynamic) + small diurnal (AZ ambient) component + Gaussian measurement noise,
#     clipped to physical bounds. The walk has light mean-reversion so it wanders
#     like a real random walk but stays physically plausible. Labeled anomaly
#     windows (spike/ramp/drift/stuck) are injected on top so Lab 2 and the served
#     anomaly model have ground truth and a clearly-separable signal to detect.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SITE = "Regional Plant 01"
ENTERPRISE = "Regional Brewery"
CADENCE_MIN = 5  # analytics cadence

# COMMAND ----------

# ---- dim_asset : ISA-95 equipment hierarchy --------------------------------
# (asset_id, asset_type, area, work_center)
_ASSETS = [
    # Brewhouse
    ("MT-01", "Mash Tun",        "Brewhouse",    "Brewhouse"),
    ("LT-01", "Lauter Tun",      "Brewhouse",    "Brewhouse"),
    ("BK-01", "Brew Kettle",     "Brewhouse",    "Brewhouse"),
    ("WP-01", "Whirlpool",       "Brewhouse",    "Brewhouse"),
    ("WC-01", "Wort Chiller",    "Brewhouse",    "Brewhouse"),
    ("BH-00", "Brewhouse Steam", "Brewhouse",    "Brewhouse"),
    # Fermentation cellar — 6 cylindroconical fermenters
    *[(f"FV-{i:03d}", "Cylindroconical Fermenter", "Fermentation", "Cellar") for i in range(1, 7)],
    # Glycol / refrigeration
    ("GLY-01", "Glycol Chiller", "Glycol", "Refrigeration"),
    ("GLY-02", "Glycol Chiller", "Glycol", "Refrigeration"),
    ("GP-01",  "Glycol Pump",    "Glycol", "Refrigeration"),
    # Canning line
    ("FILL-01", "Filler",       "Canning", "Packaging"),
    ("SEAM-01", "Seamer",       "Canning", "Packaging"),
    ("CONV-01", "Conveyor",     "Canning", "Packaging"),
    ("PAST-01", "Pasteurizer",  "Canning", "Packaging"),
    ("LBL-01",  "Labeler",      "Canning", "Packaging"),
    # CIP
    ("CIP-01", "CIP Skid", "CIP", "Cleaning"),
    # Utilities
    ("AC-01",  "Air Compressor", "Utilities", "Utilities"),
    ("BLR-01", "Boiler",         "Utilities", "Utilities"),
    ("CO2-01", "CO2 Recovery",   "Utilities", "Utilities"),
]

def asset_rows():
    return [dict(enterprise=ENTERPRISE, site=SITE, area=a[2], work_center=a[3],
                 asset_id=a[0], asset_type=a[1]) for a in _ASSETS]

# COMMAND ----------

# ---- dim_tag : ~60 tags. Fields drive the value model. ---------------------
# (asset_id, tag_id, metric, unit, lo, hi, warn, crit, noise_frac, diurnal_frac,
#  drift_frac, ambient)  where lo/hi = normal operating band; warn/crit = alarm
# thresholds; *_frac are fractions of the (hi-lo) band; ambient couples the tag
# to AZ outdoor temperature (cooling load).
def _T(asset, tag, metric, unit, lo, hi, warn, crit, noise=0.03, diurnal=0.0, drift=0.05, ambient=False, dtype="float"):
    return dict(asset_id=asset, tag_id=tag, metric=metric, unit=unit,
                normal_low=lo, normal_high=hi, warn_threshold=warn, crit_threshold=crit,
                noise_frac=noise, diurnal_frac=diurnal, drift_frac=drift, ambient=ambient, datatype=dtype)

_TAGS = [
    # --- Brewhouse (8) ---
    _T("MT-01", "BRW.MT01.TEMP_MASH",      "Mash Temperature",   "C",    62, 70, 71, 73, drift=0.04),
    _T("BK-01", "BRW.BK01.TEMP_KETTLE",    "Kettle Temperature", "C",    98, 101, 101.5, 103, noise=0.015),
    _T("BK-01", "BRW.BK01.PRESS",          "Kettle Pressure",    "kPa",  100, 130, 140, 160),
    _T("WC-01", "BRW.WC01.WORT_OUT_TEMP",  "Wort Outlet Temp",   "C",    8, 14, 16, 20, diurnal=0.12, ambient=True),
    _T("WC-01", "BRW.WC01.FLOW",           "Wort Flow",          "hL/h", 40, 70, 75, 85),
    _T("WP-01", "BRW.WP01.TEMP",           "Whirlpool Temp",     "C",    90, 96, 97, 99),
    _T("LT-01", "BRW.LT01.TURBIDITY",      "Lauter Turbidity",   "EBC",  0.5, 5, 8, 12, noise=0.08),
    _T("BH-00", "BRW.BH.STEAM_PRESS",      "Brewhouse Steam",    "bar",  2.0, 3.5, 3.8, 4.2),
    # --- Fermentation: 6 fermenters x 4 tags (24) ---
]
for i in range(1, 7):
    fv = f"FV-{i:03d}"
    _TAGS += [
        _T(fv, f"FMC.{fv}.TEMP_WORT",   "Fermentation Temp",   "C",   20, 28, 30, 33, noise=0.02, drift=0.06),
        _T(fv, f"FMC.{fv}.TEMP_JACKET", "Glycol Jacket Temp",  "C",   0, 6, 8, 12, noise=0.04, drift=0.05),
        _T(fv, f"FMC.{fv}.PRESS_HEAD",  "Head Pressure",       "bar", 0.5, 2.5, 2.8, 3.2, noise=0.03),
        _T(fv, f"FMC.{fv}.GRAVITY",     "Inline Gravity",      "SG",  1.005, 1.055, 1.002, 1.075, noise=0.015, drift=0.08),
    ]
_TAGS += [
    # --- Glycol / refrigeration (8) ---
    _T("GLY-01", "GLY.GLY01.SUPPLY_TEMP", "Glycol Supply Temp", "F",   24, 30, 32, 34, noise=0.02),
    _T("GLY-01", "GLY.GLY01.RETURN_TEMP", "Glycol Return Temp", "F",   30, 38, 42, 46, diurnal=0.10, ambient=True),
    _T("GLY-01", "GLY.GLY01.DELTA_T",     "Glycol ΔT",          "F",   4, 10, 12, 16),
    _T("GLY-01", "GLY.GLY01.COMP_AMPS",   "Compressor Amps",    "A",   80, 160, 180, 200, diurnal=0.15, ambient=True),
    _T("GLY-01", "GLY.GLY01.DISCH_PRESS", "Discharge Pressure", "psi", 180, 240, 260, 290, diurnal=0.10, ambient=True),
    _T("GLY-02", "GLY.GLY02.SUPPLY_TEMP", "Glycol Supply Temp", "F",   24, 30, 32, 34, noise=0.02),
    _T("GP-01",  "GLY.GP01.FLOW",         "Glycol Flow",        "gpm", 120, 200, 210, 230),
    _T("GP-01",  "GLY.GP01.VIBRATION",    "Pump Vibration RMS", "mm/s",0.3, 2.0, 2.8, 4.5, noise=0.06, drift=0.04),
    # --- Canning line (10) ---
    _T("FILL-01", "CAN.FILL01.FILL_VOLUME", "Fill Volume",      "mL",  352, 358, 350, 365, noise=0.01, drift=0.03),
    _T("FILL-01", "CAN.FILL01.FILL_PRESS",  "Fill Pressure",    "bar", 2.5, 3.5, 3.8, 4.2),
    _T("FILL-01", "CAN.FILL01.HEAD_TEMP",   "Fill Head Temp",   "C",   2, 6, 8, 10, ambient=True),
    _T("SEAM-01", "CAN.SEAM01.SEAM_HEIGHT", "Seam Height",      "mm",  2.9, 3.1, 2.7, 3.3, noise=0.02),
    _T("SEAM-01", "CAN.SEAM01.VIBRATION",   "Seamer Vibration RMS","mm/s",0.4, 2.0, 2.8, 4.5, noise=0.05, drift=0.03),
    _T("SEAM-01", "CAN.SEAM01.TORQUE",      "Seam Torque",      "Nm",  12, 20, 22, 26),
    _T("CONV-01", "CAN.CONV01.LINE_SPEED",  "Line Speed",       "cpm", 1600, 2000, 1500, 2100, noise=0.03),
    _T("PAST-01", "CAN.PAST01.PU",          "Pasteurization Units","PU",8, 18, 6, 22, noise=0.05),
    _T("PAST-01", "CAN.PAST01.ZONE_TEMP",   "Pasteurizer Zone Temp","C",58, 64, 66, 70),
    _T("LBL-01",  "CAN.LBL01.REJECT_RATE",  "Label Reject Rate","%",   0.1, 1.5, 2.5, 4.0, noise=0.10),
    # --- CIP (4) ---
    _T("CIP-01", "CIP.CIP01.CAUSTIC_COND", "Caustic Conductivity","mS/cm",40, 80, 90, 100, noise=0.04),
    _T("CIP-01", "CIP.CIP01.CAUSTIC_TEMP", "Caustic Temp",       "C",   70, 85, 88, 92),
    _T("CIP-01", "CIP.CIP01.ACID_COND",    "Acid Conductivity",  "mS/cm",10, 30, 35, 40, noise=0.04),
    _T("CIP-01", "CIP.CIP01.FLOW",         "CIP Flow",           "gpm", 80, 140, 150, 170),
    # --- Utilities (6) ---
    _T("AC-01",  "UTL.AC01.DISCH_PRESS",   "Air Discharge Press","bar", 6.5, 8.0, 8.5, 9.0),
    _T("AC-01",  "UTL.AC01.MOTOR_AMPS",    "Compressor Amps",    "A",   60, 110, 120, 135, diurnal=0.12, ambient=True),
    _T("BLR-01", "UTL.BLR01.STEAM_PRESS",  "Boiler Steam Press", "bar", 7, 9, 9.5, 10.5),
    _T("BLR-01", "UTL.BLR01.FEEDWATER_TEMP","Feedwater Temp",    "C",   80, 95, 98, 102),
    _T("CO2-01", "UTL.CO201.PURITY",       "CO2 Recovery Purity","%",   99.0, 99.9, 98.5, 99.95, noise=0.01),
    _T("CO2-01", "UTL.CO201.TANK_PRESS",   "CO2 Tank Pressure",  "bar", 14, 18, 19, 21),
]

def tag_rows():
    return [dict(t) for t in _TAGS]

TAG_BY_ID = {t["tag_id"]: t for t in _TAGS}
print(f"brewery_generator: {len(_ASSETS)} assets, {len(_TAGS)} tags defined.")

# COMMAND ----------

# ---- Labeled anomaly schedule ----------------------------------------------
# Each entry: dict(type, severity, start 'YYYY-MM-DD HH:MM', dur_hours,
#   effects={tag_id: (mode, magnitude)}). mode in {add, ramp, drift, set, mult}.
#   add   = +magnitude held across the window
#   ramp  = linear 0 -> magnitude across window then back to 0 (transient)
#   drift = linear 0 -> magnitude across window, NOT recovered (step left)
#   set   = freeze at magnitude (stuck-at sensor)
#   mult  = multiply band-noise by magnitude (e.g. brief instability)
ANOMALIES = [
    # Fermentation runaway: wort+jacket temp climb, gravity drops fast
    dict(type="fermentation_runaway", severity="critical", start="2024-07-18 02:00", dur_hours=30,
         effects={"FMC.FV-003.TEMP_WORT": ("ramp", 7.0), "FMC.FV-003.TEMP_JACKET": ("ramp", 6.0),
                  "FMC.FV-003.GRAVITY": ("drift", -0.030)}),
    dict(type="fermentation_runaway", severity="high", start="2025-03-09 14:00", dur_hours=24,
         effects={"FMC.FV-005.TEMP_WORT": ("ramp", 5.5), "FMC.FV-005.TEMP_JACKET": ("ramp", 5.0),
                  "FMC.FV-005.GRAVITY": ("drift", -0.025)}),
    dict(type="fermentation_runaway", severity="critical", start="2025-11-22 06:00", dur_hours=36,
         effects={"FMC.FV-002.TEMP_WORT": ("ramp", 8.0), "FMC.FV-002.TEMP_JACKET": ("ramp", 7.0),
                  "FMC.FV-002.GRAVITY": ("drift", -0.035)}),
    # Chiller fault: glycol return temp ramps up, compressor amps + discharge press follow
    dict(type="chiller_fault_drift", severity="high", start="2024-08-12 11:00", dur_hours=10,
         effects={"GLY.GLY01.RETURN_TEMP": ("ramp", 12.0), "GLY.GLY01.COMP_AMPS": ("ramp", 35.0),
                  "GLY.GLY01.DISCH_PRESS": ("ramp", 45.0)}),
    dict(type="chiller_fault_drift", severity="high", start="2025-06-27 13:00", dur_hours=8,
         effects={"GLY.GLY01.RETURN_TEMP": ("ramp", 10.0), "GLY.GLY01.COMP_AMPS": ("ramp", 30.0)}),
    dict(type="chiller_fault_drift", severity="critical", start="2026-05-19 12:00", dur_hours=14,
         effects={"GLY.GLY01.RETURN_TEMP": ("ramp", 15.0), "GLY.GLY01.COMP_AMPS": ("ramp", 42.0),
                  "GLY.GLY01.DISCH_PRESS": ("ramp", 55.0)}),
    # Fill-volume drift: slow underfill across a shift
    dict(type="fill_volume_drift", severity="medium", start="2024-05-03 22:00", dur_hours=8,
         effects={"CAN.FILL01.FILL_VOLUME": ("drift", -4.5)}),
    dict(type="fill_volume_drift", severity="medium", start="2024-12-14 06:00", dur_hours=8,
         effects={"CAN.FILL01.FILL_VOLUME": ("drift", -3.5)}),
    dict(type="fill_volume_drift", severity="high", start="2025-08-08 22:00", dur_hours=9,
         effects={"CAN.FILL01.FILL_VOLUME": ("drift", -5.5)}),
    dict(type="fill_volume_drift", severity="medium", start="2026-02-21 14:00", dur_hours=8,
         effects={"CAN.FILL01.FILL_VOLUME": ("drift", -3.8)}),
    # Bearing vibration ramp (multi-day) then maintenance reset
    dict(type="bearing_vibration_ramp", severity="high", start="2024-10-15 00:00", dur_hours=96,
         effects={"CAN.SEAM01.VIBRATION": ("drift", 2.6)}),
    dict(type="bearing_vibration_ramp", severity="critical", start="2025-09-02 00:00", dur_hours=120,
         effects={"CAN.SEAM01.VIBRATION": ("drift", 3.2)}),
    dict(type="bearing_vibration_ramp", severity="high", start="2025-04-11 00:00", dur_hours=72,
         effects={"GLY.GP01.VIBRATION": ("drift", 2.4)}),
    # CIP conductivity stuck-at (sensor fault)
    dict(type="cip_conductivity_stuck", severity="medium", start="2024-09-20 03:00", dur_hours=6,
         effects={"CIP.CIP01.CAUSTIC_COND": ("set", 62.0)}),
    dict(type="cip_conductivity_stuck", severity="medium", start="2025-07-14 04:00", dur_hours=5,
         effects={"CIP.CIP01.CAUSTIC_COND": ("set", 58.0)}),
    # Pasteurizer over-PU excursion
    dict(type="pasteurizer_excursion", severity="high", start="2025-01-30 09:00", dur_hours=6,
         effects={"CAN.PAST01.PU": ("ramp", 10.0), "CAN.PAST01.ZONE_TEMP": ("ramp", 8.0)}),
    # CO2 purity dip
    dict(type="co2_purity_dip", severity="medium", start="2025-10-05 16:00", dur_hours=7,
         effects={"UTL.CO201.PURITY": ("drift", -1.5)}),
]

def _parse(ts):
    return datetime.strptime(ts, "%Y-%m-%d %H:%M")

def anomaly_label_rows():
    """One label row per (affected tag, window) — ground truth for Lab 2 eval."""
    rows = []
    for a in ANOMALIES:
        start = _parse(a["start"]); end = start + timedelta(hours=a["dur_hours"])
        for tag_id in a["effects"]:
            rows.append(dict(asset_id=TAG_BY_ID[tag_id]["asset_id"], tag_id=tag_id,
                             start_ts=start, end_ts=end,
                             anomaly_type=a["type"], severity=a["severity"]))
    return rows

# COMMAND ----------

# ---- Value model -----------------------------------------------------------
def _az_ambient_f(dt_index):
    """Regional plant outdoor temp (deg F): annual + diurnal sinusoids."""
    doy = dt_index.dayofyear.to_numpy()
    hod = (dt_index.hour + dt_index.minute / 60.0).to_numpy()
    annual = 75 + 22 * np.cos(2 * np.pi * (doy - 200) / 365.0)   # hot summer
    diurnal = 12 * np.cos(2 * np.pi * (hod - 16) / 24.0)         # peak ~4pm
    return annual + diurnal

def _bounded_random_walk(n, step_sigma, mid, seed, reversion=0.015):
    """Bounded random walk at the analytics cadence: each step adds Gaussian noise
    to the previous value (a true random walk), with light mean-reversion toward
    `mid` so the series wanders but stays physically plausible instead of drifting
    to infinity. Stationary std ≈ step_sigma / sqrt(2*reversion). Vectorised-ish
    cumulative form would lose the reversion, so we loop — n is per-tag (≤ a few
    hundred k) and this runs in well under a second."""
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype="float64")
    x[0] = mid
    steps = rng.normal(0.0, step_sigma, n)
    keep = 1.0 - reversion
    for i in range(1, n):
        x[i] = keep * x[i - 1] + reversion * mid + steps[i]
    return x

def generate_tag_series(tag, dt_index, ambient_f=None, seed=None):
    """Return (values float64[], quality_code str[]) for one tag over dt_index
    (a pandas DatetimeIndex at CADENCE_MIN). Applies baseline + diurnal + drift +
    noise, clips to physical bounds, injects scheduled anomalies, sets quality."""
    n = len(dt_index)
    if seed is None:
        seed = abs(hash(tag["tag_id"])) % (2**31)
    rng = np.random.default_rng(seed)
    lo, hi = tag["normal_low"], tag["normal_high"]
    band = hi - lo
    mid = (lo + hi) / 2.0
    if ambient_f is None:
        ambient_f = _az_ambient_f(dt_index)

    # ---- random-walk baseline (dominant dynamic) + measurement noise ----
    # Each tag wanders as a bounded random walk around the middle of its normal
    # band; drift_frac scales the per-step size. Small iid noise on top is the
    # sensor's own measurement jitter. (Constants tuned so normal data mostly
    # stays in-band and injected anomalies remain clearly separable — verified on
    # the first real run.)
    step_sigma = band * tag["drift_frac"] * 0.5
    vals = _bounded_random_walk(n, step_sigma, mid, seed + 7, reversion=0.015)
    vals += rng.normal(0, band * tag["noise_frac"], n)

    # diurnal component
    if tag["diurnal_frac"] > 0:
        if tag["ambient"]:
            amb_norm = (ambient_f - ambient_f.mean()) / (ambient_f.std() + 1e-9)
            vals += amb_norm * band * tag["diurnal_frac"]
        else:
            hod = (dt_index.hour + dt_index.minute / 60.0).to_numpy()
            vals += np.cos(2 * np.pi * (hod - 16) / 24.0) * band * tag["diurnal_frac"]

    # ---- inject scheduled anomalies ----
    anomaly_mask = np.zeros(n, dtype=bool)
    t0 = dt_index[0].to_pydatetime()
    for a in ANOMALIES:
        eff = a["effects"].get(tag["tag_id"])
        if eff is None:
            continue
        start = _parse(a["start"]); end = start + timedelta(hours=a["dur_hours"])
        idx = np.where((dt_index >= start) & (dt_index < end))[0]
        if len(idx) == 0:
            continue
        anomaly_mask[idx] = True
        mode, mag = eff
        frac = np.linspace(0, 1, len(idx))
        if mode == "add":
            vals[idx] += mag
        elif mode == "ramp":          # 0 -> mag -> 0 (triangular transient)
            tri = np.where(frac < 0.5, frac * 2, (1 - frac) * 2)
            vals[idx] += mag * tri
        elif mode == "drift":         # 0 -> mag, held to end (step left behind)
            vals[idx] += mag * frac
        elif mode == "set":           # stuck-at
            vals[idx] = mag + rng.normal(0, band * 0.002, len(idx))
        elif mode == "mult":
            vals[idx] += rng.normal(0, band * tag["noise_frac"] * mag, len(idx))

    # physical clipping (allow anomalies to exceed the normal band but stay sane)
    phys_lo = lo - band * 1.5
    phys_hi = hi + band * 2.5
    vals = np.clip(vals, phys_lo, phys_hi)
    vals = np.round(vals, 4)

    # ---- OPC-style quality codes ----
    quality = np.array(["Good"] * n, dtype=object)
    r = rng.random(n)
    quality[r < 0.004] = "Uncertain"
    quality[r < 0.001] = "Bad"
    # during anomalies, a few readings get Substituted (sensor under stress)
    sub = anomaly_mask & (rng.random(n) < 0.05)
    quality[sub] = "Substituted"
    return vals, quality.astype(str)

def build_timestamp_index(start_dt, end_dt):
    """5-minute DatetimeIndex over [start_dt, end_dt)."""
    return pd.date_range(start=start_dt, end=end_dt, freq=f"{CADENCE_MIN}min", inclusive="left")

