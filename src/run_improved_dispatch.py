"""Original script of the dispatch-strategy test of Option A; it uses an older ROSIM factor, so it documents the
method, not the final numbers. Needs the raw data, not run by run_all.py."""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import csv
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SIM_START
from pipeline.data_pipeline import build_sim_df_tank
from post_midterm.pipeline.forecast_prices import load_forecast_prices
from post_midterm.models.simulation_ates_option_a_v7 import run_ates_option_a_v6
from post_midterm.scenarios.scenarios_v4 import SCENARIO_ATES_OPTION_A_V3

YEARS = 3.0
YEARS_LIST = [2023, 2024, 2025]
price_col = "Prijs_Gemiddelde_EPEX_Spot"

BASELINE = 1_073_985
ROSIM_F = 0.713
DEMAND_CORRECTION = 1.038
ENABLE_BEC_SUMMER_CAP = True
ENABLE_MIN_RUNTIME = True
MIN_RUNTIME_HOURS = 18
MIN_DOWNTIME_HOURS = 2
CHARGE_CAP_MW = 100.0     # UPDATED: symmetric 100 MW for S0-S5 (see CAPACITY NOTE)
DISCHARGE_CAP_MW = 100.0  # was 50/30 asymmetric -- corrected after S0 mismatch was found and diagnosed

CONFIRMED_S0 = 2_249_171          # 50 MW symmetric, run_ates_a_final.py -- historical reference only
CONFIRMED_S0_CORRECTED = 1_911_893

ATES_CAPEX_ASSUMPTION = 1_500_000
WACC = 0.05
LIFETIME = 30
ANNUITY = WACC / (1 - (1 + WACC) ** (-LIFETIME))

OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "analysis", "improved_dispatch_results.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report_figures", "ch6_ates")
OUT_PNG = os.path.join(FIG_DIR, "ch6_08_improved_dispatch.png")

TU_DELFT_BLUE = "#00A6D6"
DARK_BLUE = "#003D6E"
GREY = "#6D6E71"


def eur0(x):
    return f"{x:,.0f}".replace(",", ".")


def eur0_signed(x):
    return f"{x:+,.0f}".replace(",", ".")


def rosim_corrected(saving, f_corr=ROSIM_F):
    return BASELINE + f_corr * (saving - BASELINE)


# ═══════════════════════════════════════════════════════════════════════════
# Seasonal calendar builder (given verbatim)
# ═══════════════════════════════════════════════════════════════════════════
def build_seasonal_calendar(price_series, smooth_days=14):
    """
    For each day-of-year (1-365), compute the mean price
    using a ±smooth_days rolling window across all years.
    Returns dict: doy -> expected_price
    """
    df = pd.DataFrame({'price': price_series})
    df['doy'] = df.index.dayofyear

    calendar = {}
    for doy in range(1, 366):
        dist = abs((df['doy'] - doy + 182) % 365 - 182)
        mask = dist <= smooth_days
        calendar[doy] = df.loc[mask, 'price'].mean()

    cal_series = pd.Series(calendar).rolling(7, center=True, min_periods=1).mean()
    return cal_series.to_dict()


def get_trajectory(calendar, doy, lookahead_days):
    """Price trajectory: expected % change in next N days."""
    future_doy = (doy + lookahead_days - 1) % 365 + 1
    current = calendar[doy]
    future = calendar[future_doy]
    if current == 0:
        return 0.0
    return (future - current) / abs(current)


# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════
print("Loading data...")
(sim_df_tank, _temp_ds, df1_2025, _df3h, _df3q, _df_prod, _hist_cols, cost_hist) = build_sim_df_tank()
print(f"  Historische dervingskosten (werkelijke EPEX, excl. BEC, 3yr): {cost_hist:,.0f} EUR\n")

buffer_at_sim_start = (
    df1_2025["Bufferinhoud MWh"].loc[df1_2025.index >= SIM_START].dropna().iloc[0]
)
small_initial_soc = buffer_at_sim_start / 250.0

print("Loading forecast prices (validated loader, TZ-correct) ...")
forecast_hourly = load_forecast_prices()
sim_df_tank = sim_df_tank.join(
    forecast_hourly[["voorspelling_prijs"]].rename(columns={"voorspelling_prijs": "forecast_price"}),
    how="left",
)
sim_df_tank["forecast_price"] = sim_df_tank["forecast_price"].fillna(sim_df_tank[price_col])
print()

# ── S1 inputs: forecast ratio vs 30-day trailing mean ──────────────────────
TRAILING_MEAN_WINDOW_H = 30 * 24
sim_df_tank["trailing_mean_price"] = (
    sim_df_tank[price_col].rolling(window=TRAILING_MEAN_WINDOW_H, min_periods=24).mean()
)
sim_df_tank["trailing_mean_price"] = sim_df_tank["trailing_mean_price"].bfill()
sim_df_tank["forecast_ratio"] = np.where(
    sim_df_tank["trailing_mean_price"].abs() > 1e-6,
    sim_df_tank["forecast_price"] / sim_df_tank["trailing_mean_price"],
    1.0,
)

# ── S2 inputs: day-ahead 24h lookahead (actual EPEX) ────────────────────────
prices_arr = sim_df_tank[price_col]
sim_df_tank["next24h_min_price"] = prices_arr[::-1].rolling(window=24, min_periods=1).min()[::-1]
sim_df_tank["next24h_max_price"] = prices_arr[::-1].rolling(window=24, min_periods=1).max()[::-1]

# ── S3 inputs: seasonal calendar trajectory (causal, prior-year only) ──────
print("Building causal seasonal calendars (prior-year price data only) ...")
calendars_by_year = {}
for y in YEARS_LIST:
    source_year = y - 1
    if source_year not in prices_arr.index.year.unique():
        source_year = y
        print(f"  {y}: no prior-year data available, using {y}'s own data (flagged limitation).")
    else:
        print(f"  {y}: calendar built from {source_year} prices.")
    src_prices = sim_df_tank.loc[sim_df_tank.index.year == source_year, price_col]
    calendars_by_year[y] = build_seasonal_calendar(src_prices)
print()

traj_7d_vals = np.zeros(len(sim_df_tank))
traj_14d_vals = np.zeros(len(sim_df_tank))
doy_arr = sim_df_tank.index.dayofyear.values
year_arr = sim_df_tank.index.year.values
for idx in range(len(sim_df_tank)):
    cal = calendars_by_year[int(year_arr[idx])]
    doy = int(doy_arr[idx])
    if doy > 365:
        doy = 365
    traj_7d_vals[idx] = get_trajectory(cal, doy, 7)
    traj_14d_vals[idx] = get_trajectory(cal, doy, 14)
sim_df_tank["calendar_traj_7d"] = traj_7d_vals
sim_df_tank["calendar_traj_14d"] = traj_14d_vals

CORRECTIONS_KWARGS = dict(
    demand_correction_factor=DEMAND_CORRECTION,
    enable_bec_summer_cap=ENABLE_BEC_SUMMER_CAP,
    enable_min_runtime=ENABLE_MIN_RUNTIME,
    min_runtime_hours=MIN_RUNTIME_HOURS,
    min_downtime_hours=MIN_DOWNTIME_HOURS,
)
cfg_base = SCENARIO_ATES_OPTION_A_V3
import dataclasses
cfg_base = dataclasses.replace(cfg_base, ates_charge_max_mw=CHARGE_CAP_MW, ates_discharge_max_mw=DISCHARGE_CAP_MW)


def run_strategy(strategy_id, **extra_kwargs):
    df = run_ates_option_a_v6(
        sim_df_tank, cfg_base,
        small_initial_soc=small_initial_soc,
        dispatch_strategy=strategy_id,
        **CORRECTIONS_KWARGS,
        **extra_kwargs,
    )
    saving = (cost_hist - df["total_system_cost_eur_corrected"].sum()) / YEARS
    return df, saving


def per_year_metrics(df):
    out = {}
    for y in YEARS_LIST:
        sub = df[df.index.year == y]
        cost_hist_y = None  # not needed per-year here; use total_system_cost_eur_corrected directly
        charged = sub["ates_charge_mw"].sum()
        discharged = sub["ates_discharge_mw"].sum()
        util = discharged / charged * 100 if charged > 0 else 0.0
        out[y] = dict(charged=charged, discharged=discharged, util=util)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# S0 — reference point at 100 MW (NOT a regression test against the
# historical 50 MW confirmed baseline -- see CAPACITY NOTE below)
# ═══════════════════════════════════════════════════════════════════════════
# The S0 hook reproduces the original 50/50 MW result (EUR 2.249.171/yr to within EUR 0.30).
# This experiment raises both caps to 100 MW, so S0 here differs from that 50 MW figure.
print("Running S0 at 100 MW (reference point for this experiment) ...")
df_s0, saving_s0 = run_strategy("S0", annual_charge_pct=25, annual_discharge_pct=75)
print(f"  Done. Saving = €{eur0(saving_s0)}/yr\n")

print("CAPACITY NOTE:")
print(f"  Confirmed baseline (50 MW, run_ates_a_final.py):")
print(f"    Python €{eur0(CONFIRMED_S0)}/yr → ROSIM-corr €{eur0(CONFIRMED_S0_CORRECTED)}/yr")
print(f"  This experiment uses 100 MW caps throughout.")
print(f"  S0 at 100 MW is a new reference point, not the historical baseline.")
print(f"  All S1-S5 improvements are reported relative to S0@100MW.")
print(f"  S0@100MW result: Python €{eur0(saving_s0)}/yr → ROSIM-corr €{eur0(rosim_corrected(saving_s0))}/yr")
print()

# ═══════════════════════════════════════════════════════════════════════════
# S1-S4
# ═══════════════════════════════════════════════════════════════════════════
print("Running S1 (forecast directional signal) ...")
df_s1, saving_s1 = run_strategy("S1")
print(f"  Done. Saving = €{eur0(saving_s1)}/yr\n")

print("Running S2 (day-ahead 24h lookahead) ...")
df_s2, saving_s2 = run_strategy("S2", annual_charge_pct=25, annual_discharge_pct=75)
print(f"  Done. Saving = €{eur0(saving_s2)}/yr\n")

print("Running S3 (seasonal calendar trajectory) ...")
df_s3, saving_s3 = run_strategy("S3")
print(f"  Done. Saving = €{eur0(saving_s3)}/yr\n")

print("Running S4 (combined S1+S2+S3) ...")
df_s4, saving_s4 = run_strategy("S4", annual_charge_pct=25, annual_discharge_pct=75)
print(f"  Done. Saving = €{eur0(saving_s4)}/yr\n")


# ═══════════════════════════════════════════════════════════════════════════
# S5 — Oracle
# ═══════════════════════════════════════════════════════════════════════════
ATES_CAPACITY_MWH = SCENARIO_ATES_OPTION_A_V3.ates_capacity_m3 * (110.0 - 58.0) * 0.001157
ETA = 0.95
CHARGE_PCT_GRID = [5, 10, 15, 20, 25, 30, 35, 40]
DISCHARGE_PCT_GRID = [60, 65, 70, 75, 80, 85, 90, 95]


def run_oracle_quick(prices, charge_pct, discharge_pct, capacity_mwh, charge_cap_mw, discharge_cap_mw, eta):
    """Fast greedy SOC-tracked price-arbitrage simulator (no merit order/BEC/demand
    coupling) -- used ONLY to search for the best-performing percentile pair;
    the winning pair is then run through the real dispatch engine for S5's
    reported saving_eur_yr (see module docstring)."""
    thresh_c = np.percentile(prices, charge_pct)
    thresh_d = np.percentile(prices, discharge_pct)
    soc = 0.5 * capacity_mwh
    profit = 0.0
    charged_mwh = 0.0
    discharged_mwh = 0.0
    for p in prices:
        if p <= thresh_c and soc < capacity_mwh:
            chg = min(charge_cap_mw, capacity_mwh - soc)
            soc += chg
            profit -= chg * p
            charged_mwh += chg
        elif p >= thresh_d and soc > 0:
            dis = min(discharge_cap_mw, soc)
            soc -= dis
            profit += dis * eta * p
            discharged_mwh += dis
    return profit, charged_mwh, discharged_mwh


def search_best_pct(prices):
    best = None
    for cp in CHARGE_PCT_GRID:
        for dp in DISCHARGE_PCT_GRID:
            profit, chg, dis = run_oracle_quick(prices, cp, dp, ATES_CAPACITY_MWH, CHARGE_CAP_MW, DISCHARGE_CAP_MW, ETA)
            if best is None or profit > best[0]:
                best = (profit, cp, dp, chg, dis)
    return best


print("Running S5 oracle grid search (fast arbitrage simulator) ...")
all_prices = sim_df_tank[price_col].values
overall_best = search_best_pct(all_prices)
oracle_charge_pct_overall, oracle_discharge_pct_overall = overall_best[1], overall_best[2]
print(f"  Overall best: charge<=P{oracle_charge_pct_overall}, discharge>=P{oracle_discharge_pct_overall} "
      f"(arbitrage profit EUR {overall_best[0]:,.0f} over 3yr)\n".replace(",", "."))

oracle_per_year = {}
for y in YEARS_LIST:
    yr_prices = sim_df_tank.loc[sim_df_tank.index.year == y, price_col].values
    best_y = search_best_pct(yr_prices)
    oracle_per_year[y] = dict(charge_pct=best_y[1], discharge_pct=best_y[2], profit=best_y[0])

print("Running S5 (oracle-derived percentiles through real dispatch engine) ...")
df_s5, saving_s5 = run_strategy(
    "S0", annual_charge_pct=oracle_charge_pct_overall, annual_discharge_pct=oracle_discharge_pct_overall,
)
print(f"  Done. Saving = €{eur0(saving_s5)}/yr\n")

# ═══════════════════════════════════════════════════════════════════════════
# Assemble strategy results
# ═══════════════════════════════════════════════════════════════════════════
strategies = [
    ("S0", "S0 Current (365d P25/75)", df_s0, saving_s0),
    ("S1", "S1 Forecast signal", df_s1, saving_s1),
    ("S2", "S2 Day-ahead 24h", df_s2, saving_s2),
    ("S3", "S3 Seasonal calendar", df_s3, saving_s3),
    ("S4", "S4 Combined (S1+S2+S3)", df_s4, saving_s4),
    ("S5", "S5 Oracle (upper bound)", df_s5, saving_s5),
]

strategy_saving = {sid: sv for sid, _, _, sv in strategies}
strategy_corrected = {sid: rosim_corrected(sv) for sid, _, _, sv in strategies}
strategy_df = {sid: df for sid, _, df, _ in strategies}

# ═══════════════════════════════════════════════════════════════════════════
# Print
# ═══════════════════════════════════════════════════════════════════════════
print("IMPROVED DISPATCH ANALYSIS — OPTION A HT-ATES")
print("Baseline: €1.073.985/yr dispatch-only")
print(f"ROSIM correction: f = {ROSIM_F} (applied to incremental only)")
print("═" * 81)
print()

print("STRATEGY COMPARISON:")
print("─" * 81)
print(f"{'Strategy':<24}{'Python (€/yr)':>15}{'ROSIM-corr (€/yr)':>19}{'vs S0':>13}{'vs S0%':>10}")
for sid, label, df, sv in strategies:
    corr = strategy_corrected[sid]
    if sid == "S0":
        vs0_str, vs0pct_str = "---", "---"
    else:
        vs0 = corr - strategy_corrected["S0"]
        vs0pct = vs0 / strategy_corrected["S0"] * 100
        vs0_str, vs0pct_str = eur0_signed(vs0), f"{vs0pct:+.1f}%"
    print(f"{label:<24}{eur0(sv):>15}{eur0(corr):>19}{vs0_str:>13}{vs0pct_str:>10}")
print("─" * 81)
s4_pct_of_oracle = (strategy_corrected["S4"] - BASELINE) / (strategy_corrected["S5"] - BASELINE) * 100 if strategy_corrected["S5"] != BASELINE else float('nan')
print(f"S4 achieves {s4_pct_of_oracle:.1f}% of theoretical oracle optimum")
print()

# ── Seasonal calendar insights ──────────────────────────────────────────────
overall_cal = build_seasonal_calendar(sim_df_tank[price_col])
cal_series_full = pd.Series(overall_cal).sort_index()
expensive_doy = cal_series_full.idxmax()
cheap_doy = cal_series_full.idxmin()
traj_series = pd.Series({d: get_trajectory(overall_cal, d, 14) for d in range(1, 366)})
strongest_doy = traj_series.idxmax()

print("SEASONAL CALENDAR INSIGHTS:")
print(f"Most expensive seasonal period: around day {expensive_doy} (avg €{cal_series_full.max():.1f}/MWh)")
print(f"Cheapest seasonal period:       around day {cheap_doy} (avg €{cal_series_full.min():.1f}/MWh)")
print(f"Strongest upward trajectory:    starting day {strongest_doy}, {traj_series.max()*100:+.1f}% in 14 days")
precharge_start = max(1, strongest_doy - 14)
print(f"  → Optimal pre-charge window:  day {precharge_start} to day {strongest_doy}")

s0_charge_hours_in_window = ((df_s0.index.dayofyear >= precharge_start) & (df_s0.index.dayofyear <= strongest_doy) & (df_s0["ates_charge_mw"] > 0.01)).sum()
s0_hours_in_window = ((df_s0.index.dayofyear >= precharge_start) & (df_s0.index.dayofyear <= strongest_doy)).sum()
s3_charge_hours_in_window = ((df_s3.index.dayofyear >= precharge_start) & (df_s3.index.dayofyear <= strongest_doy) & (df_s3["ates_charge_mw"] > 0.01)).sum()
pct_s0 = s0_charge_hours_in_window / s0_hours_in_window * 100 if s0_hours_in_window > 0 else 0.0
pct_s3 = s3_charge_hours_in_window / s0_hours_in_window * 100 if s0_hours_in_window > 0 else 0.0
print(f"Currently charged in window:    {pct_s0:.1f}% of available hours (S0)")
print(f"With S3 improvement:            {pct_s3:.1f}% of available hours")
print()

# ── Forecast signal analysis ────────────────────────────────────────────────
urgent_mask = sim_df_tank["forecast_ratio"] > 1.25
defer_mask = sim_df_tank["forecast_ratio"] < 0.80
hours_urgent = urgent_mask.sum() / YEARS
hours_defer = defer_mask.sum() / YEARS
extra_charge_s1 = (df_s1["ates_charge_mw"].sum() - df_s0["ates_charge_mw"].sum()) / YEARS
saving_from_defer = (strategy_saving["S1"] - strategy_saving["S0"])

print("FORECAST SIGNAL ANALYSIS:")
print(f"Hours where forecast ratio > 1.25 (urgent):  {hours_urgent:,.0f} hrs/yr".replace(",", "."))
print(f"Additional charge captured vs S0:            {extra_charge_s1:+,.0f} MWh/yr".replace(",", "."))
print(f"Hours where forecast ratio < 0.80 (defer):   {hours_defer:,.0f} hrs/yr".replace(",", "."))
print(f"Savings from correctly deferring:            {eur0_signed(saving_from_defer)}/yr")
print()

# ── Day-ahead improvement breakdown ─────────────────────────────────────────
s2_charge_bool = df_s2["ates_charge_mw"] > 0.01
s0_charge_bool = df_s0["ates_charge_mw"] > 0.01
s2_earlier = ((s2_charge_bool) & (~s0_charge_bool)).sum() / YEARS
extra_mwh_s2 = (df_s2["ates_charge_mw"].sum() - df_s0["ates_charge_mw"].sum()) / YEARS
s2_disch_bool = df_s2["ates_discharge_mw"] > 0.01
s0_disch_bool = df_s0["ates_discharge_mw"] > 0.01
deferred_disch_hours = ((~s2_disch_bool) & (s0_disch_bool)).sum() / YEARS
saving_from_defer_disch = strategy_saving["S2"] - strategy_saving["S0"]
s2_disch_active = df_s2["ates_discharge_mw"] > 0.01
n_disch_hours_s2 = int(s2_disch_active.sum())
if "next24h_max_price" in df_s2.columns and n_disch_hours_s2 > 0:
    disch_at_peak = s2_disch_active & (df_s2[price_col] >= df_s2["next24h_max_price"] * 0.95)
    pct_disch_at_peak = disch_at_peak.sum() / n_disch_hours_s2 * 100
else:
    pct_disch_at_peak = float('nan')

print("DAY-AHEAD IMPROVEMENT BREAKDOWN:")
print(f"Hours where S2 charged earlier than S0:      {s2_earlier:,.0f} hrs/yr → {extra_mwh_s2:+,.0f} MWh charged".replace(",", "."))
print(f"Hours where S2 deferred discharge correctly: {deferred_disch_hours:,.0f} hrs/yr → {eur0_signed(saving_from_defer_disch)}/yr".replace(",", "."))
print(f"Hours where S2 discharged at daily peak:      {pct_disch_at_peak:.1f}% of discharge hours")
print()

# ── Oracle-derived optimal thresholds ───────────────────────────────────────
print("ORACLE-DERIVED OPTIMAL THRESHOLDS:")
print(f"  Empirically optimal charge up to:    P{oracle_charge_pct_overall} (current model uses P25)")
print(f"  Empirically optimal discharge above: P{oracle_discharge_pct_overall} (current model uses P75)")
print("  Per year:")
for y in YEARS_LIST:
    o = oracle_per_year[y]
    print(f"    {y}: charge P{o['charge_pct']}, discharge P{o['discharge_pct']}")
print(f"  → Recommendation: update fixed gates to P{oracle_charge_pct_overall}/P{oracle_discharge_pct_overall}")
print()

# ── Per year — best strategy (S4) ───────────────────────────────────────────
print("PER YEAR — BEST STRATEGY (S4):")
for y in YEARS_LIST:
    s4_y_cost = df_s4.loc[df_s4.index.year == y, "total_system_cost_eur_corrected"].sum()
    s0_y_cost = df_s0.loc[df_s0.index.year == y, "total_system_cost_eur_corrected"].sum()
    hist_cost_y = None
    # per-year saving needs a per-year hist_cost reference; approximate via
    # proportion of 3yr cost_hist is not exact -- report the COST DELTA
    # (S0 vs S4) per year instead, which is exact and avoids needing a
    # per-year hist_cost split.
    delta = s0_y_cost - s4_y_cost
    print(f"  {y}: €{eur0(delta)}/yr cost reduction vs S0-{y} (S4 vs S0, same year)")
print()

# ═══════════════════════════════════════════════════════════════════════════
# Financial viability
# ═══════════════════════════════════════════════════════════════════════════
best_id = max(strategy_corrected, key=strategy_corrected.get)
best_corrected = strategy_corrected[best_id]
s0_100mw_corrected = strategy_corrected["S0"]
additional_saving = best_corrected - s0_100mw_corrected

ann_capex = ATES_CAPEX_ASSUMPTION * ANNUITY
current_payback = ATES_CAPEX_ASSUMPTION / (s0_100mw_corrected - BASELINE) if (s0_100mw_corrected - BASELINE) > 0 else float('nan')
improved_payback = ATES_CAPEX_ASSUMPTION / (best_corrected - BASELINE) if (best_corrected - BASELINE) > 0 else float('nan')

print("FINANCIAL VIABILITY IMPACT:")
print(f"  Current S0@100MW ROSIM-corrected: €{eur0(s0_100mw_corrected)}/yr")
print(f"  (historical 50MW reference:       €{eur0(CONFIRMED_S0_CORRECTED)}/yr)")
print(f"  Best improved strategy ({best_id}):  €{eur0(best_corrected)}/yr")
print(f"  Additional annual saving:     {eur0_signed(additional_saving)}/yr")
print("  Impact on payback period:")
print(f"    ATES CAPEX assumption:      €{eur0(ATES_CAPEX_ASSUMPTION)}")
print(f"    Current payback:            {current_payback:.2f} yr")
print(f"    Improved payback:           {improved_payback:.2f} yr")
print("═" * 81)

# ═══════════════════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════════════════
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
csv_rows = []
for sid, label, df, sv in strategies:
    corr = strategy_corrected[sid]
    per_year = per_year_metrics(df)
    for y in YEARS_LIST:
        m = per_year[y]
        # per-year python_saving/rosim_corrected: using the 3-year figures
        # as a representative value is not meaningful per year without a
        # year-specific hist_cost reference; instead we report
        # the year-specific total_system_cost_eur_corrected directly.
        year_cost = df.loc[df.index.year == y, "total_system_cost_eur_corrected"].sum()
        csv_rows.append(dict(
            strategy=sid, year=y,
            python_saving=round(sv, 2), rosim_corrected=round(corr, 2),
            incremental_vs_s0=round(corr - strategy_corrected["S0"], 2) if sid != "S0" else 0.0,
            charged_mwh=round(m['charged'], 2), discharged_mwh=round(m['discharged'], 2),
            utilisation_pct=round(m['util'], 2),
        ))
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["strategy", "year", "python_saving", "rosim_corrected",
                                            "incremental_vs_s0", "charged_mwh", "discharged_mwh", "utilisation_pct"])
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)
print(f"\nOpgeslagen: {os.path.abspath(OUT_CSV)}")

# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════
os.makedirs(FIG_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams["font.family"] = "DejaVu Sans"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

labels_short = ["S0", "S1", "S2", "S3", "S4", "S5"]
colors_bar = [GREY, "#8FD9EE", "#5CC3E8", TU_DELFT_BLUE, DARK_BLUE, "#001F38"]
vals = [strategy_corrected[sid] for sid in ["S0", "S1", "S2", "S3", "S4", "S5"]]
bars = ax1.bar(labels_short[:-1], vals[:-1], color=colors_bar[:-1])
ax1.axhline(vals[-1], color=colors_bar[-1], linestyle='--', linewidth=1.5, label='S5 Oracle (upper bound)')
ax1.text(4.5, vals[-1], f" S4 achieves {s4_pct_of_oracle:.0f}% of oracle", va='bottom', ha='right', fontsize=8, color=GREY)
ax1.set_ylabel('ROSIM-corrected annual saving (€/yr)')
ax1.set_title('Strategy comparison', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='lower right')

doy_x = list(range(1, 366))
price_y = [overall_cal[d] for d in doy_x]
ax2.plot(doy_x, price_y, color='black', linewidth=1.2)
ax2.axvspan(precharge_start, strongest_doy, color='green', alpha=0.15, label='Charging season')
ax2.axvspan(max(1, expensive_doy - 20), min(365, expensive_doy + 20), color='red', alpha=0.12, label='Discharging season')
ax2.axvline(precharge_start, color='green', linestyle=':', linewidth=1)
ax2.axvline(strongest_doy, color='green', linestyle=':', linewidth=1)
ax2.set_xlabel('Day of year')
ax2.set_ylabel('Mean price (€/MWh)')
ax2.set_title('Seasonal price pattern — basis for calendar strategy', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"✓ Figuur opgeslagen: {os.path.abspath(OUT_PNG)}")
