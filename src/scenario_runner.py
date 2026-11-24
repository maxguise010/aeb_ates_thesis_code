#!/usr/bin/env python3
"""Decomposition of the annual saving (S_ref, S_flex, S_stor) and NPV per scenario and storage option, with the
adapter to the dispatch models. Run by run_all.py from the cache; new runs need the raw data."""
from __future__ import annotations
import argparse, json, hashlib, re, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================================
# 1. SETTINGS
# =============================================================================================
import config as C  # repository paths (config.py)
OUT_DIR   = C.RES_SCENARIO_RUNNER
CACHE_DIR = C.SCENARIO_CACHE

PRICE_COLS  = ["Prijs_Gemiddelde_EPEX_Spot", "EPEX_price", "forecast_price"]   # transformed if present
# The thesis frame (build_sim_df_tank) only contains Prijs_Gemiddelde_EPEX_Spot. The forecast price
# (voorspelling_prijs) is loaded from file inside run_tank_simulation_v5 and is NOT transformed; the TTES
# option therefore uses the actual-EPEX threshold variant (see adapter), which only reads the frame price.
DEMAND_COLS = ["heat_demand_mw", "heat_demand_corrected_mw",                    # scaled if present
               "demand_westpoort_mw", "demand_noord_mw", "demand_nieuwwest_mw"]  # loop demands (Option B)
SCALE_HISTORY = True          # also scale the historical per-unit columns in demand scenarios
# Historical unit outputs in the thesis frame are WW_BEC, WW_10, WW_30, WW_01, WW_00, WW_02 (config.
# historical_cols) plus their *_hist copies. The original regex (WW..._mw) matched none of them; unit
# capacities (WW10_max_mw, WW30_aftap_cap_mw, ...) must NOT be scaled.
HISTORY_COLS_REGEX = r"^(WW_(BEC|10|30|01|00|02)(_hist)?|total_supply_mw|unmet_demand_mw)$"
# The potential tier uses S_ref, which is measured against the historical operation. It is only valid
# when the adapter rebuilds that benchmark from the scenario frame (hist_cost_per_year). Set True when done.
HISTORY_REBUILT = True        # adapter rebuilds the benchmark from the frame (hist_cost_per_year);
                              # equals the stored build_sim_df_tank value to 0.00 EUR at the base scenario

import final_config as FC; import tank_capex as TC  # single source of rate, lifetime and tank CAPEX (src/final_config.py)
# annuity factor from final_config.py (5 %, 30 years = 15.3725).
F_PV = FC.annuity_factor()    # 5 % and 30 years, as in Equation 7.2
PHI  = 1.0                    # share of S_ref credited in the potential tier
REFERENCE_FAMILY = "ates"     # model family that supplies S_ref and Z in the decomposition

# name: family, CAPEX (low, mid, high) in M EUR, OPEX in k EUR/yr, f_corr
OPTIONS = {
    "TTES5k": dict(family="tank", capex=TC.CASE_B_TUPLE, opex=TC.opex_keur(TC.CASE_B_TUPLE[1]), f_corr=1.000),  # case B, OPEX 1.4 %
    # Option A has its own zero-capacity reference (the Option A model has no standby logic, so its
    # constrained zero-capacity run is 1,697.2, not S_ref); agreed by the author when the scenarios were set up.
    "A1D":    dict(family="ates_a", capex=(8.5, 10.5, 14.0),  opex=175.0, f_corr=0.567),
    "A2D":    dict(family="ates_a", capex=(13.5, 17.0, 22.5), opex=255.0, f_corr=0.560),
    "A4D":    dict(family="ates_a", capex=(23.5, 30.0, 40.0), opex=495.0, f_corr=0.539),
    "BWN110": dict(family="ates", capex=(8.5, 10.5, 14.0),  opex=175.0, f_corr=0.665),
    # B-std: locked f_corr (basis R) of the runs at the configured return temperature (final_numbers_locked.md)
    "Bstd":   dict(family="ates", capex=(17.7, 22.2, 30.2), opex=378.0,
                 f_corr={58.0: 0.617, 70.0: 0.490}[C.BSTD_T_RETURN_C]),  # OPEX as thesis Table 7.4
}
LABELS = {"TTES5k": "TTES 5k", "A1D": "Option A 1D", "A2D": "Option A 2D", "A4D": "Option A 4D",
          "BWN110": r"B-WN 110\textdegree C", "Bstd": "B standard"}

# Regression gate: the base scenario must reproduce the thesis (k EUR/yr)
EXPECTED = dict(
    S_ref=1074.0, Z=1849.7,
    S_stor={"TTES5k": 51.0, "A1D": 163.2, "A2D": 307.1, "A4D": 463.7, "BWN110": 128.9,
            "Bstd": {58.0: 116.8, 70.0: 118.3}[C.BSTD_T_RETURN_C]},
    gain_lower={"TTES5k": 34.0, "A1D": 73.9, "A2D": 138.2, "A4D": 236.6, "BWN110": 77.4,
                "Bstd": {58.0: 66.8, 70.0: 52.8}[C.BSTD_T_RETURN_C]},
)
TOL_MAIN, TOL_ROUNDED = 1.0, 1.5     # k EUR/yr

SCENARIOS = {
    "base":       dict(demand=1.00, level=1.00, spread=1.00),
    "demand_m20": dict(demand=0.80, level=1.00, spread=1.00),
    "demand_p20": dict(demand=1.20, level=1.00, spread=1.00),
    "spread_075": dict(demand=1.00, level=1.00, spread=0.75),
    "spread_125": dict(demand=1.00, level=1.00, spread=1.25),
    "spread_150": dict(demand=1.00, level=1.00, spread=1.50),
    "level_080":  dict(demand=1.00, level=0.80, spread=1.00),
    "level_120":  dict(demand=1.00, level=1.20, spread=1.00),
}
DEFAULT_SCENARIOS = ["base", "demand_m20", "demand_p20", "spread_075", "spread_125", "spread_150"]
SCEN_LABEL = {"base": "Base", "demand_m20": r"Demand $-$20\%", "demand_p20": r"Demand $+$20\%",
              "spread_075": r"Spread $\times$0.75", "spread_125": r"Spread $\times$1.25",
              "spread_150": r"Spread $\times$1.50", "level_080": r"Level $\times$0.8",
              "level_120": r"Level $\times$1.2", "price_file": "Price file"}

# =============================================================================================
# 2. ADAPTER  (the only part that touches your model code)
# =============================================================================================
# Code that produced the thesis numbers (all relaxed/constrained runs of Chapter 7):
#   tank (TTES 5k, none_tank): simulation_v7.run_tank_simulation_v5, patched in memory with a large_price_col
#       argument so that the large-tank thresholds use the actual EPEX price; thermal replay N=20, H/D 0.6,
#       16 sub-steps (ttes_thermal_replay.py) with the shortfall penalty. Values: data/ttes_price_input_test(_relaxed).csv.
#   Option A: simulation_ates_option_a_v7.run_ates_option_a_v6 + SOC+25pp->P45 patch, built by the set-up
#       part of generate_rosim_1and2doublet_schedules.py.
#   Option B: simulation_ates_three_loops_v7 -- WN110 with the patched SOC function of
#       generate_option_b_wn_110c_schedule.py, standard with run_ates_three_loops_v6.
_AD: dict = {}


def _adapter() -> dict:
    """Build (once per process) the patched dispatch functions exactly as in the thesis scripts."""
    if _AD:
        return _AD
    import dataclasses, inspect, os
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    runs = root          # flat src/: the two schedule scripts are next to this module

    def prefix(script, marker):          # execute a script's set-up part only (up to its data loading)
        src = (runs / script).read_text(encoding="utf-8")
        assert src.count(marker) == 1, f"{script}: marker found {src.count(marker)}x"
        ns = {"__file__": str(runs / script), "__name__": f"prefix_{script}"}
        exec(compile(src[:src.index(marker)], f"<prefix {script}>", "exec"), ns)
        return ns

    A = prefix("generate_rosim_1and2doublet_schedules.py", 'print("\\nLoading data...")')
    B = prefix("generate_option_b_wn_110c_schedule.py", 'print("OPTION B WN')
    import simulation_v7 as v7
    src = inspect.getsource(v7.run_tank_simulation_v5)
    for old, new in [
        ('''    demand_correction_factor: float = 1.0,  # default = no correction; 1.025 = +2.5%
):''', '''    demand_correction_factor: float = 1.0,  # default = no correction; 1.025 = +2.5%
    large_price_col: str = 'voorspelling_prijs',
):'''),
        ("""    dispatch_price_col = 'voorspelling_prijs'   # for planning""",
         """    dispatch_price_col = large_price_col   # for planning (patched: selectable)""")]:
        assert src.count(old) == 1
        src = src.replace(old, new, 1)
    src = src.replace("def run_tank_simulation_v5(", "def run_tank_simulation_pc(", 1)
    ns = dict(vars(v7))
    exec(compile(src, "<scenario_runner tank patch>", "exec"), ns)
    import simulation_ates_three_loops_v7 as b_mod
    from ttes_thermal_replay import ttes_model, replay
    from scenarios_v4 import (SCENARIO_ATES_OPTION_A_V3, SCENARIO_ATES_B_WESTPOORT,
                                                     SCENARIO_ATES_B_NOORD, SCENARIO_ATES_B_NIEUWWEST)
    from config import derivation_factors
    _AD.update(A=A, B=B, tank=ns["run_tank_simulation_pc"], b_v6=b_mod.run_ates_three_loops_v6,
               ttes_model=ttes_model, replay=replay, dc=dataclasses, f=derivation_factors,
               SA=SCENARIO_ATES_OPTION_A_V3, SWP=SCENARIO_ATES_B_WESTPOORT, SN=SCENARIO_ATES_B_NOORD,
               SNW=SCENARIO_ATES_B_NIEUWWEST)
    return _AD


PRICE = "Prijs_Gemiddelde_EPEX_Spot"


def load_sim_df() -> pd.DataFrame:
    """Return the hourly input frame EXACTLY as used for the thesis runs (the output of
    build_sim_df_tank(...) or the equivalent), indexed by timestamp, 2023-2025."""
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from config import SIM_START
    from data_pipeline import build_sim_df_tank
    out = build_sim_df_tank()
    df = out[0]
    # Initial buffer SOC used by every thesis tank run (not a column, carried in attrs; copies keep it)
    df.attrs["small_initial_soc"] = float(out[2]["Bufferinhoud MWh"].loc[out[2].index >= SIM_START]
                                          .dropna().iloc[0] / 250.0)
    df.attrs["stored_hist_cost_eur"] = float(out[-1])     # benchmark as stored at the original prices
    return df


def hist_cost_per_year(df: pd.DataFrame) -> pd.Series:
    """Historical benchmark rebuilt from the (scenario) frame: historical heat output of each unit x
    derivation factor x price, per year (EUR). Same formula as build_sim_df_tank / the thesis scripts.
    BEC is excluded, as in the thesis (extra BEC above history enters the model cost as revenue)."""
    f = _adapter()["f"]
    ww30 = df["WW_30"].fillna(0.0)
    aftap = ww30.clip(upper=df["WW30_aftap_cap_mw"])
    loss = (df["WW_10"].fillna(0.0) * f["WW10"] + aftap * f["WW30_aftap"]
            + (ww30 - aftap).clip(lower=0.0) * f["WW30_reductie"]
            + df["WW_01"].fillna(0.0) * f["WW01"] + df["WW_00"].fillna(0.0) * f["WW00"]
            + df["WW_02"].fillna(0.0) * f["WW02"])
    return (loss * df[PRICE]).groupby(df.index.year).sum()


def run_saving(sim_df: pd.DataFrame, option: str, schedule: str, buffer_kw: dict | None = None,
               keep_frame: bool = False) -> dict:
    """Run one case and return
         {"saving_keur_yr": float,   mean annual saving 2023-2025 in k EUR/yr, same definition as the thesis
          "per_year": {2023: ..., 2024: ..., 2025: ...},     k EUR
          "unmet_mwh": float}        unmet heat demand over the whole period
    option   : "none_tank", "none_ates", "none_ates_a" (zero capacity) or a key of OPTIONS
    schedule : "constrained" (6 h minimum downtime plus standby rule) or "relaxed" (2 h, no standby rule)
    The frame sim_df is already transformed for the scenario, and must not be modified in place.
    buffer_kw : optional overrides of the existing-buffer parameters (n_small_tanks, small_soc_min,
                small_soc_max, daily_charge_rank_n, daily_discharge_rank_n, reserve_horizon_h,
                reserve_max_fraction). Only the tank and Option A models expose them; None = thesis.
    keep_frame: also return the model output frame under "frame" (not cached; diagnostics only)."""
    ad = _adapter()
    dc = ad["dc"]
    bkw = dict(buffer_kw or {})
    # Arguments of every model, incl. three-loop: existing_buffer_rule ("rank" default / "aeb_demand"),
    # small_soc_max, reserve_horizon_h, reserve_max_fraction, daily charge/discharge rank n.
    # The three-loop model still hardcodes n_small_tanks, the tank volume, small_soc_min and the initial content.
    rule_kw = {k: bkw.pop(k) for k in ("existing_buffer_rule", "small_soc_max", "reserve_horizon_h",
                                       "reserve_max_fraction", "daily_charge_rank_n", "daily_discharge_rank_n",
                                       "buffer_release_rule", "release_reserve_horizon_h",
                                       "release_reserve_max_fraction", "release_power_horizon_h")
               if k in bkw}
    if bkw and option not in ("none_tank", "TTES5k", "none_ates_a", "A1D", "A2D", "A4D"):
        raise ValueError(f"{option}: the three-loop model hardcodes {sorted(bkw)}; not supported")
    constrained = schedule == "constrained"
    common = dict(demand_correction_factor=1.038, enable_bec_summer_cap=True, enable_min_runtime=True,
                  min_runtime_hours=18, min_downtime_hours=6 if constrained else 2)
    standby = dict(enable_ww_standby_correction=constrained)
    hist = hist_cost_per_year(sim_df)
    years = [2023, 2024, 2025]

    def per_year_saving(df, cost_col):
        return {y: float(hist[y] - df.loc[df.index.year == y, cost_col].sum()) for y in years}

    def b_cfgs(wn_mw, nw_mw, temps=None):
        s = wn_mw / (ad["SWP"].ates_charge_max_mw + ad["SN"].ates_charge_max_mw)
        ex = dict(ates_injection_temp_c=temps[0], ates_cold_temp_c=temps[1]) if temps else {}
        return (dc.replace(ad["SWP"], ates_charge_max_mw=ad["SWP"].ates_charge_max_mw * s,
                           ates_discharge_max_mw=ad["SWP"].ates_charge_max_mw * s, **ex),
                dc.replace(ad["SN"], ates_charge_max_mw=ad["SN"].ates_charge_max_mw * s,
                           ates_discharge_max_mw=ad["SN"].ates_charge_max_mw * s, **ex),
                dc.replace(ad["SNW"], ates_charge_max_mw=nw_mw, ates_discharge_max_mw=nw_mw))

    penalty = 0.0
    if option in ("none_tank", "TTES5k"):
        # Tank model: P20/P80, large-tank thresholds on actual EPEX; 50 MW; pure charge on (default)
        kw = dict(small_initial_soc=sim_df.attrs["small_initial_soc"], bec_fixed_historical=False,
                  annual_charge_pct=20, annual_discharge_pct=80, large_price_col=PRICE, **common, **standby,
                  **bkw, **rule_kw)
        if option == "none_tank":
            df = ad["tank"](sim_df, large_tank_volume_m3=1.0, enable_pure_charge_large=False, **kw)
        else:
            df = ad["tank"](sim_df, large_tank_volume_m3=5000.0, large_charge_max_mw=50.0,
                            large_discharge_max_mw=50.0, **kw)
            ch, dis = df["large_tank_charge_mw"].values, df["large_tank_discharge_mw"].values
            r = ad["replay"](ad["ttes_model"](5000, h_over_d=0.6), ch - dis, substeps=16)
            non_bec = sum(df[c].sum() for c in ["WW10_sim_tank_mw", "WW30_sim_tank_mw", "WW01_sim_tank_mw",
                                                 "WW00_sim_tank_mw", "WW02_sim_tank_mw"])
            avg_derving = df["derving_cost_eur"].sum() / non_bec if non_bec > 0 else 0.0
            penalty = max(0.0, dis.sum() - r["useful"].sum()) / 3.0 * avg_derving   # EUR/yr
        py = per_year_saving(df, "total_cost_eur_corrected")
    elif option in ("A1D", "A2D", "A4D", "none_ates_a"):
        # Option A: no standby logic in this model, so "constrained" = 6 h only (as in the thesis runs).
        # none_ates_a = the same model at zero ATES capacity.
        cap = {"A1D": 13.3, "A2D": 26.6, "A4D": 53.2, "none_ates_a": 0.0}[option]
        scen = dc.replace(ad["SA"], ates_charge_max_mw=cap, ates_discharge_max_mw=cap)
        df = ad["A"]["run_ates_option_a_soc_driven"](
            sim_df, scen, dispatch_strategy="S0", trigger_pp=25, permissive_discharge_pct=45.0,
            annual_charge_pct=30, annual_discharge_pct=60, **common, **bkw, **rule_kw)
        py = per_year_saving(df, "total_system_cost_eur_corrected")
    elif option == "BWN110":
        wp, n, nw = b_cfgs(13.3, 0.0, temps=(110.0, 58.0))
        df = ad["B"]["run_ates_three_loops_soc_driven"](
            sim_df, wp, n, nw, trigger_pp=25, permissive_discharge_pct=45.0,
            annual_charge_pct=15, annual_discharge_pct=75, **common, **standby, **rule_kw)
        py = per_year_saving(df, "total_system_cost_eur_corrected")
    elif option in ("Bstd", "none_ates"):
        # none_ates = zero-capacity run of the three-loop ATES model; it reproduces S_ref = 1,073.7 and
        # Z = 1,849.7 k EUR/yr. (The Option A model has no standby logic: its own zero-capacity
        # constrained run is 1,697.2, not S_ref -- see the regression-gate note in the report.)
        wn_nw = (6.64, 6.64) if option == "Bstd" else (0.0, 0.0)
        wp, n, nw = b_cfgs(*wn_nw)
        if option == "none_ates":
            wp, n = (dc.replace(c, ates_charge_max_mw=0.0, ates_discharge_max_mw=0.0) for c in (wp, n))
        df = ad["b_v6"](sim_df, wp, n, nw, annual_charge_pct=15, annual_discharge_pct=75, **common, **standby,
                        **rule_kw)
        py = per_year_saving(df, "total_system_cost_eur_corrected")
    else:
        raise KeyError(option)
    per_year = {y: (v - penalty) / 1000.0 for y, v in py.items()}
    out = {"saving_keur_yr": float(np.mean(list(per_year.values()))), "per_year": per_year,
           "unmet_mwh": float(df["unmet_demand_tank_mw"].sum())}
    if keep_frame:
        out["frame"] = df
    return out

# =============================================================================================
# 3. SCENARIO TRANSFORMS
# =============================================================================================
def apply_scenario(df: pd.DataFrame, sc: dict, price_file: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if sc.get("price_file"):
        pf = pd.read_csv(price_file, sep=None, engine="python", index_col=0, parse_dates=True)
        s = pf.iloc[:, 0].astype(float).reindex(out.index)
        if s.isna().any():
            raise ValueError(f"price file does not cover the model index ({int(s.isna().sum())} hours missing)")
        for c in PRICE_COLS:
            if c in out.columns and c != "forecast_price":
                out[c] = s.values
    else:
        main = next((c for c in PRICE_COLS if c in out.columns), None)
        if main is None:
            raise KeyError(f"none of {PRICE_COLS} in the frame")
        m = float(out[main].mean())
        for c in PRICE_COLS:
            if c in out.columns:
                out[c] = sc["level"] * (m + sc["spread"] * (out[c] - m))
    if sc["demand"] != 1.0:
        cols = [c for c in DEMAND_COLS if c in out.columns]
        if SCALE_HISTORY:
            cols += [c for c in out.columns if re.match(HISTORY_COLS_REGEX, c) and c not in cols]
        for c in cols:
            out[c] = out[c] * sc["demand"]
    return out

# =============================================================================================
# 4. MOCK MODEL (plumbing test only, numbers are NOT results)
# =============================================================================================
def _mock_run(sim_df, option, schedule):
    p = sim_df[next(c for c in PRICE_COLS if c in sim_df.columns)]
    d = sim_df["heat_demand_mw"] if "heat_demand_mw" in sim_df.columns else pd.Series(50.0, index=sim_df.index)
    ref = _mock_df()
    sd = float(p.std()) / float(ref["EPEX_price"].std())
    dm = float(d.mean()) / float(ref["heat_demand_mw"].mean())
    base = {"none_tank": 1074.0, "none_ates": 1074.0,
            "TTES5k": 1074.0 + 51, "A1D": 1074.0 + 163.2, "A2D": 1074.0 + 307.1, "A4D": 1074.0 + 463.7,
            "BWN110": 1074.0 + 128.9, "Bstd": 1074.0 + 118.3}[option]
    if schedule == "relaxed":
        base += 775.7
    else:
        base -= {"TTES5k": 51 - 34, "A1D": 163.2 - 130.7, "A2D": 307.1 - 247.5, "A4D": 463.7 - 438.7,
                 "BWN110": 128.9 - 115.3, "Bstd": 118.3 - 107.1}.get(option, 0.0)
    v = base * (0.5 + 0.5 * sd) * (0.6 + 0.4 * dm)
    return {"saving_keur_yr": float(v), "per_year": {}, "unmet_mwh": 0.0}

def _mock_df():
    idx = pd.date_range("2023-01-01", pd.Timestamp(2025, 12, 31, 23), freq="h")
    rng = np.random.default_rng(1)
    return pd.DataFrame({"EPEX_price": 80 + 40 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi) + rng.normal(0, 20, len(idx)),
                         "heat_demand_mw": 50 + 20 * np.cos(np.arange(len(idx)) / 8760 * 2 * np.pi)}, index=idx)

# =============================================================================================
# 5. EXECUTION WITH CACHE
# =============================================================================================
def _key(scn, option, schedule, extra=""):
    if option == "Bstd" and C.bstd_tag():         # B-std at another return temperature than 70 C: own cache files
        extra = f"{extra}{C.bstd_tag()}"
    h = hashlib.md5(json.dumps([scn, option, schedule, SCALE_HISTORY, extra], sort_keys=True).encode()).hexdigest()[:8]
    return CACHE_DIR / f"{scn['name']}__{option}__{schedule}__{h}.json"

def _task(scn, option, schedule, price_file, mock, force):
    f = _key(scn, option, schedule, price_file or "")
    if f.exists() and not force:
        return scn["name"], option, schedule, json.loads(f.read_text()), True
    df = _mock_df() if mock else load_sim_df()
    df = apply_scenario(df, scn, price_file)
    t0 = time.time()
    res = _mock_run(df, option, schedule) if mock else run_saving(df, option, schedule)
    res["seconds"] = round(time.time() - t0, 1)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res))
    return scn["name"], option, schedule, res, False

def run_scenarios(scn_list, options, lower, workers, price_file, mock, force):
    fams = sorted({OPTIONS[o]["family"] for o in options} | {REFERENCE_FAMILY})
    cases = [(f"none_{f}", sch) for f in fams for sch in ("constrained", "relaxed")]
    cases += [(o, "relaxed") for o in options]
    if lower:
        cases += [(o, "constrained") for o in options]
    jobs = [(s, o, sch) for s in scn_list for (o, sch) in cases]
    out = {}
    def collect(r):
        name, o, sch, res, cached = r
        out[(name, o, sch)] = res
        print(f"  {name:11s} {o:10s} {sch:11s} {res['saving_keur_yr']:9.1f} k EUR/yr   unmet {res.get('unmet_mwh', 0):8.1f} MWh" + ("  (cache)" if cached else f"  ({res.get('seconds', '?')} s)"), flush=True)
    if workers <= 1 or mock:
        for s, o, sch in jobs:
            collect(_task(s, o, sch, price_file, mock, force))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_task, s, o, sch, price_file, mock, force) for s, o, sch in jobs]
            for fu in as_completed(futs):
                collect(fu.result())
    return out

# =============================================================================================
# 6. DECOMPOSITION AND NPV
# =============================================================================================
def npv(capex, opex, s_attr):
    return -capex + (s_attr - opex) * F_PV / 1000.0

# decomposition: S_ref = constrained zero-capacity run; S_flex = relaxed minus constrained zero run; S_stor = option minus
# the zero run of its own model family. f_corr (ROSIM calibration) acts only on S_stor.
def decompose(raw, scn_names, options, lower):
    rows = []
    for s in scn_names:
        g = lambda o, sch: raw[(s, o, sch)]["saving_keur_yr"]
        S_ref = g(f"none_{REFERENCE_FAMILY}", "constrained")
        Z     = g(f"none_{REFERENCE_FAMILY}", "relaxed")
        S_flex = Z - S_ref
        for o in options:
            c = OPTIONS[o]; fam = c["family"]
            S_stor = g(o, "relaxed") - g(f"none_{fam}", "relaxed")
            S_stor_c = (g(o, "constrained") - g(f"none_{fam}", "constrained")) if lower else np.nan
            f = c["f_corr"]
            tiers = {"lower": f * S_stor_c, "base": S_flex + f * S_stor, "potential": PHI * S_ref + S_flex + f * S_stor}
            r = dict(scenario=s, option=o, S_ref=S_ref, Z=Z, S_flex=S_flex, S_stor=S_stor, f_corr=f,
                     S_stor_cal=f * S_stor, gain_lower=f * S_stor_c if lower else np.nan,
                     unmet_mwh=raw[(s, o, "relaxed")].get("unmet_mwh", 0.0), opex=c["opex"])
            for t, sa in tiers.items():
                for name, cap in zip(("low", "mid", "high"), c["capex"]):
                    r[f"npv_{t}_{name}"] = npv(cap, c["opex"], sa)
                r[f"S_attr_{t}"] = sa
            r["potential_valid"] = bool(HISTORY_REBUILT) or s == "base"
            rows.append(r)
    return pd.DataFrame(rows)

def regression_gate(res, options, lower):
    bad = []
    b = res[res.scenario == "base"]
    r0 = b.iloc[0]
    for name, exp in (("S_ref", EXPECTED["S_ref"]), ("Z", EXPECTED["Z"])):
        if abs(r0[name] - exp) > TOL_MAIN:
            bad.append(f"{name}: got {r0[name]:.1f}, thesis {exp:.1f}")
    for _, r in b.iterrows():
        e = EXPECTED["S_stor"].get(r.option)
        if e is not None and abs(r.S_stor - e) > (TOL_ROUNDED if r.option == "TTES5k" else TOL_MAIN):
            bad.append(f"S_stor {r.option}: got {r.S_stor:.1f}, thesis {e:.1f}")
        e = EXPECTED["gain_lower"].get(r.option)
        if lower and e is not None and abs(r.gain_lower - e) > TOL_ROUNDED:
            bad.append(f"lower-bound gain {r.option}: got {r.gain_lower:.1f}, thesis {e:.1f}")
    return bad

# =============================================================================================
# 7. OUTPUT
# =============================================================================================
def fmt(x, d=2):
    if pd.isna(x): return "--"
    s = f"{abs(x):.{d}f}"
    return (r"$+$" if x >= 0 else r"$-$") + s

def write_outputs(res, scn_names, options, lower, raw=None):
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    res.to_csv(OUT_DIR / "scenario_results.csv", index=False, float_format="%.3f")
    cols = " & ".join(rf"\textbf{{{SCEN_LABEL.get(s, s)}}}" for s in scn_names)
    def table(tier, caption, label):
        L = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
             r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%", r"\begin{tabular}{l" + "r" * len(scn_names) + "}",
             r"\toprule", rf"\textbf{{Option}} & {cols} \\", r"\midrule"]
        for o in options:
            v = [fmt(res[(res.scenario == s) & (res.option == o)][f"npv_{tier}_mid"].iloc[0]) for s in scn_names]
            L.append(f"{LABELS[o]} & " + " & ".join(v) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        return "\n".join(L)
    t = table("base", r"NPV of the base case at mid CAPEX (million euro, 5\% WACC, 30-year lifetime) for the demand and price scenarios. The calibration factors $f_\text{corr}$, the CAPEX and the OPEX are those of the base scenario.", "tab:scenario-npv")
    d = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Decomposition of the annual saving (k\texteuro/yr) for the demand and price scenarios. $S_\text{ref}$ is the dispatch-only reference, $S_\text{flex}$ the value of the relaxed schedule and $S_\text{stor}$ the uncalibrated storage effect of Option~A 1D.}",
         r"\label{tab:scenario-decomposition}", r"\small", r"\begin{tabular}{l" + "r" * len(scn_names) + "}", r"\toprule",
         rf"\textbf{{Component}} & {cols} \\", r"\midrule"]
    ref = options[0] if "A1D" not in options else "A1D"
    for name, key in ((r"$S_\text{ref}$", "S_ref"), (r"$S_\text{flex}$", "S_flex"), (r"$S_\text{stor}$ (" + LABELS[ref] + ")", "S_stor")):
        v = [f"{res[(res.scenario == s) & (res.option == ref)][key].iloc[0]:,.0f}".replace(",", "{,}") for s in scn_names]
        d.append(f"{name} & " + " & ".join(v) + r" \\")
    d += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_DIR / "scenario_npv_base.tex").write_text(t + "\n\n" + "\n".join(d) + "\n")
    # console summary
    print("\nBase-case NPV at mid CAPEX (M EUR)")
    piv = res.pivot(index="option", columns="scenario", values="npv_base_mid").reindex(options)[scn_names]
    print(piv.round(2).to_string())
    if lower:
        print("\nLower-bound NPV at mid CAPEX (M EUR)")
        print(res.pivot(index="option", columns="scenario", values="npv_lower_mid").reindex(options)[scn_names].round(2).to_string())
    print("\nS_ref, S_flex per scenario (k EUR/yr)")
    print(res.groupby("scenario")[["S_ref", "S_flex"]].first().reindex(scn_names).round(1).to_string())
    if raw:
        # Unmet demand is compared with the SAME option and schedule in the base scenario.
        base = {(o, sch): r.get("unmet_mwh", 0.0) for (s, o, sch), r in raw.items() if s == "base"}
        print("\nUnmet heat demand in the base scenario (MWh over 2023-2025):")
        for (o, sch), v in sorted(base.items()):
            print(f"  {o:12s} {sch:11s} {v:8.0f}")
        print("  (Option A constrained: the Option A model has no standby rule, so with 6 h minimum downtime\n"
              "   units switch off and cannot restart in time -- unmet even at zero capacity (none_ates_a);\n"
              "   the 4 % delivery loss below 90 degC adds at most a few hundred MWh.)")
        warn = []
        for (s, o, sch), r in sorted(raw.items()):
            if s != "base" and (o, sch) in base and r.get("unmet_mwh", 0.0) - base[(o, sch)] > 100:
                warn.append(f"{s} {o} {sch}: {r['unmet_mwh']:.0f} MWh (base {base[(o, sch)]:.0f}, "
                            f"+{r['unmet_mwh'] - base[(o, sch)]:.0f})")
        if warn:
            print("\nWARNING unmet heat demand more than 100 MWh above the base scenario "
                  "(capacity limit reached, report this in the text):")
            print("\n".join("  " + w for w in warn))
    if not HISTORY_REBUILT:
        print("\nNOTE potential tier outside the base scenario is only valid when HISTORY_REBUILT = True (benchmark rebuilt per scenario).")
    print(f"\nWritten: {OUT_DIR/'scenario_results.csv'} and {OUT_DIR/'scenario_npv_base.tex'}")

# =============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--options", default=",".join(OPTIONS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-lower-bound", action="store_true")
    ap.add_argument("--include-level", action="store_true")
    ap.add_argument("--price-file")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-check", action="store_true")
    a = ap.parse_args()
    options = [o for o in a.options.split(",") if o]
    lower = not a.no_lower_bound
    names = [s for s in a.scenarios.split(",") if s]
    if a.include_level:
        names += [s for s in ("level_080", "level_120") if s not in names]
    if a.price_file:
        names.append("price_file")
    if "base" not in names: names.insert(0, "base")
    scn = {n: dict(SCENARIOS.get(n, SCENARIOS["base"]), name=n, price_file=(n == "price_file")) for n in names}
    for n in names:
        if n not in SCENARIOS and n != "price_file": sys.exit(f"unknown scenario {n}")
    print("BASE SCENARIO FIRST (regression gate)")
    raw = run_scenarios([scn["base"]], options, lower, a.workers, a.price_file, a.mock, a.force)
    res0 = decompose(raw, ["base"], options, lower)
    bad = regression_gate(res0, options, lower)
    if bad and not a.mock and not a.skip_check:
        print("\nREGRESSION GATE FAILED. The base scenario does not reproduce the thesis:")
        print("\n".join("  " + b for b in bad)); sys.exit(1)
    print("regression gate:", "passed" if not bad else "skipped (mock)")
    rest = [scn[n] for n in names if n != "base"]
    if rest:
        print("\nOTHER SCENARIOS")
        raw.update(run_scenarios(rest, options, lower, a.workers, a.price_file, a.mock, a.force))
    res = decompose(raw, names, options, lower)
    write_outputs(res, names, options, lower, raw)

if __name__ == "__main__":
    main()