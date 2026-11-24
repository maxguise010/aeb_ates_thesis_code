#!/usr/bin/env python3
"""TTES runs with shorter price windows; the package only uses its BEC-term function (through fcorr_runs.py). The runs
themselves need the raw data."""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
HERE = C.SHORT_WINDOW   # work folder (hourly frames contain AEB data)
CACHE, FRAMES = HERE / "cache", HERE / "frames"
PRICE = "Prijs_Gemiddelde_EPEX_Spot"
YEARS = [2023, 2024, 2025]
CAL = dict(reserve_max_fraction=0.35, reserve_horizon_h=24, small_soc_max=0.68,
           daily_charge_rank_n=3, daily_discharge_rank_n=3)
NON_BEC = ["WW10_sim_tank_mw", "WW30_sim_tank_mw", "WW01_sim_tank_mw", "WW00_sim_tank_mw", "WW02_sim_tank_mw"]
KEEP = [PRICE, "heat_demand_mw", "delta_t_c", "large_tank_volume_m3", "large_tank_charge_mw",
        "large_tank_discharge_mw", "small_tank_volume_m3", "small_min_volume_required_m3",
        "unmet_demand_tank_mw", "WWBEC_sim_tank_mw", "WW_BEC", "bec_extra_revenue_eur", "total_cost_eur_corrected",
        "bec_displaced_flag"]
WINDOWS = [365, 120, 90, 60, 45, 30, 14]
PAIRS = [(20, 80), (10, 90), (25, 75), (30, 70), (30, 60)]


def tag_of(schedule, buffer, volume, window, cp, dp):
    return f"{schedule}__{buffer}__V{volume}__W{window}__P{cp}-{dp}"


def bec_terms(df):
    from simulation_v7 import bec_net_revenue_per_gj
    rate = bec_net_revenue_per_gj(df[PRICE]) * 3.6
    d = df["WWBEC_sim_tank_mw"] - df["WW_BEC"]
    act = df["WW_BEC"] >= 1.0
    yr = df.index.year
    return ({y: float(df.loc[yr == y, "bec_extra_revenue_eur"].sum()) for y in YEARS},
            {y: float((d[(yr == y) & act] * rate[(yr == y) & act]).sum()) for y in YEARS})


def job(schedule, buffer, volume, window, cp, dp):
    tag = tag_of(schedule, buffer, volume, window, cp, dp)
    f = CACHE / f"{tag}.json"
    if f.exists():
        return tag, json.loads(f.read_text()), True
    import scenario_runner as sr
    from run_ttes_low_temp import topup_cost
    ad = sr._adapter()
    sim = sr.load_sim_df()
    hist = sr.hist_cost_per_year(sim)
    con = schedule == "constrained"
    kw = dict(small_initial_soc=sim.attrs["small_initial_soc"], bec_fixed_historical=False,
              annual_charge_pct=cp, annual_discharge_pct=dp, large_price_col=PRICE,
              demand_correction_factor=1.038, enable_bec_summer_cap=True, enable_min_runtime=True,
              min_runtime_hours=18, min_downtime_hours=6 if con else 2, enable_ww_standby_correction=con,
              large_price_window_days=None if window == 365 else window,
              **(CAL if buffer in ("cal", "relH", "relD") else {}),
              **({"buffer_release_rule": {"relH": "hourly", "relD": "daily"}[buffer]}
                 if buffer in ("relH", "relD") else {}))
    t0 = time.time()
    yr = sim.index.year
    out = dict(tag=tag, schedule=schedule, buffer=buffer, volume=volume, window=window, cp=cp, dp=dp)
    if volume == 0:
        df = ad["tank"](sim, large_tank_volume_m3=1.0, enable_pure_charge_large=False, **kw)
        penalty_yr = 0.0
        topup = np.zeros(len(df)); tc = np.zeros(len(df)); useful = stored = lowT = np.zeros(len(df)); loss = 0.0
    else:
        df = ad["tank"](sim, large_tank_volume_m3=float(volume), large_charge_max_mw=50.0,
                        large_discharge_max_mw=50.0, **kw)
        ch, dis = df["large_tank_charge_mw"].values, df["large_tank_discharge_mw"].values
        r = ad["replay"](ad["ttes_model"](volume, h_over_d=0.6), ch - dis, substeps=16, T_supply_min=90.0)
        non_bec = sum(df[c].sum() for c in NON_BEC)
        avg_derving = df["derving_cost_eur"].sum() / non_bec if non_bec > 0 else 0.0
        penalty_yr = max(0.0, dis.sum() - r["useful"].sum()) / 3.0 * avg_derving
        topup, useful, stored, lowT, loss = r["topup"], r["useful"], r["stored"], r["lowT_frac"], r["loss_mwh"]
        tc = topup_cost(df, topup)
        out.update(curtailed_mwh=float(r["curtailed"].sum()), shortfall_mwh=float(max(0.0, dis.sum() - useful.sum())),
                   avg_derving=float(avg_derving))
    b_asym, b_sym = bec_terms(df)
    out.update(
        saving_py={y: float(hist[y] - df.loc[yr == y, "total_cost_eur_corrected"].sum() - penalty_yr) for y in YEARS},
        bec_asym_py=b_asym, bec_sym_py=b_sym,
        topup_mwh_py={y: float(topup[yr == y].sum()) for y in YEARS},
        topup_cost_py={y: float(tc[yr == y].sum()) for y in YEARS},
        loss_mwh=float(loss), unmet_mwh=float(df["unmet_demand_tank_mw"].sum()),
        seconds=round(time.time() - t0))
    fr = df[KEEP + [c for c in ("buffer_release_flag",) if c in df.columns]].copy()
    fr["replay_useful_mwh"], fr["replay_stored_mwh"], fr["topup_mwh"], fr["lowT_frac"] = useful, stored, topup, lowT
    fr.to_pickle(FRAMES / f"{tag}.pkl")
    f.write_text(json.dumps(out))
    return tag, out, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase"); ap.add_argument("args", nargs="*"); ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    CACHE.mkdir(exist_ok=True); FRAMES.mkdir(exist_ok=True)
    S = ("relaxed", "constrained")
    if a.phase == "zero":
        jobs = [(s, b, 0, 365, 20, 80) for s in S for b in ("thesis", "cal")]
    elif a.phase == "windows":
        jobs = [(s, b, 5000, w, 20, 80) for w in WINDOWS for s in S for b in ("thesis", "cal")]
    elif a.phase == "pairs":
        ws = [365] + [int(x) for x in a.args[0].split(",")]
        jobs = [(s, b, 5000, w, cp, dp) for w in ws for (cp, dp) in PAIRS[1:] for s in S for b in ("thesis", "cal")]
    elif a.phase == "release":          # buffer release variants A1 (relH) and A2 (relD)
        cfg = [(5000, 365), (5000, 120), (10000, 120)]
        jobs = [(s, b, v, w, 20, 80) for v, w in cfg for s in S for b in ("relH", "relD")]
        jobs += [(s, "thesis", 5000, 365, 20, 80) for s in S]      # regression (cached)
    elif a.phase == "big":
        w = int(a.args[0]); cp, dp = (int(x) for x in a.args[1].split("-"))
        jobs = [(s, b, 10000, ww, c, d) for s in S for b in ("thesis", "cal")
                for ww, c, d in {(w, cp, dp), (365, 20, 80)}]
    else:
        raise SystemExit(a.phase)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fu in as_completed([ex.submit(job, *j) for j in jobs]):
            tag, o, cached = fu.result()
            print(f"  {tag:50s} {np.mean(list(o['saving_py'].values())) / 1e3:9.2f} k EUR/yr "
                  f"({'cache' if cached else str(o['seconds']) + ' s'})", flush=True)


if __name__ == "__main__":
    main()
