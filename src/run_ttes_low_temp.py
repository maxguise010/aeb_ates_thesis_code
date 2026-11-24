#!/usr/bin/env python3
"""TTES runs at 5,000 and 10,000 m3 charged at 110, 95 and 90 C, with the thermal replay; gives the discharged heat of
the tank. Run by run_all.py."""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
HERE = C.RES_TTES   # output folder

PRICE = "Prijs_Gemiddelde_EPEX_Spot"
YEARS = [2023, 2024, 2025]
T_RET, RHO, CP = 58.0, 950.0, 4220.0          # TTES_DEFAULTS (ttes_thermal_replay.py)
T_SUPPLY_MIN = 90.0
NON_BEC = ["WW10_sim_tank_mw", "WW30_sim_tank_mw", "WW01_sim_tank_mw", "WW00_sim_tank_mw", "WW02_sim_tank_mw"]
CACHE = C.TTES_LOW_TEMP_CACHE


def topup_cost(df, topup):
    """EUR per hour for the top-up heat, assigned along the dispatch order (see module docstring)."""
    from config import derivation_factors as f
    from simulation_v7 import bec_net_revenue_per_gj
    cost = np.zeros(len(df))
    p = df[PRICE].values
    for i in np.nonzero(topup > 1e-9)[0]:
        r = df.iloc[i]
        need = topup[i]
        # BEC first (must-run, demand following); its extra heat earns the BEC net revenue when active
        if r["WWBEC_hist_available"] == 1:
            room = max(0.0, r["WWBEC_max_mw_effective"] - r["WWBEC_sim_tank_mw"])
            take = min(need, room)
            if take > 0:
                rev = bec_net_revenue_per_gj(p[i]) * 3.6 if r["WW_BEC"] >= 1.0 else 0.0
                cost[i] -= take * rev
                need -= take
        aftap_cap = r["WW30_aftap_cap_mw"]
        ww30 = r["WW30_sim_tank_mw"]
        stages = [("WW10", r["WW10_max_mw"] - r["WW10_sim_tank_mw"], f["WW10"]),
                  ("WW30", min(r["WW30_max_mw"], aftap_cap) - min(ww30, aftap_cap), f["WW30_aftap"]),
                  ("WW01", r["WW01_max_mw"] - r["WW01_sim_tank_mw"], f["WW01"]),
                  ("WW00", r["WW00_max_mw"] - r["WW00_sim_tank_mw"], f["WW00"]),
                  ("WW02", r["WW02_max_mw"] - r["WW02_sim_tank_mw"], f["WW02"]),
                  ("WW30", max(0.0, r["WW30_max_mw"] - aftap_cap) - max(0.0, ww30 - aftap_cap), f["WW30_reductie"])]
        for hx, room, fac in stages:
            if need <= 1e-9:
                break
            if r[f"{hx}_hist_available"] != 1 or room <= 1e-9:
                continue
            take = min(need, room)
            cost[i] += take * fac * p[i]
            need -= take
        if need > 1e-9:                      # no spare unit: value at the dearest factor (conservative)
            cost[i] += need * f["WW30_reductie"] * p[i]
    return cost


CAL_BUFFER = dict(reserve_max_fraction=0.35, reserve_horizon_h=24, small_soc_max=0.68,
                  daily_charge_rank_n=3, daily_discharge_rank_n=3)   # calibrated buffer setting


def job(schedule, volume, t_charge, force=False, ss10k=4, buffer="thesis"):
    key = CACHE / (f"{schedule}__{volume}__{t_charge:.0f}" + (f"__ss{ss10k}" if volume >= 10_000 and ss10k != 4 else "")
                   + ("__cal" if buffer == "calibrated" else "") + ".json")
    if key.exists() and not force:
        return json.loads(key.read_text())
    import scenario_runner as sr
    ad = sr._adapter()
    sim = sr.load_sim_df()
    hist = sr.hist_cost_per_year(sim)
    con = schedule == "constrained"
    kw = dict(small_initial_soc=sim.attrs["small_initial_soc"], bec_fixed_historical=False,
              annual_charge_pct=20, annual_discharge_pct=80, large_price_col=PRICE,
              demand_correction_factor=1.038, enable_bec_summer_cap=True, enable_min_runtime=True,
              min_runtime_hours=18, min_downtime_hours=6 if con else 2, enable_ww_standby_correction=con,
              **(CAL_BUFFER if buffer == "calibrated" else {}))
    t0 = time.time()
    yr = sim.index.year
    if volume == 0:
        df = ad["tank"](sim, large_tank_volume_m3=1.0, enable_pure_charge_large=False, **kw)
        py = {y: float(hist[y] - df.loc[yr == y, "total_cost_eur_corrected"].sum()) for y in YEARS}
        out = dict(schedule=schedule, volume_m3=0, t_charge_c=t_charge, saving_py=py)
    else:
        dT = None if t_charge == 110 else t_charge - T_RET
        df = ad["tank"](sim, large_tank_volume_m3=float(volume), large_charge_max_mw=50.0,
                        large_discharge_max_mw=50.0, large_tank_delta_t_k=dT, **kw)
        ch, dis = df["large_tank_charge_mw"].values, df["large_tank_discharge_mw"].values
        model = ad["ttes_model"](volume, h_over_d=0.6, T_hot=float(t_charge))
        r = ad["replay"](model, ch - dis, T_inj=float(t_charge), substeps=16 if volume < 10_000 else ss10k,
                         T_supply_min=T_SUPPLY_MIN)
        non_bec = sum(df[c].sum() for c in NON_BEC)
        avg_derving = df["derving_cost_eur"].sum() / non_bec if non_bec > 0 else 0.0
        penalty_yr = max(0.0, dis.sum() - r["useful"].sum()) / 3.0 * avg_derving
        py = {y: float(hist[y] - df.loc[yr == y, "total_cost_eur_corrected"].sum() - penalty_yr) for y in YEARS}
        tc = topup_cost(df, r["topup"])
        disch_h = dis > 1e-6
        out = dict(schedule=schedule, volume_m3=volume, t_charge_c=t_charge, saving_py=py,
                   topup_cost_py={y: float(tc[yr == y].sum()) for y in YEARS},
                   topup_mwh_py={y: float(r["topup"][yr == y].sum()) for y in YEARS},
                   charged_dispatch_mwh=float(ch.sum()), discharged_dispatch_mwh=float(dis.sum()),
                   stored_mwh=float(r["stored"].sum()), useful_mwh=float(r["useful"].sum()),
                   curtailed_mwh=float(r["curtailed"].sum()), loss_mwh=float(r["loss_mwh"]),
                   shortfall_mwh=float(max(0.0, dis.sum() - r["useful"].sum())), avg_derving=float(avg_derving),
                   discharge_hours=int(disch_h.sum()),
                   lowT_share_of_discharge_time=float(r["lowT_frac"][disch_h].mean()) if disch_h.any() else 0.0,
                   lowT_hours_any=int((r["lowT_frac"][disch_h] > 0).sum()))
    out["seconds"] = round(time.time() - t0)
    CACHE.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--substeps-10k", type=int, default=4, help="replay sub-steps from 10,000 m3 (thesis: 4)")
    ap.add_argument("--buffer", default="thesis", choices=("thesis", "calibrated"))
    a = ap.parse_args()
    jobs = [(s, 0, 110) for s in ("relaxed", "constrained")]
    jobs += [(s, v, t) for s in ("relaxed", "constrained") for v in (5_000, 10_000) for t in (110, 95, 90)]
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(job, *j, a.force, a.substeps_10k, a.buffer): j for j in jobs}
        for fu in as_completed(futs):
            o = fu.result()
            res.append(o)
            print(f"  {o['schedule']:11s} {o['volume_m3']:>6} m3 {o['t_charge_c']:>4.0f} C  "
                  f"{np.mean(list(o['saving_py'].values())) / 1e3:9.1f} k EUR/yr  ({o['seconds']} s)", flush=True)

    zero = {o["schedule"]: o for o in res if o["volume_m3"] == 0}
    rows, pyrows = [], []
    for o in sorted((o for o in res if o["volume_m3"] > 0), key=lambda o: (o["schedule"], o["volume_m3"], -o["t_charge_c"])):
        z = zero[o["schedule"]]
        v, t = o["volume_m3"], o["t_charge_c"]
        dens = RHO * CP * (t - T_RET) / 3.6e6                       # kWh/m3
        cap = v * dens / 1e3                                        # MWh
        s_py = {y: (o["saving_py"][str(y) if str(y) in o["saving_py"] else y] -
                    z["saving_py"][str(y) if str(y) in z["saving_py"] else y]) / 1e3 for y in YEARS}
        g = lambda d, y: d[str(y)] if str(y) in d else d[y]
        tc_py = {y: g(o["topup_cost_py"], y) / 1e3 for y in YEARS}
        for y in YEARS:
            pyrows.append(dict(schedule=o["schedule"], volume_m3=v, t_charge_c=t, year=y,
                               S_stor_keur=s_py[y], topup_mwh=g(o["topup_mwh_py"], y), topup_cost_keur=tc_py[y],
                               S_stor_net_topup_keur=s_py[y] - tc_py[y]))
        rows.append(dict(
            schedule=o["schedule"], volume_m3=v, t_charge_c=t, delta_t_k=t - T_RET, capacity_mwh=cap,
            density_kwh_m3=dens,
            charged_dispatch_mwh_yr=o["charged_dispatch_mwh"] / 3, discharged_dispatch_mwh_yr=o["discharged_dispatch_mwh"] / 3,
            stored_replay_mwh_yr=o["stored_mwh"] / 3, discharged_replay_mwh_yr=o["useful_mwh"] / 3,
            curtailed_replay_mwh_yr=o["curtailed_mwh"] / 3, shortfall_mwh_yr=o["shortfall_mwh"] / 3,
            loss_mwh_yr=o["loss_mwh"] / 3, loss_pct_of_stored=100 * o["loss_mwh"] / o["stored_mwh"] if o["stored_mwh"] else np.nan,
            cycles_per_yr=o["charged_dispatch_mwh"] / 3 / cap,
            topup_mwh_yr=np.mean([g(o["topup_mwh_py"], y) for y in YEARS]),
            topup_cost_keur_yr=np.mean(list(tc_py.values())),
            S_stor_keur_yr=np.mean(list(s_py.values())),
            S_stor_net_topup_keur_yr=np.mean([s_py[y] - tc_py[y] for y in YEARS]),
            **{f"S_stor_{y}": s_py[y] for y in YEARS},
            **{f"S_stor_net_{y}": s_py[y] - tc_py[y] for y in YEARS},
            lowT_share_of_discharge_time=o["lowT_share_of_discharge_time"],
            lowT_share_of_discharge_hours_any=o["lowT_hours_any"] / o["discharge_hours"] if o["discharge_hours"] else np.nan,
            Z_or_Sref_keur_yr=np.mean(list(z["saving_py"].values())) / 1e3))
    t = pd.DataFrame(rows)
    sfx = ("" if a.substeps_10k == 4 else f"_ss{a.substeps_10k}") + ("_cal" if a.buffer == "calibrated" else "")
    t["substeps"] = np.where(t.volume_m3 < 10_000, 16, a.substeps_10k)
    t.to_csv(HERE / f"ttes_low_temp_results{sfx}.csv", index=False, float_format="%.3f")
    pd.DataFrame(pyrows).to_csv(HERE / f"ttes_low_temp_per_year{sfx}.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
