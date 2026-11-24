#!/usr/bin/env python3
"""Splits the failed restart events of the relaxed schedule by cause and gives the share of S_flex that the storage
can bridge (Table 7.10). Needs the raw data, not run by run_all.py."""
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
HERE = C.RESTART_SPLIT   # output folder (s_flex_shares.csv and the other tables)
STANDBY_SUMMARY = C.STANDBY_BRIDGING_SUMMARY   # module level: analyse() uses the local name C for a frame
CONST = 0.001157
RS = (1, 2, 4)
UNITS = {"WW30": "ww30_hist_active", "WW01": "ww01_hist_active"}
RATED = {"TTES5k": 50.0, "A1D": 13.3, "A2D": 26.6, "A4D": 53.2, "BWN110": 13.3, "Bstd": 13.28}
B_LOOPS = {"westpoort": 95_000.0, "noord": 140_000.0, "nieuwwest": 265_000.0}
B_TCOLD = {"BWN110": {"westpoort": 58.0, "noord": 58.0, "nieuwwest": 70.0},
           "Bstd": {k: C.BSTD_T_RETURN_C for k in ("westpoort", "noord", "nieuwwest")}}   # config.BSTD_T_RETURN_C
CASES = [("none_tank", "constrained"), ("none_tank", "relaxed")] + [(o, "relaxed") for o in RATED]


def run_one(o, s, out):
    f = Path(out) / f"{C.tagged(o)}__{s}.pkl"
    if f.exists():
        return o, s, "cache"
    import scenario_runner as sr
    r = sr.run_saving(sr.load_sim_df(), o, s, keep_frame=True)
    df = r.pop("frame")
    keep = [c for c in df.columns if c in (
        "total_cost_eur_corrected", "total_system_cost_eur_corrected", "WW30_sim_tank_mw", "WW01_sim_tank_mw",
        "ww30_hist_active", "ww01_hist_active", "WW30_min_mw", "WW01_min_mw", "delta_t_c",
        "large_tank_volume_m3", "large_tank_discharge_mw", "ates_volume_m3", "ates_aquifer_temp_c",
        "ates_discharge_mw", "ates_westpoort_soc", "ates_noord_soc", "ates_nieuwwest_soc",
        "ates_westpoort_temp_c", "ates_noord_temp_c", "ates_nieuwwest_temp_c")]
    df[keep].to_pickle(f)
    return o, s, f"{r['saving_keur_yr']:.2f}"


def energies(name, F):
    """usable energy (MWh) with the SOC minimum, with SOC minimum 0, and when full (SOC 0.95)."""
    if name == "TTES5k":
        dt = np.clip(F["delta_t_c"].values, 10, None) * CONST
        V = F["large_tank_volume_m3"].values
        return (np.clip(V - 250.0, 0, None) * dt, V * dt, np.full(len(V), (0.95 - 0.05) * 5000.0) * dt)
    if name.startswith("A"):
        dT = np.clip(F["ates_aquifer_temp_c"].values - 58.0, 0, None) * CONST
        dT_full = (110.0 - 58.0) * CONST
        V = F["ates_volume_m3"].values
        return (np.clip(V - 25_000.0, 0, None) * dT, V * dT, np.full(len(V), 0.90 * 500_000.0 * dT_full))
    e1 = e0 = ef = 0.0
    for loop, cap in B_LOOPS.items():
        tc = B_TCOLD[name][loop]
        t_inj = 110.0 if (name == "BWN110" and loop != "nieuwwest") else 90.0
        if name == "BWN110" and loop == "nieuwwest":
            continue                                       # no NW store in B-WN110
        V = F[f"ates_{loop}_soc"].values * cap
        dT = np.clip(F[f"ates_{loop}_temp_c"].values - tc, 0, None) * CONST
        e1 = e1 + np.clip(V - 0.05 * cap, 0, None) * dT
        e0 = e0 + V * dT
        ef = ef + np.full(len(V), 0.90 * cap * (t_inj - tc) * CONST)
    return e1, e0, ef


def analyse(out):
    out = Path(out)
    C, Rr = pd.read_pickle(out / "none_tank__constrained.pkl"), pd.read_pickle(out / "none_tank__relaxed.pkl")
    n = len(Rr)
    dcost = (C["total_cost_eur_corrected"] - Rr["total_cost_eur_corrected"]).values
    s_flex_total = dcost.sum()
    sto = {}
    for name in RATED:
        F = pd.read_pickle(out / f"{__import__('config').tagged(name)}__relaxed.pkl")
        disch = F["large_tank_discharge_mw"].values if name == "TTES5k" else F["ates_discharge_mw"].values
        sto[name] = (energies(name, F), disch)
    events = []
    for u, act_col in UNITS.items():
        o = Rr[f"{u}_sim_tank_mw"].values
        off = o < 1e-6
        act = Rr[act_col].values.astype(bool)
        d = np.diff(np.concatenate(([False], off & act, [False])).astype(int))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if s == 0 or e >= n or off[s - 1] or off[e]:
                continue
            events.append((u, s, e))
    share = np.zeros(n)
    for _, s, e in events:
        share[s:e] += 1
    rows = []
    for u, s, e in events:
        o = Rr[f"{u}_sim_tank_mw"].values
        val = float((dcost[s:e] / share[s:e]).sum())
        for R in RS:
            w = slice(e, min(n, e + R))
            sf = o[w].copy()
            k = np.flatnonzero(sf < 1e-6)
            if len(k):
                sf[k[0]:] = 0.0
            for name, ((e1, e0, ef), disch) in sto.items():
                spare = RATED[name] - disch[w]
                used = disch[w].sum()
                avail, avail0, availf = e1[e - 1] - used, e0[e - 1] - used, ef[e - 1] - used
                p_ok = bool((sf <= spare + 1e-9).all())
                h_ok = bool(sf.sum() <= avail + 1e-9)
                cause = "covered" if p_ok and h_ok else ("both" if not p_ok and not h_ok else
                                                         ("discharge power" if not p_ok else "stored heat"))
                rows.append(dict(option=name, R_h=R, unit=u, start=str(Rr.index[s]), duration_h=e - s,
                                 schedule_value_eur=val, heat_needed_mwh=float(sf.sum()),
                                 power_needed_mw=float(sf.max()), usable_energy_mwh=float(avail),
                                 soc_proxy_energy_mwh=float(e1[e - 1]), spare_power_min_mw=float(spare.min()),
                                 cause=cause, covered=cause == "covered",
                                 heat_shortfall_mwh=max(0.0, float(sf.sum() - avail)),
                                 power_shortfall_mw=max(0.0, float((sf - spare).max())),
                                 lever_soc_min_0=p_ok and sf.sum() <= avail0 + 1e-9,
                                 lever_full_at_start=p_ok and sf.sum() <= availf + 1e-9,
                                 lever_no_power_limit=h_ok))
    ev = pd.DataFrame(rows)
    ev.to_csv(HERE / "restart_events.csv", index=False, float_format="%.3f")
    # reproduction check against the earlier standby-bridging result (data/standby_bridging_summary.csv)
    old = pd.read_csv(STANDBY_SUMMARY)
    rep = ev.groupby(["option", "R_h"]).covered.mean().rename("share_covered_now").reset_index()
    rep = rep.merge(old[["option", "R_h", "share_fully_covered"]], how="left", on=["option", "R_h"])
    rep.to_csv(HERE / "reproduction.csv", index=False, float_format="%.4f")
    # cause table
    cause = (ev.groupby(["option", "R_h", "cause"]).size().unstack(fill_value=0))
    cause["events"] = cause.sum(axis=1)
    for c in ("covered", "stored heat", "discharge power", "both"):
        if c not in cause:
            cause[c] = 0
        cause[f"share_{c}"] = cause[c] / cause["events"]
    cause.to_csv(HERE / "causes.csv", float_format="%.4f")
    dist = []
    for (o, R, c), g in ev[~ev.covered].groupby(["option", "R_h", "cause"]):
        dist.append(dict(option=o, R_h=R, cause=c, n=len(g),
                         heat_short_P50=g.heat_shortfall_mwh.median(), heat_short_P90=g.heat_shortfall_mwh.quantile(.9),
                         heat_short_max=g.heat_shortfall_mwh.max(), power_short_P50=g.power_shortfall_mw.median(),
                         power_short_P90=g.power_shortfall_mw.quantile(.9), power_short_max=g.power_shortfall_mw.max()))
    pd.DataFrame(dist).to_csv(HERE / "shortfall_distribution.csv", index=False, float_format="%.3f")
    lev = []
    for (o, R), g in ev.groupby(["option", "R_h"]):
        f = g[~g.covered]
        lev.append(dict(option=o, R_h=R, events=len(g), covered=g.covered.mean(), failed=len(f),
                        **{f"extra_cov_{l}": f[f"lever_{l}"].mean() if len(f) else 0.0
                           for l in ("soc_min_0", "full_at_start", "no_power_limit")},
                        **{f"covered_with_{l}": (g.covered | g[f"lever_{l}"]).mean()
                           for l in ("soc_min_0", "full_at_start", "no_power_limit")}))
    pd.DataFrame(lev).to_csv(HERE / "levers.csv", index=False, float_format="%.4f")
    sh = []
    for (o, R), g in ev.groupby(["option", "R_h"]):
        row = dict(option=o, R_h=R, s_flex_total_keur_yr=s_flex_total / 3e3,
                   s_flex_in_events_keur_yr=g.schedule_value_eur.sum() / 3e3)
        for c in ("covered", "stored heat", "discharge power", "both"):
            gc = g[g.cause == c]
            row[f"value_{c}_keur_yr"] = gc.schedule_value_eur.sum() / 3e3
            row[f"share_value_{c}"] = gc.schedule_value_eur.sum() / s_flex_total
            row[f"share_count_{c}"] = len(gc) / len(g)
            row[f"share_heat_{c}"] = gc.heat_needed_mwh.sum() / g.heat_needed_mwh.sum()
        sh.append(row)
    pd.DataFrame(sh).to_csv(__import__("config").restart_shares(), index=False, float_format="%.4f")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(f"events {len(events)}; S_flex {s_flex_total / 3e3:.1f} k EUR/yr")
    print(rep.round(4).to_string(index=False))
    print(cause.round(3).to_string())
    print(pd.DataFrame(dist).round(2).to_string(index=False))
    print(pd.DataFrame(lev).round(3).to_string(index=False))
    print(pd.DataFrame(sh).round(3).to_string(index=False))


if __name__ == "__main__":
    out, phase = sys.argv[1], sys.argv[2]
    Path(out).mkdir(parents=True, exist_ok=True)
    if phase == "run":
        with ProcessPoolExecutor(max_workers=7) as ex:
            for fu in as_completed([ex.submit(run_one, o, s, out) for o, s in CASES]):
                print(*fu.result(), flush=True)
    else:
        analyse(out)
