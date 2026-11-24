#!/usr/bin/env python3
"""Well-limit sensitivity: hours above the well flow limit and the dispatch with capped power per doublet (summary
from the cache). Run by run_all.py."""
import argparse, dataclasses, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
import data_store as DS
HERE = C.RES_SCEN   # output folder
import fcorr_runs as fr  # noqa: E402

RHO = 1046.9142626824998                                   # audit.py:15
CP = {110.0: 3892.308650728968, 90.0: 3874.6101748438413}  # audit.py:16
LIMIT = {110.0: 221.0, 90.0: 179.0}                        # m3/h per doublet, thesis Q_Thiem
DISPATCH = C.DISPATCH_HOURLY
RUNS = {"A1D": ("A1D", "A", 1, 110.0, 58.0), "A2D": ("A2D", "A", 2, 110.0, 58.0), "A4D": ("A4D", "A", 4, 110.0, 58.0),
        "B-WN110": ("BWN110", "WN", 1, 110.0, 58.0), "B-std WN": ("Bstd", "WN", 1, 90.0, C.BSTD_T_RETURN_C),
        "B-std NW": ("Bstd", "NW", 1, 90.0, C.BSTD_T_RETURN_C)}           # = audit_lib.RUNS / TEMPS
MY_EVALS = C.EVALS_WELLLIMIT
_Y7 = pd.read_csv(C.RES_ROSIM / "rosim_check.csv").query("year == 7").set_index("run").recomputed   # B-std: as bstd_fcorr_v3
TARGET = {"A1D": {"a": 45.161}, "BWN110": {"wn": 48.674},
          "Bstd": {"wn": round(float(_Y7["B-std WN"]), 2), "nw": round(float(_Y7["B-std NW"]), 2)}}   # targets.py R7
UNCAPPED = {"A1D": (5.1551, 1.0), "BWN110": (3.6025, 1.0), "Bstd": (3.1914, 5.2398)}          # direct runs, basis R


# well limit: flow per MW = 3.6e9 / (rho x cp x dT): 16.99 m3/h per MW at 52 K (110/58 C) and 27.7 m3/h per MW
# at the 32 K window of B-std (90/58 C). Limits per doublet: 221 m3/h (110 C) and 179 m3/h (90 C).
def k_m3h_per_mw(t_in, t_ret):
    return 3.6e9 / (RHO * CP[t_in] * (t_in - t_ret))


MW_CAP_A = LIMIT[110.0] / k_m3h_per_mw(110.0, 58.0)
MW_CAP_B = LIMIT[90.0] / k_m3h_per_mw(90.0, C.BSTD_T_RETURN_C)


# ------------------------------------------------------------------------------------------------ part 1
def hours():
    rows = []
    for run, (disp, key, n, t_in, t_ret) in RUNS.items():
        # weekly aggregate of the hourly dispatch (data/dispatch_weekly.csv): per ISO week the sums, maxima and hour counts
        # of the per-doublet charge and discharge that the hourly version computed here (same limit and cap)
        w = DS.dispatch_weekly(C.DISPATCH_WEEKLY, run)
        k, lim = k_m3h_per_mw(t_in, t_ret), LIMIT[t_in]
        cap_mw = lim / k
        rows.append(dict(run=run, doublets=n, k_m3h_per_mw=k, limit_m3h=lim, cap_mw_per_doublet=cap_mw,
                         peak_inj_m3h=w.ch_d_max.max() * k, peak_ext_m3h=w.dis_d_max.max() * k,
                         hours_inj_above_yr=float(w.n_inj_above.sum()) / 3, hours_ext_above_yr=float(w.n_ext_above.sum()) / 3,
                         charged_mwh_yr=float(w.ch_d_sum.sum()) * n / 3, discharged_mwh_yr=float(w.dis_d_sum.sum()) * n / 3,
                         share_discharge_in_hours_above_pct=float(w.dis_d_sum_above.sum() / w.dis_d_sum.sum() * 100),
                         share_discharge_above_cap_pct=float(w.dis_d_excess.sum() / w.dis_d_sum.sum() * 100),
                         share_charge_in_hours_above_pct=float(w.ch_d_sum_above.sum() / w.ch_d_sum.sum() * 100),
                         share_charge_above_cap_pct=float(w.ch_d_excess.sum() / w.ch_d_sum.sum() * 100)))
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "well_limit_hours.csv", index=False, float_format="%.3f")
    print(t.round(2).to_string(index=False))
    return t


# ------------------------------------------------------------------------------------------------ part 2
# sensitivity: the same dispatch with the power per doublet capped at limit / k MW; f_corr is
# recalibrated to the same year-7 target. The headline numbers do not change because of this.
def capped_run(cfg, setting, s1, s2):
    """fcorr_runs.run with the MW values replaced by the well-limit caps (everything else identical)."""
    ad, sim = fr.W["ad"], fr.W["sim"]
    kw = dict(fr.COMMON, **(fr.CAL if setting == "C" else {}))
    if cfg == "A1D":
        SA = ad["SA"]
        scen = dataclasses.replace(SA, ates_charge_max_mw=MW_CAP_A, ates_discharge_max_mw=MW_CAP_A,
                                   aquifer_thermal_conductivity_wmk=SA.aquifer_thermal_conductivity_wmk * s1)
        df = ad["A"]["run_ates_option_a_soc_driven"](sim, scen, dispatch_strategy="S0", trigger_pp=25,
                                                    permissive_discharge_pct=45.0, annual_charge_pct=30,
                                                    annual_discharge_pct=60, **kw)
        return df, {"a": [(None, scen)]}
    if cfg == "BWN110":
        wp, n, nw = fr.b_cfgs(MW_CAP_A, 0.0, s1, 1.0, temps=(110.0, 58.0))
        df = ad["B"]["run_ates_three_loops_soc_driven"](sim, wp, n, nw, trigger_pp=25, permissive_discharge_pct=45.0,
                                                       annual_charge_pct=15, annual_discharge_pct=75,
                                                       enable_ww_standby_correction=False, **kw)
        return df, {"wn": [("westpoort", wp), ("noord", n)]}
    wp, n, nw = fr.b_cfgs(MW_CAP_B, MW_CAP_B, s1, s2)
    df = ad["b_v6"](sim, wp, n, nw, annual_charge_pct=15, annual_discharge_pct=75,
                    enable_ww_standby_correction=False, **kw)
    return df, {"wn": [("westpoort", wp), ("noord", n)], "nw": [("nieuwwest", nw)]}


def _init():
    fr.EVALS = MY_EVALS
    fr.run = capped_run
    fr.CAP = {"A1D": MW_CAP_A}          # evaluate() uses `cfg in CAP` to pick the Option A zero family / columns
    fr.init()


def _job(a):
    fr.EVALS = MY_EVALS
    fr.run = capped_run
    fr.CAP = {"A1D": MW_CAP_A}
    return fr.evaluate(*a)


def points(cfg):
    rows = [json.loads(p.read_text()) for p in MY_EVALS.glob(f"{C.tagged(cfg)}__T__*.json")]
    return pd.DataFrame(rows)


def within(cfg, d):
    if not len(d):
        return d
    m = np.all([(d[f"{k}_eta_cycle_pct"] - t).abs() <= fr.TOL_PP for k, t in TARGET[cfg].items()], axis=0)
    return d[m & ((d.scale_1 != 1.0) | (d.scale_2 != 1.0))]


def next_runs(cfg, d):
    tg = TARGET[cfg]
    s1u, s2u = UNCAPPED[cfg] if cfg != "Bstd" else tuple(pd.read_csv(C.RES_FCORR / "bstd_fcorr_v3_R.csv").iloc[0][["scale_1", "scale_2"]])
    pts = d[(d.scale_1 != 1.0) | (d.scale_2 != 1.0)] if len(d) else d
    if cfg != "Bstd":
        k = list(tg)[0]
        if len(pts) == 0:
            return [(s1u, 1.0)]
        return [(fr.guess(d, f"{k}_eta_cycle_pct", "scale_1", tg[k]), 1.0)]
    if len(pts) < 3:
        return [(s1u, s2u), (round(s1u * 1.04, 4), s2u), (s1u, round(s2u * 1.04, 4))]
    dist = sum((pts[f"{k}_eta_cycle_pct"] - t) ** 2 for k, t in tg.items())
    near = pts.loc[dist.nsmallest(6).index]
    X = np.c_[np.ones(len(near)), np.log(near.scale_1), np.log(near.scale_2)]
    cw, *_ = np.linalg.lstsq(X, near.wn_eta_cycle_pct.values, rcond=None)
    cn, *_ = np.linalg.lstsq(X, near.nw_eta_cycle_pct.values, rcond=None)
    x = np.linalg.solve(np.array([[cw[1], cw[2]], [cn[1], cn[2]]]), np.array([tg["wn"] - cw[0], tg["nw"] - cn[0]]))
    s1, s2 = float(np.exp(x[0])), float(np.exp(x[1]))
    return [(round(s1, 4), round(s2, 4)), (round(s1 * 1.02, 4), round(s2, 4)), (round(s1, 4), round(s2 * 1.02, 4))]


def search(workers, rounds=8):
    MY_EVALS.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        first = [(c, "T", 1.0, 1.0) for c in TARGET]
        for r in ex.map(_job, first):
            print(f"  scale 1 {r['config']}: S_stor {r['s_stor_eur_yr']:,.0f}, discharged {r['discharged_mwh_yr']:,.0f} [{r['runtime_s']} s]", flush=True)
        for rnd in range(rounds):
            todo = []
            for cfg in TARGET:
                d = points(cfg)
                if len(within(cfg, d)):
                    continue
                todo += [(cfg, "T", round(a, 4), round(b, 4)) for a, b in next_runs(cfg, d)]
            print(f"round {rnd + 1}: {todo}", flush=True)
            if not todo:
                break
            for r in ex.map(_job, todo):
                eff = ", ".join(f"{k} {r[f'{k}_eta_cycle_pct']:.2f}" for k in TARGET[r["config"]])
                print(f"  {r['config']} ({r['scale_1']}, {r['scale_2']}): {eff}, S_stor {r['s_stor_eur_yr']:,.0f}, "
                      f"discharged {r['discharged_mwh_yr']:,.0f} [{r['runtime_s']} s]", flush=True)
    summarise()


# summary of the capped runs from the cache (no dispatch needed); search() calls this after the runs.
def summarise():
    """Table of the capped direct runs from the evaluation cache (no dispatch run)."""
    out = []
    for cfg in TARGET:
        d = points(cfg)
        ok = within(cfg, d)
        base = d[(d.scale_1 == 1.0) & (d.scale_2 == 1.0)].iloc[0]
        if not len(ok):
            out.append(dict(config=cfg, found=False)); continue
        dist = sum((ok[f"{k}_eta_cycle_pct"] - t).abs() for k, t in TARGET[cfg].items())
        r = ok.loc[dist.idxmin()]
        out.append(dict(config=cfg, found=True, mw_cap=MW_CAP_A if cfg != "Bstd" else MW_CAP_B, scale_1=r.scale_1,
                        scale_2=r.scale_2, **{f"{k}_eta_cycle_pct": r[f"{k}_eta_cycle_pct"] for k in TARGET[cfg]},
                        s_stor_base_capped=base.s_stor_eur_yr, s_stor_direct_capped=r.s_stor_eur_yr,
                        f_corr_capped=r.s_stor_eur_yr / base.s_stor_eur_yr,
                        discharged_direct_capped=r.discharged_mwh_yr, discharged_base_capped=base.discharged_mwh_yr,
                        flagged=bool(r.flagged)))
    pd.DataFrame(out).to_csv(HERE / "well_limit_capped_runs.csv", index=False)
    print(pd.DataFrame(out).T)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase"); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    hours() if a.phase == "hours" else (summarise() if a.phase == "summary" else search(a.workers))
