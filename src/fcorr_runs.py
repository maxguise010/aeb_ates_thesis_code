#!/usr/bin/env python3
"""f_corr calibration: scales the ground-loss parameter of the dispatch model until its efficiency equals the ROSIM
efficiency. Run by run_all.py."""
import argparse, dataclasses, json, math, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
HERE = C.RES_FCORR          # output folder (direct_runs.csv)
EVALS = C.FCORR_EVALS
OLD = C.FCORR_VALIDATION     # evaluation CSVs of the thesis runs (seeds of the search)
AST = C.AST_CACHE
# TOL_PP = 0.3 percentage point: this close must the dispatch efficiency be to the ROSIM target for a 'direct run'.
CONST, ZERO_REF, TOL_PP = 0.001157, 1_849_658, 0.3
YEARS = [2023, 2024, 2025]
PRICE = "Prijs_Gemiddelde_EPEX_Spot"
COMMON = dict(demand_correction_factor=1.038, enable_bec_summer_cap=True, enable_min_runtime=True,
              min_runtime_hours=18, min_downtime_hours=2)
CAL = dict(reserve_max_fraction=0.35, reserve_horizon_h=24, small_soc_max=0.68,
           daily_charge_rank_n=3, daily_discharge_rank_n=3)
CAP = {"A1D": 13.3, "A2D": 26.6, "A4D": 53.2}
DOUBLETS = {"A1D": ["a"], "A2D": ["a"], "A4D": ["a"], "BWN110": ["wn"], "Bstd": ["wn", "nw"]}
W = {}


def init():
    import scenario_runner as sr
    from data_pipeline import build_sim_df_tank
    from config import historical_cols, derivation_factors
    from run_short_window import bec_terms
    ad = sr._adapter()
    sim = build_sim_df_tank()[0]
    ww30 = sim[historical_cols["WW30"]].fillna(0.0)
    aftap = ww30.clip(upper=sim["WW30_aftap_cap_mw"])
    loss = (sim[historical_cols["WW10"]].fillna(0.0) * derivation_factors["WW10"]
            + aftap * derivation_factors["WW30_aftap"]
            + (ww30 - aftap).clip(lower=0.0) * derivation_factors["WW30_reductie"]
            + sim[historical_cols["WW01"]].fillna(0.0) * derivation_factors["WW01"]
            + sim[historical_cols["WW00"]].fillna(0.0) * derivation_factors["WW00"]
            + sim[historical_cols["WW02"]].fillna(0.0) * derivation_factors["WW02"])
    W.update(ad=ad, sim=sim, hist=(loss * sim[PRICE]).groupby(sim.index.year).sum(), bec_terms=bec_terms)


def zero_sym(family):
    """Calibrated zero run with the symmetric BEC term, EUR/yr per year (analyse_ates.py:47-49)."""
    r = json.loads((AST / f"{family}__relaxed__cal.json").read_text())
    return {y: r["per_year"][str(y)] * 1e3 + (r["bec_sym_py"][str(y)] - r["bec_asym_py"][str(y)]) for y in YEARS}


def b_cfgs(wn_mw, nw_mw, s_wn, s_nw, temps=None):
    """fcorr_validation_followup.py:101-111."""
    ad = W["ad"]
    SWP, SN, SNW = ad["SWP"], ad["SN"], ad["SNW"]
    s = wn_mw / (SWP.ates_charge_max_mw + SN.ates_charge_max_mw)
    extra = dict(ates_injection_temp_c=temps[0], ates_cold_temp_c=temps[1]) if temps else {}
    k = lambda cfg, sc: cfg.aquifer_thermal_conductivity_wmk * sc
    wp = dataclasses.replace(SWP, ates_charge_max_mw=SWP.ates_charge_max_mw * s, ates_discharge_max_mw=SWP.ates_charge_max_mw * s,
                             aquifer_thermal_conductivity_wmk=k(SWP, s_wn), **extra)
    n = dataclasses.replace(SN, ates_charge_max_mw=SN.ates_charge_max_mw * s, ates_discharge_max_mw=SN.ates_charge_max_mw * s,
                            aquifer_thermal_conductivity_wmk=k(SN, s_wn), **extra)
    nw = dataclasses.replace(SNW, ates_charge_max_mw=nw_mw, ates_discharge_max_mw=nw_mw,
                             aquifer_thermal_conductivity_wmk=k(SNW, s_nw))
    return wp, n, nw


def run(cfg, setting, s1, s2):
    ad, sim = W["ad"], W["sim"]
    kw = dict(COMMON, **(CAL if setting == "C" else {}))
    if cfg in CAP:
        SA = ad["SA"]
        scen = dataclasses.replace(SA, ates_charge_max_mw=CAP[cfg], ates_discharge_max_mw=CAP[cfg],
                                   aquifer_thermal_conductivity_wmk=SA.aquifer_thermal_conductivity_wmk * s1)
        df = ad["A"]["run_ates_option_a_soc_driven"](sim, scen, dispatch_strategy="S0", trigger_pp=25,
                                                    permissive_discharge_pct=45.0, annual_charge_pct=30,
                                                    annual_discharge_pct=60, **kw)
        return df, {"a": [(None, scen)]}
    if cfg == "BWN110":
        wp, n, nw = b_cfgs(13.3, 0.0, s1, 1.0, temps=(110.0, 58.0))
        df = ad["B"]["run_ates_three_loops_soc_driven"](sim, wp, n, nw, trigger_pp=25, permissive_discharge_pct=45.0,
                                                       annual_charge_pct=15, annual_discharge_pct=75,
                                                       enable_ww_standby_correction=False, **kw)
        return df, {"wn": [("westpoort", wp), ("noord", n)]}
    wp, n, nw = b_cfgs(6.64, 6.64, s1, s2)
    df = ad["b_v6"](sim, wp, n, nw, annual_charge_pct=15, annual_discharge_pct=75,
                    enable_ww_standby_correction=False, **kw)
    return df, {"wn": [("westpoort", wp), ("noord", n)], "nw": [("nieuwwest", nw)]}


def tag(cfg, setting, s1, s2):
    return f"{C.tagged(cfg)}__{setting}__{s1:.5f}__{s2:.5f}"   # B-std at 58 C: Bstd_T58


# one calibration run: the same dispatch with the ground-loss knob (thermal conductivity x scale). Cycle efficiency
# per doublet = discharged / (charged - heat still in the ground at the end). f_corr = S_stor of the run that
# hits the ROSIM target / S_stor at scale 1. Each run is cached (one JSON), so repeating costs no dispatch.
def evaluate(cfg, setting, s1, s2=1.0):
    f = EVALS / f"{tag(cfg, setting, s1, s2)}.json"
    if f.exists():
        return json.loads(f.read_text())
    if not W:
        init()
    t0 = time.time()
    df, doublets = run(cfg, setting, s1, s2)
    hist = W["hist"]
    per = {y: float(hist[y] - df.loc[df.index.year == y, "total_system_cost_eur_corrected"].sum()) for y in YEARS}
    r = dict(config=cfg, setting=setting, scale_1=s1, scale_2=s2, saving_eur_yr=float(np.mean(list(per.values()))))
    if setting == "T":
        r["s_stor_eur_yr"] = r["saving_eur_yr"] - ZERO_REF
    else:
        b_asym, b_sym = W["bec_terms"](df)
        z = zero_sym("none_ates_a" if cfg in CAP else "none_ates")
        r["s_stor_eur_yr"] = float(np.mean([per[y] + b_sym[y] - b_asym[y] - z[y] for y in YEARS]))
    CH = DIS = EEND = 0.0
    zero_vol = 0
    for d, loops in doublets.items():
        ch = dis = e_end = 0.0
        for pre, c in loops:
            if pre is None:
                cc, s = df["ates_charge_mw"].values, df["ates_discharge_mw"].values
                vol, temp = df["ates_volume_m3"].values, df["ates_aquifer_temp_c"].values
            else:
                cc, s = df[f"ates_{pre}_charge_mw"].values, df[f"ates_{pre}_discharge_mw"].values
                vol = df[f"ates_{pre}_soc"].values * c.ates_capacity_m3
                temp = df[f"ates_{pre}_temp_c"].values
            ch += cc.sum(); dis += s.sum()
            e_end += vol[-1] * CONST * (temp[-1] - c.ates_cold_temp_c)
            zero_vol += int(((vol <= 1e-6) & (s > 1e-6)).sum())
        r[f"{d}_eta_cycle_pct"] = float(dis / (ch - e_end) * 100) if ch - e_end > 0 else 0.0
        r[f"{d}_eta_disp_pct"] = float(dis / ch * 100) if ch else 0.0
        CH += ch; DIS += dis; EEND += e_end
    loss = df["ates_ground_loss_mw"].sum()
    resid = CH - DIS - loss - EEND
    r.update(charged_mwh_yr=float(CH / 3), discharged_mwh_yr=float(DIS / 3), e_end_mwh=float(EEND),
             eta_cycle_pct=float(DIS / (CH - EEND) * 100) if CH - EEND > 0 else 0.0,
             balance_residual_pct=float(resid / CH * 100) if CH else 0.0, hours_discharging_at_zero_volume=zero_vol,
             flagged=bool(zero_vol > 0 or abs(resid) > 0.01 * CH), runtime_s=round(time.time() - t0))
    EVALS.mkdir(exist_ok=True)
    f.write_text(json.dumps(r))
    return r


def _job(args):
    return evaluate(*args)


def all_evals(cfg, setting):
    rows = [json.loads(p.read_text()) for p in EVALS.glob(f"{C.tagged(cfg)}__{setting}__*.json")]
    return pd.DataFrame(rows)


def old_points(cfg):
    """Scales and efficiencies of the old runs, only used to seed the first guesses."""
    if cfg in CAP:
        d = pd.read_csv(OLD / f"A-{cfg[1:]}_evaluations.csv").rename(columns={"ground_loss_scale": "scale_1", "eta_cycle_pct": "a_eta_cycle_pct"})
        d["scale_2"] = 1.0
    else:
        d = pd.read_csv(OLD / f"followup_{'B-WN110' if cfg == 'BWN110' else 'B-std'}_evaluations.csv")
    return d[~d.flagged_artefact.astype(bool)]


def guess(pts, eta_col, scale_col, target):
    """Scale for the target efficiency by interpolation of eta against log(scale) over the runs so far."""
    p = pts[pts[scale_col] > 0].sort_values(scale_col)
    p = p.groupby(scale_col, as_index=False)[eta_col].mean()
    x, y = np.log(p[scale_col].values), p[eta_col].values
    if len(x) == 1:
        return float(p[scale_col].iloc[0] * (y[0] / target) ** 1.5)
    o = np.argsort(-y)                       # eta decreases with scale; interpolate in eta
    xs, ys = x[o], y[o]
    if target >= ys[0]:
        i = 0
    elif target <= ys[-1]:
        i = len(ys) - 2
    else:
        i = int(np.searchsorted(-ys, -target) - 1)
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
    xt = x0 + (target - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
    return float(round(math.exp(min(max(xt, math.log(0.2)), math.log(80))), 4))


def solve(jobs, workers, max_rounds=8, allow_flagged=()):
    """jobs: list of (cfg, setting, targets dict). Returns {(cfg, setting, key): direct-run dict}.
    allow_flagged: configurations whose flagged runs may be used (B-std: every run in the new target range has an
    energy-balance residual of 1 to 3 %, above the 1 % flag of the old scripts)."""
    pool = ProcessPoolExecutor(max_workers=workers, initializer=init)
    done = {}
    try:
        for rnd in range(max_rounds):
            todo = set()
            for cfg, setting, key, targets in jobs:
                if (cfg, setting, key) in done:
                    continue
                ev = all_evals(cfg, setting)
                ok = ev[~ev.flagged | (cfg in allow_flagged)] if len(ev) else ev
                if len(ok):
                    dist = sum((ok[f"{d}_eta_cycle_pct"] - t).abs() for d, t in targets.items())
                    within = ok[np.all([(ok[f"{d}_eta_cycle_pct"] - t).abs() <= TOL_PP for d, t in targets.items()], axis=0)]
                    if len(within):
                        done[(cfg, setting, key)] = within.loc[dist[within.index].idxmin()].to_dict()
                        continue
                seed = pd.concat([old_points(cfg)[[c for c in old_points(cfg).columns if c.endswith("_eta_cycle_pct") or c.startswith("scale_")]],
                                  ok]) if len(ok) else old_points(cfg)
                if len(ok) >= 2:
                    seed = ok                  # own runs only once there are enough
                if cfg == "Bstd":
                    s2_now = float(np.median(ok.scale_2)) if len(ok) else 5.1
                    s1_now = float(np.median(ok.scale_1)) if len(ok) else 3.0
                    near1 = seed[(seed.scale_2 - s2_now).abs() <= 0.25 * s2_now] if len(seed) > 3 else seed
                    near2 = seed[(seed.scale_1 - s1_now).abs() <= 0.25 * s1_now] if len(seed) > 3 else seed
                    s1 = guess(near1 if len(near1) >= 2 else seed, "wn_eta_cycle_pct", "scale_1", targets["wn"])
                    s2 = guess(near2 if len(near2) >= 2 else seed, "nw_eta_cycle_pct", "scale_2", targets["nw"])
                else:
                    d = DOUBLETS[cfg][0]
                    s1, s2 = guess(seed, f"{d}_eta_cycle_pct", "scale_1", targets[d]), 1.0
                todo.add((cfg, setting, round(s1, 4), round(s2, 4)))
            left = [j for j in jobs if (j[0], j[1], j[2]) not in done]
            print(f"round {rnd + 1}: {len(done)} targets met, {len(left)} open, {len(todo)} runs", flush=True)
            if not todo:
                break
            for fu in as_completed([pool.submit(_job, t) for t in sorted(todo)]):
                r = fu.result()
                eff = ", ".join(f"{d} {r[f'{d}_eta_cycle_pct']:.2f}" for d in DOUBLETS[r["config"]])
                print(f"  {r['config']:7s} {r['setting']} ({r['scale_1']:.4f}, {r['scale_2']:.4f}): {eff}, "
                      f"S_stor {r['s_stor_eur_yr']:,.0f}{' FLAGGED' if r['flagged'] else ''} [{r['runtime_s']} s]", flush=True)
    finally:
        pool.shutdown()
    return done


REPRO = {"A1D": [(1.0, 1.0), (5.43791, 1.0)], "A2D": [(1.0, 1.0), (9.76402, 1.0)], "A4D": [(1.0, 1.0), (15.61655, 1.0)],
         "BWN110": [(1.0, 1.0), (4.36313, 1.0)], "Bstd": [(1.0, 1.0), (2.97323, 5.10597)]}


def reproduce(workers):
    """Rerun the base and the old direct run of each configuration at the exact old scales (setting T), and the base
    of setting C."""
    exact = {}
    for cfg in REPRO:
        d = old_points(cfg) if cfg in CAP else None
        old = (pd.read_csv(OLD / f"A-{cfg[1:]}_evaluations.csv") if cfg in CAP else
               pd.read_csv(OLD / f"followup_{'B-WN110' if cfg == 'BWN110' else 'B-std'}_evaluations.csv"))
        sc1 = "ground_loss_scale" if cfg in CAP else "scale_1"
        for s1, s2 in REPRO[cfg]:
            m = old[(old[sc1] - s1).abs() < 1e-3]
            if cfg == "Bstd":
                m = m[(m.scale_2 - s2).abs() < 1e-3]
            exact[(cfg, s1, s2)] = (float(m[sc1].iloc[0]), float(m["scale_2"].iloc[0]) if cfg not in CAP else 1.0)
    jobs = [(cfg, "T", *exact[(cfg, s1, s2)]) for cfg in REPRO for s1, s2 in REPRO[cfg]]
    jobs += [(cfg, "C", 1.0, 1.0) for cfg in REPRO]
    with ProcessPoolExecutor(max_workers=workers, initializer=init) as ex:
        for fu in as_completed([ex.submit(_job, j) for j in jobs]):
            r = fu.result()
            print(f"  {r['config']} {r['setting']} ({r['scale_1']}, {r['scale_2']}): eta {r['eta_cycle_pct']:.3f}, "
                  f"S_stor {r['s_stor_eur_yr']:,.1f} [{r['runtime_s']} s]", flush=True)
    return exact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--allow-flagged", default="", help="comma-separated configurations whose flagged runs may be used")
    a = ap.parse_args()
    EVALS.mkdir(exist_ok=True)
    if a.phase == "reproduce":
        reproduce(a.workers)
    else:
        from targets import JOBS
        res = solve(JOBS, a.workers, allow_flagged=tuple(x for x in a.allow_flagged.split(",") if x))
        out = [dict(config=c, setting=s, target=k, **{kk: v for kk, v in r.items() if kk not in ("config", "setting")})
               for (c, s, k), r in res.items()]
        pd.DataFrame(out).to_csv(HERE / "direct_runs.csv", index=False)
        print(f"{len(out)} of {len(JOBS)} targets met")
