#!/usr/bin/env python3
"""f_corr calibration of Option B standard (basis R or D, year 7) from the cached dispatch runs. Run by run_all.py."""
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
HERE = C.RES_FCORR   # output folder
import fcorr_runs as fr  # noqa: E402

MY_EVALS = C.EVALS_BSTD
# year-7 target efficiencies of the B-std runs at the configured return temperature (config.BSTD_T_RETURN_C), from
# rosim_check.csv: basis R = reported by ROSIM (2 decimals), basis D = dispatch reference (3 decimals).
_Y7 = pd.read_csv(C.RES_ROSIM / "rosim_check.csv").query("year == 7").set_index("run")
TARGET = {"wn": round(float(_Y7.loc["B-std WN", "recomputed"]), 2),            # basis R: ROSIM reported, year 7
          "nw": round(float(_Y7.loc["B-std NW", "recomputed"]), 2)}
TARGET_D = {"wn": round(float(_Y7.loc["B-std WN", "dispatch_reference"]), 3),  # basis D: dispatch reference
            "nw": round(float(_Y7.loc["B-std NW", "dispatch_reference"]), 3)}


def _init():
    fr.EVALS = MY_EVALS
    fr.init()


def _job(a):
    fr.EVALS = MY_EVALS
    return fr.evaluate("Bstd", "T", *a)


def points():
    rows = [json.loads(p.read_text()) for p in MY_EVALS.glob(f"{C.tagged('Bstd')}__T__*.json")]
    d = pd.DataFrame(rows)
    return d[(d.scale_1 != 1.0) | (d.scale_2 != 1.0)]


def within(d):
    m = np.all([(d[f"{k}_eta_cycle_pct"] - t).abs() <= fr.TOL_PP for k, t in TARGET.items()], axis=0)
    return d[m]


def next_guess(d, n=6):
    dist = sum((d[f"{k}_eta_cycle_pct"] - t) ** 2 for k, t in TARGET.items())
    near = d.loc[dist.nsmallest(n).index]
    X = np.c_[np.ones(len(near)), np.log(near.scale_1), np.log(near.scale_2)]
    cw, *_ = np.linalg.lstsq(X, near.wn_eta_cycle_pct.values, rcond=None)
    cn, *_ = np.linalg.lstsq(X, near.nw_eta_cycle_pct.values, rcond=None)
    A = np.array([[cw[1], cw[2]], [cn[1], cn[2]]])
    b = np.array([TARGET["wn"] - cw[0], TARGET["nw"] - cn[0]])
    x = np.linalg.solve(A, b)
    return float(np.exp(x[0])), float(np.exp(x[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--wn", type=float, default=None, help="target WN (default: rosim_check.csv, year 7, of the basis)")
    ap.add_argument("--nw", type=float, default=None)
    ap.add_argument("--basis", default="R", help="R = ROSIM reported (default), D = dispatch reference (rosim_check.csv)")
    a = ap.parse_args()
    t0 = dict(TARGET_D if a.basis == "D" else TARGET)
    TARGET.update(wn=t0["wn"] if a.wn is None else a.wn, nw=t0["nw"] if a.nw is None else a.nw)
    MY_EVALS.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        for rnd in range(a.rounds):
            d = points()
            ok = within(d)
            if len(ok):
                break
            s1, s2 = next_guess(d)
            cand = [(round(s1, 4), round(s2, 4)), (round(s1 * 1.03, 4), round(s2, 4)), (round(s1, 4), round(s2 * 1.03, 4))]
            print(f"round {rnd + 1}: runs {cand}", flush=True)
            for r in ex.map(_job, cand[:a.workers]):
                print(f"  ({r['scale_1']}, {r['scale_2']}): WN {r['wn_eta_cycle_pct']:.2f}, NW {r['nw_eta_cycle_pct']:.2f}, "
                      f"S_stor {r['s_stor_eur_yr']:,.0f}, discharged {r['discharged_mwh_yr']:,.0f} MWh/yr, "
                      f"residual {r['balance_residual_pct']:.2f} % [{r['runtime_s']} s]", flush=True)
    d = points()
    ok = within(d)
    base = json.loads((MY_EVALS / f"{fr.tag('Bstd', 'T', 1.0, 1.0)}.json").read_text())
    if not len(ok):
        print("no run within 0.3 pp of both targets")
        return
    dist = sum((ok[f"{k}_eta_cycle_pct"] - t).abs() for k, t in TARGET.items())
    r = ok.loc[dist.idxmin()].to_dict()
    out = dict(config="Bstd", setting="T", basis=a.basis, year=7, target_wn=TARGET["wn"], target_nw=TARGET["nw"],
               scale_1=r["scale_1"], scale_2=r["scale_2"], wn_eta_cycle_pct=r["wn_eta_cycle_pct"],
               nw_eta_cycle_pct=r["nw_eta_cycle_pct"], s_stor_direct_eur_yr=r["s_stor_eur_yr"],
               s_stor_base_eur_yr=base["s_stor_eur_yr"], f_corr=r["s_stor_eur_yr"] / base["s_stor_eur_yr"],
               discharged_mwh_yr=r["discharged_mwh_yr"], balance_residual_pct=r["balance_residual_pct"], flagged=r["flagged"])
    pd.DataFrame([out]).to_csv(HERE / f"bstd_fcorr_v3_{a.basis}.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
