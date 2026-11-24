#!/usr/bin/env python3
"""Temperature sensitivity, step 1: warm-well production temperatures of the six final ROSIM runs per year. Run by
run_all.py."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
HERE = C.RES_SENS / "temp_sensitivity"   # output folder of this script
HERE.mkdir(parents=True, exist_ok=True)
import audit as au  # noqa: E402
from audit_lib import read_wells, H_WEEK  # noqa: E402
from rosim_check import RUNS  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
T_SUPPLY = 90.0


def clean(d, rc):
    bad = ((d["T"].abs() > 130) | (d["T"] < 5) | (d["Tavg"] > 130) | (d["Tavg"] < 5)).values
    bad[0] = False
    T, Ta = d["T"].values.copy(), d["Tavg"].values.copy()
    if bad.any():
        good = np.flatnonzero(~bad)
        T[bad] = np.interp(np.flatnonzero(bad), good, T[good])
        Ta[bad] = np.interp(np.flatnonzero(bad), good, Ta[good])
    return T, Ta, int(bad.sum())


def main():
    rows = []
    for run, (proj, _f, t_inj, t_ret, _tab) in RUNS.items():
        eff, per = au.read_eff_full(proj)
        vox = read_wells(proj)
        warm = [w for w in per if w.startswith("warm")]
        n = len(eff)
        V = np.zeros(n); VT = np.zeros(n); VTa = np.zeros(n); E = np.zeros(n); nbad = 0
        for w in warm:
            d = per[w]
            rc = (vox[w].density * vox[w].cp).values[:len(d)]
            T, Ta, b = clean(d, rc)
            nbad += b
            v = np.clip(-d["Q"].values, 0, None) * H_WEEK          # m3 extracted per week
            V += v; VT += v * T; VTa += v * Ta
            E += v * rc * (T - t_ret) / 3.6e9                      # MWh relative to the dispatch return temperature
        wk_T = np.where(V > 0, VT / np.where(V > 0, V, 1), np.nan)
        wk_Ta = np.where(V > 0, VTa / np.where(V > 0, V, 1), np.nan)
        for yi, end in enumerate(au.YEAR_END):
            sl = slice(end - 51, end + 1)
            v, t, ta, e = V[sl], wk_T[sl], wk_Ta[sl], E[sl]
            p = v > 0
            Tm = (v * np.nan_to_num(t)).sum() / v.sum()
            Tam = (v * np.nan_to_num(ta)).sum() / v.sum()
            below = p & (t < T_SUPPLY)
            rows.append(dict(run=run, year=yi + 2, t_inj_dispatch=t_inj, t_ret_dispatch=t_ret, n_warm_wells=len(warm),
                             production_weeks=int(p.sum()), produced_volume_m3=float(v.sum()),
                             T_prod_vw=Tm, T_prod_min=float(np.nanmin(t[p])), T_prod_max=float(np.nanmax(t[p])),
                             Tavg_prod_vw=Tam,
                             share_weeks_below_90=float(below.sum() / p.sum()),
                             share_volume_below_90=float(v[below].sum() / v.sum()),
                             share_energy_below_90=float(e[below].sum() / e[p].sum()),
                             produced_energy_mwh_vs_Tret=float(e[p].sum()),
                             energy_below_90_mwh=float(e[below].sum()),
                             ratio=(Tm - t_ret) / (t_inj - t_ret), cleaned_weeks_all_years=nbad))
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "rosim_production_temps.csv", index=False, float_format="%.4f")
    t[t.year == 7].to_csv(HERE / "rosim_production_temps_y7.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
