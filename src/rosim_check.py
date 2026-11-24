#!/usr/bin/env python3
"""Checks the six final ROSIM runs: yearly efficiency (reported, cleaned and with the dispatch reference temperature),
well spacing and cell size. Run by run_all.py."""
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
HERE = C.RES_ROSIM   # output folder of this script
import audit as au  # noqa: E402  (read-only helpers: read_eff_full, window_eff, YEAR_END)
from audit_lib import read_wells, read_settings, H_WEEK  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
FINAL = C.ROSIM_FINAL_COPY
# run: (ROSIM project, file name in data/rosim_final_copy, T_inj, T_ref, table years 6/7/8, table note)
# B-std: the pair of ROSIM runs at the return temperature of config.py (BSTD_T_RETURN_C). Table values (years
# 6/7/8) as supplied by the author; the script stops when a run differs by more than 0.02 points.
# B-std: injection 90 C and return config.BSTD_T_RETURN_C (58 C: window 32 K; 70 C only in the sensitivity).
BSTD_RUNS = {
    58.0: {"WN": ("Option B WN 58 90 def", "B_std_WN_58.txt", 90.0, 58.0, (42.90, 44.14, 45.18)),
           "NW": ("Option B NW 58 90 def", "B_std_NW_58.txt", 90.0, 58.0, (44.40, 45.89, 47.11))},
    70.0: {"WN": ("Option B WN 8 years adjusted v3 70m", "B_std_WN_70m.txt", 90.0, 70.0, (34.00, 35.57, 36.88)),
           "NW": ("Option B 8 years NW adjusted v3", "B_std_NW_70m.txt", 90.0, 70.0, (35.14, 36.90, 38.37))}}
RUNS = {
    "A1D":      ("Option A 1 Doublet 8 years changed parameters", "A1D.txt", 110.0, 58.0, (43.76, 45.16, 46.33)),
    "A2D":      ("Option A 2 doublets 8 years changed parameters", "A2D.txt", 110.0, 58.0, (48.12, 49.62, 50.92)),
    "A4D":      ("Option A 4 doublets 8 years changed parameters", "A4D.txt", 110.0, 58.0, (49.80, 51.43, 52.85)),
    "B-WN110":  ("Option B WN 110C 8 year adjusted", "B_WN110_125m.txt", 110.0, 58.0, (47.62, 48.68, 50.03)),
    "B-std WN": BSTD_RUNS[C.BSTD_T_RETURN_C]["WN"],
    "B-std NW": BSTD_RUNS[C.BSTD_T_RETURN_C]["NW"],
}
TOL = 0.02


# efficiency per year = positive / negative weekly energy over the 52 weeks ending at week 103, 155, ..., 415
# (simulation years 2..8). 'cleaned': weeks with |T| > 130 C or T < 5 C are ROSIM glitches and are linearly
# interpolated (for B-WN110 weeks 339 and 340 of the cold well). 'dispatch_reference': energy computed with
# the return temperature of the dispatch: 58 C for A and B-WN110, config.BSTD_T_RETURN_C for B-std (58 C).
def efficiencies(project, t_inj, t_ref):
    eff, per = au.read_eff_full(project)
    vox = read_wells(project)
    n = len(eff)
    tot_clean, ch, dis = np.zeros(n), np.zeros(n), np.zeros(n)
    bad_weeks = []
    for w, d in per.items():
        v = vox[w]
        rc = (v.density * v.cp).values[:len(d)]
        bad = ((d["T"].abs() > 130) | (d["T"] < 5) | (d["Tavg"] > 130) | (d["Tavg"] < 5)).values
        bad[0] = False
        Tavg, rcc = d["Tavg"].values.copy(), rc.copy()
        if bad.any():
            good = np.flatnonzero(~bad)
            Tavg[bad] = np.interp(np.flatnonzero(bad), good, Tavg[good])
            rcc[bad] = np.interp(np.flatnonzero(bad), good, rc[good])
            bad_weeks += [(w, int(eff.week.iloc[i])) for i in np.flatnonzero(bad)]
        E = d["E"].values.copy()
        E[bad] = -d["Q"].values[bad] * H_WEEK * rcc[bad] * (Tavg[bad] + 273.15) / 3.6e9
        tot_clean += E
        if w.startswith("warm"):
            Q = d["Q"].values
            ch += np.clip(Q, 0, None) * H_WEEK * rcc * (t_inj - t_ref) / 3.6e9
            dis += np.clip(-Q, 0, None) * H_WEEK * rcc * (Tavg - t_ref) / 3.6e9
    out = []
    for yi, end in enumerate(au.YEAR_END):
        sl = slice(end - 51, end + 1)
        out.append(dict(year=yi + 2, recomputed=au.window_eff(eff.total.values, end),
                        reported=float(eff.set_index("week").loc[end, "yearly"]),
                        cleaned=au.window_eff(tot_clean, end), dispatch_reference=dis[sl].sum() / ch[sl].sum() * 100))
    return pd.DataFrame(out), bad_weeks


def final_copy(fname):
    """Recompute from the user's copy in analysis/rosim_final (same tab-separated layout), if present."""
    p = FINAL / fname
    if not p.exists():
        return None
    L = [l.split("\t") for l in p.read_text(encoding="latin-1").splitlines()][2:]
    tot = np.array([float(r[2]) for r in L])
    return [au.window_eff(tot, e) for e in au.YEAR_END]


def geometry(project):
    s = read_settings(project)
    names = s["wells"]
    x = list(map(float, s["x_coordinate_well_head"].split(";")))
    y = list(map(float, s["y_coordinate_well_head"].split(";")))
    pos = dict(zip(names, zip(x, y)))
    pairs = [(f"warm{i}", f"cold{i}") for i in range(1, len(names) // 2 + 1)]
    wc = [math.dist(pos[a], pos[b]) for a, b in pairs if a in pos and b in pos]
    cold_T = []
    for w, prof in s["profiles"].items():
        if w.startswith("cold"):
            inj = prof[prof.flow > 0]
            cold_T.append(float(inj.temp.median()) if len(inj) else float("nan"))
    warm_T = [float(p[p.flow > 0].temp.median()) for w, p in s["profiles"].items() if w.startswith("warm") and (p.flow > 0).any()]
    return dict(project=project, dx_m=float(s["x_spacing__grid_cell_size_"]), dy_m=float(s["y_spacing__grid_cell_size_"]),
                warm_cold_m=";".join(f"{v:g}" for v in wc), warm_injection_T_median=";".join(f"{t:g}" for t in warm_T),
                cold_injection_T_median=";".join(f"{t:g}" for t in cold_T))


def main():
    rows, geo, stop = [], [], []
    for run, (proj, fname, t_inj, t_ref, table) in RUNS.items():
        e, bad = efficiencies(proj, t_inj, t_ref)
        fc = final_copy(fname)
        g = geometry(proj)
        g["run"] = run
        g["artefact_weeks"] = ";".join(f"{w}:{k}" for w, k in bad)
        geo.append(g)
        for _, r in e.iterrows():
            row = dict(run=run, project=proj, year=int(r.year), reported=r.reported, recomputed=r.recomputed,
                       cleaned=r.cleaned, dispatch_reference=r.dispatch_reference,
                       final_copy=(fc[int(r.year) - 2] if fc else np.nan), T_ref=t_ref)
            if r.year in (6, 7, 8):
                tv = table[int(r.year) - 6]
                row["table"] = tv
                # the table gives ROSIM-reported values, except B-WN110 year 7, which is the cleaned value
                cmp = r.cleaned if (run == "B-WN110" and r.year == 7) else r.recomputed
                row["diff_vs_table"] = cmp - tv
                if abs(cmp - tv) > TOL:
                    stop.append(f"{run} year {int(r.year)}: {cmp:.2f} against table {tv:.2f}")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "rosim_check.csv", index=False, float_format="%.3f")
    pd.DataFrame(geo).to_csv(HERE / "rosim_geometry.csv", index=False)
    print(df[df.year >= 6].round(3).to_string(index=False))
    print(pd.DataFrame(geo).to_string(index=False))
    print("rosim_final copies present:", sorted(p.name for p in FINAL.glob("*.txt")) if FINAL.exists() else "folder missing")
    if stop:
        print("STOP: differences > 0.02:\n  " + "\n  ".join(stop))
        sys.exit(1)
    print("all year 6-8 values within 0.02 of the table")


if __name__ == "__main__":
    main()
