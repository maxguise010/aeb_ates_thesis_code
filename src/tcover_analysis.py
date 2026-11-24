#!/usr/bin/env python3
"""T_cover sensitivity: bridged attribution tier and NPV for T_cover from 1 to 12 hours. Run by run_all.py."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
HERE = C.RES_SCEN   # output folder
import final_config as FC  # noqa: E402
import tank_capex as TC  # noqa: E402
import final_numbers as FN  # noqa: E402

YEARS = [2023, 2024, 2025]
OPT = ["A1D", "BWN110", "Bstd"]
FAM = {"A1D": "none_ates_a", "BWN110": "none_ates", "Bstd": "none_ates", "TTES5k": "none_tank"}
PMAX = {"A1D": 13.3, "BWN110": 13.3, "Bstd": 13.28}
# T_cover = release_power_horizon_h: the existing buffers may only drop to the lower floor if the ATES can deliver their
# reserve R (30.9-61.3 MWh) within T_cover hours (P x T_cover >= R). Thesis value 3 h. From 4.61 h
# (61.3 / 13.3 MW) the power test passes in every hour and the run equals A1h.
GRID = [1.0, 2.0, 3.0, 3.25, 3.5, 4.0, 6.0, 8.0, 12.0]
AF = FC.annuity_factor()


def load(p):
    r = json.loads(Path(p).read_text())
    for k in [k for k in r if k.endswith("_py") or k == "per_year"]:
        r[k] = {int(y): v for y, v in r[k].items()}
    return r


def sav_sym(r):
    return np.array([r["per_year"][y] + (r["bec_sym_py"][y] - r["bec_asym_py"][y]) / 1e3 for y in YEARS])


def run_for(o, t):
    if t == 3.0:
        return load(C.AST_CACHE / f"{C.tagged(o)}__relaxed__A1p.json"), "ates_same_treatment A1p"
    if t >= 6.0:
        return load(C.AST_CACHE / f"{C.tagged(o)}__relaxed__A1h.json"), "= A1h (power condition true in every hour)"
    return load(C.TCOVER_CACHE / f"{C.tagged(o)}__relaxed__T{t:g}.json"), "tcover_runs.py"


def release_stats(o, t, r):
    if "release_share" in r:
        return r["release_share"], r["power_ok_share"]
    tag = "A1p" if t == 3.0 else "A1h"
    fp = C.FRAMES_AST / f"{C.tagged(o)}__relaxed__{tag}.pkl"
    if not fp.exists():   # the hourly frames contain AEB data and are not included: same two shares, derived table
        s = pd.read_csv(C.AST_RELEASE_STATS).set_index(["option", "t_cover_h"]).loc[(C.tagged(o), t)]
        return float(s.release_share), float(s.power_ok_share)
    df = pd.read_pickle(fp)
    return float((df["buffer_release_flag"] > 0.5).mean()), float((PMAX[o] * t >= df["backup_reserve_mwh"]).mean())


def main():
    I = FN.load_inputs()
    f, share = I["f"]["R"], I["share"]
    zr = {fam: sav_sym(load(C.AST_CACHE / f"{fam}__relaxed__cal.json")) for fam in ("none_ates", "none_ates_a", "none_tank")}
    zc = sav_sym(load(C.AST_CACHE / "none_ates__constrained__cal.json"))
    s_flex = float((zr["none_ates"] - zc).mean())
    c_cal = {o: float((sav_sym(load(C.AST_CACHE / f"{C.tagged(o)}__relaxed__cal.json")) - zr[FAM[o]]).mean()) for o in OPT}
    # identity check T = 12 (run) vs A1h (cache)
    p12 = C.TCOVER_CACHE / "A1D__relaxed__T12.json"
    ident = (abs(load(p12)["saving_keur_yr"] - load(C.AST_CACHE / "A1D__relaxed__A1h.json")["saving_keur_yr"])
             if p12.exists() else np.nan)
    rows = []
    for o in OPT:
        a = FC.ATES[o]; cx, ox = a["capex_meur"][1], a["opex_keur"]
        for t in GRID:
            r, src = run_for(o, t)
            s = float((sav_sym(r) - zr[FAM[o]]).mean())
            rel, pok = release_stats(o, t, r)
            sa = share[o] * s_flex + f[o] * s
            rows.append(dict(option=o, t_cover_h=t, source=src, power_ok_share=pok, release_share=rel,
                             release_hours_yr=rel * 8760, S_stor_relaxed_sym=s, freed_value_keur_yr=s - c_cal[o],
                             freed_value_cal_keur_yr=f[o] * (s - c_cal[o]), S_attr_bridged=sa,
                             npv_bridged_mid=-cx + (sa - ox) * AF / 1000, breakeven_capex_bridged=(sa - ox) * AF / 1000))
    # tank (no T_cover in the tank model)
    tt = float((sav_sym(load(C.AST_CACHE / "TTES5k__relaxed__cal.json")) - zr["none_tank"]).mean())
    cx = TC.CASES["B"]["capex_meur"]; ox = TC.opex_keur(cx)
    sa_t = share["TTES5k"] * s_flex + tt
    for t in GRID:
        rows.append(dict(option="TTES5k", t_cover_h=t, source="tank model has no T_cover (same for every T)",
                         power_ok_share=np.nan, release_share=np.nan, release_hours_yr=np.nan, S_stor_relaxed_sym=tt,
                         freed_value_keur_yr=0.0, freed_value_cal_keur_yr=0.0, S_attr_bridged=sa_t,
                         npv_bridged_mid=-cx + (sa_t - ox) * AF / 1000,
                         breakeven_capex_bridged=sa_t * AF / 1000 / (1 + FC.TANK_OPEX_SHARE * AF)))
    T = pd.DataFrame(rows)
    ref = T[T.t_cover_h == 3.0].set_index("option")
    T["S_attr_bridged_change_vs_3h_pct"] = [100 * (r.S_attr_bridged / ref.loc[r.option, "S_attr_bridged"] - 1) for r in T.itertuples()]
    T["npv_bridged_change_vs_3h"] = [r.npv_bridged_mid - ref.loc[r.option, "npv_bridged_mid"] for r in T.itertuples()]
    T.to_csv(HERE / "tcover_sweep.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"S_flex calibrated (Option B family, symmetric) = {s_flex:.2f}; tank S_stor (cal, sym) = {tt:.2f}; "
          f"shares {dict((k, round(v, 4)) for k, v in share.items())}")
    print(f"identity T = 12 h run vs A1h cache (A1D): {ident:.4f} k EUR/yr")
    print(T.drop(columns=["source"]).round(3).to_string(index=False))
    json.dump(dict(s_flex_cal=s_flex, tank_s_stor_cal=tt, identity_T12_vs_A1h_keur=ident,
                   threshold_median_h=44.25 / 13.3, threshold_all_hours_h=61.32 / 13.3),
              open(HERE / "tcover_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
