"""Detail numbers of Option B standard at 90/58 C, read from the outputs of the chain and added to the locked-numbers
files. Run by run_all.py."""
import inspect
import json
import sys

import numpy as np
import pandas as pd

import config as C
import final_config as FC
import final_numbers as FN
import audit as au
import rosim_check as RCK

sys.stdout.reconfigure(encoding="utf-8")
CHAPTER = "B-std at 90/58 C: details"
SECTION = "## B-std at 90/58 C: details"
P = "B-std 58 C | "
ROWS = []


def put(key, value, unit, source, decimals=2):
    line = inspect.currentframe().f_back.f_lineno
    v = value if isinstance(value, str) else round(float(value), decimals)
    ROWS.append(dict(chapter=CHAPTER, key=P + key, value=v, unit=unit, script=f"src/bstd_numbers.py:{line}", source=source,
                     author_estimate="", old_value=None, old_location=""))


def rosim():
    rc = pd.read_csv(C.RES_ROSIM / "rosim_check.csv")
    pt = pd.read_csv(C.RES_SENS / "temp_sensitivity" / "rosim_production_temps_y7.csv").set_index("run")
    t_inj, t_ret = C.BSTD_T_INJ_C, C.BSTD_T_RETURN_C
    for lp in ("WN", "NW"):
        r = rc[rc.run == f"B-std {lp}"].set_index("year")
        eff, per = au.read_eff_full(RCK.RUNS[f"B-std {lp}"][0])
        for yr, end in zip(range(2, 9), au.YEAR_END):
            if yr != 7:                                             # year 7: final_numbers.py
                put(f"B standard {lp} efficiency year {yr} (basis R)", au.window_eff(eff.total.values, end), "%",
                    "ROSIM export, as rosim_check.py ('recomputed', full precision)")
        put(f"{lp} efficiency year 7 (dispatch reference 58 C, basis D)", r.loc[7, "dispatch_reference"], "%",
            "results/rosim/rosim_check.csv (week-average T, rho cp)")
        warm = per[next(w for w in per if w.startswith("warm"))]
        cold = per[next(w for w in per if w.startswith("cold"))]
        w = warm.iloc[363 - 51:363 + 1]
        q, T = w["Q"].values, w["T"].values
        pr, ij = q < 0, q > 0
        put(f"{lp} efficiency year 7, fixed 58 C reference with end-of-week T (definition of the independent check)",
            100 * (-q[pr] * (T[pr] - t_ret)).sum() / (q[ij] * (t_inj - t_ret)).sum(), "%", "ROSIM export (column T)")
        put(f"{lp} warm-well production temperature year 7, volume-weighted", pt.loc[f"B-std {lp}", "T_prod_vw"], "C",
            "results/sensitivity/temp_sensitivity/rosim_production_temps_y7.csv", 1)
        put(f"{lp} injected volume per year (years 3-8)", q[ij].sum() * 168.0, "m3/yr", "ROSIM export, year 7", 0)
        put(f"{lp} extracted volume per year (years 3-8)", -q[pr].sum() * 168.0, "m3/yr", "ROSIM export, year 7", 0)
        put(f"{lp} maximum cold-well temperature", cold["T"].max(), "C", "ROSIM export, all weeks")
        put(f"{lp} maximum warm-well temperature", warm["T"].max(), "C", "ROSIM export, all weeks")
    put("return (cold-well) temperature", t_ret, "C", "src/config.py BSTD_T_RETURN_C (AEB mean return 2023-2025)", 0)
    put("temperature window (90 C injection)", t_inj - t_ret, "K", "src/config.py", 0)
    for b in ("R", "D"):
        f = pd.read_csv(C.RES_FCORR / f"bstd_fcorr_v3_{b}.csv").iloc[0]
        put(f"calibration scales WN / NW (basis {b})", f"{f.scale_1:.4f} / {f.scale_2:.4f}", "-", f"results/fcorr/bstd_fcorr_v3_{b}.csv")


def economics():
    I = FN.load_inputs()
    a = FC.ATES["Bstd"]
    put("storage effect before calibration (S_stor, scale 1)", I["base"]["Bstd"] / 1000, "k EUR/yr",
        "data/derived/fcorr_evals (scale-1 run)")
    put("bridged share of S_flex (R = 2 h)", I["share"]["Bstd"], "-", "data/derived/restart_split (s_flex_shares)", 4)
    n = pd.read_csv(C.RES_FINAL / "step4_npv_ates_basisR.csv").query("option == 'Bstd' and f_set == 'new'").iloc[0]
    for case, i in (("low", 0), ("high", 2)):
        put(f"NPV base, CAPEX {case} ({a['capex_meur'][i]} M EUR)", FN.npv(a["capex_meur"][i], a["opex_keur"], n.S_attr_base), "M EUR",
            "results/final_numbers/step4_npv_ates_basisR.csv (S_attr base)")
    net = n.S_attr_base - a["opex_keur"]
    put("simple payback (CAPEX / (base S_attr - OPEX))", a["capex_meur"][1] * 1000 / net if net > 0 else np.inf, "yr",
        "step4_npv_ates_basisR.csv", 1)
    d = pd.read_csv(C.RES_FINAL / "step4_npv_ates_basisD.csv").query("option == 'Bstd' and f_set == 'new'").iloc[0]
    put("NPV base (basis D)", d.npv_base, "M EUR", "results/final_numbers/step4_npv_ates_basisD.csv")
    l = pd.read_csv(C.RES_FINAL / "step1_lcoh_basisD.csv").set_index("option").loc["Bstd"]
    put("LCOH (basis D)", l.lcoh_discounted, "EUR/MWh", "results/final_numbers/step1_lcoh_basisD.csv", 0)


def scenarios():
    s = pd.read_csv(C.RES_SCEN / "scenarios_old_vs_new.csv").query("option == 'Bstd'")
    for r in s.itertuples():
        put(f"rank in {r.scenario_label} (f_corr R, tank B; 1 = best of six)", r.new_rank, "-",
            "results/scenarios/scenarios_old_vs_new.csv", 0)
    put("scenario range of the base NPV (13 scenarios), lowest", s.new_npv.min(), "M EUR", "results/scenarios/scenarios_old_vs_new.csv")
    put("scenario range of the base NPV (13 scenarios), highest", s.new_npv.max(), "M EUR", "results/scenarios/scenarios_old_vs_new.csv")


def sensitivities():
    t = pd.read_csv(C.RES_SENS / "temp_sensitivity" / "step2_derated.csv").query("option == 'Bstd' and case == 'derated'").iloc[0]
    src = "results/sensitivity/temp_sensitivity/step2_derated.csv"
    put("derated power: cap WN + NW", t.cap_mw, "MW", src)
    put("derated power: storage effect before calibration", t.s_stor_uncal_keur, "k EUR/yr", src, 1)
    put("derated power: base NPV", t.npv_base_meur, "M EUR", src)
    put("derated power: change in base NPV", t.npv_change_meur, "M EUR", src)
    put("derated power: LCOH", t.lcoh_eur_mwh, "EUR/MWh", src, 0)
    hp = pd.read_csv(C.RES_SENS / "temp_sensitivity" / "step3_heat_pump.csv")
    src = "results/sensitivity/temp_sensitivity/step3_heat_pump.csv"
    cal = hp[(hp.loop == "NW") & hp.run.str.startswith("calibrated") & (hp.ates_heat_counted_above_c == 58.0)]
    th = cal[cal.temperature_basis.str.startswith("weekly") & cal.cop_basis.str.startswith("COP 4.38")].iloc[0]
    put("heat pump NW, thesis method (weekly T, COP 4.38), calibrated run: electricity", th.W_el_mwh_yr, "MWh_el/yr", src, 0)
    put("heat pump NW, thesis method, calibrated run: cost", th.cost_keur_yr, "k EUR/yr", src, 1)
    put("heat pump NW, thesis method, calibrated run: base NPV with this cost", th.npv_base_meur, "M EUR", src)
    put("heat pump NW cost, calibrated run, lowest variant (heat above 58 C)", cal.cost_keur_yr.min(), "k EUR/yr", src, 1)
    put("heat pump NW cost, calibrated run, highest variant (heat above 58 C)", cal.cost_keur_yr.max(), "k EUR/yr", src, 1)
    put("heat-pump electricity in the locked OPEX (unchanged)", 378.0 - 200.0 - 124.0, "k EUR/yr",
        "thesis Table 7.4 (OPEX 378 = 200 + 54 + 124, final_config.ATES)", 0)
    wl = pd.read_csv(C.RES_SCEN / "well_limit_hours.csv").set_index("run")
    for lp in ("WN", "NW"):
        r = wl.loc[f"B-std {lp}"]
        put(f"well limit {lp}: flow per MW (builder factor)", r.k_m3h_per_mw, "m3/h per MW", "results/scenarios/well_limit_hours.csv")
        put(f"well limit {lp}: dispatch peak flow", r.peak_ext_m3h, "m3/h", "results/scenarios/well_limit_hours.csv", 0)
        put(f"well limit {lp}: 179 m3/h as MW", r.cap_mw_per_doublet, "MW", "results/scenarios/well_limit_hours.csv")
    tc = pd.read_csv(C.RES_SCEN / "tcover_sweep.csv").query("option == 'Bstd' and t_cover_h == 8.0").iloc[0]
    put("T_cover 8 h: bridged NPV B-std (A1p, calibrated buffers)", tc.npv_bridged_mid, "M EUR", "results/scenarios/tcover_sweep.csv")


def write():
    mp = C.RES_FINAL / "chapter_number_map.csv"
    m = pd.read_csv(mp)
    m = pd.concat([m[m.chapter != CHAPTER], pd.DataFrame(ROWS)], ignore_index=True)
    m.to_csv(mp, index=False)
    lk = C.RES_FINAL / "final_numbers_locked.md"
    txt = lk.read_text(encoding="utf-8")
    if SECTION in txt:
        a = txt.index("\n" + SECTION)
        b = txt.find("\n## ", a + 1 + len(SECTION))
        txt = txt[:a] + (txt[b:] if b >= 0 else "\n")
    sec = ["", SECTION, "", f"B standard at {C.BSTD_T_INJ_C:g}/{C.BSTD_T_RETURN_C:g} C (src/config.py BSTD_T_RETURN_C): the "
           "numbers that have no own line in the sections above. Read from the outputs of the chain; nothing is typed in.", ""]
    sec += [f"- {r['key'][len(P):]}: **{r['value']} {r['unit']}** | `{r['script']}` | {r['source']}" for r in ROWS]
    lk.write_text(txt.rstrip() + "\n" + "\n".join(sec) + "\n", encoding="utf-8")
    print(f"{len(ROWS)} B-std detail numbers written to chapter_number_map.csv and final_numbers_locked.md")


def main():
    rosim()
    economics()
    scenarios()
    sensitivities()
    write()


if __name__ == "__main__":
    main()
