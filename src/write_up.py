#!/usr/bin/env python3
"""Checks scenario_runner.py against the locked numbers and adds the section on scenarios, T_cover and well limit to
the locked-numbers files. Run by run_all.py."""
import difflib, inspect, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
HERE = C.RES_SCEN     # scenario outputs
FN_DIR = C.RES_FINAL  # locked numbers and chapter map
import final_config as FC  # noqa: E402
import tank_capex as TC  # noqa: E402
import scenarios as SC  # noqa: E402
import tcover_analysis as TA  # noqa: E402

AF, CRF = FC.annuity_factor(), FC.crf()
LBL = {"TTES5k": "TTES 5,000 m3", "A1D": "A1D", "BWN110": "B-WN110", "A2D": "A2D", "Bstd": "B-std", "A4D": "A4D"}
SECTION = "## Scenarios, T_cover and well limit"
CH = "Ch. 7.7 / App. B (scenarios, T_cover, well limit)"


def line_of(module, marker):
    """first line number of `marker` in the module source (producing line for the map)."""
    src = Path(module.__file__).read_text(encoding="utf-8").splitlines()
    for i, l in enumerate(src, 1):
        if marker in l:
            return f"src/{Path(module.__file__).name}:{i}"
    raise KeyError(marker)


def fmt(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−")


# ------------------------------------------------------------------------------------------------ 1. step 0 check
def step0_check():
    sr = pd.read_csv(C.RES_SCENARIO_RUNNER / "scenario_results.csv")
    sr = sr[sr.scenario == "base"].set_index("option")
    a = pd.read_csv(FN_DIR / "step4_npv_ates_basisR.csv")
    a = a[a.f_set == "new"].set_index("option")
    t = pd.read_csv(FN_DIR / "step4_npv_tank.csv")
    tb = t[(t.opex_rule == "scaled") & (t.case == "B")].iloc[0]
    rows = []
    for o in ["TTES5k", "A1D", "A2D", "A4D", "BWN110", "Bstd"]:
        for tier in ("lower", "base", "potential"):
            locked = tb[f"npv_{tier}"] if o == "TTES5k" else a.loc[o, f"npv_{tier}"]
            rows.append(dict(item=f"NPV {tier} {o}", patched_script=sr.loc[o, f"npv_{tier}_mid"], locked=locked))
        if o != "TTES5k":
            rows.append(dict(item=f"f_corr {o}", patched_script=sr.loc[o, "f_corr"], locked=a.loc[o, "f_corr"]))
    d = pd.DataFrame(rows); d["diff"] = d.patched_script - d.locked
    d["within_0.01"] = d["diff"].abs() <= 0.01 + 1e-9
    d.to_csv(HERE / "step0_check.csv", index=False, float_format="%.4f")
    return d


# ------------------------------------------------------------------------------------------------ 2. well limit
def well_limit_derived():
    p = HERE / "well_limit_capped_runs.csv"
    if not p.exists():
        return None
    cap = pd.read_csv(p).set_index("config")
    dr = pd.read_csv(C.RES_FCORR / "direct_runs.csv")
    ev = C.FCORR_EVALS
    bst = pd.read_csv(C.RES_FCORR / "bstd_fcorr_v3_R.csv").iloc[0]
    dec = pd.read_csv(HERE / "scenario_decomposition.csv").query("scenario == 'base'").set_index("option")
    s_flex = float(dec.S_flex.iloc[0])
    a = pd.read_csv(FN_DIR / "step4_npv_ates_basisR.csv").query("f_set == 'new'").set_index("option")
    rows = []
    for o in ("A1D", "BWN110", "Bstd"):
        if o == "Bstd":
            s_dir, s_base, q = bst.s_stor_direct_eur_yr, bst.s_stor_base_eur_yr, bst.discharged_mwh_yr
            base_q = json.loads((C.EVALS_BSTD / f"{C.tagged('Bstd')}__T__1.00000__1.00000.json").read_text())["discharged_mwh_yr"]
        else:
            r = dr[(dr.config == o) & (dr.setting == "T") & (dr.target == "R7")].iloc[0]
            b = json.loads((ev / f"{o}__T__1.00000__1.00000.json").read_text())
            s_dir, s_base, q, base_q = r.s_stor_eur_yr, b["s_stor_eur_yr"], r.discharged_mwh_yr, b["discharged_mwh_yr"]
        c = cap.loc[o]
        cx, ox = FC.ATES[o]["capex_meur"][1], FC.ATES[o]["opex_keur"]
        npv_u = -cx + (s_flex + s_dir / 1000 - ox) * AF / 1000
        npv_c = -cx + (s_flex + c.s_stor_direct_capped / 1000 - ox) * AF / 1000 if c.found else np.nan
        lc = lambda qq: (cx * 1e6 * CRF + ox * 1e3) / qq
        rows.append(dict(option=o, mw_uncapped=13.3 if o != "Bstd" else 6.64, mw_capped=c.mw_cap,
                         discharged_scale1_uncapped=base_q, discharged_scale1_capped=c.discharged_base_capped,
                         discharged_direct_uncapped=q, discharged_direct_capped=c.discharged_direct_capped,
                         discharged_change_pct=100 * (c.discharged_direct_capped / q - 1),
                         s_stor_scale1_uncapped=s_base / 1000, s_stor_scale1_capped=c.s_stor_base_capped / 1000,
                         f_corr_uncapped=s_dir / s_base, f_corr_capped=c.f_corr_capped,
                         s_stor_cal_uncapped=s_dir / 1000, s_stor_cal_capped=c.s_stor_direct_capped / 1000,
                         npv_base_locked=a.loc[o, "npv_base"], npv_base_uncapped_recomputed=npv_u, npv_base_capped=npv_c,
                         npv_base_change=npv_c - npv_u, lcoh_uncapped=lc(q), lcoh_capped=lc(c.discharged_direct_capped),
                         scales_capped=f"{c.scale_1:g} / {c.scale_2:g}"))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "well_limit_derived.csv", index=False, float_format="%.4f")
    return d


# ------------------------------------------------------------------------------------------------ 3. markdown
def md_table(df, cols, heads, fmts):
    out = ["| " + " | ".join(heads) + " |", "|" + "|".join("---" for _ in heads) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f(r[c]) for c, f in zip(cols, fmts)) + " |")
    return "\n".join(out)


def tables(s0, wl):
    T = {}
    T["step0"] = md_table(s0, ["item", "patched_script", "locked", "diff"], ["number", "patched scenario_runner.py", "locked", "diff"],
                          [str, lambda x: f"{x:.3f}", lambda x: f"{x:.3f}", lambda x: f"{x:+.4f}"])
    v = pd.read_csv(HERE / "scenarios_old_vs_new.csv")
    v["option"] = v.option.map(LBL)
    for c in ("sign_change", "rank_change", "sign_change_any_case", "rank_change_any_case"):
        v[c] = v[c].fillna("")
    T["scen_old_new"] = md_table(
        v, ["scenario_label", "thesis_table", "option", "old_npv", "new_npv", "change", "sign_change", "old_rank", "new_rank",
            "rank_change", "new_npv_fcorr_D", "new_npv_tank_D", "new_npv_tank_H", "sign_change_any_case", "rank_change_any_case"],
        ["scenario", "thesis", "option", "old NPV", "new NPV", "change", "sign flip", "old rank", "new rank", "rank change",
         "new, f_corr D", "new, tank D", "new, tank H", "sign flip, any case", "rank change, any case"],
        [str, str, str, fmt, fmt, fmt, str, lambda x: str(int(x)), lambda x: str(int(x)), str, fmt,
         lambda x: fmt(x), lambda x: fmt(x), str, str])
    ts = pd.read_csv(HERE / "scenario_tank_sign.csv")
    cols = [c for c in ts.columns if c not in ("case", "tier")]
    T["tank_sign"] = md_table(ts, ["case", "tier"] + cols, ["case", "tier"] + [SC.LABEL.get(c, c) for c in cols],
                              [str, str] + [fmt] * len(cols))
    be = pd.read_csv(HERE / "scenario_breakeven.csv")
    be = be[be.set.isin(["old", "new_R_B", "new_R_H"])].copy()
    be["option"] = be.option.map(LBL)
    be["set"] = be.set.map({"old": "thesis", "new_R_B": "new (tank B)", "new_R_H": "new (tank H)"})
    sk = lambda x: x if isinstance(x, str) and not re.match(r"^-?\d", str(x)) else f"{float(x):.2f}"
    T["breakeven"] = md_table(be, ["set", "option", "npv_base", "breakeven_spread_k", "breakeven_growth_pct", "breakeven_rate_pct",
                                   "breakeven_capex_base"],
                              ["inputs", "option", "base NPV", "break-even spread factor k", "break-even growth g (%/yr)",
                               "break-even real rate (%)", "break-even CAPEX (M EUR)"],
                              [str, str, fmt, sk, lambda x: fmt(x, 2), lambda x: fmt(x, 2, False), lambda x: fmt(x, 2, False)])
    tor = pd.read_csv(HERE / "tornado_a1d.csv", index_col=0).reset_index().rename(columns={"index": "bar"})
    T["tornado"] = md_table(tor, ["bar", "old", "new_R_B", "new_D_B"], ["Figure 7.2 bar (A1D)", "thesis", "new (f_corr R)", "new (f_corr D)"],
                            [str, fmt, fmt, fmt])
    cn = pd.read_csv(HERE / "conservative_npv.csv").pivot(index="option", columns="set", values="npv_conservative").reset_index()
    cn["option"] = cn.option.map(LBL)
    T["conservative"] = md_table(cn, ["option", "old", "new_R_B", "new_R_D", "new_R_H", "new_D_B"],
                                 ["option", "thesis", "new (f_corr R, tank B)", "tank D", "tank H", "f_corr D"], [str] + [fmt] * 5)
    tc = pd.read_csv(HERE / "tcover_sweep.csv")
    tc["option"] = tc.option.map(LBL)
    T["tcover"] = md_table(tc, ["option", "t_cover_h", "power_ok_share", "release_hours_yr", "S_stor_relaxed_sym", "freed_value_keur_yr",
                                "S_attr_bridged", "S_attr_bridged_change_vs_3h_pct", "npv_bridged_mid", "npv_bridged_change_vs_3h"],
                           ["option", "T_cover (h)", "power test met (share of hours)", "buffer released (h/yr)",
                            "S_stor relaxed (k EUR/yr)", "freed value vs C (k EUR/yr, before f_corr)", "bridged S_attr (k EUR/yr)",
                            "change vs 3 h (%)", "bridged NPV (M EUR)", "change vs 3 h (M EUR)"],
                           [str, lambda x: f"{x:g}", lambda x: "–" if np.isnan(x) else f"{x:.3f}",
                            lambda x: "–" if np.isnan(x) else f"{x:,.0f}", lambda x: f"{x:.1f}", lambda x: f"{x:+.1f}",
                            lambda x: f"{x:.1f}", lambda x: f"{x:+.2f}", fmt, fmt])
    wh = pd.read_csv(HERE / "well_limit_hours.csv")
    T["wl_hours"] = md_table(wh, ["run", "doublets", "limit_m3h", "peak_ext_m3h", "cap_mw_per_doublet", "hours_inj_above_yr", "hours_ext_above_yr",
                                  "share_discharge_in_hours_above_pct", "share_discharge_above_cap_pct", "share_charge_above_cap_pct"],
                             ["run (per doublet)", "doublets", "limit (m3/h)", "dispatch peak (m3/h)", "limit as MW", "hours above, charging (h/yr)",
                              "hours above, discharging (h/yr)", "discharge in those hours (% of yearly)", "discharge above the limit (% of yearly)",
                              "charge above the limit (% of yearly)"],
                             [str, lambda x: str(int(x)), lambda x: f"{x:.0f}", lambda x: f"{x:.0f}", lambda x: f"{x:.2f}",
                              lambda x: f"{x:.0f}", lambda x: f"{x:.0f}", lambda x: f"{x:.1f}", lambda x: f"{x:.1f}", lambda x: f"{x:.1f}"])
    if wl is not None:
        w = wl.copy(); w["option"] = w.option.map(LBL)
        T["wl_runs"] = md_table(w, ["option", "mw_uncapped", "mw_capped", "discharged_direct_uncapped", "discharged_direct_capped",
                                    "discharged_change_pct", "f_corr_uncapped", "f_corr_capped", "s_stor_cal_uncapped", "s_stor_cal_capped",
                                    "npv_base_uncapped_recomputed", "npv_base_capped", "npv_base_change", "lcoh_uncapped", "lcoh_capped"],
                                ["option", "MW per doublet, now", "MW capped", "discharged, calibrated (MWh/yr), now", "capped",
                                 "change (%)", "f_corr now", "f_corr capped", "S_stor after cal. now (k EUR/yr)", "capped",
                                 "base NPV now (M EUR)", "base NPV capped", "change (M EUR)", "LCOH now (EUR/MWh)", "LCOH capped"],
                                [str, lambda x: f"{x:.2f}", lambda x: f"{x:.2f}", lambda x: f"{x:,.0f}", lambda x: f"{x:,.0f}",
                                 lambda x: f"{x:+.1f}", lambda x: f"{x:.3f}", lambda x: f"{x:.3f}", lambda x: f"{x:.1f}",
                                 lambda x: f"{x:.1f}", fmt, fmt, fmt, lambda x: f"{x:.0f}", lambda x: f"{x:.0f}"])
    return T


def fill_readme(T):
    p = HERE / "README.md"
    if not p.exists():
        return
    txt = p.read_text(encoding="utf-8")
    for k, v in T.items():
        txt = re.sub(rf"(<!-- TABLE:{k} -->\r?\n).*?(<!-- END:{k} -->)", lambda m: m.group(1) + v + "\n" + m.group(2), txt, flags=re.S)
    p.write_text(txt, encoding="utf-8")


# ------------------------------------------------------------------------------------------------ 4. locked + map
def locked_rows(wl):
    rows = []
    def put(key, value, unit, script, source, est="", old=None, old_loc=""):
        rows.append(dict(chapter=CH, key=key, value=value, unit=unit, script=script, source=source, author_estimate=est,
                         old_value=old, old_location=old_loc))
    v = pd.read_csv(HERE / "scenarios_old_vs_new.csv")
    l_npv = line_of(SC, "npv=npv_generic(cx, ox, sa, s[\"fpv\"], s[\"fpv\"]),")
    l_g = line_of(SC, "npv=npv_generic(cx, ox, sa, fg, s[\"fpv\"])")
    l_r = line_of(SC, "npv=npv_generic(cx, ox, sa, fr_, fr_)")
    for r in v.itertuples():
        scr = l_g if r.scenario.startswith("g_") else (l_r if r.scenario.startswith("rate") else l_npv)
        loc = {"Table 7.13": "thesis PDF p. 80", "Table B.5": "thesis PDF p. 116", "Table B.7": "thesis PDF p. 118"}.get(r.thesis_table, "")
        put(f"Base NPV mid CAPEX, {SC.LABEL[r.scenario]}, {LBL[r.option]} (f_corr R, tank B)", round(r.new_npv, 2), "M EUR", scr,
            "scenario_runner savings x final inputs", "AUTHOR ESTIMATE (tank CAPEX)" if r.option == "TTES5k" else "",
            round(r.old_npv, 2), loc)
    ts = pd.read_csv(HERE / "scenario_tank_sign.csv").set_index(["case", "tier"])
    for (case, tier), row in ts.iterrows():
        if tier in ("bridged", "potential") or case == "H":
            for scn in ("demand_m20", "demand_p20", "spread_075", "spread_150", "level_080"):
                put(f"Tank NPV {tier}, case {case}, {SC.LABEL[scn]}", round(row[scn], 2), "M EUR", l_npv,
                    "scenario_runner savings x final inputs", "AUTHOR ESTIMATE (tank CAPEX)")
    be = pd.read_csv(HERE / "scenario_breakeven.csv").query("set == 'new_R_B'").set_index("option")
    l_be = line_of(SC, "breakeven_rate_pct=bisect(")
    for o in ("TTES5k", "A1D", "BWN110"):
        r = be.loc[o]
        put(f"Break-even real discount rate, {LBL[o]}", round(float(r.breakeven_rate_pct), 2), "%", l_be, "NPV = 0, base tier, mid CAPEX")
        put(f"Break-even savings growth g, {LBL[o]}", round(float(r.breakeven_growth_pct), 2), "%/yr", l_be, "Eq. B.1, NPV = 0")
        if o != "TTES5k":
            put(f"Break-even spread factor k, {LBL[o]}", round(float(r.breakeven_spread_k), 3), "-", line_of(SC, "k0 = kk[i] +"),
                "linear interpolation between the spread scenarios")
    cn = pd.read_csv(HERE / "conservative_npv.csv").query("set == 'new_R_B'").set_index("option")
    oldc = pd.read_csv(HERE / "conservative_npv.csv").query("set == 'old'").set_index("option")
    for o in cn.index:
        put(f"NPV mid, conservative (f_corr on S_flex + S_stor), {LBL[o]}", round(float(cn.loc[o, "npv_conservative"]), 2), "M EUR",
            line_of(SC, "npv_conservative=npv_generic("), "Table 7.14 row", "", round(float(oldc.loc[o, "printed_thesis"]), 2), "thesis PDF p. 82")
    tc = pd.read_csv(HERE / "tcover_sweep.csv")
    l_tc = line_of(TA, "npv_bridged_mid=-cx + (sa - ox) * AF / 1000, breakeven_capex_bridged")
    for r in tc.itertuples():
        if r.t_cover_h in (1.0, 2.0, 3.0, 3.25, 3.5, 4.0, 6.0, 12.0) and r.option != "TTES5k":
            put(f"T_cover {r.t_cover_h:g} h: bridged NPV {LBL[r.option]} (A1p, calibrated buffers)", round(r.npv_bridged_mid, 2), "M EUR",
                l_tc, "tcover_runs.py / ates_same_treatment", "")
    tt = tc[tc.option == "TTES5k"].iloc[0]
    put("T_cover: bridged NPV tank case B (calibrated buffers, any T_cover)", round(tt.npv_bridged_mid, 2), "M EUR",
        line_of(TA, "npv_bridged_mid=-cx + (sa_t - ox) * AF / 1000"), "tank model has no T_cover", "AUTHOR ESTIMATE (tank CAPEX)")
    put("T_cover threshold (half the hours pass the power test), 13.3 MW", round(44.25 / 13.3, 2), "h",
        line_of(TA, "threshold_median_h="), "median R 44.25 MWh / 13.3 MW")
    put("T_cover above which every hour passes the power test, 13.3 MW", round(61.32 / 13.3, 2), "h",
        line_of(TA, "threshold_all_hours_h="), "max R 61.32 MWh / 13.3 MW")
    wh = pd.read_csv(HERE / "well_limit_hours.csv").set_index("run")
    l_wh = line_of(sys.modules["well_limit"], "hours_inj_above_yr=float(")
    for run in wh.index:
        r = wh.loc[run]
        put(f"Well limit: discharging hours above limit, {run}", round(float(r.hours_ext_above_yr), 0), "h/yr", l_wh, "dispatch_hourly/<cfg>__thesis.pkl")
        put(f"Well limit: discharge above the limit, {run}", round(float(r.share_discharge_above_cap_pct), 1), "% of yearly discharge", l_wh,
            "same")
    if wl is not None:
        l_wl = line_of(sys.modules["write_up"] if "write_up" in sys.modules else sys.modules[__name__], "npv_c = -cx + (s_flex + c.s_stor_direct_capped")
        for r in wl.itertuples():
            put(f"Well limit sensitivity: f_corr capped, {LBL[r.option]}", round(r.f_corr_capped, 3), "-", l_wl, "well_limit.py search")
            put(f"Well limit sensitivity: discharged heat capped, {LBL[r.option]}", round(r.discharged_direct_capped, 0), "MWh/yr", l_wl, "well_limit.py search")
            put(f"Well limit sensitivity: base NPV capped, {LBL[r.option]}", round(r.npv_base_capped, 2), "M EUR", l_wl, "well_limit.py search")
            put(f"Well limit sensitivity: change in base NPV, {LBL[r.option]}", round(r.npv_base_change, 2), "M EUR", l_wl, "capped - uncapped")
    return pd.DataFrame(rows)


# adds the section 'Scenarios, T_cover and well limit' to final_numbers_locked.md and chapter_number_map.csv.
# final_numbers.py rewrites both files; this script must therefore run AFTER final_numbers.py.
def write_locked(L):
    mp = FN_DIR / "chapter_number_map.csv"
    m = pd.read_csv(mp)
    m = m[m.chapter.astype(str) != CH]
    pd.concat([m, L], ignore_index=True).to_csv(mp, index=False)
    lk = FN_DIR / "final_numbers_locked.md"
    txt = lk.read_text(encoding="utf-8")
    txt = txt.split("\n" + SECTION)[0].rstrip() + "\n"
    out = ["", SECTION, "",
           "Generated by `src/write_up.py` from the outputs of `scenarios.py`, `tcover_analysis.py` and "
           "`well_limit.py` (tables in `results/scenarios/`). Headline: f_corr basis R, tank case B (5.04 M EUR, OPEX 1.4 %), B-std OPEX 378, "
           "f_PV 15.3725. The well-limit numbers are a sensitivity only; the headline numbers above do not change.", ""]
    for r in L.itertuples():
        old = f" | old: {r.old_value}" + (f" ({r.old_location})" if r.old_location else "") if pd.notna(r.old_value) else ""
        est = f" **{r.author_estimate}**" if r.author_estimate else ""
        out.append(f"- {r.key}: **{r.value} {r.unit}** | `{r.script}` | {r.source}{est}{old}")
    lk.write_text(txt + "\n".join(out) + "\n", encoding="utf-8")


def main():
    import well_limit  # noqa: F401  (for line_of)
    s0 = step0_check()
    print(s0.round(4).to_string(index=False)); print("all within 0.01:", bool(s0["within_0.01"].all()))
    wl = well_limit_derived()
    if wl is not None:
        print(wl.round(3).T.to_string())
    T = tables(s0, wl)
    fill_readme(T)
    (HERE / "_tables_preview.md").write_text("\n\n".join(f"### {k}\n\n{v}" for k, v in T.items()), encoding="utf-8")
    L = locked_rows(wl)
    write_locked(L)
    print(f"{len(L)} scenario, T_cover and well-limit numbers written to final_numbers_locked.md and chapter_number_map.csv")


if __name__ == "__main__":
    main()
