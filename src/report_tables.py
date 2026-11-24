"""Builders of all 50 thesis tables in final form; called by report_assets.py."""
import functools
import json
import math
import re

import numpy as np
import pandas as pd

import config as C
import final_config as FC
import report_numbers as RN
from report_numbers import fmt_num, fmt_signed

PRINTED = json.loads((C.DATA / "thesis_printed_tables.json").read_text(encoding="utf-8"))
NAME = {"TTES5k": "TTES 5k", "A1D": "Option A 1D", "BWN110": "B-WN 110 °C", "A2D": "Option A 2D",
        "Bstd": "B standard", "A4D": "Option A 4D"}
LK = {"A1D": "Option A 1D", "A2D": "Option A 2D", "A4D": "Option A 4D", "BWN110": "B-WN110", "Bstd": "B standard"}
pct1 = lambda v: f"{v:.1f}%"
# the locked scenario numbers (write_up.py) were first rounded to 3 and then to 2 decimals; the same path here,
# so that table and locked value are equal (difference at most 0.01 with single rounding).
lk2 = lambda v: round(round(float(v), 3), 2)
D32 = (3, 2)


def T(df, status, depends, source, note="", checks=()):
    return dict(df=df, status=status, depends=depends, source=source, note=note, checks=list(checks))


def printed(num, status="thesis input, unchanged", depends="no", source=None, note=""):
    p = PRINTED[f"Table {num}"]
    return T(pd.DataFrame(p["rows"], columns=p["columns"]), status, depends,
             source or f"thesis PDF p. {p['page']} (printed values)", note)


def lock_npv(o, tier):
    return f"TTES NPV {tier}, case B" if o == "TTES5k" else f"{LK[o]} NPV {tier} (mid CAPEX)"


@functools.lru_cache(maxsize=None)
def rosim():
    return pd.read_csv(C.RES_ROSIM / "rosim_check.csv")


def eff_years(run, col="recomputed"):
    r = rosim()
    return r[r.run == run].set_index("year")[col]


@functools.lru_cache(maxsize=None)
def scen():
    import scenarios as SC
    D = SC.decompose()
    sets, share = SC.inputs()
    return SC, D, sets, share, SC.all_rows(D, sets, share)


@functools.lru_cache(maxsize=None)
def ttes_sweep(schedule="relaxed"):
    f = "ttes_price_input_test_relaxed.csv" if schedule == "relaxed" else "ttes_price_input_test.csv"
    return pd.read_csv(C.RES_TTES_SWEEP / f)


def ttes_capacity_mwh(v):
    import run_ttes_low_temp as TL
    return v * TL.RHO * TL.CP * (110.0 - TL.T_RET) / 3.6e9


# ============================================================================== Chapter 3
def t3_1():
    return printed("3.1", note="Computed from the hourly AEB loop data (not included); does not depend on changed numbers.")


def t3_2():
    return printed("3.2", note="The dispatch capacities are unchanged (6.64 MW per B-std doublet). With the 58 °C return of the "
                   "final B-std runs (90/58 °C, 32 K), the Thiem flow of 179 m3/h corresponds to 6.6 MW (Table 3.4), as "
                   "printed; the dispatch peak of 6.64 MW is 184 m3/h (well-limit sensitivity, Appendix B).")


def t3_3():
    from audit_lib import read_settings
    s = read_settings("Option A 1 Doublet 8 years changed parameters")
    lay = {r[0]: r for r in s["layers"]}["default layer"]
    out = printed("3.3", note="Thesis inputs, unchanged; aquifer thickness, porosity, permeabilities, cell size and grid checked "
                  "against the ROSIM settings of the final runs.")
    out["checks"] = [("aquifer thickness = ROSIM settings", float(lay[3]), 33.8, 1),
                     ("porosity = ROSIM settings", float(lay[7]), 0.28, 2),
                     ("horizontal permeability = ROSIM settings", float(lay[9]), 4000, 0),
                     ("vertical permeability = ROSIM settings", float(lay[11]), 300, 0),
                     ("cell size = ROSIM settings", float(s["x_spacing__grid_cell_size_"]), 25, 0),
                     ("grid cells x = ROSIM settings", float(s["x_number_of_cells"]), 80, 0)]
    return out


def t3_4():
    """Thiem flow (Eq. 3.1) from the printed parameters; thermal power with the final return temperatures."""
    import rosim_check as RCK
    p = {r[0]: r[1:] for r in PRINTED["Table 3.4"]["rows"]}
    num = lambda s: float(s.replace(",", ""))
    MD = 9.869233e-16                                    # m2 per mD
    rows, Q = [], []
    for j, (t_in, run) in enumerate(((110.0, "A1D"), (90.0, "B-std WN"), (58.0, None))):
        k, H, dh, rw, R, mu = (num(p[x][j]) for x in ("k (mD)", "H (m)", "Δh (m)", "r_w (m)", "R (m)", "μ (Pa·s)"))
        Q.append(2 * math.pi * k * MD * H * FC.RHO_G_THIEM * dh / (mu * math.log(R / rw)) * 3600)
    for x in ("k (mD)", "H (m)", "Δh (m)", "r_w (m)", "R (m)", "μ (Pa·s)"):
        rows.append([x] + p[x])
    rows.append(["Q_Thiem (m3/h)"] + [f"{q:.0f}" for q in Q])
    dts, pd_, pi_ = [], [], []
    for j, run in enumerate(("A1D", "B-std WN")):
        t_in, t_ret = RCK.RUNS[run][2], RCK.RUNS[run][3]      # B-std: the return temperature of config.py (58 C)
        dt = t_in - t_ret
        rho, cp = FC.WATER_PROPS[t_in]
        dts.append(dt)
        pd_.append(Q[j] * C.CONST * dt)
        pi_.append(Q[j] * rho / 1000 * rho * cp * dt / 3.6e9)
    rows.append([f"ΔT with final return (K): 58 °C (110 °C well), {RCK.RUNS['B-std WN'][3]:.0f} °C (90 °C well)", f"{dts[0]:.0f}", f"{dts[1]:.0f}", "–"])
    rows.append(["Thermal power, dispatch basis (MW)", f"{pd_[0]:.1f}", f"{pd_[1]:.1f}", "–"])
    rows.append(["Thermal power, in-situ properties (MW)", f"{pi_[0]:.1f}", f"{pi_[1]:.1f}", "–"])
    df = pd.DataFrame(rows, columns=PRINTED["Table 3.4"]["columns"])
    # check: the same formulas with the printed 58 C return reproduce the printed power rows
    chk = [("Q_Thiem 110 °C reproduces printed 221", Q[0], 221, 0), ("Q_Thiem 90 °C reproduces printed 179", Q[1], 179, 0),
           ("Q_Thiem cold reproduces printed 117", Q[2], 117, 0),
           ("dispatch-basis power 110 °C (52 K) = printed 13.3", Q[0] * C.CONST * 52, 13.3, 1),
           ("in-situ power 110 °C at 52 K = printed 12.2", Q[0] * FC.WATER_PROPS[110.0][0] ** 2 / 1000 * FC.WATER_PROPS[110.0][1] * 52 / 3.6e9, 12.2, 1),
           ("in-situ power 90 °C at 32 K = printed 6.2", Q[1] * FC.WATER_PROPS[90.0][0] ** 2 / 1000 * FC.WATER_PROPS[90.0][1] * 32 / 3.6e9, 6.2, 1),
           ("dispatch-basis power 90 °C (32 K, final B-std window) = printed 6.6", pd_[1], 6.6, 1)]
    return T(df, "final numbers", "yes (90 °C well: return 58 °C, as printed)", "Eq. 3.1 with the printed parameters; final "
             "temperatures (rosim_check.RUNS; B-std at config.BSTD_T_RETURN_C = 58 °C); water properties at the injection "
             "temperature (final_config.WATER_PROPS)",
             "Parameters are thesis inputs. The 90 °C column uses the 58 °C return of the final B-std runs (ΔT 32 K): "
             "179 m3/h = 6.6 MW, as printed.", chk)


def t3_5():
    return printed("3.5", note="AEB buffer data and the dispatch-only run; does not depend on changed numbers.")


def t3_6():
    e = pd.read_csv(C.EPEX_STATS)
    e = e[(e.series == "Prijs_Gemiddelde_EPEX_Spot") & e.period.isin(["2023", "2024", "2025", "2023-2025"])]
    rows = [[r.period.replace("-", "–"), f"{r.mean:.1f}", f"{r.p30:.1f}", f"{r.p60:.1f}", f"{r.p75:.1f}", f"{r.spread:.1f}",
             fmt_num(r.neg_hours)] for r in e.itertuples()]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 3.6"]["columns"]), "unchanged, scripted", "no",
             "data/derived/epex_stats.csv (epex_stats.py)")


def t4_1():
    return printed("4.1")


# ============================================================================== Chapter 5
def t5_1():
    d = ttes_sweep().query("variant == 'actual'")
    rows = []
    for v in sorted(d.volume_m3.unique()):
        g = {r.dispatch: r for r in d[d.volume_m3 == v].itertuples()}
        rows.append([fmt_num(v), fmt_num(ttes_capacity_mwh(v)), fmt_signed(g["P20/P80"].increment_eur_yr / 1000, 0),
                     f"{g['P20/P80'].cycles_per_yr:.0f}", fmt_signed(g["P30/P60"].increment_eur_yr / 1000, 0),
                     f"{g['P30/P60'].cycles_per_yr:.0f}"])
    return T(pd.DataFrame(rows, columns=PRINTED["Table 5.1"]["columns"]), "unchanged, scripted", "no",
             "data/derived/ttes_price_input_test_relaxed.csv (actual EPEX thresholds)")


def t5_2():
    out = printed("5.2", note="Geometry, temperatures and U_eff are thesis inputs; capacity, discharged heat and cycles checked "
                  "against the TTES sweep.")
    d = ttes_sweep().query("variant == 'actual' and dispatch == 'P20/P80'").set_index("volume_m3")
    out["checks"] = [("capacity 5,000 m3 (MWh)", ttes_capacity_mwh(5000), 290, 0),
                     ("discharged 5,000 m3 (MWh/yr) = locked", RN.q_disch("TTES5k"), "TTES discharged heat (dispatch, relaxed, 5,000 m3)", 0),
                     ("discharged 19,000 m3 (MWh/yr)", d.loc[19000, "discharged_mwh_yr"], 9541, 0),
                     ("cycles 5,000 m3", round(d.loc[5000, "cycles_per_yr"]), 20, 0)]
    return out


# ============================================================================== Chapter 6
RUN_PROJ = {"1D": "Option A 1 Doublet 8 years changed parameters", "2D": "Option A 2 doublets 8 years changed parameters",
            "4D": "Option A 4 doublets 8 years changed parameters"}


def t6_1():
    from audit_lib import read_settings, read_wells
    rows = []
    for cfg, proj in RUN_PROJ.items():
        s = read_settings(proj)
        xs = dict(zip(s["wells"], map(float, s["x_coordinate_well_head"].split(";"))))
        ys = dict(zip(s["wells"], map(float, s["y_coordinate_well_head"].split(";"))))
        vox = {st.split("_")[0]: st for st in read_wells(proj)}       # well file names (cell i, j) of the ROSIM run
        n = len(s["wells"]) // 2
        for d in range(1, n + 1):
            for w, lab in ((f"warm{d}", "Warm well"), (f"cold{d}", "Cold well")):
                m = re.search(r"_i(\d+)j(\d+)", vox[w])
                rows.append([cfg, f"D{d}", lab, m.group(1), m.group(2), fmt_num(xs[w]), fmt_num(ys[w])])
    return T(pd.DataFrame(rows, columns=PRINTED["Table 6.1"]["columns"]), "final numbers", "yes (final ROSIM runs, 1D spacing 100 m)",
             "ROSIM settings (well heads) and well-file names (cell i, j) of the final runs",
             "Cell indices are those of the ROSIM well files; X, Y are the well-head coordinates of the settings.")


def t6_2():
    import fcorr_runs as fr
    rows = []
    for cfg in ("A1D", "A2D", "A4D"):
        j = json.loads((C.FCORR_EVALS / f"{fr.tag(cfg, 'T', 1.0, 1.0)}.json").read_text())
        ch, dis = j["charged_mwh_yr"], j["discharged_mwh_yr"]
        added = j["e_end_mwh"] / len(fr.YEARS)
        loss = ch - dis - added - j["balance_residual_pct"] / 100 * ch
        rows.append([cfg[1:], f"{fr.CAP[cfg]:.1f}", fmt_num(ch), fmt_num(dis), fmt_num(added), fmt_num(loss), f"{j['eta_cycle_pct']:.1f}"])
    return T(pd.DataFrame(rows, columns=PRINTED["Table 6.2"]["columns"]), "unchanged, scripted", "no (uncalibrated dispatch)",
             "data/derived/fcorr_evals (loss scale 1)",
             "Stock = heat in the aquifer at the end of 2025 / 3; ground loss closes the balance with the run's residual "
             "(the printed split differs by up to 13 MWh/yr).")


@functools.lru_cache(maxsize=None)
def thermal_radius():
    import tab_6_3_6_10_thermal_radius as M
    geo = pd.read_csv(C.RES_ROSIM / "rosim_geometry.csv").set_index("run")
    out = {}
    for run, (proj, wb, disp, key, n, label) in M.RUNS.items():
        H = M.aquifer_thickness(proj)
        v = M.injected_volume(wb)
        rth = math.sqrt(v * FC.R_TH / (math.pi * H))
        L = float(str(geo.loc[run, "warm_cold_m"]).split(";")[0])
        import data_store as DS
        w = DS.dispatch_weekly(C.DISPATCH_WEEKLY, run)    # weekly aggregate of the hourly dispatch (sums and maxima)
        out[run] = dict(L=L, rth=rth, cap=w.charge_max_mw.max(), charged=w.charge_mwh.sum() / 3 / n)
    return out


def t6_3():
    t = thermal_radius()
    rows = [[c, f"{t[r]['L']:.0f}", f"{t[r]['rth']:.0f}", f"{t[r]['L'] / t[r]['rth']:.1f}"] for c, r in (("1D", "A1D"), ("2D", "A2D"), ("4D", "A4D"))]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 6.3"]["columns"]), "final numbers", "yes (1D spacing 100 m, final schedules)",
             "Eq. 3.3 with the year-7 injection of the final schedules, H from the ROSIM settings, R_th = 1.51; L from rosim_geometry.csv")


def _eff_table(cols):
    rows = []
    for y in range(2, 8):
        rows.append([str(y)] + [pct1(eff_years(run, col).loc[y]) for run, col in cols])
    return rows


def t6_4():
    cols = [("A1D", "recomputed"), ("A2D", "recomputed"), ("A4D", "recomputed")]
    chk = [(f"{r} year 7", eff_years(r).loc[7], f"{LK[r]} efficiency year 7 (basis R)", 2) for r, _ in cols]
    return T(pd.DataFrame(_eff_table(cols), columns=PRINTED["Table 6.4"]["columns"]), "final numbers", "yes (final ROSIM runs)",
             "results/rosim/rosim_check.csv (basis R)", checks=chk)


def t6_5():
    I = RN.inputs()
    import fcorr_runs as fr
    rows, chk = [], []
    for cfg in ("A1D", "A2D", "A4D"):
        ev = json.loads((C.FCORR_EVALS / f"{fr.tag(cfg, 'T', 1.0, 1.0)}.json").read_text())
        er = float(eff_years(cfg).loc[7])
        f = RN.f_corr(cfg)
        rows.append([cfg[1:], f"{er:.1f}", f"{ev['eta_cycle_pct']:.1f}", fmt_num(RN.storage_before(cfg)), fmt_num(RN.storage_after(cfg)),
                     f"{f:.3f}", f"{er / ev['eta_cycle_pct']:.3f}", fmt_num(ev["discharged_mwh_yr"]), fmt_num(I["q"]["R"][cfg])])
        chk += [(f"f_corr {cfg}", f, f"{LK[cfg]} f_corr (basis R)", 3),
                (f"storage effect after calibration {cfg}", RN.storage_after(cfg), f"{LK[cfg]} storage effect after calibration", 1),
                (f"Q_disch,cal {cfg}", I["q"]["R"][cfg], f"{LK[cfg]} Q_disch (LCOH energy)", 0)]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 6.5"]["columns"]), "final numbers", "yes (η_ROSIM, f_corr)",
             "rosim_check.csv, data/derived/fcorr_evals, results/fcorr/direct_runs.csv", checks=chk)


def t6_6():
    rows = []
    for cfg, lab in (("A1D", "1D (13.3 MW)"), ("A2D", "2D (26.6 MW)"), ("A4D", "4D (53.2 MW)")):
        se = RN.storage_after(cfg)
        tot = RN.s_ref() + RN.s_flex() + se
        rows.append([lab, fmt_num(tot), fmt_num(RN.s_ref()), fmt_num(RN.s_flex()), fmt_num(se), f"{100 * se / tot:.1f}"])
    return T(pd.DataFrame(rows, columns=PRINTED["Table 6.6"]["columns"]), "final numbers", "yes (f_corr)",
             "final_numbers inputs (S_ref, S_flex, storage effect after calibration)")


def t6_7():
    lim = [r[2] for r in PRINTED["Table 6.7"]["rows"]]
    rows = [[str(y), pct1(eff_years("A2D").loc[y]), lim[y - 2]] for y in range(2, 8)]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 6.7"]["columns"]), "partly", "base column yes; limited column not reproducible",
             "base: rosim_check.csv (final A2D run); cold-well-limited: printed (separate ROSIM run, not repeated)",
             "The cold-well-limited ROSIM run was not repeated with the final settings; its column is the printed one.",
             [("A2D base year 7", eff_years("A2D").loc[7], "Option A 2D efficiency year 7 (basis R)", 2)])


def t6_8():
    pr = PRINTED["Table 6.8"]["rows"]
    rows = []
    for (cfg, r) in zip(("A1D", "A2D", "A4D"), pr):
        base = RN.storage_before(cfg)
        lim = float(r[4])
        rows.append([r[0], r[1], fmt_num(base), r[3], r[4], f"{100 * (lim / base - 1):.0f}".replace("-", "−")])
    return T(pd.DataFrame(rows, columns=PRINTED["Table 6.8"]["columns"]), "partly", "no (dispatch before calibration)",
             "base: fcorr_evals scale 1; cold-well-limited dispatch runs: printed (not in the repository)")


def t6_9():
    t = thermal_radius()
    cols = [f"WN ({t['B-std WN']['L']:.0f} m, r_th = {t['B-std WN']['rth']:.0f} m, L/r_th = {t['B-std WN']['L'] / t['B-std WN']['rth']:.1f})",
            f"NW ({t['B-std NW']['L']:.0f} m, r_th = {t['B-std NW']['rth']:.0f} m, L/r_th = {t['B-std NW']['L'] / t['B-std NW']['rth']:.1f})"]
    rows = [[str(y), pct1(eff_years("B-std WN").loc[y]), pct1(eff_years("B-std NW").loc[y])] for y in range(2, 8)]
    chk = [("B-std WN year 7", eff_years("B-std WN").loc[7], "B standard WN efficiency year 7 (basis R)", 2),
           ("B-std NW year 7", eff_years("B-std NW").loc[7], "B standard NW efficiency year 7 (basis R)", 2)]
    return T(pd.DataFrame(rows, columns=["Year"] + cols), "final numbers", "yes (final runs, 75 m, 90/58 °C)",
             "rosim_check.csv; thermal radius as Table 6.10", checks=chk)


def t6_10():
    t = thermal_radius()
    rows = [[lab, f"{t[r]['cap']:.1f}", fmt_num(t[r]["charged"]), f"{t[r]['rth']:.0f}", f"{t[r]['L']:.0f}", f"{t[r]['L'] / t[r]['rth']:.1f}"]
            for lab, r in (("WN 90 °C", "B-std WN"), ("WN 110 °C", "B-WN110"), ("NW 90 °C", "B-std NW"))]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 6.10"]["columns"]), "final numbers", "yes (B-std 75 m, 90/58 °C, 32 K schedules)",
             "dispatch_hourly, final schedules, ROSIM settings")


def t6_11():
    rows = [[str(y), pct1(eff_years("B-WN110", "cleaned").loc[y])] for y in range(2, 8)]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 6.11"]["columns"]), "final numbers", "yes (final run)",
             "rosim_check.csv (cleaned: weeks 339-340 of the cold well interpolated)",
             checks=[("B-WN110 year 7", eff_years("B-WN110", "cleaned").loc[7], "B-WN110 efficiency year 7 (basis R)", 2)])


def t6_12():
    s = pd.read_csv(C.FIG63_STATS, index_col=0)
    rows = []
    for k in s.index:
        m = re.match(r"cum_energy_le_([\d.]+)$", str(k))
        if m:
            cap = float(m.group(1))
            rows.append([f"{round(cap / 13.3):.0f}", f"{cap:.1f}", f"{s.loc[k, 'charge']:.1f}", f"{s.loc[k, 'discharge']:.1f}"])
    return T(pd.DataFrame(rows, columns=PRINTED["Table 6.12"]["columns"]), "unchanged, scripted", "no (dispatch)",
             "data/derived/charge_discharge_power_stats.csv")


def t6_13():
    rows, chk = [], []
    for o, temp, hp in (("A1D", "110 °C", "No"), ("A2D", "110 °C", "No"), ("A4D", "110 °C", "No"), ("BWN110", "110 °C", "No"),
                        ("Bstd", "90 °C", "Yes")):
        if o == "Bstd":
            eta = (eff_years("B-std WN").loc[7] + eff_years("B-std NW").loc[7]) / 2
        elif o == "BWN110":
            eta = eff_years("B-WN110", "cleaned").loc[7]
        else:
            eta = eff_years(o).loc[7]
        se = RN.storage_after(o)
        rows.append([RN_label(o), temp, pct1(eta), f"{RN.f_corr(o):.3f}", fmt_num(RN.s_ref() + RN.s_flex() + se), fmt_num(se), hp])
        chk.append((f"f_corr {o}", RN.f_corr(o), f"{LK[o]} f_corr (basis R)", 3))
    st = RN.storage_after("TTES5k")
    rows.append(["TTES 5,000 m3", "110 °C", "–", "–", fmt_num(RN.s_ref() + RN.s_flex() + st), fmt_num(st), "No"])
    return T(pd.DataFrame(rows, columns=PRINTED["Table 6.13"]["columns"]), "final numbers", "yes", "final numbers",
             "B standard η_yr7 is the mean of the WN and NW doublets.", chk)


def RN_label(o):
    return {"A1D": "Option A 1D", "A2D": "Option A 2D", "A4D": "Option A 4D", "BWN110": "B-WN 110 °C", "Bstd": "B standard"}[o]


# ============================================================================== Chapter 7
def t7_1():
    return printed("7.1")


def t7_2():
    rows = [["1D / B-WN 110 °C (13.3 MW)"] + [f"{v:.1f}" for v in FC.ATES["A1D"]["capex_meur"]],
            ["2D (26.6 MW)"] + [f"{v:.1f}" for v in FC.ATES["A2D"]["capex_meur"]],
            ["4D (53.2 MW)"] + [f"{v:.1f}" for v in FC.ATES["A4D"]["capex_meur"]]]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 7.2"]["columns"]), "thesis input, unchanged", "no", "final_config.ATES",
             "The printed 4D high value is 39.5 M€; the model (final_config.py, scenario_runner.py) uses 40.0 M€, which is "
             "the value behind every NPV high of Option A 4D. The table shows the model value.")


def t7_3():
    out = printed("7.3", note="Components are thesis inputs; the total equals final_config.ATES['Bstd'].")
    out["checks"] = [("B-std total mid = final_config", FC.ATES["Bstd"]["capex_meur"][1], 22.2, 1)]
    return out


def t7_4():
    rows = [r[:] for r in PRINTED["Table 7.4"]["rows"]]
    rows[0][2] = fmt_num(FC.ATES["A1D"]["opex_keur"]); rows[1][2] = fmt_num(FC.ATES["A2D"]["opex_keur"])
    rows[2][2] = fmt_num(FC.ATES["A4D"]["opex_keur"]); rows[6][2] = fmt_num(FC.ATES["Bstd"]["opex_keur"])
    rows[7] = ["TTES 5,000 m3", "1.4% of CAPEX (case B, 5.04 M€)", f"{RN.opex('TTES5k'):.1f}"]
    lo, mid, hi = RN.tank_case_b(19000)
    import tank_capex as TC
    rows[8] = ["TTES 19,000 m3", f"1.4% of CAPEX (case-B method at 19,000 m3, {mid:.2f} M€)", f"{TC.opex_keur(mid):.1f}"]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 7.4"]["columns"]), "final numbers", "yes (tank OPEX = 1.4% of case-B CAPEX)",
             "final_config.ATES, tank_capex", "TTES OPEX is 1.4% of its own CAPEX; the 19,000 m3 CAPEX follows the case-B method "
             "(Duesseldorf and Duisburg scaled to 19,000 m3 and escalated), which is a derivation for this table.",
             [("TTES 5k OPEX", RN.opex("TTES5k"), "Tank OPEX, headline (1.4 % of case B)", 2)])


def t7_5():
    cols = ["TTES5k", "A1D", "A2D", "A4D", "BWN110", "Bstd"]
    st = {o: RN.storage_before(o) for o in cols}
    sa = {o: RN.storage_after(o) for o in cols}
    rows = [["Dispatch-only reference S_ref"] + [fmt_num(RN.s_ref())] * 6,
            ["Operating flexibility S_flex"] + [fmt_num(RN.s_flex())] * 6,
            ["Storage effect S_stor"] + [fmt_num(st[o]) for o in cols],
            ["f_corr"] + ["1.00"] + [f"{RN.f_corr(o):.3f}" for o in cols[1:]],
            ["Storage effect after calibration"] + [fmt_num(sa[o]) for o in cols],
            ["S_corr"] + [fmt_num(RN.s_ref() + RN.s_flex() + sa[o]) for o in cols],
            ["Increment above reference"] + [fmt_num(RN.s_flex() + sa[o]) for o in cols]]
    chk = [(f"storage after calibration {o}", sa[o], f"{LK[o]} storage effect after calibration", 1) for o in cols[1:]]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 7.5"]["columns"]), "final numbers", "yes (f_corr)", "final numbers", checks=chk)


def t7_6():
    rows, chk = [], []
    for o in RN.OPTIONS:
        incr = RN.tiers(o)["base"]
        net = incr - RN.opex(o)
        pb = RN.simple_payback(o)
        cx = RN.capex(o)
        rows.append([NAME[o] if o != "A1D" else "Option A 1D (13.3 MW)", fmt_num(RN.s_ref() + incr), fmt_num(incr), fmt_num(RN.opex(o)),
                     fmt_num(net), f"{cx:.2f}" if o == "TTES5k" else f"{cx:.1f}", fmt_signed(RN.npv(o)),
                     ">30" if pb > FC.LIFETIME else f"{pb:.1f}"])
        chk.append((f"NPV {o}", RN.npv(o), lock_npv(o, "base"), 2))
    return T(pd.DataFrame(rows, columns=PRINTED["Table 7.6"]["columns"]), "final numbers", "yes",
             "final numbers; payback = CAPEX / (S_flex + f_corr S_stor - OPEX), simple and undiscounted (thesis convention)",
             "TTES: case B CAPEX 5.04 M€, OPEX 1.4% of CAPEX.", chk)


def conservative(o):
    SC, D, sets, share, A = scen()
    d = D.loc[("base", o)]
    s = sets["new_R_B"]
    cx, ox = SC.capex_opex(o, s)
    return SC.npv_generic(cx, ox, s["f"][o] * (d.S_flex + d.S_stor), s["fpv"], s["fpv"])


def t7_7():
    rows, chk = [], []
    for o in RN.OPTIONS:
        lo, mid, hi = RN.npv(o, case="low"), RN.npv(o), RN.npv(o, case="high")
        cons = conservative(o)
        rows.append([NAME[o] if o != "A1D" else "Option A 1D (13.3 MW)", fmt_signed(lo), fmt_signed(mid), fmt_signed(hi), fmt_signed(lk2(cons))])
        chk.append((f"NPV mid {o}", mid, lock_npv(o, "base"), 2))
        chk.append((f"NPV conservative {o}", cons, f"NPV mid, conservative (f_corr on S_flex + S_stor), {short(o)}", D32))
        if o == "TTES5k":
            chk += [("TTES NPV low (case B_low)", lo, "TTES NPV base, case B_low", 2), ("TTES NPV high (case B_high)", hi, "TTES NPV base, case B_high", 2)]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 7.7"]["columns"]), "final numbers", "yes",
             "final numbers; TTES low/high = case B low/high with OPEX 1.4% of their own CAPEX", checks=chk)


def short(o):
    return {"TTES5k": "TTES 5,000 m3", "A1D": "A1D", "BWN110": "B-WN110", "A2D": "A2D", "Bstd": "B-std", "A4D": "A4D"}[o]


def t7_8():
    rows, chk = [], []
    for o in RN.OPTIONS:
        bb, bp = RN.breakeven(o, "base"), RN.breakeven(o, "potential")
        d50 = 0.5 * RN.opex(o) * RN.AF / 1000
        cx = RN.capex(o)
        rows.append([NAME[o], f"{cx:.2f}" if o == "TTES5k" else f"{cx:.1f}", f"{bb:.1f}", f"{bp:.1f}", f"±{d50:.2f}"])
        key_b = "TTES break-even CAPEX base (OPEX scaled)" if o == "TTES5k" else f"{LK[o]} break-even CAPEX base"
        key_p = "TTES break-even CAPEX potential (OPEX scaled)" if o == "TTES5k" else f"{LK[o]} break-even CAPEX potential"
        chk += [(f"break-even base {o}", bb, key_b, 2), (f"break-even potential {o}", bp, key_p, 2)]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 7.8"]["columns"]), "final numbers", "yes", "final numbers",
             "TTES break-even with OPEX 1.4% of CAPEX: S AF / (1 + 0.014 AF).", chk)


def t7_9():
    rows, chk = [], []
    for o in RN.OPTIONS:
        lo, ba, po = RN.npv(o, "lower"), RN.npv(o, "base"), RN.npv(o, "potential")
        phi = "–" if ba >= 0 else (f"{-ba / (RN.s_ref() * RN.AF / 1000):.2f}" if -ba / (RN.s_ref() * RN.AF / 1000) <= 1 else ">1")
        se = RN.tiers(o)["base"] - RN.s_flex()                      # f_corr S_stor of the tier calculation
        psi = (RN.capex(o) * 1000 / RN.AF + RN.opex(o) - se) / RN.s_flex()
        rows.append([NAME[o] + (", P20/P80" if o == "TTES5k" else ""), fmt_signed(lo), fmt_signed(ba), fmt_signed(po), phi,
                     ">1" if psi > 1 else f"{psi:.2f}"])
        chk += [(f"NPV {t} {o}", RN.npv(o, t), lock_npv(o, t), 2) for t in ("lower", "base", "potential")]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 7.9"]["columns"]), "final numbers", "yes",
             "final numbers; φ* = -NPV_base / (S_ref f_PV); ψ* = (CAPEX/f_PV + OPEX - f_corr S_stor) / S_flex", checks=chk)


def t7_10():
    sh = pd.read_csv(C.RESTART_SPLIT / "s_flex_shares.csv")
    rows = []
    for o in ("TTES5k", "A1D"):
        g = sh[sh.option == o].set_index("R_h")
        npvs = [RN.npv(o, s=g.loc[r, "share_value_covered"] * RN.s_flex() + RN.tiers(o)["base"] - RN.s_flex()) for r in (1, 2, 4)]
        rows.append([NAME[o], "1 / 2 / 4", " / ".join(f"{100 * g.loc[r, 'share_count_covered']:.0f}" for r in (1, 2, 4)) + "%",
                     " / ".join(fmt_num(g.loc[r, "value_covered_keur_yr"]) for r in (1, 2, 4)), " / ".join(fmt_signed(v) for v in npvs)])
    chk = [("A1D NPV bridged (R = 2 h)", RN.npv("A1D", "bridged"), "Option A 1D NPV bridged (mid CAPEX)", 2),
           ("TTES NPV bridged (R = 2 h)", RN.npv("TTES5k", "bridged"), "TTES NPV bridged, case B", 2)]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 7.10"]["columns"]), "final numbers", "yes (NPV); shares unchanged",
             "data/derived/restart_split/s_flex_shares.csv; NPV with the final inputs", checks=chk)


def t7_11():
    import tank_capex as TC
    d = ttes_sweep().query("variant == 'actual' and dispatch == 'P20/P80'").set_index("volume_m3")
    lo, mid, hi = RN.tank_case_b(19000)
    import final_numbers as FN
    q19 = d.loc[19000, "discharged_mwh_yr"]
    l19 = FN.lcoh_discounted(mid * 1e6, TC.opex_keur(mid) * 1e3, q19)
    rows = [["TTES 5,000 m3", f"{RN.capex('TTES5k'):.2f}", f"{RN.opex('TTES5k'):.1f}", fmt_num(RN.q_disch("TTES5k")), f"{RN.lcoh('TTES5k'):.0f}"],
            ["TTES 19,000 m3", f"{mid:.2f}", f"{TC.opex_keur(mid):.1f}", fmt_num(q19), f"{l19:.0f}"]]
    chk = [("LCOH TTES case B", RN.lcoh("TTES5k"), "TTES LCOH case B", 1)]
    for o in ("A2D", "A4D", "A1D", "BWN110", "Bstd"):
        rows.append([NAME[o], f"{RN.capex(o):.1f}", fmt_num(RN.opex(o)), fmt_num(RN.q_disch(o)), f"{RN.lcoh(o):.0f}"])
        chk += [(f"LCOH {o}", RN.lcoh(o), f"{LK[o]} LCOH (discounted)", 0), (f"Q_disch {o}", RN.q_disch(o), f"{LK[o]} Q_disch (LCOH energy)", 0)]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 7.11"]["columns"]), "final numbers", "yes",
             "final numbers (discounted LCOH, 5%, 30 yr)", "LCOH is the discounted form (Eq. 7.5 discounted). The 19,000 m3 tank "
             "is costed with the case-B method at its own volume (derived here, not a locked number).", chk)


def t7_12():
    import tank_capex as TC
    rows, chk = [], []
    tl = RN.tiers("TTES5k")["lower"]
    c = ttes_sweep("constrained").query("variant == 'actual' and dispatch == 'P30/P60'").set_index("volume_m3")
    for lab, v, gain in (("TTES 5k, P20/P80", 5000, tl), ("TTES 5k, P30/P60", 5000, c.loc[5000, "increment_eur_yr"] / 1000),
                         ("TTES 19k, P30/P60", 19000, c.loc[19000, "increment_eur_yr"] / 1000)):
        lo, mid, hi = RN.tank_case_b(v)
        rows.append([lab, fmt_num(gain), f"{TC.opex_keur(mid):.0f}"] + [fmt_signed(RN.tank_npv(v, gain, k)) for k in ("low", "mid", "high")])
    chk.append(("TTES lower NPV mid (P20/P80)", RN.tank_npv(5000, tl, "mid"), "TTES NPV lower, case B", 2))
    for o in ("A1D", "A2D", "A4D", "BWN110", "Bstd"):
        g = RN.tiers(o)["lower"]
        rows.append([NAME[o], fmt_num(g), fmt_num(RN.opex(o))] + [fmt_signed(RN.npv(o, "lower", k)) for k in ("low", "mid", "high")])
        chk.append((f"lower NPV mid {o}", RN.npv(o, "lower"), lock_npv(o, "lower"), 2))
    return T(pd.DataFrame(rows, columns=PRINTED["Table 7.12"]["columns"]), "final numbers", "yes",
             "final numbers; TTES rows with the case-B cost at their volume and OPEX 1.4% (the low/high TTES columns use the "
             "low/high case-B cost with OPEX 1.4% of that cost)", checks=chk)


SCEN_COLS = ["base", "demand_m20", "demand_p20", "spread_075", "spread_125", "spread_150"]


def _scen_npv(o, scn):
    SC, D, sets, share, A = scen()
    r = A[(A.set == "new_R_B") & (A.scenario == scn) & (A.option == o) & (A.tier == "base")]
    return float(r.npv.iloc[0])


def t7_13():
    SC = scen()[0]
    rows, chk = [], []
    for o in RN.OPTIONS:
        vals = [_scen_npv(o, s) for s in SCEN_COLS]
        rows.append([NAME[o]] + [fmt_signed(lk2(v)) for v in vals])
        for s, v in zip(SCEN_COLS, vals):
            chk.append((f"{o} {s}", v, f"Base NPV mid CAPEX, {SC.LABEL[s]}, {lbl22(o)} (f_corr R, tank B)", D32))
    return T(pd.DataFrame(rows, columns=PRINTED["Table 7.13"]["columns"]), "final numbers", "yes", "scenarios.py (all_rows, set new_R_B)",
             checks=chk)


def lbl22(o):
    return {"TTES5k": "TTES 5,000 m3", "A1D": "A1D", "BWN110": "B-WN110", "A2D": "A2D", "Bstd": "B-std", "A4D": "A4D"}[o]


def t7_14():
    order = ["TTES5k", "A1D", "BWN110", "A2D", "Bstd", "A4D"]
    pb = lambda o: ">30 yr" if RN.simple_payback(o) > FC.LIFETIME else f"{RN.simple_payback(o):.1f} yr"
    rows = [["Storage type", "Buffer"] + ["Seasonal"] * 5,
            ["S_corr (k€/yr)"] + [fmt_num(RN.s_ref() + RN.tiers(o)["base"]) for o in order],
            ["Storage effect after calibration (k€/yr)"] + [fmt_num(RN.storage_after(o)) for o in order],
            ["Net annual (k€/yr)"] + [fmt_num(RN.tiers(o)["base"] - RN.opex(o)) for o in order],
            ["NPV mid, base case (M€)"] + [fmt_signed(RN.npv(o)) for o in order],
            ["NPV mid, lower bound (M€)"] + [fmt_signed(RN.npv(o, "lower")) for o in order],
            ["NPV mid, potential (M€)"] + [fmt_signed(RN.npv(o, "potential")) for o in order],
            ["Payback mid"] + [pb(o) for o in order],
            ["LCOH (€/MWh)"] + [f"{RN.lcoh(o):.0f}" for o in order],
            ["NPV low CAPEX (M€)"] + [fmt_signed(RN.npv(o, case="low")) for o in order],
            ["NPV mid, conservative (M€)"] + [fmt_signed(conservative(o)) for o in order],
            ["HP required", "No", "No", "No", "No", "Yes", "No"], ["NW loop served", "No", "No", "No", "No", "Yes", "No"]]
    chk = [(f"NPV base {o}", RN.npv(o), lock_npv(o, "base"), 2) for o in order]
    return T(pd.DataFrame(rows, columns=PRINTED["Table 7.14"]["columns"]), "final numbers", "yes", "final numbers", checks=chk)


# ============================================================================== Chapter 8, 9
def t8_1():
    df = pd.DataFrame(PRINTED["Table 8.1"]["rows"], columns=PRINTED["Table 8.1"]["columns"])
    y7 = [eff_years(r, "cleaned" if r == "B-WN110" else "recomputed").loc[7] for r in ("A1D", "A2D", "A4D", "B-WN110", "B-std WN", "B-std NW")]
    df.loc[df.Check == "ROSIM against field data", "Outcome"] = f"{min(y7):.0f}–{max(y7):.0f}% simulated (year 7) against 28–49% measured in the first cycles"
    return T(df, "partly", "one number (year-7 ROSIM range)", "printed table; ROSIM range from rosim_check.csv",
             "Qualitative table. Only the simulated ROSIM range changes (final year-7 efficiencies).")


def t9_1():
    rows, chk = [], []
    lab = {"TTES5k": "TTES 5,000 m3", "A1D": "Option A 1D (13.3 MW)", "BWN110": "B-WN 110 °C only", "A2D": "Option A 2D (26.6 MW)",
           "Bstd": "B standard (90 °C)", "A4D": "Option A 4D (53.2 MW)"}
    for o in ["TTES5k", "A1D", "BWN110", "A2D", "Bstd", "A4D"]:
        cx = RN.capex(o)
        rows.append([lab[o], "–" if o == "TTES5k" else f"{RN.f_corr(o):.3f}", fmt_num(RN.s_ref() + RN.tiers(o)["base"]),
                     f"{cx:.2f}" if o == "TTES5k" else f"{cx:.1f}", fmt_signed(RN.npv(o)), fmt_signed(lk2(conservative(o))), f"{RN.lcoh(o):.0f}"])
        chk.append((f"NPV {o}", RN.npv(o), lock_npv(o, "base"), 2))
    rows.append(["Dispatch-only reference", "–", fmt_num(RN.s_ref()), "0", "–", "–", "–"])
    return T(pd.DataFrame(rows, columns=PRINTED["Table 9.1"]["columns"]), "final numbers", "yes", "final numbers", checks=chk)


# ============================================================================== Appendix A
def ta_1():
    return printed("A.1")


def ta_2():
    import scenario_runner as sr
    base = dict(sr.SCENARIOS["base"], name="base", price_file=False)
    keys = [("none_ates", "constrained"), ("none_ates", "relaxed"), ("TTES5k", "relaxed"), ("A1D", "relaxed"), ("A2D", "relaxed"),
            ("A4D", "relaxed"), ("Bstd", "relaxed"), ("BWN110", "relaxed")]
    rows = [r[:] for r in PRINTED["Table A.2"]["rows"]]
    # 'vs P25/P75': change of the total dispatch saving against the P25/P75 run of the same configuration without the
    # SOC rule (data/extra_results.json, table_a2_p25p75: the grid-search runs, which need the raw data)
    import data_store as DS
    ref = DS.extra_table(C.EXTRA_RESULTS, "table_a2_p25p75").set_index("configuration").saving_p25p75_keur_yr
    p25 = {"A1D": ref["Option A 1D"], "A2D": ref["Option A 2D"], "A4D": ref["Option A 4D"],
           "Bstd": ref["Option B (WN+NW 90 C, 58 C return)"], "BWN110": ref["Option B WN 110 C"]}
    for r, (o, s) in zip(rows, keys):
        v = json.loads(sr._key(base, o, s, "").read_text())["saving_keur_yr"]
        r[2] = fmt_num(v)
        if o in p25:
            r[3] = fmt_signed(100 * (v / p25[o] - 1), 1) + "%"
    return T(pd.DataFrame(rows, columns=PRINTED["Table A.2"]["columns"]), "final numbers", "no (dispatch before calibration)",
             "saving column: scenario_runner cache; 'vs P25/P75': against the P25/P75 runs without the SOC rule (data/extra_results.json)",
             "The P25/P75 runs of the grid search need the raw data; their savings are included as data.")


# ============================================================================== Appendix B
def tb_1():
    return printed("B.1")


def tb_2():
    p = PRINTED["Table B.2"]
    rows = []
    for r in p["rows"]:
        total = float(r[5])
        rows.append(r + [fmt_signed(RN.npv("A1D", s=RN.tiers("A1D")["base"]) + RN.capex("A1D") - total)])
    be = RN.breakeven("A1D")
    return T(pd.DataFrame(rows, columns=p["columns"] + ["NPV of Option A 1D at this CAPEX (M€)"]), "partly", "text yes (break-even, NPV)",
             "components: printed (thesis input, unchanged); NPV with the final inputs: -CAPEX + (S_attr,base - OPEX) f_PV",
             f"All layouts are below the low scenario of 8.5 M€ and below the final break-even CAPEX of Option A 1D of {be:.2f} M€. "
             f"At 4 M€ the base-case NPV is {fmt_signed(RN.npv('A1D', s=RN.tiers('A1D')['base']) + RN.capex('A1D') - 4.0)} M€.",
             [("A1D break-even base", be, "Option A 1D break-even CAPEX base", 2)])


def _ttes_vol_rows(variant):
    d = ttes_sweep().query(f"variant == '{variant}'")
    out = {}
    for r in d.itertuples():
        out[(r.dispatch, r.volume_m3)] = r
    return out


def tb_3():
    import tank_capex as TC
    rows = []
    a = _ttes_vol_rows("actual")
    for pair in ("P20/P80", "P30/P60"):
        for v in (2500, 5000, 10000, 19000):
            r = a[(pair, v)]
            s = r.increment_eur_yr / 1000
            incr = RN.s_flex() + s
            lo, mid, hi = RN.tank_case_b(v)
            rows.append([pair, fmt_num(v), fmt_signed(s, 0), fmt_num(incr), fmt_num(TC.opex_keur(mid))] +
                        [fmt_signed(RN.tank_npv(v, incr, k)) for k in ("low", "mid", "high")] + [f"{r.cycles_per_yr:.0f}"])
    chk = [("TTES 5k P20/P80 NPV mid", RN.tank_npv(5000, RN.s_flex() + a[("P20/P80", 5000)].increment_eur_yr / 1000), "TTES NPV base, case B", 2)]
    return T(pd.DataFrame(rows, columns=PRINTED["Table B.3"]["columns"]), "final numbers", "yes (tank CAPEX)",
             "TTES sweep; CAPEX with the case-B method at each volume (low/mid/high = Duesseldorf / mean / Duisburg), OPEX 1.4%",
             "The case-B costing replaces the €200/300/400 per m3 of the printed table; for volumes other than 5,000 m3 it is "
             "a derivation for this table (not a locked number).", chk)


def tb_4():
    rows = []
    a, f = _ttes_vol_rows("actual"), _ttes_vol_rows("forecast")
    for pair in ("P20/P80", "P30/P60"):
        for v in (2500, 5000, 10000, 19000):
            sf, sa = f[(pair, v)].increment_eur_yr / 1000, a[(pair, v)].increment_eur_yr / 1000
            rows.append([pair, fmt_num(v), fmt_signed(sf, 0), fmt_signed(RN.tank_npv(v, RN.s_flex() + sf)), fmt_signed(sa, 0),
                         fmt_signed(RN.tank_npv(v, RN.s_flex() + sa))])
    return T(pd.DataFrame(rows, columns=PRINTED["Table B.4"]["columns"]), "final numbers", "yes (tank CAPEX)",
             "TTES sweep (forecast and actual thresholds), case-B costing at each volume")


def tb_5():
    SC, D, sets, share, A = scen()
    rows, chk = [], []
    for o in RN.OPTIONS:
        vals = []
        for g in ("g_m1", "base", "g_p1", "g_p2"):
            r = A[(A.set == "new_R_B") & (A.scenario == g) & (A.option == o) & (A.tier == "base")]
            vals.append(float(r.npv.iloc[0]))
        rows.append([NAME[o]] + [fmt_signed(lk2(v)) for v in vals])
        for g, v in zip(("g_m1", "g_p1", "g_p2"), (vals[0], vals[2], vals[3])):
            chk.append((f"{o} {g}", v, f"Base NPV mid CAPEX, {SC.LABEL[g]}, {lbl22(o)} (f_corr R, tank B)", D32))
    return T(pd.DataFrame(rows, columns=PRINTED["Table B.5"]["columns"]), "final numbers", "yes", "scenarios.py (Eq. B.1)", checks=chk)


def tb_6():
    SC, D, sets, share, A = scen()
    rows = [["S_ref"] + [fmt_num(D.loc[(s, "A1D")].S_ref) for s in SCEN_COLS],
            ["S_flex"] + [fmt_num(D.loc[(s, "A1D")].S_flex) for s in SCEN_COLS],
            ["S_stor, Option A 1D"] + [fmt_num(D.loc[(s, "A1D")].S_stor) for s in SCEN_COLS]]
    return T(pd.DataFrame(rows, columns=PRINTED["Table B.6"]["columns"]), "unchanged, scripted", "no (dispatch before calibration)",
             "data/derived/scenario_runs_cache (scenarios.decompose)")


def tb_7():
    SC, D, sets, share, A = scen()
    rows, chk = [], []
    for o in RN.OPTIONS:
        vals = []
        for s in ("rate_265", "base", "rate_600"):
            r = A[(A.set == "new_R_B") & (A.scenario == s) & (A.option == o) & (A.tier == "base")]
            vals.append(float(r.npv.iloc[0]))
        rows.append([NAME[o]] + [fmt_signed(lk2(v)) for v in vals])
        for s, v in (("rate_265", vals[0]), ("rate_600", vals[2])):
            chk.append((f"{o} {s}", v, f"Base NPV mid CAPEX, {SC.LABEL[s]}, {lbl22(o)} (f_corr R, tank B)", D32))
    return T(pd.DataFrame(rows, columns=PRINTED["Table B.7"]["columns"]), "final numbers", "yes", "scenarios.py", checks=chk)


def tb_8():
    p = {r[0]: r for r in PRINTED["Table B.8"]["rows"]}
    if not RN.buffer_cases_available():
        return printed("B.8", status="not reproducible", depends="text yes (f_corr)",
                       note="the buffer_use_cache (data/derived/buffer_use_cache.json) is missing; run scripts/model/m09_buffer_use_runs.py with the raw data.")
    rows = []
    for case, lab in (("n6", "Base case (n = 6)"), ("nobuf", "No buffers"), ("n1", "n = 1"), ("n2", "n = 2"), ("n3", "n = 3"), ("n4", "n = 4")):
        b = RN.buffer_case(case)
        pr = p[lab]
        rows.append([lab, fmt_num(b["S_ref"]), fmt_num(b["Z"]), fmt_num(b["S_flex"]), fmt_num(b["S_stor_A1D"]),
                     fmt_num(RN.f_corr("A1D") * b["S_stor_A1D"]), fmt_num(b["S_stor_TTES"]),
                     fmt_signed(RN.buffer_npv("A1D", case)), fmt_signed(RN.buffer_npv("TTES5k", case)), pr[6], pr[7], pr[8]])
    cols = ["Case", "S_ref", "Z", "S_flex", "S_stor A1D", "S_stor A1D after calibration (final f_corr)", "S_stor TTES",
            "NPV A1D base (M€)", "NPV TTES base, case B (M€)", "Swing", "Passages/day", "Reserve (h)"]
    chk = [("base case NPV A1D", RN.buffer_npv("A1D", "n6"), "Option A 1D NPV base (mid CAPEX)", 2),
           ("base case NPV TTES", RN.buffer_npv("TTES5k", "n6"), "TTES NPV base, case B", 2)]
    return T(pd.DataFrame(rows, columns=cols), "final numbers", "yes (calibrated storage effect, NPV)",
             "dispatch runs data/derived/buffer_use_cache (buffer_use_runs.py) and the base case; swing, passages and reserve: "
             "printed (buffer statistics of the hourly frames, unchanged dispatch)",
             f"After calibration with the final f_corr of Option A 1D ({RN.f_corr('A1D'):.3f}); NPV = -CAPEX + (S_flex + f_corr S_stor - OPEX) f_PV.",
             chk)


# ============================================================================== Appendix C
def tc_1():
    return printed("C.1", note="Module names as in the thesis; the repository paths are given in README.md and INVENTORY.md.")


def tc_2():
    return printed("C.2")


BUILDERS = {
    "3.1": t3_1, "3.2": t3_2, "3.3": t3_3, "3.4": t3_4, "3.5": t3_5, "3.6": t3_6, "4.1": t4_1, "5.1": t5_1, "5.2": t5_2,
    "6.1": t6_1, "6.2": t6_2, "6.3": t6_3, "6.4": t6_4, "6.5": t6_5, "6.6": t6_6, "6.7": t6_7, "6.8": t6_8, "6.9": t6_9,
    "6.10": t6_10, "6.11": t6_11, "6.12": t6_12, "6.13": t6_13, "7.1": t7_1, "7.2": t7_2, "7.3": t7_3, "7.4": t7_4,
    "7.5": t7_5, "7.6": t7_6, "7.7": t7_7, "7.8": t7_8, "7.9": t7_9, "7.10": t7_10, "7.11": t7_11, "7.12": t7_12,
    "7.13": t7_13, "7.14": t7_14, "8.1": t8_1, "9.1": t9_1, "A.1": ta_1, "A.2": ta_2, "B.1": tb_1, "B.2": tb_2, "B.3": tb_3,
    "B.4": tb_4, "B.5": tb_5, "B.6": tb_6, "B.7": tb_7, "B.8": tb_8, "C.1": tc_1, "C.2": tc_2}

