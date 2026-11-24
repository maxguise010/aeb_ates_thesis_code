"""Derived numbers of Chapters 3 and 4, consistency checks of data/extra_results.json and a smoke test of the model
code. Run by run_all.py (plain, --check and --smoke)."""
import importlib
import json
import math
import py_compile
import sys
import traceback

import numpy as np
import pandas as pd

import config as C
import data_store as DS

OUT_NUMBERS = C.OUTPUT / "derived_numbers.csv"
R, N = 0.05, 30
F_PV = (1 - (1 + R) ** -N) / R
F_ANN = 1 / F_PV
ORDER = ["TTES5k", "A1D", "A2D", "A4D", "BWN110", "Bstd"]
SPREADS = {"spread_075": 0.75, "base": 1.0, "spread_125": 1.25, "spread_150": 1.5}
LEVELS = {"level_080": 0.8, "base": 1.0, "level_120": 1.2}
TANK_MW = 50.0          # scenario_runner.run_saving, tank run: large_charge_max_mw = large_discharge_max_mw = 50.0


def inputs():
    sc = DS.extra_table(C.EXTRA_RESULTS, "scenarios_v2").set_index(["option", "scenario"])
    ce = DS.extra_table(C.EXTRA_RESULTS, "cycle_efficiency").set_index("run")
    dfac = DS.extra_table(C.EXTRA_RESULTS, "demand_factor").set_index("quantity")
    lc = pd.read_csv(C.RES_FINAL / "step1_lcoh_basisR.csv").set_index("option")
    q = {o: float(lc.loc[o, "e_new_mwh"]) for o in ORDER if o in lc.index}
    q["TTES5k"] = float(lc.loc["TTES5k case B", "e_new_mwh"])
    rc = pd.read_csv(C.RES_ROSIM / "rosim_check.csv")
    y7 = rc[rc.year == 7].set_index("run")
    eta_rosim = {"A1D": y7.loc["A1D", "reported"], "A2D": y7.loc["A2D", "reported"], "A4D": y7.loc["A4D", "reported"],
                 "BWN110": y7.loc["B-WN110", "cleaned"],            # artefact weeks 339-340 interpolated (thesis convention)
                 "Bstd": (y7.loc["B-std WN", "reported"] + y7.loc["B-std NW", "reported"]) / 2}   # the two doublets together
    lk = pd.read_csv(C.RES_FINAL / "chapter_number_map.csv").drop_duplicates("key").set_index("key").value
    f63 = pd.read_csv(C.FIG63_STATS, index_col=0)
    return sc, ce, dfac, q, eta_rosim, lk, f63


def ground_loss_share(cfg):
    """Ground loss of the uncalibrated Option A run as a share of the charged heat (as report_tables.t6_2)."""
    import fcorr_runs as fr
    j = json.loads((C.FCORR_EVALS / f"{fr.tag(cfg, 'T', 1.0, 1.0)}.json").read_text())
    ch, dis = j["charged_mwh_yr"], j["discharged_mwh_yr"]
    added = j["e_end_mwh"] / len(fr.YEARS)
    loss = ch - dis - added - j["balance_residual_pct"] / 100 * ch
    return 100 * loss / ch


def cold_well_temperature(project):
    """Flow-weighted mean temperature (column T) of the water extracted from the cold wells in year 7 (ROSIM weeks 312-363)."""
    import audit as au
    eff, per = au.read_eff_full(project)
    end = au.YEAR_END[5]                      # year 7
    sl = slice(end - 51, end + 1)
    v = t = 0.0
    for w, d in per.items():
        if w.startswith("cold"):
            q = np.clip(-d["Q"].values[sl], 0, None)
            v += q.sum(); t += (q * d["T"].values[sl]).sum()
    return t / v


def derive():
    sc, ce, dfac, q, eta_rosim, lk, f63 = inputs()
    rows = []

    def put(i, v, unit, where, definition):
        rows.append(dict(id=i, value=v, unit=unit, thesis_location=where, definition=definition))

    S = lambda o, s="base": sc.loc[(o, s)]
    cal = lambda o, s="base": float(S(o, s).S_stor * S(o, s).f_corr_R)
    # ---------------------------------------------------------------- general factors (Chapter 3)
    put("3-econ|present value factor f_PV", F_PV, "-", "Section 3 economics, Eq. LCOH/NPV", "(1 - 1.05^-30) / 0.05")
    put("3-econ|annuity factor f_ann", F_ANN, "-", "Section 3 economics", "1 / f_PV")
    f1 = F_PV / (F_PV - 1 / (1 + R))
    put("lcoh|first-year-zero factor f1", f1, "-", "Section 4 LCOH", "f_PV / (f_PV - 1/1.05)")
    # ---------------------------------------------------------------- per option
    tab = {}
    for o in ORDER:
        b = S(o)
        lcoh_pv = (b.capex_meur * 1e6 + b.opex_keur * 1e3 * F_PV) / (q[o] * F_PV)
        lcoh_ann = (b.capex_meur * 1e6 * F_ANN + b.opex_keur * 1e3) / q[o]
        put(f"lcoh|{o}|PV costs (M EUR)", b.capex_meur + b.opex_keur * F_PV / 1000, "M EUR", "Section 4 LCOH", "CAPEX + OPEX f_PV")
        put(f"lcoh|{o}|PV heat (GWh)", q[o] * F_PV / 1000, "GWh", "Section 4 LCOH", "Q f_PV")
        put(f"lcoh|{o}|LCOH = PV costs / PV heat (EUR/MWh)", lcoh_pv, "EUR/MWh", "Section 4 LCOH", "PV(costs) / PV(heat)")
        put(f"lcoh|{o}|LCOH = (CAPEX f_ann + OPEX) / Q (EUR/MWh)", lcoh_ann, "EUR/MWh", "Eq. LCOH", "(CAPEX f_ann + OPEX) / Q")
        put(f"lcoh|{o}|difference of the two LCOH forms (EUR/MWh)", lcoh_pv - lcoh_ann, "EUR/MWh", "check", "must be 0")
        put(f"lcoh|{o}|LCOH without discharge in year 1 (EUR/MWh)", float(b.lcoh_eur_mwh) * f1, "EUR/MWh", "Section 4 LCOH", "LCOH x f1")
        sc_ = cal(o)
        tab[o] = dict(S_cal=sc_, share=100 * sc_ / b.S_corr_total, Q=q[o], vpm=sc_ / q[o] * 1000, lcoh=float(b.lcoh_eur_mwh),
                      vl=100 * (sc_ / q[o] * 1000) / float(b.lcoh_eur_mwh), f=float(b.f_corr_R), npv=float(b.npv_meur),
                      npv1=float(b.npv_meur) + float(b.S_stor) * (1 - float(b.f_corr_R)) * F_PV / 1000)
        for k, u in (("S_cal", "k EUR/yr"), ("share", "%"), ("Q", "MWh/yr"), ("vpm", "EUR/MWh"), ("vl", "%"), ("npv1", "M EUR")):
            put(f"option|{o}|{k}", tab[o][k], u, "Chapter 4", {"S_cal": "S_stor x f_corr", "share": "S_cal / S_corr_total", "Q": "step1_lcoh_basisR e_new_mwh",
                                                             "vpm": "S_cal / Q x 1000", "vl": "value per MWh / LCOH", "npv1": "NPV + S_stor (1-f) f_PV / 1000"}[k])
        # spread and level analysis
        for term in ("S_ref", "S_flex", "S_stor", "S_cal", "npv_meur"):
            val = lambda s: cal(o, s) if term == "S_cal" else float(S(o, s)[term])
            xs = {f: val(s) for s, f in SPREADS.items()}
            slope = (xs[1.5] - xs[0.75]) / 0.75
            slope1 = (xs[1.0] - xs[0.75]) / 0.25
            put(f"spread|{o}|{term}|overall slope", slope, "per unit spread", "Section 4 spread", "(X(1.5) - X(0.75)) / 0.75")
            put(f"spread|{o}|{term}|first-interval slope", slope1, "per unit spread", "Section 4 spread", "(X(1) - X(0.75)) / 0.25")
            put(f"spread|{o}|{term}|flat price (overall slope)", xs[1.0] - slope, "", "Section 4 spread", "X(1) - overall slope")
            put(f"spread|{o}|{term}|flat price (first-interval slope)", xs[1.0] - slope1, "", "Section 4 spread", "X(1) - first-interval slope")
            for s, f in LEVELS.items():
                put(f"level|{o}|{term}|level {f}", val(s), "", "Section 4 level", "scenario value")
        n = {f: float(S(o, s).npv_meur) for s, f in SPREADS.items()}
        zero = np.nan
        for a, b_ in ((0.75, 1.0), (1.0, 1.25), (1.25, 1.5)):
            if n[a] <= 0 <= n[b_] or n[a] >= 0 >= n[b_]:
                zero = a + (b_ - a) * (0 - n[a]) / (n[b_] - n[a])
                break
        if np.isnan(zero):
            zero = 1.0 - n[1.0] / ((n[1.5] - n[0.75]) / 0.75)
        put(f"spread|{o}|NPV zero spread factor", zero, "-", "Section 4 spread", "linear interpolation, else extrapolation with the overall slope")
    # ---------------------------------------------------------------- the six tables (row order as printed)
    T = {}
    T["tab:arbitrage-share"] = [[tab[o][k] for k in ("S_cal", "share", "Q", "vpm", "lcoh", "vl")] for o in ORDER]
    sens_rows = ["spread_075", "base", "spread_125", "spread_150", "level_080", "level_120"]
    T["tab:arbitrage-sensitivity"] = [[float(S("A1D", s).S_ref), float(S("A1D", s).S_flex), cal("TTES5k", s), cal("A1D", s),
                                       float(S("TTES5k", s).npv_meur), float(S("A1D", s).npv_meur)] for s in sens_rows]
    fp = []
    for term, o, first in (("S_ref", "A1D", False), ("S_flex", "A1D", False), ("S_cal", "TTES5k", False), ("S_cal", "A1D", True)):
        val = lambda s: cal(o, s) if term == "S_cal" else float(S(o, s)[term])
        x1 = val("base")
        slope = (val("base") - val("spread_075")) / 0.25 if first else (val("spread_150") - val("spread_075")) / 0.75
        flat = x1 - slope
        fp.append([x1, slope, flat, "all" if flat <= 0 else 100 * slope / x1])
    T["tab:flat-price"] = fp
    eg = []
    for o, k in (("A1D", "A1D"), ("A2D", "A2D"), ("A4D", "A4D"), ("BWN110", "BWN110"), ("Bstd", "B-std 58 C (headline)")):
        ec = float(ce.loc[k, "eta_cycle_pct"])
        qq = float(ce.loc[k, "eta_disp_pct"]) if o.startswith("A") else "--"
        eg.append([ec, qq, eta_rosim[o], eta_rosim[o] / ec, tab[o]["f"]])
    T["tab:efficiency-gap"] = eg
    T["tab:fcorr-bound"] = [[tab[o]["f"], tab[o]["npv"], tab[o]["npv1"], tab[o]["npv1"] - tab[o]["npv"]] for o in ORDER]
    import fcorr_runs as fr
    mw = {"TTES5k": TANK_MW, "A1D": fr.CAP["A1D"], "A4D": fr.CAP["A4D"]}
    tv = []
    for k in ("capex", "opex", "mw", "capex_mw", "Q", "Q_mw", "capex_q", "S_cal", "S_attr", "vpm", "lcoh", "npv"):
        r_ = []
        for o in ("TTES5k", "A1D", "A4D"):
            b = S(o)
            r_.append({"capex": float(b.capex_meur), "opex": float(b.opex_keur), "mw": mw[o], "capex_mw": float(b.capex_meur) / mw[o],
                       "Q": q[o], "Q_mw": q[o] / mw[o], "capex_q": float(b.capex_meur) * 1e6 / q[o], "S_cal": tab[o]["S_cal"],
                       "S_attr": float(b.S_attr_base), "vpm": tab[o]["vpm"], "lcoh": tab[o]["lcoh"], "npv": tab[o]["npv"]}[k])
        tv.append(r_)
    T["tab:tank-vs-ates"] = tv
    for lab, M in T.items():
        for i, r_ in enumerate(M):
            for j, v in enumerate(r_, start=1):
                rows.append(dict(id=f"{lab}|row{i}|col{j}", value=v, unit="", thesis_location=f"{lab}, row {i}, column {j}", definition="table cell"))
    # ---------------------------------------------------------------- prose numbers
    shares = [tab[o]["share"] for o in ORDER[1:]]
    P = {}
    P["4-arb-share|storage effect share, tank (%)"] = tab["TTES5k"]["share"]
    P["4-arb-share|storage effect share, aquifer min (%)"] = min(shares)
    P["4-arb-share|storage effect share, aquifer max (%)"] = max(shares)
    P["4-arb-share|dispatch share min (%)"] = 100 - max(tab[o]["share"] for o in ORDER)
    P["4-arb-share|dispatch share max (%)"] = 100 - min(tab[o]["share"] for o in ORDER)
    P["4-arb-share|value per MWh / LCOH, max over options (%)"] = max(tab[o]["vl"] for o in ORDER)
    heat = float(dfac.loc["simulated non-BEC heat df1038_none_tank_constrained (MWh/yr)", "f1038"])
    qa = [q[o] for o in ("A1D", "A2D", "A4D", "BWN110", "Bstd")]
    P["4-arb-why|heat produced by the heat exchangers (GWh/yr)"] = heat / 1000
    P["4-arb-why|discharged heat share of HX production, min (%)"] = 100 * min(qa) / heat
    P["4-arb-why|discharged heat share of HX production, max (%)"] = 100 * max(qa) / heat
    P["4-arb-why|charged energy above one doublet (%)"] = float(f63.loc["share_energy_above_13_3", "charge"])
    nobuf = float(lk["Report: Table B.8 S_ref, nobuf"])
    P["4-arb-why|S_ref without buffers (k EUR/yr)"] = nobuf
    P["4-arb-why|share of S_ref carried by the buffers (%)"] = 100 * (1 - nobuf / float(S("A1D").S_ref))
    P["4-arb-sens|S_flex at level 0.8 (k EUR/yr)"] = float(S("A1D", "level_080").S_flex)
    P["4-arb-sens|S_flex at level 1.2 (k EUR/yr)"] = float(S("A1D", "level_120").S_flex)
    sf = [float(S("A1D", s).S_flex) for s in SPREADS]
    P["4-arb-sens|S_flex min over spreads (k EUR/yr)"] = min(sf)
    P["4-arb-sens|S_flex max over spreads (k EUR/yr)"] = max(sf)
    sr = [float(S("A1D", s).S_ref) for s in SPREADS]
    P["4-arb-sens|S_ref rise per 0.25 of spread (k EUR/yr)"] = (sr[3] - sr[0]) / 3
    P["4-arb-sens|S_ref change at level 0.8 (%)"] = 100 * (float(S("A1D", "level_080").S_ref) / sr[1] - 1)
    P["4-arb-sens|S_ref change at level 1.2 (%)"] = 100 * (float(S("A1D", "level_120").S_ref) / sr[1] - 1)
    P["4-arb-sens|S_stor A1D change per 25% of spread (%)"] = 100 * (cal("A1D", "spread_125") / cal("A1D") - 1)
    P["4-arb-sens|S_stor tank change per 25% of spread (%)"] = 100 * (cal("TTES5k", "spread_125") / cal("TTES5k") - 1)
    P["4-arb-sens|NPV A1D change per 25% of spread (M EUR)"] = (float(S("A1D", "spread_150").npv_meur) - float(S("A1D", "spread_075").npv_meur)) / 3
    P["4-arb-sens|lowest tank NPV, level 0.8 (M EUR)"] = float(S("TTES5k", "level_080").npv_meur)
    zero = {r["id"].split("|")[1]: r["value"] for r in rows if r["id"].endswith("NPV zero spread factor")}
    P["4-arb-sens|break-even spread factor A1D"] = zero["A1D"]
    P["4-arb-sens|break-even spread factor B-WN110"] = zero["BWN110"]
    P["4-arb-sens|zero-NPV spread factor A2D (linear extrapolation)"] = zero["A2D"]
    P["4-arb-sens|zero-NPV spread factor A4D (linear extrapolation)"] = zero["A4D"]
    P["4-arb-sens|zero-NPV spread factor B-std (linear extrapolation)"] = zero["Bstd"]
    P["4-arb-isolate|S_ref step 0.75 to 1.0 (k EUR/yr)"] = sr[1] - sr[0]
    P["4-arb-isolate|S_ref step 1.0 to 1.25 (k EUR/yr)"] = sr[2] - sr[1]
    P["4-arb-isolate|S_ref step 1.25 to 1.5 (k EUR/yr)"] = sr[3] - sr[2]
    ua = [float(S("A1D", s).S_stor) for s in SPREADS]
    P["4-arb-isolate|A1D uncalibrated step 0.75 to 1.0 (k EUR/yr)"] = ua[1] - ua[0]
    P["4-arb-isolate|A1D uncalibrated step 1.0 to 1.25 (k EUR/yr)"] = ua[2] - ua[1]
    P["4-arb-isolate|A1D uncalibrated step 1.25 to 1.5 (k EUR/yr)"] = ua[3] - ua[2]
    fr_ref, fr_flex = fp[0], fp[1]
    P["4-arb-isolate|S_ref at a flat price (k EUR/yr)"] = fr_ref[2]
    P["4-arb-isolate|S_flex at a flat price (k EUR/yr)"] = fr_flex[2]
    P["4-arb-isolate|S_ref + S_flex at a flat price (M EUR/yr)"] = (fr_ref[2] + fr_flex[2]) / 1000
    P["4-arb-isolate|S_ref + S_flex at the 2023-2025 spread (k EUR/yr)"] = fr_ref[0] + fr_flex[0]
    P["4-arb-isolate|spread-dependent part of S_ref (k EUR/yr)"] = fr_ref[1]
    P["4-arb-isolate|spread-dependent part of S_flex (k EUR/yr)"] = fr_flex[1]
    dep = fr_ref[1] + fr_flex[1]
    P["4-arb-isolate|spread-dependent value of S_ref + S_flex (k EUR/yr)"] = dep
    P["4-arb-isolate|434 / largest storage effect"] = dep / max(tab[o]["S_cal"] for o in ORDER)
    P["4-arb-isolate|434 / smallest storage effect"] = dep / min(tab[o]["S_cal"] for o in ORDER)
    ratio = {o: eg[i][3] for i, o in enumerate(("A1D", "A2D", "A4D", "BWN110", "Bstd"))}
    P["4-rosim-gap|eta_ROSIM / eta_cycle, Option A min"] = min(ratio[o] for o in ("A1D", "A2D", "A4D"))
    P["4-rosim-gap|eta_ROSIM / eta_cycle, B-WN110"] = ratio["BWN110"]
    P["4-rosim-gap|eta_ROSIM / eta_cycle, B-std"] = ratio["Bstd"]
    projects = {"A1D": "Option A 1 Doublet 8 years changed parameters", "A2D": "Option A 2 doublets 8 years changed parameters",
                "A4D": "Option A 4 doublets 8 years changed parameters"}
    for i, o in enumerate(("A1D", "A2D", "A4D")):
        P[f"4-rosim-gap|Q_disch/Q_ch {o} (%)"] = eg[i][1]
        P[f"4-rosim-gap|Q_disch/Q_ch minus eta_ROSIM {o} (pp)"] = eg[i][1] - eg[i][2]
        P[f"4-rosim-gap|eta_cycle minus eta_ROSIM {o} (pp)"] = eg[i][0] - eg[i][2]
        P[f"4-rosim-gap|ground loss share of charged heat {o} (%)"] = ground_loss_share(o)
        P[f"4-rosim-gap|cold-well delivered temperature {o}, year 7 (C)"] = cold_well_temperature(projects[o])
        name = {"A1D": "Option A 1D", "A2D": "Option A 2D", "A4D": "Option A 4D"}[o]
        P[f"4-rosim-gap|f_corr basis D {o}"] = float(lk[f"{name} f_corr (basis D, dispatch reference)"])
    ch1 = float(ce.loc["A1D", "charged_mwh_yr"])
    P["4-rosim-gap|A1D discharge at eta_ROSIM x charged heat (MWh/yr)"] = eta_rosim["A1D"] / 100 * ch1
    P["4-rosim-gap|A1D charged heat (MWh/yr)"] = ch1
    P["4-rosim-gap|A1D discharged heat, calibrated run (MWh/yr)"] = q["A1D"]
    P["4-rosim-gap|A1D discharged heat, uncalibrated run (MWh/yr)"] = float(ce.loc["A1D", "discharged_mwh_yr"])
    t, a = S("TTES5k"), S("A1D")
    P["4-tank|CAPEX ratio A1D / tank"] = a.capex_meur / t.capex_meur
    P["4-tank|OPEX ratio A1D / tank"] = a.opex_keur / t.opex_keur
    P["4-tank|S_attr A1D above tank (%)"] = 100 * (a.S_attr_base / t.S_attr_base - 1)
    d_npv = t.npv_meur - a.npv_meur
    d_cx, d_ox, d_s = a.capex_meur - t.capex_meur, (a.opex_keur - t.opex_keur) * F_PV / 1000, (a.S_attr_base - t.S_attr_base) * F_PV / 1000
    P["4-tank|NPV difference tank - A1D (M EUR)"] = d_npv
    P["4-tank|CAPEX term of the NPV difference (M EUR)"] = d_cx
    P["4-tank|OPEX term of the NPV difference (M EUR)"] = d_ox
    P["4-tank|saving term of the NPV difference (M EUR)"] = d_s
    P["4-tank|CAPEX share of the NPV difference (%)"] = 100 * d_cx / d_npv
    P["4-tank|OPEX share of the NPV difference (%)"] = 100 * d_ox / d_npv
    P["4-tank|saving share of the NPV difference (%)"] = 100 * d_s / d_npv
    P["4-tank|PV of net annual benefit, tank (M EUR)"] = (t.S_attr_base - t.opex_keur) * F_PV / 1000
    P["4-tank|PV of net annual benefit, A1D (M EUR)"] = (a.S_attr_base - a.opex_keur) * F_PV / 1000
    P["4-tank|storage effect / S_flex, tank (%)"] = 100 * tab["TTES5k"]["S_cal"] / t.S_flex
    P["4-tank|storage effect / S_flex, A1D (%)"] = 100 * tab["A1D"]["S_cal"] / a.S_flex
    P["4-tank|CAPEX per MW ratio A1D / tank"] = (a.capex_meur / mw["A1D"]) / (t.capex_meur / mw["TTES5k"])
    P["4-tank|discharge per MW ratio A1D / tank"] = (q["A1D"] / mw["A1D"]) / (q["TTES5k"] / mw["TTES5k"])
    P["4-tank|CAPEX per MWh/yr ratio A1D / tank"] = (a.capex_meur / q["A1D"]) / (t.capex_meur / q["TTES5k"])
    P["4-tank|value per MWh ratio A1D / tank"] = tab["A1D"]["vpm"] / tab["TTES5k"]["vpm"]
    P["4-lcoh|PV of costs, tank (M EUR)"] = t.capex_meur + t.opex_keur * F_PV / 1000
    P["4-lcoh|PV of discharged heat, tank (GWh)"] = q["TTES5k"] * F_PV / 1000
    P["4-lcoh|LCOH tank from the PV ratio (EUR/MWh)"] = (t.capex_meur * 1e6 + t.opex_keur * 1e3 * F_PV) / (q["TTES5k"] * F_PV)
    P["4-lcoh|PV of costs, A1D (M EUR)"] = a.capex_meur + a.opex_keur * F_PV / 1000
    P["4-lcoh|PV of discharged heat, A1D (GWh)"] = q["A1D"] * F_PV / 1000
    P["4-lcoh|LCOH A1D from the PV ratio (EUR/MWh)"] = (a.capex_meur * 1e6 + a.opex_keur * 1e3 * F_PV) / (q["A1D"] * F_PV)
    P["4-lcoh|PV of heat lower without year-1 discharge (%)"] = 100 * (1 / (1 + R)) / F_PV
    P["4-lcoh|LCOH higher without year-1 discharge (%)"] = 100 * (f1 - 1)
    P["4-lcoh|LCOH A1D without year-1 discharge (EUR/MWh)"] = float(a.lcoh_eur_mwh) * f1
    P["4-lcoh|LCOH A2D without year-1 discharge (EUR/MWh)"] = float(S("A2D").lcoh_eur_mwh) * f1
    P["3-econ|annuity factor f_ann"] = F_ANN
    P["3-econ|present value factor f_PV"] = F_PV
    for k, v in P.items():
        rows.append(dict(id=k, value=v, unit="", thesis_location="prose", definition="see derive()"))
    return pd.DataFrame(rows, columns=["id", "value", "unit", "thesis_location", "definition"])


def tolerance(r):
    dec = 0 if pd.isna(r.decimals) else int(r.decimals)
    tol = 10.0 ** -dec
    p = str(r.printed).lstrip("+-")
    if dec == 0 and p.endswith("0") and "about" in str(r.context):
        z = len(p) - len(p.rstrip("0"))
        tol = 0.5 * 10 ** z
    return tol + 1e-9


def check():
    """derived numbers against the printed values; extra-analysis data against the outputs of the chain."""
    d = pd.read_csv(OUT_NUMBERS)
    ref = pd.read_csv(C.REF_DERIVED, keep_default_na=False, na_values=[""])
    val = dict(zip(d.id, d.value))
    out = []
    for r in ref.itertuples():
        key = f"{r.table}|row{int(r.row)}|col{int(r.col)}" if isinstance(r.table, str) and r.table else r.id
        v = val.get(key)
        if v is None:
            out.append(dict(id=r.id, printed=r.printed, reproduced="", abs_diff="", tolerance="", status="NOT REPRODUCED", location=r.location)); continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = None
        if pd.isna(r.value) or fv is None:
            ok = str(v).strip() == str(r.printed).strip()
            out.append(dict(id=r.id, printed=r.printed, reproduced=v, abs_diff="", tolerance="text", status="AGREE" if ok else "DISAGREE", location=r.location))
            continue
        tol = tolerance(r)
        diff = abs(fv - float(r.value))
        out.append(dict(id=r.id, printed=r.printed, reproduced=round(fv, 6), abs_diff=round(diff, 6), tolerance=round(tol, 6),
                        status="AGREE" if diff <= tol else "DISAGREE", location=r.location))
    t = pd.DataFrame(out)
    t.to_csv(C.RESULTS / "derived_numbers_check.csv", index=False)
    print(t.status.value_counts().to_string())
    if (t.status != "AGREE").any():
        print(t[t.status != "AGREE"].to_string(index=False))
    e = extra_checks()
    return int((t.status == "DISAGREE").sum()) + int((e.status == "FAIL").sum())


def extra_checks():
    """The included results of the extra analyses that overlap with outputs of the chain must equal them."""
    rows = []

    def chk(item, a, b, tol):
        ok = abs(float(a) - float(b)) <= tol
        rows.append(dict(item=item, data_value=a, chain_value=b, tolerance=tol, status="ok" if ok else "FAIL"))

    sc = DS.extra_table(C.EXTRA_RESULTS, "scenarios_v2")
    dec = pd.read_csv(C.RES_SCEN / "scenario_decomposition.csv")
    m = sc.merge(dec, left_on=["scenario", "option"], right_on=["scenario", "option"], suffixes=("", "_chain"))
    for r in m.itertuples():
        for k in ("S_ref", "S_flex", "S_stor"):
            chk(f"scenarios_v2 {r.option} {r.scenario} {k} = scenario_decomposition.csv", getattr(r, k), getattr(r, k + "_chain"), 0.0005)
    npv = pd.read_csv(C.RES_FINAL / "step4_npv_ates_basisR.csv")
    npv = npv[npv.f_set == "new"].set_index("option")
    base = sc[sc.scenario == "base"].set_index("option")
    for o in npv.index:
        chk(f"scenarios_v2 {o} base NPV = step4_npv_ates_basisR.csv", base.loc[o, "npv_meur"], npv.loc[o, "npv_base"], 0.0006)
        chk(f"scenarios_v2 {o} base f_corr = step4_npv_ates_basisR.csv", base.loc[o, "f_corr_R"], npv.loc[o, "f_corr"], 0.0006)
    lc = pd.read_csv(C.RES_FINAL / "step1_lcoh_basisR.csv").set_index("option")
    for o in ("A1D", "A2D", "A4D", "BWN110", "Bstd"):
        chk(f"scenarios_v2 {o} base LCOH = step1_lcoh_basisR.csv", base.loc[o, "lcoh_eur_mwh"], lc.loc[o, "lcoh_discounted"], 0.0006)
    ce = DS.extra_table(C.EXTRA_RESULTS, "cycle_efficiency").set_index("run")
    import fcorr_runs as fr
    for o in ("A1D", "A2D", "A4D"):
        j = json.loads((C.FCORR_EVALS / f"{fr.tag(o, 'T', 1.0, 1.0)}.json").read_text())
        chk(f"cycle_efficiency {o} eta_cycle = fcorr_evals (scale 1)", ce.loc[o, "eta_cycle_pct"], j["eta_cycle_pct"], 0.00006)
        chk(f"cycle_efficiency {o} charged = fcorr_evals (scale 1)", ce.loc[o, "charged_mwh_yr"], j["charged_mwh_yr"], 0.0006)
    cal = DS.extra_table(C.EXTRA_RESULTS, "calibration").set_index("run")
    rc = pd.read_csv(C.RES_ROSIM / "rosim_check.csv")
    y7 = rc[rc.year == 7].set_index("run")
    for o in ("A1D", "A2D", "A4D"):
        chk(f"calibration {o} target = rosim_check.csv year 7 (reported)", cal.loc[o, "target_pct"], y7.loc[o, "reported"], 0.00006)
    sp = DS.extra_table(C.EXTRA_RESULTS, "spacing_effective").set_index("run")
    geo = pd.read_csv(C.RES_ROSIM / "rosim_geometry.csv").set_index("run")
    for o in ("A1D", "A2D", "A4D"):
        chk(f"spacing_effective {o} L (settings) = rosim_geometry.csv", sp.loc[o, "L_settings_m"], float(str(geo.loc[o, "warm_cold_m"]).split(";")[0]), 0.0006)
    import tab_6_3_6_10_thermal_radius as M
    for o in ("A1D", "A2D", "A4D"):
        proj, wb = M.RUNS[o][0], M.RUNS[o][1]
        chk(f"spacing_effective {o} V_inj = final schedule year 7", sp.loc[o, "v_inj_m3_yr"], M.injected_volume(wb), 0.0006)
    oos = DS.extra_table(C.EXTRA_RESULTS, "oos_results")
    zcache = json.loads((C.SAC_CACHE / "thesis__base__none_ates__relaxed.json").read_text())["per_year"]
    for v in sorted(set(oos.Z_2025)):
        chk("oos_results Z_2025 = zero-capacity relaxed run 2025 (scenario cache)", v, zcache["2025"], 0.0006)
    for name in ("mixing", "oos_summary", "table_a2_p25p75"):
        n = len(DS.extra_table(C.EXTRA_RESULTS, name))
        rows.append(dict(item=f"{name}: {n} rows, used as published result (needs the raw data; no overlapping chain output)",
                         data_value="", chain_value="", tolerance="", status="ok"))
    e = pd.DataFrame(rows)
    e.to_csv(C.RESULTS / "extra_results_check.csv", index=False)
    print(f"extra-analysis data: {len(e)} checks, {int((e.status == 'FAIL').sum())} failed")
    if (e.status == "FAIL").any():
        print(e[e.status == "FAIL"].to_string(index=False))
    return e


METHOD_IMPORT = ["ates_thermal_v2", "ptes_loss", "aeb_buffer_rule", "ptes_stratification", "ttes_thermal_replay", "scenarios_v4",
                 "simulation_v7", "simulation_ates_option_a_v7", "simulation_ates_three_loops_v7", "utils", "data_pipeline",
                 "forecast_prices"]
METHOD_COMPILE = ["buffer_cycling", "restart_split", "run_short_window", "generate_rosim_1and2doublet_schedules",
                  "generate_option_b_wn_110c_schedule"]


def smoke():
    """The model code needs the raw AEB data to run, but it can be imported and the patched dispatch functions of the
    thesis runs (scenario_runner._adapter: Option A and B model patches, tank patch) can be built without data."""
    rows = []
    for m in METHOD_IMPORT:
        try:
            importlib.import_module(m)
            rows.append(dict(item=f"import {m}", status="ok", detail=""))
        except Exception as e:
            rows.append(dict(item=f"import {m}", status="FAIL", detail=repr(e)[:300]))
    for m in METHOD_COMPILE:
        try:
            py_compile.compile(str(C.SRC / f"{m}.py"), doraise=True)
            rows.append(dict(item=f"compile {m}", status="ok", detail=""))
        except Exception as e:
            rows.append(dict(item=f"compile {m}", status="FAIL", detail=repr(e)[:300]))
    try:
        import scenario_runner as sr
        ad = sr._adapter()
        need = {"A", "B", "tank", "b_v6", "ttes_model", "replay"}
        ok = need <= set(ad) and "run_ates_option_a_soc_driven" in ad["A"] and "run_ates_three_loops_soc_driven" in ad["B"]
        rows.append(dict(item="scenario_runner._adapter (model source patches of the thesis runs)", status="ok" if ok else "FAIL",
                         detail=", ".join(sorted(ad))))
    except Exception:
        rows.append(dict(item="scenario_runner._adapter (model source patches of the thesis runs)", status="FAIL",
                         detail=traceback.format_exc()[-300:]))
    t = pd.DataFrame(rows)
    t.to_csv(C.RESULTS / "smoke_test.csv", index=False)
    print(t.to_string(index=False))
    return int((t.status != "ok").sum())


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        sys.exit(1 if smoke() else 0)
    elif "--check" in sys.argv:
        sys.exit(1 if check() else 0)
    else:
        d = derive()
        d.to_csv(OUT_NUMBERS, index=False)
        print(f"{len(d)} derived numbers written to {OUT_NUMBERS}")

