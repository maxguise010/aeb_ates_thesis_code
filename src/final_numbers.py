#!/usr/bin/env python3
"""Final numbers: discounted LCOH, tank costs, f_corr from the final ROSIM runs, NPV per attribution tier and ranking;
writes the locked-numbers files. Run by run_all.py."""
import inspect, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
HERE = C.RES_FINAL   # output folder of this script
import final_config as FC  # noqa: E402
import tank_capex as TC  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
CFGS = ["A1D", "A2D", "A4D", "BWN110", "Bstd"]
LABEL = {"TTES5k": "TTES 5,000 m3", "A1D": "Option A 1D", "A2D": "Option A 2D", "A4D": "Option A 4D",
         "BWN110": "B-WN110", "Bstd": "B standard"}
TIERS = ("lower", "base", "bridged", "potential")
AF, CRF = FC.annuity_factor(), FC.crf()
THESIS_PDF = C.THESIS_TEXT   # optional, not included (only used to look up where an old value is printed)
H5_TEX = C.THESIS_TEX        # optional, not included

# Published thesis values, comparison only (Master_Thesis_GL_Final (3).pdf; Tables 7.6, 7.8, 7.9, 7.10, 7.11, 7.14)
THESIS_REF = {
    "npv_base":      {"TTES5k": 10.89, "A1D": 0.09, "A2D": -6.63, "A4D": -22.37, "BWN110": -0.06, "Bstd": -15.13},
    "npv_lower":     {"TTES5k": -1.30, "A1D": -12.10, "A2D": -19.01, "A4D": -34.47, "BWN110": -12.11, "Bstd": -27.14},
    "npv_potential": {"TTES5k": 27.40, "A1D": 16.60, "A2D": 9.88, "A4D": -5.87, "BWN110": 16.44, "Bstd": 1.38},
    "npv_bridged":   {"TTES5k": 6.34, "A1D": -7.58},
    "breakeven_base": {"TTES5k": 12.4, "A1D": 10.6, "A2D": 10.4, "A4D": 7.6, "BWN110": 10.4, "Bstd": 7.1},
    "lcoh":          {"TTES5k": 22, "A1D": 207, "A2D": 175, "A4D": 192, "BWN110": 190, "Bstd": 428, "TTES19k": 47},
    "q_disch":       {"TTES5k": 5321, "A1D": 4143, "A2D": 7775, "A4D": 12782, "BWN110": 4513, "Bstd": 4256, "TTES19k": 9541},
}
TTES19K = dict(capex_meur=5.7, opex_keur=80.0)      # thesis Table 7.11 row, reproduced only (not re-costed)

LOCKED = []   # rows of the locked-numbers file


def put(chapter, key, value, unit, source, estimate="", old=None, old_search=None, decimals=2):
    """Register one locked number; records this script's calling line."""
    line = inspect.currentframe().f_back.f_lineno
    LOCKED.append(dict(chapter=chapter, key=key, value=round(float(value), decimals) if value == value else value,
                       unit=unit, script=f"src/final_numbers.py:{line}", source=source,
                       author_estimate=estimate, old_value=old, old_location=where_old(old_search) if old_search else ""))
    return value


_PDF_PAGES = None


def num_re(v, d=2):
    """Regex for an old thesis number as printed (sign as + or the minus sign, thousands comma)."""
    if v is None or v != v:
        return None
    s = f"{abs(v):,.{d}f}" if d else f"{abs(v):,.0f}"
    return r"(?<![\d.])" + (r"[−\-]" if v < 0 else r"[+]?") + re.escape(s).replace(",", r"(,|\{,\})") + r"(?![\d])"


def where_old(pattern):
    """Read-only lookup of an old value: H5_ttes.tex lines and thesis PDF pages (the .tex of other chapters is not on disk)."""
    global _PDF_PAGES
    out = []
    if H5_TEX.exists():
        for i, l in enumerate(H5_TEX.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if re.search(pattern, l):
                out.append(f"H5_ttes.tex:{i}")
    if _PDF_PAGES is None:
        txt = THESIS_PDF.read_text(encoding="utf-8") if THESIS_PDF.exists() else ""
        _PDF_PAGES = [(int(m.group(1)), seg) for m, seg in
                      zip(re.finditer(r"=====PAGE (\d+)=====", txt), re.split(r"=====PAGE \d+=====", txt)[1:])]
    pages = sorted({p for p, seg in _PDF_PAGES if re.search(pattern, seg.replace("\n", " "))})
    if pages:
        out.append("thesis PDF p. " + ", ".join(map(str, pages[:8])) + (" ..." if len(pages) > 8 else ""))
    return "; ".join(out)


# ------------------------------------------------------------------------------------------- LCOH
# discounted LCOH = (CAPEX + present value of the OPEX) / present value of the delivered heat (5 %, 30 years).
# With the same OPEX and energy every year this equals (CAPEX x CRF + OPEX) / E. E = the year-7 level of the
# calibrated dispatch, used for every year (no ramp-up years).
def lcoh_discounted(capex_eur, opex_eur_yr, e_mwh_yr, r=FC.RATE, n=FC.LIFETIME):
    """[CAPEX + sum_t OPEX_t (1+r)^-t] / [sum_t E_t (1+r)^-t], t = 1..n, constant OPEX and E (steady year-7 basis)."""
    t = np.arange(1, n + 1)
    d = (1.0 + r) ** -t
    opex = np.full(n, opex_eur_yr)
    e = np.full(n, e_mwh_yr)
    return (capex_eur + (opex * d).sum()) / (e * d).sum()


def lcoh_thesis(capex_eur, opex_eur_yr, e_mwh_yr):
    """Thesis Eq. 7.5: (CAPEX x f_ann + OPEX) / Q, f_ann = CRF rounded to 4 decimals as in the thesis text."""
    return (capex_eur * round(CRF, 4) + opex_eur_yr) / e_mwh_yr


# ------------------------------------------------------------------------------------------- inputs
def load_inputs():
    # thesis-state copy of the scenario_runner output (the current scenario_runner writes the final f_corr; the
    # "thesis" columns of this script must keep the thesis f_corr)
    sc = pd.read_csv(C.SCENARIO_RESULTS_THESIS)
    sc = sc[sc.scenario == "base"].set_index("option")
    # B-std: storage effect from the scenario cache at the configured return temperature (config.BSTD_T_RETURN_C)
    _sv = lambda o, s: json.loads((C.SAC_CACHE / f"thesis__base__{C.tagged(o)}__{s}.json").read_text())["saving_keur_yr"]
    sc.loc["Bstd", "S_stor"] = _sv("Bstd", "relaxed") - _sv("none_ates", "relaxed")
    sc.loc["Bstd", "gain_lower"] = sc.loc["Bstd", "f_corr"] * (_sv("Bstd", "constrained") - _sv("none_ates", "constrained"))
    sh = pd.read_csv(C.restart_shares())
    share = sh[sh.R_h == FC.BRIDGE_R_H].set_index("option").share_value_covered
    dr = pd.read_csv(C.RES_FCORR / "direct_runs.csv")
    ev = C.FCORR_EVALS
    base = {c: json.loads((ev / f"{C.tagged(c)}__T__1.00000__1.00000.json").read_text())["s_stor_eur_yr"] for c in CFGS}
    f, q, eff = {"R": {}, "D": {}}, {"R": {}, "D": {}}, {"R": {}, "D": {}}
    rc = pd.read_csv(C.RES_ROSIM / "rosim_check.csv")
    y7 = rc[rc.year == 7].set_index("run")
    run_of = {"A1D": "A1D", "A2D": "A2D", "A4D": "A4D", "BWN110": "B-WN110"}
    for c in ["A1D", "A2D", "A4D", "BWN110"]:
        for b in ("R", "D"):
            # f_corr = S_stor of the 'direct run' (ground loss scaled so that the dispatch efficiency hits the ROSIM efficiency
            # of year 7, within 0.3 percentage points) / S_stor of the uncalibrated run (scale 1); for basis R and D.
            r = dr[(dr.config == c) & (dr.setting == "T") & (dr.target == f"{b}7")].iloc[0]
            f[b][c] = r.s_stor_eur_yr / base[c]
            q[b][c] = r.discharged_mwh_yr
        eff["R"][c] = y7.loc[run_of[c], "cleaned"] if c == "BWN110" else y7.loc[run_of[c], "recomputed"]
        eff["D"][c] = y7.loc[run_of[c], "dispatch_reference"]
    for b in ("R", "D"):
        p = C.RES_FCORR / f"bstd_fcorr_v3_{b}.csv"
        if p.exists():
            r = pd.read_csv(p).iloc[0]
            f[b]["Bstd"], q[b]["Bstd"] = r.f_corr, r.discharged_mwh_yr
        else:
            f[b]["Bstd"], q[b]["Bstd"] = np.nan, np.nan
        col = "recomputed" if b == "R" else "dispatch_reference"
        eff[b]["Bstd"] = f'WN {y7.loc["B-std WN", col]:.2f} / NW {y7.loc["B-std NW", col]:.2f}'
    old_q = pd.read_csv(C.RES_FCORR / "followup_final_table.csv").set_index("config")
    oq = {"A1D": "A-1D", "A2D": "A-2D", "A4D": "A-4D", "BWN110": "B-WN110", "Bstd": "B-std"}
    q_old = {c: old_q.loc[oq[c], "q_discharged_direct_mwh_yr"] for c in CFGS}
    tl = pd.read_csv(C.RES_TTES / "ttes_low_temp_results.csv")
    trow = tl[(tl.schedule == "relaxed") & (tl.volume_m3 == 5000) & (tl.t_charge_c == 110)].iloc[0]
    return dict(sc=sc, share=share, base=base, f=f, q=q, eff=eff, q_old=q_old,
                q_ttes=float(trow.discharged_dispatch_mwh_yr), q_ttes_replay=float(trow.discharged_replay_mwh_yr))


# ------------------------------------------------------------------------------------------- steps
def step0(I):
    rows = []
    for c in CFGS:
        a = FC.ATES[c]
        cap, opx, qq = a["capex_meur"][1] * 1e6, a["opex_keur"] * 1e3, I["q_old"][c]
        rows.append(dict(option=c, capex_meur=a["capex_meur"][1], opex_keur=a["opex_keur"], q_disch_mwh=qq,
                         f_ann=round(CRF, 4), lcoh_reproduced=lcoh_thesis(cap, opx, qq), lcoh_thesis=THESIS_REF["lcoh"][c]))
    for name, cap, opx, qq in (("TTES5k", FC.TANK_OLD["capex_meur"][1], FC.TANK_OLD["opex_keur"], I["q_ttes"]),
                               ("TTES19k", TTES19K["capex_meur"], TTES19K["opex_keur"], THESIS_REF["q_disch"]["TTES19k"])):
        rows.append(dict(option=name, capex_meur=cap, opex_keur=opx, q_disch_mwh=qq, f_ann=round(CRF, 4),
                         lcoh_reproduced=lcoh_thesis(cap * 1e6, opx * 1e3, qq), lcoh_thesis=THESIS_REF["lcoh"][name]))
    t = pd.DataFrame(rows)
    t["diff"] = t.lcoh_reproduced - t.lcoh_thesis
    t.to_csv(HERE / "step0_lcoh_reproduction.csv", index=False, float_format="%.3f")
    return t


def step1(I, basis):
    rows = []
    entries = [(c, FC.ATES[c]["capex_meur"][1], FC.ATES[c]["opex_keur"], I["q_old"][c], I["q"][basis][c]) for c in CFGS]
    for k in ("D", "B", "H"):
        cx = TC.CASES[k]["capex_meur"]
        entries.append((f"TTES5k case {k}", cx, TC.opex_keur(cx), I["q_ttes"], I["q_ttes"]))
    entries.append(("TTES5k thesis CAPEX", FC.TANK_OLD["capex_meur"][1], FC.TANK_OLD["opex_keur"], I["q_ttes"], I["q_ttes"]))
    for name, cx, ox, q_old, q_new in entries:
        old_cx = FC.TANK_OLD["capex_meur"][1] if name.startswith("TTES") else cx
        old_ox = FC.TANK_OLD["opex_keur"] if name.startswith("TTES") else ox
        old = lcoh_thesis(old_cx * 1e6, old_ox * 1e3, q_old)
        new = lcoh_discounted(cx * 1e6, ox * 1e3, q_new)
        cap_part, opex_part = cx * 1e6 * CRF / q_new, ox * 1e3 / q_new
        simple = cx * 1e6 / (FC.LIFETIME * q_new)
        rows.append(dict(option=name, capex_meur=cx, opex_keur=ox, e_old_mwh=q_old, e_new_mwh=q_new, lcoh_thesis_method=old,
                         lcoh_new_method_old_inputs=lcoh_discounted(old_cx * 1e6, old_ox * 1e3, q_old),
                         lcoh_discounted=new, capital_part=cap_part, opex_part=opex_part, ratio_new_old=new / old,
                         capital_over_simple=cap_part / simple))
    t = pd.DataFrame(rows)
    t.to_csv(HERE / f"step1_lcoh_basis{basis}.csv", index=False, float_format="%.3f")
    return t


def step1_rates(I, basis):
    rows = []
    for r in FC.RATES_SENS:
        a = FC.ATES["A1D"]
        rows.append(dict(option="A1D", rate=r, crf=FC.crf(r), lcoh=lcoh_discounted(a["capex_meur"][1] * 1e6, a["opex_keur"] * 1e3, I["q"][basis]["A1D"], r)))
        for k in ("D", "B", "H"):
            cx = TC.CASES[k]["capex_meur"]
            rows.append(dict(option=f"TTES5k case {k}", rate=r, crf=FC.crf(r), lcoh=lcoh_discounted(cx * 1e6, TC.opex_keur(cx) * 1e3, I["q_ttes"], r)))
    t = pd.DataFrame(rows).pivot(index="option", columns="rate", values="lcoh")
    t.to_csv(HERE / f"step1_lcoh_rate_sensitivity_basis{basis}.csv", float_format="%.2f")
    return t


# four attribution tiers: lower = f x S_stor (constrained schedule); base = S_flex + f x S_stor; bridged = share of
# bridged restarts (R = 2 h) x S_flex + f x S_stor; potential = S_ref + S_flex + f x S_stor. f = f_corr.
def s_attr(I, o, f):
    r = I["sc"].loc[o]
    s_c = r.gain_lower / r.f_corr                    # uncalibrated constrained storage effect
    return dict(lower=f * s_c, base=r.S_flex + f * r.S_stor, bridged=I["share"][o] * r.S_flex + f * r.S_stor,
                potential=r.S_ref + r.S_flex + f * r.S_stor)


def npv(capex_meur, opex_keur, s_keur, af=AF):
    return -capex_meur + (s_keur - opex_keur) * af / 1000.0


def breakeven(s_keur, opex_rule=None, opex_keur=None, af=AF):
    """CAPEX with NPV = 0. Fixed OPEX: (S - OPEX) AF. OPEX = p x CAPEX: S AF / (1 + p AF)."""
    if opex_rule == "scaled":
        p = FC.TANK_OPEX_SHARE * 1000.0          # k EUR/yr per M EUR
        return s_keur * af / 1000.0 / (1.0 + p * af / 1000.0)
    return (s_keur - opex_keur) * af / 1000.0


def step3_fcorr(I):
    rows = []
    for c in CFGS:
        rows.append(dict(option=c, eff_thesis_y7=(f'WN {FC.EFF_THESIS_Y7["Bstd WN"]} / NW {FC.EFF_THESIS_Y7["Bstd NW"]}' if c == "Bstd" else FC.EFF_THESIS_Y7[c]),
                         eff_R_y7=I["eff"]["R"][c], eff_D_y7=I["eff"]["D"][c], f_thesis=I["sc"].loc[c].f_corr,
                         f_R=I["f"]["R"][c], f_D=I["f"]["D"][c], s_stor_uncal=I["base"][c] / 1000,
                         q_R=I["q"]["R"][c], q_D=I["q"]["D"][c], spacing_m=FC.SPACING_M[c]))
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "step3_fcorr.csv", index=False, float_format="%.4f")
    return t


def step4_ates(I, basis):
    rows = []
    for c in CFGS:
        a = FC.ATES[c]
        for fs, f in (("thesis", I["sc"].loc[c].f_corr), ("new", I["f"][basis][c])):
            s = s_attr(I, c, f)
            row = dict(option=c, f_set=fs, f_corr=f, capex_mid=a["capex_meur"][1], opex=a["opex_keur"])
            for t in TIERS:
                row[f"S_attr_{t}"] = s[t]
                row[f"npv_{t}"] = npv(a["capex_meur"][1], a["opex_keur"], s[t])
                row[f"breakeven_{t}"] = breakeven(s[t], opex_keur=a["opex_keur"])
            if c == "Bstd":   # NPV with the earlier code OPEX (359), for comparison
                row["npv_base_opex_code"] = npv(a["capex_meur"][1], a["opex_code_keur"], s["base"])
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(HERE / f"step4_npv_ates_basis{basis}.csv", index=False, float_format="%.3f")
    return t


def step4_tank(I):
    s = s_attr(I, "TTES5k", 1.0)
    rows = []
    for rule in ("scaled", "absolute"):
        for k in ("D", "B_low", "B", "B_high", "H"):
            cx = TC.CASES[k]["capex_meur"]
            ox = TC.opex_keur(cx, rule)
            row = dict(opex_rule=rule, case=k, capex_meur=cx, eur_per_m3=TC.CASES[k]["eur_per_m3"], opex_keur=ox,
                       lcoh=lcoh_discounted(cx * 1e6, ox * 1e3, I["q_ttes"]))
            for t in TIERS:
                row[f"npv_{t}"] = npv(cx, ox, s[t])
            rows.append(row)
        row = dict(opex_rule=rule, case="E (break-even)", capex_meur=np.nan, eur_per_m3=np.nan, opex_keur=np.nan, lcoh=np.nan)
        for t in TIERS:
            be = breakeven(s[t], rule, FC.TANK_OLD["opex_keur"])
            row[f"npv_{t}"] = 0.0
            row[f"breakeven_{t}"] = be
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "step4_npv_tank.csv", index=False, float_format="%.3f")
    return t, s


def ranking(nt, tank_npv, tier):
    d = {c: nt[(nt.option == c) & (nt.f_set == "new")][f"npv_{tier}"].iloc[0] for c in CFGS}
    d["TTES5k"] = tank_npv
    return [k for k, _ in sorted(d.items(), key=lambda x: -x[1])]


def main():
    I = load_inputs()
    hb = FC.FCORR_BASIS_HEADLINE
    s0 = step0(I)
    print("STEP 0 -- LCOH reproduction (thesis Eq. 7.5)\n", s0.round(2).to_string(index=False))
    l1 = {b: step1(I, b) for b in ("R", "D")}
    print(f"\nSTEP 1 -- LCOH, basis {hb}\n", l1[hb].round(3).to_string(index=False))
    rs = step1_rates(I, hb)
    print("\nLCOH rate sensitivity\n", rs.round(1).to_string())
    f3 = step3_fcorr(I)
    print("\nSTEP 3 -- f_corr\n", f3.round(4).to_string(index=False))
    n4 = {b: step4_ates(I, b) for b in ("R", "D")}
    tk, s_t = step4_tank(I)
    print(f"\nSTEP 4 -- ATES NPV, basis {hb}\n", n4[hb].round(2).to_string(index=False))
    print("\nTank\n", tk.round(2).to_string(index=False))

    # ---------------------------------------------------------------- change table
    nt = n4[hb]
    ch = []
    tank_b_scaled = tk[(tk.opex_rule == "scaled") & (tk.case == "B")].iloc[0]
    for c in CFGS:
        o, n = nt[(nt.option == c) & (nt.f_set == "thesis")].iloc[0], nt[(nt.option == c) & (nt.f_set == "new")].iloc[0]
        l = l1[hb].set_index("option").loc[c]
        ch.append(dict(option=c, eff_thesis=f3.set_index("option").loc[c].eff_thesis_y7, eff_new=I["eff"][hb][c],
                       f_thesis=o.f_corr, f_new=n.f_corr, s_attr_base_thesis=o.S_attr_base, s_attr_base_new=n.S_attr_base,
                       **{f"npv_{t}_thesis": (THESIS_REF[f"npv_{t}"].get(c, o[f"npv_{t}"])) for t in TIERS},
                       **{f"npv_{t}_new": n[f"npv_{t}"] for t in TIERS},
                       lcoh_thesis=THESIS_REF["lcoh"][c], lcoh_new=l.lcoh_discounted))
    ch.append(dict(option="TTES5k (case B, OPEX 1.4 %)", eff_thesis="", eff_new="", f_thesis=1.0, f_new=1.0,
                   s_attr_base_thesis=s_t["base"], s_attr_base_new=s_t["base"],
                   **{f"npv_{t}_thesis": THESIS_REF[f"npv_{t}"].get("TTES5k", np.nan) for t in TIERS},
                   **{f"npv_{t}_new": tank_b_scaled[f"npv_{t}"] for t in TIERS},
                   lcoh_thesis=THESIS_REF["lcoh"]["TTES5k"], lcoh_new=tank_b_scaled.lcoh))
    cht = pd.DataFrame(ch)
    cht.to_csv(HERE / "step4_changes_vs_thesis.csv", index=False, float_format="%.3f")
    print("\nCHANGES\n", cht.round(2).T.to_string())
    ranks = {}
    for t in TIERS:
        for k in ("D", "B", "H"):
            row = tk[(tk.opex_rule == "scaled") & (tk.case == k)].iloc[0]
            ranks[(t, k)] = ranking(nt, row[f"npv_{t}"], t)
    json.dump({f"{t}|{k}": v for (t, k), v in ranks.items()}, open(HERE / "step4_rankings.json", "w"), indent=1)
    for (t, k), v in ranks.items():
        print(f"ranking {t:9s} tank case {k}: {' > '.join(LABEL[x] for x in v)}")

    # ---------------------------------------------------------------- locked numbers
    ch5, ch7, ch6 = "Ch. 5 (TTES)", "Ch. 7 (economics)", "Ch. 6 / 3.4 (ROSIM, f_corr)"
    put(ch7, "Discount rate (real)", FC.RATE * 100, "%", "thesis assumption", "author assumption")
    put(ch7, "Lifetime", FC.LIFETIME, "yr", "thesis assumption", "author assumption", decimals=0)
    put(ch7, "Annuity (present value) factor f_PV", AF, "-", "(1 - 1.05^-30) / 0.05", old="15.37", old_search=r"15\.37", decimals=4)
    put(ch7, "Capital recovery factor f_ann", CRF, "-", "1 / f_PV", old="0.0651", old_search=r"0\.0651", decimals=5)
    k_tank = TC.CASES
    put(ch5, "Tank CAPEX case D (Danish benchmark)", k_tank["D"]["capex_meur"], "M EUR", "DEA ch. 141 p. 62, a V^b x V")
    put(ch5, "Tank specific CAPEX case D", k_tank["D"]["eur_per_m3"], "EUR/m3", "DEA ch. 141 p. 62", decimals=0)
    put(ch5, "Tank CAPEX case B (thesis base)", k_tank["B"]["capex_meur"], "M EUR", "mean of scaled+escalated Duesseldorf, Duisburg",
        "AUTHOR ESTIMATE (derived from published projects, not a published price)", old="1.5", old_search=r"1\.5 million")
    put(ch5, "Tank specific CAPEX case B", k_tank["B"]["eur_per_m3"], "EUR/m3", "case B / 5,000 m3", "AUTHOR ESTIMATE", old="300", old_search=r"300/m", decimals=0)
    put(ch5, "Tank CAPEX case B low (Duesseldorf)", k_tank["B"]["low_meur"], "M EUR", "10 M EUR x (5000/36000)^0.691 x 129.5/90.9", "AUTHOR ESTIMATE", old="1.0", old_search=r"1\.0 million")
    put(ch5, "Tank CAPEX case B high (Duisburg)", k_tank["B"]["high_meur"], "M EUR", "20 M EUR x (5000/43000)^0.691 x 129.5/90.9", "AUTHOR ESTIMATE", old="2.0", old_search=r"2\.0 million")
    put(ch5, "Tank CAPEX case H (high sensitivity)", k_tank["H"]["capex_meur"], "M EUR", "author's earlier estimate", "AUTHOR ESTIMATE, no published source")
    put(ch5, "Ratio case B / case D", k_tank["B"]["capex_meur"] / k_tank["D"]["capex_meur"], "-", "tank_capex.py")
    put(ch5, "Ratio case H / case B", k_tank["H"]["capex_meur"] / k_tank["B"]["capex_meur"], "-", "tank_capex.py")
    for n_, r_ in TC.project_table().items():
        put(ch5, f"{n_} scaled to 5,000 m3 (project price year)", r_["scaled_meur"], "M EUR", "x (5000/V)^0.691", "derived")
        if r_["scaled_escalated_meur"] == r_["scaled_escalated_meur"]:
            put(ch5, f"{n_} scaled and escalated to 2026-09", r_["scaled_escalated_meur"], "M EUR", "x CBS 129.5/90.9", "derived")
    put(ch5, "Tank OPEX, headline (1.4 % of case B)", TC.opex_keur(k_tank["B"]["capex_meur"]), "k EUR/yr", "1.4 % of CAPEX (thesis Table 7.4 rule)",
        "author assumption (thesis OPEX rule)", old="21", old_search=r"\b21\b")
    put(ch5, "TTES discharged heat (dispatch, relaxed, 5,000 m3)", I["q_ttes"], "MWh/yr", "ttes_low_temp_results.csv", old="5,321", old_search=r"5,321|5\{,\}321", decimals=0)
    for k in ("D", "B", "B_low", "B_high", "H"):
        row = tk[(tk.opex_rule == "scaled") & (tk.case == k)].iloc[0]
        est = "" if k == "D" else "AUTHOR ESTIMATE"
        put(ch7, f"TTES LCOH case {k}", row.lcoh, "EUR/MWh", "discounted LCOH, OPEX 1.4 %", est, old="22" if k == "B" else None,
            old_search=r"€22/MWh|22/MWh" if k == "B" else None, decimals=1)
        for t in TIERS:
            put(ch7, f"TTES NPV {t}, case {k}", row[f"npv_{t}"], "M EUR", "NPV = -CAPEX + (S_attr - OPEX) f_PV", est,
                old=THESIS_REF[f"npv_{t}"].get("TTES5k") if k == "B" else None,
                old_search=num_re(THESIS_REF[f"npv_{t}"].get("TTES5k")) if k == "B" else None)
    for rule in ("scaled", "absolute"):
        row = tk[(tk.opex_rule == rule) & (tk.case == "E (break-even)")].iloc[0]
        for t in TIERS:
            put(ch7, f"TTES break-even CAPEX {t} (OPEX {rule})", row[f"breakeven_{t}"], "M EUR", "case E",
                old=THESIS_REF["breakeven_base"]["TTES5k"] if (t == "base" and rule == "absolute") else None,
                old_search=r"12\.4" if (t == "base" and rule == "absolute") else None)
    for c in CFGS:
        n = nt[(nt.option == c) & (nt.f_set == "new")].iloc[0]
        l = l1[hb].set_index("option").loc[c]
        if c == "Bstd":   # one line per loop (a single line would be nan, because the B-std entry is a "WN / NW" string)
            y7 = pd.read_csv(C.RES_ROSIM / "rosim_check.csv").query("year == 7").set_index("run")
            for lp in ("WN", "NW"):
                put(ch6, f"{LABEL[c]} {lp} efficiency year 7 (basis {hb})", y7.loc[f"B-std {lp}", "recomputed"], "%",
                    "rosim_check.csv", old=FC.EFF_THESIS_Y7[f"Bstd {lp}"],
                    old_search=re.escape(f"{FC.EFF_THESIS_Y7[f'Bstd {lp}']:.2f}"))
        else:
            put(ch6, f"{LABEL[c]} efficiency year 7 (basis {hb})", I["eff"][hb][c], "%", "rosim_check.csv",
                old=FC.EFF_THESIS_Y7.get(c), old_search=(re.escape(f"{FC.EFF_THESIS_Y7[c]:.2f}") if c in FC.EFF_THESIS_Y7 else None))
        put(ch6, f"{LABEL[c]} f_corr (basis {hb})", n.f_corr, "-", "S_stor(direct run) / S_stor(scale 1)", old=I["sc"].loc[c].f_corr,
            old_search=re.escape(f"{I['sc'].loc[c].f_corr:.3f}"), decimals=3)
        put(ch6, f"{LABEL[c]} f_corr (basis D, dispatch reference)", I["f"]["D"][c], "-", "same, dispatch reference targets", decimals=3)
        put(ch6, f"{LABEL[c]} warm-cold spacing", FC.SPACING_M[c], "m", "ROSIM settings (rosim_geometry.csv)",
            "AUTHOR CHOICE (125 m spacing)" if c == "BWN110" else "", decimals=0)
        put(ch7, f"{LABEL[c]} storage effect after calibration", n.f_corr * I["base"][c] / 1000, "k EUR/yr", "f_corr x S_stor", decimals=1)
        put(ch7, f"{LABEL[c]} OPEX", FC.ATES[c]["opex_keur"], "k EUR/yr", "thesis Table 7.4", "AUTHOR ASSUMPTION (thesis OPEX)", decimals=0)
        for t in TIERS:
            put(ch7, f"{LABEL[c]} NPV {t} (mid CAPEX)", n[f"npv_{t}"], "M EUR", "NPV formula", old=THESIS_REF[f"npv_{t}"].get(c),
                old_search=num_re(THESIS_REF[f"npv_{t}"].get(c)))
            be_old = THESIS_REF["breakeven_base"].get(c) if t == "base" else None
            put(ch7, f"{LABEL[c]} break-even CAPEX {t}", n[f"breakeven_{t}"], "M EUR", "(S_attr - OPEX) f_PV",
                old=be_old, old_search=num_re(be_old, 1) if be_old else None)
        put(ch7, f"{LABEL[c]} Q_disch (LCOH energy)", l.e_new_mwh, "MWh/yr", "dispatch run calibrated to year 7", old=THESIS_REF["q_disch"][c],
            old_search=num_re(THESIS_REF["q_disch"][c], 0), decimals=0)
        put(ch7, f"{LABEL[c]} LCOH (discounted)", l.lcoh_discounted, "EUR/MWh", "Eq. 7.5 discounted form", old=THESIS_REF["lcoh"][c],
            old_search=rf"\b{THESIS_REF['lcoh'][c]}\b.{{0,3}}(/MWh|$)|LCOH.{{0,60}}\b{THESIS_REF['lcoh'][c]}\b", decimals=0)
        put(ch7, f"{LABEL[c]} LCOH capital part", l.capital_part, "EUR/MWh", "CAPEX x CRF / E", decimals=1)
        put(ch7, f"{LABEL[c]} LCOH OPEX part", l.opex_part, "EUR/MWh", "OPEX / E", decimals=1)
    lk = pd.DataFrame(LOCKED)
    lk.to_csv(HERE / "chapter_number_map.csv", index=False)
    with open(HERE / "final_numbers_locked.md", "w", encoding="utf-8") as fh:
        fh.write("# Locked final numbers\n\nGenerated by `src/final_numbers.py`. One line per number: "
                 "value, unit, producing script line, source, and where the old value stands (read-only lookup in the "
                 "thesis text when it is present in data/private/; it is not included in the repository). "
                 "**AUTHOR ESTIMATE / ASSUMPTION** marks numbers that do not come from a published source.\n"
                 f"\nf_corr basis of the headline: {hb} (ROSIM reported, year 7). Rate {FC.RATE:.0%}, {FC.LIFETIME} years, "
                 "energy basis: steady year-7 calibration applied to every year, no ramp-up.\n")
        for chap, g in lk.groupby("chapter", sort=False):
            fh.write(f"\n## {chap}\n\n")
            for _, r in g.iterrows():
                est = f" **{r.author_estimate}**" if r.author_estimate else ""
                old = f" | old: {r.old_value}" if pd.notna(r.old_value) and r.old_value is not None else ""
                loc = f" ({r.old_location})" if r.old_location else ""
                fh.write(f"- {r.key}: **{r.value} {r.unit}** | `{r.script}` | {r.source}{est}{old}{loc}\n")
    print(f"\n{len(lk)} locked numbers written")
    # Step 6 checks
    # check: annuity factor 15.3725 and CRF 0.06505 (5 %, 30 years).
    assert abs(AF - 15.3725) < 5e-5 and abs(CRF - 0.06505) < 5e-6, (AF, CRF)
    print(f"annuity factor {AF:.4f}, CRF {CRF:.5f}")


if __name__ == "__main__":
    main()
