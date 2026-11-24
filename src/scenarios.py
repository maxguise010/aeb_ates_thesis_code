#!/usr/bin/env python3
"""Price and demand scenarios with the final inputs; with the argument 'analytic' also the growth and discount-rate
scenarios. Run by run_all.py."""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
HERE = C.RES_SCEN   # output folder
import final_config as FC  # noqa: E402
import tank_capex as TC  # noqa: E402
import final_numbers as FN  # noqa: E402

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
OPT = ["TTES5k", "A1D", "BWN110", "A2D", "Bstd", "A4D"]
ATES = ["A1D", "BWN110", "A2D", "Bstd", "A4D"]
FAM = {"TTES5k": "none_tank", "A1D": "none_ates_a", "A2D": "none_ates_a", "A4D": "none_ates_a",
       "BWN110": "none_ates", "Bstd": "none_ates"}
# level_080 and level_120 (price level x0.8 and x1.2) are only in the code, not in a thesis table: sensitivity.
DISPATCH_SCEN = ["base", "demand_m20", "demand_p20", "spread_075", "spread_125", "spread_150", "level_080", "level_120"]
LABEL = {"base": "Base", "demand_m20": "Demand -20 %", "demand_p20": "Demand +20 %", "spread_075": "Spread x0.75",
         "spread_125": "Spread x1.25", "spread_150": "Spread x1.5", "level_080": "Price level x0.8",
         "level_120": "Price level x1.2", "g_m1": "Savings growth g = -1 %/yr", "g_p1": "Savings growth g = +1 %/yr",
         "g_p2": "Savings growth g = +2 %/yr", "rate_265": "Discount rate 2.65 % (ACM)", "rate_600": "Discount rate 6.0 % (SDE++)"}
IN_THESIS = {"base": "Table 7.13", "demand_m20": "Table 7.13", "demand_p20": "Table 7.13", "spread_075": "Table 7.13",
             "spread_125": "Table 7.13", "spread_150": "Table 7.13", "level_080": "code only (no thesis table)",
             "level_120": "code only (no thesis table)", "g_m1": "Table B.5", "g_p1": "Table B.5", "g_p2": "Table B.5",
             "rate_265": "Table B.7", "rate_600": "Table B.7"}
CACHE = C.SAC_CACHE
OLD_FPV = 15.37                                                       # thesis Eq. 7.1 as printed (old column only)
THESIS_T713 = {   # thesis Table 7.13 (PDF p. 80), base tier, mid CAPEX -- comparison only
    "TTES5k": [10.89, 11.17, 10.17, 10.27, 11.52, 12.15], "A1D": [0.09, 0.74, -0.94, -0.62, 0.86, 1.68],
    "BWN110": [-0.06, 0.57, -1.10, -0.68, 0.61, 1.34], "A2D": [-6.63, -6.01, -7.53, -7.73, -5.45, -4.13],
    "Bstd": [-15.13, -14.44, -16.16, -15.67, -14.55, -13.89], "A4D": [-22.37, -22.31, -23.04, -23.98, -20.66, -18.71]}
THESIS_B5 = {"TTES5k": [9.60, 12.40, 14.19], "A1D": [-1.25, 1.67, 3.54], "BWN110": [-1.39, 1.50, 3.35],
             "A2D": [-8.09, -4.94, -2.93], "Bstd": [-16.43, -13.59, -11.78], "A4D": [-23.92, -20.56, -18.42]}   # g -1, +1, +2
THESIS_B7 = {"TTES5k": [15.04, 9.59], "A1D": [3.64, -1.02], "BWN110": [3.44, -1.16], "A2D": [-3.17, -7.72],
             "Bstd": [-12.76, -15.88], "A4D": [-19.82, -23.17]}                                                  # 2.65 %, 6.0 %
G = {"g_m1": -0.01, "g_p1": 0.01, "g_p2": 0.02}
# discount-rate scenarios (thesis Table B.7): 2.65 % (ACM) and 6.0 % (SDE++); the annuity factor changes with them.
RATES = {"rate_265": 0.0265, "rate_600": 0.06}


def af(r=FC.RATE, n=FC.LIFETIME):
    return FC.annuity_factor(r, n)


# growth scenario (thesis Table B.5, Eq. B.1): the saving grows by g per year, the OPEX stays constant.
def fpv_g(g, r=FC.RATE, n=FC.LIFETIME):
    """thesis Eq. B.1: sum_t (1+g)^(t-1) / (1+r)^t."""
    return (1 - ((1 + g) / (1 + r)) ** n) / (r - g)


def saving(scn, o, sch):
    return json.loads((CACHE / f"thesis__{scn}__{C.tagged(o)}__{sch}.json").read_text())["saving_keur_yr"]


def check_cache():
    import scenario_runner as sr
    worst = 0.0
    for scn in DISPATCH_SCEN[:6]:
        for o in OPT + ["none_ates", "none_ates_a", "none_tank"]:
            for sch in ("relaxed", "constrained"):
                f = sr._key(dict(sr.SCENARIOS[scn], name=scn, price_file=False), o, sch, "")
                worst = max(worst, abs(json.loads(f.read_text())["saving_keur_yr"] - saving(scn, o, sch)))
    return worst


def decompose():
    rows = []
    for scn in DISPATCH_SCEN:
        s_ref, z = saving(scn, "none_ates", "constrained"), saving(scn, "none_ates", "relaxed")
        for o in OPT:
            rows.append(dict(scenario=scn, option=o, S_ref=s_ref, Z=z, S_flex=z - s_ref,
                             S_stor=saving(scn, o, "relaxed") - saving(scn, FAM[o], "relaxed"),
                             S_stor_c=saving(scn, o, "constrained") - saving(scn, FAM[o], "constrained")))
    return pd.DataFrame(rows).set_index(["scenario", "option"])


def inputs():
    I = FN.load_inputs()
    share = I["share"].to_dict()
    f_old = {o: float(I["sc"].loc[o].f_corr) for o in ATES}
    sets = {"old": dict(f={**f_old, "TTES5k": 1.0}, tank=(FC.TANK_OLD["capex_meur"][1], FC.TANK_OLD["opex_keur"]), fpv=OLD_FPV)}
    for b in ("R", "D"):
        for k in ("B", "D", "H"):
            cx = TC.CASES[k]["capex_meur"]
            sets[f"new_{b}_{k}"] = dict(f={**{o: I["f"][b][o] for o in ATES}, "TTES5k": 1.0}, tank=(cx, TC.opex_keur(cx)), fpv=af())
    return sets, share


def capex_opex(o, s):
    if o == "TTES5k":
        return s["tank"]
    a = FC.ATES[o]
    return a["capex_meur"][1], a["opex_keur"]


# attribution tiers as final_numbers.s_attr; the bridged share stays at its base value in every scenario.
def tiers(d, o, f, share):
    return dict(lower=f * d.S_stor_c, base=d.S_flex + f * d.S_stor, bridged=share[o] * d.S_flex + f * d.S_stor,
                potential=d.S_ref + d.S_flex + f * d.S_stor)


def npv_generic(cx, ox, s_attr, fpv_s, fpv_o):
    return -cx + (s_attr * fpv_s - ox * fpv_o) / 1000.0


def all_rows(D, sets, share):
    rows = []
    for name, s in sets.items():
        for o in OPT:
            cx, ox = capex_opex(o, s)
            f = s["f"][o]
            # dispatch scenarios
            for scn in DISPATCH_SCEN:
                t = tiers(D.loc[(scn, o)], o, f, share)
                for tier, sa in t.items():
                    rows.append(dict(set=name, scenario=scn, option=o, tier=tier, S_attr=sa, capex=cx, opex=ox,
                                     npv=npv_generic(cx, ox, sa, s["fpv"], s["fpv"]),
                                     breakeven_capex=(sa - ox) * s["fpv"] / 1000 if o != "TTES5k" or name == "old"
                                     else sa * s["fpv"] / 1000 / (1 + FC.TANK_OPEX_SHARE * 1000 * s["fpv"] / 1000)))
            # analytic scenarios on the base-scenario saving
            t = tiers(D.loc[("base", o)], o, f, share)
            for key, g in G.items():
                fg = fpv_g(g) if name != "old" else round(fpv_g(g), 2)          # the thesis printed f_PV(g) to 2 decimals
                for tier, sa in t.items():
                    rows.append(dict(set=name, scenario=key, option=o, tier=tier, S_attr=sa, capex=cx, opex=ox,
                                     npv=npv_generic(cx, ox, sa, fg, s["fpv"]), breakeven_capex=np.nan))
            for key, r in RATES.items():
                fr_ = af(r) if name != "old" else round(af(r), 2)
                for tier, sa in t.items():
                    rows.append(dict(set=name, scenario=key, option=o, tier=tier, S_attr=sa, capex=cx, opex=ox,
                                     npv=npv_generic(cx, ox, sa, fr_, fr_), breakeven_capex=np.nan))
    return pd.DataFrame(rows)


def rank_of(df_scn):
    """rank 1 = highest NPV, over the six options."""
    return df_scn.npv.rank(ascending=False, method="first").astype(int)


def old_vs_new(A):
    base = A[A.tier == "base"]
    out = []
    scen = DISPATCH_SCEN + list(G) + list(RATES)
    for scn in scen:
        g = {k: base[(base.set == k) & (base.scenario == scn)].set_index("option") for k in base.set.unique()}
        rk = {k: rank_of(v) for k, v in g.items()}
        for o in OPT:
            old, new = g["old"].loc[o].npv, g["new_R_B"].loc[o].npv
            row = dict(scenario=scn, scenario_label=LABEL.get(scn, scn), thesis_table=IN_THESIS[scn], option=o,
                       old_npv=old, new_npv=new, change=new - old,
                       sign_change=("YES" if np.sign(old) != np.sign(new) else ""),
                       old_rank=int(rk["old"][o]), new_rank=int(rk["new_R_B"][o]),
                       rank_change=("YES" if rk["old"][o] != rk["new_R_B"][o] else ""),
                       new_npv_fcorr_D=g["new_D_B"].loc[o].npv, new_rank_fcorr_D=int(rk["new_D_B"][o]),
                       new_npv_tank_D=g["new_R_D"].loc[o].npv, new_rank_tank_D=int(rk["new_R_D"][o]),
                       new_npv_tank_H=g["new_R_H"].loc[o].npv, new_rank_tank_H=int(rk["new_R_H"][o]))
            row["sign_change_any_case"] = "YES" if any(np.sign(old) != np.sign(row[c]) for c in
                                                       ("new_npv", "new_npv_fcorr_D", "new_npv_tank_D", "new_npv_tank_H")) else ""
            row["rank_change_any_case"] = "YES" if any(row["old_rank"] != row[c] for c in
                                                       ("new_rank", "new_rank_fcorr_D", "new_rank_tank_D", "new_rank_tank_H")) else ""
            out.append(row)
    return pd.DataFrame(out)


def reproduction(A):
    """old column against the printed thesis tables."""
    b = A[(A.tier == "base") & (A.set == "old")].set_index(["scenario", "option"]).npv
    d = []
    for o, v in THESIS_T713.items():
        for scn, t in zip(DISPATCH_SCEN[:6], v):
            d.append(dict(table="7.13", scenario=scn, option=o, printed=t, recomputed=b[(scn, o)]))
        for scn, t in zip(G, THESIS_B5[o]):
            d.append(dict(table="B.5", scenario=scn, option=o, printed=t, recomputed=b[(scn, o)]))
        for scn, t in zip(RATES, THESIS_B7[o]):
            d.append(dict(table="B.7", scenario=scn, option=o, printed=t, recomputed=b[(scn, o)]))
    d = pd.DataFrame(d); d["diff"] = d.recomputed - d.printed
    return d


def breakevens(D, sets, share):
    """break-even spread factor k, growth g and real discount rate (NPV = 0, base tier, mid CAPEX) per option."""
    rows = []
    ks = {"spread_075": 0.75, "base": 1.0, "spread_125": 1.25, "spread_150": 1.5}
    for name in ("old", "new_R_B", "new_D_B", "new_R_D", "new_R_H"):
        s = sets[name]
        for o in OPT:
            cx, ox = capex_opex(o, s)
            f = s["f"][o]
            npv_k = [npv_generic(cx, ox, tiers(D.loc[(scn, o)], o, f, share)["base"], s["fpv"], s["fpv"]) for scn in ks]
            kk = list(ks.values())
            k0 = np.nan
            for i in range(3):
                if np.sign(npv_k[i]) != np.sign(npv_k[i + 1]):
                    k0 = kk[i] + (0 - npv_k[i]) * (kk[i + 1] - kk[i]) / (npv_k[i + 1] - npv_k[i])
            sa = tiers(D.loc[("base", o)], o, f, share)["base"]
            fn_g = lambda g: npv_generic(cx, ox, sa, fpv_g(g), s["fpv"])
            fn_r = lambda r: npv_generic(cx, ox, sa, af(r), af(r))
            rows.append(dict(set=name, option=o, npv_base=npv_k[1],
                             breakeven_spread_k=k0 if not np.isnan(k0) else ("> 1.5" if npv_k[-1] < 0 else "< 0.75"),
                             breakeven_growth_pct=bisect(fn_g, -0.10, 0.045) * 100,
                             breakeven_rate_pct=bisect(fn_r, 0.001, 0.60) * 100,
                             breakeven_capex_base=(sa - ox) * s["fpv"] / 1000 if (o != "TTES5k" or name == "old")
                             else sa * s["fpv"] / 1000 / (1 + FC.TANK_OPEX_SHARE * 1000 * s["fpv"] / 1000)))
    return pd.DataFrame(rows)


def bisect(fn, a, b, it=80):
    fa, fb = fn(a), fn(b)
    if np.sign(fa) == np.sign(fb):
        return np.nan
    for _ in range(it):
        m = 0.5 * (a + b); fm = fn(m)
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b, fb = m, fm
    return 0.5 * (a + b)


# Figure 7.2: change of the base NPV of A1D (mid CAPEX) per sensitivity, with old and final inputs.
def tornado(D, sets, share):
    """Figure 7.2 bars for A1D that follow from the inputs here (change in base NPV at mid CAPEX, M EUR)."""
    out = []
    for name in ("old", "new_R_B", "new_D_B"):
        s = sets[name]; o = "A1D"
        cx, ox = FC.ATES[o]["capex_meur"][1], FC.ATES[o]["opex_keur"]
        lo, hi = FC.ATES[o]["capex_meur"][0], FC.ATES[o]["capex_meur"][2]
        f = s["f"][o]; fpv = s["fpv"]
        d = D.loc[("base", o)]
        t = tiers(d, o, f, share)
        nb = npv_generic(cx, ox, t["base"], fpv, fpv)
        bar = {"CAPEX low": cx - lo, "CAPEX high": cx - hi}
        for r, k in ((0.06, "rate 6.0 %"), (0.0265, "rate 2.65 %")):
            a = af(r) if name != "old" else round(af(r), 2)
            bar[k] = npv_generic(cx, ox, t["base"], a, a) - nb
        for g, k in ((-0.01, "growth -1 %/yr"), (0.01, "growth +1 %/yr")):
            a = fpv_g(g) if name != "old" else round(fpv_g(g), 2)
            bar[k] = npv_generic(cx, ox, t["base"], a, fpv) - nb
        bar["OPEX -50 %"] = 0.5 * ox * fpv / 1000
        bar["OPEX +50 %"] = -0.5 * ox * fpv / 1000
        for scn, k in (("demand_m20", "demand -20 %"), ("demand_p20", "demand +20 %"), ("spread_075", "spread x0.75"),
                       ("spread_125", "spread x1.25")):
            bar[k] = npv_generic(cx, ox, tiers(D.loc[(scn, o)], o, f, share)["base"], fpv, fpv) - nb
        bar["full S_ref credited (potential)"] = npv_generic(cx, ox, t["potential"], fpv, fpv) - nb
        bar["relaxed schedule not credited (lower)"] = npv_generic(cx, ox, t["lower"], fpv, fpv) - nb
        bar["bridged, R = 2 h"] = npv_generic(cx, ox, t["bridged"], fpv, fpv) - nb
        bar["ROSIM correction on the whole increment"] = npv_generic(cx, ox, f * (d.S_flex + d.S_stor), fpv, fpv) - nb
        out.append(dict(set=name, npv_base=nb, **bar))
    return pd.DataFrame(out).set_index("set").T


def tank_sign(A):
    """where does the tank stop being positive: per case, tier and scenario."""
    t = A[(A.option == "TTES5k") & A.set.str.startswith("new_R")].copy()
    t["case"] = t.set.str[-1]
    p = t.pivot_table(index=["case", "tier"], columns="scenario", values="npv")
    return p[DISPATCH_SCEN + list(G) + list(RATES)]


def main():
    worst = check_cache()
    D = decompose()
    sets, share = inputs()
    A = all_rows(D, sets, share)
    A.to_csv(HERE / "scenarios_all_tiers.csv", index=False, float_format="%.4f")
    rep = reproduction(A)
    rep.to_csv(HERE / "scenarios_reproduction_old.csv", index=False, float_format="%.3f")
    T = old_vs_new(A)
    T.to_csv(HERE / "scenarios_old_vs_new.csv", index=False, float_format="%.3f")
    B = breakevens(D, sets, share)
    B.to_csv(HERE / "scenario_breakeven.csv", index=False, float_format="%.3f")
    TS = tank_sign(A)
    TS.to_csv(HERE / "scenario_tank_sign.csv", float_format="%.3f")
    dec = D.reset_index()
    dec.to_csv(HERE / "scenario_decomposition.csv", index=False, float_format="%.3f")
    print(f"cache check: the six Table 7.13 scenarios equal scenario_results/cache to {worst:.4f} k EUR/yr")
    print("\nOLD column against the printed thesis tables (max |diff|):", round(float(rep["diff"].abs().max()), 3))
    print(rep[rep["diff"].abs() > 0.015].to_string(index=False))
    print("\nOLD vs NEW (base tier, mid CAPEX; new = f_corr basis R, tank case B)")
    print(T[["scenario", "option", "old_npv", "new_npv", "change", "sign_change", "old_rank", "new_rank", "rank_change",
             "new_npv_fcorr_D", "new_npv_tank_D", "new_npv_tank_H", "sign_change_any_case", "rank_change_any_case"]]
          .round(2).to_string(index=False))
    print("\nBreak-evens\n", B.round(2).to_string(index=False))
    print("\nTank NPV per case / tier / scenario (new, basis R)\n", TS.round(2).to_string())
    # orders per scenario
    base = A[A.tier == "base"]
    orders = {}
    for k in ("old", "new_R_B", "new_D_B", "new_R_D", "new_R_H"):
        for scn in DISPATCH_SCEN + list(G) + list(RATES):
            g = base[(base.set == k) & (base.scenario == scn)].sort_values("npv", ascending=False)
            orders[f"{k}|{scn}"] = " > ".join(g.option)
    json.dump(dict(orders=orders, cache_check_keur=worst, shares=share), open(HERE / "scenarios_summary.json", "w"), indent=1)
    print("\nOrders (base tier):")
    for k, v in orders.items():
        print(f"  {k:22s} {v}")


def analytic_main():
    """Thesis Tables B.5 (savings growth) and B.7 (discount rate), Figure 7.2 (A1D tornado) and the 'conservative' row
    of Table 7.14, with the thesis inputs (set 'old') and the final inputs (sets 'new_<f_corr basis>_<tank case>')."""
    D = decompose()
    sets, share = inputs()
    A = all_rows(D, sets, share)
    base = A[A.tier == "base"]
    for keys, name, title in ((list(G), "table_B5_savings_growth.csv", "Table B.5 (savings growth g, OPEX constant)"),
                              (list(RATES), "table_B7_discount_rate.csv", "Table B.7 (real discount rate)")):
        t = base[base.scenario.isin(keys)].pivot_table(index=["set", "option"], columns="scenario", values="npv").reset_index()
        t.to_csv(HERE / name, index=False, float_format="%.3f")
        print(f"\n{title}, base NPV at mid CAPEX (M EUR)\n", t.round(2).to_string(index=False))
    Tor = tornado(D, sets, share)
    Tor.to_csv(HERE / "tornado_a1d.csv", float_format="%.3f")
    print("\nTornado A1D (Figure 7.2)\n", Tor.round(2).to_string())
    # Table 7.14 "NPV mid, conservative" (f_corr applied to the whole increment S_flex + S_stor), base scenario
    cons = []
    printed = {"TTES5k": 10.89, "A1D": -5.36, "BWN110": -4.75, "A2D": -12.59, "Bstd": -20.82, "A4D": -28.75}
    for name in ("old", "new_R_B", "new_D_B", "new_R_D", "new_R_H"):
        s = sets[name]
        for o in OPT:
            cx, ox = capex_opex(o, s); f = s["f"][o]; d = D.loc[("base", o)]
            cons.append(dict(set=name, option=o, npv_conservative=npv_generic(cx, ox, f * (d.S_flex + d.S_stor), s["fpv"], s["fpv"]),
                             printed_thesis=printed[o] if name == "old" else np.nan))
    C = pd.DataFrame(cons); C.to_csv(HERE / "conservative_npv.csv", index=False, float_format="%.3f")
    print("\nTable 7.14 conservative row\n", C.pivot(index="option", columns="set", values="npv_conservative").round(2).to_string())


if __name__ == "__main__":
    analytic_main() if "analytic" in sys.argv[1:] else main()

