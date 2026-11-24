"""Adds the numbers of the appendix sensitivity analyses, with the source line of each, to the locked-numbers files
and writes the sensitivity manifest. Run by run_all.py."""
import hashlib
import sys

import pandas as pd

import config as C

sys.stdout.reconfigure(encoding="utf-8")
CHAPTER = "Sensitivity analyses (appendix)"
SECTION = "## Sensitivity analyses (appendix)"
OPT_LABEL = {"A1D": "Option A 1D", "A2D": "Option A 2D", "A4D": "Option A 4D", "BWN110": "B-WN110", "Bstd": "B standard"}
S = C.RES_SENS


def line_of(module, anchor):
    """script:line of the first line of src/<module>.py that contains `anchor` (where the number is computed)."""
    lines = (C.SRC / f"{module}.py").read_text(encoding="utf-8").splitlines()
    hits = [i for i, l in enumerate(lines, 1) if anchor in l]
    if not hits:
        raise KeyError(f"{module}.py: anchor not found: {anchor}")
    return f"src/{module}.py:{hits[0]}"


ROWS = []


def put(key, value, unit, script, source, decimals=2):
    ROWS.append(dict(chapter=CHAPTER, key=f"Sensitivity: {key}", value=round(float(value), decimals), unit=unit,
                     script=script, source=source, author_estimate="", old_value=None, old_location=""))


def pit():
    t = pd.read_csv(S / "pit_check" / "pit_runs_table.csv")
    where = line_of("make_pit_table", '"storage_effect_keur_yr": round(s, 2)')
    where_tu = line_of("make_pit_table", '"topup_energy_mwh_yr": round(m["topup_mwh_yr"], 1)')
    for v in (5000, 10000):
        for temp in (110, 95, 90):
            for tu in ("no", "yes"):
                r = t[(t.buffer_setting == "thesis") & (t.schedule == "relaxed") & (t.volume_m3 == v)
                      & (t.charge_temperature_c == temp) & (t.topup_enforced == tu)].iloc[0]
                lab = f"pit check, storage effect {v:,} m3 at {temp} C, relaxed, {'90 C top-up enforced' if tu == 'yes' else 'no top-up'}"
                put(lab, r.storage_effect_keur_yr, "k EUR/yr", where, "pit_runs_table.csv (thesis buffers, uncalibrated)")
                if tu == "yes" and v == 5000:
                    put(f"pit check, top-up heat {v:,} m3 at {temp} C, relaxed", r.topup_energy_mwh_yr, "MWh/yr", where_tu,
                        "pit_runs_table.csv", 1)


def derated():
    t = pd.read_csv(S / "temp_sensitivity" / "step2_derated.csv")
    w_cap = line_of("analyse_sensitivity", "cap_total_mw=")
    w_s = line_of("analyse_sensitivity", "s_stor_uncal_keur=s,")
    w_n = line_of("analyse_sensitivity", "npv_base_meur=n, npv_change_meur=n - n_h,")
    w_l = line_of("analyse_sensitivity", "q_lcoh_mwh_yr=q, lcoh_eur_mwh=l,")
    for o, lab in OPT_LABEL.items():
        r = t[(t.option == o) & (t.case == "derated")].iloc[0]
        put(f"derated power, cap {lab}", r.cap_total_mw, "MW", w_cap, "headline cap x year-7 ratio (T_prod - T_ret)/(T_inj - T_ret)")
        put(f"derated power, storage effect before calibration {lab}", r.s_stor_uncal_keur, "k EUR/yr", w_s, "derated dispatch run", 1)
        put(f"derated power, NPV base {lab}", r.npv_base_meur, "M EUR", w_n, "headline f_corr, mid CAPEX (downside bound)")
        put(f"derated power, NPV change {lab}", r.npv_change_meur, "M EUR", w_n, "against the headline NPV")
        put(f"derated power, LCOH {lab}", r.lcoh_eur_mwh, "EUR/MWh", w_l, "discharge scaled with the derated run", 0)
        c = t[(t.option == o) & (t.case == "cold-well cap 117 m3/h")]
        if len(c):
            put(f"cold-well cap 117 m3/h (relaxed), NPV base {lab}", c.npv_base_meur.iloc[0], "M EUR", w_n, "6.8 MW per doublet, headline f_corr")
    rk = pd.read_csv(S / "temp_sensitivity" / "step2_ranking.csv").set_index("case").ranking
    put("derated power, ranking unchanged (1 = yes)", float(rk["headline"] == rk["derated"]), "-",
        line_of("analyse_sensitivity", 'rk.append(dict(case=case, ranking='), "step2_ranking.csv", 0)


def temperatures():
    t = pd.read_csv(S / "temp_sensitivity" / "rosim_production_temps_y7.csv")
    w = {k: line_of("rosim_production_temps", a) for k, a in
         (("T", "T_prod_vw=Tm,"), ("min", "T_prod_min="), ("e", "share_energy_below_90="), ("r", "ratio=(Tm - t_ret)"))}
    for r in t.itertuples():
        put(f"ROSIM year-7 production temperature {r.run} (volume-weighted)", r.T_prod_vw, "C", w["T"], "warm well, production weeks", 1)
        put(f"ROSIM year-7 production temperature {r.run} (weekly minimum)", r.T_prod_min, "C", w["min"], "warm well", 1)
        put(f"ROSIM year-7 production temperature {r.run} (weekly maximum)", r.T_prod_max, "C", w["min"], "warm well", 1)
        put(f"ROSIM year-7 produced energy below 90 C {r.run}", 100 * r.share_energy_below_90, "%", w["e"], "relative to the dispatch return temperature", 1)
        put(f"ROSIM year-7 ratio (T_prod - T_ret)/(T_inj - T_ret) {r.run}", r.ratio, "-", w["r"], "dispatch window of the option", 3)


def heat_pump():
    t = pd.read_csv(S / "temp_sensitivity" / "step3_heat_pump.csv")
    nw = t[(t.loop == "NW") & t.run.str.startswith("calibrated")]
    w_c = line_of("analyse_sensitivity", "W_el_mwh_yr=wel, cost_keur_yr=cost))")
    w_n = line_of("analyse_sensitivity", 't["npv_base_meur"] = ')
    put("B-std heat pump electricity, lowest (final ROSIM run, calibrated)", nw.cost_keur_yr.min(), "k EUR/yr", w_c, "step3_heat_pump.csv", 1)
    put("B-std heat pump electricity, highest (final ROSIM run, calibrated)", nw.cost_keur_yr.max(), "k EUR/yr", w_c, "step3_heat_pump.csv", 1)
    thesis = nw[nw.temperature_basis.str.startswith("weekly") & nw.cop_basis.str.contains("fixed") & (nw.ates_heat_counted_above_c == 58)]
    put("B-std heat pump electricity, thesis method with the final ROSIM run", thesis.cost_keur_yr.iloc[0], "k EUR/yr", w_c, "weekly T, COP 4.38, 58 C", 1)
    put("B-std heat pump electricity in the locked OPEX (older ROSIM run)", nw.thesis_cost_keur_yr.iloc[0], "k EUR/yr",
        line_of("analyse_sensitivity", "thesis_cost = float("), "data/derived/nw_heat_pump_allweeks.csv: Table 7.4 component (54)", 1)
    put("B-std NPV base with the recomputed heat pump, highest", nw.npv_base_meur.max(), "M EUR", w_n, "OPEX 378 - 54 + recomputed")
    put("B-std NPV base with the recomputed heat pump, lowest", nw.npv_base_meur.min(), "M EUR", w_n, "OPEX 378 - 54 + recomputed")


def write_map():
    mp = C.RES_FINAL / "chapter_number_map.csv"
    m = pd.read_csv(mp)
    m = pd.concat([m[m.chapter != CHAPTER], pd.DataFrame(ROWS)], ignore_index=True)
    m.to_csv(mp, index=False)
    lk = C.RES_FINAL / "final_numbers_locked.md"
    txt = lk.read_text(encoding="utf-8")
    if SECTION in txt:                                # drop the old section (up to the next "## " heading)
        a = txt.index("\n" + SECTION)
        b = txt.find("\n## ", a + 1 + len(SECTION))
        txt = txt[:a] + (txt[b:] if b >= 0 else "\n")
    sec = ["", SECTION, "", "Numbers of the sensitivity analyses of the appendix (`scripts/15_*` to `18_*`). They are "
           "sensitivities, **not** part of the headline numbers; nothing is typed in.", ""]
    sec += [f"- {r['key']}: **{r['value']} {r['unit']}** | `{r['script']}` | {r['source']}" for r in ROWS]
    block = "\n".join(sec) + "\n"
    fig = "\n## Figures and tables (repository scripts)"
    txt = txt.rstrip() + "\n" + block if fig not in txt else txt.replace(fig, "\n" + block.rstrip() + "\n" + fig, 1)
    lk.write_text(txt, encoding="utf-8")
    print(f"{len(ROWS)} sensitivity numbers written to chapter_number_map.csv and final_numbers_locked.md")


MANIFEST = [  # (analysis, file under results/sensitivity, producing script, needs raw data, thesis use)
    ("pit check", "pit_check/pit_runs_table.csv", "src/make_pit_table.py (scripts/15)", "no", "appendix pit table"),
    ("pit check", "pit_check/pit_runs_table.md", "src/make_pit_table.py (scripts/15)", "no", "appendix pit table"),
    ("pit check", "pit_check/verification_claims.csv", "src/make_pit_table.py (scripts/15)", "no", "the five numbers of the notes"),
    ("pit check", "pit_check/rerun_comparison.csv", "src/make_pit_table.py (scripts/15)", "no (reruns: m10, raw data)", "model-version check"),
    ("pit check", "pit_check/final_ttes_check.csv", "src/make_pit_table.py (scripts/15)", "no", "110 C run = final TTES run"),
    ("temperature sensitivity", "temp_sensitivity/rosim_production_temps.csv", "src/rosim_production_temps.py (scripts/16)", "no", "production temperatures, years 2-8"),
    ("temperature sensitivity", "temp_sensitivity/rosim_production_temps_y7.csv", "src/rosim_production_temps.py (scripts/16)", "no", "production temperatures, year 7"),
    ("temperature sensitivity", "temp_sensitivity/step2_derated.csv", "src/analyse_sensitivity.py (scripts/16)", "no (runs: m11, raw data)", "derated-power NPV/LCOH"),
    ("temperature sensitivity", "temp_sensitivity/step2_ranking.csv", "src/analyse_sensitivity.py (scripts/16)", "no", "ranking"),
    ("temperature sensitivity", "temp_sensitivity/step3_energy_below_90.csv", "src/analyse_sensitivity.py (scripts/16)", "no", "energy below 90 C"),
    ("temperature sensitivity", "temp_sensitivity/step3_heat_pump.csv", "src/analyse_sensitivity.py (scripts/model/m11)", "yes (hourly EPEX prices)", "B-std heat pump range"),
    ("B-std return temperature 70 C", "bstd_return_70/rosim/rosim_check.csv", "src/bstd_return_70.py (scripts/17)", "no", "efficiency of the 70 C runs"),
    ("B-std return temperature 70 C", "bstd_return_70/fcorr/bstd_fcorr_v3_R.csv", "src/bstd_return_70.py (scripts/17)", "no", "f_corr basis R at 70 C"),
    ("B-std return temperature 70 C", "bstd_return_70/fcorr/bstd_fcorr_v3_D.csv", "src/bstd_return_70.py (scripts/17)", "no", "f_corr basis D at 70 C"),
    ("B-std return temperature 70 C", "bstd_return_70/final_numbers/step4_npv_ates_basisR.csv", "src/bstd_return_70.py (scripts/17)", "no", "NPV tiers at 70 C"),
    ("B-std return temperature 70 C", "bstd_return_70/final_numbers/step1_lcoh_basisR.csv", "src/bstd_return_70.py (scripts/17)", "no", "LCOH at 70 C"),
    ("B-std return temperature 70 C", "bstd_return_70/chapter_number_map_70.csv", "src/bstd_return_70.py (scripts/17)", "no", "the 70 C numbers (checked by verify_locked.py)"),
]


def write_manifest():
    rows = []
    items = list(MANIFEST)
    for a, f, scr, raw, use in items:
        p = S / f
        rows.append(dict(analysis=a, file=f"results/sensitivity/{f}", producing_script=scr, needs_raw_data=raw, thesis_use=use,
                         status="sensitivity, not part of the headline", exists=p.exists(),
                         sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""))
    pd.DataFrame(rows).to_csv(S / "MANIFEST.csv", index=False)
    print(f"results/sensitivity/MANIFEST.csv: {len(rows)} files, {sum(r['exists'] for r in rows)} present")


def main():
    pit()
    derated()
    temperatures()
    heat_pump()
    write_map()
    write_manifest()


if __name__ == "__main__":
    main()
