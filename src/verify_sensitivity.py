"""Check: the appendix sensitivity results against their reference copies and against the locked numbers they depend
on. Run by run_all.py."""
import hashlib
import sys

import numpy as np
import pandas as pd

import config as C
import data_store as DS

TOL = 0.002
S = C.RES_SENS
ROWS = []


def check(name, ok, detail):
    ROWS.append(dict(check=name, status="ok" if ok else "FAILED", detail=detail))


def locked(key):
    m = pd.read_csv(C.RES_FINAL / "chapter_number_map.csv")
    v = m.loc[m.key == key, "value"]
    assert len(v) == 1, key
    return float(v.iloc[0])


def same_as_reference(new_path, ref, name, label):
    """numeric columns within TOL, other columns compared only where they do not hold paths."""
    a, b = pd.read_csv(new_path), DS.frame(ref, name)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        check(label, False, f"shape/columns differ: {a.shape} vs {b.shape}")
        return
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    d = max([float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))) if len(a) else 0.0 for c in num] or [0.0])
    nan_ok = all((a[c].isna() == b[c].isna()).all() for c in num)
    txt = [c for c in a.columns if c not in num and c not in ("script", "result_file", "source")]
    txt_ok = all((a[c].astype(str) == b[c].astype(str)).all() for c in txt)
    check(label, d <= TOL and nan_ok and txt_ok, f"{len(num)} numeric columns, largest |diff| {d:.6f}; "
          f"{len(txt)} text columns {'equal' if txt_ok else 'DIFFER'}")


def main():
    # (1) regenerated files against the frozen copies of the project (file = <analysis>/<file name>)
    ref = DS.read_long(C.REF_SENSITIVITY)
    for sub, files in {"pit_check": ("pit_runs_table.csv", "verification_claims.csv", "rerun_comparison.csv", "final_ttes_check.csv"),
                       "temp_sensitivity": ("rosim_production_temps.csv", "rosim_production_temps_y7.csv", "step2_derated.csv",
                                            "step2_ranking.csv", "step3_energy_below_90.csv", "step3_heat_pump.csv")}.items():
        for f in files:
            same_as_reference(S / sub / f, ref, f"{sub}/{f}", f"{sub}/{f} = project copy")

    # (2) against the locked numbers
    pit = pd.read_csv(S / "pit_check" / "pit_runs_table.csv")
    r110 = pit[(pit.buffer_setting == "thesis") & (pit.schedule == "relaxed") & (pit.volume_m3 == 5000)
               & (pit.charge_temperature_c == 110) & (pit.topup_enforced == "no")].storage_effect_keur_yr.iloc[0]
    sc = pd.read_csv(C.RES_SCENARIO_RUNNER / "scenario_results.csv")
    s_ttes = float(sc[(sc.scenario == "base") & (sc.option == "TTES5k")].S_stor.iloc[0])
    check("pit 5,000 m3 / 110 C (no top-up) = TTES storage effect of scenario_runner", abs(r110 - s_ttes) <= 0.01,
          f"{r110:.3f} vs {s_ttes:.3f} k EUR/yr")
    import scenario_runner as SR
    exp = SR.EXPECTED["S_stor"]["TTES5k"]
    check("pit 5,000 m3 / 110 C = locked TTES storage effect (thesis, rounded)", abs(r110 - exp) <= 0.5,
          f"{r110:.2f} vs {exp} k EUR/yr (thesis value rounded to whole k EUR)")
    ft = pd.read_csv(S / "pit_check" / "final_ttes_check.csv")
    check("pit 110 C run = final TTES5k run, per year", float(ft.difference.abs().max()) <= 1e-6,
          f"largest |diff| {ft.difference.abs().max():.2e} k EUR")
    d2 = pd.read_csv(S / "temp_sensitivity" / "step2_derated.csv")
    lab = {"A1D": "Option A 1D", "A2D": "Option A 2D", "A4D": "Option A 4D", "BWN110": "B-WN110", "Bstd": "B standard"}
    for o, L in lab.items():
        h = d2[(d2.option == o) & (d2.case == "headline")].iloc[0]
        for col, key, dec in (("npv_base_meur", f"{L} NPV base (mid CAPEX)", 2), ("lcoh_eur_mwh", f"{L} LCOH (discounted)", 0),
                              ("f_corr_headline", f"{L} f_corr (basis R)", 3),
                              ("s_stor_after_cal_keur", f"{L} storage effect after calibration", 1)):
            lv = locked(key)
            check(f"derated table, headline row {o}: {col} = locked '{key}'", abs(round(h[col], dec) - lv) <= TOL,
                  f"{h[col]:.4f} (rounded {round(h[col], dec)}) vs {lv}")
        check(f"derated table, headline rerun {o} reproduces the uncalibrated S_stor", abs(h.s_stor_rerun_check_keur - h.s_stor_uncal_keur) <= 0.01,
              f"{h.s_stor_rerun_check_keur:.3f} vs {h.s_stor_uncal_keur:.3f}")
    hp = pd.read_csv(S / "temp_sensitivity" / "step3_heat_pump.csv")
    nw = hp[hp.loop == "NW"]
    import final_config as FC
    af = FC.annuity_factor()
    back = nw.npv_base_meur + (nw.opex_keur - locked("B standard OPEX")) * af / 1000
    check("heat-pump table: NPV at the locked OPEX 378 = locked B-std NPV base", float((back - locked("B standard NPV base (mid CAPEX)")).abs().max()) <= 0.005,
          f"back-computed {back.min():.4f} .. {back.max():.4f} vs {locked('B standard NPV base (mid CAPEX)')}")

    # (3) internal checks
    vc = pd.read_csv(S / "pit_check" / "verification_claims.csv")
    check("pit: the five numbers of the notes are confirmed", (vc.verdict == "confirmed").all() and len(vc) == 5,
          ", ".join(f"{r.value_in_notes} -> {r.value_in_files:.2f}" for r in vc.itertuples()))
    rc = pd.read_csv(S / "pit_check" / "rerun_comparison.csv")
    check("pit: reruns with the current model = original runs", float(rc.difference.abs().max()) <= 1e-9 and len(rc) == 168,
          f"{len(rc)} values, largest |diff| {rc.difference.abs().max():.2e}")
    m = pd.read_csv(C.RES_FINAL / "chapter_number_map.csv")
    n = int((m.chapter == "Sensitivity analyses (appendix)").sum())
    check("sensitivity numbers present in chapter_number_map.csv", n > 0, f"{n} numbers")

    t = pd.DataFrame(ROWS)
    t.to_csv(C.RESULTS / "verification_sensitivity.csv", index=False)
    bad = t[t.status != "ok"]
    print(f"{len(t)} sensitivity checks, {len(bad)} failed")
    if len(bad):
        print(bad.to_string(index=False))
        sys.exit(1)
    print("all sensitivity checks pass")


if __name__ == "__main__":
    main()
