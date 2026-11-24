"""Thesis Tables 6.3 and 6.10: thermal radius per doublet and the ratio of well spacing to thermal radius for the
final ROSIM runs; called by report_tables.py."""
import math

import pandas as pd

import config as C
import data_store as DS
import final_config as FC
from audit_lib import read_settings
from figtab_common import Rows, write_table

# run: (ROSIM project, schedule workbook, dispatch file, dispatch column key, doublets, label)
RUNS = {
    "A1D": ("Option A 1 Doublet 8 years changed parameters", "option_a_1d_8yr.xlsx", "A1D", "A", 1, "Option A 1D"),
    "A2D": ("Option A 2 doublets 8 years changed parameters", "option_a_2d_8yr.xlsx", "A2D", "A", 2, "Option A 2D"),
    "A4D": ("Option A 4 doublets 8 years changed parameters", "option_a_4d_8yr.xlsx", "A4D", "A", 4, "Option A 4D"),
    "B-WN110": ("Option B WN 110C 8 year adjusted", "option_b_wn110_8yr.xlsx", "BWN110", "WN", 1, "B-WN 110 °C"),
    # B-std: the runs and schedules at the return temperature of config.py (58 C: 32 K schedules; 70 C: v3 schedules)
    "B-std WN": {58.0: ("Option B WN 58 90 def", "option_b_std_wn_8yr_58.xlsx"),
                 70.0: ("Option B WN 8 years adjusted v3 70m", "option_b_std_wn_8yr_v3.xlsx")}[C.BSTD_T_RETURN_C]
                + ("Bstd", "WN", 1, "B std WN 90 °C"),
    "B-std NW": {58.0: ("Option B NW 58 90 def", "option_b_std_nw_8yr_58.xlsx"),
                 70.0: ("Option B 8 years NW adjusted v3", "option_b_std_nw_8yr_v3.xlsx")}[C.BSTD_T_RETURN_C]
                + ("Bstd", "NW", 1, "B std NW 90 °C"),
}
YEAR = 7                      # reference year of the thesis
H_WEEK = 168.0


def aquifer_thickness(project):
    """thickness (m) of the layer in which the wells are open (ROSIM settings, model3d_layers)."""
    s = read_settings(project)
    open_layer = s["open_section_layer_names__comma_separated"].split(";")[0]
    row = next(r for r in s["layers"] if r[0] == open_layer)
    return float(row[3])


def injected_volume(wb):
    # the Warmwell rows of the schedule workbook = the warm1 flow profile of the ROSIM run (data/rosim/, audit_lib.read_workbook)
    from audit_lib import read_workbook
    q = list(read_workbook(wb)[0]["Warmwell"].flow)
    return sum(max(x, 0.0) for x in q[(YEAR - 1) * 52:YEAR * 52]) * H_WEEK


def main():
    geo = pd.read_csv(C.RES_ROSIM / "rosim_geometry.csv").set_index("run")
    rows = []
    for run, (proj, wb, disp, key, n, label) in RUNS.items():
        H = aquifer_thickness(proj)
        v = injected_volume(wb)
        rth = math.sqrt(v * FC.R_TH / (math.pi * H))
        L = float(str(geo.loc[run, "warm_cold_m"]).split(";")[0])
        w = DS.dispatch_weekly(C.DISPATCH_WEEKLY, run)    # weekly aggregate of the hourly dispatch (sums and maxima)
        rows.append(dict(table="6.3" if run.startswith("A") else "6.10", config=label, cap_mw=round(w.charge_max_mw.max(), 2),
                         charged_mwh_yr_per_doublet=w.charge_mwh.sum() / 3 / n, v_inj_m3_yr=v, H_m=H, R_th=FC.R_TH, r_th_m=rth,
                         L_m=L, L_over_r_th=L / rth))
    df = pd.DataFrame(rows)
    write_table(df, "table_6_3_6_10_thermal_radius", "Tables 6.3 and 6.10 - thermal radius and L/r_th (final ROSIM runs)",
                {"cap_mw": ".1f", "charged_mwh_yr_per_doublet": ",.0f", "v_inj_m3_yr": ",.0f", "H_m": ".1f", "R_th": ".2f",
                 "r_th_m": ".0f", "L_m": ".0f", "L_over_r_th": ".1f"},
                note=f"r_th = sqrt(V_inj R_th / (pi H)), thesis Eq. 3.3; V_inj = injected volume per doublet in year {YEAR} of "
                     "the final ROSIM schedule; L = warm-cold spacing of the final ROSIM run. B standard: 90/58 C, 75 m.")
    R = Rows("Tables 6.3/6.10")
    for r in df.itertuples():
        R.put(f"r_th, {r.config}", r.r_th_m, "m", "Eq. 3.3, final schedule volume", 0)
        R.put(f"L, {r.config}", r.L_m, "m", "ROSIM settings", 0)
        R.put(f"L/r_th, {r.config}", r.L_over_r_th, "-", "L / r_th", 1)
        R.put(f"charged per doublet, {r.config}", r.charged_mwh_yr_per_doublet, "MWh/yr", "dispatch_hourly", 0)
    R.write()
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
