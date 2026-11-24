#!/usr/bin/env python3
"""Builder of the earlier 7-year ROSIM input workbooks; the package only uses its conversion factor from MW to m3/h
(through audit.py)."""
import argparse, datetime as dt, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

sys.stdout.reconfigure(encoding="utf-8")
import config as C  # repository paths (config.py)
HERE = C.WORK / "rosim_inputs_v2"      # output folder of main() (not part of the repository run)
DISPATCH = C.DISPATCH_HOURLY
OLD_INPUTS = C.ROSIM_SCHEDULES
OLD_ANALYSIS = C.WORK / "post_midterm_analysis"

# =============================================================================================
# 1. CONVERSION PARAMETERS: the only place they are set (None = not known, the current value is kept and flagged)
# =============================================================================================
CORRECTED = dict(rho_inj=None, rho_ret=None, cp=None,
                 T_inj_A=None, T_ret_A=None, T_inj_BWN110=None, T_ret_BWN110=None, T_inj_Bstd=None, T_ret_Bstd=None,
                 dT_Bstd=None,  # one value for schedule and dispatch; None = T_inj_Bstd - T_ret_Bstd
                 porosity=None, transmissivity_m2d=None, thickness_m=None, kh_mD=None, kv_mD=None)
# Current values. rho and cp: the 12-year generators behind the thesis ROSIM runs (mwh_to_m3h_for, 4180 x 1000).
CURRENT = dict(rho_inj=1000.0, rho_ret=1000.0, cp=4180.0,
               T_inj_A=110.0, T_ret_A=58.0, T_inj_BWN110=110.0, T_ret_BWN110=58.0, T_inj_Bstd=90.0, T_ret_Bstd=58.0,
               dT_Bstd=None,
               porosity=0.28, transmissivity_m2d=118.0, thickness_m=33.8, kh_mD=4000.0, kv_mD=300.0)
P = {k: (CORRECTED[k] if CORRECTED[k] is not None else CURRENT[k]) for k in CURRENT}
FLAG = {k: ("corrected" if CORRECTED[k] is not None else "CURRENT, no corrected value given") for k in CURRENT}

RESCALE_TO_DISPATCH = True
BHP_ACTIVE, BHP_IDLE, SENTINEL_T = 35, 15, -1
IDLE_M3H = 1.0
SMALL_M3H = 5.0
ROUND = 4
N_YEARS, WEEKS = 7, 52
START = dt.date(2024, 12, 30)
HOURS_WEEK = 168.0
YEARS = [2023, 2024, 2025]
Q_COLD_LIMIT = 117.0     # cold-well-limited rate (m3/h)

MW_NOTE_A = ("P30/P60, SOC+25pp->P45, demand x1.038, BEC summer cap, min run 18 h, min down 2 h "
             "(relaxed; the Option A model has no standby logic)")
MW_NOTE_B = "P15/P75, demand x1.038, BEC summer cap, min run 18 h, min down 2 h, standby off (relaxed)"
CONFIGS = {
    "A1D": dict(option="A", file="ROSIM_Option_A_1D_7jaar_weekdata", label="Option A 1 doublet", T="A", limit=221.0,
                doublets={"A": dict(n=1, sheet="", mw=13.3)}, run="scenario_runner.run_saving('A1D','relaxed')", note=MW_NOTE_A),
    "A2D": dict(option="A", file="ROSIM_Option_A_2D_7jaar_weekdata", label="Option A 2 doublets", T="A", limit=221.0,
                doublets={"A": dict(n=2, sheet="", mw=26.6)}, run="scenario_runner.run_saving('A2D','relaxed')", note=MW_NOTE_A),
    "A4D": dict(option="A", file="ROSIM_Option_A_4D_7jaar_weekdata", label="Option A 4 doublets", T="A", limit=221.0,
                doublets={"A": dict(n=4, sheet="", mw=53.2)}, run="scenario_runner.run_saving('A4D','relaxed')", note=MW_NOTE_A),
    "BWN110": dict(option="B", file="ROSIM_Option_B_WN110_7jaar_weekdata", label="Option B WN 110 C", T="BWN110", limit=221.0,
                   doublets={"WN": dict(n=1, sheet="Combined_WN_", mw=13.3)},
                   run="scenario_runner.run_saving('BWN110','relaxed')",
                   note=MW_NOTE_B + ", SOC+25pp->P45, NieuwWest 0 MW"),
    "Bstd": dict(option="B", file="ROSIM_Option_B_std_7jaar_weekdata", label="Option B standard", T="Bstd", limit=179.0,
                 doublets={"WN": dict(n=1, sheet="Combined_WN_", mw=6.64), "NW": dict(n=1, sheet="NieuwWest_", mw=6.64)},
                 run="scenario_runner.run_saving('Bstd','relaxed')", note=MW_NOTE_B),
}
BUFFERS = {"thesis": dict(sub="", tag="", desc="thesis buffer rule (model defaults)"),
           "cal": dict(sub="calibrated_buffer", tag="_calibrated",
                       desc="CALIBRATED BUFFER, version C: reserve_max_fraction 0.35, reserve_horizon_h 24, small_soc_max 0.68, "
                            "daily rank n 3, in the zero run and the ATES run")}
# Older schedules with the same dispatch configuration. "run" is the file the thesis ROSIM run really used.
OLD = {("A1D", "A"): dict(own="option_a_1doublet_12yr.xlsx", run="option_a_4doublets_12yr.xlsx"),
       ("A2D", "A"): dict(own="option_a_2doublets_12yr.xlsx", run="option_a_4doublets_12yr.xlsx"),
       ("A4D", "A"): dict(own="option_a_4doublets_12yr.xlsx", run="option_a_4doublets_12yr.xlsx"),
       ("BWN110", "WN"): dict(own="option_b_wn_110c_12yr.xlsx", run="option_b_wn_110c_12yr.xlsx"),
       ("Bstd", "WN"): dict(own="option_b_wn_physical_12yr.xlsx", run="option_b_wn_physical_12yr.xlsx"),
       ("Bstd", "NW"): dict(own="option_b_nw_physical_12yr.xlsx", run="option_b_nw_physical_12yr.xlsx")}
# What-if: flow factor against the volumetric heat capacity in use. NOT applied, for the supervisors' discussion.
WHATIF = [("rho 985, cp 4182 (export_rosim_weekly_option_a.py)", 985.0 * 4182.0),
          ("rho x cp = 4.1652 MJ/m3K (dispatch CONST 0.001157)", 0.001157e6 * 3600.0),
          ("rho 950, cp 4220 (r_th and P_coldwell in the generators)", 950.0 * 4220.0),
          ("rho 951, cp 4180 (density alone, water at 110 C)", 951.0 * 4180.0),
          ("rho 1000, cp 4220 (cp alone)", 1000.0 * 4220.0),
          ("rho 1046.9, cp 3.87 kJ/kgK (ROSIM deck DENSITY and SPECHEAT, mean 58-110 C)", 1046.914 * 3870.0)]


def temps(key):
    return P[f"T_inj_{key}"], P[f"T_ret_{key}"]


def delta_t(key):
    if key == "Bstd" and P["dT_Bstd"] is not None:
        return P["dT_Bstd"]
    inj, ret = temps(key)
    return inj - ret


def m3h_per_mw(rho, cp, dT):
    return 3.6e9 / (rho * cp * dT)


def weekly_pattern_a(net_m3h):
    iso = net_m3h.index.isocalendar()
    d = pd.DataFrame({"q": net_m3h.values, "y": iso["year"].values, "w": iso["week"].values})
    wk = d.groupby(["y", "w"])["q"].mean().reset_index()
    wk = wk[wk["y"].isin(YEARS) & (wk["w"] <= WEEKS)]
    wk.loc[wk["q"].abs() < IDLE_M3H, "q"] = 0.0
    pat = wk.groupby("w")["q"].mean().reindex(range(1, WEEKS + 1)).values
    pat[np.abs(pat) < IDLE_M3H] = 0.0
    return pat


def weekly_pattern_b(net_m3h):
    month = net_m3h.groupby(net_m3h.index.month).mean()
    months = [(START + dt.timedelta(weeks=i)).month for i in range(WEEKS)]
    pat = np.array([month[m] for m in months], dtype=float)
    pat[np.abs(pat) < IDLE_M3H] = 0.0
    return pat


def rescale(pat, v_inj, v_ext):
    """Scale injection and extraction weeks separately to the yearly dispatch volumes."""
    pos, neg = pat[pat > 0].sum() * HOURS_WEEK, -pat[pat < 0].sum() * HOURS_WEEK
    s_pos = v_inj / pos if pos > 0 else 1.0
    s_neg = v_ext / neg if neg > 0 else 1.0
    out = pat.copy()
    out[out > 0] *= s_pos
    out[out < 0] *= s_neg
    return out, s_pos, s_neg


def schedule(pat):
    year1 = pat.copy()
    year1[year1 < 0] = 0.0
    return np.concatenate([year1] + [pat] * (N_YEARS - 1))


# =============================================================================================
# 2. WORKBOOKS
# =============================================================================================
HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
HEADER_FONT = Font(color="FFFFFF", bold=True)
YEAR1_FILL = PatternFill(start_color="FFF9B0", end_color="FFF9B0", fill_type="solid")
WHITE_FILL = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
GREY_FILL = PatternFill(start_color="E7E6E6", end_color="E7E6E6", fill_type="solid")
BOLD = Font(bold=True)
COL_WIDTHS = [8, 8, 6, 13, 10, 10]
HEADERS = ["Year", "Month", "Day", "Flow_m3h", "BHP_bar", "Temp_C"]


def write_sheet(ws, title, flows, inj_temp, warm_up_label):
    ws["A1"] = title
    ws["A1"].font = Font(bold=True, size=12)
    ws.merge_cells("A1:F1")
    for c, h in enumerate(HEADERS, 1):
        cell = ws.cell(row=3, column=c, value=h)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(horizontal="center")
    r = 4
    if warm_up_label:   # Option A layout has this label row; the Option B layout does not
        ws.cell(row=r, column=1, value="WARM-UP YEAR (charge only)").font = BOLD
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=6)
        for c in range(1, 7):
            ws.cell(row=r, column=c).fill = YEAR1_FILL
        r += 1
    for i, q in enumerate(flows):
        d = START + dt.timedelta(weeks=i)
        q = round(float(q), ROUND)
        injecting = q > 0
        vals = [d.year, d.month, d.day, q, BHP_ACTIVE if injecting else BHP_IDLE, inj_temp if injecting else SENTINEL_T]
        yr = i // WEEKS + 1
        fill = YEAR1_FILL if yr == 1 else (WHITE_FILL if yr % 2 == 0 else GREY_FILL)
        for c, v in enumerate(vals, 1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.fill = fill
        r += 1
    for i, w in enumerate(COL_WIDTHS, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def read_sheet(ws):
    rows = [r for r in ws.iter_rows(values_only=True) if isinstance(r[0], int)]
    return rows


def write_workbook(cfg_key, cfg, buf_key, data, sha):
    buf = BUFFERS[buf_key]
    folder = HERE / buf["sub"] if buf["sub"] else HERE
    folder.mkdir(exist_ok=True)
    path = folder / f"{cfg['file']}{buf['tag']}.xlsx"
    T_inj, T_ret = temps(cfg["T"])
    label_row = cfg["option"] == "A"
    wb = Workbook()
    wb.remove(wb.active)
    for dkey, d in cfg["doublets"].items():
        q = data[dkey]["flows"]
        q_cold = -q * P["rho_inj"] / P["rho_ret"]
        name = f"{cfg['label']}{' (calibrated buffer)' if buf_key == 'cal' else ''}"
        scope = f" {dkey}" if cfg["option"] == "B" else (f", per doublet (use for each of the {d['n']})" if d["n"] > 1 else "")
        write_sheet(wb.create_sheet(f"{d['sheet']}Warmwell"), f"{name}{scope}: 7-year ROSIM schedule (Warmwell)", q, T_inj, label_row)
        write_sheet(wb.create_sheet(f"{d['sheet']}Coldwell"), f"{name}{scope}: 7-year ROSIM schedule (Coldwell)", q_cold, T_ret, label_row)
    ws = wb.create_sheet("Metadata")
    end = START + dt.timedelta(weeks=N_YEARS * WEEKS)
    meta = [("ROSIM input", f"{cfg['label']}, {buf['desc']}"),
            ("Period", f"{START} to {START + dt.timedelta(weeks=N_YEARS * WEEKS - 1)}: 364 weekly rows, 7-day steps, strictly increasing"),
            ("ROSIM simulation end date", f"{end} (end of the last week; the last row applies until this date)"),
            ("Year 1", "Warm-up: charge only, extraction weeks idle"), ("Years 2-7", "Representative year, identical each year"),
            ("Weekly aggregation", "ISO-week mean of net flow, mean per week number over 2023-2025" if cfg["option"] == "A"
             else "V6 method: monthly mean net flow 2023-2025 of this run; each week takes the month of its Monday"),
            ("Rescaled to dispatch MWh", str(RESCALE_TO_DISPATCH)), (None, None)]
    for k, label in [("rho_inj", "Density, warm side (kg/m3)"), ("rho_ret", "Density, cold side (kg/m3)"), ("cp", "Specific heat (J/kg/K)")]:
        meta.append((label, f"{P[k]:g}  [{FLAG[k]}]"))
    meta += [("Injection temperature (C)", f"{T_inj:g}  [{FLAG['T_inj_' + cfg['T']]}]"),
             ("Return temperature (C)", f"{T_ret:g}  [{FLAG['T_ret_' + cfg['T']]}]"),
             ("Delta T used for MWh to m3 (K)", f"{delta_t(cfg['T']):g}"
              + ("  [dispatch model stores at 90-70 = 20 K: inconsistent, flagged]" if cfg["T"] == "Bstd" and P["dT_Bstd"] is None else "")),
             ("Porosity (metadata)", f"{P['porosity']:g}  [{FLAG['porosity']}]"),
             ("Transmissivity (m2/day, metadata)", f"{P['transmissivity_m2d']:g}  [{FLAG['transmissivity_m2d']}]"),
             ("Aquifer thickness (m, metadata)", f"{P['thickness_m']:g}  [{FLAG['thickness_m']}]"),
             ("kh / kv (mD, set in the ROSIM deck)", f"{P['kh_mD']:g} / {P['kv_mD']:g}  [{FLAG['kh_mD']}]"), (None, None)]
    for dkey, d in cfg["doublets"].items():
        x = data[dkey]
        meta += [(f"{dkey}: doublets x cap", f"{d['n']} x {d['mw'] / d['n']:g} MW = {d['mw']:g} MW; well limit {cfg['limit']:g} m3/h per doublet"),
                 (f"{dkey}: dispatch charged / discharged", f"{x['charged']:,.1f} / {x['discharged']:,.1f} MWh/yr (all doublets)"),
                 (f"{dkey}: scale factor injection / extraction", f"{x['s_pos']:.4f} / {x['s_neg']:.4f}")]
    meta += [(None, None), ("Idle / extraction sentinels", f"BHP {BHP_IDLE} bar, Temp {SENTINEL_T}; injection BHP {BHP_ACTIVE} bar"),
             ("Dispatch run", f"{cfg['run']}, buffer: {buf['desc']}"), ("Dispatch settings", cfg["note"]),
             ("Script", "analysis/rosim_inputs_v2/build_rosim_inputs.py"),
             ("Script version", f"sha256 {sha[:16]} (git hash not available: the project is not a git repository)"),
             ("Generated", dt.date.today().isoformat())]
    for r, (k, v) in enumerate(meta, 1):
        if k:
            ws.cell(row=r, column=1, value=k).font = BOLD if r == 1 else Font()
            ws.cell(row=r, column=2, value=v)
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 100
    wb.save(path)
    return path


# =============================================================================================
# 3. CHECKS: all from the saved workbook, not from the arrays in memory
# =============================================================================================
def check_workbook(path, cfg_key, cfg, buf_key, data):
    wb = load_workbook(path, data_only=True)
    out = []
    dT = delta_t(cfg["T"])
    T_inj, T_ret = temps(cfg["T"])
    for dkey, d in cfg["doublets"].items():
        warm = read_sheet(wb[f"{d['sheet']}Warmwell"])
        cold = read_sheet(wb[f"{d['sheet']}Coldwell"])
        assert len(warm) == len(cold) == N_YEARS * WEEKS, len(warm)
        dates = [dt.date(*r[:3]) for r in warm]
        assert dates[0] == START and all((b - a).days == 7 for a, b in zip(dates, dates[1:]))
        assert dates == [dt.date(*r[:3]) for r in cold]
        for rows, T in ((warm, T_inj), (cold, T_ret)):
            for r in rows:
                assert (r[3:6] == (r[3], BHP_ACTIVE, T)) if r[3] > 0 else (r[4] == BHP_IDLE and r[5] == SENTINEL_T), r
        q = np.array([r[3] for r in warm], float)
        yr = [q[i * WEEKS:(i + 1) * WEEKS] for i in range(N_YEARS)]
        assert not (yr[0] < 0).any()
        v_in = [y[y > 0].sum() * HOURS_WEEK for y in yr]
        v_ex = [-y[y < 0].sum() * HOURS_WEEK for y in yr]
        assert max(v_in[1:]) - min(v_in[1:]) < 1e-6 and max(v_ex[1:]) - min(v_ex[1:]) < 1e-6
        to_mwh = P["rho_inj"] * P["cp"] * dT / 3.6e9 * d["n"]
        e_c, e_d = v_in[1] * to_mwh, v_ex[1] * to_mwh
        # A reversal week: opposite sign to the last non-idle week, with |flow| below SMALL_M3H
        reversal = np.zeros(len(q), bool)
        last = 0.0
        for i, x in enumerate(q):
            if x != 0:
                reversal[i] = last != 0 and np.sign(x) != np.sign(last) and abs(x) < SMALL_M3H
                last = x
        reversals = int(reversal.sum())
        rev_year = int(reversal[2 * WEEKS:3 * WEEKS].sum())    # a full cycle year, not the warm-up
        small = int(((q != 0) & (np.abs(q) < SMALL_M3H)).sum())
        x = data[dkey]
        out.append(dict(config=cfg_key, buffer=buf_key, doublet=dkey, n_doublets=d["n"], workbook=path.name,
                        V_inj_m3_yr_per_doublet=v_in[1], V_ext_m3_yr_per_doublet=v_ex[1], ratio_ext_inj=v_ex[1] / v_in[1],
                        E_ch_repr_MWh=e_c, E_ch_disp_MWh=x["charged"], err_ch_pct=(e_c / x["charged"] - 1) * 100,
                        E_dis_repr_MWh=e_d, E_dis_disp_MWh=x["discharged"], err_dis_pct=(e_d / x["discharged"] - 1) * 100,
                        peak_inj_m3h=q.max(), peak_ext_m3h=-q.min(), well_limit_m3h=cfg["limit"],
                        within_limit=bool(max(q.max(), -q.min()) <= cfg["limit"]), over_117_coldwell=bool(max(q.max(), -q.min()) > Q_COLD_LIMIT),
                        reversal_weeks_small_364=reversals, reversal_weeks_small_rep_year=rev_year, weeks_0_to_5=small,
                        max_jump_m3h=float(np.abs(np.diff(q)).max()), warmup_inj_weeks=int((yr[0] > 0).sum()),
                        warmup_idle_weeks=int((yr[0] == 0).sum()), warmup_V_m3_per_doublet=v_in[0],
                        scale_inj=x["s_pos"], scale_ext=x["s_neg"], unscaled_V_inj=x["unscaled_inj"], unscaled_V_ext=x["unscaled_ext"]))
    return out


def old_volumes(fname, first_data_row=4):
    wb = load_workbook(OLD_INPUTS / fname, data_only=True)
    q = np.array([r[3] for r in read_sheet(wb["Warmwell"])], float)
    y = q[WEEKS:7 * WEEKS]
    return dict(V_inj=y[y > 0].sum() * HOURS_WEEK / 6, V_ext=-y[y < 0].sum() * HOURS_WEEK / 6, peak_inj=y.max(), peak_ext=-y.min(), max_jump=np.abs(np.diff(y)).max())


def regression_old_a():
    """The aggregation with the old parameters and no rescaling must reproduce the old Option A workbook."""
    f = DISPATCH / "OLD_A50__thesis.pkl"
    if not f.exists():
        print("  regression: OLD_A50 not available yet")
        return
    df = pd.read_pickle(f)
    k = m3h_per_mw(985.0, 4182.0, 52.0)
    pat = weekly_pattern_a((df["A_charge_mw"] - df["A_discharge_mw"]) * k)
    ws = load_workbook(OLD_ANALYSIS / "ROSIM_Option_A_8jaar_weekdata.xlsx", data_only=True)["Warmwell"]
    old = np.array([r[3] for r in read_sheet(ws)], float)
    new = np.round(np.concatenate([np.where(pat < 0, 0.0, pat)] + [pat] * 7), 1)
    new[np.abs(new) < IDLE_M3H] = 0.0
    print(f"  regression against ROSIM_Option_A_8jaar_weekdata.xlsx: max |diff| {np.abs(old - new).max():.3f} m3/h over {len(old)} rows;"
          f" peaks {new.max():.1f} / {new.min():.1f} (file {old.max():.1f} / {old.min():.1f});"
          f" dispatch {df['A_charge_mw'].sum() / 3:,.0f} / {df['A_discharge_mw'].sum() / 3:,.0f} MWh/yr, hourly peak {df['A_charge_mw'].max():.1f} MW")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    a = ap.parse_args()
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    checks, compare = [], []
    keys = [k for k in a.configs.split(",") if k]
    results = {}
    for buf_key in BUFFERS:
        for key in keys:
            cfg = CONFIGS[key]
            df = pd.read_pickle(DISPATCH / f"{key}__{buf_key}.pkl")
            dT = delta_t(cfg["T"])
            data = {}
            for dkey, d in cfg["doublets"].items():
                c, dis = df[f"{dkey}_charge_mw"], df[f"{dkey}_discharge_mw"]
                charged, discharged = c.sum() / 3.0, dis.sum() / 3.0
                k = m3h_per_mw(P["rho_inj"], P["cp"], dT)
                net = (c - dis) * k / d["n"]
                pat = weekly_pattern_a(net) if cfg["option"] == "A" else weekly_pattern_b(net)
                un_in, un_ex = pat[pat > 0].sum() * HOURS_WEEK, -pat[pat < 0].sum() * HOURS_WEEK
                v_in, v_ex = charged * k / d["n"], discharged * k / d["n"]
                s_pos, s_neg = 1.0, 1.0
                if RESCALE_TO_DISPATCH:
                    pat, s_pos, s_neg = rescale(pat, v_in, v_ex)
                data[dkey] = dict(flows=schedule(pat), charged=charged, discharged=discharged, s_pos=s_pos, s_neg=s_neg,
                                  unscaled_inj=un_in, unscaled_ext=un_ex)
            path = write_workbook(key, cfg, buf_key, data, sha)
            checks += check_workbook(path, key, cfg, buf_key, data)
            results[(key, buf_key)] = data
    t = pd.DataFrame(checks)
    suffix = "" if set(keys) == set(CONFIGS) else "_partial"
    t.to_csv(HERE / f"step3_checks{suffix}.csv", index=False, float_format="%.4f")

    # Step 4: old versus new, thesis-default set
    what = []
    for _, r in t[t.buffer == "thesis"].iterrows():
        cfg = CONFIGS[r.config]
        old = OLD[(r.config, r.doublet)]
        o = old_volumes(old["own"])
        ros = old_volumes(old["run"])
        rho_f, cp_f = CURRENT["rho_inj"] / P["rho_inj"], CURRENT["cp"] / P["cp"]
        T_inj, T_ret = temps(cfg["T"])
        old_dT = CURRENT[f"T_inj_{cfg['T']}"] - CURRENT[f"T_ret_{cfg['T']}"]
        dT_f = old_dT / delta_t(cfg["T"])
        compare.append(dict(config=r.config, doublet=r.doublet, old_file=old["own"], old_V_inj=o["V_inj"], new_V_inj=r.V_inj_m3_yr_per_doublet,
                            old_V_ext=o["V_ext"], new_V_ext=r.V_ext_m3_yr_per_doublet, old_peak_inj=o["peak_inj"], new_peak_inj=r.peak_inj_m3h,
                            old_peak_ext=o["peak_ext"], new_peak_ext=r.peak_ext_m3h, old_max_jump=o["max_jump"], new_max_jump=r.max_jump_m3h,
                            factor_density=rho_f, factor_cp=cp_f, factor_dT=dT_f,
                            factor_method_inj=r.V_inj_m3_yr_per_doublet / o["V_inj"] / (rho_f * cp_f * dT_f),
                            factor_method_ext=r.V_ext_m3_yr_per_doublet / o["V_ext"] / (rho_f * cp_f * dT_f),
                            rosim_run_file=old["run"], rosim_run_V_inj=ros["V_inj"], rosim_run_V_ext=ros["V_ext"]))
        rc = P["rho_inj"] * P["cp"]
        for name, x in WHATIF:
            f = rc / x
            what.append(dict(config=r.config, doublet=r.doublet, case=name, flow_factor=f, V_inj=r.V_inj_m3_yr_per_doublet * f,
                             V_ext=r.V_ext_m3_yr_per_doublet * f, peak=max(r.peak_inj_m3h, r.peak_ext_m3h) * f, limit=r.well_limit_m3h))
        if r.config == "Bstd":
            f = delta_t("Bstd") / 20.0
            what.append(dict(config=r.config, doublet=r.doublet, case="dT 20 K (as in the dispatch, 90-70 C)", flow_factor=f,
                             V_inj=r.V_inj_m3_yr_per_doublet * f, V_ext=r.V_ext_m3_yr_per_doublet * f,
                             peak=max(r.peak_inj_m3h, r.peak_ext_m3h) * f, limit=r.well_limit_m3h))
    pd.DataFrame(compare).to_csv(HERE / f"step4_old_vs_new{suffix}.csv", index=False, float_format="%.4f")
    pd.DataFrame(what).to_csv(HERE / f"step4_whatif_not_applied{suffix}.csv", index=False, float_format="%.4f")

    # Calibrated versus thesis-default volumes
    p = t.pivot(index=["config", "doublet"], columns="buffer", values=["V_inj_m3_yr_per_doublet", "V_ext_m3_yr_per_doublet"])
    cal = pd.DataFrame({"V_inj_thesis": p[("V_inj_m3_yr_per_doublet", "thesis")], "V_inj_cal": p[("V_inj_m3_yr_per_doublet", "cal")],
                        "V_ext_thesis": p[("V_ext_m3_yr_per_doublet", "thesis")], "V_ext_cal": p[("V_ext_m3_yr_per_doublet", "cal")]})
    cal["dV_inj_pct"] = (cal.V_inj_cal / cal.V_inj_thesis - 1) * 100
    cal["dV_ext_pct"] = (cal.V_ext_cal / cal.V_ext_thesis - 1) * 100
    cal.to_csv(HERE / "calibrated_buffer" / f"calibrated_vs_thesis{suffix}.csv", float_format="%.4f")

    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_colwidth", 60)
    print("\nPARAMETERS", {k: P[k] for k in ("rho_inj", "rho_ret", "cp")}, "unset corrected values:", [k for k, v in CORRECTED.items() if v is None])
    print("\nSTEP 3")
    print(t.drop(columns=["workbook", "unscaled_V_inj", "unscaled_V_ext"]).round(3).T.to_string())
    print("\nunscaled volumes per doublet"); print(t[["config", "buffer", "doublet", "unscaled_V_inj", "unscaled_V_ext", "scale_inj", "scale_ext"]].round(3).to_string())
    print("\nSTEP 4"); print(pd.DataFrame(compare).round(3).T.to_string())
    print("\nCALIBRATED"); print(cal.round(2).to_string())
    regression_old_a()


if __name__ == "__main__":
    main()
