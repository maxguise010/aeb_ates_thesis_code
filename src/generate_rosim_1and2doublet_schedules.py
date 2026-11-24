"""Original script for the ROSIM schedules of Option A (1 and 2 doublets); scenario_runner.py runs only its set-up
part. The full script needs template workbooks that are not included."""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dataclasses
import inspect
import shutil
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from openpyxl import load_workbook

from config import CONST, historical_cols, derivation_factors
from data_pipeline import build_sim_df_tank
import simulation_ates_option_a_v7 as opt_a_mod
from scenarios_v4 import SCENARIO_ATES_OPTION_A_V3

price_col = "Prijs_Gemiddelde_EPEX_Spot"
YEARS_LIST = [2023, 2024, 2025]

DELTA_T_K = 52.0
INJ_TEMP_C = 110.0
COLD_TEMP_C = 58.0
H_M = 33.8
N_E = 0.28
CELL_M = 25.0
N_YEARS = 12
YEAR1_WEEKS = 52

OPTION_A_CHARGE_PCT, OPTION_A_DISCHARGE_PCT = 30, 60
TRIGGER_PP = 25
PERMISSIVE_DISCHARGE_PCT = 45.0
CORRECTIONS_KWARGS = dict(
    demand_correction_factor=1.038,
    enable_bec_summer_cap=True,
    enable_min_runtime=True,
    min_runtime_hours=18,
    min_downtime_hours=2,
)

ROSIM_DIR = os.path.join(os.path.dirname(__file__), "..", "rosim_inputs")
TEMPLATE = os.path.join(ROSIM_DIR, "option_a_soc_driven_12yr.xlsx")

HOURS_PER_YEAR = 8760.0
HOURS_PER_WEEK = HOURS_PER_YEAR / 52.0
BHP_ACTIEF = 35.0
BHP_SENTINEL = 15.0
SENTINEL_T = -1.0
DATA_START_ROW = 4
COL_FLOW = 4
COL_BHP = 5
COL_TEMP = 6

MONTH_WEEK_CENTRES = [2.0, 6.5, 11.0, 15.5, 20.0, 24.5, 28.5, 33.0, 37.5, 41.5, 46.0, 50.5]
HOURS_PER_MONTH = np.array([744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744], dtype=float)
DEFAULT_SIGMA_RANGE = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
LOW_SIGMA_RANGE = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

CONFIGS = [
    {'label': '1doublet', 'cap_mw': 13.3, 'n': 1},
    {'label': '2doublets', 'cap_mw': 26.6, 'n': 2},
]


def mwh_to_m3h_for(delta_t_k):
    return 3_600_000.0 / (4180.0 * delta_t_k)


def flow_for_mw(mw):
    return mw / (DELTA_T_K * CONST)


def mw_for_flow(flow_m3h):
    return flow_m3h * DELTA_T_K * CONST


# ═══════════════════════════════════════════════════════════════════════════
# build_smooth_schedule / write_rosim_excel -- established, validated
# functions, copied verbatim
# ═══════════════════════════════════════════════════════════════════════════
def build_smooth_schedule(monthly_charge_mwh, monthly_discharge_mwh,
                           delta_t_k, flow_cap, n_years=N_YEARS, max_jump=30.0,
                           sigma_range=None):
    mwh_to_m3h = mwh_to_m3h_for(delta_t_k)
    charge_rate = np.array(monthly_charge_mwh) / HOURS_PER_MONTH * mwh_to_m3h
    discharge_rate = np.array(monthly_discharge_mwh) / HOURS_PER_MONTH * mwh_to_m3h
    net_rate = charge_rate - discharge_rate

    w_ext = ([c - 52 for c in MONTH_WEEK_CENTRES[-3:]] +
             MONTH_WEEK_CENTRES +
             [c + 52 for c in MONTH_WEEK_CENTRES[:3]])
    net_ext = list(net_rate[-3:]) + list(net_rate) + list(net_rate[:3])

    cs = CubicSpline(w_ext, net_ext)
    flow_52 = cs(np.arange(0.5, 52, 1.0))

    flow_624 = np.tile(flow_52, n_years)
    flow_624[:YEAR1_WEEKS] = np.where(flow_624[:YEAR1_WEEKS] < 0, 0.0, flow_624[:YEAR1_WEEKS])

    annual_charge = sum(monthly_charge_mwh) * mwh_to_m3h * 52.0 / HOURS_PER_YEAR
    annual_discharge = sum(monthly_discharge_mwh) * mwh_to_m3h * 52.0 / HOURS_PER_YEAR
    target_total_charge = annual_charge * n_years
    target_total_discharge = annual_discharge * (n_years - 1)

    sigma_range = sigma_range if sigma_range is not None else DEFAULT_SIGMA_RANGE

    attempts = []
    converged_sigma = None
    for sigma in sigma_range:
        smoothed = gaussian_filter1d(flow_624, sigma=sigma, mode="nearest")

        pos_sum = smoothed[smoothed > 0].sum()
        neg_sum = -smoothed[smoothed < 0].sum()
        if pos_sum > 0:
            smoothed = np.where(smoothed > 0, smoothed * target_total_charge / pos_sum, smoothed)
        if neg_sum > 0:
            smoothed = np.where(smoothed < 0, smoothed * target_total_discharge / neg_sum, smoothed)

        smoothed = np.clip(smoothed, -flow_cap, flow_cap)
        smoothed[:YEAR1_WEEKS] = np.where(smoothed[:YEAR1_WEEKS] < 0, 0.0, smoothed[:YEAR1_WEEKS])

        diffs = np.diff(smoothed)
        max_rise = diffs.max()
        max_fall = -diffs.min()
        max_actual = max(max_rise, max_fall)

        print(f"    sigma={sigma:.1f}: rise={max_rise:.1f}  fall={max_fall:.1f} m3/hr")
        attempts.append((sigma, smoothed, max_rise, max_fall))

        if max_actual < max_jump:
            converged_sigma = sigma
            print(f"    -> Converged at sigma={sigma:.1f}")
            break

    if converged_sigma is not None:
        sigma_used, smoothed, _, _ = attempts[-1]
    else:
        rise_ok = [a for a in attempts if a[2] < max_jump]
        pool = rise_ok if rise_ok else attempts
        sigma_used, smoothed, best_rise, best_fall = min(pool, key=lambda a: a[3])
        print(f"    WARNING: no sigma satisfied both rise and fall < {max_jump:.0f} m3/hr")
        print(f"    -> Using sigma={sigma_used:.1f}: rise={best_rise:.1f}, fall={best_fall:.1f}")

    warm_12yr = smoothed
    cold_12yr = -warm_12yr
    return warm_12yr, cold_12yr, sigma_used


def build_schedule_with_guard(monthly_c, monthly_d, delta_t_k, flow_cap, label):
    mwh_to_m3h = mwh_to_m3h_for(delta_t_k)
    warm, cold, sigma_used = build_smooth_schedule(monthly_c, monthly_d, delta_t_k=delta_t_k, flow_cap=flow_cap)

    target_discharge_mwh = float(sum(monthly_d))
    discharge_mwh = -warm[warm < 0].sum() / mwh_to_m3h * HOURS_PER_WEEK / (N_YEARS - 1)
    guard_threshold = 0.5 * target_discharge_mwh
    guard_passed = discharge_mwh >= guard_threshold

    if not guard_passed:
        print(f"\n  Discharge-volume guard FAILED for {label} at sigma={sigma_used:.1f} -- "
              f"retrying restricted to sigma in {LOW_SIGMA_RANGE}.")
        warm, cold, sigma_used = build_smooth_schedule(
            monthly_c, monthly_d, delta_t_k=delta_t_k, flow_cap=flow_cap, sigma_range=LOW_SIGMA_RANGE,
        )
        discharge_mwh = -warm[warm < 0].sum() / mwh_to_m3h * HOURS_PER_WEEK / (N_YEARS - 1)
        guard_passed = discharge_mwh >= guard_threshold
        if not guard_passed:
            raise RuntimeError(
                f"Discharge-volume guard failed even within {LOW_SIGMA_RANGE} for {label}: "
                f"{discharge_mwh:.0f} MWh/yr < 50% of target ({target_discharge_mwh:.0f} MWh/yr)."
            )

    print(f"  Discharge-volume guard ({label}): {'PASSED' if guard_passed else 'FAILED'} "
          f"({discharge_mwh:.0f} MWh/yr vs threshold {guard_threshold:.0f})")
    return warm, cold, sigma_used


def write_rosim_excel(warm_12yr, cold_12yr, inj_temp, cold_temp, template_path, output_path):
    shutil.copy2(template_path, output_path)
    wb = load_workbook(output_path)
    ws_warm = wb["Warmwell"]
    ws_cold = wb["Coldwell"]

    n = len(warm_12yr)
    for i in range(n):
        row = i + DATA_START_ROW
        w = float(warm_12yr[i])
        c = float(cold_12yr[i])

        ws_warm.cell(row=row, column=COL_FLOW).value = round(w, 4)
        if w > 0.0:
            ws_warm.cell(row=row, column=COL_BHP).value = BHP_ACTIEF
            ws_warm.cell(row=row, column=COL_TEMP).value = inj_temp
        else:
            ws_warm.cell(row=row, column=COL_BHP).value = BHP_SENTINEL
            ws_warm.cell(row=row, column=COL_TEMP).value = SENTINEL_T

        ws_cold.cell(row=row, column=COL_FLOW).value = round(c, 4)
        if c > 0.0:
            ws_cold.cell(row=row, column=COL_BHP).value = BHP_ACTIEF
            ws_cold.cell(row=row, column=COL_TEMP).value = cold_temp
        else:
            ws_cold.cell(row=row, column=COL_BHP).value = BHP_SENTINEL
            ws_cold.cell(row=row, column=COL_TEMP).value = SENTINEL_T

    wb.save(output_path)
    print(f"    Saved: {os.path.basename(output_path)} ({n} data rows)")


# ═══════════════════════════════════════════════════════════════════════════
# Build the patched SOC+25pp->P45 Option A dispatch (verbatim, validated
# patch from run_ates_soc_driven_discharge.py)
# ═══════════════════════════════════════════════════════════════════════════
OLD_PQ = '''    p01 = max(sim_df[price_col].quantile(0.01), 0.0)
    p10 = sim_df[price_col].quantile(0.10)
    p25 = sim_df[price_col].quantile(0.25)
    p75 = sim_df[price_col].quantile(0.75)
    p85 = sim_df[price_col].quantile(0.85)
    p90 = sim_df[price_col].quantile(0.90)
    p99 = sim_df[price_col].quantile(0.99)'''
NEW_PQ = '''    p01 = max(sim_df[price_col].quantile(0.01), 0.0)
    p10 = sim_df[price_col].quantile(0.10)
    p25 = sim_df[price_col].quantile(0.25)
    p75 = sim_df[price_col].quantile(0.75)
    p85 = sim_df[price_col].quantile(0.85)
    p90 = sim_df[price_col].quantile(0.90)
    p99 = sim_df[price_col].quantile(0.99)
    p_permissive = sim_df[price_col].quantile(permissive_discharge_pct / 100.0)'''

OLD_SOFT = '''        # ── Level 4 / S0 mechanism: SOC-excess adjusted trailing quantile ──
        adjusted_discharge_pct    = max(50.0, base_discharge_pct - 70.0 * max(0.0, soc_excess))
        adjusted_discharge_thresh = window_prices.quantile(adjusted_discharge_pct / 100.0)
        ates_discharge_signal     = price_now >= adjusted_discharge_thresh

        adjusted_charge_pct = base_charge_pct - 70.0 * max(0.0, soc_excess)
        if adjusted_charge_pct < 5.0:
            ates_charge_signal = False
        else:
            adjusted_charge_thresh = window_prices.quantile(adjusted_charge_pct / 100.0)
            ates_charge_signal     = price_now <= adjusted_charge_thresh'''
NEW_SOFT = '''        soc_deviation_pp = soc_excess * 100.0
        override_active = trigger_pp is not None and soc_deviation_pp > trigger_pp

        if override_active:
            adjusted_discharge_pct = permissive_discharge_pct
        else:
            adjusted_discharge_pct = max(50.0, base_discharge_pct - 70.0 * max(0.0, soc_excess))
        adjusted_discharge_thresh = window_prices.quantile(adjusted_discharge_pct / 100.0)
        ates_discharge_signal     = price_now >= adjusted_discharge_thresh

        adjusted_charge_pct = base_charge_pct - 70.0 * max(0.0, soc_excess)
        if adjusted_charge_pct < 5.0:
            ates_charge_signal = False
        else:
            adjusted_charge_thresh = window_prices.quantile(adjusted_charge_pct / 100.0)
            ates_charge_signal     = price_now <= adjusted_charge_thresh'''

OLD_HARD = '''            # ── DISCHARGING WITH VARIABLE SOC FLOOR ──────────────────────────
            if price_now >= p99:
                soc_floor = 0.05
            elif price_now >= p90:
                soc_floor = 0.10
            elif price_now >= p85:
                soc_floor = 0.20
            elif price_now >= p75:
                soc_floor = 0.30
            else:
                soc_floor = 1.00   # below P75: no discharge

            # Summer discharge gate: only above P75 in Apr-Sep
            if month in [4, 5, 6, 7, 8, 9] and price_now < p75:
                soc_floor = 1.00'''
NEW_HARD = '''            discharge_price_floor = p_permissive if override_active else p75
            if price_now >= p99:
                soc_floor = 0.05
            elif price_now >= p90:
                soc_floor = 0.10
            elif price_now >= p85:
                soc_floor = 0.20
            elif price_now >= discharge_price_floor:
                soc_floor = 0.30
            else:
                soc_floor = 1.00

            if month in [4, 5, 6, 7, 8, 9] and price_now < discharge_price_floor:
                soc_floor = 1.00'''

OLD_SIG = '''    dispatch_strategy: str = "S0",
):'''
NEW_SIG = '''    dispatch_strategy: str = "S0",
    trigger_pp=None,
    permissive_discharge_pct: float = 75.0,
):'''

_src = inspect.getsource(opt_a_mod.run_ates_option_a_v6)
for _n, _b in [("pq", OLD_PQ), ("sig", OLD_SIG), ("soft", OLD_SOFT), ("hard", OLD_HARD)]:
    assert _src.count(_b) == 1, f"[{_n}] expected 1 match, found {_src.count(_b)}"
_src = _src.replace(OLD_SIG, NEW_SIG, 1)
_src = _src.replace(OLD_PQ, NEW_PQ, 1)
_src = _src.replace(OLD_SOFT, NEW_SOFT, 1)
_src = _src.replace(OLD_HARD, NEW_HARD, 1)
_src = _src.replace("def run_ates_option_a_v6(", "def run_ates_option_a_soc_driven(", 1)
_ns = dict(vars(opt_a_mod))
exec(compile(_src, "<generate_rosim_1and2doublet_schedules patched>", "exec"), _ns)
run_ates_option_a_soc_driven = _ns["run_ates_option_a_soc_driven"]
print("Patched run_ates_option_a_soc_driven() built successfully.")

# ═══════════════════════════════════════════════════════════════════════════
# Load data, per-year historical shortfall cost
# ═══════════════════════════════════════════════════════════════════════════
print("\nLoading data...")
(sim_df_tank, _a, _b, _c, _d, _e, _f, cost_hist) = build_sim_df_tank()

ww30_hist = sim_df_tank[historical_cols["WW30"]].fillna(0.0)
aftap_cap_hist = sim_df_tank["WW30_aftap_cap_mw"]
aftap_part_hist = ww30_hist.clip(upper=aftap_cap_hist)
reductie_part_hist = (ww30_hist - aftap_part_hist).clip(lower=0.0)
ww30_hist_loss = aftap_part_hist * derivation_factors["WW30_aftap"] + reductie_part_hist * derivation_factors["WW30_reductie"]
hist_loss_series = (
    sim_df_tank[historical_cols["WW10"]].fillna(0.0) * derivation_factors["WW10"]
    + ww30_hist_loss
    + sim_df_tank[historical_cols["WW01"]].fillna(0.0) * derivation_factors["WW01"]
    + sim_df_tank[historical_cols["WW00"]].fillna(0.0) * derivation_factors["WW00"]
    + sim_df_tank[historical_cols["WW02"]].fillna(0.0) * derivation_factors["WW02"]
)
hist_cost_series = hist_loss_series * sim_df_tank[price_col]
hist_cost_per_year = hist_cost_series.groupby(sim_df_tank.index.year).sum()

# ═══════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════
for cfg in CONFIGS:
    n = cfg['n']
    cap_mw = cfg['cap_mw']
    print(f"\n{'='*55}")
    print(f"Config: {cfg['label']} | Cap: {cap_mw} MW total | "
          f"Per doublet: {cap_mw/n:.1f} MW")

    scenario = dataclasses.replace(
        SCENARIO_ATES_OPTION_A_V3, ates_charge_max_mw=cap_mw, ates_discharge_max_mw=cap_mw,
    )
    df = run_ates_option_a_soc_driven(
        sim_df_tank, scenario, dispatch_strategy="S0",
        trigger_pp=TRIGGER_PP, permissive_discharge_pct=PERMISSIVE_DISCHARGE_PCT,
        annual_charge_pct=OPTION_A_CHARGE_PCT, annual_discharge_pct=OPTION_A_DISCHARGE_PCT,
        **CORRECTIONS_KWARGS,
    )

    monthly_charge = (df[df['ates_charge_mw'] > 0]
                       .groupby(df[df['ates_charge_mw'] > 0].index.month)
                       ['ates_charge_mw'].sum() / 3).reindex(range(1, 13), fill_value=0.0)
    monthly_discharge = (df[df['ates_discharge_mw'] > 0]
                          .groupby(df[df['ates_discharge_mw'] > 0].index.month)
                          ['ates_discharge_mw'].sum() / 3).reindex(range(1, 13), fill_value=0.0)

    total_charged = monthly_charge.sum()
    total_discharged = monthly_discharge.sum()
    eta_python = total_discharged / total_charged * 100 if total_charged else 0.0

    per_year = {y: hist_cost_per_year[y] - df[df.index.year == y]["total_system_cost_eur_corrected"].sum()
                for y in YEARS_LIST}
    saving = sum(per_year.values()) / 3.0

    print(f"  Total charged:    {total_charged:.0f} MWh/yr")
    print(f"  Total discharged: {total_discharged:.0f} MWh/yr")
    print(f"  η_Python:         {eta_python:.1f}%")
    print(f"  Annual saving:    €{saving:,.0f}/yr".replace(",", "."))

    # ── r_th (derived from the charged volume of this run) ────────────
    charge_per_d_mwh = total_charged / n
    V_inj_per_d_m3 = charge_per_d_mwh * 3.6e9 / (1000 * 4180 * DELTA_T_K)
    r_th = np.sqrt(V_inj_per_d_m3 / (np.pi * H_M * N_E))
    cfg['r_th'] = r_th
    print(f"\n  Per doublet (total ÷ {n}):")
    print(f"    Charged:    {charge_per_d_mwh:.0f} MWh/yr")
    print(f"    Discharged: {total_discharged/n:.0f} MWh/yr")
    print(f"    r_th (live): {r_th:.0f} m")

    # ── Build at the COMBINED cap, THEN divide by n (same order as the 4-doublet schedule script) ──
    print(f"\n  Building schedule at COMBINED {cap_mw} MW cap ...")
    flow_cap_combined = flow_for_mw(cap_mw)
    warm_combined, cold_combined, sigma = build_schedule_with_guard(
        monthly_charge.values, monthly_discharge.values, DELTA_T_K, flow_cap_combined, cfg['label'],
    )
    warm_12yr = warm_combined / n
    cold_12yr = cold_combined / n

    max_flow = warm_12yr.max()
    max_mw = mw_for_flow(max_flow)
    yr1_d = int((cold_12yr[:52] > 0).sum())
    # NOT diff(warm_12yr - cold_12yr) -- 2x doubling, see note 3.
    warm_diffs = np.diff(warm_12yr)
    max_jump = max(warm_diffs.max(), -warm_diffs.min())

    print(f"\n  Schedule per doublet:")
    print(f"    Peak flow:   {max_flow:.1f} m³/hr = {max_mw:.1f} MW")
    print(f"    Thiem limit: 221 m³/hr = 13.3 MW per doublet")
    print(f"    Max jump:    {max_jump:.1f} m³/hr (target <30)")
    print(f"    Year-1 disch:{yr1_d} (must be 0)")
    print(f"    Weeks:       {len(warm_12yr)} (must be 624)")
    ok = max_flow <= 221 and yr1_d == 0 and len(warm_12yr) == 624
    print(f"    Status:      {'✓ READY' if ok else '✗ CHECK'}")

    output = os.path.join(ROSIM_DIR, f"option_a_{cfg['label']}_12yr.xlsx")
    write_rosim_excel(warm_12yr, cold_12yr, INJ_TEMP_C, COLD_TEMP_C, TEMPLATE, output)

    print(f"\n  First 10 warmwell rows:")
    for i in range(10):
        flag = " ← idle" if warm_12yr[i] == 0 else ""
        print(f"    Week {i+1:>3}: {warm_12yr[i]:>7.2f} m³/hr{flag}")

# ═══════════════════════════════════════════════════════════════════════════
# Well coordinates
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\nWELL COORDINATES FOR ROSIM")
print(f"{'='*55}")

# 1 doublet: keep the SAME physical doublet's established coordinates,
# as used in the earlier ROSIM schedules of this doublet.
r_th_1d = CONFIGS[0]['r_th']
print(f"\n1 doublet (13.3 MW):")
print(f"  Warmwell: i=40, j=42  (Y=1050m)")
print(f"  Coldwell: i=39, j=36  (Y=900m, 150m spacing)")
print(f"  r_th (live): {r_th_1d:.0f}m | L/r_th = {150/r_th_1d:.2f} "
      f"(established 150m comfortably exceeds 1.5×r_th={1.5*r_th_1d:.0f}m)")

# 2 doublets: no prior precedent -- derive spacing from the live r_th
# using the SAME formula as the 4/6-doublet script, same-i/different-j
# internal layout, offset-i between doublets.
r_th_2d = CONFIGS[1]['r_th']
L_internal = max(1.5 * r_th_2d, 75)
L_internal_cells = round(L_internal / CELL_M)
L_internal_m = L_internal_cells * CELL_M
L_between = max(2.5 * r_th_2d, 150)
L_between_cells = round(L_between / CELL_M)
L_between_m = L_between_cells * CELL_M

centre_i, centre_j = 40, 42
i1 = centre_i - L_between_cells // 2
i2 = centre_i + L_between_cells // 2
w_j = centre_j - L_internal_cells // 2   # warm outward-consistent tie-break
c_j = centre_j + L_internal_cells // 2

print(f"\n2 doublets (26.6 MW):")
print(f"  D1 Warmwell: i={i1}, j={w_j}  (Y={w_j*CELL_M:.0f}m)")
print(f"  D1 Coldwell: i={i1}, j={c_j}  (Y={c_j*CELL_M:.0f}m)")
print(f"  D2 Warmwell: i={i2}, j={w_j}  (Y={w_j*CELL_M:.0f}m)")
print(f"  D2 Coldwell: i={i2}, j={c_j}  (Y={c_j*CELL_M:.0f}m)")
print(f"  Internal spacing: {L_internal_m:.0f}m | Between doublets: {L_between_m:.0f}m")
print(f"  r_th (live) per doublet: {r_th_2d:.0f}m | "
      f"L/r_th internal = {L_internal_m/r_th_2d:.2f} | "
      f"L/r_th between = {L_between_m/r_th_2d:.2f} "
      f"{'✓' if L_between_m/r_th_2d >= 2.5 else '✗'}")

print(f"\nDo not modify any files outside post_midterm/.")
