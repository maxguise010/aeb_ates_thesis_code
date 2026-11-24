"""Original script for the ROSIM schedule of Option B-WN at 110 C; scenario_runner.py runs only its set-up part. The
full script needs template workbooks that are not included."""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dataclasses
import inspect
import shutil
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from openpyxl import load_workbook

from config import historical_cols, derivation_factors
from data_pipeline import build_sim_df_tank
import simulation_ates_three_loops_v7 as b_mod
from scenarios_v4 import (
    SCENARIO_ATES_B_WESTPOORT, SCENARIO_ATES_B_NOORD, SCENARIO_ATES_B_NIEUWWEST,
)

# Option B WN at 110°C — same Thiem limit as Option A
DELTA_T_WN  = 52.0    # K (was 32K at 90°C)
INJ_TEMP_WN = 110.0   # °C (was 90°C)
COLD_TEMP   = 58.0    # °C (unchanged)
FLOW_CAP    = 221.0   # m³/hr (same Thiem limit as Option A)
P_THIEM_WN  = 13.3    # MW
N_YEARS     = 12
H_AQ, N_E   = 33.8, 0.28

price_col = "Prijs_Gemiddelde_EPEX_Spot"
YEARS_LIST = [2023, 2024, 2025]
ROSIM_DIR = os.path.join(os.path.dirname(__file__), "..", "rosim_inputs")
OPTION_B_CHARGE_PCT, OPTION_B_DISCHARGE_PCT = 15, 75
TRIGGER_PP = 25
PERMISSIVE_DISCHARGE_PCT = 45.0
CORRECTIONS_KWARGS = dict(
    demand_correction_factor=1.038,
    enable_bec_summer_cap=True,
    enable_min_runtime=True,
    min_runtime_hours=18,
    min_downtime_hours=2,
    enable_ww_standby_correction=False,
)
OPTION_A_1D_PYTHON_SAVING = 2_012_898   # Option A 1 doublet, 13.3 MW (plot_option_a_rosim_comparison.py)

YEAR1_WEEKS = 52
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


# ═══════════════════════════════════════════════════════════════════════════
# Schedule helpers -- copied unchanged from the 2-doublet Option B schedule script
# ═══════════════════════════════════════════════════════════════════════════
def mwh_to_m3h_for(delta_t_k):
    return 3_600_000.0 / (4180.0 * delta_t_k)


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

        pos_vol = smoothed[smoothed > 0].sum()
        neg_vol = -smoothed[smoothed < 0].sum()
        chg_pct = (pos_vol - target_total_charge) / target_total_charge * 100 if target_total_charge else 0.0
        dsc_pct = (neg_vol - target_total_discharge) / target_total_discharge * 100 if target_total_discharge else 0.0

        print(f"    sigma={sigma:.1f}: rise={max_rise:.1f}  fall={max_fall:.1f} m3/hr "
              f"| chg {chg_pct:+.1f}%  dsc {dsc_pct:+.1f}%")

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
        reason = "lowest max fall with rise < target" if rise_ok else "lowest max fall (rise target unmet by any sigma)"
        print(f"    WARNING: no sigma satisfied both rise and fall < {max_jump:.0f} m3/hr")
        print(f"    -> Using sigma={sigma_used:.1f} ({reason}): rise={best_rise:.1f}, fall={best_fall:.1f}")

    warm_12yr = smoothed
    cold_12yr = -warm_12yr

    print(f"\n  FINAL 624-week check:")
    print(f"    Max rise: {np.diff(warm_12yr).max():.1f} m3/hr")
    print(f"    Max fall: {-np.diff(warm_12yr).min():.1f} m3/hr")
    print(f"    Year-1 discharge weeks: {int((cold_12yr[:YEAR1_WEEKS] > 0).sum())} (must be 0)")
    print(f"    Total weeks: {len(warm_12yr)} (must be {n_years * 52})")

    return warm_12yr, cold_12yr, sigma_used


def build_schedule_with_guard(monthly_c, monthly_d, delta_t_k, flow_cap, label):
    mwh_to_m3h = mwh_to_m3h_for(delta_t_k)
    warm, cold, sigma_used = build_smooth_schedule(monthly_c, monthly_d, delta_t_k=delta_t_k, flow_cap=flow_cap)

    target_discharge_mwh = float(sum(monthly_d))
    discharge_mwh = -warm[warm < 0].sum() / mwh_to_m3h * HOURS_PER_WEEK / (N_YEARS - 1)
    guard_threshold = 0.5 * target_discharge_mwh
    guard_passed = discharge_mwh >= guard_threshold

    if not guard_passed:
        print(f"\n  Discharge-volume guard FAILED for {label} at sigma={sigma_used:.1f} "
              f"({discharge_mwh:.0f} MWh/yr < 50% of {target_discharge_mwh:.0f}) -- "
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


def clip_and_rescale(warm, max_jump_target=25.0):
    signed = warm.copy()
    for i in range(1, len(signed)):
        delta = signed[i] - signed[i - 1]
        if abs(delta) > max_jump_target:
            signed[i] = signed[i - 1] + np.sign(delta) * max_jump_target

    target_charge = warm[warm > 0].sum()
    target_discharge = -warm[warm < 0].sum()
    pos_sum = signed[signed > 0].sum()
    neg_sum = -signed[signed < 0].sum()

    rescaled = signed.copy()
    if pos_sum > 0:
        rescaled = np.where(rescaled > 0, rescaled * target_charge / pos_sum, rescaled)
    if neg_sum > 0:
        rescaled = np.where(rescaled < 0, rescaled * target_discharge / neg_sum, rescaled)

    warm_out = rescaled
    warm_out[:YEAR1_WEEKS] = np.where(warm_out[:YEAR1_WEEKS] < 0, 0.0, warm_out[:YEAR1_WEEKS])
    cold_out = -warm_out

    jump_diffs = np.diff(warm_out)
    max_jump_after = max(jump_diffs.max(), -jump_diffs.min())

    return warm_out, cold_out, max_jump_after


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
    print(f"  Saved: {output_path} ({n} data rows)")


# ═══════════════════════════════════════════════════════════════════════════
# SOC+25pp->P45 patch for the three-loop model (verbatim from
# generate_optionb_soc_driven_schedules.py)
# ═══════════════════════════════════════════════════════════════════════════
OLD_LOOP_MODE_SRC = inspect.getsource(b_mod._loop_ates_mode)
OLD_LOOP_BODY = '''    soc_excess = soc - soc_target

    adj_disch_pct    = max(50.0, annual_discharge_pct - 70.0 * max(0.0, soc_excess))
    adj_disch_thresh = window_prices.quantile(adj_disch_pct / 100.0)
    want_discharge   = (price_now >= adj_disch_thresh) and (soc > soc_min)

    adj_charge_pct = annual_charge_pct - 70.0 * max(0.0, soc_excess)
    if adj_charge_pct < 5.0 or soc >= soc_max:
        want_charge = False
    else:
        adj_charge_thresh = window_prices.quantile(adj_charge_pct / 100.0)
        want_charge = (price_now <= adj_charge_thresh) and (soc < soc_max)

    if want_discharge:
        want_charge = False

    return "discharge" if want_discharge else ("charge" if want_charge else "drift")'''
NEW_LOOP_BODY = '''    soc_excess = soc - soc_target
    soc_deviation_pp = soc_excess * 100.0
    override_active = trigger_pp is not None and soc_deviation_pp > trigger_pp

    if override_active:
        adj_disch_pct = permissive_discharge_pct
    else:
        adj_disch_pct = max(50.0, annual_discharge_pct - 70.0 * max(0.0, soc_excess))
    adj_disch_thresh = window_prices.quantile(adj_disch_pct / 100.0)
    want_discharge   = (price_now >= adj_disch_thresh) and (soc > soc_min)

    adj_charge_pct = annual_charge_pct - 70.0 * max(0.0, soc_excess)
    if adj_charge_pct < 5.0 or soc >= soc_max:
        want_charge = False
    else:
        adj_charge_thresh = window_prices.quantile(adj_charge_pct / 100.0)
        want_charge = (price_now <= adj_charge_thresh) and (soc < soc_max)

    if want_discharge:
        want_charge = False

    mode = "discharge" if want_discharge else ("charge" if want_charge else "drift")
    return mode, override_active'''
assert OLD_LOOP_MODE_SRC.count(OLD_LOOP_BODY) == 1
loop_mode_src = OLD_LOOP_MODE_SRC.replace(OLD_LOOP_BODY, NEW_LOOP_BODY, 1)
loop_mode_src = loop_mode_src.replace(
    "    annual_discharge_pct,\n):",
    "    annual_discharge_pct,\n    trigger_pp=None,\n    permissive_discharge_pct=75.0,\n):",
    1,
)
assert loop_mode_src != OLD_LOOP_MODE_SRC
ns = dict(vars(b_mod))
exec(compile(loop_mode_src, "<_loop_ates_mode patched>", "exec"), ns)

OLD_PQUANTILE_BLOCK = '''    p90 = sim_df[price_col].quantile(0.90)              # very expensive (SOC floor 10%)
    p99 = sim_df[price_col].quantile(0.99)              # extreme spike (SOC floor 5%)'''
NEW_PQUANTILE_BLOCK = '''    p90 = sim_df[price_col].quantile(0.90)              # very expensive (SOC floor 10%)
    p99 = sim_df[price_col].quantile(0.99)              # extreme spike (SOC floor 5%)
    p_permissive = sim_df[price_col].quantile(permissive_discharge_pct / 100.0)'''

OLD_CALLS_BLOCK = '''        mode_wp = _loop_ates_mode(
            soc_wp, soc_target, ates_soc_min, ates_soc_max,
            price_now, window_prices, annual_charge_pct, annual_discharge_pct,
        )
        mode_n = _loop_ates_mode(
            soc_n, soc_target, ates_soc_min, ates_soc_max,
            price_now, window_prices, annual_charge_pct, annual_discharge_pct,
        )
        mode_nw = _loop_ates_mode(
            soc_nw, soc_target, ates_soc_min, ates_soc_max,
            price_now, window_prices, annual_charge_pct, annual_discharge_pct,
        )'''
NEW_CALLS_BLOCK = '''        mode_wp, override_wp = _loop_ates_mode(
            soc_wp, soc_target, ates_soc_min, ates_soc_max,
            price_now, window_prices, annual_charge_pct, annual_discharge_pct,
            trigger_pp=trigger_pp, permissive_discharge_pct=permissive_discharge_pct,
        )
        mode_n, override_n = _loop_ates_mode(
            soc_n, soc_target, ates_soc_min, ates_soc_max,
            price_now, window_prices, annual_charge_pct, annual_discharge_pct,
            trigger_pp=trigger_pp, permissive_discharge_pct=permissive_discharge_pct,
        )
        mode_nw, override_nw = _loop_ates_mode(
            soc_nw, soc_target, ates_soc_min, ates_soc_max,
            price_now, window_prices, annual_charge_pct, annual_discharge_pct,
            trigger_pp=trigger_pp, permissive_discharge_pct=permissive_discharge_pct,
        )'''

OLD_HARD_GATE_BLOCK = '''        # ── DISCHARGING: SOC floor varies by price tier ───────────────────────
        if price_now >= p99:
            soc_floor = 0.05
        elif price_now >= p90:
            soc_floor = 0.10
        elif price_now >= p85:
            soc_floor = 0.20
        elif price_now >= p75:
            soc_floor = 0.30
        else:
            soc_floor = 1.00

        # Summer discharge gate
        if month in [4, 5, 6, 7, 8, 9] and price_now < p75:
            soc_floor = 1.00

        # December reserve: preserve at least 35% SOC heading into year-end
        if month == 12:
            soc_floor = max(soc_floor, 0.35)

        # November reserve constraint: protect December/January capacity
        # Early November (days 1-15): floor at 25%; late November: floor at 35%
        if month == 11:
            day = sim_df.index[i].day
            november_floor = 0.25 if day <= 15 else 0.35
            soc_floor = max(soc_floor, november_floor)

        # Apply SOC floor per loop
        discharge_eligible_wp = ates_soc_wp > soc_floor
        discharge_eligible_no = ates_soc_no > soc_floor
        discharge_eligible_nw = ates_soc_nw > soc_floor'''
NEW_HARD_GATE_BLOCK = '''        def _soc_floor_for(override_active, month_, day_):
            price_floor = p_permissive if override_active else p75
            if price_now >= p99:
                floor_ = 0.05
            elif price_now >= p90:
                floor_ = 0.10
            elif price_now >= p85:
                floor_ = 0.20
            elif price_now >= price_floor:
                floor_ = 0.30
            else:
                floor_ = 1.00
            if month_ in [4, 5, 6, 7, 8, 9] and price_now < price_floor:
                floor_ = 1.00
            if month_ == 12:
                floor_ = max(floor_, 0.35)
            if month_ == 11:
                november_floor = 0.25 if day_ <= 15 else 0.35
                floor_ = max(floor_, november_floor)
            return floor_

        day = sim_df.index[i].day
        soc_floor_wp = _soc_floor_for(override_wp, month, day)
        soc_floor_no = _soc_floor_for(override_n, month, day)
        soc_floor_nw = _soc_floor_for(override_nw, month, day)

        discharge_eligible_wp = ates_soc_wp > soc_floor_wp
        discharge_eligible_no = ates_soc_no > soc_floor_no
        discharge_eligible_nw = ates_soc_nw > soc_floor_nw'''

OLD_SIG_TAIL = '''    demand_correction_factor: float = 1.0,
):'''
NEW_SIG_TAIL = '''    demand_correction_factor: float = 1.0,
    trigger_pp=None,
    permissive_discharge_pct: float = 75.0,
):'''

_src = inspect.getsource(b_mod.run_ates_three_loops_v6)
for _name, _block in [("p-quantile block", OLD_PQUANTILE_BLOCK), ("signature tail", OLD_SIG_TAIL),
                       ("loop-mode calls block", OLD_CALLS_BLOCK), ("hard gate block", OLD_HARD_GATE_BLOCK)]:
    _count = _src.count(_block)
    assert _count == 1, f"[{_name}] expected exactly 1 match, found {_count}"
_src = _src.replace(OLD_SIG_TAIL, NEW_SIG_TAIL, 1)
_src = _src.replace(OLD_PQUANTILE_BLOCK, NEW_PQUANTILE_BLOCK, 1)
_src = _src.replace(OLD_CALLS_BLOCK, NEW_CALLS_BLOCK, 1)
_src = _src.replace(OLD_HARD_GATE_BLOCK, NEW_HARD_GATE_BLOCK, 1)
_src = _src.replace("def run_ates_three_loops_v6(", "def run_ates_three_loops_soc_driven(", 1)
exec(compile(_src, "<generate_option_b_wn_110c_schedule patched>", "exec"), ns)
run_ates_three_loops_soc_driven = ns["run_ates_three_loops_soc_driven"]
print("Patched run_ates_three_loops_soc_driven() built successfully.\n")

print("OPTION B WN — 110°C VARIANT")
print("=" * 55)
print(f"  Injection temp:  {INJ_TEMP_WN}°C (was 90°C)")
print(f"  ΔT:              {DELTA_T_WN}K (was 32K)")
print(f"  Thiem flow:      {FLOW_CAP} m³/hr = {P_THIEM_WN} MW")
print(f"  (vs Option A:    221 m³/hr = 13.3 MW — identical)")

# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — WN-only dispatch at 13.3 MW, 110/58°C, P15/P75 + SOC+25pp->P45
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
hist_cost_per_year = (hist_loss_series * sim_df_tank[price_col]).groupby(sim_df_tank.index.year).sum()

BASE_WP = SCENARIO_ATES_B_WESTPOORT.ates_charge_max_mw
BASE_N = SCENARIO_ATES_B_NOORD.ates_charge_max_mw
scale_wn = P_THIEM_WN / (BASE_WP + BASE_N)
cap_wp, cap_n = BASE_WP * scale_wn, BASE_N * scale_wn
cfg_wp = dataclasses.replace(SCENARIO_ATES_B_WESTPOORT, ates_charge_max_mw=cap_wp, ates_discharge_max_mw=cap_wp,
                             ates_injection_temp_c=INJ_TEMP_WN, ates_cold_temp_c=COLD_TEMP)
cfg_n = dataclasses.replace(SCENARIO_ATES_B_NOORD, ates_charge_max_mw=cap_n, ates_discharge_max_mw=cap_n,
                            ates_injection_temp_c=INJ_TEMP_WN, ates_cold_temp_c=COLD_TEMP)
cfg_nw = dataclasses.replace(SCENARIO_ATES_B_NIEUWWEST, ates_charge_max_mw=0.0, ates_discharge_max_mw=0.0)

print(f"Running WN-only dispatch: WestPoort {cap_wp:.2f} + Noord {cap_n:.2f} = {cap_wp+cap_n:.1f} MW "
      f"at {INJ_TEMP_WN:.0f}/{COLD_TEMP:.0f}°C, NieuwWest 0 MW, "
      f"P{OPTION_B_CHARGE_PCT}/P{OPTION_B_DISCHARGE_PCT} + SOC+{TRIGGER_PP}pp->P{PERMISSIVE_DISCHARGE_PCT:.0f} ...")
df = run_ates_three_loops_soc_driven(
    sim_df_tank, cfg_wp, cfg_n, cfg_nw,
    trigger_pp=TRIGGER_PP, permissive_discharge_pct=PERMISSIVE_DISCHARGE_PCT,
    annual_charge_pct=OPTION_B_CHARGE_PCT, annual_discharge_pct=OPTION_B_DISCHARGE_PCT,
    **CORRECTIONS_KWARGS,
)
print("  Done.")
df = df.copy()
df['wn_charge_mw'] = df['ates_westpoort_charge_mw'] + df['ates_noord_charge_mw']
df['wn_discharge_mw'] = df['ates_westpoort_discharge_mw'] + df['ates_noord_discharge_mw']

saving = sum(hist_cost_per_year[y] - df[df.index.year == y]["total_system_cost_eur_corrected"].sum()
             for y in YEARS_LIST) / 3.0

monthly_charge = (df[df['wn_charge_mw'] > 0].groupby(df[df['wn_charge_mw'] > 0].index.month)
                  ['wn_charge_mw'].sum() / 3).reindex(range(1, 13), fill_value=0.0)
monthly_discharge = (df[df['wn_discharge_mw'] > 0].groupby(df[df['wn_discharge_mw'] > 0].index.month)
                     ['wn_discharge_mw'].sum() / 3).reindex(range(1, 13), fill_value=0.0)

print(f"\nStep 1 — WN dispatch at {P_THIEM_WN} MW, P15/P75:")
print(f"  Hourly max:        charge {df['wn_charge_mw'].max():.2f} MW, discharge {df['wn_discharge_mw'].max():.2f} MW")
print(f"  NieuwWest check:   charged {df['ates_nieuwwest_charge_mw'].sum()/3:.1f} MWh/yr (must be 0)")
print(f"  Annual charged:    {monthly_charge.sum():.0f} MWh/yr")
print(f"  Annual discharged: {monthly_discharge.sum():.0f} MWh/yr")
print(f"  η_Python:          {monthly_discharge.sum()/monthly_charge.sum()*100:.1f}%")
print(f"  Annual saving:     €{saving:,.0f}/yr  (Option A 1D, 13.3 MW: €{OPTION_A_1D_PYTHON_SAVING:,.0f}/yr)")

# r_th (project convention sqrt(V/(pi*H*n)), the radius of the injected water)
V_inj = monthly_charge.sum() * 3.6e9 / (1000 * 4180 * DELTA_T_WN)
r_th  = np.sqrt(V_inj / (np.pi * H_AQ * N_E))
print(f"\n  V_inj:  {V_inj:,.0f} m³/yr")
print(f"  r_th:   {r_th:.1f} m")
print(f"  Recommended spacing: {1.5*r_th:.0f}m (1.5×r_th)")
print(f"  Current WN spacing:  125m → L/r_th = {125/r_th:.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Build schedule from monthly totals
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nStep 2 — Building 624-week schedule:")
warm_12yr, cold_12yr, sigma = build_schedule_with_guard(
    monthly_charge.values, monthly_discharge.values,
    delta_t_k=DELTA_T_WN, flow_cap=FLOW_CAP, label="Option B WN 110°C",
)
d = np.diff(warm_12yr)
if max(d.max(), -d.min()) > 30.0:
    print(f"  Max jump {max(d.max(), -d.min()):.1f} m³/hr > 30 -- applying clip_and_rescale(25.0)")
    warm_12yr, cold_12yr, _ = clip_and_rescale(warm_12yr, max_jump_target=25.0)

# Year-1: volume-matched sine from 5 m³/hr (same injected volume as year 1 of the built schedule)
MIN_START = 5.0
orig_yr1_m3 = warm_12yr[:52][warm_12yr[:52] > 0].sum() * 168
sine_yr1  = np.sin(np.pi * np.arange(52) / 51)
yr1_peak  = MIN_START + (orig_yr1_m3 / 168 - 52 * MIN_START) / sine_yr1.sum()
yr1_peak  = max(yr1_peak, MIN_START + 1.0)
yr1_flows = MIN_START + (yr1_peak - MIN_START) * sine_yr1
warm_12yr = warm_12yr.copy()
warm_12yr[:52] = yr1_flows
cold_12yr = -warm_12yr
yr1_mwh = yr1_flows.sum() * 168 * 1000 * 4180 * DELTA_T_WN / 3.6e9
orig_yr1_mwh = orig_yr1_m3 * 1000 * 4180 * DELTA_T_WN / 3.6e9
print(f"\n  Year-1 replaced: sine 5.0 → {yr1_peak:.1f} m³/hr, {yr1_mwh:.0f} MWh "
      f"(original year 1: {orig_yr1_mwh:.0f} MWh)")

max_flow  = np.abs(warm_12yr).max()
max_mw    = max_flow * 1000 * 4180 * DELTA_T_WN / 3.6e9
yr1_disch = int((cold_12yr[:52] > 0).sum())
diffs     = np.diff(warm_12yr)
max_jump  = max(diffs.max(), -diffs.min())

print(f"\nStep 2 — Schedule:")
print(f"  Sigma:        {sigma:.1f}")
print(f"  Peak flow:    {max_flow:.1f} m³/hr = {max_mw:.2f} MW")
print(f"  Thiem limit:  {FLOW_CAP:.0f} m³/hr = {P_THIEM_WN} MW")
print(f"  Max jump:     {max_jump:.1f} m³/hr (target <30)")
print(f"  Year-1 disch: {yr1_disch} (must be 0)")
print(f"  Weeks:        {len(warm_12yr)} (must be 624)")

ok = max_flow <= FLOW_CAP and yr1_disch == 0 and len(warm_12yr) == 624 and max_jump < 30.0
print(f"  Status:       {'✓ READY' if ok else '✗ CHECK'}")

# Write file
output   = os.path.join(ROSIM_DIR, 'option_b_wn_110c_12yr.xlsx')
template = os.path.join(ROSIM_DIR, 'option_b_wn_p15p75_12yr_fixed.xlsx')
write_rosim_excel(warm_12yr, cold_12yr, INJ_TEMP_WN, COLD_TEMP, template, output)
wb = load_workbook(output)
for sheet in ["Warmwell", "Coldwell"]:
    wb[sheet]["A1"].value = f"Option B WestPoort+Noord (combined) — 110°C, 13.3 MW, P15/P75 SOC+25pp→P45 — 12-jaars schema — {sheet}"
wb.save(output)
print(f"\nSaved: option_b_wn_110c_12yr.xlsx")

print(f"\nFirst 10 warmwell rows:")
for i in range(10):
    print(f"  Week {i+1:>3}: {warm_12yr[i]:>7.2f} m³/hr")
print(f"\nWeeks 50-55 (yr1→yr2):")
for i in range(49, 55):
    print(f"  Week {i+1:>3}: {warm_12yr[i]:>7.2f} m³/hr")

print(f"\nROSIM CONFIGURATION — Option B WN at 110°C:")
print(f"  File:     option_b_wn_110c_12yr.xlsx")
print(f"  Warmwell: i=40, j=42")
print(f"  Coldwell: i=40, j=37  (125m spacing, L/r_th={125/r_th:.2f})")
print(f"  Inj temp: 110°C | ΔT=52K | 12 jaar")
print(f"\n  Note: NW doublet remains at 90°C (separate loop, no 110°C available)")
print(f"\nComparison with original Option B WN (90°C):")
print(f"  90°C: Thiem=179 m³/hr=6.6MW, ΔT=32K")
print(f"  110°C: Thiem=221 m³/hr=13.3MW, ΔT=52K")
print(f"  Power ratio: {13.3/6.6:.1f}× higher at 110°C")
print(f"\nDo not modify any files outside post_midterm/.")
