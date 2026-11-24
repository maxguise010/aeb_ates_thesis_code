"""Dispatch model of Option A (central ATES with 1, 2 or 4 doublets) with the eligibility gates of Appendix A. Needs
the raw data."""
import numpy as np
import pandas as pd

from ates_thermal_v2 import ATESThermalState
from aeb_buffer_rule import AebBufferRule, check_rule
from config import (
    CONST,
    priority_order,
    hx_params,
    derivation_factors,
    bec_own_use_mw,
    tank_charge_max_mw,
    tank_discharge_max_mw,
)

# ── POST-MIDTERM BEC REVENUE FORMULA (version 2) — Option A ──
# Identical to the three-loop v7 implementation.
# WW standby correction NOT applied — same reasoning:
# with ATES storage WW30/WW01 may legitimately be switched off.
#
# Source: supervisor analysis BEC_extra_warmte_zomer_analyse_v2
# Correction: BEC yields a net revenue per GJ of heat,
# not a cost item. Extra BEC production above the historical
# baseline is counted as additional system revenue.
# Fully replaces the v1 cost correction (33.8 EUR/MWh) —
# that formula and BEC_MARGINAL_COST_EUR_PER_MWH no longer
# exist here.

BEC_BIOMASSA_PRIJS_EUR_PER_GJ = 5.69    # average 2025
BEC_SUBSIDIE_PRIJS_EUR_PER_GJ = 9.17    # subsidy per GJ
BEC_ELEK_PER_WARMTE = 1.09 / 3.93      # GJ electricity per GJ heat
BEC_BIOMASSA_PER_WARMTE = 5.69 / 3.93  # GJ biomass per GJ heat
BEC_SUBSIDIE_PER_WARMTE = 1 + BEC_ELEK_PER_WARMTE  # GJ eligible for subsidy per GJ heat
BEC_MIN_OPERATIONAL_MW = 1.0            # MW threshold for standstill
BEC_BASELINE_CORRECTION = True
enable_ww_standby_correction = False

# ── MINIMUM RUNTIME CONSTRAINT (ATES models) ───────────────
# Heat exchangers may not be switched on/off too often.
# WW standby correction is OFF for ATES models — ATES covers
# the heat demand, so that WWs can legitimately be switched off.
# Minimum runtime, however, is a physical constraint that always
# applies, whether or not ATES is present.
# Source: supervisor advice July 2026
MIN_RUNTIME_HX = ['WW10', 'WW30', 'WW01']

# ── BEC ZOMERBEPERKING ────────────────────────────────────────
# Between 1 May and 1 October at most 70% of the nominal BEC power.
# Also applies to ATES simulations.
BEC_SUMMER_CAP_FRACTION = 0.70
BEC_SUMMER_START_MONTH  = 5
BEC_SUMMER_END_MONTH    = 10


def bec_net_revenue_per_gj(epex_eur_per_mwh):
    # Convert EPEX from €/MWh to €/GJ (1 MWh = 3.6 GJ)
    epex_gj = epex_eur_per_mwh / 3.6

    biomassa_bedrag = -BEC_BIOMASSA_PER_WARMTE * BEC_BIOMASSA_PRIJS_EUR_PER_GJ
    elek_bedrag     =  BEC_ELEK_PER_WARMTE     * epex_gj
    subsidie_bedrag =  BEC_SUBSIDIE_PER_WARMTE * BEC_SUBSIDIE_PRIJS_EUR_PER_GJ

    return biomassa_bedrag + elek_bedrag + subsidie_bedrag  # €/GJ heat


def run_ates_option_a_v6(
    sim_input_df,
    config,
    use_dynamic_cold_temp: bool = False,
    n_small_tanks=3,
    small_tank_volume_m3=1200.0,
    small_charge_max_mw=None,
    small_discharge_max_mw=None,
    small_initial_soc=0.5,
    small_soc_max=0.95,
    small_soc_min=0.10,
    ates_soc_max=0.95,
    ates_soc_min=0.05,
    ates_initial_soc=0.0,
    annual_charge_pct=None,
    annual_discharge_pct=None,
    daily_charge_rank_n=6,
    daily_discharge_rank_n=6,
    reserve_horizon_h=6,
    reserve_max_fraction=0.20,
    enable_pure_charge_ates=True,
    ates_mech_pump_mwh_per_mwh_th=72.91 / 60_000,
    price_col="Prijs_Gemiddelde_EPEX_Spot",
    perfect_foresight=False,
    # Minimum runtime constraint -- see MIN_RUNTIME_HX above.
    enable_min_runtime: bool = False,
    min_runtime_hours: int = 18,   # at least 18 hours on once started
    min_downtime_hours: int = 6,   # at least 6 hours off after switching off
    # BEC summer cap -- see module-level BEC_SUMMER_CAP_FRACTION above.
    enable_bec_summer_cap: bool = False,
    # Heat demand correction factor -- same as simulation_v7.py.
    demand_correction_factor: float = 1.0,
    # Rule for the existing buffers: "rank" (default) or "aeb_demand" (aeb_buffer_rule.py).
    existing_buffer_rule: str = "rank",
    # Conditional release of the existing buffers (default None = off), as in simulation_v7.py:
    # "hourly": buffer floor = release reserve (demand of the next release_reserve_horizon_h hours, max.
    # release_reserve_max_fraction x buffer volume) in hours with ATES backup, otherwise the base floor.
    # Backup: H = (V_ates - ates_soc_min x cap) x CONST x dT_inj >= R (MWh) and, if
    # release_power_horizon_h is set, also discharge limit x release_power_horizon_h >= R.
    buffer_release_rule=None,
    release_reserve_horizon_h=6,
    release_reserve_max_fraction=0.20,
    release_power_horizon_h=None,
    # ── Dispatch strategy hook (run_improved_dispatch.py) ──
    # 'S0' (default) is the original threshold logic, unchanged (checked in run_improved_dispatch.py against
    # EUR 2.249.171/yr); 'S1'-'S4' replace only the threshold block below, every other mechanism stays the same.
    # Their input columns (forecast_ratio, next24h_min_price, next24h_max_price, calendar_traj_7d,
    # calendar_traj_14d) must be joined onto sim_input_df by the caller; this function does not compute them.
    dispatch_strategy: str = "S0",
):
    """
    Option A single-store HT-ATES with V6 graduated dispatch.

    Rolling SOC-excess quantile signal provides base charge/discharge
    want; V6 graduated tier gates and seasonal reserve constraints
    then gate eligibility. Negative prices always trigger unconditional
    charging regardless of season or SOC ceiling.

    Dispatches against total heat_demand_mw (full WPW network demand),
    not per-loop columns.
    """
    sim_df = sim_input_df.copy()

    if annual_charge_pct is None:
        annual_charge_pct = config.annual_charge_pct
    if annual_discharge_pct is None:
        annual_discharge_pct = config.annual_discharge_pct

    if small_charge_max_mw is None:
        small_charge_max_mw = tank_charge_max_mw * (n_small_tanks / 3.0)
    if small_discharge_max_mw is None:
        small_discharge_max_mw = tank_discharge_max_mw * (n_small_tanks / 3.0)

    small_volume_max = n_small_tanks * small_tank_volume_m3
    small_volume     = small_volume_max * small_initial_soc

    dT_inj = max(1.0, config.ates_injection_temp_c - config.ates_cold_temp_c)

    ates_capacity_m3   = config.ates_capacity_m3
    ates_charge_max    = config.ates_charge_max_mw
    ates_discharge_max = config.ates_discharge_max_mw

    ates_state = ATESThermalState(config)
    ates_state.V_stored = ates_capacity_m3 * ates_initial_soc

    _DYN_COL   = "T_return_C"
    _dyn_avail = use_dynamic_cold_temp and (_DYN_COL in sim_df.columns)
    if use_dynamic_cold_temp and not _dyn_avail:
        print(
            f"[option_a_v6 WARNING] use_dynamic_cold_temp=True but column "
            f"'{_DYN_COL}' not found. Falling back to config.ates_cold_temp_c = "
            f"{config.ates_cold_temp_c:.1f} degC."
        )

    trailing_window_h = 365 * 24

    # ── Graduated price thresholds ────────────────────────────────────────────
    p01 = max(sim_df[price_col].quantile(0.01), 0.0)
    p10 = sim_df[price_col].quantile(0.10)
    p25 = sim_df[price_col].quantile(0.25)
    p75 = sim_df[price_col].quantile(0.75)
    p85 = sim_df[price_col].quantile(0.85)
    p90 = sim_df[price_col].quantile(0.90)
    p99 = sim_df[price_col].quantile(0.99)

    print(f"  Option A V6 graduated thresholds (EUR/MWh):")
    print(f"    P01={p01:.1f}  P10={p10:.1f}  P25={p25:.1f}")
    print(f"    P75={p75:.1f}  P85={p85:.1f}  P90={p90:.1f}  P99={p99:.1f}")
    print(f"    Negative-price unconditional charge: active below 0.0 EUR/MWh")

    # Annual discharge tracker — reset each January, cap at 3x capacity
    annual_discharge_m3 = 0.0
    MAX_DISCHARGE_RATIO = 3.0

    # ── Pre-compute dispatch signals ──────────────────────────────────────────
    if perfect_foresight:
        sim_df["ates_charge_thresh"]    = sim_df[price_col].quantile(annual_charge_pct / 100)
        sim_df["ates_discharge_thresh"] = sim_df[price_col].quantile(annual_discharge_pct / 100)
    else:
        sim_df["ates_charge_thresh"] = (
            sim_df[price_col]
            .rolling(window=trailing_window_h, min_periods=24)
            .quantile(annual_charge_pct / 100)
        )
        sim_df["ates_discharge_thresh"] = (
            sim_df[price_col]
            .rolling(window=trailing_window_h, min_periods=24)
            .quantile(annual_discharge_pct / 100)
        )

    rank_asc  = sim_df.groupby(sim_df.index.date)[price_col].rank(method="first", ascending=True)
    rank_desc = sim_df.groupby(sim_df.index.date)[price_col].rank(method="first", ascending=False)
    sim_df["small_charge_signal"]    = rank_asc  <= daily_charge_rank_n
    sim_df["small_discharge_signal"] = rank_desc <= daily_discharge_rank_n

    demand        = sim_df["heat_demand_mw"].copy()
    demand_padded = pd.concat([demand, demand.iloc[:90 * 24]])
    forward_mean  = demand_padded.rolling(window=90 * 24, min_periods=24).mean().shift(-(90 * 24 - 1))
    forward_mean  = forward_mean.iloc[:len(demand)].bfill().ffill()
    fd_min, fd_max = forward_mean.min(), forward_mean.max()
    if fd_max > fd_min:
        sim_df["ates_soc_target_seasonal"] = (
            ates_soc_min
            + (ates_soc_max - ates_soc_min)
            * (forward_mean - fd_min) / (fd_max - fd_min)
        ).values
    else:
        sim_df["ates_soc_target_seasonal"] = 0.5

    if perfect_foresight:
        sim_df["ates_annual_50_thresh"] = sim_df[price_col].quantile(0.50)
    else:
        sim_df["ates_annual_50_thresh"] = (
            sim_df[price_col]
            .rolling(window=trailing_window_h, min_periods=24)
            .quantile(0.50)
        )

    MIN_PHYSICAL_DELTA_T = 10.0
    safe_delta_t = sim_df["delta_t_c"].clip(lower=MIN_PHYSICAL_DELTA_T)
    sim_df["required_volume_for_demand_m3"] = sim_df["heat_demand_mw"] / (CONST * safe_delta_t)
    reserve_raw = sum(
        sim_df["required_volume_for_demand_m3"].shift(-h).fillna(0.0)
        for h in range(1, reserve_horizon_h + 1)
    )
    reserve_cap_abs = reserve_max_fraction * small_volume_max
    sim_df["small_min_volume_required_m3"] = np.clip(reserve_raw, 0.0, reserve_cap_abs)
    if buffer_release_rule is not None:
        if buffer_release_rule != "hourly":
            raise ValueError(f"buffer_release_rule must be None or 'hourly', got {buffer_release_rule!r}")
        rel_raw = sum(sim_df["required_volume_for_demand_m3"].shift(-h).fillna(0.0)
                      for h in range(1, release_reserve_horizon_h + 1))
        sim_df["release_min_volume_m3"] = np.clip(rel_raw, 0.0, release_reserve_max_fraction * small_volume_max)
        for c in ("buffer_release_flag", "backup_heat_ok", "backup_power_ok", "backup_heat_mwh", "backup_reserve_mwh"):
            sim_df[c] = 0.0

    # Baseline only active if historical BEC >= threshold
    sim_df["bec_hist_active"] = sim_df["WW_BEC"] >= BEC_MIN_OPERATIONAL_MW

    # ── BEC summer cap (May to September) ────────────────────────────
    if enable_bec_summer_cap:
        bec_max_mw_col = sim_df.index.month.map(
            lambda m: (hx_params['WWBEC']['max_mw'] * BEC_SUMMER_CAP_FRACTION)
                      if (BEC_SUMMER_START_MONTH <= m < BEC_SUMMER_END_MONTH)
                      else hx_params['WWBEC']['max_mw']
        )
        sim_df['WWBEC_max_mw_effective'] = bec_max_mw_col
    else:
        sim_df['WWBEC_max_mw_effective'] = hx_params['WWBEC']['max_mw']

    for hx in priority_order:
        sim_df[f"{hx}_sim_tank_mw"]         = 0.0
        sim_df[f"{hx}_on_tank"]             = 0
        sim_df[f"{hx}_electricity_loss_mw"] = 0.0

    output_cols = [
        "small_tank_volume_m3",
        "small_tank_charge_mw", "small_tank_discharge_mw",
        "ates_volume_m3", "ates_soc",
        "ates_charge_mw", "ates_discharge_mw",
        "ates_aquifer_temp_c", "ates_outlet_temp_c", "ates_temp_gap_c",
        "ates_heat_pump_mwel", "ates_useful_discharge_mw", "ates_ground_loss_mw",
        "ates_mech_pump_mwh", "ates_pump_cost_eur",
        "tank_charge_mw", "tank_discharge_mw",
        "direct_from_hx_mw", "direct_from_tank_mw",
        "total_generation_tank_model_mw", "BEC_own_use_mw",
        "total_generation_usable_mw", "total_supply_to_customer_mw",
        "unmet_demand_tank_mw", "curtailed_or_excess_mw",
        "total_electricity_loss_mw", "derving_cost_eur",
        "total_system_cost_eur",
        "bec_hist_baseline_mw", "bec_extra_production_mw",
        "bec_extra_revenue_eur", "bec_correction_cost_eur", "total_system_cost_eur_corrected",
        "pure_charge_ates_mw", "small_mode_v2", "ates_mode_v2", "bec_displaced_flag",
        "ates_dT_inj_t",
    ]
    for hx in MIN_RUNTIME_HX:
        output_cols.append(f"{hx}_minruntime_forced_on_flag")
        output_cols.append(f"{hx}_minruntime_forced_off_flag")
    for c in output_cols:
        sim_df[c] = 0.0

    check_rule(existing_buffer_rule)
    aeb = (AebBufferRule(sim_df, small_volume_max, price_col)
           if existing_buffer_rule == "aeb_demand" else None)

    prev_dispatch = {hx: 0.0 for hx in priority_order}

    if enable_min_runtime:
        # Tracks how many hours each HX has been continuously on or off
        hx_consecutive_on  = {hx: 0 for hx in MIN_RUNTIME_HX}
        hx_consecutive_off = {hx: 999 for hx in MIN_RUNTIME_HX}
        # 999 = off long enough at the start of the simulation

    non_bec_dispatch_sequence = [
        "WW10", "WW30_aftap", "WW01", "WW00", "WW02", "WW30_reductie",
    ]
    pure_charge_order = sorted(
        [hx for hx in priority_order if hx != "WWBEC"],
        key=lambda h: derivation_factors["WW30_aftap"] if h == "WW30" else derivation_factors[h],
    )
    col_idx = {c: sim_df.columns.get_loc(c) for c in sim_df.columns}

    # ── Main simulation loop ──────────────────────────────────────────────────
    for i in range(len(sim_df)):
        row       = sim_df.iloc[i]
        demand_t  = row["heat_demand_mw"] * demand_correction_factor
        delta_t   = max(row["delta_t_c"], MIN_PHYSICAL_DELTA_T)
        price_now = row[price_col]
        month     = sim_df.index[i].month
        day       = sim_df.index[i].day

        if _dyn_avail:
            dT_inj_t = max(1.0, config.ates_injection_temp_c - row[_DYN_COL])
        else:
            dT_inj_t = dT_inj

        small_min_vol = row["small_min_volume_required_m3"]
        if buffer_release_rule is not None:
            heat_mwh = max(0.0, ates_state.V_stored - ates_soc_min * ates_capacity_m3) * CONST * dT_inj_t
            res_mwh = row["release_min_volume_m3"] * CONST * delta_t
            heat_ok = heat_mwh >= res_mwh
            power_ok = (release_power_horizon_h is None
                        or ates_discharge_max * release_power_horizon_h >= res_mwh)
            if heat_ok and power_ok:
                small_min_vol = row["release_min_volume_m3"]
            for c, v in (("buffer_release_flag", float(heat_ok and power_ok)), ("backup_heat_ok", float(heat_ok)),
                         ("backup_power_ok", float(power_ok)), ("backup_heat_mwh", heat_mwh),
                         ("backup_reserve_mwh", res_mwh)):
                sim_df.iat[i, col_idx[c]] = v
        small_soc     = small_volume / small_volume_max if small_volume_max > 0 else 0.0
        ates_soc      = ates_state.V_stored / ates_capacity_m3 if ates_capacity_m3 > 0 else 0.0

        # ── Small tank mode (daily tier-2 price-rank signal) ─────────────────
        small_can_charge    = small_soc < small_soc_max
        small_can_discharge = small_soc > small_soc_min
        small_want_charge    = bool(row["small_charge_signal"]) and small_can_charge
        small_want_discharge = bool(row["small_discharge_signal"]) and small_can_discharge
        if small_want_discharge:
            small_want_charge = False
        small_mode = "charge" if small_want_charge else ("discharge" if small_want_discharge else "drift")
        if aeb is not None:
            small_mode, small_min_vol = aeb.hour(i)

        # ── ATES rolling signal (SOC-excess adjusted trailing quantile) ───────
        ates_can_charge    = ates_soc < ates_soc_max
        ates_can_discharge = ates_soc > ates_soc_min
        ates_target        = row["ates_soc_target_seasonal"]
        soc_excess         = ates_soc - ates_target

        if perfect_foresight:
            window_prices = sim_df[price_col]
        else:
            window_start_idx = max(0, i - trailing_window_h + 1)
            window_prices    = sim_df[price_col].iloc[window_start_idx:i + 1]

        # ── Strategy-specific base percentile + rate multiplier ───────────────
        # dispatch_strategy='S0': base_charge_pct/base_discharge_pct stay
        # exactly annual_charge_pct/annual_discharge_pct as passed in, and
        # both rate multipliers stay 1.0 -- byte-for-byte identical to the
        # original (pre-hook) logic. See the function
        # signature comment above for the full strategy catalogue; the
        # strategy-specific input columns are computed/joined by the
        # caller (run_improved_dispatch.py), not here.
        base_charge_pct    = annual_charge_pct
        base_discharge_pct = annual_discharge_pct
        ates_charge_rate_mult    = 1.0
        ates_discharge_rate_mult = 1.0
        force_charge_override    = False
        force_discharge_override = False

        if dispatch_strategy == "S0":
            pass

        elif dispatch_strategy == "S1":
            fr = row["forecast_ratio"]
            if pd.isna(fr):
                fr = 1.0
            if fr > 1.50:
                base_charge_pct = 45
            elif fr > 1.25:
                base_charge_pct = 35
            elif fr < 0.80:
                base_charge_pct = 15
            else:
                base_charge_pct = 25
            if fr < 0.75:
                base_discharge_pct = 65
            elif fr > 1.30:
                base_discharge_pct = 85
            else:
                base_discharge_pct = 75

        elif dispatch_strategy == "S2":
            da_min = row["next24h_min_price"]
            da_max = row["next24h_max_price"]
            if not pd.isna(da_min):
                if price_now <= da_min * 1.05:
                    force_charge_override = True
                elif price_now > da_min * 1.30:
                    ates_charge_rate_mult = 0.2
            if not pd.isna(da_max):
                if price_now >= da_max * 0.95:
                    force_discharge_override = True
                elif price_now < da_max * 0.70:
                    ates_discharge_rate_mult = 0.2

        elif dispatch_strategy == "S3":
            traj_14d = row["calendar_traj_14d"]
            traj_7d  = row["calendar_traj_7d"]
            if pd.isna(traj_14d):
                traj_14d = 0.0
            if pd.isna(traj_7d):
                traj_7d = 0.0
            if traj_14d > 0.40:
                base_charge_pct = 50
            elif traj_14d > 0.20:
                base_charge_pct = 40
            elif traj_14d < -0.20:
                base_charge_pct = 12
            else:
                base_charge_pct = 25
            if traj_7d < -0.15:
                base_discharge_pct = 65
            elif traj_7d > 0.25:
                base_discharge_pct = 88
            else:
                base_discharge_pct = 75

        elif dispatch_strategy == "S4":
            # Level 2+3: seasonal trajectory (S3 rule), then forecast ratio
            # (S1 rule) layered on top -- S1's "else" branch becomes the
            # S3-adjusted percentile rather than the flat 25/75 default.
            traj_14d = row["calendar_traj_14d"]
            traj_7d  = row["calendar_traj_7d"]
            if pd.isna(traj_14d):
                traj_14d = 0.0
            if pd.isna(traj_7d):
                traj_7d = 0.0
            if traj_14d > 0.40:
                base_charge_pct = 50
            elif traj_14d > 0.20:
                base_charge_pct = 40
            elif traj_14d < -0.20:
                base_charge_pct = 12
            else:
                base_charge_pct = 25
            if traj_7d < -0.15:
                base_discharge_pct = 65
            elif traj_7d > 0.25:
                base_discharge_pct = 88
            else:
                base_discharge_pct = 75

            fr = row["forecast_ratio"]
            if pd.isna(fr):
                fr = 1.0
            if fr > 1.50:
                base_charge_pct = 45
            elif fr > 1.25:
                base_charge_pct = 35
            elif fr < 0.80:
                base_charge_pct = 15
            if fr < 0.75:
                base_discharge_pct = 65
            elif fr > 1.30:
                base_discharge_pct = 85

            # Level 1: day-ahead, highest priority -- overrides/dampens
            # whatever Level 2+3 (+ Level 4 gate below) decided.
            da_min = row["next24h_min_price"]
            da_max = row["next24h_max_price"]
            if not pd.isna(da_min):
                if price_now <= da_min * 1.05:
                    force_charge_override = True
                elif price_now > da_min * 1.30:
                    ates_charge_rate_mult = 0.15
            if not pd.isna(da_max):
                if price_now >= da_max * 0.95:
                    force_discharge_override = True
                elif price_now < da_max * 0.70:
                    ates_discharge_rate_mult = 0.15

        # ── Level 4 / S0 mechanism: SOC-excess adjusted trailing quantile ──
        adjusted_discharge_pct    = max(50.0, base_discharge_pct - 70.0 * max(0.0, soc_excess))
        adjusted_discharge_thresh = window_prices.quantile(adjusted_discharge_pct / 100.0)
        ates_discharge_signal     = price_now >= adjusted_discharge_thresh

        adjusted_charge_pct = base_charge_pct - 70.0 * max(0.0, soc_excess)
        if adjusted_charge_pct < 5.0:
            ates_charge_signal = False
        else:
            adjusted_charge_thresh = window_prices.quantile(adjusted_charge_pct / 100.0)
            ates_charge_signal     = price_now <= adjusted_charge_thresh

        if force_charge_override:
            ates_charge_signal = True
        if force_discharge_override:
            ates_discharge_signal = True

        ates_want_charge    = ates_charge_signal    and ates_can_charge
        ates_want_discharge = ates_discharge_signal and ates_can_discharge
        if ates_want_discharge:
            ates_want_charge = False

        # ── V6 Graduated dispatch — Option A single store ─────────────────────
        if price_now < 0:
            # Unconditional negative-price charging: force eligible, block discharge
            charge_eligible    = True
            discharge_eligible = False

        else:
            # ── CHARGING TIERS ────────────────────────────────────────────────
            if price_now <= p01:
                charge_eligible = True           # extreme bargain: charge any season
            elif price_now <= p10:
                charge_eligible = True           # very cheap: charge any season
            elif price_now <= p25:
                charge_eligible = month in [4, 5, 6, 7, 8, 9, 10]  # Apr-Oct only
            else:
                charge_eligible = False

            # ── DISCHARGING WITH VARIABLE SOC FLOOR ──────────────────────────
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
                soc_floor = 1.00

            # November reserve constraint
            if month == 11:
                november_floor = 0.25 if day <= 15 else 0.35
                soc_floor = max(soc_floor, november_floor)

            # December reserve constraint
            if month == 12:
                soc_floor = max(soc_floor, 0.35)

            # October SOC gate: must be above 50% before winter discharge
            if month == 10 and ates_soc < 0.50:
                soc_floor = 1.00

            # Annual discharge volume cap
            if annual_discharge_m3 > MAX_DISCHARGE_RATIO * config.ates_capacity_m3:
                soc_floor = 1.00

            discharge_eligible = ates_soc > soc_floor

        # Apply eligibility to existing want signals
        ates_want_charge    = ates_want_charge    and charge_eligible
        ates_want_discharge = ates_want_discharge and discharge_eligible

        ates_mode = "charge" if ates_want_charge else ("discharge" if ates_want_discharge else "drift")

        # Reset annual discharge tracker on 1 January
        if month == 1 and day == 1:
            annual_discharge_m3 = 0.0

        sim_df.iat[i, col_idx["small_mode_v2"]] = (
            1.0 if small_mode == "charge" else (-1.0 if small_mode == "discharge" else 0.0)
        )
        sim_df.iat[i, col_idx["ates_mode_v2"]] = (
            1.0 if ates_mode == "charge" else (-1.0 if ates_mode == "discharge" else 0.0)
        )

        for hx in priority_order:
            sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]] = 0.0
            sim_df.iat[i, col_idx[f"{hx}_on_tank"]]     = 0

        # ── BEC dispatch ──────────────────────────────────────────────────────
        bec_p    = hx_params["WWBEC"]
        bec_min  = row["WWBEC_min_mw"]
        bec_max  = row["WWBEC_max_mw_effective"]
        bec_prev = prev_dispatch["WWBEC"]
        bec_hist = row["WWBEC_hist_available"] == 1

        annual_50 = row["ates_annual_50_thresh"]
        if pd.isna(annual_50):
            annual_50 = window_prices.quantile(0.50)

        bec_displace = (
            ates_capacity_m3 > 0
            and ates_soc > 0.85
            and price_now >= annual_50
            and ates_can_discharge
            and bec_hist
        )

        if not bec_hist:
            bec_dispatch = 0.0
        else:
            bec_target     = bec_min if bec_displace else max(bec_min, min(demand_t + bec_own_use_mw, bec_max))
            bec_ramp_lower = max(0.0, bec_prev - bec_p["ramp_down_mwph"])
            bec_ramp_upper = min(bec_max, bec_prev + bec_p["ramp_up_mwph"])
            bec_dispatch   = float(np.clip(bec_target, bec_ramp_lower, bec_ramp_upper))
            bec_dispatch   = max(0.0, min(bec_dispatch, bec_max))

        sim_df.iat[i, col_idx["bec_displaced_flag"]] = 1.0 if bec_displace else 0.0
        bec_own_use   = min(bec_own_use_mw, bec_dispatch)
        bec_avail     = max(0.0, bec_dispatch - bec_own_use)
        bec_to_cust   = min(bec_avail, demand_t)
        bec_curtailed = max(0.0, bec_avail - bec_to_cust)
        sim_df.iat[i, col_idx["WWBEC_sim_tank_mw"]] = bec_dispatch
        sim_df.iat[i, col_idx["WWBEC_on_tank"]]     = int(bec_dispatch > 1e-9)
        prev_dispatch["WWBEC"] = bec_dispatch
        demand_after_bec = max(0.0, demand_t - bec_to_cust)

        # ── BEC net-revenue correction (v2) ──────────────────────────────────
        # Baseline = measured historical BEC production in this hour (WW_BEC).
        # The correction is only applied in hours in which the historical BEC
        # is above the operational threshold (bec_hist_active) -- hours with
        # near-zero historical values (standstill) do not count as "extra".
        # Extra BEC production above the baseline yields a net revenue
        # (electricity + subsidy, minus biomass costs) instead of a
        # cost item (v1). The revenue is entered as a negative "correction cost"
        # stored, so that the existing summation formula
        # (total_..._corrected = total_..._eur + bec_correction_cost_eur)
        # keeps working unchanged but now lowers the costs instead of raising them.
        bec_hist_baseline_mw = row["WW_BEC"] if BEC_BASELINE_CORRECTION else 0.0
        bec_hist_active_now  = bool(row["bec_hist_active"])
        if BEC_BASELINE_CORRECTION and bec_hist_active_now:
            bec_extra_mw = max(0.0, bec_dispatch - bec_hist_baseline_mw)
        else:
            bec_extra_mw = 0.0
        bec_extra_production_mw = bec_extra_mw
        bec_extra_gj      = bec_extra_mw * 3.6
        bec_extra_revenue = bec_extra_gj * bec_net_revenue_per_gj(price_now)
        bec_correction_cost_eur = -bec_extra_revenue
        sim_df.iat[i, col_idx["bec_hist_baseline_mw"]]    = bec_hist_baseline_mw
        sim_df.iat[i, col_idx["bec_extra_production_mw"]] = bec_extra_production_mw
        sim_df.iat[i, col_idx["bec_extra_revenue_eur"]]   = bec_extra_revenue
        sim_df.iat[i, col_idx["bec_correction_cost_eur"]] = bec_correction_cost_eur

        # ── Small tank discharge ──────────────────────────────────────────────
        small_avail_disch  = max(0.0, small_volume - small_min_vol)
        small_max_disch    = min(small_discharge_max_mw, small_avail_disch * CONST * delta_t)
        small_planned_disch = min(demand_after_bec, small_max_disch) if small_mode == "discharge" else 0.0
        demand_after_small  = max(0.0, demand_after_bec - small_planned_disch)

        # ── ATES discharge feasibility ────────────────────────────────────────
        # ates_discharge_rate_mult stays 1.0 for S0/S1/S3 (no-op multiply);
        # only S2/S4 ever set it below 1.0 (see strategy block above).
        ates_disch_avail   = max(0.0, ates_state.V_stored - ates_soc_min * ates_capacity_m3)
        ates_max_disch     = min(ates_discharge_max * ates_discharge_rate_mult, ates_disch_avail * CONST * dT_inj_t)
        ates_planned_disch = min(demand_after_small, ates_max_disch) if ates_mode == "discharge" else 0.0
        demand_remaining   = max(0.0, demand_after_small - ates_planned_disch)

        # Update annual discharge volume tracker
        discharge_vol_m3 = ates_planned_disch / (CONST * dT_inj_t) if ates_planned_disch > 0 else 0.0
        annual_discharge_m3 += discharge_vol_m3

        # ── Small tank charge headroom ────────────────────────────────────────
        small_avail_chg   = max(0.0, small_volume_max - small_volume)
        small_chg_room    = small_avail_chg * CONST * delta_t
        small_soc_room    = max(0.0, (small_soc_max * small_volume_max - small_volume) * CONST * delta_t)
        small_feas_chg    = min(small_charge_max_mw, small_chg_room, small_soc_room) if small_mode == "charge" else 0.0
        if aeb is not None and small_mode == "charge":
            small_feas_chg = aeb.feasible_charge(i, small_volume, delta_t, small_charge_max_mw)

        # ── ATES charge headroom ──────────────────────────────────────────────
        # ates_charge_rate_mult stays 1.0 for S0/S1/S3 (no-op multiply);
        # only S2/S4 ever set it below 1.0 (see strategy block above).
        ates_chg_vol_room = max(0.0, ates_soc_max * ates_capacity_m3 - ates_state.V_stored)
        ates_chg_room     = ates_chg_vol_room * CONST * dT_inj_t
        ates_feas_chg     = min(ates_charge_max * ates_charge_rate_mult, ates_chg_room) if ates_mode == "charge" else 0.0

        non_bec_target = demand_remaining + small_feas_chg

        # ── Minimum runtime: update the consecutive on/off counters ───────────
        # Uses the committed dispatch of the PREVIOUS time step (prev_dispatch,
        # not yet overwritten in this iteration) -- identical to simulation_v7.py.
        if enable_min_runtime:
            for hx_name in MIN_RUNTIME_HX:
                prev_was_on = prev_dispatch[hx_name] > 1e-9
                if prev_was_on:
                    hx_consecutive_on[hx_name] += 1
                    hx_consecutive_off[hx_name] = 0
                else:
                    hx_consecutive_off[hx_name] += 1
                    hx_consecutive_on[hx_name] = 0
                forced_on_now = 0 < hx_consecutive_on[hx_name] < min_runtime_hours
                forced_off_now = 0 < hx_consecutive_off[hx_name] < min_downtime_hours
                sim_df.iat[i, col_idx[f"{hx_name}_minruntime_forced_on_flag"]] = 1.0 if forced_on_now else 0.0
                sim_df.iat[i, col_idx[f"{hx_name}_minruntime_forced_off_flag"]] = 1.0 if forced_off_now else 0.0

        # ── Non-BEC demand-following dispatch ─────────────────────────────────
        dispatched_non_bec = 0.0
        for stage in non_bec_dispatch_sequence:
            if stage == "WW30_aftap":
                hx = "WW30"; ww30_stage = "aftap"
            elif stage == "WW30_reductie":
                hx = "WW30"; ww30_stage = "reductie"
            else:
                hx = stage; ww30_stage = None

            p          = hx_params[hx]
            min_cap    = row[f"{hx}_min_mw"]
            max_cap    = row[f"{hx}_max_mw"]
            prev       = prev_dispatch[hx]
            hist_avail = row[f"{hx}_hist_available"] == 1

            if hx == "WW30":
                ww30_aftap_cap = row["WW30_aftap_cap_mw"]
                stage_min_cap  = 0.0
                stage_max_cap  = (
                    min(max_cap, ww30_aftap_cap) if ww30_stage == "aftap"
                    else max(0.0, max_cap - ww30_aftap_cap)
                )
            else:
                stage_min_cap = min_cap
                stage_max_cap = max_cap

            # ── Minimum runtime constraint (WW10/WW30/WW01 only) ─────────────
            if enable_min_runtime and hx in MIN_RUNTIME_HX:
                if 0 < hx_consecutive_off[hx] < min_downtime_hours:
                    stage_max_cap = 0.0
                elif 0 < hx_consecutive_on[hx] < min_runtime_hours and ww30_stage != "reductie":
                    stage_min_cap = max(stage_min_cap, min_cap)

            useful_remaining = max(0.0, non_bec_target - dispatched_non_bec)
            allow_on = hist_avail and stage_max_cap > 1e-9

            if not allow_on or useful_remaining <= 1e-9:
                desired_stage = 0.0
            else:
                desired_stage = min(stage_max_cap, useful_remaining)
                if 0 < desired_stage < stage_min_cap:
                    desired_stage = stage_min_cap

            already      = sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
            ramp_lo      = max(0.0, prev - p["ramp_down_mwph"])
            ramp_hi      = min(max_cap, prev + p["ramp_up_mwph"])
            desired_total = float(np.clip(already + desired_stage, ramp_lo, ramp_hi))
            dispatch_stage = max(0.0, desired_total - already)
            dispatch_stage = min(dispatch_stage, max(0.0, stage_max_cap - already))
            forced = max(0.0, ramp_lo - already)
            dispatch_stage = min(dispatch_stage, max(useful_remaining, forced))

            new_total = max(0.0, min(already + dispatch_stage, max_cap))
            sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]] = new_total
            sim_df.iat[i, col_idx[f"{hx}_on_tank"]]     = int(new_total > 1e-9)
            dispatched_non_bec += dispatch_stage
            prev_dispatch[hx]  = new_total

        # ── Pure-charge ATES using spare HX capacity ──────────────────────────
        pure_charge_added = 0.0
        pure_ceil = min(ates_soc_max, ates_target + 0.10)
        if enable_pure_charge_ates and ates_mode == "charge" and ates_feas_chg > 0 and ates_soc < pure_ceil:
            pure_room   = max(0.0, (pure_ceil * ates_capacity_m3 - ates_state.V_stored) * CONST * dT_inj_t)
            charge_room = min(ates_feas_chg, pure_room)
            for hx in pure_charge_order:
                if charge_room <= 1e-9:
                    break
                p          = hx_params[hx]
                min_cap    = row[f"{hx}_min_mw"]
                max_cap    = row[f"{hx}_max_mw"]
                prev       = prev_dispatch[hx]
                hist_avail = row[f"{hx}_hist_available"] == 1
                if not hist_avail:
                    continue
                already  = sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
                ramp_hi  = min(max_cap, prev + p["ramp_up_mwph"])
                add_room = min(max(0.0, ramp_hi - already), max_cap - already)
                add_now  = min(add_room, charge_room)
                if already < 1e-9 and add_now > 0 and add_now < min_cap:
                    add_now = min_cap if (min_cap <= add_room and min_cap <= charge_room) else 0.0
                if add_now <= 1e-9:
                    continue
                new_total = already + add_now
                sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]] = new_total
                sim_df.iat[i, col_idx[f"{hx}_on_tank"]]     = int(new_total > 1e-9)
                prev_dispatch[hx] = new_total
                pure_charge_added += add_now
                charge_room       -= add_now

        sim_df.iat[i, col_idx["pure_charge_ates_mw"]] = pure_charge_added

        non_bec_gen     = sum(
            sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
            for hx in priority_order if hx != "WWBEC"
        )
        non_bec_to_cust = min(non_bec_gen, demand_remaining)
        surplus         = max(0.0, non_bec_gen - non_bec_to_cust)

        small_chg         = min(surplus, small_feas_chg)
        if aeb is not None:
            aeb.record(small_chg)
        surplus_after_sml = surplus - small_chg
        ates_chg_actual   = min(surplus_after_sml, ates_feas_chg)
        leftover_curtail  = max(0.0, surplus_after_sml - ates_chg_actual)

        # ── Update small tank volume ──────────────────────────────────────────
        small_volume = min(small_volume + small_chg / (CONST * delta_t), small_volume_max)
        if buffer_release_rule is None:
            small_volume = max(small_volume - small_planned_disch / (CONST * delta_t), small_min_vol)
        else:   # a floor above the content only blocks discharge; it does not raise the content
            small_volume = max(small_volume - small_planned_disch / (CONST * delta_t), min(small_min_vol, small_volume))

        # ── Advance ATES thermal state ────────────────────────────────────────
        T_out, hp_el, useful_disch, ground_loss = ates_state.step(ates_chg_actual, ates_planned_disch)
        T_gap = max(0.0, config.network_supply_temp_c - T_out)

        mech_pump   = (ates_chg_actual + ates_planned_disch) * ates_mech_pump_mwh_per_mwh_th
        pump_cost   = (hp_el + mech_pump) * price_now

        # ── Energy accounting ─────────────────────────────────────────────────
        total_supply = bec_to_cust + non_bec_to_cust + small_planned_disch + useful_disch
        unmet        = max(0.0, demand_t - total_supply)
        curtailed    = leftover_curtail + bec_curtailed

        # ── Write results ─────────────────────────────────────────────────────
        sim_df.iat[i, col_idx["small_tank_volume_m3"]]      = small_volume
        sim_df.iat[i, col_idx["small_tank_charge_mw"]]      = small_chg
        sim_df.iat[i, col_idx["small_tank_discharge_mw"]]   = small_planned_disch
        sim_df.iat[i, col_idx["ates_volume_m3"]]            = ates_state.V_stored
        sim_df.iat[i, col_idx["ates_soc"]]                  = ates_state.V_stored / ates_capacity_m3 if ates_capacity_m3 > 0 else 0.0
        sim_df.iat[i, col_idx["ates_charge_mw"]]            = ates_chg_actual
        sim_df.iat[i, col_idx["ates_discharge_mw"]]         = ates_planned_disch
        sim_df.iat[i, col_idx["ates_aquifer_temp_c"]]       = ates_state.T_aq
        sim_df.iat[i, col_idx["ates_outlet_temp_c"]]        = T_out
        sim_df.iat[i, col_idx["ates_temp_gap_c"]]           = T_gap
        sim_df.iat[i, col_idx["ates_heat_pump_mwel"]]       = hp_el
        sim_df.iat[i, col_idx["ates_useful_discharge_mw"]]  = useful_disch
        sim_df.iat[i, col_idx["ates_ground_loss_mw"]]       = ground_loss
        sim_df.iat[i, col_idx["ates_mech_pump_mwh"]]        = mech_pump
        sim_df.iat[i, col_idx["ates_pump_cost_eur"]]        = pump_cost
        sim_df.iat[i, col_idx["tank_charge_mw"]]            = small_chg + ates_chg_actual
        sim_df.iat[i, col_idx["tank_discharge_mw"]]         = small_planned_disch + ates_planned_disch
        sim_df.iat[i, col_idx["direct_from_hx_mw"]]         = bec_to_cust + non_bec_to_cust
        sim_df.iat[i, col_idx["direct_from_tank_mw"]]       = small_planned_disch + useful_disch

        total_gen        = bec_dispatch + non_bec_gen
        total_gen_usable = bec_avail + non_bec_gen
        sim_df.iat[i, col_idx["total_generation_tank_model_mw"]] = total_gen
        sim_df.iat[i, col_idx["BEC_own_use_mw"]]                 = bec_own_use
        sim_df.iat[i, col_idx["total_generation_usable_mw"]]     = total_gen_usable
        sim_df.iat[i, col_idx["total_supply_to_customer_mw"]]    = total_supply
        sim_df.iat[i, col_idx["unmet_demand_tank_mw"]]           = unmet
        sim_df.iat[i, col_idx["curtailed_or_excess_mw"]]         = curtailed

        # ── Derving cost (electricity opportunity cost, BEC excluded) ─────────
        total_elec_loss = 0.0
        for hx in priority_order:
            heat_out = sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
            if hx == "WW30":
                ww30_aftap_cap = row["WW30_aftap_cap_mw"]
                aftap_p        = min(heat_out, ww30_aftap_cap)
                reduct_p       = max(0.0, heat_out - aftap_p)
                elec_loss      = (
                    aftap_p  * derivation_factors["WW30_aftap"]
                    + reduct_p * derivation_factors["WW30_reductie"]
                )
            else:
                elec_loss = heat_out * derivation_factors[hx]
            sim_df.iat[i, col_idx[f"{hx}_electricity_loss_mw"]] = elec_loss
            if hx != "WWBEC":
                total_elec_loss += elec_loss

        derving_cost      = total_elec_loss * price_now
        total_system_cost = derving_cost + pump_cost
        total_system_cost_corrected = total_system_cost + bec_correction_cost_eur

        sim_df.iat[i, col_idx["total_electricity_loss_mw"]]       = total_elec_loss
        sim_df.iat[i, col_idx["derving_cost_eur"]]                = derving_cost
        sim_df.iat[i, col_idx["total_system_cost_eur"]]           = total_system_cost
        sim_df.iat[i, col_idx["total_system_cost_eur_corrected"]] = total_system_cost_corrected
        sim_df.iat[i, col_idx["ates_dT_inj_t"]]            = dT_inj_t

    return sim_df


def print_hx_cycling_stats(sim_df, priority_order, label=""):
    # NB: hours_on/hours_off/starts are divided by a hardcoded 3 (years),
    # matching the YEARS=3.0 convention used throughout the post-midterm
    # run scripts for the 2023-01-01..2025-12-31 simulation window -- not
    # a general-purpose per-year normalisation for other date ranges.
    print(f"\nWARMTEWISSELAAR ON/OFF CYCLI — {label}")
    print("─" * 70)
    print(f"{'HX':<12} {'Uren aan/jr':>12} {'Uren uit/jr':>12} "
          f"{'Starts/jr':>10} {'Gem. run (u)':>13} {'Gem. stop (u)':>13}")
    print("─" * 70)

    for hx in priority_order:
        col = f"{hx}_on_tank" if f"{hx}_on_tank" in sim_df.columns \
              else f"{hx}_on"
        if col not in sim_df.columns:
            continue

        on_series = sim_df[col].fillna(0).astype(int)
        hours_on  = on_series.sum() / 3        # per year
        hours_off = (1 - on_series).sum() / 3

        # Count starts = transition from 0 to 1
        starts = ((on_series.diff() == 1).sum()) / 3

        mean_run  = hours_on  / starts if starts > 0 else 0
        mean_stop = hours_off / starts if starts > 0 else 0

        print(f"{hx:<12} {hours_on:>12.0f} {hours_off:>12.0f} "
              f"{starts:>10.1f} {mean_run:>13.1f} {mean_stop:>13.1f}")
    print("─" * 70)
