"""Dispatch model of Option B (one ATES per loop), also used for B-WN 110 C and the zero-capacity reference run. Needs
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

# ── POST-MIDTERM BEC REVENUE FORMULA (version 2) ─────────────
# Source: supervisor analysis BEC_extra_warmte_zomer_analyse_v2
# Correction: BEC yields a net revenue per GJ of heat,
# not a cost item. Extra BEC production above the historical
# baseline is counted as additional system revenue.
#
# Ratios (from the supervisor's presentation):
# Electricity per GJ heat:  1.09 / 3.93 = 0.2774
# Biomass per GJ heat:       5.69 / 3.93 = 1.4478
# Subsidisable per GJ of heat:    heat + electricity = 1.2774
#
# Fully replaces the v1 cost correction (fixed marginal cost factor
# of 33.8 EUR/MWh) — that formula and BEC_MARGINAL_COST_EUR_PER_MWH
# no longer exist here.
#
# Fixed prices:
BEC_BIOMASSA_PRIJS_EUR_PER_GJ = 5.69    # average 2025
BEC_SUBSIDIE_PRIJS_EUR_PER_GJ = 9.17    # subsidy per GJ
BEC_ELEK_PER_WARMTE = 1.09 / 3.93      # GJ electricity per GJ heat
BEC_BIOMASSA_PER_WARMTE = 5.69 / 3.93  # GJ biomass per GJ heat
BEC_SUBSIDIE_PER_WARMTE = 1 + BEC_ELEK_PER_WARMTE  # GJ eligible for subsidy per GJ heat
BEC_MIN_OPERATIONAL_MW = 1.0            # MW threshold for standstill
BEC_BASELINE_CORRECTION = True


def bec_net_revenue_per_gj(epex_eur_per_mwh):
    # Convert EPEX from €/MWh to €/GJ (1 MWh = 3.6 GJ)
    epex_gj = epex_eur_per_mwh / 3.6

    biomassa_bedrag = -BEC_BIOMASSA_PER_WARMTE * BEC_BIOMASSA_PRIJS_EUR_PER_GJ
    elek_bedrag     =  BEC_ELEK_PER_WARMTE     * epex_gj
    subsidie_bedrag =  BEC_SUBSIDIE_PER_WARMTE * BEC_SUBSIDIE_PRIJS_EUR_PER_GJ

    return biomassa_bedrag + elek_bedrag + subsidie_bedrag  # €/GJ heat

# ── WW standby-minimum correction ────────────────────────────
# If a heat exchanger was historically operational at
# time step t (production >= threshold), the simulation must keep it
# at least at its standby minimum at that same time step.
# Reason: in reality heat exchangers that are in
# operation may not simply be switched off completely —
# they must stay at least at their technical minimum load.
#
# Threshold values to be submitted to the supervisor for verification.
#
# Note: hx_params['WW30'] and hx_params['WW01'] are of the
# 'dynamic'/'special' type and have — unlike WWBEC ('fixed')
# — no static 'min_mw' key in config.py. Their technical
# minimum depends on delta_t_c and is already computed per hour
# by data_pipeline.py in the columns WW30_min_mw and
# WW01_min_mw. Those hourly column values are used below
# instead of a fixed module constant.

WW_STANDBY_MIN_OPERATIONAL_THRESHOLD_MW = 1.0  # MW — below this value = switched off
WW30_STANDBY_MIN_MW = None   # hourly: row["WW30_min_mw"] (computed by the pipeline, no static config key)
WW01_STANDBY_MIN_MW = None   # hourly: row["WW01_min_mw"] (computed by the pipeline, no static config key)
WW30_HIST_COL = "WW_30"      # historical production column in sim_df_tank
WW01_HIST_COL = "WW_01"      # historical production column in sim_df_tank (WW01 exists — no WW10 fallback needed)

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


def _loop_ates_mode(
    soc,
    soc_target,
    soc_min,
    soc_max,
    price_now,
    window_prices,
    annual_charge_pct,
    annual_discharge_pct,
):
    """
    Determine charge/discharge/drift for one ATES loop.

    SOC-excess shifts thresholds: a loop with more stored energy than the
    seasonal target discharges more aggressively and charges less eagerly.
    Identical to v3/v5 — provides SOC-excess adjusted baseline signal that
    the v6 graduated constraint block then gates.
    """
    soc_excess = soc - soc_target

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

    return "discharge" if want_discharge else ("charge" if want_charge else "drift")


def run_ates_three_loops_v6(
    sim_input_df,
    config_westpoort,
    config_noord,
    config_nieuwwest,
    col_demand_westpoort='demand_westpoort_mw',
    col_demand_noord='demand_noord_mw',
    col_demand_nieuwwest='demand_nieuwwest_mw',
    annual_charge_pct=25,
    annual_discharge_pct=75,
    daily_charge_rank_n=6,
    daily_discharge_rank_n=6,
    ates_soc_max=0.95,
    ates_soc_min=0.05,
    ates_initial_soc=0.0,
    price_col='Prijs_Gemiddelde_EPEX_Spot',
    perfect_foresight=False,
    # WW standby correction disabled for ATES model:
    # When ATES is present, WW30/WW01 can legitimately be switched
    # off because stored heat covers demand. Applying standby
    # correction here would penalise the ATES for doing exactly
    # what it is designed to do. Standby correction only applies
    # to the dispatch-only baseline model (simulation_v7.py).
    enable_ww_standby_correction=False,
    # Minimum runtime constraint -- see MIN_RUNTIME_HX above.
    enable_min_runtime: bool = False,
    min_runtime_hours: int = 18,   # at least 18 hours on once started
    min_downtime_hours: int = 6,   # at least 6 hours off after switching off
    # BEC summer cap -- see module-level BEC_SUMMER_CAP_FRACTION above.
    enable_bec_summer_cap: bool = False,
    # Rule for the existing buffers: "rank" (default) or "aeb_demand" (aeb_buffer_rule.py).
    existing_buffer_rule: str = "rank",
    # Existing buffers: ceiling and reserve (defaults = the previously hard-coded values).
    small_soc_max: float = 0.95,
    reserve_horizon_h: int = 6,
    reserve_max_fraction: float = 0.20,
    # Conditional release of the existing buffers (default None = off), as in simulation_v7.py.
    # "hourly": buffer floor = release reserve in hours with ATES backup: H = sum over the loops of
    # (V_lus - ates_soc_min x cap_lus) x CONST x dT_lus >= R (MWh) and, if release_power_horizon_h
    # is set, also (sum of the discharge limits of the loops) x release_power_horizon_h >= R.
    buffer_release_rule=None,
    release_reserve_horizon_h: int = 6,
    release_reserve_max_fraction: float = 0.20,
    release_power_horizon_h=None,
    # Heat demand correction factor -- same as simulation_v7.py.
    demand_correction_factor: float = 1.0,
):
    """
    Three-loop ATES dispatch v6: negative-price rule + November reserve.

    Extends v5 with two additions:
      1. Negative prices (<0 EUR/MWh) trigger unconditional charging and
         block all discharge for that timestep.
      2. November SOC floor: early Nov 25%, late Nov 35% — prevents the
         over-discharge that left V5 with December SOC = 0.227.
    """
    sim_df = sim_input_df.copy()

    # ── Initialise three ATES thermal states ─────────────────────────────────
    ates_wp = ATESThermalState(config_westpoort)
    ates_n  = ATESThermalState(config_noord)
    ates_nw = ATESThermalState(config_nieuwwest)

    cap_wp = config_westpoort.ates_capacity_m3
    cap_n  = config_noord.ates_capacity_m3
    cap_nw = config_nieuwwest.ates_capacity_m3
    total_capacity = cap_wp + cap_n + cap_nw

    ates_wp.V_stored = cap_wp * ates_initial_soc
    ates_n.V_stored  = cap_n  * ates_initial_soc
    ates_nw.V_stored = cap_nw * ates_initial_soc

    dT_wp = max(1.0, config_westpoort.ates_injection_temp_c - config_westpoort.ates_cold_temp_c)
    dT_n  = max(1.0, config_noord.ates_injection_temp_c    - config_noord.ates_cold_temp_c)
    dT_nw = max(1.0, config_nieuwwest.ates_injection_temp_c - config_nieuwwest.ates_cold_temp_c)

    max_disch_wp = config_westpoort.ates_discharge_max_mw
    max_disch_n  = config_noord.ates_discharge_max_mw
    max_disch_nw = config_nieuwwest.ates_discharge_max_mw

    max_charge_wp = config_westpoort.ates_charge_max_mw
    max_charge_n  = config_noord.ates_charge_max_mw
    max_charge_nw = config_nieuwwest.ates_charge_max_mw

    # ── Small buffer tank geometry (AEB plant) ────────────────────────────────
    n_small_tanks       = 3
    small_tank_vol      = 1200.0
    small_volume_max    = n_small_tanks * small_tank_vol
    small_volume        = small_volume_max * 0.5
    small_soc_max_val   = small_soc_max
    small_soc_min_val   = 0.10
    small_charge_max    = tank_charge_max_mw
    small_discharge_max = tank_discharge_max_mw
    reserve_max_frac    = reserve_max_fraction
    mech_pump_factor    = 72.91 / 60_000

    MIN_PHYSICAL_DELTA_T = 10.0
    trailing_window_h    = 365 * 24

    # ── Graduated price thresholds ────────────────────────────────────────────
    # Computed once from full 3-year price series (not rolling).
    # p01 clamped to 0.0 so the P01 tier always covers zero-price hours.
    p01 = max(sim_df[price_col].quantile(0.01), 0.0)   # clamped: min zero-price boundary
    p10 = sim_df[price_col].quantile(0.10)              # very cheap: charge regardless of season
    p25 = sim_df[price_col].quantile(0.25)              # normal charge gate (Apr-Oct only)
    p75 = sim_df[price_col].quantile(0.75)              # normal discharge gate (SOC floor 30%)
    p85 = sim_df[price_col].quantile(0.85)              # moderately expensive (SOC floor 20%)
    p90 = sim_df[price_col].quantile(0.90)              # very expensive (SOC floor 10%)
    p99 = sim_df[price_col].quantile(0.99)              # extreme spike (SOC floor 5%)

    print(f"  Graduated price thresholds (EUR/MWh):")
    print(f"    P01={p01:.1f}  P10={p10:.1f}  P25={p25:.1f}")
    print(f"    P75={p75:.1f}  P85={p85:.1f}  P90={p90:.1f}  P99={p99:.1f}")
    print(f"    Negative-price rule active below 0.0 EUR/MWh (unconditional charge)")

    # Annual discharge volume trackers — reset each January, cap at 3x capacity
    annual_discharge_m3_wp = 0.0
    annual_discharge_m3_no = 0.0
    annual_discharge_m3_nw = 0.0

    # ── Pre-compute shared price thresholds ───────────────────────────────────
    if perfect_foresight:
        sim_df["_charge_thresh"]    = sim_df[price_col].quantile(annual_charge_pct / 100)
        sim_df["_discharge_thresh"] = sim_df[price_col].quantile(annual_discharge_pct / 100)
        sim_df["_annual_50_thresh"] = sim_df[price_col].quantile(0.50)
    else:
        sim_df["_charge_thresh"] = (
            sim_df[price_col]
            .rolling(window=trailing_window_h, min_periods=24)
            .quantile(annual_charge_pct / 100)
        )
        sim_df["_discharge_thresh"] = (
            sim_df[price_col]
            .rolling(window=trailing_window_h, min_periods=24)
            .quantile(annual_discharge_pct / 100)
        )
        sim_df["_annual_50_thresh"] = (
            sim_df[price_col]
            .rolling(window=trailing_window_h, min_periods=24)
            .quantile(0.50)
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

    # Baseline only active if historical BEC >= threshold
    sim_df["bec_hist_active"] = sim_df["WW_BEC"] >= BEC_MIN_OPERATIONAL_MW

    # WW30/WW01 historically operational in this hour (standby-minimum correction)
    sim_df["ww30_hist_active"] = sim_df[WW30_HIST_COL] >= WW_STANDBY_MIN_OPERATIONAL_THRESHOLD_MW
    sim_df["ww01_hist_active"] = sim_df[WW01_HIST_COL] >= WW_STANDBY_MIN_OPERATIONAL_THRESHOLD_MW

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

    safe_dt = sim_df["delta_t_c"].clip(lower=MIN_PHYSICAL_DELTA_T)
    sim_df["required_volume_for_demand_m3"] = sim_df["heat_demand_mw"] / (CONST * safe_dt)
    reserve_raw = sum(
        sim_df["required_volume_for_demand_m3"].shift(-h).fillna(0.0)
        for h in range(1, reserve_horizon_h + 1)
    )
    reserve_cap_abs = reserve_max_frac * small_volume_max
    sim_df["small_min_volume_required_m3"] = np.clip(reserve_raw, 0.0, reserve_cap_abs)
    if buffer_release_rule is not None:
        if buffer_release_rule != "hourly":
            raise ValueError(f"buffer_release_rule must be None or 'hourly', got {buffer_release_rule!r}")
        rel_raw = sum(sim_df["required_volume_for_demand_m3"].shift(-h).fillna(0.0)
                      for h in range(1, release_reserve_horizon_h + 1))
        sim_df["release_min_volume_m3"] = np.clip(rel_raw, 0.0, release_reserve_max_fraction * small_volume_max)
        for c in ("buffer_release_flag", "backup_heat_ok", "backup_power_ok", "backup_heat_mwh", "backup_reserve_mwh"):
            sim_df[c] = 0.0

    has_wp = col_demand_westpoort in sim_df.columns
    has_n  = col_demand_noord     in sim_df.columns
    has_nw = col_demand_nieuwwest in sim_df.columns
    for flag, name in [(has_wp, col_demand_westpoort), (has_n, col_demand_noord), (has_nw, col_demand_nieuwwest)]:
        if not flag:
            print(f"[three_loops_v6 WARNING] column '{name}' not found -- using 0.0")

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
        "ww30_standby_forced_flag", "ww30_standby_extra_mw",
        "ww01_standby_forced_flag", "ww01_standby_extra_mw",
        "pure_charge_ates_mw", "small_mode_v2", "ates_mode_v2", "bec_displaced_flag",
        "effective_demand_mw",
        "ates_westpoort_soc", "ates_noord_soc", "ates_nieuwwest_soc",
        "ates_westpoort_discharge_mw", "ates_noord_discharge_mw", "ates_nieuwwest_discharge_mw",
        "ates_westpoort_charge_mw", "ates_noord_charge_mw", "ates_nieuwwest_charge_mw",
        "ates_westpoort_temp_c", "ates_noord_temp_c", "ates_nieuwwest_temp_c",
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

    cidx_wp = col_idx[col_demand_westpoort] if has_wp else None
    cidx_n  = col_idx[col_demand_noord]     if has_n  else None
    cidx_nw = col_idx[col_demand_nieuwwest] if has_nw else None

    # ── Main simulation loop ──────────────────────────────────────────────────
    for i in range(len(sim_df)):
        row      = sim_df.iloc[i]
        demand_t = row["heat_demand_mw"] * demand_correction_factor
        delta_t  = max(row["delta_t_c"], MIN_PHYSICAL_DELTA_T)
        price_now = row[price_col]

        demand_wp = sim_df.iat[i, cidx_wp] if has_wp else 0.0
        demand_n  = sim_df.iat[i, cidx_n]  if has_n  else 0.0
        demand_nw = sim_df.iat[i, cidx_nw] if has_nw else 0.0

        small_min_vol = row["small_min_volume_required_m3"]
        if buffer_release_rule is not None:
            heat_mwh = sum(max(0.0, st.V_stored - ates_soc_min * cp) * CONST * dt_
                           for st, cp, dt_ in ((ates_wp, cap_wp, dT_wp), (ates_n, cap_n, dT_n), (ates_nw, cap_nw, dT_nw)))
            res_mwh = row["release_min_volume_m3"] * CONST * delta_t
            heat_ok = heat_mwh >= res_mwh
            power_ok = (release_power_horizon_h is None
                        or (max_disch_wp + max_disch_n + max_disch_nw) * release_power_horizon_h >= res_mwh)
            if heat_ok and power_ok:
                small_min_vol = row["release_min_volume_m3"]
            for c, v in (("buffer_release_flag", float(heat_ok and power_ok)), ("backup_heat_ok", float(heat_ok)),
                         ("backup_power_ok", float(power_ok)), ("backup_heat_mwh", heat_mwh),
                         ("backup_reserve_mwh", res_mwh)):
                sim_df.iat[i, col_idx[c]] = v

        soc_wp = ates_wp.V_stored / cap_wp if cap_wp > 0 else 0.0
        soc_n  = ates_n.V_stored  / cap_n  if cap_n  > 0 else 0.0
        soc_nw = ates_nw.V_stored / cap_nw if cap_nw > 0 else 0.0

        total_vol = ates_wp.V_stored + ates_n.V_stored + ates_nw.V_stored
        soc_agg   = total_vol / total_capacity if total_capacity > 0 else 0.0

        soc_target = row["ates_soc_target_seasonal"]

        if perfect_foresight:
            window_prices = sim_df[price_col]
        else:
            win_start = max(0, i - trailing_window_h + 1)
            window_prices = sim_df[price_col].iloc[win_start:i + 1]

        # ── Small tank mode (daily tier-2 price-rank signal) ─────────────────
        small_soc          = small_volume / small_volume_max if small_volume_max > 0 else 0.0
        small_can_charge   = small_soc < small_soc_max_val
        small_can_disch    = small_soc > small_soc_min_val
        small_want_charge  = bool(row["small_charge_signal"])    and small_can_charge
        small_want_disch   = bool(row["small_discharge_signal"]) and small_can_disch
        if small_want_disch:
            small_want_charge = False
        small_mode = "charge" if small_want_charge else ("discharge" if small_want_disch else "drift")
        if aeb is not None:
            small_mode, small_min_vol = aeb.hour(i)

        # ── Per-loop ATES mode from price signal (pre-graduated-constraint) ───
        mode_wp = _loop_ates_mode(
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
        )

        # ── Convert to boolean wants for constraint logic ─────────────────────
        ates_want_discharge_wp = (mode_wp == "discharge")
        ates_want_charge_wp    = (mode_wp == "charge")
        ates_soc_wp            = soc_wp

        ates_want_discharge_no = (mode_n == "discharge")
        ates_want_charge_no    = (mode_n == "charge")
        ates_soc_no            = soc_n

        ates_want_discharge_nw = (mode_nw == "discharge")
        ates_want_charge_nw    = (mode_nw == "charge")
        ates_soc_nw            = soc_nw

        # ── Graduated dispatch logic (v6) ─────────────────────────────────────
        month = sim_df.index[i].month

        # ── CHARGING: tier-based eligibility ─────────────────────────────────
        if price_now <= p10:
            charge_eligible_wp = True
            charge_eligible_no = True
            charge_eligible_nw = True
        elif price_now <= p25:
            if month in [4, 5, 6, 7, 8, 9, 10]:
                charge_eligible_wp = True
                charge_eligible_no = True
                charge_eligible_nw = True
            else:
                charge_eligible_wp = False
                charge_eligible_no = False
                charge_eligible_nw = False
        else:
            charge_eligible_wp = False
            charge_eligible_no = False
            charge_eligible_nw = False

        # ── DISCHARGING: SOC floor varies by price tier ───────────────────────
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
        discharge_eligible_nw = ates_soc_nw > soc_floor

        # October SOC gate (retained from v3)
        if month == 10:
            if ates_soc_wp < 0.50:
                discharge_eligible_wp = False
            if ates_soc_no < 0.50:
                discharge_eligible_no = False
            if ates_soc_nw < 0.50:
                discharge_eligible_nw = False

        # Annual discharge volume cap (retained from v3)
        MAX_DISCHARGE_RATIO = 3.0
        if annual_discharge_m3_wp > MAX_DISCHARGE_RATIO * config_westpoort.ates_capacity_m3:
            discharge_eligible_wp = False
        if annual_discharge_m3_no > MAX_DISCHARGE_RATIO * config_noord.ates_capacity_m3:
            discharge_eligible_no = False
        if annual_discharge_m3_nw > MAX_DISCHARGE_RATIO * config_nieuwwest.ates_capacity_m3:
            discharge_eligible_nw = False

        # ── Unconditional negative-price override (v6) ────────────────────────
        # Applied last so it overrides all tier and reserve logic above.
        # Only the eligibility flags are changed; all downstream calculations
        # (BEC dispatch, small tank, charge/discharge allocation) run normally.
        if price_now < 0:
            charge_eligible_wp = True
            charge_eligible_no = True
            charge_eligible_nw = True
            discharge_eligible_wp = False
            discharge_eligible_no = False
            discharge_eligible_nw = False

        # Reset annual discharge trackers on 1 January each year (always runs)
        if sim_df.index[i].month == 1 and sim_df.index[i].day <= 7:
            if i == 0 or sim_df.index[i - 1].month == 12:
                annual_discharge_m3_wp = 0.0
                annual_discharge_m3_no = 0.0
                annual_discharge_m3_nw = 0.0

        # ── Apply eligibility to existing charge/discharge want signals ────────
        ates_want_charge_wp    = ates_want_charge_wp    and charge_eligible_wp
        ates_want_charge_no    = ates_want_charge_no    and charge_eligible_no
        ates_want_charge_nw    = ates_want_charge_nw    and charge_eligible_nw
        ates_want_discharge_wp = ates_want_discharge_wp and discharge_eligible_wp
        ates_want_discharge_no = ates_want_discharge_no and discharge_eligible_no
        ates_want_discharge_nw = ates_want_discharge_nw and discharge_eligible_nw

        # ── Reconstruct mode strings from constrained booleans ────────────────
        mode_wp = "discharge" if ates_want_discharge_wp else ("charge" if ates_want_charge_wp else "drift")
        mode_n  = "discharge" if ates_want_discharge_no else ("charge" if ates_want_charge_no else "drift")
        mode_nw = "discharge" if ates_want_discharge_nw else ("charge" if ates_want_charge_nw else "drift")

        # Aggregate mode for reporting
        modes = [mode_wp, mode_n, mode_nw]
        if modes.count("discharge") >= 2:
            ates_mode_agg = "discharge"
        elif modes.count("charge") >= 2:
            ates_mode_agg = "charge"
        else:
            ates_mode_agg = "drift"

        sim_df.iat[i, col_idx["small_mode_v2"]] = (
            1.0 if small_mode == "charge" else (-1.0 if small_mode == "discharge" else 0.0)
        )
        sim_df.iat[i, col_idx["ates_mode_v2"]] = (
            1.0 if ates_mode_agg == "charge" else (-1.0 if ates_mode_agg == "discharge" else 0.0)
        )

        # ── Per-loop planned discharges (each against its own loop demand) ────
        def _planned_disch(state, capacity, dT_nom, max_rate, loop_demand, mode):
            if mode != "discharge":
                return 0.0
            vol_avail = max(0.0, state.V_stored - ates_soc_min * capacity)
            max_from_vol = min(max_rate, vol_avail * CONST * dT_nom)
            return min(loop_demand, max_from_vol)

        disch_wp = _planned_disch(ates_wp, cap_wp, dT_wp, max_disch_wp, demand_wp, mode_wp)
        disch_n  = _planned_disch(ates_n,  cap_n,  dT_n,  max_disch_n,  demand_n,  mode_n)
        disch_nw = _planned_disch(ates_nw, cap_nw, dT_nw, max_disch_nw, demand_nw, mode_nw)

        # ── Update annual discharge volume trackers ────────────────────────────
        discharge_vol_westpoort_m3 = disch_wp / (CONST * dT_wp) if disch_wp > 0 else 0.0
        discharge_vol_noord_m3     = disch_n  / (CONST * dT_n)  if disch_n  > 0 else 0.0
        discharge_vol_nieuwwest_m3 = disch_nw / (CONST * dT_nw) if disch_nw > 0 else 0.0
        annual_discharge_m3_wp += discharge_vol_westpoort_m3
        annual_discharge_m3_no += discharge_vol_noord_m3
        annual_discharge_m3_nw += discharge_vol_nieuwwest_m3

        total_ates_disch = disch_wp + disch_n + disch_nw

        effective_demand_t = max(0.0, demand_t - total_ates_disch)

        # ── Reset per-HX dispatch ─────────────────────────────────────────────
        for hx in priority_order:
            sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]] = 0.0
            sim_df.iat[i, col_idx[f"{hx}_on_tank"]]     = 0

        # ── BEC dispatch against effective demand ─────────────────────────────
        bec_p              = hx_params["WWBEC"]
        bec_min_cap        = row["WWBEC_min_mw"]
        bec_max_cap        = row["WWBEC_max_mw_effective"]
        bec_prev           = prev_dispatch["WWBEC"]
        bec_hist_available = row["WWBEC_hist_available"] == 1

        annual_50_now = row["ates_annual_50_thresh"]
        if pd.isna(annual_50_now):
            annual_50_now = window_prices.quantile(0.50)

        bec_displace_active = (
            total_capacity > 0
            and soc_agg > 0.85
            and price_now >= annual_50_now
            and soc_agg > ates_soc_min
            and bec_hist_available
        )

        if not bec_hist_available:
            bec_dispatch = 0.0
        else:
            bec_target     = bec_min_cap if bec_displace_active else max(
                bec_min_cap, min(effective_demand_t + bec_own_use_mw, bec_max_cap)
            )
            bec_ramp_lower = max(0.0, bec_prev - bec_p["ramp_down_mwph"])
            bec_ramp_upper = min(bec_max_cap, bec_prev + bec_p["ramp_up_mwph"])
            bec_dispatch   = float(np.clip(bec_target, bec_ramp_lower, bec_ramp_upper))
            bec_dispatch   = max(0.0, min(bec_dispatch, bec_max_cap))

        sim_df.iat[i, col_idx["bec_displaced_flag"]] = 1.0 if bec_displace_active else 0.0

        bec_own_use              = min(bec_own_use_mw, bec_dispatch)
        bec_available_for_cust   = max(0.0, bec_dispatch - bec_own_use)
        bec_to_customer          = min(bec_available_for_cust, effective_demand_t)
        bec_curtailed            = max(0.0, bec_available_for_cust - bec_to_customer)
        sim_df.iat[i, col_idx["WWBEC_sim_tank_mw"]] = bec_dispatch
        sim_df.iat[i, col_idx["WWBEC_on_tank"]]     = int(bec_dispatch > 1e-9)
        prev_dispatch["WWBEC"]   = bec_dispatch
        demand_after_bec         = max(0.0, effective_demand_t - bec_to_customer)

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
        small_avail_disch_vol    = max(0.0, small_volume - small_min_vol)
        small_max_disch_mw       = min(small_discharge_max, small_avail_disch_vol * CONST * delta_t)
        small_planned_disch      = min(demand_after_bec, small_max_disch_mw) if small_mode == "discharge" else 0.0
        demand_after_small       = max(0.0, demand_after_bec - small_planned_disch)

        # ── Small tank charge headroom ────────────────────────────────────────
        small_avail_charge_vol   = max(0.0, small_volume_max - small_volume)
        small_charge_room_mw     = small_avail_charge_vol * CONST * delta_t
        small_soc_room_mw        = max(0.0, (small_soc_max_val * small_volume_max - small_volume) * CONST * delta_t)
        small_feasible_charge    = min(small_charge_max, small_charge_room_mw, small_soc_room_mw) if small_mode == "charge" else 0.0
        if aeb is not None and small_mode == "charge":
            small_feasible_charge = aeb.feasible_charge(i, small_volume, delta_t, small_charge_max)

        demand_remaining = demand_after_small
        non_bec_target   = demand_remaining + small_feasible_charge

        # ── Non-BEC demand-following dispatch ─────────────────────────────────
        # Captured before the stage loop -- these are the previous timestep's
        # committed dispatch values, needed for the standby-minimum ramp check
        # below (the loop itself overwrites prev_dispatch["WW30"]/["WW01"]).
        prev_dispatch_before_ww30 = prev_dispatch["WW30"]
        prev_dispatch_before_ww01 = prev_dispatch["WW01"]

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

        dispatched_non_bec = 0.0
        for stage in non_bec_dispatch_sequence:
            if stage == "WW30_aftap":
                hx = "WW30"; ww30_stage = "aftap"
            elif stage == "WW30_reductie":
                hx = "WW30"; ww30_stage = "reductie"
            else:
                hx = stage; ww30_stage = None

            p              = hx_params[hx]
            min_cap        = row[f"{hx}_min_mw"]
            max_cap        = row[f"{hx}_max_mw"]
            prev           = prev_dispatch[hx]
            hist_available = row[f"{hx}_hist_available"] == 1

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
            allow_on         = hist_available and stage_max_cap > 1e-9

            if not allow_on or useful_remaining <= 1e-9:
                desired_stage_dispatch = 0.0
            else:
                desired_stage_dispatch = min(stage_max_cap, useful_remaining)
                if 0 < desired_stage_dispatch < stage_min_cap:
                    desired_stage_dispatch = stage_min_cap

            already_for_hx   = sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
            ramp_lower_total = max(0.0, prev - p["ramp_down_mwph"])
            ramp_upper_total = min(max_cap, prev + p["ramp_up_mwph"])
            desired_total    = float(np.clip(
                already_for_hx + desired_stage_dispatch, ramp_lower_total, ramp_upper_total
            ))
            dispatch_stage   = max(0.0, desired_total - already_for_hx)
            dispatch_stage   = min(dispatch_stage, max(0.0, stage_max_cap - already_for_hx))
            forced_due_to_ramp = max(0.0, ramp_lower_total - already_for_hx)
            dispatch_stage   = min(dispatch_stage, max(useful_remaining, forced_due_to_ramp))

            new_total_for_hx = max(0.0, min(already_for_hx + dispatch_stage, max_cap))
            sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]] = new_total_for_hx
            sim_df.iat[i, col_idx[f"{hx}_on_tank"]]     = int(new_total_for_hx > 1e-9)
            dispatched_non_bec += dispatch_stage
            prev_dispatch[hx]   = new_total_for_hx

        # ── WW standby-minimum correction (WW30) ───────────────────────────────
        # If WW30 was historically operational in this hour (production >= threshold),
        # the dispatch logic may not have switched it off completely. Force
        # at least the technical minimum (or the ramp lower bound, if higher).
        ww30_hist_active = row["ww30_hist_active"]
        if ww30_hist_active and enable_ww_standby_correction:
            ww30_min_cap       = row["WW30_min_mw"]
            ww30_max_cap       = row["WW30_max_mw"]
            ramp_lower_ww30    = max(0.0, prev_dispatch_before_ww30 - hx_params["WW30"]["ramp_down_mwph"])
            ww30_min_this_timestep = max(ww30_min_cap, ramp_lower_ww30)
            ww30_dispatch      = sim_df.iat[i, col_idx["WW30_sim_tank_mw"]]
            ww30_forced        = min(max(ww30_dispatch, ww30_min_this_timestep), ww30_max_cap)
            if ww30_forced > ww30_dispatch + 1e-9:
                sim_df.iat[i, col_idx["ww30_standby_forced_flag"]] = 1.0
                sim_df.iat[i, col_idx["ww30_standby_extra_mw"]]    = ww30_forced - ww30_dispatch
                sim_df.iat[i, col_idx["WW30_sim_tank_mw"]] = ww30_forced
                sim_df.iat[i, col_idx["WW30_on_tank"]]     = int(ww30_forced > 1e-9)
                prev_dispatch["WW30"] = ww30_forced

        # ── WW standby-minimum correction (WW01) ───────────────────────────────
        ww01_hist_active = row["ww01_hist_active"]
        if ww01_hist_active and enable_ww_standby_correction:
            ww01_min_cap       = row["WW01_min_mw"]
            ww01_max_cap       = row["WW01_max_mw"]
            ramp_lower_ww01    = max(0.0, prev_dispatch_before_ww01 - hx_params["WW01"]["ramp_down_mwph"])
            ww01_min_this_timestep = max(ww01_min_cap, ramp_lower_ww01)
            ww01_dispatch      = sim_df.iat[i, col_idx["WW01_sim_tank_mw"]]
            ww01_forced        = min(max(ww01_dispatch, ww01_min_this_timestep), ww01_max_cap)
            if ww01_forced > ww01_dispatch + 1e-9:
                sim_df.iat[i, col_idx["ww01_standby_forced_flag"]] = 1.0
                sim_df.iat[i, col_idx["ww01_standby_extra_mw"]]    = ww01_forced - ww01_dispatch
                sim_df.iat[i, col_idx["WW01_sim_tank_mw"]] = ww01_forced
                sim_df.iat[i, col_idx["WW01_on_tank"]]     = int(ww01_forced > 1e-9)
                prev_dispatch["WW01"] = ww01_forced

        non_bec_generation  = sum(
            sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
            for hx in priority_order if hx != "WWBEC"
        )
        non_bec_to_customer = min(non_bec_generation, demand_remaining)
        surplus             = max(0.0, non_bec_generation - non_bec_to_customer)

        # ── Pure-charge via spare HX capacity ─────────────────────────────────
        pure_charge_added_mw   = 0.0
        seasonal_buffer        = 0.10
        pure_charge_ceiling_wp = min(ates_soc_max, soc_target + seasonal_buffer)
        pure_charge_ceiling_n  = pure_charge_ceiling_wp
        pure_charge_ceiling_nw = pure_charge_ceiling_wp

        headroom_pure_wp = (
            max(0.0, pure_charge_ceiling_wp * cap_wp - ates_wp.V_stored) * CONST * dT_wp
            if mode_wp == "charge" and soc_wp < pure_charge_ceiling_wp else 0.0
        )
        headroom_pure_n  = (
            max(0.0, pure_charge_ceiling_n  * cap_n  - ates_n.V_stored)  * CONST * dT_n
            if mode_n  == "charge" and soc_n  < pure_charge_ceiling_n  else 0.0
        )
        headroom_pure_nw = (
            max(0.0, pure_charge_ceiling_nw * cap_nw - ates_nw.V_stored) * CONST * dT_nw
            if mode_nw == "charge" and soc_nw < pure_charge_ceiling_nw else 0.0
        )
        headroom_pure_wp = min(headroom_pure_wp, max_charge_wp)
        headroom_pure_n  = min(headroom_pure_n,  max_charge_n)
        headroom_pure_nw = min(headroom_pure_nw, max_charge_nw)
        total_pure_headroom = headroom_pure_wp + headroom_pure_n + headroom_pure_nw

        pure_charge_extra = 0.0
        if total_pure_headroom > 1e-9:
            for hx in pure_charge_order:
                if pure_charge_extra >= total_pure_headroom:
                    break
                p              = hx_params[hx]
                min_cap        = row[f"{hx}_min_mw"]
                max_cap        = row[f"{hx}_max_mw"]
                prev_hx        = prev_dispatch[hx]
                hist_available = row[f"{hx}_hist_available"] == 1
                if not hist_available:
                    continue
                already_for_hx   = sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
                ramp_upper_total = min(max_cap, prev_hx + p["ramp_up_mwph"])
                additional_room  = min(max(0.0, ramp_upper_total - already_for_hx),
                                       max_cap - already_for_hx)
                add_now = min(additional_room, total_pure_headroom - pure_charge_extra)
                if already_for_hx < 1e-9 and add_now > 0:
                    if add_now < min_cap:
                        add_now = min_cap if (min_cap <= additional_room and min_cap <= (total_pure_headroom - pure_charge_extra)) else 0.0
                if add_now <= 1e-9:
                    continue
                new_total = already_for_hx + add_now
                sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]] = new_total
                sim_df.iat[i, col_idx[f"{hx}_on_tank"]]     = int(new_total > 1e-9)
                prev_dispatch[hx]     = new_total
                pure_charge_extra    += add_now

        sim_df.iat[i, col_idx["pure_charge_ates_mw"]] = pure_charge_extra

        non_bec_generation = sum(
            sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
            for hx in priority_order if hx != "WWBEC"
        )
        surplus = max(0.0, non_bec_generation - non_bec_to_customer)

        small_charge        = min(surplus, small_feasible_charge)
        if aeb is not None:
            aeb.record(small_charge)
        surplus_after_small = max(0.0, surplus - small_charge)

        headroom_wp = (
            min(max_charge_wp, max(0.0, ates_soc_max * cap_wp - ates_wp.V_stored) * CONST * dT_wp)
            if mode_wp == "charge" else 0.0
        )
        headroom_n  = (
            min(max_charge_n,  max(0.0, ates_soc_max * cap_n  - ates_n.V_stored)  * CONST * dT_n)
            if mode_n  == "charge" else 0.0
        )
        headroom_nw = (
            min(max_charge_nw, max(0.0, ates_soc_max * cap_nw - ates_nw.V_stored) * CONST * dT_nw)
            if mode_nw == "charge" else 0.0
        )
        total_headroom = headroom_wp + headroom_n + headroom_nw

        if total_headroom > 1e-9 and surplus_after_small > 1e-9:
            allocatable = min(surplus_after_small, total_headroom)
            charge_wp   = allocatable * headroom_wp / total_headroom
            charge_n    = allocatable * headroom_n  / total_headroom
            charge_nw   = allocatable * headroom_nw / total_headroom
            leftover_curtailed = max(0.0, surplus_after_small - allocatable)
        else:
            charge_wp = charge_n = charge_nw = 0.0
            leftover_curtailed = surplus_after_small

        small_charge_vol_rate = small_charge        / (CONST * delta_t)
        small_disch_vol_rate  = small_planned_disch / (CONST * delta_t)
        small_volume = min(small_volume + small_charge_vol_rate, small_volume_max)
        if buffer_release_rule is None:
            small_volume = max(small_volume - small_disch_vol_rate, small_min_vol)
        else:   # a floor above the content only blocks discharge; it does not raise the content
            small_volume = max(small_volume - small_disch_vol_rate, min(small_min_vol, small_volume))

        T_out_wp, hp_wp, useful_wp, loss_wp = ates_wp.step(charge_wp, disch_wp)
        T_out_n,  hp_n,  useful_n,  loss_n  = ates_n.step(charge_n,  disch_n)
        T_out_nw, hp_nw, useful_nw, loss_nw = ates_nw.step(charge_nw, disch_nw)

        total_charge_mw      = charge_wp    + charge_n    + charge_nw
        total_disch_mw       = disch_wp     + disch_n     + disch_nw
        total_useful_mw      = useful_wp    + useful_n    + useful_nw
        total_ground_loss_mw = loss_wp      + loss_n      + loss_nw
        total_hp_el_mwh      = hp_wp        + hp_n        + hp_nw
        mech_pump_mwh        = (total_charge_mw + total_disch_mw) * mech_pump_factor
        total_pump_cost      = (total_hp_el_mwh + mech_pump_mwh)  * price_now

        T_out_rep = T_out_nw
        T_gap_rep = max(0.0, config_nieuwwest.network_supply_temp_c - T_out_rep)

        total_supply_to_customer = (
            bec_to_customer + non_bec_to_customer + small_planned_disch + total_useful_mw
        )
        unmet_demand = max(0.0, demand_t - total_supply_to_customer)
        curtailed    = leftover_curtailed + bec_curtailed

        sim_df.iat[i, col_idx["ates_westpoort_soc"]]          = ates_wp.V_stored / cap_wp if cap_wp > 0 else 0.0
        sim_df.iat[i, col_idx["ates_noord_soc"]]              = ates_n.V_stored  / cap_n  if cap_n  > 0 else 0.0
        sim_df.iat[i, col_idx["ates_nieuwwest_soc"]]          = ates_nw.V_stored / cap_nw if cap_nw > 0 else 0.0
        sim_df.iat[i, col_idx["ates_westpoort_discharge_mw"]] = disch_wp
        sim_df.iat[i, col_idx["ates_noord_discharge_mw"]]     = disch_n
        sim_df.iat[i, col_idx["ates_nieuwwest_discharge_mw"]] = disch_nw
        sim_df.iat[i, col_idx["ates_westpoort_charge_mw"]]    = charge_wp
        sim_df.iat[i, col_idx["ates_noord_charge_mw"]]        = charge_n
        sim_df.iat[i, col_idx["ates_nieuwwest_charge_mw"]]    = charge_nw
        sim_df.iat[i, col_idx["ates_westpoort_temp_c"]]       = ates_wp.T_aq
        sim_df.iat[i, col_idx["ates_noord_temp_c"]]           = ates_n.T_aq
        sim_df.iat[i, col_idx["ates_nieuwwest_temp_c"]]       = ates_nw.T_aq

        agg_vol = ates_wp.V_stored + ates_n.V_stored + ates_nw.V_stored
        sim_df.iat[i, col_idx["ates_volume_m3"]]           = agg_vol
        sim_df.iat[i, col_idx["ates_soc"]]                 = agg_vol / total_capacity if total_capacity > 0 else 0.0
        sim_df.iat[i, col_idx["ates_charge_mw"]]           = total_charge_mw
        sim_df.iat[i, col_idx["ates_discharge_mw"]]        = total_disch_mw
        sim_df.iat[i, col_idx["ates_aquifer_temp_c"]]      = T_out_rep
        sim_df.iat[i, col_idx["ates_outlet_temp_c"]]       = T_out_rep
        sim_df.iat[i, col_idx["ates_temp_gap_c"]]          = T_gap_rep
        sim_df.iat[i, col_idx["ates_heat_pump_mwel"]]      = total_hp_el_mwh
        sim_df.iat[i, col_idx["ates_useful_discharge_mw"]] = total_useful_mw
        sim_df.iat[i, col_idx["ates_ground_loss_mw"]]      = total_ground_loss_mw
        sim_df.iat[i, col_idx["ates_mech_pump_mwh"]]       = mech_pump_mwh
        sim_df.iat[i, col_idx["ates_pump_cost_eur"]]       = total_pump_cost
        sim_df.iat[i, col_idx["effective_demand_mw"]]      = effective_demand_t

        total_generation        = bec_dispatch + non_bec_generation
        total_generation_usable = bec_available_for_cust + non_bec_generation
        sim_df.iat[i, col_idx["small_tank_volume_m3"]]           = small_volume
        sim_df.iat[i, col_idx["small_tank_charge_mw"]]           = small_charge
        sim_df.iat[i, col_idx["small_tank_discharge_mw"]]        = small_planned_disch
        sim_df.iat[i, col_idx["tank_charge_mw"]]                 = small_charge + total_charge_mw
        sim_df.iat[i, col_idx["tank_discharge_mw"]]              = small_planned_disch + total_disch_mw
        sim_df.iat[i, col_idx["direct_from_hx_mw"]]              = bec_to_customer + non_bec_to_customer
        sim_df.iat[i, col_idx["direct_from_tank_mw"]]            = small_planned_disch + total_useful_mw
        sim_df.iat[i, col_idx["total_generation_tank_model_mw"]] = total_generation
        sim_df.iat[i, col_idx["BEC_own_use_mw"]]                 = bec_own_use
        sim_df.iat[i, col_idx["total_generation_usable_mw"]]     = total_generation_usable
        sim_df.iat[i, col_idx["total_supply_to_customer_mw"]]    = total_supply_to_customer
        sim_df.iat[i, col_idx["unmet_demand_tank_mw"]]           = unmet_demand
        sim_df.iat[i, col_idx["curtailed_or_excess_mw"]]         = curtailed

        total_electricity_loss = 0.0
        for hx in priority_order:
            heat_output = sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
            if hx == "WW30":
                ww30_aftap_cap   = row["WW30_aftap_cap_mw"]
                aftap_part       = min(heat_output, ww30_aftap_cap)
                reductie_part    = max(0.0, heat_output - aftap_part)
                electricity_loss = (
                    aftap_part    * derivation_factors["WW30_aftap"]
                    + reductie_part * derivation_factors["WW30_reductie"]
                )
            else:
                electricity_loss = heat_output * derivation_factors[hx]
            sim_df.iat[i, col_idx[f"{hx}_electricity_loss_mw"]] = electricity_loss
            if hx != "WWBEC":
                total_electricity_loss += electricity_loss

        derving_cost      = total_electricity_loss * price_now
        total_system_cost = derving_cost + total_pump_cost
        total_system_cost_corrected = total_system_cost + bec_correction_cost_eur

        sim_df.iat[i, col_idx["total_electricity_loss_mw"]]        = total_electricity_loss
        sim_df.iat[i, col_idx["derving_cost_eur"]]                 = derving_cost
        sim_df.iat[i, col_idx["total_system_cost_eur"]]            = total_system_cost
        sim_df.iat[i, col_idx["total_system_cost_eur_corrected"]]  = total_system_cost_corrected

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
