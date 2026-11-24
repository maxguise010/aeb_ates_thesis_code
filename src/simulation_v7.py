"""Dispatch model with the existing buffers and a large tank (TTES), including the BEC term; also gives the
zero-capacity tank runs. Needs the raw data."""

import numpy as np
import pandas as pd

from ptes_loss import ptes_loss_mw
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
# no longer exist here. Baseline = hour-by-hour historical
# BEC production (WW_BEC). Hours with historical BEC < 1.0 MW
# are treated as standstill and do not count as
# a valid baseline (see simulation_ates_three_loops_v7.py).
#
# Fixed prices:
BEC_BIOMASSA_PRIJS_EUR_PER_GJ = 5.69    # average 2025
BEC_SUBSIDIE_PRIJS_EUR_PER_GJ = 9.17    # subsidy per GJ
BEC_ELEK_PER_WARMTE = 1.09 / 3.93      # GJ electricity per GJ heat
BEC_BIOMASSA_PER_WARMTE = 5.69 / 3.93  # GJ biomass per GJ heat
BEC_SUBSIDIE_PER_WARMTE = 1 + BEC_ELEK_PER_WARMTE  # GJ eligible for subsidy per GJ heat
BEC_MIN_OPERATIONAL_MW = 1.0  # MW — below this value = standstill, no baseline
BEC_BASELINE_CORRECTION = True


def bec_net_revenue_per_gj(epex_eur_per_mwh):
    # Convert EPEX from €/MWh to €/GJ (1 MWh = 3.6 GJ)
    epex_gj = epex_eur_per_mwh / 3.6

    biomassa_bedrag = -BEC_BIOMASSA_PER_WARMTE * BEC_BIOMASSA_PRIJS_EUR_PER_GJ
    elek_bedrag     =  BEC_ELEK_PER_WARMTE     * epex_gj
    subsidie_bedrag =  BEC_SUBSIDIE_PER_WARMTE * BEC_SUBSIDIE_PRIJS_EUR_PER_GJ

    return biomassa_bedrag + elek_bedrag + subsidie_bedrag  # €/GJ heat

# ── WW standby-minimum correction ────────────────────────────
# If WW30 or WW01 was historically operational at time step t
# (production >= threshold), the simulation must keep it at that same
# time step at least at its standby minimum. This model has
# no ATES/large seasonal storage (large_tank_volume_m3 = 1.0,
# dispatch-only baseline) that can legitimately replace these HXs,
# so the correction is switched on here — unlike in
# simulation_ates_three_loops_v7.py, where ATES storage can rightly
# switch off WW30/WW01.
#
# Threshold values to be submitted to the supervisor for verification.

WW_STANDBY_MIN_OPERATIONAL_THRESHOLD_MW = 1.0  # MW — below this value = switched off
WW30_HIST_COL = "WW_30"      # historical production column in sim_df_tank
WW01_HIST_COL = "WW_01"      # historical production column in sim_df_tank

# ── PRIJSGESTUURDE WARMTEPLANNING (v5) ──────────────────────────
# Instead of following demand, AEB delivers based on the forecast price
# how much heat is delivered in that hour. High forecast price = less
# deliver heat ("sell" more electricity by using less HX power
# to use); low forecast price = deliver more heat than the demand
# (tops up the small buffer). Settlement is always on the actual
# EPEX price -- see price_driven_planning in run_tank_simulation_v5().
#
# Delivery fractions per price level (percentile of the forecast-price
# distribution over the whole simulation period). This is the central/default
# set; alternative sets can be passed per run via the
# delivery_fractions parameter (see the sensitivity analysis in
# run_dispatch_only_v5_prijsgestuurd.py).
DELIVERY_FRACTIONS = {
    'p99': 0.50,   # above P99: deliver only 50% of the demand
    'p90': 0.65,   # above P90: deliver 65% of the demand
    'p75': 0.80,   # above P75: deliver 80% of the demand
    'p50': 1.00,   # between P25 and P75 (around the median): deliver the full demand
    'p25': 1.10,   # below P25: deliver 110% (also charges the buffer)
    'p10': 1.20,   # below P10: deliver 120% (fills the buffer to the maximum)
}

# Average derving factor (MWh of electricity forgone per MWh of heat),
# used to express the price-mismatch cost (price_miss_eur) in the same unit
# expressed as derving_cost_eur elsewhere in this file. WWBEC is
# excluded, as in the derving cost count itself (BEC is a
# captive steam by-product, not an "avoided HX use").
DERVING_FACTOR_AVG = float(
    np.mean([v for k, v in derivation_factors.items() if k != "WWBEC"])
)

# ── BEC ZOMERBEPERKING ────────────────────────────────────────
# Between 1 May and 1 October the BEC may deliver at most 70% of its
# use its nominal capacity.
# Reason: maintenance, lower heat demand, technical limitations.
# Applies ONLY to the dispatch-only baseline — not to ATES/PTES.
# Source: supervisor setting July 2026.

enable_bec_summer_cap = False  # off by default — switch on explicitly in the run scripts

BEC_SUMMER_CAP_FRACTION = 0.70        # 70% van max_mw
BEC_SUMMER_START_MONTH  = 5           # May
BEC_SUMMER_END_MONTH    = 10          # to September inclusive (< October)

# ── MINIMUM RUNTIME CONSTRAINT ───────────────────────────────
# As soon as a heat exchanger runs above its minimum value,
# it must stay active for at least min_runtime_hours before
# it can be switched off completely.
# After switching off it must stay off for at least min_downtime_hours
# before it can be started again.
#
# Reason: frequent on/off switching is operationally not
# realistic and causes wear of the installation.
# Expected effect: HXs that now run for only 1-2 hours are
# forced to stay on longer, which brings the simulated production
# closer to the historical production.
#
# Applies to: WW10, WW30, WW01
# Not for:  WWBEC (own logic), WW00/WW02 (small auxiliary sources)
# Source: supervisor setting July 2026
#
# Implementation notes:
# - hx_params[hx] has NO static 'min_mw' key for WW10/WW30/WW01
# (only WWBEC has one) -- the forced-on floor therefore uses the
# real per-time-step technical minimum (min_cap = row[f"{hx}_min_mw"]),
#   not hx_params[hx].get('min_mw', ...).
# - WW30 has two sub-stages (aftap/reductie) within the same
# time-step iteration (non_bec_dispatch_sequence). The forced-on floor
# is therefore applied only to the aftap sub-stage (otherwise the
# floor would be added twice); forced-off (max=0) does apply to both
# sub-stages, because that can safely be added up (0+0=0).
# - Known limitation: if useful_remaining for an HX is already (almost) 0
# because other sources already meet the demand in that hour, the
# dispatch still set to 0 despite a raised stage_min_cap --
# forced-on floor is then not effectively enforced. This is a
# limitation of the supplied approach (raising min instead of forcing the dispatch
# directly), not corrected so as not to move further from the requested
# set-up than necessary.
MIN_RUNTIME_HX = ['WW10', 'WW30', 'WW01']

# ── WARMTEVRAAG CORRECTIEFACTOR ──────────────────────────────
# The simulation historically produces 2.5% less non-BEC heat
# than the historical data. On the supervisor's advice the
# heat demand per time step is raised by a fixed factor so that
# the total heat production matches the historical data.
# This compensates for pipe losses and other unmodelled
# losses that are included in the historical production figures.
#
# Passed as the function parameter demand_correction_factor
# (default = 1.0, no correction). Set to 1.025 for the 2.5%
# correction. Source: supervisor setting July 2026.


def run_tank_simulation_v5(
    sim_input_df,
    n_small_tanks=3,
    small_tank_volume_m3=1200.0,
    small_charge_max_mw=None,
    small_discharge_max_mw=None,
    small_initial_soc=0.5,
    small_soc_max=0.95,
    small_soc_min=0.10,
    large_tank_volume_m3=150_000.0,
    large_charge_max_mw=200.0,
    large_discharge_max_mw=200.0,
    large_initial_soc=0.20,
    large_soc_max=0.95,
    large_soc_min=0.05,
    annual_charge_pct=25,
    annual_discharge_pct=75,
    daily_charge_rank_n=6,
    daily_discharge_rank_n=6,
    reserve_horizon_h=6,
    reserve_max_fraction=0.20,
    enable_pure_charge_large=True,
    large_tank_water_temp_c=90.0,
    price_col="Prijs_Gemiddelde_EPEX_Spot",
    perfect_foresight=False,
    enable_ww_standby_correction=True,
    # BEC fixed at historical values — no optimisation of BEC.
    # BEC v2 correction switched off because BEC never deviates from history.
    bec_fixed_historical=False,
    # Price-driven heat planning (v5) — instead of following demand, AEB delivers
    # based on the forecast price. Settlement is always on the actual EPEX.
    price_driven_planning=False,
    delivery_fractions=None,
    # BEC summer limit — ONLY for the dispatch-only baseline (see
    # module-level BEC_SUMMER_CAP_FRACTION/_START_MONTH/_END_MONTH above).
    enable_bec_summer_cap=False,
    # Minimum runtime constraint — ONLY for WW10/WW30/WW01 (see
    # module-level MIN_RUNTIME_HX above).
    enable_min_runtime: bool = False,
    min_runtime_hours: int = 18,   # at least 18 hours on once started
    min_downtime_hours: int = 2,   # at least 2 hours off after switching off
    # ΔT (K) for volume <-> energy of the LARGE tank. None (default) = the
    # hourly network ΔT (delta_t_c), as in all thesis runs. A fixed value
    # (e.g. T_charge - T_return = 95 - 58 = 37 K) models a tank charged at a
    # lower temperature; the small buffers stay at delta_t_c.
    large_tank_delta_t_k=None,
    # Rule for the existing buffers: "rank" (default, daily price rank) or
    # "aeb_demand" (old AEB demand rule, see aeb_buffer_rule.py).
    existing_buffer_rule: str = "rank",
    # Length (days) of the rolling price window for the charge/discharge thresholds of the
    # LARGE tank. None (default) = 365 days, as in all thesis runs. Always causal (only
    # prices up to the current hour). The BEC displacement threshold (annual_50) stays at 365 days.
    large_price_window_days=None,
    # Conditional release of the existing buffers (default None = off). The buffer arguments
    # above then describe the base rule (e.g. calibrated); the 'release' arguments the rule that
    # may apply when the large tank can cover the reserve, i.e. usable tank heat
    # H = (V_tank - large_soc_min x V) x CONST x dT >= R = 'release' reserve (demand of the next
    # release_reserve_horizon_h hours, max. release_reserve_max_fraction x buffer volume) in MWh.
    # "hourly": per hour only the floor: release reserve if H >= R, otherwise the base floor
    # (ceiling and rank stay those of the base rule);
    # "daily":  at the first hour of each (UTC) day: if H >= max_day(R) then for the whole day the
    # release rule (floor, ceiling release_small_soc_max, rank release_rank_n), otherwise the base rule.
    # A floor higher than the content only blocks discharging (no raising of the content).
    buffer_release_rule=None,
    release_small_soc_max=0.95,
    release_rank_n=6,
    release_reserve_horizon_h=6,
    release_reserve_max_fraction=0.20,
    # Diagnostic/relaxation options for the large tank (defaults = thesis behaviour):
    # large_threshold_mode "percentile" (default) or "daily_rank": charge in the large_daily_rank_n
    # cheapest, discharge in the large_daily_rank_n most expensive hours of each (UTC) day (day-ahead
    # prices, without SOC adjustment);
    # large_ignore_seasonal_target: no SOC-excess adjustment of the percentiles and no
    # seasonal ceiling (target + 10 pp) for pure charge (ceiling = large_soc_max);
    # pure_charge_ignore_ramp: pure charge may ramp every available unit up to its maximum;
    # record_tank_diagnostics: extra columns diag_* (thresholds, feasible charge/discharge, residual demand).
    large_threshold_mode="percentile",
    large_daily_rank_n=6,
    large_ignore_seasonal_target=False,
    pure_charge_ignore_ramp=False,
    record_tank_diagnostics=False,
    # Heat demand correction factor — see the module-level explanation above.
    demand_correction_factor: float = 1.0,  # default = no correction; 1.025 = +2.5%
):
    """
    Two-tier storage:
      - n_small_tanks of small_tank_volume_m3 each (default: 3 x 1200 m3)
      - 1 large tank of large_tank_volume_m3
    Small tanks do daily arbitrage, large tank does seasonal.
    All HX min/max capacity and ramping constraints enforced.
    Trailing 365-day rolling percentile thresholds (no future info).
    """
    sim_df = sim_input_df.copy()

    # ── CHANGE v3: dispatch thresholds based on forecast prices ─────────
    # AEB plans heat delivery day-ahead based on the forecast.
    # The thresholds determine when heat is delivered vs electricity.
    # Source: supervisor e-mail July 2026
    from forecast_prices import load_forecast_prices

    # Load the forecast prices (already time-zone corrected to UTC)
    forecast_df = load_forecast_prices()
    # Add to sim_df as an extra column
    sim_df = sim_df.join(forecast_df['voorspelling_prijs'], how='left')
    sim_df['voorspelling_prijs'] = sim_df['voorspelling_prijs'].fillna(sim_df['Prijs_Gemiddelde_EPEX_Spot'])

    # FOR (historical baseline without storage):
    # Use the forecast price for the dispatch decision (P25/P75 thresholds)
    dispatch_price_col = 'voorspelling_prijs'   # for planning

    # FOR the derving calculation: always use the actual EPEX
    # (settled at the realised price) -- price_col (default
    # 'Prijs_Gemiddelde_EPEX_Spot') is still used unchanged for
    # derving_cost_eur and the BEC v2 revenue formula.

    # ── CHANGE v5: price-driven heat planning ──────────────────────
    # AEB decides day-ahead how much heat is delivered based on
    # the forecast price (instead of following demand). The thresholds are percentiles
    # of the forecast-price distribution over the whole simulation period (fixed
    # policy rule, no day-by-day optimisation or trailing window such as
    # at the tier-1 thresholds above).
    if price_driven_planning:
        fractions = delivery_fractions if delivery_fractions is not None else DELIVERY_FRACTIONS

        p10_thresh = sim_df['voorspelling_prijs'].quantile(0.10)
        p25_thresh = sim_df['voorspelling_prijs'].quantile(0.25)
        p50_thresh = sim_df['voorspelling_prijs'].quantile(0.50)
        p75_thresh = sim_df['voorspelling_prijs'].quantile(0.75)
        p90_thresh = sim_df['voorspelling_prijs'].quantile(0.90)
        p99_thresh = sim_df['voorspelling_prijs'].quantile(0.99)

        print("  Voorspelprijs drempelwaarden (EUR/MWh):")
        print(f"    P10={p10_thresh:.1f}  P25={p25_thresh:.1f}  P50={p50_thresh:.1f}")
        print(f"    P75={p75_thresh:.1f}  P90={p90_thresh:.1f}  P99={p99_thresh:.1f}")

        forecast_prijs = sim_df['voorspelling_prijs']
        sim_df['delivery_fraction'] = np.select(
            [
                forecast_prijs >= p99_thresh,
                forecast_prijs >= p90_thresh,
                forecast_prijs >= p75_thresh,
                forecast_prijs >= p25_thresh,
                forecast_prijs >= p10_thresh,
            ],
            [
                fractions['p99'],
                fractions['p90'],
                fractions['p75'],
                fractions['p50'],
                fractions['p25'],
            ],
            default=fractions['p10'],
        )
        sim_df['planned_delivery_mw'] = sim_df['heat_demand_mw'] * sim_df['delivery_fraction']
    else:
        sim_df['delivery_fraction'] = 1.0
        sim_df['planned_delivery_mw'] = sim_df['heat_demand_mw']

    if small_charge_max_mw is None:
        small_charge_max_mw = tank_charge_max_mw * (n_small_tanks / 3.0)
    if small_discharge_max_mw is None:
        small_discharge_max_mw = tank_discharge_max_mw * (n_small_tanks / 3.0)

    small_volume_max = n_small_tanks * small_tank_volume_m3
    small_volume_init = small_volume_max * small_initial_soc
    large_volume_init = large_tank_volume_m3 * large_initial_soc

    trailing_window_h = 365 * 24
    tank_window_h = (trailing_window_h if large_price_window_days is None
                     else int(round(float(large_price_window_days) * 24)))

    if perfect_foresight:
        tier1_charge_thresh_scalar = sim_df[dispatch_price_col].quantile(annual_charge_pct / 100)
        tier1_discharge_thresh_scalar = sim_df[dispatch_price_col].quantile(annual_discharge_pct / 100)
        sim_df["tier1_charge_thresh"] = tier1_charge_thresh_scalar
        sim_df["tier1_discharge_thresh"] = tier1_discharge_thresh_scalar
    else:
        sim_df["tier1_charge_thresh"] = (
            sim_df[dispatch_price_col]
            .rolling(window=tank_window_h, min_periods=24)
            .quantile(annual_charge_pct / 100)
        )
        sim_df["tier1_discharge_thresh"] = (
            sim_df[dispatch_price_col]
            .rolling(window=tank_window_h, min_periods=24)
            .quantile(annual_discharge_pct / 100)
        )
    sim_df["tier1_charge"] = sim_df[dispatch_price_col] <= sim_df["tier1_charge_thresh"]
    sim_df["tier1_discharge"] = sim_df[dispatch_price_col] >= sim_df["tier1_discharge_thresh"]

    rank_asc = sim_df.groupby(sim_df.index.date)[price_col].rank(method="first", ascending=True)
    rank_desc = sim_df.groupby(sim_df.index.date)[price_col].rank(method="first", ascending=False)
    sim_df["tier2_charge"] = rank_asc <= daily_charge_rank_n
    sim_df["tier2_discharge"] = rank_desc <= daily_discharge_rank_n
    if large_threshold_mode not in ("percentile", "daily_rank"):
        raise ValueError(f"large_threshold_mode must be 'percentile' or 'daily_rank', got {large_threshold_mode!r}")
    if large_threshold_mode == "daily_rank":
        sim_df["large_rank_asc"] = sim_df.groupby(sim_df.index.date)[dispatch_price_col].rank(method="first", ascending=True)
        sim_df["large_rank_desc"] = sim_df.groupby(sim_df.index.date)[dispatch_price_col].rank(method="first", ascending=False)
    if record_tank_diagnostics:
        for c in ("diag_charge_thresh", "diag_discharge_thresh", "diag_large_feasible_charge",
                  "diag_demand_after_small", "diag_large_max_disch", "diag_pure_charge_room",
                  "diag_large_charge_room_mw", "diag_pure_ceiling"):
            sim_df[c] = np.nan

    demand = sim_df["heat_demand_mw"].copy()
    demand_padded = pd.concat([demand, demand.iloc[:90*24]])
    forward_mean = demand_padded.rolling(window=90*24, min_periods=24).mean().shift(-(90*24-1))
    forward_mean = forward_mean.iloc[:len(demand)]
    forward_mean = forward_mean.bfill().ffill()
    fd_min, fd_max = forward_mean.min(), forward_mean.max()
    sim_df["large_soc_target_seasonal"] = (
        large_soc_min + (large_soc_max - large_soc_min)
        * (forward_mean - fd_min) / (fd_max - fd_min)
    ).values

    if perfect_foresight:
        sim_df["annual_50_thresh"] = sim_df[price_col].quantile(0.50)
    else:
        sim_df["annual_50_thresh"] = (
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
        if buffer_release_rule not in ("hourly", "daily"):
            raise ValueError(f"buffer_release_rule must be None, 'hourly' or 'daily', got {buffer_release_rule!r}")
        rel_raw = sum(sim_df["required_volume_for_demand_m3"].shift(-h).fillna(0.0)
                      for h in range(1, release_reserve_horizon_h + 1))
        sim_df["release_min_volume_m3"] = np.clip(rel_raw, 0.0, release_reserve_max_fraction * small_volume_max)
        rel_res_mwh = sim_df["release_min_volume_m3"] * CONST * safe_delta_t
        sim_df["release_reserve_day_max_mwh"] = rel_res_mwh.groupby(sim_df.index.date).transform("max")
        sim_df["release_tier2_charge"] = rank_asc <= release_rank_n
        sim_df["release_tier2_discharge"] = rank_desc <= release_rank_n
        sim_df["buffer_release_flag"] = 0.0

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

    for hx in priority_order:
        sim_df[f"{hx}_sim_tank_mw"] = 0.0
        sim_df[f"{hx}_on_tank"] = 0
        sim_df[f"{hx}_electricity_loss_mw"] = 0.0
    new_cols = [
        "small_tank_volume_m3", "large_tank_volume_m3",
        "small_tank_charge_mw", "small_tank_discharge_mw",
        "large_tank_charge_mw", "large_tank_discharge_mw",
        "tank_charge_mw", "tank_discharge_mw",
        "tank_volume_m3", "tank_energy_mwh",
        "direct_from_hx_mw", "direct_from_tank_mw",
        "total_generation_tank_model_mw", "BEC_own_use_mw",
        "total_generation_usable_mw", "total_supply_to_customer_mw",
        "unmet_demand_tank_mw", "curtailed_or_excess_mw",
        "total_electricity_loss_mw",
        "pure_charge_large_mw", "derving_cost_eur",
        "small_mode_v5", "large_mode_v5", "bec_displaced_flag",
        "large_tank_thermal_loss_mw",
        "bec_hist_baseline_mw", "bec_extra_production_mw",
        "bec_extra_revenue_eur", "bec_correction_cost_eur", "total_cost_eur_corrected",
        "ww30_standby_forced_flag", "ww30_standby_extra_mw",
        "ww01_standby_forced_flag", "ww01_standby_extra_mw",
        "forecast_price_t", "epex_price_t", "price_miss_eur",
    ]
    for hx in MIN_RUNTIME_HX:
        new_cols.append(f"{hx}_minruntime_forced_on_flag")
        new_cols.append(f"{hx}_minruntime_forced_off_flag")
    for c in new_cols:
        sim_df[c] = 0.0

    check_rule(existing_buffer_rule)
    aeb = (AebBufferRule(sim_df, small_volume_max, price_col)
           if existing_buffer_rule == "aeb_demand" else None)

    prev_dispatch = {hx: 0.0 for hx in priority_order}
    small_volume = small_volume_init
    large_volume = large_volume_init

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

    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        if price_driven_planning:
            # Planned delivery (based on the forecast price) instead of the actual
            # heat demand drives the rest of the dispatch loop below.
            # A shortfall against the actual demand is covered by the
            # existing small-buffer logic (small tank discharge below),
            # or counts as unmet_demand_tank_mw if the buffer is empty.
            demand_t = row["planned_delivery_mw"]
            forecast_price_t = row["voorspelling_prijs"]
            epex_price_t = row[price_col]
            sim_df.iat[i, col_idx["forecast_price_t"]] = forecast_price_t
            sim_df.iat[i, col_idx["epex_price_t"]] = epex_price_t
            # Price mismatch against the planned delivery: positive = the
            # actual EPEX price turned out higher than forecast (AEB has, at
            # on the basis of the too-low forecast more heat was planned than
            # optimal in hindsight = missed electricity revenue).
            # Expressed in the same unit as derving_cost_eur below
            # (MW electricity equivalent * EUR/MWh, no /1000: that would give
            # EUR/1000, not comparable with derving_cost_eur elsewhere in this file).
            sim_df.iat[i, col_idx["price_miss_eur"]] = (
                (epex_price_t - forecast_price_t) * demand_t * DERVING_FACTOR_AVG
            )
        else:
            demand_t = row["heat_demand_mw"] * demand_correction_factor
        # MIN_PHYSICAL_DELTA_T: guard against bad measurement values where supply ≤ return
        delta_t = max(row["delta_t_c"], MIN_PHYSICAL_DELTA_T)
        large_dt = delta_t if large_tank_delta_t_k is None else float(large_tank_delta_t_k)
        small_min_vol = row["small_min_volume_required_m3"]
        price_now = row[price_col]
        cur_soc_max = small_soc_max
        cur_t2_charge, cur_t2_discharge = row["tier2_charge"], row["tier2_discharge"]
        if buffer_release_rule is not None:
            tank_heat_mwh = max(0.0, large_volume - large_soc_min * large_tank_volume_m3) * CONST * large_dt
            if buffer_release_rule == "hourly":
                release_now = tank_heat_mwh >= row["release_min_volume_m3"] * CONST * delta_t
                if release_now:
                    small_min_vol = row["release_min_volume_m3"]
            else:
                if i == 0 or sim_df.index[i].date() != sim_df.index[i - 1].date():
                    release_day = tank_heat_mwh >= row["release_reserve_day_max_mwh"]
                release_now = release_day
                if release_now:
                    small_min_vol = row["release_min_volume_m3"]
                    cur_soc_max = release_small_soc_max
                    cur_t2_charge, cur_t2_discharge = row["release_tier2_charge"], row["release_tier2_discharge"]
            sim_df.iat[i, col_idx["buffer_release_flag"]] = 1.0 if release_now else 0.0

        small_soc = small_volume / small_volume_max if small_volume_max > 0 else 0.0
        large_soc = large_volume / large_tank_volume_m3 if large_tank_volume_m3 > 0 else 0.0

        small_can_charge = small_soc < cur_soc_max
        small_can_discharge = small_soc > small_soc_min
        small_want_charge = bool(cur_t2_charge) and small_can_charge
        small_want_discharge = bool(cur_t2_discharge) and small_can_discharge
        if small_want_discharge:
            small_want_charge = False
        small_mode = "charge" if small_want_charge else ("discharge" if small_want_discharge else "drift")
        if aeb is not None:
            small_mode, small_min_vol = aeb.hour(i)

        large_can_charge = large_soc < large_soc_max
        large_can_discharge = large_soc > large_soc_min
        large_seasonal_target = row["large_soc_target_seasonal"]
        soc_excess = large_soc - large_seasonal_target
        if large_ignore_seasonal_target:
            soc_excess = 0.0

        # dispatch_price_now / dispatch_window_prices: forecast price, used
        # for the P25/P75 planning thresholds below (large-tank charge/
        # discharge decision). window_prices (EPEX) stays unchanged for
        # the BEC displacement threshold (annual_50_now) further on.
        dispatch_price_now = row[dispatch_price_col]
        if perfect_foresight:
            window_prices = sim_df[price_col]
            dispatch_window_prices = sim_df[dispatch_price_col]
        else:
            window_start_idx = max(0, i - trailing_window_h + 1)
            window_prices = sim_df[price_col].iloc[window_start_idx:i+1]
            dispatch_window_prices = sim_df[dispatch_price_col].iloc[max(0, i - tank_window_h + 1):i+1]

        adjusted_discharge_pct = max(50.0, annual_discharge_pct - 70.0 * max(0.0, soc_excess))
        adjusted_discharge_thresh = dispatch_window_prices.quantile(adjusted_discharge_pct / 100.0)
        tier1_discharge_now = dispatch_price_now >= adjusted_discharge_thresh

        adjusted_charge_pct = annual_charge_pct - 70.0 * max(0.0, soc_excess)
        if adjusted_charge_pct < 5.0:
            tier1_charge_now = False
        else:
            adjusted_charge_thresh = dispatch_window_prices.quantile(adjusted_charge_pct / 100.0)
            tier1_charge_now = dispatch_price_now <= adjusted_charge_thresh
        if large_threshold_mode == "daily_rank":
            tier1_charge_now = row["large_rank_asc"] <= large_daily_rank_n
            tier1_discharge_now = row["large_rank_desc"] <= large_daily_rank_n
        if record_tank_diagnostics:
            sim_df.iat[i, col_idx["diag_discharge_thresh"]] = adjusted_discharge_thresh
            sim_df.iat[i, col_idx["diag_charge_thresh"]] = (adjusted_charge_thresh if adjusted_charge_pct >= 5.0
                                                            else np.nan)

        large_want_charge = tier1_charge_now and large_can_charge
        large_want_discharge = tier1_discharge_now and large_can_discharge
        if large_want_discharge:
            large_want_charge = False
        large_mode = "charge" if large_want_charge else ("discharge" if large_want_discharge else "drift")

        sim_df.iat[i, col_idx["small_mode_v5"]] = (
            1.0 if small_mode == "charge" else (-1.0 if small_mode == "discharge" else 0.0)
        )
        sim_df.iat[i, col_idx["large_mode_v5"]] = (
            1.0 if large_mode == "charge" else (-1.0 if large_mode == "discharge" else 0.0)
        )

        for hx in priority_order:
            sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]] = 0.0
            sim_df.iat[i, col_idx[f"{hx}_on_tank"]] = 0

        # BEC: must-run; displaced to minimum when large tank overfull
        bec_p = hx_params["WWBEC"]
        # BEC maximum — taking the summer cap into account
        bec_max_cap = row["WWBEC_max_mw_effective"]
        # minimum stays unchanged
        bec_min_cap = row["WWBEC_min_mw"]
        bec_prev = prev_dispatch["WWBEC"]
        bec_hist_available = row["WWBEC_hist_available"] == 1

        annual_50_now = row["annual_50_thresh"]
        if pd.isna(annual_50_now):
            annual_50_now = window_prices.quantile(0.50)

        bec_displace_active = (
            large_tank_volume_m3 > 1.0
            and large_soc > 0.85
            and price_now >= annual_50_now
            and large_can_discharge
            and bec_hist_available
        )

        if bec_fixed_historical:
            # BEC always exactly equal to the historical value --
            # no optimisation, no extra production, no v2 correction.
            bec_dispatch = row['WW_BEC']
            bec_displace_active = False
        elif not bec_hist_available:
            bec_dispatch = 0.0
        else:
            if bec_displace_active:
                bec_target = bec_min_cap
            else:
                bec_target = max(bec_min_cap, min(demand_t + bec_own_use_mw, bec_max_cap))
            bec_ramp_lower = max(0.0, bec_prev - bec_p["ramp_down_mwph"])
            bec_ramp_upper = min(bec_max_cap, bec_prev + bec_p["ramp_up_mwph"])
            bec_dispatch = float(np.clip(bec_target, bec_ramp_lower, bec_ramp_upper))
            bec_dispatch = max(0.0, min(bec_dispatch, bec_max_cap))

        sim_df.iat[i, col_idx["bec_displaced_flag"]] = 1.0 if bec_displace_active else 0.0

        bec_own_use = min(bec_own_use_mw, bec_dispatch)
        bec_available_for_customer = max(0.0, bec_dispatch - bec_own_use)
        bec_to_customer = min(bec_available_for_customer, demand_t)
        bec_curtailed = max(0.0, bec_available_for_customer - bec_to_customer)
        sim_df.iat[i, col_idx["WWBEC_sim_tank_mw"]] = bec_dispatch
        sim_df.iat[i, col_idx["WWBEC_on_tank"]] = int(bec_dispatch > 1e-9)
        prev_dispatch["WWBEC"] = bec_dispatch
        demand_after_bec = max(0.0, demand_t - bec_to_customer)

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
        if bec_fixed_historical:
            # BEC never deviates from history -- no v2 correction.
            bec_extra_mw     = 0.0
            bec_extra_revenue = 0.0
        elif BEC_BASELINE_CORRECTION and bec_hist_active_now:
            bec_extra_mw = max(0.0, bec_dispatch - bec_hist_baseline_mw)
            bec_extra_gj      = bec_extra_mw * 3.6
            bec_extra_revenue = bec_extra_gj * bec_net_revenue_per_gj(price_now)
        else:
            bec_extra_mw = 0.0
            bec_extra_revenue = 0.0
        bec_extra_production_mw = bec_extra_mw
        bec_correction_cost_eur = -bec_extra_revenue
        sim_df.iat[i, col_idx["bec_hist_baseline_mw"]]    = bec_hist_baseline_mw
        sim_df.iat[i, col_idx["bec_extra_production_mw"]] = bec_extra_production_mw
        sim_df.iat[i, col_idx["bec_extra_revenue_eur"]]   = bec_extra_revenue
        sim_df.iat[i, col_idx["bec_correction_cost_eur"]] = bec_correction_cost_eur

        # Small tank discharge
        small_avail_discharge_vol = max(0.0, small_volume - small_min_vol)
        small_max_disch_mw = min(small_discharge_max_mw, small_avail_discharge_vol * CONST * delta_t)
        if small_mode == "discharge":
            small_planned_discharge = min(demand_after_bec, small_max_disch_mw)
        else:
            small_planned_discharge = 0.0
        demand_after_small = max(0.0, demand_after_bec - small_planned_discharge)

        # Large tank discharge
        large_avail_discharge_vol = max(0.0, large_volume - large_soc_min * large_tank_volume_m3)
        large_max_disch_mw = min(large_discharge_max_mw, large_avail_discharge_vol * CONST * large_dt)
        if large_mode == "discharge":
            large_planned_discharge = min(demand_after_small, large_max_disch_mw)
        else:
            large_planned_discharge = 0.0
        demand_remaining = max(0.0, demand_after_small - large_planned_discharge)
        if record_tank_diagnostics:
            sim_df.iat[i, col_idx["diag_demand_after_small"]] = demand_after_small
            sim_df.iat[i, col_idx["diag_large_max_disch"]] = large_max_disch_mw

        # Small tank charge headroom
        small_avail_charge_vol = max(0.0, small_volume_max - small_volume)
        small_charge_room_mw = small_avail_charge_vol * CONST * delta_t
        small_soc_room_mw = max(0.0, (cur_soc_max * small_volume_max - small_volume) * CONST * delta_t)
        if small_mode == "charge":
            small_feasible_charge = min(small_charge_max_mw, small_charge_room_mw, small_soc_room_mw)
        else:
            small_feasible_charge = 0.0
        if aeb is not None and small_mode == "charge":
            small_feasible_charge = aeb.feasible_charge(i, small_volume, delta_t, small_charge_max_mw)

        # Large tank charge headroom
        large_avail_charge_vol = max(0.0, large_tank_volume_m3 - large_volume)
        large_charge_room_mw = large_avail_charge_vol * CONST * large_dt
        large_soc_room_mw = max(0.0, (large_soc_max * large_tank_volume_m3 - large_volume) * CONST * large_dt)
        if large_mode == "charge":
            large_feasible_charge = min(large_charge_max_mw, large_charge_room_mw, large_soc_room_mw)
        else:
            large_feasible_charge = 0.0

        non_bec_target = demand_remaining + small_feasible_charge

        # Non-BEC demand-following dispatch
        # Captured before the stage loop -- these are the previous timestep's
        # committed dispatch values, needed for the standby-minimum ramp check
        # below (the loop itself overwrites prev_dispatch["WW30"]/["WW01"]).
        prev_dispatch_before_ww30 = prev_dispatch["WW30"]
        prev_dispatch_before_ww01 = prev_dispatch["WW01"]

        # ── Minimum runtime: update the consecutive on/off counters ───────────
        # Uses the committed dispatch of the PREVIOUS time step (prev_dispatch,
        # not yet overwritten in this iteration) -- updated exactly once per time step
        # per HX, so no double update for WW30's two sub-stages.
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

            p = hx_params[hx]
            min_cap = row[f"{hx}_min_mw"]
            max_cap = row[f"{hx}_max_mw"]
            prev = prev_dispatch[hx]
            hist_available = row[f"{hx}_hist_available"] == 1

            if hx == "WW30":
                ww30_aftap_cap = row["WW30_aftap_cap_mw"]
                if ww30_stage == "aftap":
                    stage_min_cap = 0.0
                    stage_max_cap = min(max_cap, ww30_aftap_cap)
                else:
                    stage_min_cap = 0.0
                    stage_max_cap = max(0.0, max_cap - ww30_aftap_cap)
            else:
                stage_min_cap = min_cap
                stage_max_cap = max_cap

            # ── Minimum runtime constraint (WW10/WW30/WW01 only) ─────────────
            if enable_min_runtime and hx in MIN_RUNTIME_HX:
                if 0 < hx_consecutive_off[hx] < min_downtime_hours:
                    # Forced off -- set maximum to zero (applies to both
                    # WW30 sub-stages, safe to add up: 0 + 0 = 0)
                    stage_max_cap = 0.0
                elif 0 < hx_consecutive_on[hx] < min_runtime_hours and ww30_stage != "reductie":
                    # Forced on -- raise the dispatch minimum to the real
                    # per-time-step technical minimum (min_cap, not a
                    # hx_params fallback -- see the module-level explanation
                    # above). For WW30 only on the aftap sub-stage
                    # applied to avoid double counting.
                    stage_min_cap = max(stage_min_cap, min_cap)

            useful_remaining = max(0.0, non_bec_target - dispatched_non_bec)
            allow_on = hist_available and (stage_max_cap > 1e-9)

            if not allow_on or useful_remaining <= 1e-9:
                desired_stage_dispatch = 0.0
            else:
                desired_stage_dispatch = min(stage_max_cap, useful_remaining)
                if 0 < desired_stage_dispatch < stage_min_cap:
                    desired_stage_dispatch = stage_min_cap

            already_for_hx = sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
            ramp_lower_total = max(0.0, prev - p["ramp_down_mwph"])
            ramp_upper_total = min(max_cap, prev + p["ramp_up_mwph"])
            desired_total = already_for_hx + desired_stage_dispatch
            desired_total = float(np.clip(desired_total, ramp_lower_total, ramp_upper_total))
            dispatch_stage = max(0.0, desired_total - already_for_hx)
            remaining_stage_room = max(0.0, stage_max_cap - already_for_hx)
            dispatch_stage = min(dispatch_stage, remaining_stage_room)
            forced_due_to_ramp = max(0.0, ramp_lower_total - already_for_hx)
            dispatch_stage = min(dispatch_stage, max(useful_remaining, forced_due_to_ramp))

            new_total_for_hx = already_for_hx + dispatch_stage
            new_total_for_hx = max(0.0, min(new_total_for_hx, max_cap))
            sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]] = new_total_for_hx
            sim_df.iat[i, col_idx[f"{hx}_on_tank"]] = int(new_total_for_hx > 1e-9)
            dispatched_non_bec += dispatch_stage
            prev_dispatch[hx] = new_total_for_hx

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

        # Pure-charge for LARGE tank only
        pure_charge_added_mw = 0.0
        seasonal_buffer = 0.10
        pure_charge_ceiling = min(large_soc_max, large_seasonal_target + seasonal_buffer)
        if large_ignore_seasonal_target:
            pure_charge_ceiling = large_soc_max
        if record_tank_diagnostics:
            sim_df.iat[i, col_idx["diag_large_feasible_charge"]] = large_feasible_charge
            sim_df.iat[i, col_idx["diag_large_charge_room_mw"]] = large_soc_room_mw
            sim_df.iat[i, col_idx["diag_pure_ceiling"]] = pure_charge_ceiling
            sim_df.iat[i, col_idx["diag_pure_charge_room"]] = max(
                0.0, (pure_charge_ceiling * large_tank_volume_m3 - large_volume) * CONST * large_dt)
        pure_charge_allowed = (
            enable_pure_charge_large
            and large_mode == "charge"
            and large_feasible_charge > 0
            and large_soc < pure_charge_ceiling
        )
        if pure_charge_allowed:
            pure_charge_room_mw = max(
                0.0,
                (pure_charge_ceiling * large_tank_volume_m3 - large_volume) * CONST * large_dt
            )
            charge_room = min(large_feasible_charge, pure_charge_room_mw)

            for hx in pure_charge_order:
                if charge_room <= 1e-9:
                    break
                p = hx_params[hx]
                min_cap = row[f"{hx}_min_mw"]
                max_cap = row[f"{hx}_max_mw"]
                prev = prev_dispatch[hx]
                hist_available = row[f"{hx}_hist_available"] == 1
                if not hist_available:
                    continue
                already_for_hx = sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
                ramp_upper_total = min(max_cap, prev + p["ramp_up_mwph"])
                if pure_charge_ignore_ramp:
                    ramp_upper_total = max_cap
                additional_room = max(0.0, ramp_upper_total - already_for_hx)
                additional_room = min(additional_room, max_cap - already_for_hx)
                add_now = min(additional_room, charge_room)
                if already_for_hx < 1e-9 and add_now > 0:
                    if add_now < min_cap:
                        if min_cap <= additional_room and min_cap <= charge_room:
                            add_now = min_cap
                        else:
                            add_now = 0.0
                if add_now <= 1e-9:
                    continue
                new_total = already_for_hx + add_now
                sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]] = new_total
                sim_df.iat[i, col_idx[f"{hx}_on_tank"]] = int(new_total > 1e-9)
                prev_dispatch[hx] = new_total
                pure_charge_added_mw += add_now
                charge_room -= add_now

        sim_df.iat[i, col_idx["pure_charge_large_mw"]] = pure_charge_added_mw

        non_bec_generation = sum(
            sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
            for hx in priority_order if hx != "WWBEC"
        )

        # Allocate non-BEC output
        non_bec_to_customer = min(non_bec_generation, demand_remaining)
        surplus = max(0.0, non_bec_generation - non_bec_to_customer)

        small_charge = min(surplus, small_feasible_charge)
        if aeb is not None:
            aeb.record(small_charge)
        surplus_after_small = surplus - small_charge

        large_charge = min(surplus_after_small, large_feasible_charge)
        leftover_curtailed = max(0.0, surplus_after_small - large_charge)

        small_charge_flow_m3h = small_charge / (CONST * delta_t)
        small_discharge_flow_m3h = small_planned_discharge / (CONST * delta_t)
        large_charge_flow_m3h = large_charge / (CONST * large_dt)
        large_discharge_flow_m3h = large_planned_discharge / (CONST * large_dt)

        small_volume = min(small_volume + small_charge_flow_m3h, small_volume_max)
        if buffer_release_rule is None:
            small_volume = max(small_volume - small_discharge_flow_m3h, small_min_vol)
        else:
            # a floor above the content only blocks discharge; it does not raise the content
            small_volume = max(small_volume - small_discharge_flow_m3h, min(small_min_vol, small_volume))
        large_volume = min(large_volume + large_charge_flow_m3h, large_tank_volume_m3)
        large_volume = max(large_volume - large_discharge_flow_m3h,
                           large_soc_min * large_tank_volume_m3)

        # Standing thermal loss through PTES walls (heat leaks to surrounding soil each hour).
        # Expressed as volume lost so it reduces SOC and constrains future dispatch.
        ptes_wall_loss_mw = 0.0
        if large_tank_volume_m3 > 1.0:
            ptes_wall_loss_mw = ptes_loss_mw(large_tank_volume_m3, large_tank_water_temp_c)
            loss_vol_m3 = ptes_wall_loss_mw / (CONST * large_dt)
            large_volume = max(large_soc_min * large_tank_volume_m3, large_volume - loss_vol_m3)

        total_supply_to_customer = (
            bec_to_customer + non_bec_to_customer
            + small_planned_discharge + large_planned_discharge
        )
        unmet_demand = max(0.0, demand_t - total_supply_to_customer)
        curtailed = leftover_curtailed + bec_curtailed

        sim_df.iat[i, col_idx["small_tank_volume_m3"]] = small_volume
        sim_df.iat[i, col_idx["large_tank_volume_m3"]] = large_volume
        sim_df.iat[i, col_idx["large_tank_thermal_loss_mw"]] = ptes_wall_loss_mw
        sim_df.iat[i, col_idx["small_tank_charge_mw"]] = small_charge
        sim_df.iat[i, col_idx["small_tank_discharge_mw"]] = small_planned_discharge
        sim_df.iat[i, col_idx["large_tank_charge_mw"]] = large_charge
        sim_df.iat[i, col_idx["large_tank_discharge_mw"]] = large_planned_discharge
        sim_df.iat[i, col_idx["tank_charge_mw"]] = small_charge + large_charge
        sim_df.iat[i, col_idx["tank_discharge_mw"]] = small_planned_discharge + large_planned_discharge
        sim_df.iat[i, col_idx["tank_volume_m3"]] = small_volume + large_volume
        sim_df.iat[i, col_idx["tank_energy_mwh"]] = (
            (small_volume + large_volume) * CONST * delta_t if large_tank_delta_t_k is None
            else small_volume * CONST * delta_t + large_volume * CONST * large_dt
        )
        sim_df.iat[i, col_idx["direct_from_hx_mw"]] = bec_to_customer + non_bec_to_customer
        sim_df.iat[i, col_idx["direct_from_tank_mw"]] = small_planned_discharge + large_planned_discharge

        total_generation = bec_dispatch + non_bec_generation
        total_generation_usable = bec_available_for_customer + non_bec_generation
        sim_df.iat[i, col_idx["total_generation_tank_model_mw"]] = total_generation
        sim_df.iat[i, col_idx["BEC_own_use_mw"]] = bec_own_use
        sim_df.iat[i, col_idx["total_generation_usable_mw"]] = total_generation_usable
        sim_df.iat[i, col_idx["total_supply_to_customer_mw"]] = total_supply_to_customer
        sim_df.iat[i, col_idx["unmet_demand_tank_mw"]] = unmet_demand
        sim_df.iat[i, col_idx["curtailed_or_excess_mw"]] = curtailed

        total_electricity_loss = 0.0
        for hx in priority_order:
            heat_output = sim_df.iat[i, col_idx[f"{hx}_sim_tank_mw"]]
            if hx == "WW30":
                ww30_aftap_cap = row["WW30_aftap_cap_mw"]
                aftap_part = min(heat_output, ww30_aftap_cap)
                reductie_part = max(0.0, heat_output - aftap_part)
                electricity_loss = (
                    aftap_part * derivation_factors["WW30_aftap"] +
                    reductie_part * derivation_factors["WW30_reductie"]
                )
            else:
                electricity_loss = heat_output * derivation_factors[hx]
            sim_df.iat[i, col_idx[f"{hx}_electricity_loss_mw"]] = electricity_loss
            # BEC is captive steam byproduct — excluded from total derving cost
            if hx != "WWBEC":
                total_electricity_loss += electricity_loss
        sim_df.iat[i, col_idx["total_electricity_loss_mw"]] = total_electricity_loss
        derving_cost = total_electricity_loss * price_now
        sim_df.iat[i, col_idx["derving_cost_eur"]] = derving_cost
        sim_df.iat[i, col_idx["total_cost_eur_corrected"]] = derving_cost + bec_correction_cost_eur

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
