"""Parameters of the ATES configurations (capacity, temperatures, aquifer) for the dispatch models."""
from dataclasses import dataclass, field
from typing import Optional, Callable
import pandas as pd

# -- Option B standard: return temperature -------------------------------------------------------------------
# One return temperature for Option B standard: the cold-well temperature of the dispatch window (ates_cold_temp_c
# of the three B-std loop configs; ates_thermal_v2.py converts power to flow with T_inj - T_cold). It is set
# in config.py (BSTD_T_RETURN_C: 58 C = headline, the AEB mean return temperature 2023-2025; 70 C only in the
# return-temperature sensitivity) and checked at import (check_bstd_return). The field network_return_temp_c of the
# original file is removed: no model or analysis script read it.
import config as _C  # noqa: E402
BSTD_T_INJ_C = float(_C.BSTD_T_INJ_C)
BSTD_T_RETURN_C = float(_C.BSTD_T_RETURN_C)


@dataclass
class ScenarioConfig:
    # ── Identity ──────────────────────────────────────────────────────────────
    name:        str = "unnamed"
    description: str = ""
    technology:  str = "ptes"   # "ptes" | "ates_option_a" | "ates_option_b" | "ates_option_c"

    # ── PTES sizing (technology="ptes" only) ──────────────────────────────────
    large_tank_volume_m3:    float = 100_000.0
    large_charge_max_mw:     float = 60.0
    large_discharge_max_mw:  float = 60.0
    n_small_tanks:           int   = 3
    small_tank_volume_m3:    float = 1_200.0

    # ── ATES sizing ───────────────────────────────────────────────────────────
    ates_capacity_m3:        float = 1_000_000.0
    ates_charge_max_mw:      float = 25.0
    ates_discharge_max_mw:   float = 25.0

    # ── ATES temperature operating range ─────────────────────────────────────
    ates_injection_temp_c:   float = 90.0
    ates_cold_temp_c:        float = 70.0
    ates_ground_ambient_c:   float = 12.0

    # ── ATES aquifer thermal properties (site-specific — sweep as sensitivity) ─
    aquifer_thermal_conductivity_wmk: float = 2.0   # W/(m·K)
    aquifer_porosity:                 float = 0.30
    aquifer_depth_m:                  float = 100.0
    aquifer_thickness_m:              float = 30.0

    # ── Network interface ─────────────────────────────────────────────────────
    network_supply_temp_c:   float = 90.0
    ates_requires_heat_pump: bool  = False
    heat_pump_cop:           float = 5.0
    hx_efficiency:           float = 0.96

    # ── Dispatch thresholds ───────────────────────────────────────────────────
    annual_charge_pct:      int = 25
    annual_discharge_pct:   int = 75
    daily_charge_rank_n:    int = 6
    daily_discharge_rank_n: int = 6

    # ── Price scenario ────────────────────────────────────────────────────────
    price_scenario_fn: Optional[Callable] = None

    # ── Flags ─────────────────────────────────────────────────────────────────
    perfect_foresight:       bool = False
    enable_pure_charge:      bool = True
    include_thermal_losses:  bool = True


# ── Price-growth helper ───────────────────────────────────────────────────────

def make_price_growth(multiplier: float, start_yr: int = 2026, end_yr: int = 2030):
    """Return a function that scales a price Series by a linear ramp to `multiplier`."""
    import numpy as np

    def fn(prices: pd.Series) -> pd.Series:
        t = ((prices.index.year - start_yr) / (end_yr - start_yr)).clip(0, 1)
        return prices * (1.0 + (multiplier - 1.0) * t)

    return fn


# ── Named PTES scenarios ──────────────────────────────────────────────────────

SCENARIO_PTES_BASELINE = ScenarioConfig(
    name="ptes_100k",
    technology="ptes",
    large_tank_volume_m3=100_000.0,
    include_thermal_losses=True,
)

# ── Named ATES scenarios (v2 — updated parameters, see header) ───────────────

# Option A: HT-ATES at AEB plant site.
# Injection at 110°C into a pressurised deep aquifer (~200m).
# Ambient temperature elevated to 16°C to reflect geothermal gradient at depth.
# Porosity 0.25 and thickness 25m reflect typical NL deep sandy aquifer.
# No heat pump required: outlet blended down to 90°C via mixing valve when hot,
# topped up from plant supply via HX when cooled below 90°C.
SCENARIO_ATES_OPTION_A = ScenarioConfig(
    name="ates_option_a_v2",
    description="HT-ATES at AEB plant, injection at 110°C (deep pressurised aquifer, 200m)",
    technology="ates_option_a",
    ates_injection_temp_c=110.0,
    ates_cold_temp_c=58.0,
    network_supply_temp_c=90.0,
    ates_requires_heat_pump=False,
    ates_capacity_m3=1_000_000.0,
    ates_charge_max_mw=25.0,
    ates_discharge_max_mw=25.0,
    aquifer_depth_m=200.0,
    aquifer_thickness_m=25.0,
    aquifer_thermal_conductivity_wmk=2.0,
    aquifer_porosity=0.25,
    ates_ground_ambient_c=16.0,
)

# Option B: Two ATES systems at grid bottleneck nodes, modelled as one combined store.
# Each system: 250k m³ capacity, 25 MW charge/discharge.
# Combined: 500k m³, 50 MW charge/discharge.
# Injection at 90°C (grid supply temperature at bottleneck nodes).
# Heat pump required as soon as outlet temp drops below 90°C network setpoint.
# COP = 5.0 is a placeholder; actual COP depends on temperature lift per hour
# (computed from T_gap inside ATESThermalState — update to Carnot-based COP in v3).
SCENARIO_ATES_OPTION_B = ScenarioConfig(
    name="ates_option_b_v2",
    description="Two ATES systems at grid bottleneck nodes, injection at 90°C — modelled as one combined store (2 × 250k m³, 2 × 25 MW)",
    technology="ates_option_b",
    ates_injection_temp_c=90.0,
    ates_cold_temp_c=70.0,
    network_supply_temp_c=90.0,
    ates_requires_heat_pump=True,
    heat_pump_cop=5.0,
    ates_capacity_m3=500_000.0,
    ates_charge_max_mw=50.0,
    ates_discharge_max_mw=50.0,
    aquifer_depth_m=100.0,
    aquifer_thickness_m=20.0,
    aquifer_thermal_conductivity_wmk=2.0,
    aquifer_porosity=0.28,
    ates_ground_ambient_c=13.5,
)

SCENARIO_ATES_OPTION_C = ScenarioConfig(
    name="ates_option_c_ltdh",
    description="Low-temp ATES for new neighbourhood (4GDH pilot)",
    technology="ates_option_c",
    ates_injection_temp_c=60.0,
    ates_cold_temp_c=45.0,
    network_supply_temp_c=55.0,
    ates_requires_heat_pump=False,
    ates_capacity_m3=50_000.0,
    aquifer_depth_m=50.0,
    aquifer_thickness_m=15.0,
)

# ── Price scenarios ───────────────────────────────────────────────────────────

SCENARIO_PRICE_MODERATE = ScenarioConfig(
    name="price_moderate",
    price_scenario_fn=make_price_growth(1.5),
)
SCENARIO_PRICE_HIGH = ScenarioConfig(
    name="price_high",
    price_scenario_fn=make_price_growth(2.0),
)
SCENARIO_PRICE_EXTREME = ScenarioConfig(
    name="price_extreme",
    price_scenario_fn=make_price_growth(3.0),
)


# ── Three-loop Option B scenarios (v3 — site-specific Amsterdam subsurface) ───
#
# Shared aquifer parameters for Amsterdam subsurface (Dinoloket REGIS II)
# Depth: 288.61 – 322.40m (midpoint 305m)
# Aquifer thickness: 33.8m
# k_h: 2.5 – 5.0 m/d (central estimate 3.5 m/d)
# k_v: 0.01 – 0.05 m/d (central estimate 0.025 m/d)
# k_D: 100 – 250 m²/d (consistent with k_h × thickness)
# c-value: 1000 – 5000 days (good aquitard confinement)
# Ambient temp at 305m: 11 + 31 × 0.305 = 20.5°C
# Source: Dinoloket REGIS II, AEB-area Amsterdam

SCENARIO_ATES_B_WESTPOORT = ScenarioConfig(
    name="ates_option_b_westpoort",
    description="ATES at WestPoort grid node, 90°C injection — "
                "sized for WestPoort loop demand (19% of total). "
                "Amsterdam subsurface params from Dinoloket REGIS II.",
    technology="ates_option_b",
    ates_injection_temp_c=BSTD_T_INJ_C,
    ates_cold_temp_c=BSTD_T_RETURN_C,
    network_supply_temp_c=90.0,
    ates_requires_heat_pump=False,
    ates_capacity_m3=95_000.0,
    ates_charge_max_mw=9.5,
    ates_discharge_max_mw=9.5,
    aquifer_depth_m=305.0,
    aquifer_thickness_m=33.8,
    aquifer_thermal_conductivity_wmk=2.0,
    aquifer_porosity=0.28,
    ates_ground_ambient_c=20.5,
    # Site-specific: k_h=3.5 m/d (lower end for ATES — flag as constraint)
    # k_v=0.025 m/d, anisotropy ratio 140:1
    # Buoyancy losses not modelled — ROSIM validation planned
)

SCENARIO_ATES_B_NOORD = ScenarioConfig(
    name="ates_option_b_noord",
    description="ATES at Noord grid node, 90°C injection — "
                "sized for Noord loop demand (28% of total, "
                "upward adjusted for planned network growth). "
                "Amsterdam subsurface params from Dinoloket REGIS II.",
    technology="ates_option_b",
    ates_injection_temp_c=BSTD_T_INJ_C,
    ates_cold_temp_c=BSTD_T_RETURN_C,
    network_supply_temp_c=90.0,
    ates_requires_heat_pump=False,
    ates_capacity_m3=140_000.0,
    ates_charge_max_mw=14.0,
    ates_discharge_max_mw=14.0,
    aquifer_depth_m=305.0,
    aquifer_thickness_m=33.8,
    aquifer_thermal_conductivity_wmk=2.0,
    aquifer_porosity=0.28,
    ates_ground_ambient_c=20.5,
    # Noord sized slightly above proportional share to accommodate
    # planned network expansion per AEB operator input
)

SCENARIO_ATES_B_NIEUWWEST = ScenarioConfig(
    name="ates_option_b_nieuwwest",
    description="ATES at NieuwWest grid node, 90°C injection — "
                "sized for NieuwWest loop demand (53% of total). "
                "Amsterdam subsurface params from Dinoloket REGIS II.",
    technology="ates_option_b",
    ates_injection_temp_c=BSTD_T_INJ_C,
    ates_cold_temp_c=BSTD_T_RETURN_C,
    network_supply_temp_c=90.0,
    ates_requires_heat_pump=False,
    ates_capacity_m3=265_000.0,
    ates_charge_max_mw=26.5,
    ates_discharge_max_mw=26.5,
    aquifer_depth_m=305.0,
    aquifer_thickness_m=33.8,
    aquifer_thermal_conductivity_wmk=2.0,
    aquifer_porosity=0.28,
    ates_ground_ambient_c=20.5,
)


# ── Option A corrected reference case (v3) ────────────────────────────────────
#
# Parameters updated from scenarios_v2.py to reflect the 305m Dinoloket REGIS II
# aquifer layer (same layer used for three-loop Option B configs above).
# Sized to match Option B three-loop for a fair head-to-head comparison:
#   - Total volume: 500,000 m3 (same as Option B three-loop combined)
#   - Combined rate: +/-50 MW (same as Option B three-loop: 9.5+14.0+26.5 MW)
# Key distinction from Option B: injection at 110 degC (HT-ATES) vs 90 degC —
# higher energy density per m3 stored (52 K dT vs 20 K dT) but larger conductive
# losses due to greater temperature excess above the 20.5 degC ambient.
# No heat pump required: outlet blended to 90 degC via mixing valve when hot,
# topped up from plant HX supply when aquifer cools below 90 degC.
# Note: buoyancy losses (free convection at high Ra*) are not yet modelled —
# thermal recovery here is an upper bound.  ROSIM validation planned.
SCENARIO_ATES_OPTION_A_V3 = ScenarioConfig(
    name="ates_option_a_v3",
    description=(
        "HT-ATES at AEB plant, injection 110 degC. "
        "Dinoloket REGIS II 305m aquifer. "
        "Sized for fair comparison with Option B three-loop: 500k m3, +/-50 MW."
    ),
    technology="ates_option_a",
    ates_injection_temp_c=110.0,
    ates_cold_temp_c=58.0,
    network_supply_temp_c=90.0,
    ates_requires_heat_pump=False,
    ates_capacity_m3=500_000.0,
    ates_charge_max_mw=50.0,
    ates_discharge_max_mw=50.0,
    aquifer_depth_m=305.0,
    aquifer_thickness_m=33.8,
    aquifer_thermal_conductivity_wmk=2.0,
    aquifer_porosity=0.28,
    ates_ground_ambient_c=20.5,
)


# -- One B-std return temperature ----------------------------------------------------------
BSTD_CONFIGS = (SCENARIO_ATES_B_WESTPOORT, SCENARIO_ATES_B_NOORD, SCENARIO_ATES_B_NIEUWWEST)


def check_bstd_return(t_return, where=""):
    """Fails if a B-std value differs from the one return temperature BSTD_T_RETURN_C."""
    assert t_return == BSTD_T_RETURN_C, f"{where}: return temperature {t_return} != BSTD_T_RETURN_C {BSTD_T_RETURN_C}"
    for c in BSTD_CONFIGS:
        assert c.ates_cold_temp_c == BSTD_T_RETURN_C and c.ates_injection_temp_c == BSTD_T_INJ_C, \
            f"{where}: {c.name} window {c.ates_injection_temp_c}/{c.ates_cold_temp_c} != {BSTD_T_INJ_C}/{BSTD_T_RETURN_C}"


check_bstd_return(BSTD_T_RETURN_C, "scenarios_v4")
