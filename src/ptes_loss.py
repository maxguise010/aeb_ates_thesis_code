"""Steady-state heat-loss model of a large hot-water store (PTES/TTES), used by the tank dispatch model."""

import numpy as np

# ── U-values (W/m²·K) ────────────────────────────────────────────────────────
PTES_U_TOP    = 0.20   # insulated floating cover
PTES_U_SIDE   = 0.50   # liner + soil at high operating temperature
PTES_U_BOTTOM = 0.35   # liner + deep soil

# ── Ambient temperature references (°C) ──────────────────────────────────────
T_AMBIENT_AIR_NL  = 11.0   # annual mean air temperature, KNMI De Bilt
T_SOIL_SHALLOW_NL = 12.0   # annual mean soil temp at PTES side depth (5–15 m)
T_SOIL_DEEP_NL    = 10.0   # annual mean soil temp below PTES bottom (>20 m)


def ptes_geometry(volume_m3: float):
    """
    Approximate PTES geometry as a cylinder with depth/diameter ≈ 0.20.

    Returns
    -------
    (depth_m, diameter_m, top_area_m2, side_area_m2, bottom_area_m2)
    """
    if volume_m3 <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    diameter   = (volume_m3 / 0.55) ** (1 / 3) * 1.5
    depth      = volume_m3 / (np.pi * (diameter / 2) ** 2)
    top_area   = np.pi * (diameter / 2) ** 2
    bottom_area = top_area
    side_area  = np.pi * diameter * depth
    return depth, diameter, top_area, side_area, bottom_area


def ptes_loss_mw(volume_m3: float, water_temp_c: float) -> float:
    """
    Steady-state thermal power loss (MW) for a PTES of given volume and
    mean water temperature.  Each surface uses its own U-value and ambient
    temperature reference.

    Called each timestep by run_tank_simulation_v5() via the large_tank_water_temp_c
    parameter.  Result is converted to m³ and subtracted from large_volume.
    """
    if volume_m3 <= 0:
        return 0.0
    _, _, top_a, side_a, bot_a = ptes_geometry(volume_m3)
    q_top    = PTES_U_TOP    * top_a  * max(0.0, water_temp_c - T_AMBIENT_AIR_NL)
    q_side   = PTES_U_SIDE   * side_a * max(0.0, water_temp_c - T_SOIL_SHALLOW_NL)
    q_bottom = PTES_U_BOTTOM * bot_a  * max(0.0, water_temp_c - T_SOIL_DEEP_NL)
    return (q_top + q_side + q_bottom) / 1e6   # W → MW
