"""Thermal model of the ATES storage zone (volume, temperature, heat loss to the ground), used by the ATES dispatch
models."""

import numpy as np


class ATESThermalState:
    """
    Per-timestep aquifer thermal state for one ATES storage zone.

    Parameters
    ----------
    config : ScenarioConfig
        Must provide:
        ates_ground_ambient_c, ates_injection_temp_c, ates_cold_temp_c,
        aquifer_thickness_m, aquifer_thermal_conductivity_wmk,
        network_supply_temp_c, ates_requires_heat_pump, heat_pump_cop,
        hx_efficiency
    """

    RHO_WATER = 1000.0   # kg/m³
    CP_WATER  = 4180.0   # J/(kg·K)

    def __init__(self, config):
        self.cfg = config
        self.T_aq  = config.ates_ground_ambient_c   # start at undisturbed ambient
        self.V_stored = 0.0                          # m³ currently in aquifer
        self.heat_loss_total_mwh = 0.0

    def step(
        self,
        charge_mw: float,
        discharge_mw: float,
        dt_h: float = 1.0,
    ) -> tuple:
        """
        Advance the aquifer thermal state by one timestep.

        Parameters
        ----------
        charge_mw    : heat power injected this hour (MW)
        discharge_mw : heat power withdrawn this hour (MW)
        dt_h         : timestep length in hours (default 1)

        Returns
        -------
        (T_out, pump_el_mwh, useful_mw, loss_mw)
        T_out       : outlet temperature = current mean aquifer temperature (°C)
        pump_el_mwh : heat-pump electricity consumed this hour (MWh_el)
        useful_mw   : useful heat delivered to network after top-up (MW)
        loss_mw     : conductive loss to ground (MW)
        """
        cfg   = self.cfg
        dt_s  = dt_h * 3600.0
        CONST = 0.001157   # MW/(K·m³/h)

        T_inj  = cfg.ates_injection_temp_c
        T_cold = cfg.ates_cold_temp_c
        T_amb  = cfg.ates_ground_ambient_c

        # ── Volume balance ────────────────────────────────────────────────────
        # Charge: inject hot water at T_inj, displace cold water from cold well.
        # Volume pumped = power / (CONST × ΔT_nominal).
        dT_inj    = max(1.0, T_inj - T_cold)
        charge_vol = charge_mw / (CONST * dT_inj) * dt_h   # m³ injected

        # Discharge: extract hot water at T_aq, replace with cold water.
        # Volume per MW depends on the ACTUAL aquifer ΔT vs cold return.
        dT_discharge  = max(1.0, self.T_aq - T_cold)
        discharge_vol = discharge_mw / (CONST * dT_discharge) * dt_h   # m³ extracted

        V_old = self.V_stored
        V_new = max(0.0, V_old + charge_vol - discharge_vol)
        self.V_stored = V_new

        # ── Conductive loss to surrounding rock (cylindrical plume) ──────────
        if V_new > 0 and cfg.aquifer_thickness_m > 0:
            r        = (V_new / (np.pi * cfg.aquifer_thickness_m)) ** 0.5
            A        = 2 * np.pi * r * cfg.aquifer_thickness_m + 2 * np.pi * r ** 2
            r_char   = max(1.0, 0.5 * r)
            dT_loss  = max(0.0, self.T_aq - T_amb)
            Q_loss_w = cfg.aquifer_thermal_conductivity_wmk * A * dT_loss / r_char
        else:
            Q_loss_w = 0.0

        Q_loss_J = Q_loss_w * dt_s          # J
        loss_mw  = Q_loss_w / 1e6           # MW

        # ── Temperature update: enthalpy-mixing formula ───────────────────────
        # Conservation of enthalpy for a well-mixed doublet:
        #   T_new × V_new = T_old × V_old
        #                 + charge_vol   × T_inj      (hot water in)
        #                 − discharge_vol × T_old      (hot water out at T_aq)
        #                 − Q_loss / (ρ·Cp)            (ground loss in K·m³)
        if V_new > 0:
            Q_loss_K = Q_loss_J / (self.RHO_WATER * self.CP_WATER)   # m³·K
            T_new = (
                self.T_aq * V_old
                + charge_vol * T_inj
                - discharge_vol * self.T_aq
                - Q_loss_K
            ) / V_new

            # Physical sanity check — should never trigger after this fix.
            # If it does, the diagnostic values below identify the source.
            if T_new > T_inj + 5.0:
                C_stored   = self.RHO_WATER * V_old * self.CP_WATER
                Q_in_diag  = self.RHO_WATER * self.CP_WATER * charge_vol * (T_inj - self.T_aq)
                Q_out_diag = self.RHO_WATER * self.CP_WATER * discharge_vol * (self.T_aq - T_cold)
                dT_diag    = T_new - self.T_aq
                print(
                    f"[ates_thermal_v2 WARNING] T_aq_new={T_new:.1f}°C exceeds "
                    f"T_inj+5={T_inj + 5.0:.1f}°C\n"
                    f"  C_stored={C_stored:.3e} J/K  Q_in={Q_in_diag:.3e} J  "
                    f"Q_out={Q_out_diag:.3e} J  Q_loss={Q_loss_J:.3e} J  dT={dT_diag:.2f} K\n"
                    f"  V_old={V_old:.1f} m³  V_new={V_new:.1f} m³  "
                    f"charge_vol={charge_vol:.2f} m³  discharge_vol={discharge_vol:.2f} m³"
                )

            # Clamp to physically valid range [T_amb, T_inj]
            self.T_aq = max(T_amb, min(T_inj, T_new))
        else:
            self.T_aq = T_amb

        self.heat_loss_total_mwh += loss_mw

        # ── Outlet temperature and top-up logic ──────────────────────────────
        T_out = self.T_aq
        T_gap = max(0.0, cfg.network_supply_temp_c - T_out)

        if T_gap > 0 and cfg.ates_requires_heat_pump:
            # Heat pump electricity to top up from T_aq to network setpoint
            pump_el_mwh = discharge_mw * T_gap / (cfg.heat_pump_cop * cfg.network_supply_temp_c)
            useful_mw   = discharge_mw
        elif T_gap > 0:
            # HX top-up from plant hot supply (Option A) — small efficiency loss only
            pump_el_mwh = 0.0
            useful_mw   = discharge_mw * cfg.hx_efficiency
        else:
            pump_el_mwh = 0.0
            useful_mw   = discharge_mw

        return T_out, pump_el_mwh, useful_mw, loss_mw

    def reset(self):
        """Reset state to initial conditions."""
        self.T_aq              = self.cfg.ates_ground_ambient_c
        self.V_stored          = 0.0
        self.heat_loss_total_mwh = 0.0
