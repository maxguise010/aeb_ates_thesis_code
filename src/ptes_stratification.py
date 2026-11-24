"""Stratified tank model with N layers (conduction, buoyancy mixing, plug flow), used by the thermal replay of the
TTES."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PTESStratificationConfig:
    # Geometrie
    volume_m3:          float = 25_000.0   # total volume [m³]
    height_m:           float = 10.0       # tank depth [m]
    slope_ratio:        float = 1.0        # horizontal distance per unit height (side slope)

    # Stratificatie
    n_layers:           int   = 20         # number of layers (Formhals 2024: 20-50 recommended)

    # Thermal properties of water
    rho_water:          float = 985.0      # density [kg/m³] @ ~60°C average
    cp_water:           float = 4182.0     # specific heat [J/(kg·K)]
    lambda_water:       float = 0.6        # thermal conductivity of water [W/(m·K)]

    # Mixing coefficient of turbulent mixing during active through-flow
    # Calibration parameter — higher value = poorer stratification
    # Typical range: 0.01 (good diffuser design) to 0.5 (poor design)
    alpha_turb:         float = 0.05       # [W/(m²·K)] during charging/discharging

    # U-values of the insulation [W/(m²·K)]
    U_dek:              float = 0.15       # floating cover at the top
    U_zijwand:          float = 0.30       # insulated side walls
    U_bodem:            float = 0.10       # bottom (the ground insulates)

    # Temperaturen
    T_omgeving:         float = 10.0       # ambient temperature [°C]
    T_netwerk_setpoint: float = 90.0       # minimum useful outlet temperature [°C]
    T_koud_retour:      float = 58.0       # network return temperature [°C]
    T_initieel:         float = 15.0       # initial tank temperature [°C]


class PTESStratificationModel:
    """
    Multi-layer stratified thermal model for PTES.

    Models thermal stratification with N horizontal layers
    with heat conduction, buoyancy correction and turbulent mixing.

    Layer 0   = top layer (hottest, charging/discharging here)
    Layer N-1 = bottom layer (coldest, cold return water here)

    Referentie: Formhals et al. (2024) Geothermal Energy 12:23
    DOI: 10.1186/s40517-024-00302-9
    """

    def __init__(self, config: PTESStratificationConfig):
        self.cfg = config
        self.n = config.n_layers

        # Compute the layer geometry
        self._calc_geometry()

        # Initialise the temperature profile — uniformly cold
        self.T = np.full(self.n, config.T_initieel, dtype=float)

        # Tracking
        self.total_charged_mwh   = 0.0
        self.total_discharged_mwh = 0.0
        self.total_loss_mwh      = 0.0
        self.total_useful_mwh    = 0.0  # only energy above T_netwerk_setpoint (quality correction)
        self.total_curtailed_mwh = 0.0  # energy that did not fit in the tank because of the CFL limit on m_flow

    def _calc_geometry(self):
        cfg = self.cfg
        # Height per layer
        self.dz = cfg.height_m / self.n

        # Volumes per layer (trapezoid — lower layers smaller)
        # Top width larger than bottom width because of the side slope
        # Simplified: uniform volume per layer
        self.V_layer = np.full(self.n, cfg.volume_m3 / self.n)
        self.m_layer = self.V_layer * cfg.rho_water  # mass per layer [kg]

        # Areas for the heat loss per layer [m²]
        # Layer 0 (top): cover + side wall
        # Layers 1..N-2: side wall only
        # Layer N-1 (bottom): bottom + side wall
        A_dwarsdoorsnede = cfg.volume_m3 / cfg.height_m  # average cross-section [m²]
        omtrek = 4 * np.sqrt(A_dwarsdoorsnede)            # square approximation
        A_zijwand_per_laag = omtrek * self.dz

        self.UA_verlies = np.zeros(self.n)
        for i in range(self.n):
            UA = A_zijwand_per_laag * cfg.U_zijwand
            if i == 0:       # top layer — add the cover
                UA += A_dwarsdoorsnede * cfg.U_dek
            if i == self.n-1: # bottom layer — add the bottom
                UA += A_dwarsdoorsnede * cfg.U_bodem
            self.UA_verlies[i] = UA

        # Conduction area between layers [m²]
        self.A_geleiding = A_dwarsdoorsnede  # cross-section between layers

    def step(self,
             power_mw:        float,    # positive = charging, negative = discharging [MW]
             T_injectie:      float,    # injection temperature when charging [°C]
             dt_hours:        float = 1.0) -> dict:
        """
        Carry out one time step.

        Returns dict with: T_uitlaat, nuttige_energie_mwh, verlies_mwh,
                          T_profiel, gemiddelde_T, recovery_fractie (outlet temperature, useful energy, loss, profile, mean T, recovery fraction)
        """
        cfg = self.cfg
        dt = dt_hours * 3600  # to seconds
        T = self.T.copy()
        n = self.n

        # ── 1. HEAT CONDUCTION BETWEEN ADJACENT LAYERS ─────────
        Q_cond = np.zeros(n)
        for i in range(1, n):
            q = cfg.lambda_water * self.A_geleiding * (T[i-1] - T[i]) / self.dz
            Q_cond[i-1] += q
            Q_cond[i]   -= q

        # ── 2. TURBULENT MIXING DURING ACTIVE USE ─────────
        Q_turb = np.zeros(n)
        is_active = abs(power_mw) > 1e-6
        if is_active:
            for i in range(1, n):
                q = cfg.alpha_turb * self.A_geleiding * (T[i-1] - T[i])
                Q_turb[i-1] += q
                Q_turb[i]   -= q

        # ── 3. HEAT LOSS TO THE SURROUNDINGS ───────────────────────
        Q_loss = self.UA_verlies * (T - cfg.T_omgeving)

        # ── 4. MASS TRANSFER (charging/discharging) — plug flow through ALL
        # layers in series (upwind/donor-cell advection), not
        # only between the top and bottom layer.
        Q_massaflow = np.zeros(n)
        T_uitlaat = T[0]  # default outlet = top layer
        nuttige_mwh = 0.0
        nuttige_mwh_kwaliteit = 0.0  # quality-corrected variant (only relevant when discharging)
        # Actually transported energy based on the (possibly CFL-limited)
        # m_flow -- can be lower than the power command power_mw*dt_hours
        # when m_flow is limited below this by the stability cap
        # (see module analysis: happens in ~34% of the active hours, mainly
        # when T[0] is close to T_injectie or T_koud_retour respectively).
        actual_charged_mwh = 0.0
        actual_discharged_mwh = 0.0
        curtailed_mwh = 0.0
        min_layer_mass = np.min(self.m_layer)

        if power_mw > 1e-6:
            # CHARGING: hot water enters at the top (layer 0) at
            # T_injection. Each layer passes its content to the layer
            # below it (the same m_flow flows in series through the whole
            # column); the bottom layer gives its content to the
            # return network.
            energy_in_w = power_mw * 1e6
            m_flow = energy_in_w / (cfg.cp_water * max(1.0, T_injectie - T[0]))
            m_flow = min(m_flow, min_layer_mass / dt * 0.5)  # CFL-like limit (stability)
            Q_massaflow[0] = m_flow * cfg.cp_water * (T_injectie - T[0])
            for i in range(1, n):
                Q_massaflow[i] = m_flow * cfg.cp_water * (T[i-1] - T[i])

            # Actually stored energy based on the (possibly limited)
            # m_flow, instead of the claimed power command. This is the sum
            # of all Q_massflow terms (telescopes to
            # m_flow*cp*(T_injection - T[n-1]) -- NOT T[0]: there is no
            # explicit outflow term at the bottom during charging, so the net
            # energy entering the whole tank depends on the
            # bottom-layer temperature, not the top one. Purely the
            # using the local top flux (T_injectie - T[0]) would
            # strongly underestimate the stored energy as soon as T[0], towards
            # T_injection rises, while the tank via the bottom layers still
            # is still taking up a lot of energy.
            Q_actual_charge_w = max(0.0, float(np.sum(Q_massaflow)))
            actual_charged_mwh = Q_actual_charge_w * dt / 3.6e9  # W·s -> MWh
            curtailed_mwh = power_mw * dt_hours - actual_charged_mwh
            nuttige_mwh = actual_charged_mwh
            # nuttige_mwh_kwaliteit stays 0.0 during charging: the
            # quality criterion (>=T_netwerk_setpoint) concerns
            # DELIVERY to the network, which only happens when discharging. Would
            # count the charging energy here as well, it would
            # useful_recovery_efficiency() = total_useful_mwh/total_charged_mwh
            # partly mix the charging with the discharging side and no pure
            # "useful discharge relative to charged" fraction any more.

        elif power_mw < -1e-6:
            # DISCHARGING: cold return water enters at the bottom (layer
            # N-1). Each layer passes its content to the layer
            # above; the top layer delivers to the network
            # (T_outlet, measured BEFORE this update).
            T_uitlaat = T[0]

            energy_out_w = abs(power_mw) * 1e6
            m_flow = energy_out_w / (cfg.cp_water * max(1.0, T[0] - cfg.T_koud_retour))
            m_flow = min(m_flow, min_layer_mass / dt * 0.5)
            for i in range(0, n - 1):
                Q_massaflow[i] = m_flow * cfg.cp_water * (T[i+1] - T[i])
            Q_massaflow[n-1] = m_flow * cfg.cp_water * (cfg.T_koud_retour - T[n-1])

            # Actually extracted energy based on the (possibly limited)
            # m_flow, instead of the claimed power command. Sum of all
            # Q_massaflow terms (telescopes to -m_flow*cp*(T[0] -
            # T_koud_retour); negative = energy leaves the tank) --
            # same principle as for charging, here coincidentally equal to the
            # local top flux, because T_koud_retour is the fixed
            # boundary condition here (instead of T[n-1] as when charging).
            Q_actual_discharge_w = max(0.0, -float(np.sum(Q_massaflow)))
            actual_discharged_mwh = Q_actual_discharge_w * dt / 3.6e9  # W·s -> MWh

            if T_uitlaat >= cfg.T_netwerk_setpoint:
                # Fully useful
                nuttige_mwh = actual_discharged_mwh
            else:
                # Partly useful — only energy above the setpoint
                dT_nuttig = max(0.0, T_uitlaat - cfg.T_koud_retour)
                dT_totaal = max(1.0, T_uitlaat - cfg.T_koud_retour)
                nuttige_mwh = actual_discharged_mwh * (dT_nuttig / dT_totaal)

            # ── CORRECTED useful-energy metric (quality correction) ──
            # NB: nuttige_mwh above stays UNCHANGED (dT_nuttig and
            # dT_totaal are mathematically identical as soon as T_uitlaat-T_koud_retour
            # > 1.0, so that branch is a no-op in practice -- useful_mwh
            # in fact simply measures the total discharge, not the discharge above the
            # setpoint). This separate calculation implements the actual
            # temperature/quality-corrected variant with linear
            # interpolation, kept separately in self.total_useful_mwh so that
            # the existing nuttige_mwh/'Energie-recovery' remains traceable
            # for comparison.
            if T_uitlaat >= cfg.T_netwerk_setpoint:
                nuttige_mwh_kwaliteit = actual_discharged_mwh
            elif T_uitlaat <= cfg.T_koud_retour:
                nuttige_mwh_kwaliteit = 0.0
            else:
                fractie = (T_uitlaat - cfg.T_koud_retour) / (cfg.T_netwerk_setpoint - cfg.T_koud_retour)
                nuttige_mwh_kwaliteit = actual_discharged_mwh * min(1.0, fractie)

        # ── 5. TEMPERATUURUPDATE ──────────────────────────────────
        for i in range(n):
            dQ = (Q_cond[i] + Q_turb[i] - Q_loss[i] + Q_massaflow[i])
            T[i] += dQ * dt / (self.m_layer[i] * cfg.cp_water)

        # ── 6. BUOYANCY CORRECTION (thermal inversion) ──────────
        # Formhals et al. 2024: if layer i is warmer than layer i-1 (above it)
        # → mix directly (resolve the density inversion)
        #
        # Instrumentatie (buoyancy_events_this_step /
        # buoyancy_exergy_lost_mwh_this_step): the mixing itself conserves
        # energy exactly (mass-weighted mean), so "energy lost"
        # is here not an enthalpy loss but the loss of usable
        # energy above T_network_setpoint -- the energy that before
        # mixing was still above the network setpoint but after mixing is not
        # any more.
        buoyancy_events_this_step = 0
        buoyancy_exergy_lost_mwh_this_step = 0.0
        changed = True
        while changed:
            changed = False
            for i in range(1, n):
                if T[i] > T[i-1] + 0.01:  # 0.01°C tolerance
                    m_a, m_b = self.m_layer[i-1], self.m_layer[i]
                    T_mix = (m_b * T[i] + m_a * T[i-1]) / (m_a + m_b)
                    E_before = (
                        m_a * max(0.0, T[i-1] - cfg.T_netwerk_setpoint)
                        + m_b * max(0.0, T[i] - cfg.T_netwerk_setpoint)
                    ) * cfg.cp_water
                    E_after = (m_a + m_b) * max(0.0, T_mix - cfg.T_netwerk_setpoint) * cfg.cp_water
                    buoyancy_exergy_lost_mwh_this_step += max(0.0, E_before - E_after) / 3.6e9
                    buoyancy_events_this_step += 1
                    T[i]   = T_mix
                    T[i-1] = T_mix
                    changed = True

        # ── 7. UPDATE STATE ───────────────────────────────────────
        self.T = T
        verlies_mwh = float(np.sum(Q_loss)) * dt / 3.6e9

        self.total_charged_mwh    += actual_charged_mwh
        self.total_discharged_mwh += actual_discharged_mwh
        self.total_useful_mwh     += nuttige_mwh_kwaliteit
        self.total_curtailed_mwh  += curtailed_mwh
        self.total_buoyancy_events = getattr(self, 'total_buoyancy_events', 0) + buoyancy_events_this_step
        self.total_buoyancy_exergy_lost_mwh = (
            getattr(self, 'total_buoyancy_exergy_lost_mwh', 0.0) + buoyancy_exergy_lost_mwh_this_step
        )
        self.total_loss_mwh       += verlies_mwh

        return {
            'T_uitlaat':      T_uitlaat,
            'T_profiel':      T.copy(),
            'gemiddelde_T':   float(np.mean(T)),
            'T_bovenste':     float(T[0]),
            'T_onderste':     float(T[-1]),
            'nuttige_mwh':    nuttige_mwh,
            'nuttige_mwh_kwaliteit': nuttige_mwh_kwaliteit,
            'actual_charged_mwh':    actual_charged_mwh,
            'actual_discharged_mwh': actual_discharged_mwh,
            'curtailed_mwh':         curtailed_mwh,
            'verlies_mwh':    verlies_mwh,
            'thermocline_idx': int(np.argmax(np.diff(T) < -2.0)) if np.any(np.diff(T) < -2.0) else n//2,
            'buoyancy_events':          buoyancy_events_this_step,
            'buoyancy_exergy_lost_mwh': buoyancy_exergy_lost_mwh_this_step,
        }

    def recovery_efficiency(self) -> float:
        # Plain energy round trip: all discharged energy counts,
        # regardless of delivery temperature. See also useful_recovery_efficiency().
        if self.total_charged_mwh < 1e-6:
            return 0.0
        return self.total_discharged_mwh / self.total_charged_mwh

    def useful_recovery_efficiency(self) -> float:
        # Quality-corrected: only discharge >= T_netwerk_setpoint counts
        # fully, with linear interpolation between T_koud_retour and
        # T_netwerk_setpoint for partly usable discharge.
        if self.total_charged_mwh < 1e-6:
            return 0.0
        return self.total_useful_mwh / self.total_charged_mwh

    def soc(self) -> float:
        cfg = self.cfg
        E_huidig = sum(self.m_layer[i] * cfg.cp_water * (self.T[i] - cfg.T_koud_retour)
                       for i in range(self.n)) / 3.6e9  # MWh
        E_max = sum(self.m_layer[i] * cfg.cp_water * (cfg.T_netwerk_setpoint - cfg.T_koud_retour)
                    for i in range(self.n)) / 3.6e9  # MWh
        return max(0.0, min(1.0, E_huidig / E_max)) if E_max > 0 else 0.0
