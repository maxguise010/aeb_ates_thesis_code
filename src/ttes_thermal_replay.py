"""Replays a tank dispatch series through the stratified TTES model (N = 20 layers, with sub-steps) to get the usable
discharge heat and the shortfall."""
import numpy as np

from ptes_stratification import PTESStratificationConfig, PTESStratificationModel

# TTES defaults (identical to ttes_final_n20.py)
TTES_DEFAULTS = dict(
    T_hot=110.0, T_cold=58.0, rho_w=950.0, cp_w=4220.0, T_amb=21.0, n_layers=20,
    U_wall=0.20, U_top=0.12, U_bottom=0.20, alpha_turb=0.05, T_min_useful=90.0, T_init=15.0,
)


class CylinderTTESModel(PTESStratificationModel):
    """Same physics as PTESStratificationModel; true cylinder geometry
    (wall pi*D*dz per layer, top/bottom pi*D²/4) instead of a square plan."""
    def __init__(self, config, diameter_m):
        self._D = diameter_m
        super().__init__(config)

    def _calc_geometry(self):
        cfg = self.cfg
        self.dz = cfg.height_m / self.n
        self.V_layer = np.full(self.n, cfg.volume_m3 / self.n)
        self.m_layer = self.V_layer * cfg.rho_water
        A_cross = np.pi * self._D ** 2 / 4
        A_wall_layer = np.pi * self._D * self.dz
        self.UA_verlies = np.zeros(self.n)
        for i in range(self.n):
            UA = A_wall_layer * cfg.U_zijwand
            if i == 0:
                UA += A_cross * cfg.U_dek
            if i == self.n - 1:
                UA += A_cross * cfg.U_bodem
            self.UA_verlies[i] = UA
        self.A_geleiding = A_cross


def cylinder_dims(volume_m3, h_over_d=1 / 3):
    """Diameter and height (m) of a cylinder with the given volume and H/D."""
    D = (4 * volume_m3 / (np.pi * h_over_d)) ** (1 / 3)
    return D, D * h_over_d


def ttes_model(volume_m3, h_over_d=1 / 3, **overrides):
    """Cylinder TTES; default H:D = 1:3."""
    p = {**TTES_DEFAULTS, **overrides}
    D, H = cylinder_dims(volume_m3, h_over_d)
    cfg = PTESStratificationConfig(
        volume_m3=float(volume_m3), height_m=H, n_layers=p['n_layers'], alpha_turb=p['alpha_turb'],
        rho_water=p['rho_w'], cp_water=p['cp_w'],
        U_dek=p['U_top'], U_zijwand=p['U_wall'], U_bodem=p['U_bottom'],
        T_omgeving=p['T_amb'], T_netwerk_setpoint=p['T_min_useful'], T_koud_retour=p['T_cold'],
        T_initieel=p['T_init'],
    )
    return CylinderTTESModel(cfg, D)


def replay(model, power_series, T_inj=110.0, substeps=4, T_supply_min=None):
    """Replay hourly dispatch power (MW, + charge / − discharge) through
    `model`, splitting each hour into `substeps` equal steps.

    Returns hourly arrays (MWh per dispatch hour): 'stored', 'curtailed',
    'useful' (discharge), plus run totals 'loss_mwh' and 'cfl_hours'
    (charge hours in which the flow cap still bound).

    T_supply_min (optional, default None = no change): top-up accounting. In a
    discharge sub-step with outlet temperature T_out < T_supply_min, the heat
    m*cp*(T_supply_min - T_out) needed to lift the (unchanged) discharge flow
    to T_supply_min is added to the extra hourly array 'topup' (MWh_th, to be
    supplied by the plant); 'lowT_frac' is the share of that hour's discharge
    sub-steps with T_out < T_supply_min. The mass flow is the one the model
    uses in step() (same formula and stability cap, ptes_stratification.py
    lines 229-231), evaluated on the profile before the step."""
    n = len(power_series)
    stored = np.zeros(n)
    curtailed = np.zeros(n)
    useful = np.zeros(n)
    topup = np.zeros(n)
    lowT_frac = np.zeros(n)
    cfl_hours = 0
    dt = 1.0 / substeps
    cfg = model.cfg
    m_cap = np.min(model.m_layer) / (dt * 3600) * 0.5
    for t, p in enumerate(power_series):
        p = float(p)
        for _ in range(substeps):
            if T_supply_min is not None and p < -1e-6:
                T_out = float(model.T[0])
                if T_out < T_supply_min:
                    m_flow = min(abs(p) * 1e6 / (cfg.cp_water * max(1.0, T_out - cfg.T_koud_retour)), m_cap)
                    topup[t] += m_flow * cfg.cp_water * (T_supply_min - T_out) * dt * 3600 / 3.6e9
                    lowT_frac[t] += 1.0 / substeps
            r = model.step(power_mw=p, T_injectie=T_inj, dt_hours=dt)
            if p > 1e-6:
                stored[t] += r['actual_charged_mwh']
                curtailed[t] += r['curtailed_mwh']
            elif p < -1e-6:
                useful[t] += r['nuttige_mwh']
        if p > 1e-6 and curtailed[t] > 1e-6:
            cfl_hours += 1
    out = dict(stored=stored, curtailed=curtailed, useful=useful,
               loss_mwh=model.total_loss_mwh, cfl_hours=cfl_hours)
    if T_supply_min is not None:
        out.update(topup=topup, lowT_frac=lowT_frac)
    return out
