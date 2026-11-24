"""Perfect-foresight upper bound (one LP per window) on the extra value of a large tank on the relaxed schedule. Needs
the raw data, not run by run_all.py."""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from config import SIM_START, derivation_factors
from pipeline.data_pipeline import build_sim_df_tank
from models.ptes_loss import ptes_loss_mw
from post_midterm.models.simulation_v7 import run_tank_simulation_v5
import inspect
import pandas as pd
from post_midterm.models import simulation_v7 as v7
from post_midterm.models.ttes_thermal_replay import ttes_model, replay


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "ttes_perfect_foresight_bound_relaxed_output.log")
OUT_CSV = os.path.join(HERE, "..", "analysis", "ttes_perfect_foresight_bound_relaxed.csv")
assert not os.path.exists(LOG_PATH) and not os.path.exists(OUT_CSV), "output exists -- refusing to overwrite"
_log = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log)
T_START = time.time()

YEARS = 3.0
RATE_MW = 50.0
CASES = {5_000: dict(eta=0.987, breakeven=119_000), 19_000: dict(eta=0.963, breakeven=451_000)}
CAP_MWH_PER_M3 = 950 * 4220 * 52 / 3.6e9   # MWh per m³ (110 → 58 °C)
STANDBY_ON = False      # relaxed schedule
MIN_DOWNTIME_H = 2      # relaxed schedule

print("Loading data...")
(sim, _t, df1, _a, _b, _c, _d, cost_hist) = build_sim_df_tank()
soc0 = df1["Bufferinhoud MWh"].loc[df1.index >= SIM_START].dropna().iloc[0] / 250.0
print(f"Schedule: relaxed (min downtime {MIN_DOWNTIME_H} h, standby {'on' if STANDBY_ON else 'off'})")
t0 = time.time()
df = run_tank_simulation_v5(sim, large_tank_volume_m3=1.0, enable_pure_charge_large=False, small_initial_soc=soc0,
                            demand_correction_factor=1.038, enable_bec_summer_cap=True, enable_min_runtime=True,
                            min_runtime_hours=18, bec_fixed_historical=False, enable_ww_standby_correction=STANDBY_ON,
                            min_downtime_hours=MIN_DOWNTIME_H)
ZERO_REF = (cost_hist - df['total_cost_eur_corrected'].sum()) / YEARS
print(f"  No-tank relaxed run: €{ZERO_REF:,.0f}/yr (expected €1,849,652 ± 10) [{time.time() - t0:.0f}s]")
CHECKS = [("zero-capacity reference", ZERO_REF, 1_849_652)]

# ── Reproduction check: threshold rule (actual P20/P80), as ttes_price_input_test.py --schedule relaxed ──
PATCHES = [
    ('''    demand_correction_factor: float = 1.0,  # default = no correction; 1.025 = +2.5%
):''',
     '''    demand_correction_factor: float = 1.0,  # default = no correction; 1.025 = +2.5%
    large_price_col: str = 'voorspelling_prijs',
):'''),
    ("""    dispatch_price_col = 'voorspelling_prijs'   # for planning""",
     """    dispatch_price_col = large_price_col   # for planning (patched: selectable)"""),
]
_src = inspect.getsource(v7.run_tank_simulation_v5)
for old, new in PATCHES:
    assert _src.count(old) == 1, f"patch anchor found {_src.count(old)}×:\n{old}"
    _src = _src.replace(old, new, 1)
_src = _src.replace("def run_tank_simulation_v5(", "def run_tank_simulation_pc(", 1)
_ns = dict(vars(v7))
exec(compile(_src, "<pf_relaxed patched>", "exec"), _ns)
run_pc = _ns["run_tank_simulation_pc"]
CON = dict(small_initial_soc=soc0, demand_correction_factor=1.038, enable_bec_summer_cap=True,
           enable_min_runtime=True, min_runtime_hours=18, bec_fixed_historical=False,
           enable_ww_standby_correction=STANDBY_ON, min_downtime_hours=MIN_DOWNTIME_H)
NON_BEC_COLS = ["WW10_sim_tank_mw", "WW30_sim_tank_mw", "WW01_sim_tank_mw", "WW00_sim_tank_mw", "WW02_sim_tank_mw"]
_ACT = 'Prijs_Gemiddelde_EPEX_Spot'
t0 = time.time()
ref_chk = (cost_hist - run_pc(sim, large_tank_volume_m3=1.0, enable_pure_charge_large=False, annual_charge_pct=20,
                              annual_discharge_pct=80, large_price_col=_ACT, **CON)["total_cost_eur_corrected"].sum()) / YEARS
print(f"  Check: no tank, actual, P20/P80: €{ref_chk:,.0f}/yr [{time.time() - t0:.0f}s]")
for v, expected in ((5_000, 51_345), (19_000, 128_940)):
    t0 = time.time()
    ss = 16 if v < 10_000 else 4
    dfc = run_pc(sim, large_tank_volume_m3=float(v), large_charge_max_mw=50.0, large_discharge_max_mw=50.0,
                 annual_charge_pct=20, annual_discharge_pct=80, large_price_col=_ACT, **CON)
    ch_c = dfc["large_tank_charge_mw"].values
    dis_c = dfc["large_tank_discharge_mw"].values
    r_c = replay(ttes_model(v, h_over_d=0.6), ch_c - dis_c, substeps=ss)
    non_bec = sum(dfc[c].sum() for c in NON_BEC_COLS)
    avg_derving = dfc["derving_cost_eur"].sum() / non_bec if non_bec > 0 else 0.0
    shortfall = max(0.0, dis_c.sum() - r_c['useful'].sum()) / YEARS
    sav = (cost_hist - dfc["total_cost_eur_corrected"].sum()) / YEARS - shortfall * avg_derving
    print(f"  Check: actual P20/P80 {v:>6,} m³: incr €{sav - ref_chk:+,.0f}/yr (expected €{expected:+,}) "
          f"[{time.time() - t0:.0f}s]")
    CHECKS.append((f"threshold rule {v:,} m3", sav - ref_chk, expected))
_fail = [c for c in CHECKS if abs(c[1] - c[2]) > 10]
if _fail:
    print("\nREPRODUCTION CHECK FAILED -- stopping, bound not computed:")
    for name, got, exp in _fail:
        print(f"  {name}: got {got:,.0f}, expected {exp:,} (diff {got - exp:+,.0f})")
    sys.exit(1)
print("  Reproduction check passed (all within ±10 EUR/yr).")

price = sim["Prijs_Gemiddelde_EPEX_Spot"].values
n = len(sim)
aftap_cap = sim["WW30_aftap_cap_mw"].values
ww30_out = df["WW30_sim_tank_mw"].values
ww30_max = sim["WW30_max_mw"].values
ww30_av = sim["WW30_hist_available"].values == 1
# Forced minimums the tank may NOT displace (the constrained schedule's own rules):
# WW30/WW01 standby minimum when historically active, and min-runtime forced-on
# (WW10/WW30/WW01) -> output can only be reduced to min_mw. Min-downtime forced-off
# units have no spare capacity.
def floor_of(hx, out):
    forced = np.zeros(n, dtype=bool)
    if f"{hx}_minruntime_forced_on_flag" in df:
        forced |= df[f"{hx}_minruntime_forced_on_flag"].values == 1
    if hx in ("WW30", "WW01") and STANDBY_ON:   # relaxed: no standby rule -> no standby floor
        forced |= df[f"{hx.lower()}_hist_active"].values.astype(bool)
    return np.where(forced & (out > 1e-6), np.minimum(out, sim[f"{hx}_min_mw"].values), 0.0)


def spare_of(hx, out, cap):
    av = sim[f"{hx}_hist_available"].values == 1
    if f"{hx}_minruntime_forced_off_flag" in df:
        av &= df[f"{hx}_minruntime_forced_off_flag"].values != 1
    return np.where(av, np.clip(cap - out, 0, None), 0.0)


# pseudo-units: (name, factor, displaceable output, spare capacity)
units = []
for hx in ["WW10", "WW01", "WW00", "WW02"]:
    out = df[f"{hx}_sim_tank_mw"].values
    units.append((hx, derivation_factors[hx], np.clip(out - floor_of(hx, out), 0, None),
                  spare_of(hx, out, sim[f"{hx}_max_mw"].values)))
a_out = np.minimum(ww30_out, aftap_cap)
r_out = np.clip(ww30_out - aftap_cap, 0, None)
a_max = np.minimum(ww30_max, aftap_cap)
r_max = np.clip(ww30_max - aftap_cap, 0, None)
disp30 = np.clip(ww30_out - floor_of("WW30", ww30_out), 0, None)
r_disp = np.minimum(r_out, disp30)
units.append(("WW30_aftap", derivation_factors["WW30_aftap"], disp30 - r_disp, spare_of("WW30", a_out, a_max)))
units.append(("WW30_reductie", derivation_factors["WW30_reductie"], r_disp, spare_of("WW30", r_out, r_max)))
U = len(units)
F = np.array([u[1] for u in units])
OUT = np.nan_to_num(np.stack([u[2] for u in units], axis=1))     # n × U
SPARE = np.nan_to_num(np.stack([u[3] for u in units], axis=1))
price = np.nan_to_num(price)
print(f"  Units: " + ", ".join(f"{u[0]} f={u[1]}" for u in units))


def solve_window(sl, cap_mwh, eta):
    T = sl.stop - sl.start
    p = price[sl]
    nq = T * U
    nv = 2 * nq + T
    c = np.concatenate([(F[None, :] * p[:, None]).ravel(), -(F[None, :] * p[:, None]).ravel(), np.zeros(T)])
    A_ub = lil_matrix((T, nv))
    for t in range(T):
        A_ub[t, t * U:(t + 1) * U] = 1
        A_ub[t, nq + t * U:nq + (t + 1) * U] = 1
    A_eq = lil_matrix((T, nv))
    for t in range(T):
        A_eq[t, 2 * nq + t] = 1
        if t > 0:
            A_eq[t, 2 * nq + t - 1] = -1
        A_eq[t, t * U:(t + 1) * U] = -1
        A_eq[t, nq + t * U:nq + (t + 1) * U] = 1 / eta
    bounds = ([(0, s) for s in SPARE[sl].ravel()] + [(0, o) for o in OUT[sl].ravel()]
              + [(0, cap_mwh)] * (T - 1) + [(0, 0)])
    r = linprog(c, A_ub=A_ub.tocsr(), b_ub=np.full(T, RATE_MW), A_eq=A_eq.tocsr(), b_eq=np.zeros(T),
                bounds=bounds, method="highs")
    if r.status != 0:
        return 0.0, 0.0, 0.0, 0.0
    q = r.x[:nq].reshape(T, U)
    d = r.x[nq:2 * nq].reshape(T, U)
    qs, ds = q.sum(axis=1), d.sum(axis=1)
    return -r.fun, qs.sum(), ds.sum(), (p * ds).sum() - (p * qs).sum()


def windows(hours):
    if hours == 24:
        days = sim.index.normalize()
        edges = np.flatnonzero(np.r_[True, days[1:] != days[:-1], True])
    else:
        edges = np.r_[np.arange(0, n, hours), n]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a > 1]


print("\nPERFECT-FORESIGHT UPPER BOUND (relaxed schedule, 50 MW, after the 3 × 1,200 m³ buffers)")
print("=" * 118)
print(f"  {'Volume':>8}{'Capacity':>10}{'η':>7}{'Horizon':>9}{'Bound €/yr':>13}{'Charged':>10}{'Discharged':>12}"
      f"{'Cycles':>8}{'EPEX spread':>13}{'Standing loss':>15}{'Break-even':>12}  Verdict")
print(f"  {'m³':>8}{'MWh':>10}{'':>7}{'':>9}{'':>13}{'MWh/yr':>10}{'MWh/yr':>12}{'/yr':>8}{'€/MWh_e':>13}"
      f"{'MWh/yr':>15}{'€/yr':>12}")
print("  " + "-" * 116)
results = {}
rows = []
N_DAYS = len(np.unique(sim.index.normalize()))
for v, cfg in CASES.items():
    cap = v * CAP_MWH_PER_M3
    for hours, label in ((24, "daily"), (168, "7-day")):
        t0 = time.time()
        tot = np.array([solve_window(w, cap, cfg['eta']) for w in windows(hours)]).sum(axis=0)
        val, qch, qdis, sp = tot / YEARS
        rows.append(dict(schedule="relaxed (2 h, standby off)", volume_m3=v, capacity_mwh=cap, eta=cfg['eta'],
                         horizon=label, bound_eur_yr=val, charged_mwh_yr=qch, discharged_mwh_yr=qdis,
                         charged_mwh_per_day=tot[1] / N_DAYS, cycles_per_yr=qch / cap,
                         epex_spread_eur_mwh=(sp / qdis if qdis > 0 else 0.0),
                         standing_loss_mwh_yr=ptes_loss_mw(v, 90.0) * 8760, breakeven_eur_yr=cfg['breakeven'],
                         zero_ref_eur_yr=ZERO_REF))
        spread = sp / qdis if qdis > 0 else 0.0
        loss = ptes_loss_mw(v, 90.0) * 8760
        verdict = "BELOW break-even" if val < cfg['breakeven'] else "above break-even"
        results[(v, label)] = val
        print(f"  {v:>8,}{cap:>10,.0f}{cfg['eta']:>7.3f}{label:>9}{val:>13,.0f}{qch:>10,.0f}{qdis:>12,.0f}"
              f"{qch / cap:>8.0f}{spread:>13.1f}{loss:>15,.0f}{cfg['breakeven']:>12,}  {verdict}  [{time.time() - t0:.0f}s]")
res = pd.DataFrame(rows)
res.to_csv(OUT_CSV, index=False)
print("\n  Charged heat per day (MWh/day, over all " + f"{N_DAYS} days):")
for _, s in res.iterrows():
    print(f"    {s['volume_m3']:>6,} m³ {s['horizon']:>6}: {s['charged_mwh_per_day']:,.1f} MWh/day")
print("\n  Bound = value relative to the relaxed no-tank run (same basis as the increments).")
print("  Standing loss = the dispatch model's ptes_loss_mw at 90 °C (not subtracted from the bound).")
if all(results[(v, lab)] < CASES[v]['breakeven'] for v in CASES for lab in ("daily", "7-day")):
    print("\n  All bounds are below break-even -> step (2) not run, as instructed.")
print(f"\nSaved: {os.path.abspath(OUT_CSV)}\nLog saved: {LOG_PATH}\nTotal runtime: {time.time() - T_START:.0f}s")
