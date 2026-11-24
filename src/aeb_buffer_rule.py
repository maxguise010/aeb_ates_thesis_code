"""Optional rule for the three existing AEB buffers in the dispatch models that copies how AEB runs them (charging in
the 10 cheapest hours of each day)."""
import numpy as np
import pandas as pd

from config import CONST

N_CHARGE_HOURS_PER_DAY = 10           # notebook cell 45: N_CHEAPEST_HOURS_PER_DAY
MIN_VOL_FLOOR_FRAC = 1200.0 / 3600.0  # notebook cell 39: tank_min_volume_floor_m3 / tank_volume_max_m3
MIN_VOL_CAP_FRAC = 2400.0 / 3600.0    # notebook cell 39: tank_min_volume_cap_m3 / tank_volume_max_m3
RULES = ("rank", "aeb_demand")


def check_rule(rule):
    if rule not in RULES:
        raise ValueError(f"existing_buffer_rule must be one of {RULES}, got {rule!r}")


class AebBufferRule:
    def __init__(self, sim_df, small_volume_max, price_col):
        dates = pd.Series(sim_df.index.date, index=sim_df.index)
        rank = sim_df.groupby(dates.values)[price_col].rank(method="first", ascending=True)
        charge = (rank <= N_CHARGE_HOURS_PER_DAY).values
        demand = sim_df["heat_demand_mw"].values
        nc = pd.Series(np.where(charge, 0.0, demand), index=sim_df.index).groupby(dates.values).sum()
        days = list(nc.index)
        forecast = {d: float(nc[days[k - 1]] if k > 0 else nc[d]) for k, d in enumerate(days)}
        self.charge = charge
        self.target = dates.map(forecast).values.astype(float)
        # remaining charging hours of the same day from hour t onwards (including t)
        rev = pd.Series(charge.astype(int)[::-1], index=sim_df.index[::-1])
        self.remaining = rev.groupby(dates.values[::-1]).cumsum().values[::-1]
        dt = sim_df["delta_t_c"].clip(lower=1e-9)
        vreq = sim_df["heat_demand_mw"] / (CONST * dt)
        nxt2 = vreq.shift(-1).fillna(0.0) + vreq.shift(-2).fillna(0.0)
        self.min_vol = np.clip(nxt2.values, MIN_VOL_FLOOR_FRAC * small_volume_max,
                               MIN_VOL_CAP_FRAC * small_volume_max)
        self.dates = dates.values
        self.v_max = small_volume_max
        self._day = None
        self._achieved = 0.0
        self._target = 0.0

    def hour(self, i):
        """Buffer mode and minimum volume for hour i (call once per hour, before the dispatch)."""
        if self.dates[i] != self._day:
            self._day = self.dates[i]
            self._target = self.target[i]
            self._achieved = 0.0
        return ("charge" if self.charge[i] else "discharge"), float(self.min_vol[i])

    def feasible_charge(self, i, small_volume, delta_t, charge_max_mw):
        """Requested charge (MW) for a charging hour: remaining target / remaining charging hours."""
        rem_h = int(self.remaining[i])
        requested = max(0.0, self._target - self._achieved) / rem_h if rem_h > 0 else 0.0
        room = max(0.0, self.v_max - small_volume) * CONST * delta_t
        return min(requested, charge_max_mw, room)

    def record(self, small_charge_mw):
        self._achieved += small_charge_mw
