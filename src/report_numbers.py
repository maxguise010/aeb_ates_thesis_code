"""Shared calculations for the report figures and tables, with the same functions and final inputs as
final_numbers.py."""
import functools
import json

import numpy as np
import pandas as pd

import config as C
import final_config as FC
import final_numbers as FN
import tank_capex as TC

OPTIONS = ["TTES5k", "A1D", "BWN110", "A2D", "Bstd", "A4D"]          # thesis order of the economic tables
ATES = ["A1D", "A2D", "A4D", "BWN110", "Bstd"]
AF, CRF = FC.annuity_factor(), FC.crf()
BASIS = FC.FCORR_BASIS_HEADLINE


@functools.lru_cache(maxsize=None)
def inputs():
    return FN.load_inputs()


def f_corr(o):
    return 1.0 if o == "TTES5k" else float(inputs()["f"][BASIS][o])


def tiers(o):
    """lower / base / bridged / potential attributed saving (k EUR/yr) as final_numbers.s_attr."""
    return FN.s_attr(inputs(), o, f_corr(o))


def capex(o, case="mid"):
    if o == "TTES5k":
        return TC.CASES[{"low": "B_low", "mid": "B", "high": "B_high"}[case]]["capex_meur"]
    return FC.ATES[o]["capex_meur"][{"low": 0, "mid": 1, "high": 2}[case]]


def opex(o, case="mid"):
    return TC.opex_keur(capex(o, case)) if o == "TTES5k" else FC.ATES[o]["opex_keur"]


def npv(o, tier="base", case="mid", s=None):
    return FN.npv(capex(o, case), opex(o, case), tiers(o)[tier] if s is None else s)


def breakeven(o, tier="base"):
    s = tiers(o)[tier]
    return FN.breakeven(s, "scaled") if o == "TTES5k" else FN.breakeven(s, opex_keur=opex(o))


def storage_before(o):
    """S_stor of the uncalibrated run (k EUR/yr): fcorr_evals scale 1 for ATES, scenario_runner for the tank."""
    I = inputs()
    return float(I["sc"].loc["TTES5k"].S_stor) if o == "TTES5k" else I["base"][o] / 1000


def storage_after(o):
    """= locked 'storage effect after calibration' (f_corr x S_stor of scale 1)."""
    return f_corr(o) * storage_before(o)


def s_ref():
    return float(inputs()["sc"].loc["A1D"].S_ref)


def s_flex():
    return float(inputs()["sc"].loc["A1D"].S_flex)


def lcoh(o):
    I = inputs()
    if o == "TTES5k":
        return FN.lcoh_discounted(capex(o) * 1e6, opex(o) * 1e3, I["q_ttes"])
    return FN.lcoh_discounted(capex(o) * 1e6, opex(o) * 1e3, I["q"][BASIS][o])


def q_disch(o):
    I = inputs()
    return I["q_ttes"] if o == "TTES5k" else I["q"][BASIS][o]


# ---------------------------------------------------------------------------- tank cost at any volume (case B method)
def tank_case_b(volume_m3):
    """(low, mid, high) CAPEX in M EUR of a tank of the given volume with the case-B method: the Duesseldorf and
    Duisburg totals scaled with (V/V_project)^0.691 and escalated with CBS; mid = their mean, low/high = the two."""
    pt = TC.project_table(volume_m3)
    vals = [pt[n]["scaled_escalated_meur"] for n in FC.BASE_PROJECTS]
    return min(vals), sum(vals) / len(vals), max(vals)


def tank_npv(volume_m3, s_attr_keur, case="mid"):
    lo, mid, hi = tank_case_b(volume_m3)
    cx = {"low": lo, "mid": mid, "high": hi}[case]
    return FN.npv(cx, TC.opex_keur(cx), s_attr_keur)


# ---------------------------------------------------------------------------- pump replacement (thesis Appendix B.3)
def pump_replacement_pv_meur(o):
    """Present value of replacing the submersible pumps: n_doublets x pump cost (2020 prices x CBS civil-engineering
    factor) at every PUMP_INTERVAL years within the lifetime, discounted at the real rate (thesis Appendix B.3)."""
    if o == "TTES5k":
        return 0.0
    cost = FC.PUMP_COST_2020_KEUR * FC.CBS_CIVIL_2020_2025
    years = range(FC.PUMP_INTERVAL_YR, FC.LIFETIME, FC.PUMP_INTERVAL_YR)      # 7, 14, 21, 28
    return FC.DOUBLETS[o] * cost * sum((1 + FC.RATE) ** -t for t in years) / 1000


# ---------------------------------------------------------------------------- buffer-use runs (Table B.8)
def _saving(path):
    return json.loads(path.read_text())["saving_keur_yr"]


def buffer_case(case):
    """S_ref, Z, S_flex, S_stor A1D, S_stor TTES (k EUR/yr) of one buffer case; 'n6' = base case (scenario cache)."""
    if case == "n6":
        import scenario_runner as sr
        base = dict(sr.SCENARIOS["base"], name="base", price_file=False)
        g = lambda o, s: _saving(sr._key(base, o, s, ""))
    else:
        d = C.BUFFER_USE_CACHE
        g = lambda o, s: _saving(d / f"{case}__{o}__{s}.json")
    sref, z = g("none_ates", "constrained"), g("none_ates", "relaxed")
    return dict(S_ref=sref, Z=z, S_flex=z - sref, S_stor_A1D=g("A1D", "relaxed") - g("none_ates_a", "relaxed"),
                S_stor_TTES=g("TTES5k", "relaxed") - g("none_tank", "relaxed"))


def buffer_npv(o, case):
    b = buffer_case(case)
    if o == "TTES5k":
        return FN.npv(capex(o), opex(o), b["S_flex"] + b["S_stor_TTES"])
    return FN.npv(capex(o), opex(o), b["S_flex"] + f_corr(o) * b["S_stor_A1D"])


def buffer_cases_available():
    d = C.BUFFER_USE_CACHE
    return d.exists() and len(list(d.glob("*.json"))) >= 30


def simple_payback(o):
    net = tiers(o)["base"] - opex(o)
    return capex(o) * 1000 / net if net > 0 else np.inf


def fmt_signed(v, d=2):
    return (f"{v:+.{d}f}").replace("-", "−")


def fmt_num(v, d=0):
    return f"{v:,.{d}f}".replace("-", "−")


def frame(rows, columns):
    return pd.DataFrame(rows, columns=columns)
