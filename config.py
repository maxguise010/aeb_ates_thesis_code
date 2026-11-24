"""Central settings: all paths (relative to the package folder) and the constants of the dispatch model. Everything
the scripts write goes to outputs/; paths such as results/... in the comments of the scripts are relative to outputs/.
The raw AEB data are not included: put them in data/raw/ or point AEB_RAW_DATA_DIR to their folder; only the model
code needs them."""
import os
import sys
from pathlib import Path

import pandas as pd

# Package paths (the only place where paths are set)
REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(1, str(SRC))
DATA = REPO / "data"
OUTPUT = REPO / "outputs"       # everything the scripts write (created on the first run; not part of the package)
RESULTS = OUTPUT / "results"
FIGURES = OUTPUT / "figures"
WORK = OUTPUT / "work"          # working folders of the model runs and of report_assets.py
CACHE = OUTPUT / "cache"        # the unpacked caches of data/run_caches.jsonl (src/data_store.py)

# Raw AEB data and price series (not included; only the model code needs them)
RAW = Path(os.environ.get("AEB_RAW_DATA_DIR", str(DATA / "raw")))
DATA_DIR = str(RAW)             # name used by data_pipeline.py and forecast_prices.py

# Derived inputs that are included: the caches of the dispatch runs (one line per run in data/run_caches.jsonl,
# unpacked below to outputs/cache/<kind>/) and small result tables of steps that need the raw data.
DERIVED = DATA
RUN_CACHES = DATA / "run_caches.jsonl"
SCENARIO_CACHE = CACHE / "scenario_runner_cache"              # scenario_runner.py, thesis setting, six scenarios
SAC_CACHE = CACHE / "scenario_runs_cache"                     # dispatch runs of the eight scenarios incl. price level
SCENARIO_RESULTS_THESIS = DATA / "scenario_results_thesis_fcorr.csv"   # scenario_runner output with the thesis f_corr
AST_CACHE = CACHE / "ates_same_treatment_cache"               # ATES runs with calibrated buffers, T_cover = 3 h
AST_RELEASE_STATS = DATA / "ates_same_treatment_release_stats.csv"
FCORR_EVALS = CACHE / "fcorr_evals"                           # fcorr_runs.py, one JSON per calibration run
EVALS_BSTD = CACHE / "fcorr_evals_bstd"                       # bstd_fcorr_v3.py
EVALS_WELLLIMIT = CACHE / "fcorr_evals_well_limit"            # well_limit.py search
TCOVER_CACHE = CACHE / "tcover_cache"                         # T_cover runs
TTES_LOW_TEMP_CACHE = CACHE / "ttes_low_temp_cache"           # run_ttes_low_temp.py
BUFFER_USE_CACHE = CACHE / "buffer_use_cache"                 # buffer-use runs (Table B.8, Figure 7.2)
FCORR_VALIDATION = CACHE / "fcorr_validation"                 # evaluation CSVs of the thesis f_corr runs
RESTART_SPLIT = CACHE / "restart_split"                       # restart_split.py (s_flex_shares.csv)
STANDBY_BRIDGING_SUMMARY = DATA / "standby_bridging_summary.csv"
PART_E_EFFICIENCY = DATA / "rosim_efficiency_per_year.csv"          # efficiency targets of the f_corr search (targets.py)
# hourly dispatch: not included. The chain uses its exact weekly aggregate; the model layer writes the hourly frames here
DISPATCH_HOURLY = RAW / "dispatch_hourly"
DISPATCH_WEEKLY = DATA / "dispatch_weekly.csv"                # weekly sums, maxima and hour counts (data_store.dispatch_weekly)
# results of steps that need the raw AEB data or the hourly EPEX prices (aggregated; included as inputs)
RES_TTES_SWEEP = DATA                                         # ttes_price_input_test(_relaxed).csv (tank sweep)
EPEX_STATS = DATA / "epex_stats.csv"                          # yearly price statistics (Table 3.6)
FIG63_STATS = DATA / "charge_discharge_power_stats.csv"                     # power-distribution statistics (Table 6.12)
STEP3_HEAT_PUMP = DATA / "step3_heat_pump.csv"                # heat pump of B-std, step 3.2 (hourly EPEX prices)
SOC_WEEKLY = DATA / "soc_weekly.csv"                          # Figure 6.2: weekly state of charge of Option A 4D
POWER_BINS = DATA / "power_distribution_bins.csv"             # Figure 6.3: histogram bins of the hourly power
EXTRA_RESULTS = DATA / "extra_results.json"                   # results of the extra analyses (out-of-sample, mixing, ...)

# ROSIM output of the final runs (consolidated CSV: data/rosim/<run code>.csv and settings.csv; no AEB data)
ROSIM_PROJECTS = DATA / "rosim"
ROSIM_SCHEDULES = RAW / "rosim_schedules"                     # schedule workbooks: not included (rows = flow profiles in data/rosim/)
ROSIM_FINAL_COPY = DATA / "rosim_final_copy"                  # optional second copy of ATESEfficiency.txt (not included)

# Optional thesis text, only used to look up where an old value is printed (not included)
THESIS_TEXT = DATA / "private" / "thesis_text.txt"
THESIS_TEX = DATA / "private" / "H5_ttes.tex"

# Reference results (included): what the checks compare against
RES_REFERENCE = REPO / "reference"
REF_LOCKED = RES_REFERENCE / "locked_numbers.csv"             # the locked numbers (id, value, unit, thesis_location, source)
REF_TABLES = RES_REFERENCE / "tables_reference.csv"           # every thesis table, long form (data_store.to_long)
REF_FIGURES_DATA = RES_REFERENCE / "figures_data_reference.csv"   # every figure data set (not Figures 3.1-3.3), long form
REF_SENSITIVITY = RES_REFERENCE / "sensitivity_reference.csv"     # result files of the appendix sensitivity analyses
REF_DERIVED = RES_REFERENCE / "derived_numbers_reference.csv"     # derived numbers of Chapters 3 and 4 as printed in the thesis

# Outputs (all under outputs/)
RES_SCENARIO_RUNNER = RESULTS / "scenario_runner"
RES_TTES = RESULTS / "ttes_low_temp"
RES_ROSIM = RESULTS / "rosim"
RES_FCORR = RESULTS / "fcorr"
RES_FINAL = RESULTS / "final_numbers"
RES_SCEN = RESULTS / "scenarios"

# Sensitivity analyses of the appendix (not part of the headline numbers)
RES_SENS = RESULTS / "sensitivity"                            # pit_check/, temp_sensitivity/, bstd_return_70/
PIT_RERUN_CACHE = CACHE / "pit_rerun_cache"                   # pit runs with the current model
TEMP_SENS_RUNS = CACHE / "temp_sensitivity_runs"              # derated-power dispatch runs (JSON)
NW_HEAT_PUMP_THESIS = DATA / "nw_heat_pump_allweeks.csv"      # source of the 54 k EUR/yr heat-pump electricity (Table 7.4)

# Option B standard: return (cold-well) temperature
# The ONE parameter for the return temperature of B-std. 58 C = headline (mean return in the AEB data, window 90/58 C =
# 32 K); every script reads this value: the dispatch configuration (scenarios_v4.py), the ROSIM runs (rosim_check.RUNS),
# f_corr, the well limit, the thermal radius and the file names of the caches.
# Only src/bstd_return_70.py sets it to 70 C (sensitivity, output in results/sensitivity/bstd_return_70/).
BSTD_T_RETURN_C = 58.0
BSTD_T_INJ_C = 90.0
RES_BSTD_RETURN_70 = RES_SENS / "bstd_return_70"              # output of the return-temperature sensitivity (70 C)


def bstd_tag():
    """Suffix of the cache files of B-std runs: '' for the 70 C runs (their original file names), '_T58' at 58 C."""
    return "" if BSTD_T_RETURN_C == 70.0 else f"_T{BSTD_T_RETURN_C:g}"


def tagged(option):
    """Name of an option in cache file names: 'Bstd' + bstd_tag() for B-std, the option itself otherwise."""
    return option + bstd_tag() if option == "Bstd" else option


def restart_shares():
    """Bridged share of S_flex per option (restart_split.py) at the configured B-std return temperature."""
    return RESTART_SPLIT / f"s_flex_shares{bstd_tag()}.csv"


# Work folders of the model runs (need the raw AEB data; never part of the package)
FRAMES_AST = WORK / "ates_same_treatment_frames"
FRAMES_RESTART = WORK / "restart_split_frames"
SHORT_WINDOW = WORK / "ttes_short_window"
MODEL_OUT = WORK / "model_outputs"
TEMP_SENS_HOURLY = WORK / "temp_sensitivity_hourly"           # hourly B-std discharge + EPEX price (model layer)

for _d in (RES_SCENARIO_RUNNER, RES_TTES, RES_ROSIM, RES_FCORR, RES_FINAL, RES_SCEN, FIGURES, RES_SENS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# unpack the caches of data/run_caches.jsonl to outputs/cache/<kind>/ (only when the content changed)
import data_store as _data_store  # noqa: E402

_data_store.unpack(RUN_CACHES, CACHE)

# Simulation window
SIM_START = pd.Timestamp(2023, 1, 1, 0, 0)    # first hour of the simulation window
SIM_END   = pd.Timestamp(2025, 12, 31, 23, 0)  # last hour of the simulation window

# Physical constants
CONST    = 0.001157   # MW = CONST x delta T (K) x flow (m3/h)
ZERO_TOL = 1e-9
MIN_PHYSICAL_DELTA_T = 10.0   # Kelvin - guard against near-zero delta T in loops

# BEC internal own-use
bec_own_use_mw = 1.0

# Default small-tank piping limits
tank_charge_max_mw    = 50.0
tank_discharge_max_mw = 50.0

# Heat-exchanger dispatch priority
priority_order = ["WWBEC", "WW10", "WW30", "WW01", "WW00", "WW02"]

# HX parameterisation
hx_params = {
    "WWBEC": {
        "type": "fixed",
        "min_mw": 17,
        "max_mw": 25,
        "ramp_up_mwph": 25,
        "ramp_down_mwph": 25,
    },
    "WW10": {
        "type": "dynamic",
        "min_m3h": 100,
        "max_m3h": 850,
        "max_mw_const": 62,
        "ramp_up_mwph": 62,
        "ramp_down_mwph": 62,
    },
    "WW30": {
        "type": "dynamic",
        "min_m3h": 14,
        "max_m3h": 377,
        "max_mw_const": 24,
        "ramp_up_mwph": 24,
        "ramp_down_mwph": 24,
    },
    "WW01": {
        "type": "special",
        "min_m3h": 50,
        "max_mw_const": 30,
        "ramp_up_mwph": 30,
        "ramp_down_mwph": 30,
    },
    "WW00": {
        "type": "dynamic",
        "min_m3h": 85,
        "max_m3h": 850,
        "max_mw_const": 50,
        "ramp_up_mwph": 50,
        "ramp_down_mwph": 50,
    },
    "WW02": {
        "type": "dynamic",
        "min_m3h": 85,
        "max_m3h": 1000,
        "max_mw_const": 56,
        "ramp_up_mwph": 56,
        "ramp_down_mwph": 56,
    },
}

# Opportunity-cost factors of the electricity not produced ("derving")
# MWh_electricity lost per MWh_heat supplied. WWBEC excluded from cost totals.
derivation_factors = {
    "WW00":          0.3024,
    "WW01":          0.3024,
    "WW02":          0.3024,
    "WW10":          0.20232,
    "WW30_reductie": 0.486,
    "WW30_aftap":    0.2862,
    "WWBEC":         1.0,
}

# Mapping: simulation names -> historical column names in df_prod
historical_cols = {
    "WWBEC": "WW_BEC",
    "WW10":  "WW_10",
    "WW30":  "WW_30",
    "WW01":  "WW_01",
    "WW00":  "WW_00",
    "WW02":  "WW_02",
}

# Heat price ("warmteprijs") tariff coefficients
gamma_basis = 0.05708
gamma_volg  = 0.07227
gamma_piek  = 0.10495

