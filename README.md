# Code for my master thesis: heat storage for AEB and Westpoort Warmte

This repository holds the Python code and the derived data behind the numbers, tables and figures of my master thesis at TU Delft, *Thermal Energy Storage for Optimising Heat and Power Production under Time-Varying Electricity Prices* (November 2026). The thesis looks at a stratified tank and at high-temperature aquifer storage (HT-ATES) for the AEB waste-to-energy plant and the Westpoort Warmte heat network in Amsterdam.

## Quick start

```
pip install -r requirements.txt     # tested with Python 3.13.9 on Windows 10
python run_all.py                    # takes about 80 seconds
```

`run_all.py` rebuilds the numbers, tables and figure data of the thesis from the derived results in `data/` and compares them with the values printed in the thesis. It prints one line per group and exits with code 0 if everything agrees. What it checks:

- 556 locked numbers (all equal)
- 50 tables (all equal)
- 9 of the 10 figure data sets (Figure 3.1 needs the raw data)
- the appendix sensitivity analyses and the numbers that the thesis derives by arithmetic in Chapters 3 and 4
- a quick import test of the model code

Everything it writes goes to `outputs/`, which is created on the first run.

## What is where

| Folder or file | Content |
|---|---|
| `run_all.py` | The one entry point. `python run_all.py --step <module>` runs a single module of `src/`. |
| `config.py` | All paths and parameters. |
| `src/` | The code (see below). |
| `data/` | Derived inputs: results of the dispatch runs, weekly aggregates, and the ROSIM output of the final runs (`data/rosim/`). |
| `reference/` | The thesis values that `run_all.py` compares against. |
| `results/` | My hand checks of the numbers (`hand_checks.md`) and the log of the perfect-foresight bound. |

## The code in `src/`

Each module starts with a short note on what it does.

- **Chain run by `run_all.py`**: the scenario and NPV calculations (`scenario_runner.py`, `scenarios.py`, `final_numbers.py`, `final_config.py`, `tank_capex.py`), the tank runs (`run_ttes_low_temp.py`), the ROSIM readers and checks (`rosim_check.py`, `audit.py`, `build_rosim_inputs.py`), the f_corr calibration (`fcorr_runs.py`, `bstd_fcorr_v3.py`), the sensitivity analyses, the table and figure builders (`report_*.py`) and the checks (`verify_*.py`).
- **Model code that needs the raw data**: the dispatch models (`simulation_v7.py`, `simulation_ates_option_a_v7.py`, `simulation_ates_three_loops_v7.py`), the thermal ATES model (`ates_thermal_v2.py`), the data pipeline (`data_pipeline.py`) and the tank and buffer models. It is included so the method can be read. It can't run here, because the raw data is not in the repository.
- **Scripts included as written in the project**: `ttes_perfect_foresight_bound_relaxed.py` (perfect-foresight upper bound of the tank) and `run_improved_dispatch.py` (day-ahead and simple-forecast test). They still use the imports of my project layout, so they don't run in this package. `run_improved_dispatch.py` also uses an older ROSIM factor, so it documents the method and not the final numbers.

## What is not reproduced here

- The dispatch runs themselves: they need the hourly AEB data and the EPEX prices. Their results are in `data/run_caches.jsonl`, which the chain reads.
- Figures 3.1 to 3.3, because they show confidential AEB data and the EPEX price series.
- The out-of-sample test, the electricity use of the heat pump and the perfect-foresight bound: the results are included as data (`data/extra_results.json`, `data/nw_heat_pump_allweeks.csv`, `data/step3_heat_pump.csv`, `data/ttes_perfect_foresight_bound_relaxed.csv`).
- The day-ahead and simple-forecast test of Option A: only the script is included (src/run_improved_dispatch.py); the result is reported in Appendix A of the thesis.
- ROSIM itself and the ROSIM projects. The ROSIM output that the code reads is in `data/rosim/`.

## Data and licence

The repository contains derived data only (yearly and weekly aggregates and the ROSIM output). There is no raw AEB data, no EPEX price series and no hourly data. AEB and EPEX gave me permission to publish the derived data.

The code is released under the MIT licence, see `LICENSE`. The licence applies to what is in this repository. It does not cover the raw data of AEB or the EPEX price series, which are not part of it.

## Use of generative AI

During this thesis, generative AI tools (mainly Claude, by Anthropic) were used as a supporting aid. For the text, AI was used to refine language and academic style, give feedback on structure, point out redundant information, and challenge the author's reasoning where arguments were incomplete. For the code, AI mainly helped with debugging Python code in VS Code, keeping an overview of tasks and code functionality, identifying the extensions and packages needed for the simulations, and improving the layout and clarity of generated figures. All AI output was critically reviewed by the author and adapted or rejected where appropriate. The research design, model development, data analysis, interpretation of results, and conclusions are the author's own work, and the author takes full responsibility for the content of this thesis.

The hand checks of the numbers are documented in `results/hand_checks.md`.
