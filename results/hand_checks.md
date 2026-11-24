# Hand checks

The numbers of the repository were checked against the ROSIM outputs and by hand. Each check below can be repeated
with a calculator from the values in `outputs/results/` (written by `run_all.py`).

## 1. Annuity factor and capital recovery factor (`src/final_config.py`, asserted in `src/final_numbers.py`)

f_PV = (1 − 1.05^−30) / 0.05 = (1 − 0.231377) / 0.05 = **15.3725**; CRF = 1 / 15.3725 = **0.06505**.
`final_numbers.py` stops if either differs (assert at the end of `main()`).

## 2. Discounted LCOH of Option A 1D (`outputs/results/final_numbers/step1_lcoh_basisR.csv`)

CAPEX 10.5 M EUR, OPEX 175 k EUR/yr, discharged heat 4,251.38 MWh/yr (calibrated dispatch, year 7):

(10,500,000 × 0.0650514 + 175,000) / 4,251.38 = (683,040 + 175,000) / 4,251.38 = **201.83 EUR/MWh** — equal to the
locked value (202 EUR/MWh rounded).

## 3. Scenario NPV, Option A 1D at price spread ×1.25 (`outputs/results/scenarios/scenarios_old_vs_new.csv`)

S_flex = 780.32 k EUR/yr, S_stor = 246.15 k EUR/yr (`scenario_decomposition.csv`), f_corr = 0.5670 (basis R):

S_attr = 780.32 + 0.5670 × 246.15 = 919.90; NPV = −10.5 + (919.90 − 175) × 15.3725 / 1000 = **+0.951 M EUR** — equal to
the table.

## 4. ROSIM efficiencies against the ROSIM output (`outputs/results/rosim/rosim_check.csv`)

`rosim_check.py` recomputes the yearly efficiency from the weekly energies of the ROSIM output in `data/rosim/` (positive / negative
weekly energy over the 52 weeks that end at week 363 for year 7) and compares it with the value ROSIM reports and with
the efficiency table of the final numbers. Year 7:

| run | ROSIM reported | recomputed | cleaned | table | difference (compared value − table) |
|---|---|---|---|---|---|
| A1D | 45.16 | 45.161 | 45.161 | 45.16 | +0.001 |
| A2D | 49.62 | 49.624 | 49.625 | 49.62 | +0.004 |
| A4D | 51.43 | 51.427 | 51.427 | 51.43 | −0.003 |
| B-WN110 | 48.86 | 48.859 | 48.674 | 48.68 | −0.006 (cleaned: weeks 339-340 of the cold well interpolated) |
| B-std WN | 35.57 | 35.566 | 35.566 | 35.57 | −0.004 |
| B-std NW | 36.90 | 36.899 | 36.899 | 36.90 | −0.001 |

The script stops when a year 6-8 value differs by more than 0.02.

## 5. Reproduction of the submitted thesis

- `outputs/results/final_numbers/step0_lcoh_reproduction.csv`: the thesis LCOH (Eq. 7.5, old inputs) reproduced per option.
- `outputs/results/scenarios/scenarios_reproduction_old.csv`: Tables 7.13, B.5 and B.7 with the old inputs against the printed
  values (largest difference 0.017 M EUR).
- `outputs/results/scenarios/step0_check.csv`: the NPVs of `scenario_runner.py` against the locked numbers (all within 0.01).

## 6. Repository against the locked numbers

`run_all.py` (with `src/verify_locked.py`) compares every locked number in `outputs/results/final_numbers/chapter_number_map.csv`
with the frozen copy in `reference/locked_numbers.csv` (tolerance 0.002) and writes `outputs/results/verification.csv`.
