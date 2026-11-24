"""Target efficiencies (years 6, 7 and 8; basis R and D) for the f_corr calibration in fcorr_runs.py."""
import pandas as pd
import config as C  # repository paths (config.py)

E = pd.read_csv(C.PART_E_EFFICIENCY)
YEARS = (6, 7, 8)
RUN = {"A1D": "A1D", "A2D": "A2D", "A4D": "A4D", "BWN110": "B-WN110"}


def eff(run, col):
    return {y: float(E[(E.run == run) & (E.year == y)][col].iloc[0]) for y in YEARS}


EFF = {}
for basis, col in (("R", "rosim_cleaned"), ("D", "dispatch_reference")):
    for cfg, run in RUN.items():
        EFF[(cfg, basis)] = eff(run, col)
    EFF[("WN", basis)] = eff("B-std WN", col)
    EFF[("NW70", basis)] = eff("B-std NW", col)
EFF[("NW125", "R")] = {6: 27.14, 7: 29.12, 8: 30.76}      # supplied by the author, not recomputed (file missing)
EFF[("NW125", "D")] = {6: 30.0, 7: 32.1, 8: 33.8}          # rough values supplied by the author (file missing)
SOURCE_NOTE = {"NW125 R": "supplied by the author", "NW125 D": "rough, supplied by the author"}

JOBS = []
for setting in ("T", "C"):
    for basis in ("R", "D"):
        for y in YEARS:
            for cfg in RUN:
                d = "a" if cfg != "BWN110" else "wn"
                JOBS.append((cfg, setting, f"{basis}{y}", {d: EFF[(cfg, basis)][y]}))
            # (B-std is calibrated by bstd_fcorr_v3.py on the final runs; its older targets are not part of the chain)
# the old targets, so that Step 1 also works through the same search (setting T only)
OLD_TARGETS = {"A1D": {"a": 43.56}, "A2D": {"a": 45.07}, "A4D": {"a": 46.20}, "BWN110": {"wn": 44.2}}
JOBS += [(cfg, "T", "old", t) for cfg, t in OLD_TARGETS.items()]
