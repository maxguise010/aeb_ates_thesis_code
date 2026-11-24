"""Readers for the ROSIM project files (settings, well files, ATESEfficiency.txt) and the schedule workbooks, used by
the ROSIM scripts."""
import datetime as dt
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import functools

import config as C  # repository paths (config.py)
HERE = C.WORK / "rosim_input_audit"   # output folder of audit.py (not part of the repository run)
RESINSIGHT = C.ROSIM_PROJECTS          # ROSIM projects (data/rosim/<name>/: flat subset of each project)
INPUTS = C.ROSIM_SCHEDULES             # 8-year schedule workbooks
DISPATCH = C.DISPATCH_HOURLY
WEEKS, N_YEARS, H_WEEK = 52, 8, 168.0

# run -> workbook (the author's list), ROSIM project (matched by flow profile in projects.py), dispatch column key
RUNS = {
    "A1D": dict(wb="option_a_1d_8yr.xlsx", project="Option A 1 Doublet 8 years changed parameters", disp="A1D", key="A", n=1),
    "A2D": dict(wb="option_a_2d_8yr.xlsx", project="Option A 2 doublets 8 years changed parameters", disp="A2D", key="A", n=2),
    "A4D": dict(wb="option_a_4d_8yr.xlsx", project="Option A 4 doublets 8 years changed parameters", disp="A4D", key="A", n=4),
    "B-WN110": dict(wb="option_b_wn110_8yr.xlsx", project="Option B WN 110C 8 year adjusted", disp="BWN110", key="WN", n=1),
    "B-std WN": dict(wb="option_b_std_wn_8yr.xlsx", project="Option B WN 8 years adjusted", disp="Bstd", key="WN", n=1),
    "B-std NW": dict(wb="option_b_std_nw_8yr.xlsx", project="Option B 8 years NW adjusted", disp="Bstd", key="NW", n=1),
}
TEMPS = {"A1D": (110.0, 58.0), "A2D": (110.0, 58.0), "A4D": (110.0, 58.0), "B-WN110": (110.0, 58.0),
         "B-std WN": (90.0, 70.0), "B-std NW": (90.0, 70.0)}


def project_dir(name):
    return RESINSIGHT / name


# Consolidated ROSIM data: data/rosim/<code>.csv holds, per ROSIM week, the columns of ATESEfficiency.txt, the density
# and cp of each well file and the flow profile of each well; data/rosim/settings.csv holds the scalar fields and layer rows
# of <project>_last_run_settings.xml. The text of the ROSIM exports is copied token by token, so every value parses to the
# same float as from the original files.
ROSIM_CODE = {"Option A 1 Doublet 8 years changed parameters": "A1D", "Option A 2 doublets 8 years changed parameters": "A2D",
              "Option A 4 doublets 8 years changed parameters": "A4D", "Option B WN 110C 8 year adjusted": "BWN110",
              "Option B WN 58 90 def": "Bstd_WN_58", "Option B NW 58 90 def": "Bstd_NW_58",
              "Option B WN 8 years adjusted v3 70m": "Bstd_WN_70", "Option B 8 years NW adjusted v3": "Bstd_NW_70"}
# schedule workbook -> ROSIM run it was loaded into; the workbook rows equal the warm1 / cold1 flow profile of that run
# (checked when the package was built: identical for the six final workbooks)
WB_PROJECT = {"option_a_1d_8yr.xlsx": "Option A 1 Doublet 8 years changed parameters",
              "option_a_2d_8yr.xlsx": "Option A 2 doublets 8 years changed parameters",
              "option_a_4d_8yr.xlsx": "Option A 4 doublets 8 years changed parameters",
              "option_b_wn110_8yr.xlsx": "Option B WN 110C 8 year adjusted",
              "option_b_std_wn_8yr_58.xlsx": "Option B WN 58 90 def", "option_b_std_nw_8yr_58.xlsx": "Option B NW 58 90 def",
              "option_b_std_wn_8yr_v3.xlsx": "Option B WN 8 years adjusted v3 70m", "option_b_std_nw_8yr_v3.xlsx": "Option B 8 years NW adjusted v3"}


@functools.lru_cache(maxsize=None)
def _rosim_text(name):
    """data/rosim/<code>.csv of a ROSIM run as text (every cell a string, as in the ROSIM export)."""
    return pd.read_csv(RESINSIGHT / f"{ROSIM_CODE[name]}.csv", dtype=str, keep_default_na=False)


@functools.lru_cache(maxsize=None)
def _rosim_cells(name):
    """density and cp columns of data/rosim/<code>.csv parsed by pandas (as pd.read_csv parsed the well files)."""
    return pd.read_csv(RESINSIGHT / f"{ROSIM_CODE[name]}.csv", usecols=lambda c: c.endswith(("__density", "__cp")))


@functools.lru_cache(maxsize=None)
def _settings_table():
    return pd.read_csv(RESINSIGHT / "settings.csv", dtype=str, keep_default_na=False)


def read_settings(name):
    """Settings of a ROSIM run: scalar fields, layer rows, and the flow profile per well as a DataFrame."""
    s = _settings_table()
    s = s[s.project == name]
    out = {f: v for f, v in zip(s.field, s.value) if f != "model3d_layers"}
    out["layers"] = [v.split(";") for v in s.loc[s.field == "model3d_layers", "value"]]
    wells = out["well_name"].split(";")
    t = _rosim_text(name)
    prof = {}
    for w in wells:
        cols = [f"{w}__p_{k}" for k in ("year", "month", "day", "flow", "bhp", "temp")]
        rows = [list(map(float, r)) for r in t[cols].itertuples(index=False) if r[0] != ""]
        prof[w] = pd.DataFrame(rows, columns=["year", "month", "day", "flow", "bhp", "temp"])
    out["profiles"] = prof
    out["wells"] = wells
    return out


def read_deck_props(name):
    """SPECHEAT (water column at 0..200 C), DENSITY, SPECROCK, RTEMPVD and WELSPECS/COMPDAT from the OPM deck."""
    lines = (project_dir(name) / f"{name}_OPM_FLOW.DATA").read_text(encoding="latin-1").splitlines()

    def block(kw):
        i = lines.index(kw) + 1
        rows = []
        while True:
            s = lines[i].strip()
            i += 1
            if not s or s.startswith("--"):
                if rows and not s:
                    break
                continue
            rows.append(s)
            if s.endswith("/"):
                break
        return rows

    sh = {float(r.split()[0]): float(r.split()[2]) for r in block("SPECHEAT")}
    dens = float(block("DENSITY")[0].split()[1])
    rock = float(block("SPECROCK")[0].split()[1])
    rt = [tuple(map(float, r.replace("/", "").split())) for r in block("RTEMPVD")]
    welspecs = [r for r in block("WELSPECS") if r != "/"]
    i = lines.index("COMPDAT") + 1
    comp = []
    while lines[i].strip() != "/":
        if lines[i].startswith("'"):
            comp.append(lines[i].split())
        i += 1
    return dict(specheat=sh, density=dens, specrock=rock, rtempvd=rt, welspecs=welspecs, compdat=comp)


def read_workbook(path):
    """Warmwell and Coldwell rows (year, month, day, flow, bhp, temp) of an 8-year schedule workbook. The workbooks are
    not shipped: their rows are the warm1 / cold1 flow profile of the ROSIM run they were loaded into (data/rosim/).
    The Metadata sheet is not available (empty list)."""
    s = read_settings(WB_PROJECT[Path(path).name])
    sheets = {"Warmwell": s["profiles"]["warm1"].copy(), "Coldwell": s["profiles"]["cold1"].copy()}
    return sheets, [], ["Warmwell", "Coldwell", "Metadata"]


def read_efficiency(name):
    """ATESEfficiency.txt: weekly rows with total energy and the reported yearly efficiency."""
    t = _rosim_text(name)
    df = pd.DataFrame({"week": [int(x) for x in t.week],
                       "total_energy": [float(x) for x in t.total],
                       "yearly_eff": [float(x) if x.strip() else np.nan for x in t.yearly]})
    return df


def read_wells(name):
    """Per-well voxet files (keyed by the file name, e.g. warm1_i39j37): the weekly density and cp of the well cell,
    the only columns the scripts use."""
    t = _rosim_cells(name)
    stems = sorted(c[:-len("__density")] for c in t.columns if c.endswith("__density"))
    return {w: pd.DataFrame({"density": t[f"{w}__density"].values, "cp": t[f"{w}__cp"].values}) for w in stems}


def years(q):
    return [np.asarray(q[i * WEEKS:(i + 1) * WEEKS], float) for i in range(len(q) // WEEKS)]


def dispatch(run):
    c = RUNS[run]
    df = pd.read_pickle(DISPATCH / f"{c['disp']}__thesis.pkl")
    return df[f"{c['key']}_charge_mw"], df[f"{c['key']}_discharge_mw"]
