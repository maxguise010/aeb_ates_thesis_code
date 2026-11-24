"""Single entry point: regenerates every thesis number, table and figure that does not need the raw AEB data, checks
them against reference/ and prints PASS or FAIL per group. Usage: python run_all.py, or python run_all.py --step
MODULE [ARGS] to run one module of src/. Everything is written to outputs/. The raw data, needed only by the model
code, go in data/raw/ or the folder named by AEB_RAW_DATA_DIR."""
import csv
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
OUT = REPO / "outputs"

STEPS = [   # (label, module of src/, arguments): the order of the thesis chain
    ("scenario decomposition (S_ref, S_flex, S_stor, NPV per scenario)", "scenario_runner", []),
    ("TTES runs at 110/95/90 C (tank LCOH)", "run_ttes_low_temp", ["--workers", "2"]),
    ("ROSIM efficiencies per year and well geometry", "rosim_check", []),
    ("f_corr table of the thesis column", "fcorr_followup_table", []),
    ("f_corr direct runs (calibration of the aquifer module)", "fcorr_runs", ["targets", "--workers", "1", "--allow-flagged", "Bstd"]),
    ("f_corr of B-std, basis R", "bstd_fcorr_v3", []),
    ("f_corr of B-std, basis D", "bstd_fcorr_v3", ["--basis", "D"]),
    ("final numbers: decomposition, NPV, LCOH, break-even", "final_numbers", []),
    ("demand and price scenarios", "scenarios", []),
    ("analytic scenarios (growth, discount rate)", "scenarios", ["analytic"]),
    ("cover-horizon sensitivity", "tcover_analysis", []),
    ("well limit: hours above the limit", "well_limit", ["hours"]),
    ("well limit: capped runs", "well_limit", ["summary"]),
    ("write-up of the scenario, T_cover and well-limit numbers", "write_up", []),
    ("pit check: TTES runs, thesis buffers", "run_ttes_low_temp", ["--workers", "2", "--substeps-10k", "16"]),
    ("pit check: TTES runs, calibrated buffers", "run_ttes_low_temp", ["--workers", "2", "--substeps-10k", "16", "--buffer", "calibrated"]),
    ("pit check table", "make_pit_table", []),
    ("ROSIM production temperatures", "rosim_production_temps", []),
    ("temperature sensitivity (derated power, heat pump)", "analyse_sensitivity", []),
    ("B-std return temperature 70 C (sensitivity)", "bstd_return_70", []),
    ("sensitivity numbers", "sensitivity_numbers", []),
    ("B-std detail numbers", "bstd_numbers", []),
    ("thesis tables and figures in final form", "report_assets", ["--repo-copy"]),
    ("derived numbers of Chapters 3 and 4", "derived_numbers", []),
]
CHECKS = [   # (group, module, arguments): run even if an earlier check fails
    ("locked numbers", "verify_locked", []),
    ("report numbers, PNG files, tables and figure data", "verify_report", []),
    ("sensitivity analyses", "verify_sensitivity", []),
    ("derived numbers and extra-analysis data", "derived_numbers", ["--check"]),
    ("model code smoke test", "derived_numbers", ["--smoke"]),
]


def run_module(module, args):
    """Run src/<module>.py as __main__ (as 'python -m module args'); a module that calls sys.exit(0) ends normally."""
    sys.path[:0] = [str(REPO), str(SRC)]
    sys.argv = [module] + list(args)
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        if e.code not in (None, 0):
            raise


def spawn(module, args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO), str(SRC)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    return subprocess.run([sys.executable, str(Path(__file__).resolve()), "--step", module, *args], cwd=str(REPO), env=env).returncode


def read(path):
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return None


def summary():
    """PASS / FAIL per group, from the files the checks write."""
    R = OUT / "results"
    groups = []
    v = read(R / "verification.csv")
    if v is not None:
        cmp = [r for r in v if not r["status"].startswith("new")]
        diffs = [abs(float(r["diff"])) for r in cmp if r["diff"] not in ("", "nan")]
        bad = [r for r in cmp if r["status"] != "ok"]
        groups.append(("locked numbers", not bad, f"{len(cmp)} compared, largest |diff| {max(diffs) if diffs else 0:.6f}, {len(bad)} differ"))
    rv, rp = read(R / "report_verification.csv"), read(R / "report_png_checks.csv")
    if rv is not None and rp is not None:
        nb = sum(r["passed"] != "True" for r in rv) + sum(r["passed"] != "True" for r in rp)
        groups.append(("report numbers and PNG files", nb == 0, f"{len(rv)} numbers, {len(rp)} PNG files, {nb} failed"))
    tf = read(R / "verification_tables_figures.csv")
    if tf is not None:
        for kind, lab in (("tables/", "tables"), ("figures_data/", "figure data")):
            rows = [r for r in tf if r["file"].startswith(kind)]
            eq = sum(r["status"] == "equal" for r in rows)
            raw = sum(r["status"].startswith("not regenerated") for r in rows)
            bad = len(rows) - eq - raw
            groups.append((lab, bad == 0, f"{len(rows)} reference files: {eq} equal, {raw} need the raw data, {bad} differ or missing"))
    vs = read(R / "verification_sensitivity.csv")
    if vs is not None:
        nb = sum(r["status"] != "ok" for r in vs)
        groups.append(("sensitivity analyses", nb == 0, f"{len(vs)} checks, {nb} failed"))
    for f, lab in (("derived_numbers_check.csv", "derived numbers (Chapters 3 and 4)"), ("extra_results_check.csv", "extra-analysis data"),
                   ("smoke_test.csv", "model code smoke test")):
        d = read(R / f)
        if d is not None:
            st = [r["status"] for r in d]
            nb = sum(s in ("DISAGREE", "FAIL") for s in st)
            nr = sum(s == "NOT REPRODUCED" for s in st)
            groups.append((lab, nb == 0, f"{len(d)} items: {st.count('AGREE') + st.count('ok')} agree, {nr} not reproduced, {nb} disagree or fail"))
    return groups


def main():
    if sys.argv[1:2] == ["--step"]:
        run_module(sys.argv[2], sys.argv[3:])
        return
    t_all, log = time.time(), []
    stopped = False
    for label, module, args in STEPS:
        print(f"\n=== {module} {' '.join(args)}: {label} " + "=" * 20, flush=True)
        t0 = time.time()
        rc = spawn(module, args)
        log.append(dict(step=f"{module} {' '.join(args)}".strip(), label=label, seconds=round(time.time() - t0, 1), exit_code=rc))
        if rc != 0:
            print(f"\nSTOPPED: {module} ended with exit code {rc}")
            stopped = True
            break
    if not stopped:
        for group, module, args in CHECKS:
            print(f"\n=== check: {group} " + "=" * 20, flush=True)
            t0 = time.time()
            rc = spawn(module, args)
            log.append(dict(step=f"check {module} {' '.join(args)}".strip(), label=group, seconds=round(time.time() - t0, 1), exit_code=rc))
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    with open(OUT / "results" / "run_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "label", "seconds", "exit_code"])
        w.writeheader()
        w.writerows(log)
    total = time.time() - t_all
    groups = summary() if not stopped else []
    print("\n" + "=" * 100 + "\nRESULT PER GROUP\n" + "=" * 100)
    for g, ok, detail in groups:
        print(f"{'PASS' if ok else 'FAIL'}  {g:<40} {detail}")
    ok_all = (not stopped) and groups and all(ok for _, ok, _ in groups) and all(r["exit_code"] == 0 for r in log)
    print("-" * 100)
    print(f"{'PASS' if ok_all else 'FAIL'}  overall ({len(log)} steps and checks, {total:.0f} s); details in outputs/results/")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
