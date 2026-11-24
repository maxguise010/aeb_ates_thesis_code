"""Sensitivity of Option B standard to a 70 C return temperature: reruns the ROSIM check, f_corr and final numbers
into outputs/results/sensitivity/bstd_return_70/. Run by run_all.py."""
import shutil

import pandas as pd

import config as C
from repo_run import run

PREFIX = "B-std return 70 C | "


def main():
    C.BSTD_T_RETURN_C = 70.0                   # the only change: the B-std return temperature
    out = C.RES_BSTD_RETURN_70
    for sub in ("rosim", "fcorr", "final_numbers"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for f in ("direct_runs.csv", "followup_final_table.csv"):      # inputs of final_numbers.py for the other options
        shutil.copy2(C.RES_FCORR / f, out / "fcorr" / f)
    C.RES_ROSIM, C.RES_FCORR, C.RES_FINAL = out / "rosim", out / "fcorr", out / "final_numbers"   # redirect all outputs
    run("rosim_check")
    run("bstd_fcorr_v3")
    run("bstd_fcorr_v3", "--basis", "D")
    run("final_numbers")
    m = pd.read_csv(C.RES_FINAL / "chapter_number_map.csv")
    b = m[m.key.astype(str).str.startswith("B standard")].copy()
    b["key"] = PREFIX + b.key
    b["chapter"] = "Sensitivity: B-std return temperature 70 C"
    b["source"] = b.source.astype(str) + " (config.BSTD_T_RETURN_C = 70, src/bstd_return_70.py)"
    b.to_csv(out / "chapter_number_map_70.csv", index=False)
    print(f"{len(b)} B-std numbers at a 70 C return written to {out.relative_to(C.REPO).as_posix()}/chapter_number_map_70.csv")
    print(b[["key", "value", "unit"]].to_string(index=False))


if __name__ == "__main__":
    main()
