"""Check: compares every regenerated locked number with reference/locked_numbers.csv (tolerance 0.002). Run by
run_all.py."""
import sys

import numpy as np
import pandas as pd

import config as C

TOL = 0.002


def keyed(df):
    """key -> value; a key that occurs more than once gets a running number (#2, #3, ...)."""
    seen, out = {}, {}
    for k, v in zip(df["key"].astype(str), df["value"]):
        seen[k] = seen.get(k, 0) + 1
        out[k if seen[k] == 1 else f"{k} #{seen[k]}"] = v
    return out


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def main():
    nm = pd.read_csv(C.RES_FINAL / "chapter_number_map.csv")
    s70 = C.RES_BSTD_RETURN_70 / "chapter_number_map_70.csv"     # B-std return 70 C (sensitivity, bstd_return_70.py)
    if s70.exists():
        nm = pd.concat([nm, pd.read_csv(s70)], ignore_index=True)
    added = set(nm.loc[nm.chapter == "Figures and tables (repository scripts)", "key"].astype(str))
    sens = set(nm.loc[nm.chapter == "Sensitivity analyses (appendix)", "key"].astype(str))   # checked by verify_sensitivity.py
    added |= sens
    new = keyed(nm)
    ref = keyed(pd.read_csv(C.REF_LOCKED).rename(columns={"id": "key"}))   # reference/locked_numbers.csv
    rows = []
    for k in sorted(set(ref) | set(new)):
        a, b = ref.get(k), new.get(k)
        fa, fb = as_float(a), as_float(b)
        if k not in ref and k in sens:
            status = "new (sensitivity, appendix)"   # sensitivity numbers, not locked; see results/verification_sensitivity.csv
        elif k not in ref and k in added:
            status = "new (figure/table script)"     # numbers added by the figure and table scripts, not locked earlier
        elif k not in new or k not in ref:
            status = "missing in new" if k not in new else "not in reference"
        elif np.isnan(fa) or np.isnan(fb):
            status = "ok" if str(a) == str(b) else "text differs"
        else:
            status = "ok" if abs(fa - fb) <= TOL + 1e-12 else "DIFFERS"
        rows.append(dict(key=k, reference=a, new=b, diff=(fb - fa) if not (np.isnan(fa) or np.isnan(fb)) else np.nan,
                         status=status))
    t = pd.DataFrame(rows)
    t.to_csv(C.RESULTS / "verification.csv", index=False)
    new_status = ["new (figure/table script)", "new (sensitivity, appendix)"]
    bad = t[~t.status.isin(["ok"] + new_status)]
    cmp = t[~t.status.isin(new_status)]
    n_num = int(cmp["diff"].notna().sum())
    n_fig, n_sens = int((t.status == new_status[0]).sum()), int((t.status == new_status[1]).sum())
    print(f"{len(cmp)} locked numbers compared ({n_num} numeric), tolerance {TOL}; largest |diff| = "
          f"{cmp['diff'].abs().max():.6f}; {n_fig} new numbers of the figure/table scripts and {n_sens} sensitivity "
          f"numbers (not in the reference)")
    if len(bad):
        print("DIFFERENCES:")
        print(bad.to_string(index=False))
        sys.exit(1)
    print("all numbers match the reference within the tolerance")


if __name__ == "__main__":
    main()
