"""Check: report numbers and PNG files against the locked values, and every regenerated table and figure data set
against reference/, cell by cell. Run by run_all.py."""
import sys

import pandas as pd

import config as C
import data_store as DS
from report_assets import RAW_STEMS


def main():
    v = pd.read_csv(C.RESULTS / "report_verification.csv")
    p = pd.read_csv(C.RESULTS / "report_png_checks.csv")
    print(f"report: {len(v)} numbers checked against the locked values ({int(v.passed.sum())} passed); "
          f"{len(p)} PNG files checked ({int(p.passed.sum())} passed)")
    ok_tf = tables_and_figure_data()
    if not v.passed.all() or not p.passed.all() or not ok_tf:
        print(v[~v.passed].to_string(index=False)); print(p[~p.passed].to_string(index=False))
        sys.exit(1)
    print("all report numbers and figures pass")


def tables_and_figure_data():
    """every regenerated table / figure data CSV against the long reference files (exact cell text)."""
    man = pd.read_csv(C.RESULTS / "report_manifest.csv")
    raw_only = set(man.loc[man.status.astype(str).str.startswith("not reproducible here"), "item"])
    rows = []
    for ref_file, folder, pattern in ((C.REF_TABLES, C.RESULTS / "tables", "table_*.csv"),
                                      (C.REF_FIGURES_DATA, C.RESULTS / "figures_data", "*.csv")):
        ref = DS.read_long(ref_file)
        new_files = {p.name: p for p in folder.glob(pattern)}
        for name in sorted(set(ref.file) | set(new_files)):
            item = man.loc[man.files.astype(str).str.contains(name, regex=False), "item"]
            item = item.iloc[0] if len(item) else {s: i for i, s in RAW_STEMS.items()}.get(name[:-4], "")
            if name not in new_files:
                status = "not regenerated: needs the raw data" if item in raw_only else "MISSING"
                n = 0
            elif name not in set(ref.file):
                status, n = "NOT IN REFERENCE", 0
            else:
                bad = DS.compare_long(DS.to_long(new_files[name]), ref[ref.file == name])
                status, n = ("equal" if not len(bad) else "DIFFERS"), len(bad)
            rows.append(dict(file=f"{folder.name}/{name}", item=item, status=status, records_differing=n))
    t = pd.DataFrame(rows)
    t.to_csv(C.RESULTS / "verification_tables_figures.csv", index=False)
    bad = t[~t.status.isin(["equal", "not regenerated: needs the raw data"])]
    print(f"tables and figure data: {len(t)} reference files, {int((t.status == 'equal').sum())} regenerated and equal "
          f"to data/reference/, {int(t.status.str.startswith('not regenerated').sum())} need the raw data, {len(bad)} problems")
    if len(bad):
        print(bad.to_string(index=False))
    return not len(bad)


if __name__ == "__main__":
    main()
