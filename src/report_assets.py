"""Regenerates every thesis figure and table in final form, with a manifest and a check against the locked numbers.
Run by run_all.py. Figures that need the raw data are skipped."""
import argparse, hashlib, json, re, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

CHAPTER = "Figures and tables (repository scripts)"
TOL = 0.002


# ---------------------------------------------------------------------------------------------- LaTeX / Markdown
_TEX = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("€", r"\texteuro{}"),
        ("°", r"\textdegree{}"), ("−", r"$-$"), ("±", r"$\pm$"), ("×", r"$\times$"), ("≤", r"$\leq$"), ("≥", r"$\geq$"),
        ("φ", r"$\varphi$"), ("ψ", r"$\psi$"), ("η", r"$\eta$"), ("Δ", r"$\Delta$"), ("μ", r"$\mu$"), ("³", r"\textsuperscript{3}"),
        ("²", r"\textsuperscript{2}"), ("–", "--"), ("→", r"$\rightarrow$"), ("·", r"$\cdot$"), (">", r"$>$"), ("<", r"$<$")]


def tex(s):
    s = str(s)
    for a, b in _TEX:
        s = s.replace(a, b)
    return s


def numeric_like(col):
    vals = [str(v) for v in col if str(v) not in ("", "–", "-")]
    return bool(vals) and all(re.fullmatch(r"[+−\-±>]?[\d,.]+%?( ?yr)?|[\d.e\-]+|.*\d.*/.*", v) for v in vals)


def latex_tabular(df):
    spec = []
    for c in df.columns:
        longest = max([len(str(c))] + [len(str(v)) for v in df[c]])
        spec.append("r" if numeric_like(df[c]) else (f"p{{{min(6.0, 0.17 * min(longest, 45)):.1f}cm}}" if longest > 34 else "l"))
    out = [r"\begin{tabular}{" + "".join(spec) + "}", r"\toprule", " & ".join(tex(c) for c in df.columns) + r" \\", r"\midrule"]
    out += [" & ".join(tex(v) for v in r) + r" \\" for r in df.itertuples(index=False)]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def markdown(df, title, note):
    lines = [f"# {title}", ""] + ([note, ""] if note else [])
    lines += ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------- verification
def locked_values():
    m = pd.read_csv(C.RES_FINAL / "chapter_number_map.csv")
    return dict(zip(m.key.astype(str), m.value))


def check_rows(item, checks, lk):
    rows = []
    for label, value, ref, dec in checks:
        if isinstance(ref, tuple):
            src, refv = ref
        elif isinstance(ref, str):
            src, refv = ref, lk.get(ref, np.nan)
        else:
            src, refv = "reproduces the printed thesis input", float(ref)
        new = round(round(float(value), dec[0]), dec[1]) if isinstance(dec, tuple) else round(float(value), dec)
        ok = (refv == refv) and abs(new - float(refv)) <= TOL + 1e-12
        rows.append(dict(item=item, number=label, new=new, reference=refv, reference_source=src, passed=bool(ok)))
    return rows


def parse_num(s):
    s = str(s).replace("−", "-").replace(",", "").replace("%", "").replace(" yr", "").replace("±", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------------- main
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def png_check(path):
    import matplotlib.image as mpimg
    import report_style as S
    a = mpimg.imread(path)[..., :3]
    h, w = a.shape[:2]
    exp_w = round(S.WIDTH_IN * S.DPI)
    nonwhite = (a < 0.97).any(axis=2)
    border = np.concatenate([nonwhite[:2].ravel(), nonwhite[-2:].ravel(), nonwhite[:, :2].ravel(), nonwhite[:, -2:].ravel()])
    return dict(file=Path(path).name, width_px=w, height_px=h, expected_width_px=exp_w, width_ok=abs(w - exp_w) <= 1,
                not_blank=bool(nonwhite.mean() > 0.005), ink_share=round(float(nonwhite.mean()), 4),
                nothing_at_border=bool(border.mean() < 0.002), passed=bool(abs(w - exp_w) <= 1 and nonwhite.mean() > 0.005 and border.mean() < 0.002))


def contact_sheet(entries, out):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    n = len(entries)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(16, 3.4 * rows))
    for ax in np.atleast_1d(axs).flat:
        ax.axis("off")
    for ax, (label, p) in zip(np.atleast_1d(axs).flat, entries):
        ax.imshow(mpimg.imread(p))
        ax.set_title(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor="white", metadata={"Software": None})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.WORK / "report_assets"))   # outputs/work/report_assets
    ap.add_argument("--repo-copy", action="store_true", help="also copy figures, tables, data and manifest into the repository")
    a = ap.parse_args()
    out = Path(a.out)
    figdir, tabdir, datdir = out / "figures", out / "tables", out / "data"
    for d in (figdir, tabdir, datdir):
        d.mkdir(parents=True, exist_ok=True)
    import report_figures as RF
    import report_tables as RT
    items = pd.read_csv(C.DATA / "thesis_items.csv")
    lk = locked_values()
    manifest, ver, png = [], [], []
    raw_ok = RF.raw_available()
    for it in items.itertuples():
        files, info = [], None
        if it.kind == "Table":
            info = RT.BUILDERS[it.number]()
            stem = "table_" + it.number.replace(".", "_")
            df = info["df"]
            df.to_csv(tabdir / f"{stem}.csv", index=False, encoding="utf-8")
            (tabdir / f"{stem}.md").write_text(markdown(df, f"{it.item}: {it.caption}", info["note"]), encoding="utf-8")
            (tabdir / f"{stem}.tex").write_text(f"% {it.item} -- {info['status']}; source: {info['source']}\n" + latex_tabular(df),
                                                encoding="utf-8")
            files = [tabdir / f"{stem}.{e}" for e in ("csv", "md", "tex")]
            script = f"src/report_tables.py:{RT.BUILDERS[it.number].__name__}"
        elif it.item in RF.REPO or it.item in RF.RAW:
            fn = RF.REPO.get(it.item) or RF.RAW.get(it.item)
            if it.item in RF.RAW and not raw_ok:
                stem = RAW_STEMS[it.item]
                if stem in CONFIDENTIAL:   # hourly AEB data and licensed price series: figure and data never in the repository
                    info = RF.F(stem, status="not reproducible here (confidential: needs the raw AEB data; not in the repository)",
                                depends="no", source="raw AEB data (hourly buffer data / licensed price series)",
                                note="Needs data/raw/ (AEB_RAW_DATA_DIR); figure and data kept out of the repository.")
                else:   # can only be made with the raw data; the final version is with the thesis figures
                    info = RF.F(stem, status="not reproducible here (needs the raw AEB data in data/raw/)", depends="no",
                                source="raw AEB data", note="Run make_report_assets.py with data/raw/ (AEB_RAW_DATA_DIR) to draw it.")
            else:
                info = fn(figdir, datdir)
            stem = info["stem"]
            files = [p for p in (figdir / f"{stem}.png", figdir / f"{stem}.pdf", datdir / f"{stem}.csv") if p.exists()]
            if it.item == "Figure 7.2":
                files += [figdir / "ch7_npv_tornado_all_options.png", figdir / "ch7_npv_tornado_all_options.pdf",
                          datdir / "ch7_npv_tornado.csv"]
                files = [p for p in files if p.exists() and p.name != "ch7_npv_tornado_1d.csv"]
            script = f"src/report_figures.py:{fn.__name__}"
            for p in files:
                if p.suffix == ".png":
                    png.append(dict(item=it.item, **png_check(p)))
        else:
            info = dict(status="not reproducible (diagram, no data)", depends="no", source="thesis figure (drawing)", note="", checks=[])
            script = "–"
        ver += check_rows(it.item, info.get("checks", []), lk)
        manifest.append(dict(item=it.item, kind=it.kind, chapter=it.chapter, pdf_page=it.pdf_page, caption=it.caption[:120],
                             files=";".join(Path(p).relative_to(out).as_posix() for p in files), producing_script=script,
                             source_data=info["source"], depends_on_changed_numbers=info["depends"], status=info["status"],
                             note=info.get("note", ""), sha256=";".join(sha(p) for p in files)))
    M = pd.DataFrame(manifest)
    V = pd.DataFrame(ver)
    P = pd.DataFrame(png)
    V.to_csv(out / "verification.csv", index=False)
    P.to_csv(out / "png_checks.csv", index=False)
    # per item verdict
    verdict = V.groupby("item").passed.all() if len(V) else pd.Series(dtype=bool)
    M["verification"] = [("pass" if verdict[i] else "FAIL") if i in verdict.index else "n/a (no locked value)" for i in M["item"]]
    M.to_csv(out / "MANIFEST.csv", index=False)
    entries = [(r.item, out / f) for r in M.itertuples() for f in r.files.split(";") if f.endswith(".png") and "all_options" not in f]
    contact_sheet(entries, out / "contact_sheet.png")
    changes_vs_printed(out, RT)
    if a.repo_copy:
        repo_copy(out, M, RF)
    add_map_rows()
    n_fail = int((~V.passed).sum()) if len(V) else 0
    n_png = int((~P.passed).sum()) if len(P) else 0
    print(f"{len(M)} items ({(M.kind == 'Figure').sum()} figures, {(M.kind == 'Table').sum()} tables) written to {out}")
    print(f"verification: {len(V)} numbers, {n_fail} failed; PNG checks: {len(P)} files, {n_png} with a problem; "
          f"raw-data figures regenerated: {raw_ok}")
    if n_fail or n_png:
        print(V[~V.passed].to_string(index=False) if n_fail else "", P[~P.passed].to_string(index=False) if n_png else "")
        sys.exit(1)


def changes_vs_printed(out, RT):
    lines = ["# Changes against the printed thesis", "",
             "Every table cell and figure value whose content differs from `Master_Thesis_GL_Final (3).pdf`. Printed = the "
             "submitted thesis; new = this folder (final numbers). Cells of tables that are thesis inputs and unchanged are "
             "not listed.", ""]
    for num, b in RT.BUILDERS.items():
        key = f"Table {num}"
        if key not in RT.PRINTED:
            continue
        p = RT.PRINTED[key]
        info = b()
        new = info["df"]
        if info["status"].startswith("thesis input"):
            continue
        rows = []
        pdf = pd.DataFrame(p["rows"], columns=p["columns"])
        shared = [c for c in range(min(len(pdf.columns), len(new.columns)))]
        lead = 0                                    # leading text columns identify a row (e.g. config / doublet / well)
        while lead < min(3, len(new.columns) - 1) and all(parse_num(v) is None for v in new.iloc[:, lead]):
            lead += 1
        lead = max(lead, 1)
        for i in range(min(len(pdf), len(new))):
            label = " / ".join(str(new.iat[i, k]) for k in range(lead))
            for j in shared:
                pv, nv = str(pdf.iat[i, j]), str(new.iat[i, j])
                if pv.replace(" ", "") != nv.replace(" ", ""):
                    rows.append((label, str(new.columns[j]), pv, nv))
        if list(pdf.columns) != list(new.columns):
            rows.append(("(header)", "columns", " / ".join(pdf.columns), " / ".join(map(str, new.columns))))
        if rows:
            lines += [f"## {key} (p. {p['page']}) — {info['status']}", "", "| row | column | printed | new |", "|---|---|---|---|"]
            lines += [f"| {r} | {c} | {a} | {b_} |" for r, c, a, b_ in rows]
            if info.get("note"):
                lines += ["", info["note"]]
            lines.append("")
    figs = RT.PRINTED["_figures"]
    import report_numbers as RN
    lines += ["## Figures", "", "| figure | value | printed | new |", "|---|---|---|---|"]
    rc = pd.read_csv(C.RES_ROSIM / "rosim_check.csv")
    for run, v in figs["Figure 6.4"]["year7"].items():
        col = "cleaned" if run == "B-WN110" else "recomputed"
        nv = float(rc[(rc.run == run) & (rc.year == 7)][col].iloc[0])
        lines.append(f"| Figure 6.4 | year-7 efficiency {run} (%) | {v} | {nv:.2f} |")
    for o, v in figs["Figure 6.5"]["storage_after"].items():
        lines.append(f"| Figure 6.5 | storage effect after calibration {o} (k€/yr) | {v} | {RN.storage_after(o):.0f} |")
    for o, v in figs["Figure 7.1"]["base"].items():
        lines.append(f"| Figure 7.1 | NPV base {o} (M€) | {v:+.2f} | {RN.npv(o):+.2f} |")
    nb = RN.npv("A1D")
    lines.append(f"| Figure 7.2 | A1D base NPV (M€) | {figs['Figure 7.2']['npv_base']:+.2f} | {nb:+.2f} |")
    lines.append(f"| Figure 7.2 | A1D NPV with pump replacement (M€) | {figs['Figure 7.2']['pump replacement included']:+.2f} | "
                 f"{nb - RN.pump_replacement_pv_meur('A1D'):+.2f} |")
    if RN.buffer_cases_available():
        lines.append(f"| Figure 7.2 | A1D NPV with buffers as by AEB, n = 2 (M€) | {figs['Figure 7.2']['buffers used as by AEB (n = 2)']:+.2f} | "
                     f"{RN.buffer_npv('A1D', 'n2'):+.2f} |")
    lines += ["| Figure 6.1 | warm-well temperature | earlier A1D run (two discharge periods per year) | final A1D run (one per year) |",
              "| Figure B.1 | annualised tank cost | €300/m³ mid CAPEX | case-B method at each volume |"]
    (out / "changes_vs_printed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


CONFIDENTIAL = {"ch3_buffer_model_vs_measured_week", "ch3_epex_prices"}   # Figures 3.2 and 3.3
RAW_STEMS = {"Figure 3.1": "ch3_buffer_hour_profile", "Figure 3.2": "ch3_buffer_model_vs_measured_week",   # need data/raw/
             "Figure 3.3": "ch3_epex_prices"}


def repo_copy(out, M, RF):
    """figures, tables, figure data and the report files into output/ (figures/, results/tables, results/figures_data)."""
    (C.RESULTS / "tables").mkdir(exist_ok=True)
    (C.RESULTS / "figures_data").mkdir(exist_ok=True)
    confidential = CONFIDENTIAL
    for p in (out / "figures").glob("*"):
        if p.stem in confidential:
            continue                       # hourly AEB / licensed price data: kept out of the repository
        shutil.copy2(p, C.FIGURES / p.name)
    for p in (out / "tables").glob("table_*"):
        shutil.copy2(p, C.RESULTS / "tables" / p.name)
    for p in (out / "data").glob("*.csv"):
        if p.stem not in confidential:
            shutil.copy2(p, C.RESULTS / "figures_data" / p.name)
    for f in ("MANIFEST.csv", "verification.csv", "png_checks.csv", "changes_vs_printed.md", "contact_sheet.png"):
        shutil.copy2(out / f, C.RESULTS / ("report_" + f.lower()))


def add_map_rows():
    """new numbers of the report (not in the frozen reference) into chapter_number_map.csv / final_numbers_locked.md."""
    import figtab_common as FT
    import report_numbers as RN
    R = FT.Rows("Report")
    for o in RN.OPTIONS:
        pb = RN.simple_payback(o)
        R.put(f"Table 7.6 simple payback, {o}", pb if pb <= 30 else float("nan"), "yr (nan = >30)", "CAPEX / (S_attr,base - OPEX)", 1)
        R.put(f"Table 7.6 net annual benefit, {o}", RN.tiers(o)["base"] - RN.opex(o), "k EUR/yr", "S_attr,base - OPEX", 0)
        if o != "TTES5k":
            R.put(f"Figure 7.2 pump replacement PV, {o}", RN.pump_replacement_pv_meur(o), "M EUR", "App. B.3 formula", 2)
            R.put(f"Figure 7.2 NPV with pump replacement, {o}", RN.npv(o) - RN.pump_replacement_pv_meur(o), "M EUR", "NPV base - PV pumps", 2)
    if RN.buffer_cases_available():
        for case in ("nobuf", "n1", "n2", "n3", "n4"):
            b = RN.buffer_case(case)
            for k in ("S_ref", "Z", "S_flex", "S_stor_A1D", "S_stor_TTES"):
                R.put(f"Table B.8 {k}, {case}", b[k], "k EUR/yr", "buffer_use_runs.py", 0)
            R.put(f"Table B.8 NPV A1D base, {case}", RN.buffer_npv("A1D", case), "M EUR", "final f_corr", 2)
            R.put(f"Table B.8 NPV TTES base, {case}", RN.buffer_npv("TTES5k", case), "M EUR", "case B", 2)
    import tank_capex as TC
    lo, mid, hi = RN.tank_case_b(19000)
    R.put("Table 7.11 TTES 19,000 m3 CAPEX (case-B method)", mid, "M EUR", "tank_capex.project_table(19000)", 2)
    R.put("Table 7.4 TTES 19,000 m3 OPEX (1.4 %)", TC.opex_keur(mid), "k EUR/yr", "1.4 % of CAPEX", 1)
    R.write()


if __name__ == "__main__":
    main()

