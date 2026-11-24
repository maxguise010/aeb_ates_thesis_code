"""Builders of the thesis data figures, in one style and with the final numbers; called by report_assets.py."""
import contextlib
import io
import runpy

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import config as C
import final_config as FC
import report_numbers as RN
import report_style as S

RAW_FILES = ["bufferstand_warmtevraag.csv", "Warmtewisselaar.xlsx", "Elektriciteit_prijzen.xlsx",
             "Warmtelevering_metingen_details.xlsx", "EPEX_prijs_simpele_voorspelling.xlsx"]


def raw_available():
    """False: the figures with raw AEB data (3.1-3.3) are not drawn by this package."""
    return False


def F(stem, checks=(), status="final numbers", depends="yes", source="", note="", confidential=False):
    return dict(stem=stem, checks=list(checks), status=status, depends=depends, source=source, note=note,
                confidential=confidential)


def _quiet_run(path, argv=None):
    """run an original project script and return its globals (stdout discarded).

    To os.devnull as a real text stream, because some scripts call sys.stdout.reconfigure()."""
    import os, sys
    old = sys.argv
    sys.argv = [str(path)] + (argv or [])
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
            return runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = old


# ============================================================================== repository data
def fig_5_1(figdir, datadir):
    d = pd.read_csv(C.RES_TTES_SWEEP / "ttes_price_input_test_relaxed.csv").query("variant == 'actual'").copy()
    d["storage_effect_keur_yr"] = d.increment_eur_yr / 1000
    d[["dispatch", "volume_m3", "storage_effect_keur_yr", "cycles_per_yr"]].sort_values(["dispatch", "volume_m3"]).to_csv(
        datadir / "ch5_ttes_storage_effect.csv", index=False)
    fig, (a1, a2) = S.figure(6.5, 1, 2)
    col = S.OPT_COLOR["TTES5k"]
    vols = sorted(d.volume_m3.unique())
    for pair, ls, mk in (("P20/P80", "-", "o"), ("P30/P60", "--", "s")):
        g = d[d.dispatch == pair].sort_values("volume_m3")
        for ax, y in ((a1, "storage_effect_keur_yr"), (a2, "cycles_per_yr")):
            ax.plot(g.volume_m3, g[y], color=col, linestyle=ls, marker=mk, markersize=3.5, markeredgecolor="white")
            ax.annotate(pair, (g.volume_m3.iloc[-1], g[y].iloc[-1]), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=S.FS_SMALL, color=S.INK)
    a1.axhline(0, color=S.MID, linewidth=S.LW_THIN)
    for ax, yl in ((a1, "Storage effect (k€/yr)"), (a2, "Charge cycles per year (1/yr)")):
        S.despine(ax)
        ax.set_xlabel("Tank volume (m³)")
        ax.set_ylabel(yl)
        ax.set_xticks(vols)
        ax.set_xticklabels([f"{v / 1000:g}k" for v in vols])
        ax.set_xlim(1000, 23500)
    S.save(fig, figdir, "ch5_ttes_storage_effect")
    return F("ch5_ttes_storage_effect", status="unchanged, scripted", depends="no",
             source="data/derived/ttes_price_input_test_relaxed.csv")


def fig_b_1(figdir, datadir):
    import tank_capex as TC
    d = pd.read_csv(C.RES_TTES_SWEEP / "ttes_price_input_test_relaxed.csv").query("variant == 'actual'").copy()
    vols = sorted(d.volume_m3.unique())
    cost = []
    for v in vols:
        lo, mid, hi = RN.tank_case_b(v)
        cost.append(mid * 1000 * RN.CRF + TC.opex_keur(mid))
    rows = [dict(series=f"storage effect {r.dispatch}", volume_m3=r.volume_m3, keur_yr=r.increment_eur_yr / 1000) for r in d.itertuples()]
    rows += [dict(series="annualised cost, mid CAPEX (case-B method), OPEX 1.4%", volume_m3=v, keur_yr=c) for v, c in zip(vols, cost)]
    pd.DataFrame(rows).sort_values(["series", "volume_m3"]).to_csv(datadir / "ch5_ttes_sweep.csv", index=False)
    fig, ax = S.figure(7.0)
    col = S.OPT_COLOR["TTES5k"]
    for pair, ls, mk in (("P20/P80", "-", "o"), ("P30/P60", "--", "s")):
        g = d[d.dispatch == pair].sort_values("volume_m3")
        ax.plot(g.volume_m3, g.increment_eur_yr / 1000, color=col, linestyle=ls, marker=mk, markersize=3.5, markeredgecolor="white")
        ax.annotate(f"Storage effect {pair}", (g.volume_m3.iloc[-1], g.increment_eur_yr.iloc[-1] / 1000), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=S.FS_SMALL)
    ax.plot(vols, cost, color=S.DARK, linestyle=":", marker="D", markersize=3)
    ax.annotate("Annualised cost (mid CAPEX)", (vols[-1], cost[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=S.FS_SMALL, color=S.DARK)
    ax.axhline(0, color=S.MID, linewidth=S.LW_THIN)
    S.despine(ax)
    ax.set_xticks(vols); ax.set_xticklabels([f"{v:,.0f}" for v in vols])
    ax.set_xlim(1000, 29000)
    ax.set_xlabel("Tank volume (m³)"); ax.set_ylabel("Storage effect and annualised cost (k€/yr)")
    S.save(fig, figdir, "ch5_ttes_sweep")
    return F("ch5_ttes_sweep", depends="yes (tank CAPEX)", source="TTES sweep; annualised cost = case-B CAPEX at each volume "
             "x CRF + 1.4% OPEX", note="Case-B costing replaces €300/m³; derived for this figure.")


def fig_6_1(figdir, datadir):
    import audit as au
    import rosim_check as RCK
    proj, _, t_inj, t_ret, _ = RCK.RUNS["A1D"]
    eff, per = au.read_eff_full(proj)
    w = per[next(k for k in per if k.startswith("warm"))]
    t = eff.week.values / 52.0
    m = t <= 7
    T, Q, t = w["T"].values[m], w["Q"].values[m], t[m]
    pd.DataFrame(dict(week=eff.week.values[m], year=t, warm_temperature_c=T, warm_flow_m3h=Q)).to_csv(
        datadir / "ch6_option_a_1d_recovery_temperature.csv", index=False)
    fig, ax = S.figure(7.0)
    for i in np.flatnonzero(Q < 0):
        ax.axvspan(t[i] - 0.5 / 52, t[i] + 0.5 / 52, color="#ececec", linewidth=0, zorder=0)
    ax.plot(t, T, color=S.OPT_COLOR["A1D"], zorder=3)
    for y, lab, ls in ((t_inj, f"Injection {t_inj:.0f} °C", "--"), (90.0, "Minimum grid supply 90 °C", ":"), (t_ret, f"Return {t_ret:.0f} °C", "--")):
        ax.axhline(y, color=S.MID, linestyle=ls, linewidth=S.LW_THIN, zorder=2)
        ax.text(7.05, y, lab, va="center", fontsize=S.FS_SMALL, color=S.MUTED, clip_on=False)
    S.despine(ax)
    ax.set_xlim(0, 7); ax.set_ylim(15, 115)
    ax.set_xlabel("Simulation year (year 7 = reference year)"); ax.set_ylabel("Warm-well temperature (°C)")
    ax.legend([Patch(color="#ececec")], ["Discharge weeks"], loc="lower right")
    S.save(fig, figdir, "ch6_option_a_1d_recovery_temperature")
    return F("ch6_option_a_1d_recovery_temperature", depends="yes (final A1D run)",
             source="data/rosim/…A1D…/ATESEfficiency.txt (final run)", note="Final run; the printed figure used an earlier run.")


LINES_64 = [("A1D", "recomputed", "Option A 1D", "A1D", "-", "o"), ("A2D", "recomputed", "Option A 2D", "A2D", "-", "s"),
            ("A4D", "recomputed", "Option A 4D", "A4D", "-", "^"), ("B-WN110", "cleaned", "B-WN 110 °C", "BWN110", "--", "D"),
            ("B-std WN", "recomputed", "B std WN 90 °C", "Bstd", "--", "v"), ("B-std NW", "recomputed", "B std NW 90 °C", "Bstd", ":", "P")]
LOCK_EFF = {"A1D": "Option A 1D efficiency year 7 (basis R)", "A2D": "Option A 2D efficiency year 7 (basis R)",
            "A4D": "Option A 4D efficiency year 7 (basis R)", "B-WN110": "B-WN110 efficiency year 7 (basis R)",
            "B-std WN": "B standard WN efficiency year 7 (basis R)", "B-std NW": "B standard NW efficiency year 7 (basis R)"}


def fig_6_4(figdir, datadir):
    rc = pd.read_csv(C.RES_ROSIM / "rosim_check.csv")
    rows, ends, chk = [], [], []
    fig, ax = S.figure(8.0)
    for run, col, lab, o, ls, mk in LINES_64:
        g = rc[(rc.run == run) & rc.year.between(2, 7)].sort_values("year")
        ax.plot(g.year, g[col], color=S.OPT_COLOR[o], linestyle=ls, marker=mk, markersize=3.5, markeredgecolor="white")
        rows += [dict(run=run, basis=col, year=int(y), efficiency_pct=v) for y, v in zip(g.year, g[col])]
        ends.append((float(g[col].iloc[-1]), lab, S.OPT_COLOR[o]))
        chk.append((f"{run} year 7", float(g[col].iloc[-1]), LOCK_EFF[run], 2))
    pd.DataFrame(rows).to_csv(datadir / "ch6_rosim_convergence_all.csv", index=False)
    ys = S.spread_labels([e[0] for e in ends], 1.05)
    for (y0, lab, c), y in zip(ends, ys):
        ax.annotate(f"{lab} ({y0:.1f}%)", (7, y0), xytext=(7.12, y), textcoords="data", va="center", fontsize=S.FS_SMALL, color=c,
                    annotation_clip=False)
    ax.axvline(7, color=S.MID, linestyle=":", linewidth=S.LW_THIN)
    S.despine(ax)
    ax.set_xlim(1.8, 9.3); ax.set_xticks(range(2, 8))
    ax.set_xlabel("Simulation year (year 7 = reference year)"); ax.set_ylabel("Thermal recovery efficiency (%)")
    ax.spines["bottom"].set_bounds(1.8, 7.2)
    S.save(fig, figdir, "ch6_rosim_convergence_all")
    return F("ch6_rosim_convergence_all", chk, source="results/rosim/rosim_check.csv (basis R; B-WN110 cleaned)",
             note="Solid: Option A; dashed/dotted: Option B. Labels give the year-7 value.")


def fig_6_5(figdir, datadir):
    order = ["A1D", "A2D", "A4D", "BWN110", "Bstd", "TTES5k"]
    lk = {"A1D": "Option A 1D", "A2D": "Option A 2D", "A4D": "Option A 4D", "BWN110": "B-WN110", "Bstd": "B standard"}
    d = pd.DataFrame([dict(option=o, S_ref=RN.s_ref(), S_flex=RN.s_flex(), storage_before=RN.storage_before(o),
                           storage_after=RN.storage_after(o)) for o in order])
    d["total"] = d.S_ref + d.S_flex + d.storage_after
    d.to_csv(datadir / "ch6_saving_decomposition.csv", index=False)
    fig, (a1, a2) = S.figure(7.5, 1, 2, gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(d))
    a1.bar(x, d.S_ref, 0.62, color=S.DARK, label="Dispatch-only reference")
    a1.bar(x, d.S_flex, 0.62, bottom=d.S_ref, color=S.LIGHT, label="Relaxed schedule")
    a1.bar(x, d.storage_after, 0.62, bottom=d.S_ref + d.S_flex, color=[S.OPT_COLOR[o] for o in d.option])
    for i, v in enumerate(d.total):
        a1.text(i, v + 30, f"{v:,.0f}", ha="center", va="bottom", fontsize=S.FS_SMALL - 0.5)
    a1.set_ylim(0, 2450); a1.set_ylabel("Annual saving after calibration (k€/yr)")
    a1.legend(handles=[Patch(color=S.DARK), Patch(color=S.LIGHT), Patch(color=S.MID)],
              labels=["Dispatch-only reference", "Relaxed schedule", "Storage effect (option colour)"], loc="lower center",
              fontsize=S.FS_SMALL - 0.5, bbox_to_anchor=(0.5, 1.0), ncol=1)
    w = 0.38
    for i, r in d.iterrows():
        c = S.OPT_COLOR[r.option]
        if r.option != "TTES5k":
            a2.bar(i - w / 2, r.storage_before, w, color=c, alpha=0.35)
            a2.text(i - w / 2, r.storage_before + 6, f"{r.storage_before:.0f}", ha="center", va="bottom", fontsize=S.FS_SMALL - 1)
            xa = i + w / 2
        else:
            xa = i
        a2.bar(xa, r.storage_after, w, color=c)
        a2.text(xa, r.storage_after + 6, f"{r.storage_after:.0f}", ha="center", va="bottom", fontsize=S.FS_SMALL - 1)
    a2.set_ylim(0, 100 * (int(d.storage_before.max() * 1.2) // 100 + 1))
    a2.set_ylabel("Storage effect (k€/yr)")
    a2.legend(handles=[Patch(color=S.MID, alpha=0.35), Patch(color=S.MID)], labels=["Dispatch model (before calibration)",
              "After ROSIM calibration"], loc="lower center", fontsize=S.FS_SMALL - 0.5, bbox_to_anchor=(0.5, 1.0))
    for a in (a1, a2):
        S.despine(a)
        a.set_xticks(x); a.set_xticklabels([S.OPT_SHORT[o].replace("\n5,000 m³", "") for o in d.option], fontsize=S.FS_SMALL)
    S.save(fig, figdir, "ch6_saving_decomposition")
    chk = [(f"storage after calibration {o}", RN.storage_after(o), f"{lk[o]} storage effect after calibration", 1) for o in order[:-1]]
    return F("ch6_saving_decomposition", chk, depends="yes (f_corr)", source="final numbers (report_numbers.py)")


TIERS71 = [("lower", "Storage effect only"), ("base", "Base case"), ("potential", "Dispatch potential credited")]


def lock_npv(o, tier):
    lk = {"A1D": "Option A 1D", "A2D": "Option A 2D", "A4D": "Option A 4D", "BWN110": "B-WN110", "Bstd": "B standard"}
    return f"TTES NPV {tier}, case B" if o == "TTES5k" else f"{lk[o]} NPV {tier} (mid CAPEX)"


def fig_7_1(figdir, datadir):
    order = RN.OPTIONS
    d = pd.DataFrame([dict(option=o, **{t: RN.npv(o, t) for t, _ in TIERS71}) for o in order])
    d.to_csv(datadir / "ch7_npv_tiers.csv", index=False)
    fig, ax = S.figure(8.5)
    x = np.arange(len(d))
    w = 0.27
    for j, (t, lab) in enumerate(TIERS71):
        for i, o in enumerate(d.option):
            c = S.OPT_COLOR[o]
            kw = dict(color="white", edgecolor=c, linewidth=1.0) if t == "lower" else \
                dict(color=c) if t == "base" else dict(color=c, hatch="////", edgecolor="white", linewidth=0)
            v = d.loc[i, t]
            ax.bar(i + (j - 1) * w, v, w * 0.95, **kw)
            txt = RN.fmt_signed(v, 2 if abs(v) < 1 else 1)
            ax.text(i + (j - 1) * w, v + (0.6 if v >= 0 else -0.6), txt, rotation=90, ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=S.FS_SMALL - 1)
    ax.axhline(0, color=S.INK, linewidth=S.LW_THIN)
    S.despine(ax)
    ax.set_xticks(x); ax.set_xticklabels([S.OPT_SHORT[o] for o in d.option], fontsize=S.FS_SMALL)
    ax.set_ylabel("NPV at mid CAPEX (M€)")
    lo, hi = d[["lower", "base", "potential"]].min().min(), d[["lower", "base", "potential"]].max().max()
    ax.set_ylim(lo - 7, hi + 7)
    ax.legend(handles=[Patch(facecolor="white", edgecolor=S.DARK), Patch(color=S.DARK), Patch(facecolor=S.DARK, hatch="////", edgecolor="white")],
              labels=[l for _, l in TIERS71], loc="upper right", fontsize=S.FS_SMALL)
    S.save(fig, figdir, "ch7_npv_tiers")
    chk = [(f"NPV {t} {o}", d.set_index("option").loc[o, t], lock_npv(o, t), 2) for o in order for t, _ in TIERS71]
    return F("ch7_npv_tiers", chk, source="final numbers (report_numbers.npv)")


ROWS72 = [("Full dispatch-only reference credited", "full S_ref credited (potential)", None),
          ("Relaxed schedule not credited", "relaxed schedule not credited (lower)", None),
          ("Only bridged schedule value credited (2 h)", "bridged, R = 2 h", None),
          ("ROSIM correction on whole increment", "ROSIM correction on the whole increment", None),
          ("Discount rate 6.0% / 2.65% (real)", "rate 6.0 %", "rate 2.65 %"),
          ("CAPEX high / low", "CAPEX high", "CAPEX low"),
          ("Savings −1% / +1% per year", "growth -1 %/yr", "growth +1 %/yr"),
          ("OPEX +50% / −50%", "OPEX +50 %", "OPEX -50 %"),
          ("Heat demand +20% / −20%", "demand +20 %", "demand -20 %"),
          ("Price spread ×0.75 / ×1.25", "spread x0.75", "spread x1.25"),
          ("Pump replacement included", "pump replacement", None),
          ("Buffers used as by AEB (n = 2)", "buffers n = 2", None)]


def tornado_bars():
    import fig_7_2_tornado as T72
    import scenarios as SC
    D = SC.decompose()
    sets, share = SC.inputs()
    out = {}
    for o in RN.OPTIONS:
        nb, b = T72.bars(D, sets["new_R_B"], share, o)
        b = dict(b)
        b["pump replacement"] = -RN.pump_replacement_pv_meur(o) if o != "TTES5k" else np.nan
        b["buffers n = 2"] = (RN.buffer_npv(o, "n2") - nb) if (o in ("A1D", "TTES5k") and RN.buffer_cases_available()) else np.nan
        out[o] = (nb, b)
    return out


def _tornado_panel(ax, o, nb, b, labels=True, fs=S.FS_SMALL):
    n = len(ROWS72)
    span = max(abs(nb + v) for v in b.values() if v == v) or 1
    for i, (label, k1, k2) in enumerate(ROWS72):
        y = n - 1 - i
        ks = [k for k in (k1, k2) if k is not None and b.get(k) == b.get(k)]
        for k in sorted(ks, key=lambda k: -abs(b[k])):
            ax.barh(y, b[k], left=nb, height=0.6, color=S.LOWER if b[k] < 0 else S.RAISE)
        vals = [b[k] for k in ks if abs(b[k]) > 1e-9]
        if not ks:
            ax.text(nb, y, "  n/a", va="center", fontsize=fs - 1, color=S.MUTED)
        if len(vals) == 2 and (vals[0] < 0) == (vals[1] < 0):
            v = max(vals, key=abs)
            ax.text(nb + v + (0.015 if v >= 0 else -0.015) * span, y, " / ".join(RN.fmt_signed(nb + x) for x in vals), va="center",
                    ha="left" if v >= 0 else "right", fontsize=fs - 1)
        else:
            for v in vals:
                ax.text(nb + v + (0.015 if v >= 0 else -0.015) * span, y, RN.fmt_signed(nb + v), va="center",
                        ha="left" if v >= 0 else "right", fontsize=fs - 1)
    ax.axvline(nb, color=S.OPT_COLOR[o], linewidth=1.2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in ROWS72][::-1] if labels else [""] * n, fontsize=fs)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=S.GRID, linewidth=0.6); ax.set_axisbelow(True)
    vals = [nb + v for v in b.values() if v == v] + [nb]
    pad = 0.22 * (max(vals) - min(vals))
    ax.set_xlim(min(vals) - pad, max(vals) + pad)
    ax.set_ylim(-0.7, n - 0.3)


def fig_7_2(figdir, datadir):
    bars = tornado_bars()
    rows = []
    for o, (nb, b) in bars.items():
        rows.append(dict(option=o, bar="base case", change_meur=0.0, npv_meur=nb))
        for _, k1, k2 in ROWS72:
            for k in (k1, k2):
                if k:
                    rows.append(dict(option=o, bar=k, change_meur=b[k], npv_meur=nb + b[k] if b[k] == b[k] else np.nan))
    pd.DataFrame(rows).to_csv(datadir / "ch7_npv_tornado.csv", index=False)
    fig, ax = S.figure(9.5)
    nb, b = bars["A1D"]
    _tornado_panel(ax, "A1D", nb, b, fs=S.FS)
    ax.set_xlabel(f"NPV of Option A 1D at mid CAPEX (M€); base case {RN.fmt_signed(nb)}")
    ax.legend([Patch(color=S.LOWER), Patch(color=S.RAISE)], ["Lowers NPV", "Raises NPV"], loc="lower right", fontsize=S.FS_SMALL)
    S.save(fig, figdir, "ch7_npv_tornado_1d")
    fig, axs = S.figure(21.0, 3, 2)
    for k, (ax, o) in enumerate(zip(axs.flat, RN.OPTIONS)):
        nb, b = bars[o]
        _tornado_panel(ax, o, nb, b, labels=(k % 2 == 0), fs=S.FS_SMALL - 1)
        ax.set_title(f"{S.OPT_LABEL[o]} (base {RN.fmt_signed(nb)} M€)", fontsize=S.FS_SMALL, loc="left", color=S.OPT_COLOR[o])
        ax.set_xlabel("NPV at mid CAPEX (M€)", fontsize=S.FS_SMALL - 1)
        ax.tick_params(axis="x", labelsize=S.FS_SMALL - 1)
    fig.legend([Patch(color=S.LOWER), Patch(color=S.RAISE)], ["Lowers NPV", "Raises NPV"], loc="outside lower center", ncol=2,
               fontsize=S.FS_SMALL)
    S.save(fig, figdir, "ch7_npv_tornado_all_options")
    tor = pd.read_csv(C.RES_SCEN / "tornado_a1d.csv", index_col=0)["new_R_B"]
    nb, b = bars["A1D"]
    chk = [("A1D base NPV", nb, lock_npv("A1D", "base"), 2)]
    chk += [(f"A1D bar {k}", b[k], ("tornado_a1d.csv", float(tor[k])), 3) for k in tor.index if k != "npv_base"]
    return F("ch7_npv_tornado_1d", chk, source="scenarios.py functions (fig_7_2_tornado.bars); pump replacement App. B.3; "
             "buffers n = 2 from data/derived/buffer_use_cache",
             note="Second file ch7_npv_tornado_all_options shows every option; the buffer bar exists for A1D and TTES only "
                  "(the buffer runs of Table B.8).")


# ============================================================================== aggregated results of dispatch runs
# Figures 3.1-3.3 show the hourly AEB buffer data and the EPEX price series. They need the raw data and are not part of
# the package (Appendix C). Figures 6.2 and 6.3 are drawn from aggregates of the hourly dispatch runs that are included
# in data/: the weekly state of charge (6.2) and the histogram bins of the hourly charge and discharge power (6.3).
def _not_in_package(figdir, datadir):
    raise RuntimeError("this figure needs the raw AEB data and is not part of the package")


def fig_6_2(figdir, datadir):
    """Figure 6.2: weekly state of charge of Option A 4D (P30/P60 + SOC rule, relaxed, 2023-2025): weekly mean line,
    min-max band and weekly mean seasonal target (data/soc_weekly.csv, weekly aggregate of the hourly dispatch run)."""
    pd.read_csv(C.SOC_WEEKLY, dtype=str, keep_default_na=False).to_csv(datadir / "ch6_soc_option_a_final_weekly.csv", index=False)
    w = pd.read_csv(C.SOC_WEEKLY, parse_dates=["week_start"]).set_index("week_start")
    fig, axs = S.figure(13.0, 3, 1)
    col = S.OPT_COLOR["A4D"]
    for ax, year in zip(axs, (2023, 2024, 2025)):
        y = w[w.index.year == year]
        ax.fill_between(y.index, y.soc_min_pct, y.soc_max_pct, color=col, alpha=0.25, linewidth=0, step="post")
        ax.step(y.index, y.soc_mean_pct, where="post", color=col, linewidth=0.9)
        ax.step(y.index, y.target_mean_pct, where="post", color=S.DARK, linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 108); ax.set_ylabel("SOC (%)")
        ax.text(0.01, 0.92, f"{year}: peak weekly max SOC {y.soc_max_pct.max():.0f}%", transform=ax.transAxes, fontsize=S.FS_SMALL, va="top")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        S.despine(ax)
    axs[0].legend(handles=[Patch(color=col, alpha=0.4), Patch(facecolor="white", edgecolor=S.DARK, linestyle="--")],
                  labels=["Weekly SOC (mean line, min-max band; % of maximum aquifer capacity)", "Seasonal target (weekly mean)"],
                  loc="lower right", fontsize=S.FS_SMALL - 0.5, ncol=2, bbox_to_anchor=(1, 1.0))
    S.save(fig, figdir, "ch6_soc_option_a_final_weekly")
    return F("ch6_soc_option_a_final_weekly", status="final numbers (weekly aggregate)", depends="no (dispatch of Option A 4D)",
             source="data/soc_weekly.csv (weekly aggregate of the hourly dispatch run of Option A 4D)",
             note="Weekly mean line, min-max band and weekly mean seasonal target; no hourly data.")


def fig_6_3(figdir, datadir):
    """Figure 6.3: distribution of the hourly charge and discharge power of the unconstrained dispatch (150 MW), drawn
    from its histogram bins (data/power_distribution_bins.csv: 40 bins per series, hours per bin, 2023-2025)."""
    pd.read_csv(C.POWER_BINS, dtype=str, keep_default_na=False).to_csv(datadir / "ch7_power_distribution_13mw.csv", index=False)
    b = pd.read_csv(C.POWER_BINS)
    cap, esp = 13.3, 150 * 52.0 * 0.001157        # Thiem limit of one doublet (221 m3/h); 150 m3/h at 52 K
    fig, axs = S.figure(6.5, 1, 2)
    for ax, lab in zip(axs, ("Charging", "Discharging")):
        s = b[b.series == lab]
        ax.bar(s.bin_left_mw, s.hours, width=s.bin_right_mw - s.bin_left_mw, align="edge", color=S.DARK, edgecolor="white", linewidth=0.3)
        ax.axvline(cap, color=S.OPT_COLOR["A1D"], linestyle="--", linewidth=1.0)
        ax.axvline(esp, color=S.MID, linestyle=":", linewidth=1.0)
        ax.text(0.40, 0.97, f"dashed: Thiem limit {cap} MW", transform=ax.transAxes, fontsize=S.FS_SMALL - 0.5, va="top",
                color=S.OPT_COLOR["A1D"])
        ax.text(0.40, 0.89, f"dotted: 150 mÂ³/h, {esp:.1f} MW", transform=ax.transAxes, fontsize=S.FS_SMALL - 0.5, va="top", color=S.MID)
        ax.set_xlabel(f"{lab} power (MW; {int(s.hours.sum()):,} h)"); ax.set_ylabel("Hours per bin, 2023â€“2025 (h)")
        S.despine(ax)
    S.save(fig, figdir, "ch7_power_distribution_13mw")
    return F("ch7_power_distribution_13mw", status="final numbers (histogram of the dispatch run)", depends="no (dispatch at 150 MW)",
             source="data/power_distribution_bins.csv (histogram bins of the hourly dispatch run)",
             note="Dashed: single-doublet Thiem limit 13.3 MW; dotted: 150 mÂ³/h at 52 K.")


REPO = {"Figure 5.1": fig_5_1, "Figure 6.1": fig_6_1, "Figure 6.2": fig_6_2, "Figure 6.3": fig_6_3, "Figure 6.4": fig_6_4,
        "Figure 6.5": fig_6_5, "Figure 7.1": fig_7_1, "Figure 7.2": fig_7_2, "Figure B.1": fig_b_1}
RAW = {"Figure 3.1": _not_in_package, "Figure 3.2": _not_in_package, "Figure 3.3": _not_in_package}   # raw AEB data: not drawn
