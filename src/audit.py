#!/usr/bin/env python3
"""Readers and efficiency definitions for the ROSIM runs, used as a library by rosim_check.py and the other ROSIM
scripts."""
import datetime as dt
import math
import sys
import numpy as np
import pandas as pd
from audit_lib import (HERE, INPUTS, RUNS, TEMPS, WEEKS, H_WEEK, read_settings, read_deck_props, read_workbook,
                       read_wells, project_dir, dispatch, _rosim_text)

sys.stdout.reconfigure(encoding="utf-8")
import build_rosim_inputs as b  # noqa: E402

RHO = 1046.9142626824998
CP_DECK = {110.0: 3892.308650728968, 90.0: 3874.6101748438413}
START = dt.date(2024, 12, 30)
Y2, Y8 = 1, 8                     # workbook years 2..8 are rows 52..415
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# user's numbers from the ROSIM outputs (cycle year 2), only for checking
USER = {"A1D": (34, 17, 1, 72.5, 53.4, 35.7), "A2D": (36, 15, 1, 60.7, 79.1, 31.2), "A4D": (32, 14, 1, 54.7, 62.7, 23.9),
        "B-WN110": (32, 17, 3, 76.6, 77.0, 31.0), "B-std WN": (33, 19, 6, 103.8, 91.1, 40.8), "B-std NW": (32, 18, 4, 108.7, 96.4, 41.9)}
USER_VOL = {"A1D": (202040, 109507), "A2D": (167755, 105200), "A4D": (120626, 88192), "B-WN110": (161782, 109035),
            "B-std WN": (218862, 137216), "B-std NW": (215099, 151348)}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def k_m3h_per_mw(run):
    t_in, t_ret = TEMPS[run]
    return b.m3h_per_mw(RHO, CP_DECK[t_in], t_in - t_ret)


def month_split(dates, vol):
    """Split weekly volumes over calendar months by days (each week = 7 days from its Monday)."""
    out = np.zeros(12)
    for d, v in zip(dates, vol):
        for k in range(7):
            out[(d + dt.timedelta(days=k)).month - 1] += v / 7.0
    return out


def shares(x):
    s = x.sum()
    return x / s * 100 if s > 0 else x * 0


def direct_sign_changes(q):
    return int(((q[1:] * q[:-1]) < 0).sum())


def blocks(q):
    """Runs of consecutive weeks with the same nonzero sign."""
    s = np.sign(q)
    n, prev = 0, 0
    for x in s:
        if x != 0 and x != prev:
            n += 1
        prev = x
    return n


def year_stats(q):
    inj, ext = q[q > 0].sum() * H_WEEK, -q[q < 0].sum() * H_WEEK
    nz = np.abs(q[q != 0])
    return dict(v_inj=inj, v_ext=ext, ratio=ext / inj, w_inj=int((q > 0).sum()), w_ext=int((q < 0).sum()),
                w_idle=int((q == 0).sum()), sign_changes=direct_sign_changes(q), blocks=blocks(q),
                w_below10=int(((q != 0) & (np.abs(q) < 10)).sum()), peak_inj=q.max(), peak_ext=-q.min(),
                mean_abs=np.abs(q).mean(), min_nz=nz.min(), max_nz=nz.max())


# ---------------------------------------------------------------------------------------------
# Part A: workbooks
# ---------------------------------------------------------------------------------------------
def part_a():
    rows, monthly, yr5, check = [], {}, {}, []
    for run, c in RUNS.items():
        sh, meta, _ = read_workbook(INPUTS / c["wb"])
        w = sh["Warmwell"]
        q = w.flow.values.astype(float)
        dates = [dt.date(int(a), int(m_), int(d)) for a, m_, d in w[["year", "month", "day"]].values]
        per = [year_stats(q[i * WEEKS:(i + 1) * WEEKS]) for i in range(Y2, Y8)]
        st = {k: np.mean([p[k] for p in per]) for k in per[0]}
        st.update({f"{k}_min": min(p[k] for p in per) for k in ("sign_changes", "w_inj", "w_ext")})
        st.update({f"{k}_max": max(p[k] for p in per) for k in ("sign_changes", "w_inj", "w_ext")})
        span = q[WEEKS:8 * WEEKS]
        st["max_jump"] = float(np.abs(np.diff(span)).max())
        st["years_identical"] = all(np.array_equal(q[i * WEEKS:(i + 1) * WEEKS], q[7 * WEEKS:8 * WEEKS]) for i in range(2, 8))
        d27 = dates[WEEKS:8 * WEEKS]
        mi = month_split(d27, np.clip(span, 0, None) * H_WEEK)
        me = month_split(d27, np.clip(-span, 0, None) * H_WEEK)
        monthly[run] = (shares(mi), shares(me))
        rows.append(dict(run=run, workbook=c["wb"], **st))
        yr5[run] = (dates[4 * WEEKS:5 * WEEKS], q[4 * WEEKS:5 * WEEKS])
        # check against the author's ROSIM numbers for cycle year 2 (they match workbook year 2, rows 52-103)
        cy = q[52:104]
        s2 = year_stats(cy)
        u = USER[run]
        qe = month_split(dates[52:104], np.clip(-cy, 0, None) * H_WEEK)
        check.append(dict(run=run, w_inj=s2["w_inj"], user_w_inj=u[0], w_ext=s2["w_ext"], user_w_ext=u[1],
                          sign_changes=s2["sign_changes"], user_sign_changes=u[2], peak_inj=s2["peak_inj"], user_peak_inj=u[3],
                          peak_ext=s2["peak_ext"], user_peak_ext=u[4], mean_abs=s2["mean_abs"], user_mean_abs=u[5],
                          v_inj=st["v_inj"], user_v_inj=USER_VOL[run][0], v_ext=st["v_ext"], user_v_ext=USER_VOL[run][1],
                          ext_quarters=str(np.round([shares(qe)[i:i + 3].sum() for i in (0, 3, 6, 9)], 0).tolist())))
    A = pd.DataFrame(rows)
    A.to_csv(HERE / "partA_workbooks.csv", index=False, float_format="%.3f")
    pd.DataFrame(check).to_csv(HERE / "partA_check_vs_rosim_output.csv", index=False, float_format="%.2f")
    M = pd.DataFrame({f"{run} {lab}": s for run, (i, e) in monthly.items() for lab, s in (("inj", i), ("ext", e))},
                     index=MONTHS).T
    M.to_csv(HERE / "partA_monthly_shares.csv", float_format="%.1f")
    fig_year5(yr5)
    return A, monthly, pd.DataFrame(check)


def fig_year5(yr5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, surf = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True, sharex=True, facecolor=surf)
    for ax, (run, (d, q)) in zip(axes.flat, yr5.items()):
        x = np.arange(len(q))
        ax.bar(x, np.clip(q, 0, None), width=0.85, color="#2a78d6", label="injection")
        ax.bar(x, np.clip(q, None, 0), width=0.85, color="#eb6834", label="extraction")
        ax.axhline(0, color=ink2, lw=0.8)
        ax.set_title(run, loc="left", color=ink, fontsize=11)
        ax.set_facecolor(surf)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ticks = [i for i in range(1, len(d)) if d[i].month != d[i - 1].month]
        ax.set_xticks(ticks, [MONTHS[d[i].month - 1] for i in ticks], fontsize=8, color=ink2)
        ax.tick_params(colors=ink2, length=0)
        ax.yaxis.grid(True, color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("warm-well flow (m3/h)", color=ink2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Weekly warm-well flow in year 5 (2029), injection positive", x=0.01, ha="left", color=ink)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(HERE / "fig_partA_year5_flows.png", dpi=170, facecolor=surf)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Part B: workbook against dispatch
# ---------------------------------------------------------------------------------------------
def meta_value(meta, key):
    return dict(meta)[key]


def part_b(monthly_wb):
    rows, mon = [], {}
    for run, c in RUNS.items():
        ch, dis = dispatch(run)
        k, n = k_m3h_per_mw(run), c["n"]
        qin, qex = ch * k / n, dis * k / n                       # m3/h per doublet, gross
        v_in, v_ex = qin.sum() / 3, qex.sum() / 3                 # m3/yr
        di = qin.groupby(qin.index.month).sum().values / 3
        de = qex.groupby(qex.index.month).sum().values / 3
        # builder input after netting: weekly_pattern_a (Option A) or weekly_pattern_b (Option B), on the year-1 dates
        net = qin - qex
        pat = b.weekly_pattern_a(net) if run.startswith("A") else b.weekly_pattern_b(net)
        pdates = [START + dt.timedelta(weeks=i) for i in range(WEEKS)]
        ni = month_split(pdates, np.clip(pat, 0, None) * H_WEEK)
        ne = month_split(pdates, np.clip(-pat, 0, None) * H_WEEK)
        # hours with charge and discharge in the same month: what monthly netting cancels
        mnet = net.groupby([net.index.year, net.index.month]).sum()
        gross = (qin + qex).groupby([net.index.year, net.index.month]).sum()
        cancelled = 1 - mnet.abs().sum() / gross.sum()
        wk = net.groupby([net.index.isocalendar().year, net.index.isocalendar().week]).sum()
        wgross = (qin + qex).groupby([net.index.isocalendar().year, net.index.isocalendar().week]).sum()
        cancelled_w = 1 - wk.abs().sum() / wgross.sum()
        _, meta, _ = read_workbook(INPUTS / c["wb"])
        wi, we = monthly_wb[run]
        mon[run] = dict(disp_inj=shares(di), disp_ext=shares(de), net_inj=shares(ni), net_ext=shares(ne), wb_inj=wi, wb_ext=we)
        # timing difference: total variation distance between dispatch and workbook monthly shares (0 = same, 100 = disjoint)
        rows.append(dict(run=run, disp_v_inj=v_in, disp_v_ext=v_ex, disp_week_inj=v_in / 52, disp_week_ext=v_ex / 52,
                         pattern_before_smoothing=meta_value(meta, "Weekly pattern before smoothing, inj / ext"),
                         dispatch_volume=meta_value(meta, "Dispatch volume, inj / ext"),
                         rescale_unsmoothed=meta_value(meta, "Rescale factors inj / ext, unsmoothed pattern to dispatch"),
                         rescale_smoothed=meta_value(meta, "Rescale factors inj / ext, smoothed year to dispatch"),
                         smoothing=meta_value(meta, "Smoothing"), jump_limiter=meta_value(meta, "Jump limiter"),
                         aggregation=meta_value(meta, "Weekly aggregation"),
                         cancelled_by_monthly_netting_pct=cancelled * 100, cancelled_by_weekly_netting_pct=cancelled_w * 100,
                         tv_inj_disp_vs_net=0.5 * np.abs(shares(di) - shares(ni)).sum(),
                         tv_inj_net_vs_wb=0.5 * np.abs(shares(ni) - wi).sum(),
                         tv_inj_disp_vs_wb=0.5 * np.abs(shares(di) - wi).sum(),
                         tv_ext_disp_vs_net=0.5 * np.abs(shares(de) - shares(ne)).sum(),
                         tv_ext_net_vs_wb=0.5 * np.abs(shares(ne) - we).sum(),
                         tv_ext_disp_vs_wb=0.5 * np.abs(shares(de) - we).sum(),
                         hourly_peak_inj=float(qin.max()), hourly_peak_ext=float(qex.max())))
    B = pd.DataFrame(rows)
    B.to_csv(HERE / "partB_dispatch_vs_workbook.csv", index=False, float_format="%.3f")
    out = []
    for run, m in mon.items():
        for key, v in m.items():
            out.append(dict(run=run, series=key, **dict(zip(MONTHS, v))))
    pd.DataFrame(out).to_csv(HERE / "partB_monthly_shares.csv", index=False, float_format="%.1f")
    fig_monthly(mon)
    return B, mon


def fig_monthly(mon, tag="", names=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, surf = "#0b0b0b", "#52514e", "#fcfcfb"
    series = [("disp", "dispatch (gross, no netting)", "#2a78d6"), ("net", "builder input after netting", "#1baf7a"),
              ("wb", "workbook years 2-8", "#eb6834")]
    for run, m in mon.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True, facecolor=surf)
        for ax, leg, title in zip(axes, ("inj", "ext"), ("Injection", "Extraction")):
            x = np.arange(12)
            have = [s for s in series if f"{s[0]}_{leg}" in m]
            wdt = 0.8 / len(have)
            for j, (key, lab, col) in enumerate(have):
                ax.bar(x - 0.4 + wdt * (j + 0.5), m[f"{key}_{leg}"], width=wdt * 0.92, color=col, label=lab)
            ax.set_xticks(x, MONTHS, color=ink2, fontsize=8)
            ax.set_title(f"{title} share by month", loc="left", color=ink, fontsize=10)
            ax.set_facecolor(surf)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=ink2, length=0)
            ax.yaxis.grid(True, color="#e4e3df", lw=0.8)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("% of yearly volume", color=ink2)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
        fig.suptitle((names or {}).get(run, run), x=0.01, ha="left", color=ink)
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fn = run.replace(" ", "_").replace("-", "")
        fig.savefig(HERE / f"fig_partB_monthly_{fn}{tag}.png", dpi=150, facecolor=surf)
        plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Part E: efficiency with the dispatch reference temperature
# ---------------------------------------------------------------------------------------------
def read_eff_full(name):
    """ATESEfficiency.txt of a ROSIM run (from data/rosim/<code>.csv, same text): weekly total and yearly efficiency, and
    per well the columns T, Tavg, Q and E."""
    t = _rosim_text(name)
    wells = [c[:-len("__T")] for c in t.columns if c.endswith("__T")]
    df = pd.DataFrame({"week": [int(x) for x in t.week], "total": [float(x) for x in t.total],
                       "yearly": [float(x) if x.strip() else np.nan for x in t.yearly]})
    per = {}
    for w in wells:
        per[w] = pd.DataFrame({"T": [float(x) for x in t[f"{w}__T"]], "Tavg": [float(x) for x in t[f"{w}__Tavg"]],
                               "Q": [float(x) for x in t[f"{w}__Q"]], "E": [float(x) for x in t[f"{w}__E"]]})
    return df, per


# ROSIM year rows: week 103 = end of simulation year 2, then every 52 weeks, up to year 8 (week 415).
YEAR_END = [103, 155, 207, 259, 311, 363, 415]     # ROSIM's yearly rows, labelled years 2..8


# efficiency definition per week: over the 52-week window ending at `end`, the sum of the positive
# weekly energy (extracted) divided by the sum of the negative (injected), in percent.
def window_eff(e, end):
    s = e[end - 51:end + 1]
    return s[s > 0].sum() / -s[s < 0].sum() * 100


def part_e():
    rows, cleaned_weeks = [], {}
    for run, c in RUNS.items():
        name = c["project"]
        eff, per = read_eff_full(name)
        vox = read_wells(name)
        t_inj, t_ref = TEMPS[run]
        tot_clean = np.zeros(len(eff))
        ch = np.zeros(len(eff))
        dis = np.zeros(len(eff))
        v_in = np.zeros(len(eff))
        v_ex = np.zeros(len(eff))
        tq = np.zeros(len(eff))
        bad_all = []
        for w, d in per.items():
            v = vox[w]
            rc = (v.density * v.cp).values[:len(d)]
            bad = ((d["T"].abs() > 130) | (d["T"] < 5) | (d["Tavg"] > 130) | (d["Tavg"] < 5)).values
            bad[0] = False
            Tavg, rcc = d["Tavg"].values.copy(), rc.copy()
            if bad.any():
                good = np.flatnonzero(~bad)
                Tavg[bad] = np.interp(np.flatnonzero(bad), good, Tavg[good])
                rcc[bad] = np.interp(np.flatnonzero(bad), good, rc[good])
                bad_all += [(w, int(i)) for i in np.flatnonzero(bad)]
            E = d["E"].values.copy()
            E[bad] = -d["Q"].values[bad] * H_WEEK * rcc[bad] * (Tavg[bad] + 273.15) / 3.6e9
            tot_clean += E
            if w.startswith("warm"):
                Q = d["Q"].values
                ch += np.clip(Q, 0, None) * H_WEEK * rcc * (t_inj - t_ref) / 3.6e9
                dis += np.clip(-Q, 0, None) * H_WEEK * rcc * (Tavg - t_ref) / 3.6e9
                v_in += np.clip(Q, 0, None) * H_WEEK
                v_ex += np.clip(-Q, 0, None) * H_WEEK
                tq += np.clip(-Q, 0, None) * H_WEEK * Tavg
        cleaned_weeks[run] = bad_all
        # reproduce ROSIM's reported numbers from its own weekly totals
        rep = [window_eff(eff.total.values, e) for e in YEAR_END]
        reported = eff.set_index("week").loc[YEAR_END, "yearly"].values
        assert np.allclose(rep, reported, atol=0.011), (run, rep, reported)
        for yi, end in enumerate(YEAR_END):
            sl = slice(end - 51, end + 1)
            rows.append(dict(run=run, year=yi + 2, rosim_reported=reported[yi], rosim_cleaned=window_eff(tot_clean, end),
                             dispatch_reference=dis[sl].sum() / ch[sl].sum() * 100,
                             charged_mwh=ch[sl].sum(), discharged_mwh=dis[sl].sum(),
                             volume_ratio=v_ex[sl].sum() / v_in[sl].sum(), t_prod_mean=tq[sl].sum() / v_ex[sl].sum(),
                             temp_factor=(tq[sl].sum() / v_ex[sl].sum() - t_ref) / (t_inj - t_ref),
                             artefact_weeks_in_year=sum(1 for _, i in bad_all if end - 51 <= i <= end)))
    E = pd.DataFrame(rows)
    E.to_csv(HERE / "rosim_efficiency_per_year.csv", index=False, float_format="%.3f")
    return E, cleaned_weeks


# ---------------------------------------------------------------------------------------------
# Part F: geometry
# ---------------------------------------------------------------------------------------------
def part_f():
    rows, wells = [], []
    for run, c in list(RUNS.items()) + [("thesis B-std NW (Option B NW Nieuwste)", dict(project="Option B NW Nieuwste")),
                                        ("thesis B-std WN (Option B WN Nieuwste)", dict(project="Option B WN Nieuwste"))]:
        s = read_settings(c["project"])
        dk = read_deck_props(c["project"])
        names = s["wells"]
        x = list(map(float, s["x_coordinate_well_head"].split(";")))
        y = list(map(float, s["y_coordinate_well_head"].split(";")))
        pos = dict(zip(names, zip(x, y)))
        dist = lambda a, b_: math.dist(pos[a], pos[b_])
        pairs = [(f"warm{i}", f"cold{i}") for i in range(1, len(names) // 2 + 1)]
        wc = [dist(a, b_) for a, b_ in pairs]
        warms = [n_ for n_ in names if n_.startswith("warm")]
        colds = [n_ for n_ in names if n_.startswith("cold")]
        ww = min((dist(a, b_) for i, a in enumerate(warms) for b_ in warms[i + 1:]), default=np.nan)
        cc = min((dist(a, b_) for i, a in enumerate(colds) for b_ in colds[i + 1:]), default=np.nan)
        wc_other = min((dist(a, b_) for a in warms for b_ in colds if (a[4:] != b_[4:])), default=np.nan)
        lay = s["layers"]
        aq = lay[1]
        top = float(s["model_top_depth"].split(";")[0])
        cells = {}
        for cpd in dk["compdat"]:
            cells.setdefault(cpd[0].strip("'"), set()).add((int(cpd[1]), int(cpd[2])))
        rows.append(dict(run=run, project=c["project"], dx=float(s["x_spacing__grid_cell_size_"]), dy=float(s["y_spacing__grid_cell_size_"]),
                         nx=int(float(s["x_number_of_cells"])), ny=int(float(s["y_number_of_cells"])), nz=int(float(s["z_number_of_cells"])),
                         refinement=s["xy_grid_size_target__0_no_grid_refinement_"],
                         wells=";".join(f"{n_} ({pos[n_][0]:g}, {pos[n_][1]:g})" for n_ in names),
                         warm_cold_m=";".join(f"{v:g}" for v in wc), warm_warm_min_m=ww, cold_cold_min_m=cc, warm_cold_other_min_m=wc_other,
                         screen_md=f"{s['top_open_section_md_s__comma_separated'].split(';')[0]}-{s['base_open_section_md_s__comma_separated'].split(';')[0]}",
                         screen_m=float(s["base_open_section_md_s__comma_separated"].split(";")[0]) - float(s["top_open_section_md_s__comma_separated"].split(";")[0]),
                         aquifer_top_m=top + float(lay[0][3]), aquifer_m=float(aq[3]), aquifer_cells=int(aq[1]),
                         diameter_inch=s["diameter"], kh_mD=float(aq[9]), kv_mD=float(aq[11]), porosity=float(aq[7]),
                         rock_cond_wmk=float(aq[12]), water_cond_wmk=float(aq[13]), salinity_ppm=float(aq[14]),
                         rock_cp_jkgk=float(aq[15]), rock_density=float(aq[16]), specrock_kjm3k=dk["specrock"],
                         confining_kh_mD=float(lay[0][9]), confining_poro=float(lay[0][7]), confining_m=float(lay[0][3]),
                         fluid_density=dk["density"], cp_water_90=dk["specheat"][90.0], cp_water_110=dk["specheat"][110.0],
                         end_date=str(dt.date(1970, 1, 1) + dt.timedelta(days=float(s["simulation_end_date"]))),
                         well_cells=";".join(f"{w}:{sorted(v)}" for w, v in cells.items())))
    F = pd.DataFrame(rows)
    F.to_csv(HERE / "partF_geometry.csv", index=False)
    return F


def main():
    A, monthly, chk = part_a()
    B, mon = part_b(monthly)
    E, cw = part_e()
    F = part_f()
    pd.set_option("display.width", 250, "display.max_columns", 80, "display.max_colwidth", 120)
    print(A.round(2).T.to_string())
    print(chk.round(1).to_string(index=False))
    print(pd.read_csv(HERE / "partA_monthly_shares.csv", index_col=0).round(1).to_string())
    print(B.round(2).T.to_string())
    print(pd.read_csv(HERE / "partB_monthly_shares.csv").round(1).to_string(index=False))
    print(E.round(2).to_string(index=False))
    print("cleaned weeks:", cw)
    print(F.T.to_string())


if __name__ == "__main__":
    main()
