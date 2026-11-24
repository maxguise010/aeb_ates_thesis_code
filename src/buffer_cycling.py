"""Behaviour of the existing AEB buffers (hourly content, daily swing, model against measurement) for Figure 3.1 and
Table 3.5. Needs the raw data, not run by run_all.py."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
import config as C  # noqa: E402  (repository paths, config.py)
from config import DATA_DIR  # noqa: E402

BUFFER_CSV = os.path.join(DATA_DIR, "bufferstand_warmtevraag.csv")
MODEL_CSV = os.path.join(str(C.MODEL_OUT), "warmtelevering_per_hx_v9_final.csv")   # run_dispatch_only_v9_demand_corrected.py
OUT = os.path.join(str(C.WORK), "buffer_output")   # aggregated statistics (from AEB data)
os.makedirs(OUT, exist_ok=True)
# Figures: --fig-dir <folder> writes them there with thesis names (ch3_...); default: buffer_output/
if "--fig-dir" in sys.argv:
    FIG_DIR = os.path.abspath(sys.argv[sys.argv.index("--fig-dir") + 1])
    FIG_NAMES = dict(a="ch3_buffer_hour_profile.png", b="ch3_buffer_daily_swing_hist.png",
                     c="ch3_buffer_model_vs_measured_week.png")
else:
    FIG_DIR = OUT
    FIG_NAMES = dict(a="fig_a_hour_profile.png", b="fig_b_daily_swing_hist.png",
                     c="fig_c_winter_week_model_vs_measured.png")
os.makedirs(FIG_DIR, exist_ok=True)

VOL_M3 = 3 * 1200.0          # three buffers of 1,200 m3
RHO_CP = 0.001163            # MWh per m3 per K (water)
LOW, HIGH = 0.25, 0.75       # hysteresis thresholds for turning-point swings (share of C)
SEASONS = {"winter": [12, 1, 2], "summer": [6, 7, 8]}
TZ = "Europe/Amsterdam"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"

rows = []   # summary rows: metric, season, year, value, unit, n, definition


def add(metric, value, unit, n, definition, season="all", year="all"):
    rows.append(dict(metric=metric, season=season, year=year, value=value, unit=unit, n=n,
                     definition=definition))


def describe(series, metric, unit, ndesc, definition, season="all", year="all", qs=(("mean", None),
             ("median", 0.5), ("P10", 0.1), ("P90", 0.9))):
    s = series.dropna()
    for name, q in qs:
        v = s.mean() if q is None else s.quantile(q)
        add(f"{metric} {name}", v, unit, f"{len(s)} {ndesc}", definition, season, year)
    return s


# ═════════════════════════════════════════════════════════════════════════════
# 0. Inspect the input before computing anything
# ═════════════════════════════════════════════════════════════════════════════
print("INSPECTION — bufferstand_warmtevraag.csv")
raw = pd.read_csv(BUFFER_CSV)
print(f"  delimiter ',' , decimal '.', {len(raw):,} records, columns: {list(raw.columns)}")
utc = pd.to_datetime(raw["Datetime UTC"], format="%d/%m/%Y %H:%M").dt.tz_localize("UTC")
loc_given = pd.to_datetime(raw["Datetime CET/CEST"], format="%d/%m/%Y %H:%M")
loc = utc.dt.tz_convert(TZ)
mismatch = int((loc.dt.tz_localize(None) != loc_given).sum())
print(f"  timestamps: 'Datetime UTC' used as key, converted to {TZ}; local column disagrees in "
      f"{mismatch} records")
buf = pd.DataFrame({"content": raw["Bufferinhoud MWh"].astype(float).values,
                    "demand": raw["Warmtevraag WPW totaal MW"].astype(float).values},
                   index=pd.DatetimeIndex(utc))
dup = int(buf.index.duplicated().sum())
buf = buf[~buf.index.duplicated()]
full = pd.date_range(buf.index.min(), buf.index.max(), freq="h", tz="UTC")
missing_h = len(full) - len(buf)
step = pd.Series(buf.index).diff().mode()[0]
nan_c = int(buf["content"].isna().sum())
print(f"  period {buf.index.min()} → {buf.index.max()} (UTC), step {step}, duplicates {dup}, "
      f"missing hours {missing_h} ({missing_h / len(full) * 100:.2f} %), NaN content {nan_c}")
buf = buf.reindex(full)                                   # gaps stay NaN (not filled)
buf["local"] = buf.index.tz_convert(TZ)
buf["date"] = buf["local"].dt.date
buf["hour"] = buf["local"].dt.hour
buf["month"] = buf["local"].dt.month
buf["year"] = buf["local"].dt.year
hpd = buf.groupby("date").size()
print(f"  hours per local day: {hpd.value_counts().to_dict()} "
      f"(23 = spring DST day, 25 = autumn DST day)")
print(f"  DST days: {sorted(str(d) for d in hpd[hpd != 24].index)}")
print(f"  heat-demand column present: 'Warmtevraag WPW totaal MW' (+ per loop)")

mod = pd.read_csv(MODEL_CSV, sep=";", decimal=",")
mod.index = pd.DatetimeIndex(pd.to_datetime(mod["datetime_utc"]), tz="UTC")
mod = mod[~mod.index.duplicated()]
print(f"\nINSPECTION — warmtelevering_per_hx_v9_final.csv: ';' / decimal ',', {len(mod):,} records, "
      f"{mod.index.min()} → {mod.index.max()}; 'datetime' equals 'datetime_utc' in "
      f"{(pd.to_datetime(mod['datetime']).values == pd.to_datetime(mod['datetime_utc']).values).mean() * 100:.1f} % of rows")
# Alignment check: heat demand of both files at small lags
lag_corr = {}
for lag in range(-2, 3):
    j = buf["demand"].shift(lag).reindex(mod.index)
    lag_corr[lag] = j.corr(mod["heat_demand_mw"])
best_lag = max(lag_corr, key=lag_corr.get)
print("  demand correlation buffer file vs model file by lag (h): "
      + ", ".join(f"{k:+d}: {v:.4f}" for k, v in lag_corr.items()) + f"  -> using lag {best_lag:+d}")
if best_lag != 0:
    mod = mod.copy()
    mod.index = mod.index - pd.Timedelta(hours=best_lag)
overlap = buf.index.intersection(mod.index)
print(f"  overlap with buffer data: {len(overlap):,} hours ({overlap.min()} → {overlap.max()})")

# ═════════════════════════════════════════════════════════════════════════════
# 1. Data description and capacity
# ═════════════════════════════════════════════════════════════════════════════
c = buf["content"]
C = c.quantile(0.999)                       # capacity C = P99.9 of content (robust to spikes)
dT = C / (VOL_M3 * RHO_CP)                  # implied temperature difference over the full volume
nh = int(c.notna().sum())
for name, v in (("min", c.min()), ("max", c.max()), ("P0.1", c.quantile(0.001)), ("P99.9 = C", C)):
    add(f"content {name}", v, "MWh", f"{nh} h", "hourly buffer content, 3 x 1,200 m3")
add("implied delta T at C", dT, "K", f"{nh} h", "C / (3,600 m3 x 0.001163 MWh/(m3 K))")
add("records", len(raw), "-", "-", "rows in the buffer file")
add("missing hours", missing_h, "h", f"{len(full)} h", "hours absent from the hourly UTC grid")
add("duplicate records", dup, "-", "-", "repeated UTC timestamps")
print(f"\n1. C = P99.9 = {C:.1f} MWh (max {c.max():.1f}); implied ΔT {dT:.1f} K "
      f"({'plausible for a 90/≈45-60 °C buffer' if 25 <= dT <= 60 else 'CHECK: outside 25-60 K'}); "
      f"based on {nh} h")
frac = c / C

# helpers for daily statistics --------------------------------------------------------------------
day_ok = buf.groupby("date")["content"].count() >= 20        # require >= 20 valid hours per day


def daily(series, how):
    g = series.groupby(buf["date"])
    return g.agg(how)[day_ok]


day_meta = buf.groupby("date").agg(month=("month", "first"), year=("year", "first"))[day_ok]


def subsets(df_index_meta):
    yield "all", "all", np.ones(len(df_index_meta), bool)
    for s, months in SEASONS.items():
        yield s, "all", df_index_meta["month"].isin(months).values
    for y in sorted(df_index_meta["year"].unique()):
        yield "all", int(y), (df_index_meta["year"] == y).values


# ═════════════════════════════════════════════════════════════════════════════
# 2. Daily swing  (max − min) / C per local calendar day
# ═════════════════════════════════════════════════════════════════════════════
swing = (daily(c, "max") - daily(c, "min")) / C
for s, y, m in subsets(day_meta):
    sw = describe(swing[m], "daily swing", "share of C", "days", "(daily max − daily min) / C", s, y)
    for t in (0.5, 0.7, 0.8):
        add(f"share of days with swing >= {int(t * 100)} %", (sw >= t).mean(), "share", f"{len(sw)} days",
            "(daily max − daily min) / C >= threshold", s, y)

# ═════════════════════════════════════════════════════════════════════════════
# 3. Cycles per day
# ═════════════════════════════════════════════════════════════════════════════
# (a) equivalent full cycles: sum of positive hour-to-hour increases of content per day / C
#     (charging throughput per capacity); only consecutive hours, a gap breaks the difference.
dc = c.diff()
efc = daily(dc.clip(lower=0), "sum") / C
for s, y, m in subsets(day_meta):
    describe(efc[m], "equivalent full cycles per day", "cycles/day", "days",
             "sum of positive hourly content increases per day / C", s, y)
# (b) turning-point half-swings with hysteresis: state becomes LOW below 25 % of C and HIGH above
#     75 % of C; one half-swing is counted at each LOW->HIGH or HIGH->LOW change of state.
#     A full cycle = two half-swings.


def half_swings(frac_series):
    state, events = None, []
    for t, v in frac_series.dropna().items():
        if v < LOW and state != "low":
            if state == "high":
                events.append(t)
            state = "low"
        elif v > HIGH and state != "high":
            if state == "low":
                events.append(t)
            state = "high"
    return pd.Series(1, index=pd.DatetimeIndex(events, tz="UTC"), dtype=float)


def per_day(events, days):
    if len(events) == 0:
        return pd.Series(0.0, index=days)
    return events.groupby(events.index.tz_convert(TZ).date).sum().reindex(days, fill_value=0)


hs = half_swings(frac)
hs_day = per_day(hs, day_meta.index)
for s, y, m in subsets(day_meta):
    describe(hs_day[m], "half-swings 25↔75 % per day", "swings/day", "days",
             "state changes LOW(<25 % C) ↔ HIGH(>75 % C) with hysteresis, per day", s, y)
    add("full cycles 25↔75 % per day mean", hs_day[m].mean() / 2, "cycles/day", f"{int(m.sum())} days",
        "half-swings per day / 2", s, y)
    add("share of days with >= 1 half-swing", (hs_day[m] >= 1).mean(), "share", f"{int(m.sum())} days",
        "days with at least one 25↔75 % state change", s, y)

# ═════════════════════════════════════════════════════════════════════════════
# 4. Minimum content and reserve
# ═════════════════════════════════════════════════════════════════════════════
hour_meta = buf[["month", "year"]]
qs4 = (("P1", 0.01), ("P5", 0.05), ("P25", 0.25), ("median", 0.5))
for s, months in [("all", None)] + list(SEASONS.items()):
    mh = np.ones(len(buf), bool) if months is None else buf["month"].isin(months).values
    describe(frac[mh], "content", "share of C", "h", "hourly content / C", s, qs=qs4)
    ff = frac[mh].dropna()
    add("share of hours content < 10 % of C", (ff < 0.10).mean(), "share", f"{len(ff)} h", "content / C < 0.10", s)
    add("share of hours content > 90 % of C", (ff > 0.90).mean(), "share", f"{len(ff)} h", "content / C > 0.90", s)
    md = np.ones(len(day_meta), bool) if months is None else day_meta["month"].isin(months).values
    describe((daily(c, "min") / C)[md], "daily minimum", "share of C", "days", "daily minimum content / C", s, qs=qs4)
    res_h = (c / buf["demand"].where(buf["demand"] > 0))[mh]
    describe(res_h, "reserve", "hours of demand", "h", "content (MWh) / heat demand (MW) same hour",
             s, qs=(("P1", 0.01), ("P5", 0.05), ("median", 0.5)))
cap_mwh = 0.20 * VOL_M3 * RHO_CP * dT
model_res_h = np.minimum(6.0, cap_mwh / buf["demand"].where(buf["demand"] > 0))
add("model reserve cap", cap_mwh, "MWh", "-", "20 % of 3,600 m3 x 0.001163 x implied ΔT")
describe(model_res_h, "model reserve", "hours of demand", "h",
         "min(6 h, cap / demand): model rule next-6-h demand capped at 20 % of volume",
         qs=(("P5", 0.05), ("median", 0.5), ("P95", 0.95)))
add("model SOC limits", "10 % – 95 %", "share of volume", "-", "small_soc_min / small_soc_max")

# ═════════════════════════════════════════════════════════════════════════════
# 5. Price behaviour of the current operation (overlap with the model file only)
# ═════════════════════════════════════════════════════════════════════════════
# net charging power in hour t (MW) = content(t+1) − content(t), consecutive hours only
j = buf.loc[overlap].copy()
j["price"] = mod.loc[overlap, "EPEX_price"].values
j["p_ch"] = (c.shift(-1) - c).loc[overlap]
j["rank"] = j.groupby("date")["price"].rank(method="first")          # 1 = cheapest hour of the day
j["nday"] = j.groupby("date")["price"].transform("count")
for s, months in [("all", None)] + list(SEASONS.items()):
    jj = j if months is None else j[j["month"].isin(months)]
    cheap = jj[jj["rank"] <= 6]["p_ch"].dropna()
    dear = jj[jj["rank"] > jj["nday"] - 6]["p_ch"].dropna()
    add("mean net charging power, 6 cheapest hours", cheap.mean(), "MW", f"{len(cheap)} h",
        "mean content(t+1) − content(t) in the 6 cheapest hours of each day", s)
    add("mean net charging power, 6 most expensive hours", dear.mean(), "MW", f"{len(dear)} h",
        "mean content(t+1) − content(t) in the 6 most expensive hours of each day", s)
    v = jj[["p_ch", "rank"]].dropna()
    add("correlation charging power vs price rank", v["p_ch"].corr(v["rank"]), "Pearson r", f"{len(v)} h",
        "corr(content(t+1) − content(t), within-day price rank); negative = charges when cheap", s)

# ═════════════════════════════════════════════════════════════════════════════
# 6. Hour-of-day profile
# ═════════════════════════════════════════════════════════════════════════════
prof = {}
for s, months in SEASONS.items():
    g = frac[buf["month"].isin(months)].groupby(buf["hour"])
    prof[s] = pd.DataFrame({"mean": g.mean(), "P10": g.quantile(0.1), "P90": g.quantile(0.9), "n": g.count()})
    for h, r in prof[s].iterrows():
        add(f"content at hour {h:02d} mean", r["mean"], "share of C", f"{int(r['n'])} h", "hourly content / C by local hour", s)

# ═════════════════════════════════════════════════════════════════════════════
# 7. Comparison with the model's small_tank_soc (overlap period)
# ═════════════════════════════════════════════════════════════════════════════
msoc = mod.loc[overlap, "small_tank_soc"].astype(float)
meas = frac.loc[overlap]
ov_meta = j.groupby("date").agg(month=("month", "first"), year=("year", "first"))
ov_days = j.groupby("date")["content"].count() >= 20


def day_stats(series):
    g = series.groupby(j["date"])
    sw = (g.max() - g.min())[ov_days]
    up = series.diff().clip(lower=0).groupby(j["date"]).sum()[ov_days]
    hsd = per_day(half_swings(series), sw.index)
    return sw, up, hsd


for label, ser in (("measured content / C", meas), ("model small_tank_soc", msoc)):
    sw, up, hsd = day_stats(ser)
    for s, y, m in [("all", "all", np.ones(len(sw), bool))] + [
            (s, "all", ov_meta.loc[sw.index, "month"].isin(mm).values) for s, mm in SEASONS.items()]:
        n_d = int(m.sum())
        sub = ser if s == "all" else ser[j["month"].isin(SEASONS[s]).values]
        add(f"[{label}] mean", sub.mean(), "share", f"{sub.notna().sum()} h", "mean over hours", s)
        add(f"[{label}] P5", sub.quantile(0.05), "share", f"{sub.notna().sum()} h", "5th percentile", s)
        add(f"[{label}] P95", sub.quantile(0.95), "share", f"{sub.notna().sum()} h", "95th percentile", s)
        add(f"[{label}] daily swing median", sw[m].median(), "share", f"{n_d} days", "daily max − min", s)
        add(f"[{label}] equivalent full cycles per day mean", up[m].mean(), "cycles/day", f"{n_d} days",
            "sum of positive hourly increases per day (share units)", s)
        add(f"[{label}] half-swings 25↔75 % per day mean", hsd[m].mean(), "swings/day", f"{n_d} days",
            "hysteresis state changes per day", s)

# ═════════════════════════════════════════════════════════════════════════════
# 8. Free volume in the 20 % cheapest hours of each year
# ═════════════════════════════════════════════════════════════════════════════
for y in sorted(j["year"].unique()):
    jy = j[j["year"] == y]
    cheap = jy[jy["price"] <= jy["price"].quantile(0.20)]
    free = ((1 - cheap["content"] / C) * C).dropna()
    add("free volume in 20 % cheapest hours", free.mean(), "MWh", f"{len(free)} h",
        "mean (1 − content/C) x C over the 20 % cheapest hours of the year", year=int(y))
    add("free volume in 20 % cheapest hours", free.mean() / (RHO_CP * dT), "m3", f"{len(free)} h",
        "free MWh / (0.001163 x implied ΔT)", year=int(y))
cheap_all = j[j["price"] <= j.groupby("year")["price"].transform(lambda p: p.quantile(0.20))]
free_all = ((1 - cheap_all["content"] / C) * C).dropna()
add("free volume in 20 % cheapest hours", free_all.mean(), "MWh", f"{len(free_all)} h",
    "as above, all overlap years")
add("free volume in 20 % cheapest hours", free_all.mean() / (RHO_CP * dT), "m3", f"{len(free_all)} h",
    "as above, all overlap years")

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT, "buffer_summary.csv"), index=False)

# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════
CM = 1 / 2.54
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})


def style(a):
    a.set_facecolor("white")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(MUTED)
    a.tick_params(colors=MUTED)
    a.yaxis.grid(True, color=GRID, linewidth=0.6)
    a.set_axisbelow(True)


# (a) hour-of-day profile
fig, ax = plt.subplots(figsize=(16 * CM, 8 * CM)); fig.patch.set_facecolor("white"); style(ax)
for s, col in (("winter", BLUE), ("summer", ORANGE)):
    p = prof[s] * 100
    ax.fill_between(p.index, p["P10"], p["P90"], color=col, alpha=0.15, linewidth=0)
    ax.plot(p.index, p["mean"], color=col, linewidth=2)
    ax.text(23.3, p["mean"].iloc[-1], f"{s.capitalize()} (Dec–Feb)" if s == "winter" else "Summer (Jun–Aug)",
            color=INK, va="center", fontsize=7.5)
ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 3)); ax.set_ylim(0, 105)
ax.set_xlabel("Hour of day (local time)"); ax.set_ylabel("Buffer content (% of C)")
ax.text(0.3, 101, "Line = mean, band = P10–P90", fontsize=7, color=MUTED, va="top")
fig.tight_layout(); fig.subplots_adjust(right=0.80)
fig.savefig(os.path.join(FIG_DIR, FIG_NAMES["a"]), dpi=300, facecolor="white"); plt.close(fig)

# (b) histogram of the daily swing
fig, ax = plt.subplots(figsize=(16 * CM, 7.5 * CM)); fig.patch.set_facecolor("white"); style(ax)
ax.hist(swing * 100, bins=np.arange(0, 105, 5), color=BLUE, edgecolor="white", linewidth=1)
med = swing.median() * 100
top = ax.get_ylim()[1]
ax.set_ylim(0, top * 1.12)
ax.axvline(med, color=INK, linewidth=1)
ax.text(med, top * 1.10, f"Median {med:.0f} %", fontsize=7.5, color=INK, va="top", ha="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=1))
ax.set_xlim(0, 100); ax.set_xlabel("Daily swing (max − min) as % of C"); ax.set_ylabel("Number of days")
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, FIG_NAMES["b"]), dpi=300, facecolor="white"); plt.close(fig)

# (c) one winter week: measured content / C and model small_tank_soc
wk0 = pd.Timestamp("2024-01-15", tz=TZ).tz_convert("UTC")
wk = pd.date_range(wk0, periods=7 * 24, freq="h")
wk = wk.intersection(overlap)
fig, ax = plt.subplots(figsize=(16 * CM, 7.5 * CM)); fig.patch.set_facecolor("white"); style(ax)
xl = wk.tz_convert(TZ).tz_localize(None)
ax.plot(xl, frac.loc[wk] * 100, color=BLUE, linewidth=1.6)
ax.plot(xl, msoc.loc[wk] * 100, color=ORANGE, linewidth=1.6)
ax.text(xl[-1], frac.loc[wk].iloc[-1] * 100, "  AEB data (% of C)", color=INK, va="center", fontsize=7.5)
ax.text(xl[-1], msoc.loc[wk].iloc[-1] * 100, "  Dispatch model (% of C)", color=INK, va="center", fontsize=7.5)
ax.set_ylim(0, 105); ax.set_ylabel("Buffer state (%)")
import matplotlib.dates as mdates  # noqa: E402
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
fig.tight_layout(); fig.subplots_adjust(right=0.76)
fig.savefig(os.path.join(FIG_DIR, FIG_NAMES["c"]), dpi=300, facecolor="white"); plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# Console summary with N behind every statistic
# ═════════════════════════════════════════════════════════════════════════════
pd.set_option("display.width", 220, "display.max_colwidth", 60, "display.max_rows", 400)
keep = summary[~summary["metric"].str.startswith("content at hour")]
print("\nSUMMARY (value, unit, N):")
print(keep[["metric", "season", "year", "value", "unit", "n"]].to_string(index=False))


def g(metric, season="all", year="all"):
    r = summary[(summary.metric == metric) & (summary.season == season) & (summary.year == year)]
    return r["value"].iloc[0], r["n"].iloc[0]


v1, n1 = g("equivalent full cycles per day mean")
v2, n2 = g("half-swings 25↔75 % per day mean")
v3, n3 = g("daily swing median")
v4, n4 = g("content P5")
print("\nTHREE KEY NUMBERS")
print(f"  1. Equivalent full cycles per day (charging throughput / C): mean {v1:.2f}  [{n1}]")
print(f"  2. Full 25↔75 % cycles per day: mean {v2 / 2:.2f} (half-swings {v2:.2f})  [{n2}]")
print(f"  3. Median daily swing {v3 * 100:.0f} % of C; content below {v4 * 100:.0f} % of C in only 5 % of hours (P5)  [{n3}; {n4}]")
print(f"\nWritten: {OUT}" + (f"\nFigures: {FIG_DIR} ({', '.join(FIG_NAMES.values())})" if FIG_DIR != OUT else ""))
