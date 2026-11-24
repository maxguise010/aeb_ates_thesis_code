"""Builds the hourly input table 2023-2025 of the dispatch models from the AEB data and the EPEX prices. Needs the raw
data."""

import os
import numpy as np
import pandas as pd

from config import (
    DATA_DIR, SIM_START, SIM_END, CONST, ZERO_TOL, MIN_PHYSICAL_DELTA_T,
    hx_params, priority_order, historical_cols, derivation_factors,
)
from utils import filter_datetime_index, clean_numeric


# ── Dutch month-name helper ───────────────────────────────────────────────────

_DUTCH_TO_EN = {
    "jan": "01", "feb": "02", "mrt": "03", "apr": "04",
    "mei": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "okt": "10", "nov": "11", "dec": "12",
}

def _translate_dutch_dt(series: pd.Series) -> pd.Series:
    dt = series.astype(str).str.strip().str.lower()
    dt = dt.str.replace(" ", " ", regex=False).str.replace("/", "-", regex=False)
    for nl, num in _DUTCH_TO_EN.items():
        dt = dt.str.replace(f"-{nl}-", f"-{num}-", regex=False)
    parsed = pd.to_datetime(dt, format="%d-%m-%y %H:%M", errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(dt[mask], format="%d-%m-%y %H:%M:%S", errors="coerce")
    return parsed


# ── 1. bufferstand_warmtevraag.csv ────────────────────────────────────────────

def load_buffer_demand() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "bufferstand_warmtevraag.csv")
    cols = [
        "Bufferinhoud MWh",
        "Warmtevraag WPW totaal MW",
        "Warmtevraag NieuwWest MW",
        "Warmtevraag Noord MW",
    ]
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=";")

    df["Datetime UTC"] = pd.to_datetime(df["Datetime UTC"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Datetime UTC"]).set_index("Datetime UTC").sort_index()
    return filter_datetime_index(df, SIM_START, SIM_END)[cols].copy()


# ── 2. Warmtewisselaar.xlsx — Levering sheet (AEB delivery) ──────────────────

def load_aeb_delivery() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "Warmtewisselaar.xlsx")
    df2 = pd.read_excel(path, sheet_name="Levering", header=None)
    df_aeb = df2.iloc[:, [1, 3, 4, 6]].copy()
    df_aeb.columns = ["Datetime_UTC", "Warmtelevering AEB", "BEC_WPW", "WPW_Vattenfall_AD"]

    dutch_to_english = {
        "jan": "Jan", "feb": "Feb", "mrt": "Mar", "apr": "Apr",
        "mei": "May", "jun": "Jun", "jul": "Jul", "aug": "Aug",
        "sep": "Sep", "okt": "Oct", "nov": "Nov", "dec": "Dec",
    }

    def translate(s):
        if not isinstance(s, str):
            return s
        s = s.lower()
        for nl, en in dutch_to_english.items():
            if nl in s:
                return s.replace(nl, en)
        return s

    df_aeb["Datetime_UTC"] = pd.to_datetime(
        df_aeb["Datetime_UTC"].apply(translate),
        format="%d-%b-%y %H:%M",
        errors="coerce",
    )
    df_aeb = df_aeb.dropna(subset=["Datetime_UTC"]).set_index("Datetime_UTC")
    for c in ["Warmtelevering AEB", "BEC_WPW", "WPW_Vattenfall_AD"]:
        df_aeb[c] = pd.to_numeric(df_aeb[c], errors="coerce")
    return filter_datetime_index(df_aeb, SIM_START, SIM_END)


# ── 3. Elektriciteit_prijzen.xlsx — EPEX Spot ────────────────────────────────

def load_epex_prices():
    path  = os.path.join(DATA_DIR, "Elektriciteit_prijzen.xlsx")
    time_col  = "DatumUurKwartier_UTC"
    price_col = "Prijs_Gemiddelde_EPEX_Spot"

    tmp = pd.read_excel(path, header=None)
    header_row = None
    for i in range(len(tmp)):
        if time_col in tmp.iloc[i].astype(str).str.strip().values:
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"'{time_col}' not found in Elektriciteit_prijzen.xlsx")

    df3 = pd.read_excel(path, header=header_row, dtype={time_col: str})
    df3.columns = df3.columns.astype(str).str.strip()

    dt = df3[time_col].astype(str).str.strip().str.lower()
    dt = dt.str.replace(" ", " ", regex=False).str.replace("/", "-", regex=False)
    for nl, num in _DUTCH_TO_EN.items():
        dt = dt.str.replace(f"-{nl}-", f"-{num}-", regex=False)

    dt_parsed = pd.to_datetime(dt, format="%d-%m-%y %H:%M", errors="coerce")
    mask = dt_parsed.isna()
    if mask.any():
        dt_parsed.loc[mask] = pd.to_datetime(dt[mask], format="%d-%m-%y %H:%M:%S", errors="coerce")
    df3[time_col] = dt_parsed

    p = df3[price_col].astype(str).str.replace(" ", " ", regex=False).str.strip()
    p = p.str.replace("−", "-", regex=False).str.replace("–", "-", regex=False).str.replace("—", "-", regex=False)
    neg = p.str.contains(r"\(.*\)")
    p = p.str.replace("(", "", regex=False).str.replace(")", "", regex=False)
    p = p.str.replace(r"[^0-9,\.\-]", "", regex=True).str.replace(",", ".", regex=False)
    p = pd.to_numeric(p, errors="coerce")
    p.loc[neg & p.notna()] = -p.loc[neg & p.notna()].abs()
    df3[price_col] = p

    df3 = df3.dropna(subset=[time_col, price_col]).set_index(time_col).sort_index()
    df3 = df3.groupby(df3.index)[price_col].mean().to_frame()

    start_q = pd.to_datetime("01-01-2022 00:00")
    end_q   = pd.to_datetime("31-12-2025 23:45")
    end_h   = pd.to_datetime("31-12-2025 23:00")

    df3_q = df3.loc[(df3.index >= start_q) & (df3.index <= end_q)].copy()
    df3_h = df3_q.resample("h").mean()
    df3_h = df3_h.loc[(df3_h.index >= start_q) & (df3_h.index <= end_h)].copy()

    return df3_q, df3_h


# ── 4. Warmtelevering_metingen_details.xlsx — supply/return temperatures ─────

def load_temperatures() -> pd.DataFrame:
    path     = os.path.join(DATA_DIR, "Warmtelevering_metingen_details.xlsx")
    time_col = "IdDatumUurMinuut_UTC"
    id_col   = "Unnamed: 1"

    preview = pd.read_excel(path, header=None, nrows=80)
    header_row = next(
        (i for i in range(len(preview))
         if time_col in preview.iloc[i].astype(str).str.strip().values),
        None,
    )
    if header_row is None:
        raise ValueError(f"Could not find header row containing '{time_col}'")

    df = pd.read_excel(path, header=header_row)
    df.columns = df.columns.astype(str).str.replace("﻿", "").str.strip()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df_2025 = df.loc[df[id_col].notna()].copy()

    t = pd.to_numeric(df_2025[time_col], errors="coerce")
    dt = pd.to_datetime(t.astype("Int64").astype(str), format="%Y%m%d%H%M", errors="coerce")

    supply_col = df_2025.columns[7]
    return_col = df_2025.columns[29]

    T_supply = clean_numeric(df_2025[supply_col]).to_numpy()
    T_return = clean_numeric(df_2025[return_col]).to_numpy()

    temp = pd.DataFrame(
        {"T_supply_C": T_supply, "T_return_C": T_return},
        index=dt.to_numpy(),
    ).sort_index()
    temp = temp[~temp.index.isna()]
    temp_ds = temp.resample("1h").mean()
    temp_ds = temp_ds.loc[(temp_ds.index >= SIM_START) & (temp_ds.index <= SIM_END)]
    temp_ds["Delta_T_Buffer_C"] = temp_ds["T_supply_C"] - temp_ds["T_return_C"]
    return temp_ds


# ── 5. Warmtewisselaar.xlsx — Productie sheet ─────────────────────────────────

def load_hx_production() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "Warmtewisselaar.xlsx")
    df_prod_raw = pd.read_excel(path, sheet_name="Productie", header=None)

    header_row = None
    for i in range(min(20, len(df_prod_raw))):
        row_vals = df_prod_raw.iloc[i].astype(str).str.strip().str.lower()
        if row_vals.str.contains("datumuur", na=False).any():
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find header row in Productie sheet.")

    headers = df_prod_raw.iloc[header_row].astype(str).str.strip()
    df_prod = df_prod_raw.iloc[header_row + 1:].copy()
    df_prod.columns = headers
    df_prod.columns = (
        df_prod.columns.astype(str).str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    datetime_col = [c for c in df_prod.columns if "datumuur" in c.lower()][0]

    rename_map = {
        "WW00": "WW_00", "WW01": "WW_01", "WW02": "WW_02",
        "WW10": "WW_10", "WW30": "WW_30", "WWBEC": "WW_BEC", "WWBEN": "WW_BEN",
        "WW_00": "WW_00", "WW_01": "WW_01", "WW_02": "WW_02",
        "WW_10": "WW_10", "WW_30": "WW_30", "WW_BEC": "WW_BEC", "WW_BEN": "WW_BEN",
    }
    df_prod = df_prod.rename(columns={c: rename_map[c] for c in df_prod.columns if c in rename_map})

    target_cols = ["WW_00", "WW_01", "WW_02", "WW_10", "WW_30", "WW_BEC", "WW_BEN"]
    available_cols = [c for c in target_cols if c in df_prod.columns]
    if not available_cols:
        raise ValueError("None of the WW columns found in Productie sheet.")

    df_prod = df_prod[[datetime_col] + available_cols].copy()
    df_prod = df_prod.rename(columns={datetime_col: "Datetime_UTC"})
    df_prod["Datetime_UTC"] = _translate_dutch_dt(df_prod["Datetime_UTC"])

    for col in available_cols:
        df_prod[col] = clean_numeric(df_prod[col])

    df_prod = df_prod.dropna(subset=["Datetime_UTC"]).set_index("Datetime_UTC").sort_index()
    df_prod = df_prod.groupby(df_prod.index).mean()
    return df_prod.loc[SIM_START:SIM_END].copy()


# ── 6. T30 steam consumption + turbine availability ───────────────────────────

def load_turbine_data(sim_df_ref: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame aligned to sim_df_ref.index with turbine columns."""
    col_t30 = "T30_stoom_verbruik_tluur_1h[Dim 1][t/h]"
    col_t10_mw = "Turbine10_Eproductie_MW_1h[Dim 1][MW]"
    col_t30_mw = "Turbine30_Eproductie_MW_1h[Dim 1][MW]"

    # T30 steam consumption
    path_t30 = os.path.join(DATA_DIR, "T30_stoom_slok_tph.csv")
    t30_raw = pd.read_csv(path_t30).iloc[:, :2].copy()
    t30_raw.columns = ["datetime_raw", col_t30]

    HEADER_OFFSET = 1
    hours_before = int((SIM_START - pd.Timestamp("2022-01-01")).total_seconds() // 3600)
    expected_h   = int((SIM_END - SIM_START).total_seconds() // 3600) + 1
    start_idx = HEADER_OFFSET + hours_before
    t30_raw = t30_raw.iloc[start_idx: start_idx + expected_h].copy()
    t30_raw["Datetime UTC"] = pd.date_range(start=SIM_START, periods=len(t30_raw), freq="h")
    t30_raw = t30_raw.set_index("Datetime UTC").sort_index()
    t30_raw[col_t30] = pd.to_numeric(t30_raw[col_t30], errors="coerce")

    # Turbine electricity production
    path_turb = os.path.join(DATA_DIR, "T10_en_T30_productie_MW_voor_WW10_WW30_beschikbaarheid.csv")
    turb_df = pd.read_csv(path_turb).iloc[:, :3].copy()
    turb_df.columns = ["datetime_raw", col_t10_mw, col_t30_mw]
    turb_df["Datetime UTC"] = pd.to_datetime(turb_df["datetime_raw"], dayfirst=True, errors="coerce")
    turb_df = turb_df.dropna(subset=["Datetime UTC"]).set_index("Datetime UTC").sort_index()
    turb_df = turb_df.loc[(turb_df.index >= SIM_START) & (turb_df.index <= SIM_END)].copy()

    out = sim_df_ref[[col_t30]].copy() if col_t30 in sim_df_ref.columns else pd.DataFrame(index=sim_df_ref.index)
    out = out.join(t30_raw[[col_t30]], how="left").fillna(0.0)
    out = out.join(turb_df[[col_t10_mw, col_t30_mw]], how="left").fillna(0.0)
    return out


# ── Master DataFrame assembly ─────────────────────────────────────────────────

def build_sim_df_tank() -> tuple:
    """
    Load all data, join into sim_df_tank, compute availability flags,
    charging-hour labels, and pre-compute hist_loss_cols + cost_hist.

    Returns
    -------
    sim_df_tank, temp_ds, df1_2025, df3_h, df3_q, df_prod_2025,
    hist_loss_cols, cost_hist
    """
    print("Loading data...")
    df1_2025     = load_buffer_demand()
    temp_ds      = load_temperatures()
    df3_q, df3_h = load_epex_prices()
    df_prod_2025 = load_hx_production()

    price_col = "Prijs_Gemiddelde_EPEX_Spot"
    P_DPB_2025 = df3_q[price_col].mean()

    # Demand + ΔT
    df_demand  = df1_2025[["Warmtevraag WPW totaal MW"]].rename(columns={"Warmtevraag WPW totaal MW": "heat_demand_mw"})
    df_delta_t = temp_ds[["Delta_T_Buffer_C"]].rename(columns={"Delta_T_Buffer_C": "delta_t_c"})
    df_hx_hist = df_prod_2025.rename(columns={c: f"{c}_hist" for c in df_prod_2025.columns})

    df_master = df_demand.join(df_delta_t, how="inner").join(df_hx_hist, how="left")

    sim_df_tank = df_master.copy()
    sim_df_tank = sim_df_tank.dropna(subset=["heat_demand_mw", "delta_t_c"]).copy()
    sim_df_tank["heat_demand_mw"] = sim_df_tank["heat_demand_mw"].clip(lower=0)

    # Add individual loop demand columns to sim_df_tank
    # WestPoort demand is derived as residual: WPW total - NieuwWest - Noord
    # NieuwWest and Noord are loaded directly from df1_2025

    loop_cols = sim_df_tank.copy()

    for src_col, dst_col in [
        ('Warmtevraag NieuwWest MW', 'demand_nieuwwest_mw'),
        ('Warmtevraag Noord MW',     'demand_noord_mw'),
    ]:
        if src_col in df1_2025.columns:
            sim_df_tank = sim_df_tank.join(
                df1_2025[[src_col]].rename(columns={src_col: dst_col}),
                how='left'
            )
            sim_df_tank[dst_col] = sim_df_tank[dst_col].ffill().bfill()
        else:
            print(f"WARNING: {src_col} not found — filling {dst_col} with 0")
            sim_df_tank[dst_col] = 0.0

    # WestPoort = WPW total - NieuwWest - Noord
    sim_df_tank['demand_westpoort_mw'] = (
        sim_df_tank['heat_demand_mw']
        - sim_df_tank['demand_nieuwwest_mw']
        - sim_df_tank['demand_noord_mw']
    ).clip(lower=0.0)

    print(f"  Loop demands joined:")
    print(f"  WestPoort mean: {sim_df_tank['demand_westpoort_mw'].mean():.3f} MW")
    print(f"  NieuwWest mean: {sim_df_tank['demand_nieuwwest_mw'].mean():.3f} MW")
    print(f"  Noord mean:     {sim_df_tank['demand_noord_mw'].mean():.3f} MW")
    print(f"  Sum vs total:   {(sim_df_tank['demand_westpoort_mw'] + sim_df_tank['demand_nieuwwest_mw'] + sim_df_tank['demand_noord_mw']).mean():.3f} vs {sim_df_tank['heat_demand_mw'].mean():.3f} MW")

    hist_cols_to_use = list(historical_cols.values())
    hist_df = df_prod_2025[hist_cols_to_use].copy()
    sim_df_tank = sim_df_tank.join(hist_df, how="left")
    sim_df_tank[hist_cols_to_use] = sim_df_tank[hist_cols_to_use].fillna(0)

    # Turbine data
    turb_data = load_turbine_data(sim_df_tank)
    col_t30  = "T30_stoom_verbruik_tluur_1h[Dim 1][t/h]"
    col_t10_mw = "Turbine10_Eproductie_MW_1h[Dim 1][MW]"
    col_t30_mw = "Turbine30_Eproductie_MW_1h[Dim 1][MW]"
    for col in [col_t30, col_t10_mw, col_t30_mw]:
        if col in sim_df_tank.columns:
            sim_df_tank = sim_df_tank.drop(columns=[col])
    sim_df_tank = sim_df_tank.join(turb_data, how="left").fillna(0.0)

    # WW30 aftap capacity
    t30_slok = sim_df_tank[col_t30]
    import numpy as np
    sim_df_tank["WW30_aftap_cap_mw"] = np.where(
        t30_slok <= 180.0, 0.0,
        np.where(t30_slok >= 230.0, 7.0, (t30_slok - 180.0) * (7.0 / 50.0)),
    )

    # Min/max MW per HX
    safe_dt = sim_df_tank["delta_t_c"].clip(lower=MIN_PHYSICAL_DELTA_T)
    for hx, p in hx_params.items():
        if p["type"] == "fixed":
            sim_df_tank[f"{hx}_min_mw"] = p["min_mw"]
            sim_df_tank[f"{hx}_max_mw"] = p["max_mw"]
        elif p["type"] == "special":
            dyn_min = CONST * safe_dt * p["min_m3h"]
            sim_df_tank[f"{hx}_min_mw"] = np.minimum(dyn_min, p["max_mw_const"])
            sim_df_tank[f"{hx}_max_mw"] = p["max_mw_const"]
        elif p["type"] == "dynamic":
            dyn_min = CONST * safe_dt * p["min_m3h"]
            dyn_max = CONST * safe_dt * p["max_m3h"]
            sim_df_tank[f"{hx}_min_mw"] = np.minimum(dyn_min, p["max_mw_const"])
            sim_df_tank[f"{hx}_max_mw"] = np.minimum(dyn_max, p["max_mw_const"])
    for hx in priority_order:
        sim_df_tank[f"{hx}_min_mw"] = np.minimum(
            sim_df_tank[f"{hx}_min_mw"], sim_df_tank[f"{hx}_max_mw"]
        )

    # Availability flags
    for hx in ["WWBEC", "WW01", "WW00", "WW02"]:
        hist_col = historical_cols[hx]
        sim_df_tank[f"{hx}_hist_available"]  = (sim_df_tank[hist_col].abs() > ZERO_TOL).astype(int)
        sim_df_tank[f"{hx}_hist_forced_off"] = (sim_df_tank[hist_col].abs() <= ZERO_TOL).astype(int)
    sim_df_tank["WW10_hist_available"]  = (sim_df_tank[col_t10_mw] >= 5.0).astype(int)
    sim_df_tank["WW10_hist_forced_off"] = 1 - sim_df_tank["WW10_hist_available"]
    sim_df_tank["WW30_hist_available"]  = (sim_df_tank[col_t30_mw] >= 5.0).astype(int)
    sim_df_tank["WW30_hist_forced_off"] = 1 - sim_df_tank["WW30_hist_available"]

    # EPEX prices
    if price_col in sim_df_tank.columns:
        sim_df_tank = sim_df_tank.drop(columns=[price_col])
    sim_df_tank = sim_df_tank.join(df3_h[[price_col]], how="left")
    sim_df_tank[price_col] = sim_df_tank[price_col].ffill().bfill()

    # Charging-hour labels (N cheapest per day)
    N_CHEAPEST = 10
    sim_df_tank["date"] = sim_df_tank.index.date
    sim_df_tank["hour"] = sim_df_tank.index.hour
    price_rank = sim_df_tank.groupby("date")[price_col].rank(method="first", ascending=True)
    sim_df_tank["price_rank_within_day"] = price_rank
    sim_df_tank["is_charging_hour"]      = (price_rank <= N_CHEAPEST).astype(int)
    sim_df_tank["is_non_charging_hour"]  = 1 - sim_df_tank["is_charging_hour"]

    daily_noncharge_demand = (
        sim_df_tank.loc[sim_df_tank["is_non_charging_hour"] == 1]
        .groupby("date")["heat_demand_mw"].sum()
    )
    daily_charge_hours = sim_df_tank.groupby("date")["is_charging_hour"].sum()
    all_dates = pd.Index(sorted(sim_df_tank["date"].unique()))
    forecast = {}
    for i, d in enumerate(all_dates):
        forecast[d] = daily_noncharge_demand.get(all_dates[i - 1] if i > 0 else d, 0.0)
    sim_df_tank["forecast_noncharge_demand_mwh"] = sim_df_tank["date"].map(forecast)
    sim_df_tank["charging_hours_count"] = sim_df_tank["date"].map(daily_charge_hours)

    # Min required tank volume (2-h forward demand reserve)
    safe_dt2 = sim_df_tank["delta_t_c"].clip(lower=ZERO_TOL)
    req_vol = sim_df_tank["heat_demand_mw"] / (CONST * safe_dt2)
    next_2h = req_vol.shift(-1).fillna(0.0) + req_vol.shift(-2).fillna(0.0)
    sim_df_tank["tank_min_volume_required_m3"] = np.clip(next_2h, 1200.0, 2400.0)

    # Historical electricity loss columns
    hist_loss_cols = {}
    aftap_cap = sim_df_tank["WW30_aftap_cap_mw"]
    for hx in priority_order:
        hist_col  = historical_cols[hx]
        heat_hist = sim_df_tank[hist_col].fillna(0.0)
        if hx == "WW30":
            aftap_part    = np.minimum(heat_hist, aftap_cap)
            reductie_part = np.maximum(heat_hist - aftap_part, 0)
            loss = (
                aftap_part * derivation_factors["WW30_aftap"]
                + reductie_part * derivation_factors["WW30_reductie"]
            )
        else:
            loss = heat_hist * derivation_factors[hx]
        hist_loss_cols[hx] = loss

    cost_hist = sum(
        hist_loss_cols[hx] * sim_df_tank[price_col]
        for hx in priority_order if hx != "WWBEC"
    ).sum()

    # Join measured supply and return temperatures onto sim_df_tank
    # Source: Warmtelevering_metingen_details.xlsx, resampled to hourly
    # T_supply_C: network supply temperature at AEB (~90-110°C, varies per timestep)
    # T_return_C: network return temperature (~58-70°C, varies per timestep)
    if 'T_supply_C' in temp_ds.columns and 'T_return_C' in temp_ds.columns:
        sim_df_tank = sim_df_tank.join(
            temp_ds[['T_supply_C', 'T_return_C']],
            how='left'
        )
        sim_df_tank['T_supply_C'] = sim_df_tank['T_supply_C'].ffill().bfill()
        sim_df_tank['T_return_C'] = sim_df_tank['T_return_C'].ffill().bfill()
        print(f"  T_supply_C joined: mean={sim_df_tank['T_supply_C'].mean():.1f}°C, "
              f"min={sim_df_tank['T_supply_C'].min():.1f}°C, "
              f"max={sim_df_tank['T_supply_C'].max():.1f}°C")
        print(f"  T_return_C joined: mean={sim_df_tank['T_return_C'].mean():.1f}°C, "
              f"min={sim_df_tank['T_return_C'].min():.1f}°C, "
              f"max={sim_df_tank['T_return_C'].max():.1f}°C")
    else:
        print("  WARNING: T_supply_C or T_return_C not found in temp_ds — skipping join")
        sim_df_tank['T_supply_C'] = 90.0  # fallback constant
        sim_df_tank['T_return_C'] = 70.0  # fallback constant

    print(f"sim_df_tank shape: {sim_df_tank.shape}")
    print(f"Historical derving (excl. BEC): {cost_hist:,.0f} €")

    return sim_df_tank, temp_ds, df1_2025, df3_h, df3_q, df_prod_2025, hist_loss_cols, cost_hist
