"""Loads the AEB price forecast and converts its time stamps to UTC for the dispatch model. Needs the raw data."""
import os
import pandas as pd

from config import DATA_DIR

DEFAULT_FORECAST_PATH = os.path.join(DATA_DIR, "EPEX_prijs_simpele_voorspelling.xlsx")


def load_forecast_prices(filepath=DEFAULT_FORECAST_PATH):
    df = pd.read_excel(filepath)
    df.columns = ['datetime', 'voorspelling_prijs', 'realisatie_prijs']

    # Parse date-time — format is DD/MM/YYYY HH:MM
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)

    # Local Dutch time (CET winter = UTC+1, CEST summer = UTC+2)
    # convert to UTC
    df['datetime'] = (
        df['datetime']
        .dt.tz_localize('Europe/Amsterdam', ambiguous='infer', nonexistent='shift_forward')
        .dt.tz_convert('UTC')
        .dt.tz_localize(None)  # remove timezone info for the merge with sim_df_tank
    )

    df = df.set_index('datetime')

    # Resample from 15 minutes to hourly resolution via the mean
    df_hourly = df.resample('h').mean()

    # Filter on the simulation period 2023-2025
    df_hourly = df_hourly.loc['2023-01-01':'2025-12-31']

    print(f"  Voorspelprijzen geladen (tijdzone gecorrigeerd NL -> UTC):")
    print(f"  Rijen: {len(df_hourly)}")
    print(f"  Periode: {df_hourly.index[0]} t/m {df_hourly.index[-1]}")
    print(f"  Gemiddelde voorspelprijs:    {df_hourly['voorspelling_prijs'].mean():.2f} EUR/MWh")
    print(f"  Gemiddelde realisatieprijs:  {df_hourly['realisatie_prijs'].mean():.2f} EUR/MWh")
    print(f"  Verschil gem.:               {(df_hourly['voorspelling_prijs'] - df_hourly['realisatie_prijs']).mean():.2f} EUR/MWh")

    return df_hourly
