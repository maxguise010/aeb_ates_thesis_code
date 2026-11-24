"""Small helpers of the data pipeline: parsing numbers written in Dutch notation and filtering by date."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_series(df, col, title=None, ylabel=None, figsize=(12, 5)):
    plt.figure(figsize=figsize)
    df[col].plot()
    plt.title(title or col)
    plt.xlabel("Datetime")
    plt.ylabel(ylabel or col)
    plt.tight_layout()
    plt.show()


def filter_datetime_index(df, start, end):
    start = pd.to_datetime(start)
    end   = pd.to_datetime(end)
    return df.loc[start:end].copy()


def clean_numeric(s: pd.Series) -> pd.Series:
    """Parse a column with Dutch number formatting and unicode minus signs."""
    s = s.astype(str).str.strip()
    s = (s.str.replace(" ", " ", regex=False)
           .str.replace("−", "-", regex=False)
           .str.replace("–", "-", regex=False)
           .str.replace("—", "-", regex=False))
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")
