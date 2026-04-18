"""
Fama-French 3-Factor Data Collector
=====================================

Downloads Fama-French 3 factors (Market, SMB, HML) from Kenneth French's
data library via pandas_datareader and aligns them with stock return data.

Factors:
- Mkt-RF : Market excess return (same as CAPM market factor)
- SMB    : Small Minus Big (size premium)
- HML    : High Minus Low (value premium)
- RF     : Daily risk-free rate

Example:
--------
    >>> from factor_model.ff3_collector import get_ff3_factors, align_ff3_with_stocks
    >>> factors = get_ff3_factors("2023-01-01", "2026-01-01")
    >>> aligned = align_ff3_with_stocks(stock_returns, factors)
"""

from __future__ import annotations

import pandas as pd
import pandas_datareader.data as web


FF3_DATASET = "F-F_Research_Data_Factors_daily"


def get_ff3_factors(start: str, end: str) -> pd.DataFrame:
    """
    Download daily Fama-French 3 factors from Kenneth French's data library.

    Parameters
    ----------
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str
        End date in "YYYY-MM-DD" format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Mkt_RF, SMB, HML, RF] as decimals (not %).

    Raises
    ------
    ValueError
        If no data could be downloaded for the given date range.
    """
    raw = web.DataReader(FF3_DATASET, "famafrench", start, end)[0]

    if raw.empty:
        raise ValueError("No Fama-French data downloaded. Check date range and internet connection.")

    # Convert from percentage to decimal
    factors = raw / 100.0

    # Standardize column names
    factors.columns = ["Mkt_RF", "SMB", "HML", "RF"]
    factors.index = pd.to_datetime(factors.index)
    factors.index.name = "Date"

    return factors.sort_index()


def align_ff3_with_stocks(
    stock_returns: pd.DataFrame,
    ff3_factors: pd.DataFrame
) -> pd.DataFrame:
    """
    Align stock returns with FF3 factors on common trading dates.

    Computes stock excess returns using FF3's own risk-free rate (RF)
    for consistency with the factor model.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Raw log returns for each stock (not yet excess returns).
    ff3_factors : pd.DataFrame
        FF3 factors from get_ff3_factors().

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns:
        [stock1, stock2, ..., Mkt_RF, SMB, HML, RF, stock1_excess, stock2_excess, ...]

    Raises
    ------
    ValueError
        If no common dates exist between stocks and factors.
    """
    merged = stock_returns.join(ff3_factors, how="inner")

    if merged.empty:
        raise ValueError("No common dates between stock returns and FF3 factors.")

    # Compute excess returns using FF3's own RF for consistency
    stock_cols = stock_returns.columns.tolist()
    for col in stock_cols:
        merged[f"{col}_excess"] = merged[col] - merged["RF"]

    return merged.sort_index()


def get_stock_excess_and_factors(
    stock_returns: pd.DataFrame,
    ff3_factors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split aligned data into stock excess returns and factor matrix.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Raw stock log returns.
    ff3_factors : pd.DataFrame
        FF3 factors from get_ff3_factors().

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (stock_excess_df, factors_df) where:
        - stock_excess_df has one column per stock (excess returns)
        - factors_df has columns [Mkt_RF, SMB, HML]
    """
    aligned = align_ff3_with_stocks(stock_returns, ff3_factors)

    stock_cols = stock_returns.columns.tolist()
    excess_cols = [f"{col}_excess" for col in stock_cols]

    stock_excess_df = aligned[excess_cols].copy()
    stock_excess_df.columns = stock_cols

    factors_df = aligned[["Mkt_RF", "SMB", "HML"]].copy()

    return stock_excess_df, factors_df
