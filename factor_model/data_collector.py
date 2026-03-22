"""
Data Collector Module
=====================

Downloads and processes financial market data from Yahoo Finance
for CAPM (Capital Asset Pricing Model) analysis.

This module provides functions to:
- Download historical stock prices
- Calculate log returns
- Compute excess returns over the risk-free rate
- Merge stock and market data for regression analysis

Example:
--------
    >>> from factor_model.data_collector import (
    ...     get_stock_returns,
    ...     get_market_returns,
    ...     calculate_excess_returns,
    ...     merge_data
    ... )
    >>> stocks = get_stock_returns(["AAPL", "MSFT"], "2023-01-01", "2024-01-01")
    >>> market = get_market_returns("2023-01-01", "2024-01-01")
    >>> rf = get_risk_free_rate()
    >>> merged = merge_data(
    ...     calculate_excess_returns(stocks, rf),
    ...     calculate_excess_returns(market, rf)
    ... )
"""

from __future__ import annotations

import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf


# Constants
TRADING_DAYS: int = 252
"""Number of trading days per year for annualization."""

DEFAULT_RISK_FREE_RATE: float = 0.02
"""Default annual risk-free rate (2%)."""


def get_stock_returns(
    symbols: List[str],
    start: str,
    end: str
) -> pd.DataFrame:
    """
    Download historical stock data and calculate log returns.

    Parameters
    ----------
    symbols : List[str]
        List of stock ticker symbols (e.g., ["AAPL", "MSFT"]).
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str
        End date in "YYYY-MM-DD" format.

    Returns
    -------
    pd.DataFrame
        DataFrame with log returns for each stock, indexed by date.

    Raises
    ------
    ValueError
        If symbols list is empty or no data could be downloaded.

    Example
    -------
        >>> returns = get_stock_returns(["AAPL", "GOOGL"], "2023-01-01", "2024-01-01")
        >>> returns.head()
                        AAPL     GOOGL
        Date
        2023-01-04  0.0102   -0.0118
        2023-01-05 -0.0107   -0.0217
    """
    if not symbols:
        raise ValueError("symbols list cannot be empty")

    data = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise ValueError("No stock data downloaded.")

    stock_returns = {}

    for symbol in symbols:
        try:
            prices = extract_close_series(data, symbol)
            log_returns = np.log(prices / prices.shift(1)).dropna()
            stock_returns[symbol] = log_returns
        except Exception as e:
            print(f"Warning: could not process {symbol}: {e}")

    if not stock_returns:
        raise ValueError("No stock returns were created.")

    returns_df = pd.DataFrame(stock_returns).sort_index()
    return returns_df.dropna(how="all")


def get_market_returns(
    start: str,
    end: str,
    market_symbol: str = "^GSPC"
) -> pd.DataFrame:
    """
    Download market index data and calculate log returns.

    Uses S&P 500 (^GSPC) as the default market proxy.

    Parameters
    ----------
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str
        End date in "YYYY-MM-DD" format.
    market_symbol : str, optional
        Market index ticker symbol (default: "^GSPC" for S&P 500).

    Returns
    -------
    pd.DataFrame
        DataFrame with market log returns in "Market" column.

    Raises
    ------
    ValueError
        If no market data could be downloaded.

    Example
    -------
        >>> market = get_market_returns("2023-01-01", "2024-01-01")
        >>> market.head()
                      Market
        Date
        2023-01-04  0.0074
        2023-01-05 -0.0118
    """
    data = yf.download(
        tickers=market_symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise ValueError("No market data downloaded.")

    prices = extract_close_series(data, market_symbol)
    market_returns = np.log(prices / prices.shift(1)).dropna()

    return pd.DataFrame({"Market": market_returns}).sort_index()


def get_risk_free_rate(annual_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
    """
    Calculate daily risk-free rate from annual rate.

    Parameters
    ----------
    annual_rate : float, optional
        Annual risk-free rate (default: 0.02 or 2%).

    Returns
    -------
    float
        Daily risk-free rate.

    Example
    -------
        >>> rf = get_risk_free_rate()
        >>> print(f"{rf:.6f}")
        0.000079
    """
    return annual_rate / TRADING_DAYS


def calculate_excess_returns(
    returns: Union[pd.Series, pd.DataFrame],
    rf_rate: float
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate excess returns over the risk-free rate.

    Excess Return = Total Return - Risk-Free Rate

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Asset returns (daily).
    rf_rate : float
        Daily risk-free rate.

    Returns
    -------
    pd.Series or pd.DataFrame
        Excess returns with same shape as input.

    Raises
    ------
    ValueError
        If returns is empty or None.

    Example
    -------
        >>> returns = pd.Series([0.01, 0.02, -0.01])
        >>> excess = calculate_excess_returns(returns, 0.0001)
    """
    if returns is None or len(returns) == 0:
        raise ValueError("returns is empty")
    return returns - rf_rate


def merge_data(
    stock_excess: pd.DataFrame,
    market_excess: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge stock and market excess returns on common dates.

    Performs an inner join to ensure date alignment.

    Parameters
    ----------
    stock_excess : pd.DataFrame
        DataFrame of stock excess returns.
    market_excess : pd.DataFrame
        DataFrame of market excess returns with "Market" column.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all stocks and market, aligned by date.

    Raises
    ------
    ValueError
        If either input is empty or no common dates exist.

    Example
    -------
        >>> merged = merge_data(stock_excess, market_excess)
        >>> print(merged.columns.tolist())
        ['AAPL', 'MSFT', 'Market']
    """
    if stock_excess.empty:
        raise ValueError("stock_excess is empty")
    if market_excess.empty:
        raise ValueError("market_excess is empty")

    merged = stock_excess.join(market_excess, how="inner")

    if merged.empty:
        raise ValueError("Merged data is empty after aligning dates.")

    return merged.sort_index()


def annualized_mean_returns(
    returns: Union[pd.Series, pd.DataFrame]
) -> Union[float, pd.Series]:
    """
    Convert mean daily returns to annualized returns.

    Annualized Return = Daily Mean × Trading Days

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns.

    Returns
    -------
    float or pd.Series
        Annualized mean returns.

    Example
    -------
        >>> daily_returns = pd.Series([0.001] * 252)  # 0.1% daily
        >>> annual = annualized_mean_returns(daily_returns)
        >>> print(f"{annual:.2%}")
        25.20%
    """
    return returns.mean() * TRADING_DAYS


def extract_close_series(
    data: pd.DataFrame,
    symbol: Optional[str] = None
) -> pd.Series:
    """
    Extract Close prices from yfinance download data.

    Handles both single-ticker and multi-ticker downloads,
    including various MultiIndex column formats.

    Parameters
    ----------
    data : pd.DataFrame
        Raw data from yfinance.download().
    symbol : str, optional
        Ticker symbol to extract (for multi-ticker data).

    Returns
    -------
    pd.Series
        Series of closing prices.

    Raises
    ------
    ValueError
        If data is empty or Close column cannot be found.
    """
    if data.empty:
        raise ValueError("Downloaded data is empty")

    if isinstance(data.columns, pd.MultiIndex):
        # Case 1: columns like ('Close', 'AAPL')
        if "Close" in data.columns.get_level_values(0):
            close_data = data["Close"]

            if isinstance(close_data, pd.Series):
                prices = close_data
            elif symbol is not None and symbol in close_data.columns:
                prices = close_data[symbol]
            else:
                prices = close_data.iloc[:, 0]

        # Case 2: columns like ('AAPL', 'Close')
        elif "Close" in data.columns.get_level_values(1):
            if symbol is not None and symbol in data.columns.get_level_values(0):
                prices = data[symbol]["Close"]
            else:
                first_symbol = data.columns.get_level_values(0)[0]
                prices = data[first_symbol]["Close"]
        else:
            raise ValueError("Could not find Close column in MultiIndex data")
    else:
        if "Close" not in data.columns:
            raise ValueError("Could not find Close column in data")
        prices = data["Close"]

    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    prices = pd.to_numeric(prices, errors="coerce").dropna()

    if len(prices) < 2:
        raise ValueError("Not enough valid price data")

    return prices


def main() -> None:
    """
    Run data collection pipeline.

    Downloads stock and market data, calculates excess returns,
    and saves merged data to CSV.
    """
    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    start = "2023-01-01"
    end = "2026-01-01"

    print("Downloading stock returns...")
    stock_returns = get_stock_returns(symbols, start, end)

    print("Downloading market returns...")
    market_returns = get_market_returns(start, end)

    rf_rate = get_risk_free_rate()
    print(f"Daily risk-free rate: {rf_rate:.8f}")

    stock_excess = calculate_excess_returns(stock_returns, rf_rate)
    market_excess = calculate_excess_returns(market_returns, rf_rate)

    merged_data = merge_data(stock_excess, market_excess)

    print("\nFirst 5 rows of merged excess returns:")
    print(merged_data.head())

    print("\nColumns present:")
    print(merged_data.columns.tolist())

    expected_columns = symbols + ["Market"]
    missing_columns = [col for col in expected_columns if col not in merged_data.columns]

    if missing_columns:
        print("\nMissing columns:")
        print(missing_columns)
    else:
        print("\nAll expected stock and market columns are present.")

    print("\nAverage annual excess return:")
    print(annualized_mean_returns(merged_data))

    print("\nMerged data shape:")
    print(merged_data.shape)

    # Save to CSV in data/ directory
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "merged_excess_returns.csv")
    merged_data.to_csv(output_path)
    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()