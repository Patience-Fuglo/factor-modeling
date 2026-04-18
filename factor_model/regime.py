"""
Factor Timing & Market Regime Detection
=========================================

Detects market regimes (bull/bear/sideways) and analyses how each
factor (Market, SMB, HML, MOM) performs differently across regimes.
This is the foundation of factor timing — adjusting portfolio factor
exposure based on current market conditions.

Regime Detection Methods:
1. Moving Average Crossover — price above/below 200-day SMA
2. Rolling Sharpe — positive/negative rolling risk-adjusted return
3. Drawdown-based — regime defined by depth of current drawdown

Factor Timing Insight:
- Value (HML) tends to underperform in momentum-driven bull markets
- Momentum (MOM) tends to crash during sharp reversals
- Low-beta strategies outperform in bear regimes

Example:
--------
    >>> from factor_model.regime import detect_regime, factor_performance_by_regime
    >>> regimes = detect_regime(market_returns)
    >>> perf = factor_performance_by_regime(factors_df, regimes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


TRADING_DAYS: int = 252


def detect_regime(
    market_returns: pd.Series,
    method: str = "sma",
    sma_window: int = 200,
    sharpe_window: int = 60
) -> pd.Series:
    """
    Detect market regime for each trading day.

    Parameters
    ----------
    market_returns : pd.Series
        Daily market excess returns.
    method : str
        Detection method: "sma" (moving average) or "sharpe" (rolling Sharpe).
    sma_window : int
        Lookback window for SMA method (default 200 days).
    sharpe_window : int
        Lookback window for rolling Sharpe method (default 60 days).

    Returns
    -------
    pd.Series
        Regime label per day: "Bull", "Bear", or "Sideways".
    """
    if method == "sma":
        return _sma_regime(market_returns, sma_window)
    elif method == "sharpe":
        return _sharpe_regime(market_returns, sharpe_window)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'sma' or 'sharpe'.")


def _sma_regime(market_returns: pd.Series, window: int) -> pd.Series:
    """
    Regime based on cumulative return vs its rolling moving average.

    Bull  = cumulative return above its SMA
    Bear  = cumulative return below its SMA by more than 1 std
    Sideways = in between
    """
    cum_returns = (1 + market_returns).cumprod()
    sma = cum_returns.rolling(window=window, min_periods=window // 2).mean()
    std = cum_returns.rolling(window=window, min_periods=window // 2).std()

    regime = pd.Series(index=market_returns.index, dtype=str)
    regime[:] = "Sideways"
    regime[cum_returns > sma] = "Bull"
    regime[cum_returns < (sma - std)] = "Bear"

    return regime


def _sharpe_regime(market_returns: pd.Series, window: int) -> pd.Series:
    """
    Regime based on rolling Sharpe ratio.

    Bull     = rolling Sharpe > 0.5
    Bear     = rolling Sharpe < -0.5
    Sideways = between -0.5 and 0.5
    """
    rolling_mean = market_returns.rolling(window=window).mean()
    rolling_std = market_returns.rolling(window=window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS)

    regime = pd.Series(index=market_returns.index, dtype=str)
    regime[:] = "Sideways"
    regime[rolling_sharpe > 0.5] = "Bull"
    regime[rolling_sharpe < -0.5] = "Bear"

    return regime.fillna("Sideways")


def factor_performance_by_regime(
    factors_df: pd.DataFrame,
    regimes: pd.Series
) -> pd.DataFrame:
    """
    Calculate annualised mean return of each factor in each market regime.

    Parameters
    ----------
    factors_df : pd.DataFrame
        Daily factor returns with columns like [Mkt_RF, SMB, HML, MOM].
    regimes : pd.Series
        Regime labels per day from detect_regime().

    Returns
    -------
    pd.DataFrame
        Mean annualised return per factor per regime, plus day counts.
    """
    aligned = factors_df.join(regimes.rename("Regime"), how="inner")

    results = []
    for regime_label in ["Bull", "Bear", "Sideways"]:
        subset = aligned[aligned["Regime"] == regime_label]
        if len(subset) == 0:
            continue

        factor_cols = [c for c in factors_df.columns if c != "RF"]
        row = {"Regime": regime_label, "Days": len(subset)}

        for factor in factor_cols:
            ann_return = subset[factor].mean() * TRADING_DAYS
            ann_vol = subset[factor].std() * np.sqrt(TRADING_DAYS)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
            row[f"{factor}_Return"] = ann_return
            row[f"{factor}_Sharpe"] = sharpe

        results.append(row)

    return pd.DataFrame(results)


def regime_weights(
    current_regime: str,
    base_weights: dict[str, float] = None
) -> dict[str, float]:
    """
    Return factor weights adjusted for the current market regime.

    This is the core of factor timing — tilt exposure toward factors
    that historically outperform in the current regime.

    Parameters
    ----------
    current_regime : str
        "Bull", "Bear", or "Sideways".
    base_weights : dict, optional
        Base factor weights to tilt from. Defaults to equal weight.

    Returns
    -------
    dict
        Adjusted factor weights that sum to 1.
    """
    tilts = {
        "Bull":     {"Mkt_RF": 0.40, "SMB": 0.20, "HML": 0.15, "MOM": 0.25},
        "Bear":     {"Mkt_RF": 0.20, "SMB": 0.30, "HML": 0.35, "MOM": 0.15},
        "Sideways": {"Mkt_RF": 0.30, "SMB": 0.25, "HML": 0.25, "MOM": 0.20},
    }
    return tilts.get(current_regime, tilts["Sideways"])


def regime_timed_portfolio(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    regime_method: str = "sma"
) -> pd.Series:
    """
    Build a portfolio that tilts stock weights based on market regime.

    In Bull regime  → overweight high-momentum stocks
    In Bear regime  → overweight low-beta / value stocks
    In Sideways     → equal weight

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Daily excess returns for each stock.
    market_returns : pd.Series
        Daily market returns for regime detection.
    regime_method : str
        Regime detection method ("sma" or "sharpe").

    Returns
    -------
    pd.Series
        Daily portfolio returns with regime-based weights.
    """
    regimes = detect_regime(market_returns, method=regime_method)

    portfolio_returns = pd.Series(index=stock_returns.index, dtype=float)
    n_stocks = len(stock_returns.columns)

    for date in stock_returns.index:
        if date not in regimes.index:
            portfolio_returns[date] = stock_returns.loc[date].mean()
            continue

        regime = regimes[date]

        if regime == "Bull":
            # Overweight first 3 stocks (assume sorted by momentum)
            weights = pd.Series(0.0, index=stock_returns.columns)
            weights.iloc[:3] = 1 / 3
        elif regime == "Bear":
            # Overweight last 2 stocks (lower beta)
            weights = pd.Series(0.0, index=stock_returns.columns)
            weights.iloc[-2:] = 0.5
        else:
            weights = pd.Series(1 / n_stocks, index=stock_returns.columns)

        portfolio_returns[date] = (stock_returns.loc[date] * weights).sum()

    return portfolio_returns


def plot_regimes(
    market_returns: pd.Series,
    regimes: pd.Series,
    save_path: str = None
) -> None:
    """
    Plot cumulative market return with regime shading.

    Bull = green background, Bear = red, Sideways = gray.
    """
    cum_returns = (1 + market_returns).cumprod()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Shade regime backgrounds
    regime_colors = {"Bull": "#d5f5e3", "Bear": "#fadbd8", "Sideways": "#f2f3f4"}
    prev_regime = None
    start_date = None

    dates = regimes.index.tolist()
    for i, date in enumerate(dates):
        r = regimes[date]
        if r != prev_regime:
            if prev_regime is not None:
                ax.axvspan(start_date, date, alpha=0.4,
                           color=regime_colors.get(prev_regime, "white"), linewidth=0)
            start_date = date
            prev_regime = r
    if start_date is not None:
        ax.axvspan(start_date, dates[-1], alpha=0.4,
                   color=regime_colors.get(prev_regime, "white"), linewidth=0)

    ax.plot(cum_returns.index, cum_returns, color="#2c3e50", linewidth=2, label="S&P 500 (Cumulative)")

    bull_patch = mpatches.Patch(color="#d5f5e3", alpha=0.7, label="Bull Regime")
    bear_patch = mpatches.Patch(color="#fadbd8", alpha=0.7, label="Bear Regime")
    side_patch = mpatches.Patch(color="#f2f3f4", alpha=0.7, label="Sideways Regime")
    ax.legend(handles=[ax.lines[0], bull_patch, bear_patch, side_patch], fontsize=10)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.set_title("Market Regime Detection — S&P 500", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_factor_performance_by_regime(
    perf_df: pd.DataFrame,
    save_path: str = None
) -> None:
    """Bar chart showing annualised factor returns per regime."""
    factor_cols = [c for c in perf_df.columns if c.endswith("_Return")]
    factor_names = [c.replace("_Return", "") for c in factor_cols]
    regimes = perf_df["Regime"].tolist()

    x = np.arange(len(factor_names))
    width = 0.25
    colors = {"Bull": "#2ecc71", "Bear": "#e74c3c", "Sideways": "#95a5a6"}

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (_, row) in enumerate(perf_df.iterrows()):
        regime = row["Regime"]
        vals = [row[c] for c in factor_cols]
        offset = (i - len(regimes) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=regime, color=colors.get(regime, "gray"), edgecolor="black", alpha=0.85)

    ax.axhline(y=0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(factor_names, fontsize=12)
    ax.set_xlabel("Factor", fontsize=12)
    ax.set_ylabel("Annualised Return", fontsize=12)
    ax.set_title("Factor Performance by Market Regime", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main() -> None:
    """Run regime detection and factor timing analysis."""
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    import pandas as pd
    from factor_model.ff3_collector import get_ff3_factors

    data = pd.read_csv(
        os.path.join(data_dir, "merged_excess_returns.csv"),
        index_col=0, parse_dates=True
    )
    market = data["Market"]

    print("Detecting market regimes...")
    regimes = detect_regime(market, method="sma")

    regime_counts = regimes.value_counts()
    print("\nRegime distribution:")
    for r, c in regime_counts.items():
        print(f"  {r:10s}: {c} days ({c / len(regimes):.1%})")

    print("\nDownloading FF3 factors for regime analysis...")
    ff3 = get_ff3_factors("2023-01-01", "2026-01-01")

    from factor_model.momentum import get_momentum_factor, build_carhart4_factors
    mom = get_momentum_factor("2023-01-01", "2026-01-01")
    all_factors = build_carhart4_factors(ff3, mom)[["Mkt_RF", "SMB", "HML", "MOM"]]

    print("\nFactor performance by regime:")
    perf = factor_performance_by_regime(all_factors, regimes)
    print(perf.to_string(index=False))

    perf.to_csv(os.path.join(data_dir, "regime_factor_performance.csv"), index=False)

    plot_regimes(market, regimes, save_path=os.path.join(plots_dir, "market_regimes.png"))
    plot_factor_performance_by_regime(perf, save_path=os.path.join(plots_dir, "factor_by_regime.png"))

    print(f"\nPlots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
