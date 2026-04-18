"""
Momentum Factor & Carhart 4-Factor Model
==========================================

Implements the momentum factor (WML — Winners Minus Losers) and extends
the Fama-French 3-factor model to the Carhart 4-factor model.

Momentum measures whether stocks that performed well recently continue
to outperform — one of the most robust and widely documented anomalies
in financial markets.

Carhart 4-Factor Model:
    R_i - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(MOM) + ε

Cross-sectional momentum strategy:
    - Rank stocks by past 12-1 month returns (skip last month)
    - Go long winners (top tercile), short losers (bottom tercile)

Example:
--------
    >>> from factor_model.momentum import get_momentum_factor, run_carhart4_regressions
    >>> mom = get_momentum_factor("2023-01-01", "2026-01-01")
    >>> results = run_carhart4_regressions(stock_excess_df, factors_with_mom)
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


TRADING_DAYS: int = 252
SIGNIFICANCE_LEVEL: float = 0.05


def get_momentum_factor(start: str, end: str) -> pd.Series:
    """
    Download daily momentum factor (WML) from Kenneth French's data library.

    Parameters
    ----------
    start : str
        Start date "YYYY-MM-DD".
    end : str
        End date "YYYY-MM-DD".

    Returns
    -------
    pd.Series
        Daily momentum factor as decimals (not %), indexed by date.
    """
    raw = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", start, end)[0]

    if raw.empty:
        raise ValueError("No momentum factor data downloaded.")

    mom = raw["Mom"] / 100.0
    mom.index = pd.to_datetime(mom.index)
    mom.index.name = "Date"
    mom.name = "MOM"

    return mom.sort_index()


def calculate_cross_sectional_momentum(
    stock_returns: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 1
) -> pd.DataFrame:
    """
    Calculate cross-sectional momentum scores for each stock.

    Momentum = cumulative return over past (lookback - skip) months,
    skipping the most recent month to avoid short-term reversal.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Daily log returns for each stock.
    lookback_months : int
        Lookback window in months (default 12).
    skip_months : int
        Months to skip at end of window (default 1).

    Returns
    -------
    pd.DataFrame
        Monthly momentum scores ranked 1 (lowest) to N (highest).
    """
    # Resample to monthly cumulative returns
    monthly = stock_returns.resample("M").sum()

    lookback = lookback_months - skip_months
    skip = skip_months

    momentum_scores = pd.DataFrame(index=monthly.index, columns=monthly.columns, dtype=float)

    for i in range(lookback + skip, len(monthly)):
        window = monthly.iloc[i - lookback - skip: i - skip]
        cum_returns = window.sum()
        momentum_scores.iloc[i] = cum_returns.rank()

    return momentum_scores.dropna(how="all")


def build_carhart4_factors(
    ff3_factors: pd.DataFrame,
    momentum_factor: pd.Series
) -> pd.DataFrame:
    """
    Combine FF3 factors with momentum to form Carhart 4-factor matrix.

    Parameters
    ----------
    ff3_factors : pd.DataFrame
        Columns [Mkt_RF, SMB, HML, RF].
    momentum_factor : pd.Series
        Daily WML momentum factor.

    Returns
    -------
    pd.DataFrame
        Aligned DataFrame with columns [Mkt_RF, SMB, HML, MOM, RF].
    """
    combined = ff3_factors.join(momentum_factor, how="inner")

    if combined.empty:
        raise ValueError("No common dates between FF3 factors and momentum factor.")

    return combined.sort_index()


def carhart4_regression(
    stock_excess: pd.Series,
    factors: pd.DataFrame
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run Carhart 4-factor regression for a single stock.

    Parameters
    ----------
    stock_excess : pd.Series
        Daily excess returns (R_i - R_f).
    factors : pd.DataFrame
        DataFrame with columns [Mkt_RF, SMB, HML, MOM].

    Returns
    -------
    RegressionResultsWrapper
        Fitted OLS model.
    """
    X = sm.add_constant(factors[["Mkt_RF", "SMB", "HML", "MOM"]])
    return sm.OLS(stock_excess, X).fit()


def extract_carhart4_results(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    stock_name: str
) -> Dict[str, Any]:
    """Extract key metrics from Carhart 4-factor regression."""
    params = result.params
    pvalues = result.pvalues

    return {
        "Stock": stock_name,
        "Alpha": params["const"],
        "Beta_Mkt": params["Mkt_RF"],
        "Beta_SMB": params["SMB"],
        "Beta_HML": params["HML"],
        "Beta_MOM": params["MOM"],
        "R_squared": result.rsquared,
        "Alpha_pvalue": pvalues["const"],
        "Beta_Mkt_pvalue": pvalues["Mkt_RF"],
        "Beta_SMB_pvalue": pvalues["SMB"],
        "Beta_HML_pvalue": pvalues["HML"],
        "Beta_MOM_pvalue": pvalues["MOM"],
    }


def run_carhart4_regressions(
    stock_excess_df: pd.DataFrame,
    factors_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run Carhart 4-factor regressions for all stocks.

    Parameters
    ----------
    stock_excess_df : pd.DataFrame
        Stock excess returns (one column per stock).
    factors_df : pd.DataFrame
        Columns [Mkt_RF, SMB, HML, MOM].

    Returns
    -------
    pd.DataFrame
        Full results table including all four factor loadings.
    """
    # Align stock returns with 4-factor data
    aligned = stock_excess_df.join(factors_df[["Mkt_RF", "SMB", "HML", "MOM"]], how="inner")

    results = []
    for stock in stock_excess_df.columns:
        model = carhart4_regression(aligned[stock], aligned[["Mkt_RF", "SMB", "HML", "MOM"]])
        res = extract_carhart4_results(model, stock)
        results.append(res)
        _print_carhart4_summary(res)

    return pd.DataFrame(results)


def _print_carhart4_summary(res: Dict[str, Any]) -> None:
    """Print formatted Carhart 4-factor summary for one stock."""
    sig = lambda p: "sig" if p < SIGNIFICANCE_LEVEL else "n.s."
    print(f"\n--- {res['Stock']} (Carhart 4-Factor) ---")
    print(f"  Annual Alpha : {res['Alpha'] * TRADING_DAYS:+.4f}  (p={res['Alpha_pvalue']:.3f}, {sig(res['Alpha_pvalue'])})")
    print(f"  Beta Mkt     : {res['Beta_Mkt']:.4f}  (p={res['Beta_Mkt_pvalue']:.3f}, {sig(res['Beta_Mkt_pvalue'])})")
    print(f"  Beta SMB     : {res['Beta_SMB']:.4f}  (p={res['Beta_SMB_pvalue']:.3f}, {sig(res['Beta_SMB_pvalue'])})")
    print(f"  Beta HML     : {res['Beta_HML']:.4f}  (p={res['Beta_HML_pvalue']:.3f}, {sig(res['Beta_HML_pvalue'])})")
    print(f"  Beta MOM     : {res['Beta_MOM']:.4f}  (p={res['Beta_MOM_pvalue']:.3f}, {sig(res['Beta_MOM_pvalue'])})")
    print(f"  R²           : {res['R_squared']:.4f}")


def compare_all_models(
    capm_results: pd.DataFrame,
    ff3_results: pd.DataFrame,
    carhart4_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a three-way R² comparison: CAPM vs FF3 vs Carhart 4-Factor.

    Parameters
    ----------
    capm_results : pd.DataFrame
        From run_all_regressions().
    ff3_results : pd.DataFrame
        From run_ff3_regressions().
    carhart4_results : pd.DataFrame
        From run_carhart4_regressions().

    Returns
    -------
    pd.DataFrame
        Comparison with R² and improvements for each model.
    """
    capm = capm_results[["Stock", "R_squared"]].rename(columns={"R_squared": "CAPM_R2"})
    ff3 = ff3_results[["Stock", "R_squared"]].rename(columns={"R_squared": "FF3_R2"})
    c4 = carhart4_results[["Stock", "R_squared"]].rename(columns={"R_squared": "C4_R2"})

    comp = capm.merge(ff3, on="Stock").merge(c4, on="Stock")
    comp["FF3_Improvement"] = comp["FF3_R2"] - comp["CAPM_R2"]
    comp["C4_Improvement"] = comp["C4_R2"] - comp["FF3_R2"]
    comp["Total_Improvement"] = comp["C4_R2"] - comp["CAPM_R2"]

    return comp


def plot_momentum_scores(
    momentum_scores: pd.DataFrame,
    save_path: str = None
) -> None:
    """Plot cross-sectional momentum rankings over time."""
    fig, ax = plt.subplots(figsize=(13, 6))

    for col in momentum_scores.columns:
        ax.plot(momentum_scores.index, momentum_scores[col], label=col, linewidth=1.5, marker="o", markersize=3)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Momentum Rank (higher = stronger momentum)", fontsize=12)
    ax.set_title("Cross-Sectional Momentum Rankings Over Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_model_r2_comparison(
    comparison_df: pd.DataFrame,
    save_path: str = None
) -> None:
    """Plot CAPM vs FF3 vs Carhart 4-Factor R² for all stocks."""
    stocks = comparison_df["Stock"]
    x = range(len(stocks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar([i - width for i in x], comparison_df["CAPM_R2"], width,
           label="CAPM", color="#95a5a6", edgecolor="black")
    ax.bar([i for i in x], comparison_df["FF3_R2"], width,
           label="FF3", color="#2980b9", edgecolor="black")
    ax.bar([i + width for i in x], comparison_df["C4_R2"], width,
           label="Carhart 4-Factor", color="#8e44ad", edgecolor="black")

    ax.set_xticks(list(x))
    ax.set_xticklabels(stocks, fontsize=12)
    ax.set_xlabel("Stock", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("R² Comparison: CAPM vs FF3 vs Carhart 4-Factor", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main() -> None:
    """Run momentum analysis and Carhart 4-factor regression pipeline."""
    from factor_model.data_collector import get_stock_returns
    from factor_model.ff3_collector import get_ff3_factors, get_stock_excess_and_factors
    from factor_model.regression import run_all_regressions
    from factor_model.ff3_regression import run_ff3_regressions

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    symbols = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    start, end = "2023-01-01", "2026-01-01"

    print("Downloading stock returns...")
    stock_returns = get_stock_returns(symbols, start, end)

    print("Downloading FF3 factors + Momentum...")
    ff3 = get_ff3_factors(start, end)
    mom = get_momentum_factor(start, end)

    stock_excess_df, factors_df = get_stock_excess_and_factors(stock_returns, ff3)
    carhart4_factors = build_carhart4_factors(ff3, mom)

    print("\n=== Carhart 4-Factor Regressions ===")
    c4_results = run_carhart4_regressions(stock_excess_df, carhart4_factors)

    capm_results = pd.read_csv(os.path.join(data_dir, "capm_results.csv"))
    ff3_results = pd.read_csv(os.path.join(data_dir, "ff3_results.csv"))

    comparison = compare_all_models(capm_results, ff3_results, c4_results)
    print("\n=== 3-Model R² Comparison ===")
    print(comparison[["Stock", "CAPM_R2", "FF3_R2", "C4_R2", "Total_Improvement"]].to_string(index=False))

    # Cross-sectional momentum
    mom_scores = calculate_cross_sectional_momentum(stock_returns)

    c4_results.to_csv(os.path.join(data_dir, "carhart4_results.csv"), index=False)
    comparison.to_csv(os.path.join(data_dir, "model_comparison_all.csv"), index=False)

    plot_model_r2_comparison(comparison, save_path=os.path.join(plots_dir, "r2_all_models.png"))
    plot_momentum_scores(mom_scores, save_path=os.path.join(plots_dir, "momentum_scores.png"))

    print(f"\nResults saved to: {data_dir}")


if __name__ == "__main__":
    main()
