"""
Fama-French 5-Factor Model
============================

Extends FF3 by adding two additional risk factors:
- RMW (Robust Minus Weak)  : Profitability premium
- CMA (Conservative Minus Aggressive) : Investment premium

FF5 Model:
    R_i - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(RMW) + β₅(CMA) + ε

RMW captures the tendency for profitable firms to outperform unprofitable ones.
CMA captures the tendency for conservatively investing firms to outperform
aggressive investors.

Note: FF5 subsumes the value premium (HML) in many markets — HML often becomes
insignificant once RMW and CMA are included.

Example:
--------
    >>> from factor_model.ff5_regression import get_ff5_factors, run_ff5_regressions
    >>> factors = get_ff5_factors("2023-01-01", "2026-01-01")
    >>> results = run_ff5_regressions(stock_excess_df, factors)
"""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


TRADING_DAYS: int = 252
SIGNIFICANCE_LEVEL: float = 0.05
FF5_DATASET = "F-F_Research_Data_5_Factors_2x3_daily"


def get_ff5_factors(start: str, end: str) -> pd.DataFrame:
    """
    Download daily Fama-French 5 factors from Kenneth French's data library.

    Parameters
    ----------
    start : str
        Start date "YYYY-MM-DD".
    end : str
        End date "YYYY-MM-DD".

    Returns
    -------
    pd.DataFrame
        Columns [Mkt_RF, SMB, HML, RMW, CMA, RF] as decimals.
    """
    raw = web.DataReader(FF5_DATASET, "famafrench", start, end)[0]

    if raw.empty:
        raise ValueError("No FF5 data downloaded. Check date range and internet connection.")

    factors = raw / 100.0
    factors.columns = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    factors.index = pd.to_datetime(factors.index)
    factors.index.name = "Date"

    return factors.sort_index()


def get_stock_excess_ff5(
    stock_returns: pd.DataFrame,
    ff5_factors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align stock returns with FF5 factors and compute excess returns.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Raw daily log returns per stock.
    ff5_factors : pd.DataFrame
        FF5 factors from get_ff5_factors().

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (stock_excess_df, factors_df) where factors_df has
        columns [Mkt_RF, SMB, HML, RMW, CMA].
    """
    merged = stock_returns.join(ff5_factors, how="inner")

    if merged.empty:
        raise ValueError("No common dates between stock returns and FF5 factors.")

    stock_cols = stock_returns.columns.tolist()
    for col in stock_cols:
        merged[f"{col}_excess"] = merged[col] - merged["RF"]

    excess_cols = [f"{col}_excess" for col in stock_cols]
    stock_excess_df = merged[excess_cols].copy()
    stock_excess_df.columns = stock_cols

    factors_df = merged[["Mkt_RF", "SMB", "HML", "RMW", "CMA"]].copy()

    return stock_excess_df, factors_df


def ff5_regression(
    stock_excess: pd.Series,
    factors: pd.DataFrame
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run Fama-French 5-factor OLS regression for a single stock.

    Parameters
    ----------
    stock_excess : pd.Series
        Daily excess returns (R_i - R_f).
    factors : pd.DataFrame
        Columns [Mkt_RF, SMB, HML, RMW, CMA].

    Returns
    -------
    RegressionResultsWrapper
        Fitted OLS model.
    """
    X = sm.add_constant(factors[["Mkt_RF", "SMB", "HML", "RMW", "CMA"]])
    return sm.OLS(stock_excess, X).fit()


def extract_ff5_results(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    stock_name: str
) -> Dict[str, Any]:
    """Extract key metrics from FF5 regression results."""
    params = result.params
    pvalues = result.pvalues

    return {
        "Stock": stock_name,
        "Alpha": params["const"],
        "Beta_Mkt": params["Mkt_RF"],
        "Beta_SMB": params["SMB"],
        "Beta_HML": params["HML"],
        "Beta_RMW": params["RMW"],
        "Beta_CMA": params["CMA"],
        "R_squared": result.rsquared,
        "Alpha_pvalue": pvalues["const"],
        "Beta_Mkt_pvalue": pvalues["Mkt_RF"],
        "Beta_RMW_pvalue": pvalues["RMW"],
        "Beta_CMA_pvalue": pvalues["CMA"],
    }


def run_ff5_regressions(
    stock_excess_df: pd.DataFrame,
    factors_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run FF5 regressions for all stocks.

    Parameters
    ----------
    stock_excess_df : pd.DataFrame
        Stock excess returns (one column per stock).
    factors_df : pd.DataFrame
        Columns [Mkt_RF, SMB, HML, RMW, CMA].

    Returns
    -------
    pd.DataFrame
        Full results table with all five factor loadings per stock.
    """
    results = []

    for stock in stock_excess_df.columns:
        model = ff5_regression(stock_excess_df[stock], factors_df)
        res = extract_ff5_results(model, stock)
        results.append(res)
        _print_ff5_summary(res)

    return pd.DataFrame(results)


def _print_ff5_summary(res: Dict[str, Any]) -> None:
    """Print formatted FF5 summary for one stock."""
    sig = lambda p: "sig" if p < SIGNIFICANCE_LEVEL else "n.s."
    print(f"\n--- {res['Stock']} (FF5) ---")
    print(f"  Annual Alpha : {res['Alpha'] * TRADING_DAYS:+.4f}  (p={res['Alpha_pvalue']:.3f}, {sig(res['Alpha_pvalue'])})")
    print(f"  Beta Mkt     : {res['Beta_Mkt']:.4f}  (p={res['Beta_Mkt_pvalue']:.3f}, {sig(res['Beta_Mkt_pvalue'])})")
    print(f"  Beta SMB     : {res['Beta_SMB']:.4f}")
    print(f"  Beta HML     : {res['Beta_HML']:.4f}")
    print(f"  Beta RMW     : {res['Beta_RMW']:.4f}  (p={res['Beta_RMW_pvalue']:.3f}, {sig(res['Beta_RMW_pvalue'])})")
    print(f"  Beta CMA     : {res['Beta_CMA']:.4f}  (p={res['Beta_CMA_pvalue']:.3f}, {sig(res['Beta_CMA_pvalue'])})")
    print(f"  R²           : {res['R_squared']:.4f}")


def build_full_model_comparison(
    capm_r2: pd.Series,
    ff3_r2: pd.Series,
    c4_r2: pd.Series,
    ff5_r2: pd.Series,
    stocks: list[str]
) -> pd.DataFrame:
    """
    Build R² comparison table across all four models.

    Parameters
    ----------
    capm_r2, ff3_r2, c4_r2, ff5_r2 : pd.Series
        R² values indexed by stock name.

    Returns
    -------
    pd.DataFrame
        Side-by-side R² with improvement columns.
    """
    df = pd.DataFrame({
        "Stock": stocks,
        "CAPM_R2": capm_r2.values,
        "FF3_R2": ff3_r2.values,
        "C4_R2": c4_r2.values,
        "FF5_R2": ff5_r2.values,
    })
    df["FF5_vs_CAPM"] = df["FF5_R2"] - df["CAPM_R2"]
    df["FF5_vs_FF3"] = df["FF5_R2"] - df["FF3_R2"]
    return df


def plot_full_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: str = None
) -> None:
    """Bar chart comparing R² across CAPM, FF3, Carhart 4, and FF5."""
    import numpy as np

    stocks = comparison_df["Stock"]
    x = np.arange(len(stocks))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - 1.5 * width, comparison_df["CAPM_R2"], width, label="CAPM", color="#95a5a6", edgecolor="black")
    ax.bar(x - 0.5 * width, comparison_df["FF3_R2"], width, label="FF3", color="#2980b9", edgecolor="black")
    ax.bar(x + 0.5 * width, comparison_df["C4_R2"], width, label="Carhart 4", color="#8e44ad", edgecolor="black")
    ax.bar(x + 1.5 * width, comparison_df["FF5_R2"], width, label="FF5", color="#e67e22", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(stocks, fontsize=12)
    ax.set_xlabel("Stock", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("R² Across All Models: CAPM → FF3 → Carhart 4 → FF5",
                 fontsize=13, fontweight="bold")
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
    """Run FF5 analysis pipeline."""
    from factor_model.data_collector import get_stock_returns

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    symbols = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    start, end = "2023-01-01", "2026-01-01"

    print("Downloading stock returns...")
    stock_returns = get_stock_returns(symbols, start, end)

    print("Downloading FF5 factors...")
    ff5_factors = get_ff5_factors(start, end)

    stock_excess_df, factors_df = get_stock_excess_ff5(stock_returns, ff5_factors)

    print("\n=== Fama-French 5-Factor Regressions ===")
    ff5_results = run_ff5_regressions(stock_excess_df, factors_df)

    ff5_results.to_csv(os.path.join(data_dir, "ff5_results.csv"), index=False)

    # Build full 4-model comparison
    capm = pd.read_csv(os.path.join(data_dir, "capm_results.csv"))
    ff3 = pd.read_csv(os.path.join(data_dir, "ff3_results.csv"))
    c4 = pd.read_csv(os.path.join(data_dir, "carhart4_results.csv"))

    comparison = build_full_model_comparison(
        capm.set_index("Stock")["R_squared"],
        ff3.set_index("Stock")["R_squared"],
        c4.set_index("Stock")["R_squared"],
        ff5_results.set_index("Stock")["R_squared"],
        symbols
    )

    print("\n=== Full Model R² Comparison ===")
    print(comparison.to_string(index=False))

    comparison.to_csv(os.path.join(data_dir, "full_model_comparison.csv"), index=False)
    plot_full_model_comparison(
        comparison,
        save_path=os.path.join(plots_dir, "r2_all_four_models.png")
    )

    print(f"\nResults saved to: {data_dir}")


if __name__ == "__main__":
    main()
