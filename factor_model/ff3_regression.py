"""
Fama-French 3-Factor Regression Module
========================================

Runs multi-factor regressions using the Fama-French 3-factor model and
compares results against single-factor CAPM to quantify the improvement.

FF3 Model:
    R_i - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + ε

Where:
    - α  = Abnormal return unexplained by all three factors
    - β₁ = Market sensitivity (same interpretation as CAPM beta)
    - β₂ = Size factor loading (positive = small-cap behaviour)
    - β₃ = Value factor loading (positive = value stock behaviour)

Example:
--------
    >>> from factor_model.ff3_regression import run_ff3_regressions, compare_models
    >>> ff3_results = run_ff3_regressions(stock_excess_df, factors_df)
    >>> comparison = compare_models(capm_results, ff3_results)
"""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


TRADING_DAYS: int = 252
SIGNIFICANCE_LEVEL: float = 0.05


def ff3_regression(
    stock_excess: pd.Series,
    factors: pd.DataFrame
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run Fama-French 3-factor regression for a single stock.

    Parameters
    ----------
    stock_excess : pd.Series
        Daily excess returns for one stock (R_i - R_f).
    factors : pd.DataFrame
        DataFrame with columns [Mkt_RF, SMB, HML].

    Returns
    -------
    RegressionResultsWrapper
        Fitted OLS model.
    """
    X = sm.add_constant(factors[["Mkt_RF", "SMB", "HML"]])
    return sm.OLS(stock_excess, X).fit()


def extract_ff3_results(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    stock_name: str
) -> Dict[str, Any]:
    """
    Extract key metrics from FF3 regression results.

    Parameters
    ----------
    result : RegressionResultsWrapper
        Fitted FF3 OLS model.
    stock_name : str
        Ticker symbol.

    Returns
    -------
    dict
        Keys: Stock, Alpha, Beta_Mkt, Beta_SMB, Beta_HML, R_squared,
              Alpha_pvalue, Beta_Mkt_pvalue, Beta_SMB_pvalue, Beta_HML_pvalue
    """
    params = result.params
    pvalues = result.pvalues

    return {
        "Stock": stock_name,
        "Alpha": params["const"],
        "Beta_Mkt": params["Mkt_RF"],
        "Beta_SMB": params["SMB"],
        "Beta_HML": params["HML"],
        "R_squared": result.rsquared,
        "Alpha_pvalue": pvalues["const"],
        "Beta_Mkt_pvalue": pvalues["Mkt_RF"],
        "Beta_SMB_pvalue": pvalues["SMB"],
        "Beta_HML_pvalue": pvalues["HML"],
    }


def run_ff3_regressions(
    stock_excess_df: pd.DataFrame,
    factors_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run FF3 regressions for all stocks.

    Parameters
    ----------
    stock_excess_df : pd.DataFrame
        Stock excess returns (one column per stock).
    factors_df : pd.DataFrame
        FF3 factors with columns [Mkt_RF, SMB, HML].

    Returns
    -------
    pd.DataFrame
        Results table with alpha, three betas, R², and p-values per stock.
    """
    results = []

    for stock in stock_excess_df.columns:
        model = ff3_regression(stock_excess_df[stock], factors_df)
        res = extract_ff3_results(model, stock)
        results.append(res)

        _print_ff3_summary(res)

    return pd.DataFrame(results)


def _print_ff3_summary(res: Dict[str, Any]) -> None:
    """Print formatted FF3 regression summary for one stock."""
    sig = lambda p: "sig" if p < SIGNIFICANCE_LEVEL else "n.s."
    annual_alpha = res["Alpha"] * TRADING_DAYS

    print(f"\n--- {res['Stock']} ---")
    print(f"  Annual Alpha : {annual_alpha:+.4f}  (p={res['Alpha_pvalue']:.3f}, {sig(res['Alpha_pvalue'])})")
    print(f"  Beta Mkt     : {res['Beta_Mkt']:.4f}  (p={res['Beta_Mkt_pvalue']:.3f}, {sig(res['Beta_Mkt_pvalue'])})")
    print(f"  Beta SMB     : {res['Beta_SMB']:.4f}  (p={res['Beta_SMB_pvalue']:.3f}, {sig(res['Beta_SMB_pvalue'])})")
    print(f"  Beta HML     : {res['Beta_HML']:.4f}  (p={res['Beta_HML_pvalue']:.3f}, {sig(res['Beta_HML_pvalue'])})")
    print(f"  R²           : {res['R_squared']:.4f}")


def compare_models(
    capm_results: pd.DataFrame,
    ff3_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Build side-by-side comparison of CAPM vs FF3 model fit.

    Parameters
    ----------
    capm_results : pd.DataFrame
        Output from run_all_regressions() (CAPM).
    ff3_results : pd.DataFrame
        Output from run_ff3_regressions() (FF3).

    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        [Stock, CAPM_Beta, CAPM_R2, FF3_Beta_Mkt, FF3_Beta_SMB,
         FF3_Beta_HML, FF3_R2, R2_Improvement]
    """
    capm = capm_results[["Stock", "Beta", "R_squared"]].copy()
    capm.columns = ["Stock", "CAPM_Beta", "CAPM_R2"]

    ff3 = ff3_results[["Stock", "Beta_Mkt", "Beta_SMB", "Beta_HML", "R_squared"]].copy()
    ff3.columns = ["Stock", "FF3_Beta_Mkt", "FF3_Beta_SMB", "FF3_Beta_HML", "FF3_R2"]

    comparison = capm.merge(ff3, on="Stock")
    comparison["R2_Improvement"] = comparison["FF3_R2"] - comparison["CAPM_R2"]

    return comparison


def plot_factor_loadings(
    ff3_results: pd.DataFrame,
    save_path: str = None
) -> None:
    """
    Plot SMB and HML factor loadings as grouped bar chart.

    Parameters
    ----------
    ff3_results : pd.DataFrame
        Output from run_ff3_regressions().
    save_path : str, optional
        File path to save the figure. Displays if None.
    """
    stocks = ff3_results["Stock"]
    x = range(len(stocks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar([i - width for i in x], ff3_results["Beta_Mkt"], width, label="Market (β₁)", color="#3498db", edgecolor="black")
    ax.bar([i for i in x], ff3_results["Beta_SMB"], width, label="SMB (β₂)", color="#2ecc71", edgecolor="black")
    ax.bar([i + width for i in x], ff3_results["Beta_HML"], width, label="HML (β₃)", color="#e74c3c", edgecolor="black")

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(stocks, fontsize=12)
    ax.set_xlabel("Stock", fontsize=12)
    ax.set_ylabel("Factor Loading (β)", fontsize=12)
    ax.set_title("Fama-French 3-Factor Loadings by Stock", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_r2_comparison(
    comparison_df: pd.DataFrame,
    save_path: str = None
) -> None:
    """
    Plot CAPM vs FF3 R² improvement per stock.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from compare_models().
    save_path : str, optional
        File path to save the figure.
    """
    stocks = comparison_df["Stock"]
    x = range(len(stocks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar([i - width / 2 for i in x], comparison_df["CAPM_R2"], width,
           label="CAPM R²", color="#95a5a6", edgecolor="black")
    ax.bar([i + width / 2 for i in x], comparison_df["FF3_R2"], width,
           label="FF3 R²", color="#2980b9", edgecolor="black")

    for i, (capm_r2, ff3_r2, imp) in enumerate(
        zip(comparison_df["CAPM_R2"], comparison_df["FF3_R2"], comparison_df["R2_Improvement"])
    ):
        ax.annotate(
            f"+{imp:.3f}",
            xy=(i + width / 2, ff3_r2),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#2980b9",
            fontweight="bold"
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(stocks, fontsize=12)
    ax.set_xlabel("Stock", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("Model Fit: CAPM vs Fama-French 3-Factor\n(Higher R² = Better Explanatory Power)", fontsize=13, fontweight="bold")
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
    """
    Run FF3 analysis pipeline.

    Loads stock returns and FF3 factors, runs regressions,
    compares against CAPM, and saves results and plots.
    """
    from factor_model.data_collector import get_stock_returns
    from factor_model.ff3_collector import get_ff3_factors, get_stock_excess_and_factors
    from factor_model.regression import run_all_regressions

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    symbols = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    start, end = "2023-01-01", "2026-01-01"

    print("Downloading stock returns...")
    stock_returns = get_stock_returns(symbols, start, end)

    print("Downloading FF3 factors...")
    ff3_factors = get_ff3_factors(start, end)

    stock_excess_df, factors_df = get_stock_excess_and_factors(stock_returns, ff3_factors)

    print("\n=== Fama-French 3-Factor Regressions ===")
    ff3_results = run_ff3_regressions(stock_excess_df, factors_df)

    # Load existing CAPM results for comparison
    capm_path = os.path.join(data_dir, "capm_results.csv")
    if os.path.exists(capm_path):
        capm_results = pd.read_csv(capm_path)
        comparison = compare_models(capm_results, ff3_results)
        print("\n=== CAPM vs FF3 Comparison ===")
        print(comparison[["Stock", "CAPM_R2", "FF3_R2", "R2_Improvement"]].to_string(index=False))

        comparison.to_csv(os.path.join(data_dir, "model_comparison.csv"), index=False)
        plot_r2_comparison(comparison, save_path=os.path.join(plots_dir, "r2_comparison.png"))

    ff3_results.to_csv(os.path.join(data_dir, "ff3_results.csv"), index=False)
    plot_factor_loadings(ff3_results, save_path=os.path.join(plots_dir, "factor_loadings.png"))

    print(f"\nResults saved to: {data_dir}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
