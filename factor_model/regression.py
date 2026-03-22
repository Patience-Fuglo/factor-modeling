"""
Regression Module
=================

Performs CAPM (Capital Asset Pricing Model) regression analysis
to estimate alpha (abnormal return) and beta (systematic risk).

The CAPM model:
    R_i - R_f = α + β(R_m - R_f) + ε

Where:
    - R_i = Stock return
    - R_f = Risk-free rate
    - R_m = Market return
    - α (alpha) = Abnormal return (Jensen's alpha)
    - β (beta) = Systematic risk / market sensitivity
    - ε = Error term

Example:
--------
    >>> from factor_model.regression import run_all_regressions
    >>> results = run_all_regressions(stock_excess_df, market_excess)
    >>> print(results)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Any, Optional

import pandas as pd
import statsmodels.api as sm

# Configure matplotlib backend BEFORE importing pyplot
import matplotlib
if '--show' in sys.argv:
    # Use native macOS backend for interactive display
    matplotlib.use('macosx')
else:
    # Use non-interactive backend for saving files
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


# Constants
TRADING_DAYS: int = 252
SIGNIFICANCE_LEVEL: float = 0.05


def single_factor_regression(
    stock_returns: pd.Series,
    market_returns: pd.Series
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run single-factor CAPM regression.

    Estimates: stock = alpha + beta * market + error

    Parameters
    ----------
    stock_returns : pd.Series
        Excess returns of the stock.
    market_returns : pd.Series
        Excess returns of the market.

    Returns
    -------
    RegressionResultsWrapper
        Fitted OLS regression model with params, pvalues, rsquared, etc.

    Example
    -------
        >>> model = single_factor_regression(aapl_excess, market_excess)
        >>> print(f"Beta: {model.params.iloc[1]:.4f}")
        Beta: 1.1638
    """
    X = sm.add_constant(market_returns)  # adds intercept
    model = sm.OLS(stock_returns, X).fit()
    return model


def extract_results(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    stock_name: str
) -> Dict[str, Any]:
    """
    Extract key metrics from regression results.

    Parameters
    ----------
    result : RegressionResultsWrapper
        Fitted OLS regression model.
    stock_name : str
        Name/ticker of the stock.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - Stock: Stock name
        - Alpha: Daily alpha (intercept)
        - Beta: Market beta (slope)
        - R_squared: Model fit (0-1)
        - Alpha_pvalue: Statistical significance of alpha
        - Beta_pvalue: Statistical significance of beta

    Example
    -------
        >>> results = extract_results(model, "AAPL")
        >>> print(f"R²: {results['R_squared']:.4f}")
        R²: 0.4726
    """
    alpha = result.params.iloc[0]
    beta = result.params.iloc[1]

    r_squared = result.rsquared

    alpha_pvalue = result.pvalues.iloc[0]
    beta_pvalue = result.pvalues.iloc[1]

    return {
        "Stock": stock_name,
        "Alpha": alpha,
        "Beta": beta,
        "R_squared": r_squared,
        "Alpha_pvalue": alpha_pvalue,
        "Beta_pvalue": beta_pvalue,
    }


def print_regression_summary(
    stock_name: str,
    alpha: float,
    beta: float,
    r_squared: float,
    alpha_pval: float,
    beta_pval: float
) -> None:
    """
    Print formatted regression results summary.

    Parameters
    ----------
    stock_name : str
        Name/ticker of the stock.
    alpha : float
        Daily alpha (intercept).
    beta : float
        Market beta (slope).
    r_squared : float
        R-squared value.
    alpha_pval : float
        P-value for alpha.
    beta_pval : float
        P-value for beta.
    """
    print("\n------------------------------")
    print(f"Stock: {stock_name}")

    print(f"Annualized Alpha: {alpha * TRADING_DAYS:.4f}")
    print(f"Beta: {beta:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    alpha_sig = "Significant" if alpha_pval < SIGNIFICANCE_LEVEL else "Not Significant"
    beta_sig = "Significant" if beta_pval < SIGNIFICANCE_LEVEL else "Not Significant"

    print(f"Alpha p-value: {alpha_pval:.4f} ({alpha_sig})")
    print(f"Beta p-value: {beta_pval:.4f} ({beta_sig})")


def run_all_regressions(
    stock_excess_df: pd.DataFrame,
    market_excess: pd.Series
) -> pd.DataFrame:
    """
    Run CAPM regressions for all stocks in DataFrame.

    Parameters
    ----------
    stock_excess_df : pd.DataFrame
        DataFrame with stock excess returns (one column per stock).
    market_excess : pd.Series
        Series of market excess returns.

    Returns
    -------
    pd.DataFrame
        Results table with columns:
        [Stock, Alpha, Beta, R_squared, Alpha_pvalue, Beta_pvalue]

    Example
    -------
        >>> results_df = run_all_regressions(stocks, market)
        >>> print(results_df[["Stock", "Beta", "R_squared"]])
           Stock    Beta  R_squared
        0   AAPL  1.1638     0.4726
        1   MSFT  1.0402     0.4575
    """
    results = []

    for stock in stock_excess_df.columns:
        model = single_factor_regression(stock_excess_df[stock], market_excess)

        res = extract_results(model, stock)
        results.append(res)

        # print each result
        print_regression_summary(
            stock,
            res["Alpha"],
            res["Beta"],
            res["R_squared"],
            res["Alpha_pvalue"],
            res["Beta_pvalue"],
        )

    return pd.DataFrame(results)


def plot_regression(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    stock_name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Create scatter plot with CAPM regression line.

    Parameters
    ----------
    stock_returns : pd.Series
        Excess returns of the stock.
    market_returns : pd.Series
        Excess returns of the market.
    stock_name : str
        Name/ticker of the stock (for title).
    save_path : str, optional
        Path to save figure. If None, displays interactively.

    Example
    -------
        >>> plot_regression(aapl_returns, market, "AAPL", "plots/aapl_capm.png")
    """
    model = single_factor_regression(stock_returns, market_returns)

    plt.figure(figsize=(8, 6))
    plt.scatter(market_returns, stock_returns, alpha=0.5, label="Daily Returns")

    # regression line
    plt.plot(
        market_returns.sort_values(),
        model.predict(sm.add_constant(market_returns.sort_values())),
        color="red",
        linewidth=2,
        label=f"CAPM: β={model.params.iloc[1]:.2f}"
    )

    plt.xlabel("Market Excess Returns", fontsize=12)
    plt.ylabel(f"{stock_name} Excess Returns", fontsize=12)
    plt.title(f"CAPM Regression: {stock_name}", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)  # Block until window is closed


def plot_beta_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Create bar chart comparing betas across stocks.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_all_regressions().
    save_path : str, optional
        Path to save figure.
    """
    plt.figure(figsize=(10, 6))

    colors = ["#2ecc71" if b < 1 else "#e74c3c" for b in results_df["Beta"]]

    bars = plt.bar(results_df["Stock"], results_df["Beta"], color=colors, edgecolor="black")

    plt.axhline(y=1, color="black", linestyle="--", linewidth=1.5, label="Market (β=1)")

    plt.xlabel("Stock", fontsize=12)
    plt.ylabel("Beta (β)", fontsize=12)
    plt.title("CAPM Beta Comparison", fontsize=14)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for bar, beta in zip(bars, results_df["Beta"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{beta:.2f}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)  # Block until window is closed


def main() -> None:
    """
    Run CAPM analysis pipeline.

    Loads data, runs regressions, saves results and generates plots.
    
    Usage:
        python -m factor_model.regression          # Save plots to files
        python -m factor_model.regression --show   # Display plots interactively
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CAPM regression analysis")
    parser.add_argument(
        "--show", 
        action="store_true", 
        help="Display plots interactively (opens plot windows)"
    )
    parser.add_argument(
        "--save-plots", 
        action="store_true", 
        help="Generate and save plots to plots/ directory"
    )
    args = parser.parse_args()
    
    # Load data from data/ directory
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "merged_excess_returns.csv")
    data = pd.read_csv(
        data_path,
        index_col=0,
        parse_dates=True
    )

    market = data["Market"]
    stocks = data.drop(columns=["Market"])

    print("\nRunning CAPM regressions...\n")

    results_df = run_all_regressions(stocks, market)

    print("\n\n=== SUMMARY TABLE ===")
    print(results_df)

    # Save results to data/ directory
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "capm_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Handle plot generation
    if args.show:
        # Display plots interactively
        print("\nDisplaying plots (close each window to continue)...")
        plot_beta_comparison(results_df, save_path=None)
        
        for stock in stocks.columns:
            plot_regression(stocks[stock], market, stock, save_path=None)
            
    elif args.save_plots:
        # Save plots to files
        plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        print("\nGenerating plots...")
        plot_beta_comparison(
            results_df,
            save_path=os.path.join(plots_dir, "beta_comparison.png")
        )

        for stock in stocks.columns:
            plot_regression(
                stocks[stock],
                market,
                stock,
                save_path=os.path.join(plots_dir, f"capm_{stock.lower()}.png")
            )

        print(f"All plots saved to: {plots_dir}")
    else:
        print("\nTip: Use --save-plots to generate visualizations or --show to display them")


if __name__ == "__main__":
    main()