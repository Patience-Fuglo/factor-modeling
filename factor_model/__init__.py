"""
Factor Model Package
====================

A quantitative finance library for CAPM regression analysis
and factor-based portfolio risk assessment.

Modules:
--------
- data_collector: Download and process market data from Yahoo Finance
- regression: Perform CAPM regression and statistical analysis

Example:
--------
    from factor_model import (
        get_stock_returns,
        get_market_returns,
        calculate_excess_returns,
        merge_data,
        single_factor_regression,
        run_all_regressions
    )
"""

__version__ = "1.0.0"
__author__ = "Patience Fuglo"


def __getattr__(name):
    """Lazy import to avoid circular import warnings when running as module."""
    
    # Data collector exports
    if name in (
        "get_stock_returns",
        "get_market_returns", 
        "get_risk_free_rate",
        "calculate_excess_returns",
        "merge_data",
        "annualized_mean_returns",
        "TRADING_DAYS",
    ):
        from factor_model import data_collector
        return getattr(data_collector, name)
    
    # Regression exports
    if name in (
        "single_factor_regression",
        "extract_results",
        "run_all_regressions",
        "plot_regression",
    ):
        from factor_model import regression
        return getattr(regression, name)
    
    raise AttributeError(f"module 'factor_model' has no attribute '{name}'")


__all__ = [
    # Data collection
    "get_stock_returns",
    "get_market_returns",
    "get_risk_free_rate",
    "calculate_excess_returns",
    "merge_data",
    "annualized_mean_returns",
    "TRADING_DAYS",
    # Regression
    "single_factor_regression",
    "extract_results",
    "run_all_regressions",
    "plot_regression",
]
