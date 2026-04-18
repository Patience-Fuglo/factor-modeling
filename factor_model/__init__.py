"""
Factor Model Package
====================

A quantitative finance library for CAPM and Fama-French 3-factor regression
analysis, backtesting, and research report generation.

Modules:
--------
- data_collector  : Download and process market data from Yahoo Finance
- regression      : CAPM single-factor regression and analysis
- ff3_collector   : Download Fama-French 3 factors from Kenneth French's library
- ff3_regression  : Multi-factor FF3 regression and CAPM vs FF3 comparison
- backtest        : Factor-based portfolio backtesting with performance metrics

Example:
--------
    from factor_model import (
        get_stock_returns,
        get_market_returns,
        calculate_excess_returns,
        merge_data,
        single_factor_regression,
        run_all_regressions,
        get_ff3_factors,
        run_ff3_regressions,
        compare_models,
        Backtest,
    )
"""

__version__ = "2.0.0"
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

    # CAPM regression exports
    if name in (
        "single_factor_regression",
        "extract_results",
        "run_all_regressions",
        "plot_regression",
    ):
        from factor_model import regression
        return getattr(regression, name)

    # FF3 data collector exports
    if name in (
        "get_ff3_factors",
        "align_ff3_with_stocks",
        "get_stock_excess_and_factors",
    ):
        from factor_model import ff3_collector
        return getattr(ff3_collector, name)

    # FF3 regression exports
    if name in (
        "ff3_regression",
        "run_ff3_regressions",
        "compare_models",
        "plot_factor_loadings",
        "plot_r2_comparison",
    ):
        from factor_model import ff3_regression
        return getattr(ff3_regression, name)

    # Backtest exports
    if name in ("Backtest", "BacktestConfig", "BacktestResults"):
        from factor_model import backtest
        return getattr(backtest, name)

    # Momentum / Carhart 4-factor exports
    if name in (
        "get_momentum_factor",
        "calculate_cross_sectional_momentum",
        "build_carhart4_factors",
        "run_carhart4_regressions",
        "compare_all_models",
    ):
        from factor_model import momentum
        return getattr(momentum, name)

    # Regime detection exports
    if name in (
        "detect_regime",
        "factor_performance_by_regime",
        "regime_weights",
        "regime_timed_portfolio",
    ):
        from factor_model import regime
        return getattr(regime, name)

    # Portfolio optimisation exports
    if name in ("PortfolioOptimizer", "OptimizationConfig"):
        from factor_model import portfolio
        return getattr(portfolio, name)

    # FF5 exports
    if name in ("get_ff5_factors", "get_stock_excess_ff5", "run_ff5_regressions", "build_full_model_comparison"):
        from factor_model import ff5_regression
        return getattr(ff5_regression, name)

    # Walk-forward exports
    if name in ("WalkForwardTest", "SplitConfig"):
        from factor_model import walk_forward
        return getattr(walk_forward, name)

    # Rolling beta exports
    if name in ("RollingBetaAnalyzer",):
        from factor_model import rolling_beta
        return getattr(rolling_beta, name)

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
    # CAPM regression
    "single_factor_regression",
    "extract_results",
    "run_all_regressions",
    "plot_regression",
    # FF3 data
    "get_ff3_factors",
    "align_ff3_with_stocks",
    "get_stock_excess_and_factors",
    # FF3 regression
    "ff3_regression",
    "run_ff3_regressions",
    "compare_models",
    "plot_factor_loadings",
    "plot_r2_comparison",
    # Backtest
    "Backtest",
    "BacktestConfig",
    "BacktestResults",
]
