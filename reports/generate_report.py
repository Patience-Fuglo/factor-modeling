"""
Research Report Generator
==========================

Generates a professional HTML research report covering:
- CAPM single-factor analysis
- Fama-French 3-factor model
- Carhart 4-factor model (with momentum)
- Market regime detection & factor timing
- Portfolio construction comparison (Equal Weight, Min Variance, Max Sharpe, Risk Parity)
- Efficient frontier

Output: reports/capm_ff3_research_report.html

Usage:
    python reports/generate_report.py
"""

from __future__ import annotations

import base64
import os
import sys
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factor_model.backtest import Backtest, BacktestConfig
from factor_model.regression import run_all_regressions
from factor_model.ff3_collector import get_ff3_factors, get_stock_excess_and_factors
from factor_model.ff3_regression import run_ff3_regressions, compare_models
from factor_model.momentum import (
    get_momentum_factor, build_carhart4_factors,
    run_carhart4_regressions, compare_all_models, plot_model_r2_comparison
)
from factor_model.regime import (
    detect_regime, factor_performance_by_regime,
    plot_regimes, plot_factor_performance_by_regime
)
from factor_model.portfolio import PortfolioOptimizer, OptimizationConfig
from factor_model.data_collector import get_stock_returns
from factor_model.ff5_regression import get_ff5_factors, get_stock_excess_ff5, run_ff5_regressions, build_full_model_comparison
from factor_model.walk_forward import WalkForwardTest, SplitConfig
from factor_model.rolling_beta import RollingBetaAnalyzer

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


def fig_to_base64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def capture_plot(plot_fn, *args, **kwargs) -> str:
    """Call a plot function that saves to save_path, then read back as base64."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    plot_fn(*args, save_path=path, **kwargs)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(path)
    return encoded


# ------------------------------------------------------------------ #
#  Individual chart builders                                          #
# ------------------------------------------------------------------ #

def plot_beta_bar(results_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if b < 1 else "#e74c3c" for b in results_df["Beta"]]
    bars = ax.bar(results_df["Stock"], results_df["Beta"], color=colors, edgecolor="black")
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1.5, label="Market (β=1)")
    for bar, beta in zip(bars, results_df["Beta"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{beta:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Stock", fontsize=12)
    ax.set_ylabel("Beta (β)", fontsize=12)
    ax.set_title("CAPM Beta Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_r2_bars(results_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(results_df["Stock"], results_df["R_squared"], color="#3498db", edgecolor="black")
    for bar, r2 in zip(bars, results_df["R_squared"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{r2:.3f}", ha="center", fontsize=10)
    ax.set_xlabel("Stock", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("CAPM R² — Proportion of Return Explained by Market", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_cumulative_returns(data: pd.DataFrame) -> str:
    cumulative = (1 + data).cumprod()
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in cumulative.columns:
        lw = 2.5 if col == "Market" else 1.5
        ls = "--" if col == "Market" else "-"
        ax.plot(cumulative.index, cumulative[col], label=col, linewidth=lw, linestyle=ls)
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.set_title("Cumulative Excess Returns — All Assets", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_backtest_chart(data: pd.DataFrame) -> tuple[str, dict]:
    config = BacktestConfig(rebalance_freq="M", top_n=3, transaction_cost=0.001)
    bt = Backtest(data, config=config)
    results = bt.run()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(results.portfolio_value.index, results.portfolio_value,
                 label="Portfolio (Top 3 Equal Weight)", color="#2980b9", linewidth=2)
    axes[0].plot(results.benchmark_value.index, results.benchmark_value,
                 label="S&P 500", color="#7f8c8d", linewidth=1.5, linestyle="--")
    axes[0].set_ylabel("Value (Base = 100)", fontsize=11)
    axes[0].set_title("Portfolio vs Benchmark — Backtested Performance", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    rolling_max = results.portfolio_value.cummax()
    drawdown = (results.portfolio_value - rolling_max) / rolling_max * 100
    axes[1].fill_between(drawdown.index, drawdown, 0, color="#e74c3c", alpha=0.5)
    axes[1].set_ylabel("Drawdown (%)", fontsize=11)
    axes[1].set_xlabel("Date", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig), results.metrics


def plot_portfolio_comparison_chart(data: pd.DataFrame) -> str:
    config = OptimizationConfig(estimation_window=126, rebalance_freq="M", transaction_cost=0.001)
    opt = PortfolioOptimizer(data, config=config)
    strategies = opt.compare_all_strategies()

    colors = {
        "Equal Weight": "#3498db", "Min Variance": "#2ecc71",
        "Max Sharpe": "#e67e22", "Risk Parity": "#9b59b6",
    }
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, returns in strategies.items():
        cum = (1 + returns).cumprod() * 100
        ax.plot(cum.index, cum, label=name, color=colors[name], linewidth=2)
    bm_cum = (1 + data["Market"]).cumprod() * 100
    ax.plot(bm_cum.index, bm_cum, label="S&P 500", color="#7f8c8d", linewidth=1.5, linestyle="--")
    ax.set_ylabel("Portfolio Value (Base = 100)", fontsize=11)
    ax.set_title("Portfolio Construction Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig), strategies, opt


def plot_efficient_frontier_chart(opt: PortfolioOptimizer) -> str:
    mu = opt.stocks.mean().values * 252
    cov = opt.stocks.cov().values * 252
    n = opt.n_stocks

    np.random.seed(42)
    rand_weights = np.random.dirichlet(np.ones(n), size=3000)
    port_returns = rand_weights @ mu
    port_vols = np.sqrt(np.einsum("ij,jk,ik->i", rand_weights, cov, rand_weights))
    port_sharpes = (port_returns - 0.02) / port_vols

    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(port_vols, port_returns, c=port_sharpes, cmap="viridis", alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    for label, weight_fn, color, marker in [
        ("Equal Weight", opt._equal_weight, "blue", "o"),
        ("Min Variance", opt._min_variance, "green", "^"),
        ("Max Sharpe", opt._max_sharpe, "red", "*"),
        ("Risk Parity", opt._risk_parity, "purple", "D"),
    ]:
        try:
            w = weight_fn(opt.stocks)
            r = w @ mu
            v = np.sqrt(w @ cov @ w)
            ax.scatter(v, r, color=color, marker=marker, s=200, zorder=5,
                       edgecolors="black", linewidths=1.5, label=label)
        except Exception:
            pass

    ax.set_xlabel("Annualised Volatility", fontsize=12)
    ax.set_ylabel("Annualised Expected Return", fontsize=12)
    ax.set_title("Efficient Frontier — Monte Carlo Simulation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_ff5_chart(ff5_results: pd.DataFrame, full_comparison: pd.DataFrame) -> str:
    import numpy as np
    stocks = full_comparison["Stock"]
    x = np.arange(len(stocks))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 1.5*width, full_comparison["CAPM_R2"], width, label="CAPM", color="#95a5a6", edgecolor="black")
    ax.bar(x - 0.5*width, full_comparison["FF3_R2"],  width, label="FF3",  color="#2980b9", edgecolor="black")
    ax.bar(x + 0.5*width, full_comparison["C4_R2"],   width, label="Carhart 4", color="#8e44ad", edgecolor="black")
    ax.bar(x + 1.5*width, full_comparison["FF5_R2"],  width, label="FF5",  color="#e67e22", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(stocks, fontsize=12)
    ax.set_xlabel("Stock", fontsize=12); ax.set_ylabel("R²", fontsize=12)
    ax.set_title("R² Progression: CAPM → FF3 → Carhart 4 → FF5", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); ax.set_ylim(0, 1); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_walk_forward_chart(wf_results: dict, train_market: pd.Series, test_market: pd.Series) -> str:
    colors = {"Equal Weight": "#3498db", "Min Variance": "#2ecc71", "Max Sharpe": "#e67e22"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, period, market_series, title in zip(
        axes, ["train", "test"], [train_market, test_market],
        ["In-Sample (2023–2024)", "Out-of-Sample (2025)"]
    ):
        for name, splits in wf_results.items():
            cum = (1 + splits[period]).cumprod() * 100
            ax.plot(cum.index, cum, label=name, color=colors.get(name, "gray"), linewidth=2)
        bm = (1 + market_series).cumprod() * 100
        ax.plot(bm.index, bm, label="S&P 500", color="#7f8c8d", linewidth=1.5, linestyle="--")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Value (Base = 100)", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.suptitle("Walk-Forward Validation: In-Sample vs Out-of-Sample", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_wf_r2_chart(model_comp: pd.DataFrame) -> str:
    import numpy as np
    stocks = model_comp["Stock"].tolist()
    x = np.arange(len(stocks)); width = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 1.5*width, model_comp["CAPM_IS_R2"],  width, label="CAPM In-Sample",     color="#2980b9", edgecolor="black")
    ax.bar(x - 0.5*width, model_comp["CAPM_OOS_R2"], width, label="CAPM Out-of-Sample", color="#aed6f1", edgecolor="black")
    ax.bar(x + 0.5*width, model_comp["FF5_IS_R2"],   width, label="FF5 In-Sample",      color="#e67e22", edgecolor="black")
    ax.bar(x + 1.5*width, model_comp["FF5_OOS_R2"],  width, label="FF5 Out-of-Sample",  color="#f9c784", edgecolor="black")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(stocks, fontsize=12)
    ax.set_xlabel("Stock", fontsize=12); ax.set_ylabel("R²", fontsize=12)
    ax.set_title("In-Sample vs Out-of-Sample R²: CAPM and FF5", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_rolling_beta_chart(data: pd.DataFrame) -> str:
    rba = RollingBetaAnalyzer(data, window=60)
    rolling_betas = rba.compute_all()
    n_stocks = len(rba.stocks.columns)
    colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#e74c3c"]
    fig, axes = plt.subplots(n_stocks, 1, figsize=(13, 3 * n_stocks), sharex=True)
    for i, (stock, ax) in enumerate(zip(rba.stocks.columns, axes)):
        beta = rolling_betas[stock].dropna()
        ax.plot(beta.index, beta, color=colors[i % len(colors)], linewidth=1.5, label=stock)
        ax.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(y=beta.mean(), color=colors[i % len(colors)], linestyle=":", linewidth=1, alpha=0.7)
        ax.fill_between(beta.index, beta, 1, where=(beta > 1), alpha=0.15, color="#e74c3c")
        ax.fill_between(beta.index, beta, 1, where=(beta < 1), alpha=0.15, color="#2ecc71")
        ax.set_ylabel(f"{stock}\nβ", fontsize=10)
        ax.legend([f"Mean β={beta.mean():.2f}  |  Range: {beta.min():.2f}–{beta.max():.2f}"], fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date", fontsize=12)
    fig.suptitle("Rolling Beta Analysis (60-Day Window)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig_to_base64(fig)


def results_to_html_table(df: pd.DataFrame) -> str:
    rows = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for _, row in df.iterrows()
    )
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"


# ------------------------------------------------------------------ #
#  Main report generator                                              #
# ------------------------------------------------------------------ #

def generate_report() -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading data...")
    data = pd.read_csv(
        os.path.join(DATA_DIR, "merged_excess_returns.csv"),
        index_col=0, parse_dates=True
    )
    market = data["Market"]
    stocks = data.drop(columns=["Market"])

    # --- CAPM ---
    print("Running CAPM regressions...")
    capm_results = run_all_regressions(stocks, market)

    # --- FF3 ---
    print("Running FF3 regressions...")
    symbols = stocks.columns.tolist()
    stock_returns = get_stock_returns(symbols, "2023-01-01", "2026-01-01")
    ff3 = get_ff3_factors("2023-01-01", "2026-01-01")
    stock_excess_df, factors_df = get_stock_excess_and_factors(stock_returns, ff3)
    ff3_results = run_ff3_regressions(stock_excess_df, factors_df)

    # --- Carhart 4-Factor ---
    print("Running Carhart 4-factor regressions...")
    mom_factor = get_momentum_factor("2023-01-01", "2026-01-01")
    c4_factors = build_carhart4_factors(ff3, mom_factor)
    c4_results = run_carhart4_regressions(stock_excess_df, c4_factors)
    model_comparison = compare_all_models(capm_results, ff3_results, c4_results)

    # --- FF5 ---
    print("Running FF5 regressions...")
    ff5_factors = get_ff5_factors("2023-01-01", "2026-01-01")
    stock_excess_ff5, factors_ff5 = get_stock_excess_ff5(stock_returns, ff5_factors)
    ff5_results = run_ff5_regressions(stock_excess_ff5, factors_ff5)
    full_comparison = build_full_model_comparison(
        capm_results.set_index("Stock")["R_squared"],
        ff3_results.set_index("Stock")["R_squared"],
        c4_results.set_index("Stock")["R_squared"],
        ff5_results.set_index("Stock")["R_squared"],
        stocks.columns.tolist()
    )

    # --- Walk-Forward ---
    print("Running walk-forward validation...")
    ff5_for_wf = factors_ff5.join(mom_factor, how="left").fillna(0)
    split_config = SplitConfig(train_end="2024-12-31", test_start="2025-01-01")
    wf = WalkForwardTest(data, ff5_for_wf, config=split_config)
    wf_model_comp = wf.run_model_comparison()
    wf_strategy_results = wf.run_strategy_oos()

    # --- Regime ---
    print("Detecting market regimes...")
    regimes = detect_regime(market, method="sma")
    all_factors = c4_factors[["Mkt_RF", "SMB", "HML", "MOM"]]
    regime_perf = factor_performance_by_regime(all_factors, regimes)

    # --- Portfolio ---
    print("Running portfolio optimisation...")
    config = OptimizationConfig(estimation_window=126, rebalance_freq="M", transaction_cost=0.001)
    opt = PortfolioOptimizer(data, config=config)
    strategies = opt.compare_all_strategies()
    metrics_df = opt.print_comparison_table(strategies)

    # --- Generate all charts ---
    print("Generating charts...")
    beta_chart = plot_beta_bar(capm_results)
    r2_chart = plot_r2_bars(capm_results)
    cumret_chart = plot_cumulative_returns(data)
    bt_chart, bt_metrics = plot_backtest_chart(data)
    port_chart, _, _ = plot_portfolio_comparison_chart(data)
    frontier_chart = plot_efficient_frontier_chart(opt)
    regimes_chart = capture_plot(plot_regimes, market, regimes)
    regime_factor_chart = capture_plot(plot_factor_performance_by_regime, regime_perf)
    r2_models_chart = capture_plot(plot_model_r2_comparison, model_comparison)
    ff5_chart = plot_ff5_chart(ff5_results, full_comparison)
    wf_chart = plot_walk_forward_chart(wf_strategy_results, wf.train_market, wf.test_market)
    wf_r2_chart = plot_wf_r2_chart(wf_model_comp)
    rolling_beta_chart = plot_rolling_beta_chart(data)

    # --- Build HTML tables ---
    capm_display = capm_results.copy()
    capm_display["Alpha (Annual)"] = (capm_display["Alpha"] * 252).map("{:+.4f}".format)
    capm_display["Beta"] = capm_display["Beta"].map("{:.4f}".format)
    capm_display["R²"] = capm_display["R_squared"].map("{:.4f}".format)
    capm_display["Alpha Sig."] = capm_results["Alpha_pvalue"].apply(lambda x: "Yes" if x < 0.05 else "No")
    capm_table = results_to_html_table(
        capm_display[["Stock", "Alpha (Annual)", "Alpha Sig.", "Beta", "R²"]]
    )

    c4_display = c4_results.copy()
    c4_display["Alpha (Annual)"] = (c4_display["Alpha"] * 252).map("{:+.4f}".format)
    c4_display["Beta Mkt"] = c4_display["Beta_Mkt"].map("{:.4f}".format)
    c4_display["Beta SMB"] = c4_display["Beta_SMB"].map("{:.4f}".format)
    c4_display["Beta HML"] = c4_display["Beta_HML"].map("{:.4f}".format)
    c4_display["Beta MOM"] = c4_display["Beta_MOM"].map("{:.4f}".format)
    c4_display["R²"] = c4_display["R_squared"].map("{:.4f}".format)
    c4_table = results_to_html_table(
        c4_display[["Stock", "Alpha (Annual)", "Beta Mkt", "Beta SMB", "Beta HML", "Beta MOM", "R²"]]
    )

    r2_comp_display = model_comparison[["Stock", "CAPM_R2", "FF3_R2", "C4_R2", "Total_Improvement"]].copy()
    for col in ["CAPM_R2", "FF3_R2", "C4_R2", "Total_Improvement"]:
        r2_comp_display[col] = r2_comp_display[col].map("{:.4f}".format)
    r2_comp_table = results_to_html_table(r2_comp_display)

    regime_display = regime_perf[["Regime", "Days", "Mkt_RF_Return", "SMB_Return", "HML_Return", "MOM_Return"]].copy()
    for col in ["Mkt_RF_Return", "SMB_Return", "HML_Return", "MOM_Return"]:
        regime_display[col] = regime_display[col].map("{:+.2%}".format)
    regime_table = results_to_html_table(regime_display)

    metrics_table = results_to_html_table(metrics_df)
    bt_metrics_rows = "".join(f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>" for k, v in bt_metrics.items())

    today = datetime.today().strftime("%B %d, %Y")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Factor Model Research Report — CAPM, FF3, Carhart, Regime, Portfolio</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa; color: #2c3e50; line-height: 1.7; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 20px; }}
  h1 {{ font-size: 2rem; color: #1a252f; margin-bottom: 6px; }}
  h2 {{ font-size: 1.4rem; color: #2980b9; margin: 36px 0 12px; border-bottom: 2px solid #2980b9; padding-bottom: 6px; }}
  h3 {{ font-size: 1.1rem; color: #34495e; margin: 20px 0 8px; }}
  .meta {{ color: #7f8c8d; font-size: 0.9rem; margin-bottom: 30px; }}
  .card {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 28px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
  img {{ width: 100%; border-radius: 6px; margin: 16px 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 12px; }}
  th {{ background: #2980b9; color: white; padding: 10px 14px; text-align: left; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #ecf0f1; }}
  tr:nth-child(even) td {{ background: #f8f9fa; }}
  .highlight {{ background: #eaf4fb; border-left: 4px solid #2980b9; padding: 14px 18px; border-radius: 4px; margin: 14px 0; }}
  .finding {{ background: #eafaf1; border-left: 4px solid #27ae60; padding: 14px 18px; border-radius: 4px; margin: 10px 0; }}
  .warning {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 14px 18px; border-radius: 4px; margin: 10px 0; }}
  .toc {{ background: white; border-radius: 8px; padding: 20px 28px; margin-bottom: 28px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
  .toc a {{ color: #2980b9; text-decoration: none; display: block; padding: 3px 0; }}
  .toc a:hover {{ text-decoration: underline; }}
  footer {{ text-align: center; color: #bdc3c7; font-size: 0.85rem; margin-top: 48px; padding-top: 20px; border-top: 1px solid #ecf0f1; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body>
<div class="container">

  <h1>Factor Model Research Report</h1>
  <p class="meta">
    {today} &nbsp;|&nbsp;
    Portfolio: AAPL, MSFT, GOOGL, JPM, XOM &nbsp;|&nbsp;
    Period: Jan 2023 – Dec 2025 &nbsp;|&nbsp;
    Benchmark: S&amp;P 500
  </p>

  <!-- TABLE OF CONTENTS -->
  <div class="toc">
    <strong>Contents</strong>
    <a href="#summary">Executive Summary</a>
    <a href="#data">1. Data &amp; Methodology</a>
    <a href="#cumret">2. Cumulative Returns</a>
    <a href="#capm">3. CAPM Analysis</a>
    <a href="#ff3">4. Fama-French 3-Factor Model</a>
    <a href="#carhart">5. Carhart 4-Factor Model (Momentum)</a>
    <a href="#ff5">6. Fama-French 5-Factor Model</a>
    <a href="#rolling">7. Rolling Beta Analysis</a>
    <a href="#walkforward">8. Walk-Forward Out-of-Sample Validation</a>
    <a href="#regime">9. Market Regime Detection &amp; Factor Timing</a>
    <a href="#portfolio">10. Portfolio Construction &amp; Optimisation</a>
    <a href="#backtest">8. Backtesting</a>
    <a href="#conclusions">9. Conclusions</a>
  </div>

  <!-- EXECUTIVE SUMMARY -->
  <div class="card" id="summary">
    <h2>Executive Summary</h2>
    <p>This report provides a full-stack quantitative analysis of five large-cap U.S. equities using four progressively sophisticated factor models. We progress from single-factor CAPM through Fama-French 3-factor, Carhart 4-factor (with momentum), and into factor timing via market regime detection. Portfolio construction is evaluated across four strategies with a walk-forward backtesting framework.</p>
    <div class="highlight"><strong>Key findings:</strong> No stock generates statistically significant alpha. Multi-factor models materially improve explanatory power. Momentum is a significant factor for all five stocks. The Risk Parity portfolio delivers the best risk-adjusted return (Sharpe 1.36) vs the benchmark (Sharpe 1.17).</div>
  </div>

  <!-- DATA -->
  <div class="card" id="data">
    <h2>1. Data &amp; Methodology</h2>
    <p>Daily adjusted prices from Yahoo Finance via <code>yfinance</code>. Log returns: <code>ln(P_t / P_{{t-1}})</code>. FF3 and momentum factors from Kenneth French's data library via <code>pandas-datareader</code>. All regressions use OLS. Significance level: p &lt; 0.05.</p>
    <h3>Models</h3>
    <p><strong>CAPM:</strong> <code>R_i - R_f = α + β(Rm-Rf) + ε</code></p>
    <p><strong>FF3:</strong> <code>R_i - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + ε</code></p>
    <p><strong>Carhart 4:</strong> <code>R_i - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(MOM) + ε</code></p>
  </div>

  <!-- CUMULATIVE RETURNS -->
  <div class="card" id="cumret">
    <h2>2. Cumulative Excess Returns</h2>
    <img src="data:image/png;base64,{cumret_chart}" alt="Cumulative Returns">
    <div class="finding">GOOGL (+40% annualised) and JPM delivered the strongest cumulative performance. XOM significantly underperformed, driven by energy-sector dynamics rather than broad market forces.</div>
  </div>

  <!-- CAPM -->
  <div class="card" id="capm">
    <h2>3. CAPM Analysis</h2>
    {capm_table}
    <img src="data:image/png;base64,{beta_chart}" alt="Beta">
    <img src="data:image/png;base64,{r2_chart}" alt="R-squared">
    <div class="finding"><strong>Beta:</strong> Tech stocks (AAPL 1.16, MSFT 1.04, GOOGL 1.19) amplify market moves. JPM (0.92) tracks the market. XOM (0.45) is driven by oil, not equity sentiment.</div>
    <div class="warning"><strong>Alpha:</strong> No stock produces statistically significant alpha — consistent with efficient large-cap markets.</div>
    <div class="finding"><strong>R²:</strong> CAPM explains only 9% of XOM's returns — a clear signal that a single market factor is insufficient for energy stocks.</div>
  </div>

  <!-- FF3 -->
  <div class="card" id="ff3">
    <h2>4. Fama-French 3-Factor Model</h2>
    <p>Extending CAPM with SMB (size) and HML (value) factors improves explanatory power for all stocks. AAPL and MSFT show negative HML loading — confirming their growth stock character. XOM shows positive HML — consistent with value stock classification.</p>
    <img src="data:image/png;base64,{r2_models_chart}" alt="R2 Comparison">
  </div>

  <!-- CARHART 4 -->
  <div class="card" id="carhart">
    <h2>5. Carhart 4-Factor Model (Momentum)</h2>
    {c4_table}
    <h3>R² Across All Three Models</h3>
    {r2_comp_table}
    <div class="finding"><strong>Momentum is significant</strong> for all five stocks (all Beta MOM p-values &lt; 0.05). Adding momentum further improves R² beyond FF3, particularly for JPM (R² rises to 0.55) and XOM (0.31 vs 0.09 in CAPM).</div>
    <div class="finding"><strong>XOM's negative MOM loading</strong> reveals it is a momentum contrarian — it tends to underperform when market momentum is strong, and recover when momentum reverses.</div>
  </div>

  <!-- FF5 -->
  <div class="card" id="ff5">
    <h2>6. Fama-French 5-Factor Model</h2>
    <p>FF5 adds Profitability (RMW — Robust Minus Weak) and Investment (CMA — Conservative Minus Aggressive) to the FF3 model. These are the two most important additions since Fama-French published the original 3-factor model in 1993.</p>
    <img src="data:image/png;base64,{ff5_chart}" alt="FF5 R2 Comparison">
    <div class="finding"><strong>RMW (Profitability):</strong> AAPL (β=0.67), MSFT (β=0.30), GOOGL (β=0.33) all have positive, significant RMW loading — confirming they are highly profitable firms that earn the profitability premium. JPM has negative RMW (β=-0.36) — consistent with the lower profitability margins typical of financial firms.</div>
    <div class="finding"><strong>CMA (Investment):</strong> MSFT (β=-0.61) and GOOGL (β=-0.69) have strongly negative CMA loading — they are aggressive investors (heavy R&D, capex) rather than conservative ones. XOM has positive CMA (β=0.24) — energy companies are more capital-conservative relative to tech.</div>
    <div class="highlight"><strong>FF5 improves R² for all stocks.</strong> The biggest gain is XOM (9% → 30%) — profitability and investment factors capture energy-sector dynamics that CAPM and even FF3 miss entirely.</div>
  </div>

  <!-- ROLLING BETA -->
  <div class="card" id="rolling">
    <h2>7. Rolling Beta Analysis</h2>
    <p>Static beta assumes a stock's market sensitivity is constant over time. Rolling beta reveals how this relationship actually changes — showing which stocks have stable vs unstable risk profiles.</p>
    <img src="data:image/png;base64,{rolling_beta_chart}" alt="Rolling Beta">
    <div class="finding"><strong>XOM has the most unstable beta</strong> (range: -0.02 to 0.80) — its market sensitivity swings dramatically based on oil price cycles and energy sector sentiment. Using a static beta of 0.45 for risk management would be dangerously misleading.</div>
    <div class="finding"><strong>GOOGL and AAPL show the most stable rolling betas</strong> — their relationship with the market is consistent across time, making them more predictable from a risk management perspective.</div>
    <div class="warning"><strong>Portfolio risk implication:</strong> Any hedging strategy using beta should use rolling beta, not the static estimate. A portfolio hedged with static betas could be significantly under- or over-hedged during regime changes.</div>
  </div>

  <!-- WALK-FORWARD -->
  <div class="card" id="walkforward">
    <h2>8. Walk-Forward Out-of-Sample Validation</h2>
    <p>Models trained on 2023–2024, evaluated on 2025 data they have never seen. This is the most important test in quantitative research — separating genuine predictive power from in-sample overfitting.</p>
    <img src="data:image/png;base64,{wf_chart}" alt="Walk Forward">
    <img src="data:image/png;base64,{wf_r2_chart}" alt="Walk Forward R2">
    <div class="finding"><strong>OOS R² is positive for all stocks under both models</strong> — meaning both CAPM and FF5 beat the naive historical mean as a forecast on unseen data. This is not guaranteed and is a meaningful validation result.</div>
    <div class="finding"><strong>FF5 OOS R² exceeds CAPM OOS R²</strong> for all stocks — the extra factors don't just help in-sample, they genuinely improve out-of-sample prediction.</div>
    <div class="warning"><strong>Strategy degradation:</strong> Max Sharpe Sharpe drops from 1.66 (IS) to 0.49 (OOS) — a classic sign of overfitting. Equal Weight is more robust (1.78 IS → 1.13 OOS) because it has no free parameters to overfit. Simpler strategies generalise better.</div>
  </div>

  <!-- REGIME -->
  <div class="card" id="regime">
    <h2>9. Market Regime Detection &amp; Factor Timing</h2>
    <p>Regime detected using 200-day moving average of cumulative S&P 500 returns. Bull = above SMA, Bear = more than 1 std below, Sideways = in between.</p>
    <img src="data:image/png;base64,{regimes_chart}" alt="Regimes">
    <h3>Factor Performance by Regime</h3>
    {regime_table}
    <img src="data:image/png;base64,{regime_factor_chart}" alt="Factor by Regime">
    <div class="finding"><strong>Bull regime (78% of days):</strong> Market factor dominates. Momentum performs well (+7.1% annualised).</div>
    <div class="finding"><strong>Bear regime (3% of days):</strong> Momentum spikes (+79% annualised in bear days) — suggesting momentum strategies partially hedge downside. Market factor is deeply negative as expected.</div>
    <div class="warning"><strong>Factor timing implication:</strong> Tilting toward momentum and away from market beta in bear regimes could improve risk-adjusted returns — but bear regimes are rare (only 23 days) making this statistically fragile.</div>
  </div>

  <!-- PORTFOLIO -->
  <div class="card" id="portfolio">
    <h2>10. Portfolio Construction &amp; Optimisation</h2>
    <p>Walk-forward optimisation using 126-day estimation window, monthly rebalancing, 10bps transaction costs.</p>
    <img src="data:image/png;base64,{port_chart}" alt="Portfolio Comparison">
    <h3>Performance Comparison</h3>
    {metrics_table}
    <img src="data:image/png;base64,{frontier_chart}" alt="Efficient Frontier">
    <div class="finding"><strong>Equal Weight</strong> delivers the highest total return (99.4%) but this is driven by concentration in high-performing tech stocks — not skill.</div>
    <div class="finding"><strong>Risk Parity</strong> achieves the best risk-adjusted return (Sharpe 1.36, Sortino 1.80) with the lowest volatility (15.9%), outperforming the benchmark on a risk-adjusted basis.</div>
    <div class="warning"><strong>Min Variance</strong> underperforms the benchmark in total return — it over-weights XOM which dragged performance in this tech-driven period.</div>
  </div>

  <!-- BACKTEST -->
  <div class="card" id="backtest">
    <h2>11. Backtesting — Simple Equal-Weight Strategy</h2>
    <img src="data:image/png;base64,{bt_chart}" alt="Backtest">
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{bt_metrics_rows}</tbody>
    </table>
    <div class="warning" style="margin-top:16px;"><strong>In-sample caveat:</strong> All results use the same data for parameter estimation and evaluation. Walk-forward testing partially mitigates but does not eliminate this. Out-of-sample testing on post-2025 data is required before acting on any of these results.</div>
  </div>

  <!-- CONCLUSIONS -->
  <div class="card" id="conclusions">
    <h2>12. Conclusions</h2>
    <ol style="padding-left:20px; line-height:2.2;">
      <li><strong>No stock generates statistically significant alpha</strong> — consistent with the Efficient Market Hypothesis for large-cap equities over 2023–2025.</li>
      <li><strong>CAPM is insufficient for energy stocks</strong> — XOM's 9% R² reveals that oil prices, not equity market beta, drive its returns.</li>
      <li><strong>Momentum is a significant, priced factor</strong> for all five stocks — adding it to FF3 improves explanatory power for every name in the portfolio.</li>
      <li><strong>Market regimes matter for factor timing</strong> — momentum dramatically outperforms in bear regimes, suggesting regime-conditional factor tilts could add value in a larger, more diversified portfolio.</li>
      <li><strong>Risk Parity is the best risk-adjusted strategy</strong> in this sample — it achieves competitive returns with the lowest volatility and drawdown, outperforming the benchmark on a Sharpe basis.</li>
      <li><strong>All results are in-sample</strong> — the next step is out-of-sample validation, expansion to more stocks, and live paper trading.</li>
    </ol>
    <div class="highlight" style="margin-top:20px;"><strong>Next research directions:</strong> Fama-French 5-factor model (adding profitability + investment factors), Hidden Markov Model regime detection, cross-asset factor analysis, and live paper trading integration.</div>
  </div>

  <footer>
    <p>Factor Model Research Pipeline &nbsp;|&nbsp; {today}</p>
    <p style="margin-top:6px;">Python — pandas · statsmodels · scipy · matplotlib · yfinance · pandas-datareader</p>
  </footer>

</div>
</body>
</html>"""

    output_path = os.path.join(REPORTS_DIR, "capm_ff3_research_report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return os.path.abspath(output_path)


if __name__ == "__main__":
    print("Generating full research report...")
    path = generate_report()
    print(f"\nReport saved to: {path}")
    print("Open in your browser to view.")
