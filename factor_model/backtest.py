"""
Backtesting Engine
==================

Simulates a factor-based long-only equity portfolio using historical
excess returns. Includes realistic transaction costs and full suite
of performance metrics used in professional quant research.

Strategy:
- Rank stocks by market beta from FF3 regression
- Go long the top N stocks by chosen factor loading
- Rebalance monthly
- Deduct transaction costs on turnover

Performance Metrics:
- Total return, CAGR, Annualized volatility
- Sharpe ratio, Sortino ratio
- Maximum drawdown, Calmar ratio
- Win rate

Example:
--------
    >>> from factor_model.backtest import Backtest
    >>> bt = Backtest(returns_df, rebalance_freq="M", top_n=3, transaction_cost=0.001)
    >>> results = bt.run()
    >>> bt.plot_performance(results)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRADING_DAYS: int = 252


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    rebalance_freq: str = "M"         # pandas offset: "M"=month-end, "Q"=quarter-end
    top_n: int = 3                    # number of stocks to hold
    transaction_cost: float = 0.001   # 10bps per trade (one-way)
    initial_capital: float = 100.0    # starting portfolio value (index = 100)
    rank_by: str = "equal"            # "equal" = equal weight, "beta" = rank by beta


@dataclass
class BacktestResults:
    """Container for all backtest output metrics."""
    portfolio_returns: pd.Series
    portfolio_value: pd.Series
    benchmark_value: pd.Series
    turnover: pd.Series
    metrics: dict = field(default_factory=dict)


class Backtest:
    """
    Factor-based long-only backtesting engine.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily excess returns with one column per stock plus "Market" column.
    config : BacktestConfig, optional
        Backtesting parameters.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        config: Optional[BacktestConfig] = None
    ) -> None:
        self.returns_df = returns_df.copy()
        self.config = config or BacktestConfig()

        if "Market" not in self.returns_df.columns:
            raise ValueError("returns_df must contain a 'Market' column as benchmark.")

        self.market = self.returns_df["Market"]
        self.stocks = self.returns_df.drop(columns=["Market"])

    def _get_rebalance_dates(self) -> pd.DatetimeIndex:
        """Return month-end (or configured frequency) dates within the data range."""
        return self.returns_df.resample(self.config.rebalance_freq).last().index

    def _select_portfolio(self, available_stocks: pd.Index) -> list[str]:
        """Select top_n stocks (equal weight for now — extensible to factor ranking)."""
        n = min(self.config.top_n, len(available_stocks))
        return list(available_stocks[:n])

    def run(self) -> BacktestResults:
        """
        Execute the backtest.

        Returns
        -------
        BacktestResults
            Portfolio returns, value series, benchmark, turnover, and metrics.
        """
        month_end_dates = set(self._get_rebalance_dates())
        dates = self.stocks.index.tolist()

        portfolio_returns = pd.Series(index=self.stocks.index, dtype=float)
        current_holdings: list[str] = []
        previous_weights: dict[str, float] = {}
        rebalance_log: list[tuple] = []

        for i, date in enumerate(dates):
            # Rebalance on day 0, or the first trading day after each month-end
            prev_was_month_end = i > 0 and dates[i - 1] in month_end_dates
            should_rebalance = (i == 0) or prev_was_month_end

            if should_rebalance:
                new_holdings = self._select_portfolio(self.stocks.columns)
                n = len(new_holdings)
                new_weights = {s: 1.0 / n for s in new_holdings}

                all_tickers = set(list(previous_weights.keys()) + new_holdings)
                turnover = sum(
                    abs(new_weights.get(t, 0.0) - previous_weights.get(t, 0.0))
                    for t in all_tickers
                )
                rebalance_log.append((date, turnover))

                current_holdings = new_holdings
                previous_weights = new_weights.copy()

            daily_ret = self.stocks.loc[date, current_holdings].mean() if current_holdings else 0.0
            portfolio_returns[date] = daily_ret

        # Deduct transaction costs on rebalance days
        for rebal_date, turnover in rebalance_log:
            portfolio_returns[rebal_date] -= turnover * self.config.transaction_cost

        turnover_series = pd.Series(
            {d: t for d, t in rebalance_log},
            name="Turnover"
        )

        portfolio_value = (1 + portfolio_returns).cumprod() * self.config.initial_capital
        benchmark_value = (1 + self.market).cumprod() * self.config.initial_capital

        metrics = self._compute_metrics(portfolio_returns, portfolio_value, benchmark_value)

        return BacktestResults(
            portfolio_returns=portfolio_returns,
            portfolio_value=portfolio_value,
            benchmark_value=benchmark_value,
            turnover=turnover_series,
            metrics=metrics,
        )

    def _compute_metrics(
        self,
        returns: pd.Series,
        portfolio_value: pd.Series,
        benchmark_value: pd.Series
    ) -> dict:
        """Compute full suite of performance metrics."""
        n_days = len(returns)

        total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
        cagr = (1 + total_return) ** (TRADING_DAYS / n_days) - 1

        ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
        sharpe = (returns.mean() * TRADING_DAYS) / ann_vol if ann_vol > 0 else 0.0

        downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
        sortino = (returns.mean() * TRADING_DAYS) / downside if downside > 0 else 0.0

        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

        win_rate = (returns > 0).sum() / len(returns)

        bm_total = benchmark_value.iloc[-1] / benchmark_value.iloc[0] - 1
        excess_return = total_return - bm_total

        return {
            "Total Return": f"{total_return:.2%}",
            "CAGR": f"{cagr:.2%}",
            "Annualized Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.3f}",
            "Sortino Ratio": f"{sortino:.3f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Calmar Ratio": f"{calmar:.3f}",
            "Win Rate (daily)": f"{win_rate:.2%}",
            "Benchmark Total Return": f"{bm_total:.2%}",
            "Excess Return vs Benchmark": f"{excess_return:.2%}",
        }

    def plot_performance(
        self,
        results: BacktestResults,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot portfolio vs benchmark cumulative performance and drawdown.

        Parameters
        ----------
        results : BacktestResults
            Output from run().
        save_path : str, optional
            File path to save figure.
        """
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [3, 1]})

        # --- Cumulative performance ---
        ax1 = axes[0]
        ax1.plot(results.portfolio_value.index, results.portfolio_value,
                 label="Portfolio", color="#2980b9", linewidth=2)
        ax1.plot(results.benchmark_value.index, results.benchmark_value,
                 label="S&P 500 (Benchmark)", color="#7f8c8d", linewidth=1.5, linestyle="--")
        ax1.set_ylabel("Portfolio Value (Base = 100)", fontsize=11)
        ax1.set_title("Portfolio vs Benchmark — Cumulative Performance", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # --- Drawdown ---
        ax2 = axes[1]
        rolling_max = results.portfolio_value.cummax()
        drawdown = (results.portfolio_value - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, drawdown, 0, color="#e74c3c", alpha=0.5)
        ax2.plot(drawdown.index, drawdown, color="#e74c3c", linewidth=1)
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def print_metrics(self, results: BacktestResults) -> None:
        """Print formatted performance metrics table."""
        print("\n" + "=" * 45)
        print("  PORTFOLIO PERFORMANCE METRICS")
        print("=" * 45)
        for key, val in results.metrics.items():
            print(f"  {key:<35} {val}")
        print("=" * 45)


def main() -> None:
    """Run backtest on existing merged excess returns data."""
    import os

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "merged_excess_returns.csv")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    config = BacktestConfig(
        rebalance_freq="M",
        top_n=3,
        transaction_cost=0.001,
        initial_capital=100.0,
    )

    bt = Backtest(data, config=config)
    results = bt.run()

    bt.print_metrics(results)

    bt.plot_performance(
        results,
        save_path=os.path.join(plots_dir, "backtest_performance.png")
    )

    print(f"\nBacktest plot saved to: {plots_dir}/backtest_performance.png")


if __name__ == "__main__":
    main()
