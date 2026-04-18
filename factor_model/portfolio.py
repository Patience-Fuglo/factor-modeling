"""
Portfolio Construction & Optimization
========================================

Implements four professional portfolio construction methods and
compares their backtested performance:

1. Equal Weight        — baseline, simplest
2. Minimum Variance    — minimise portfolio volatility
3. Maximum Sharpe      — maximise risk-adjusted return
4. Risk Parity         — equalise risk contribution from each asset

All methods use rolling in-sample estimation windows to avoid
look-ahead bias (walk-forward optimisation).

Example:
--------
    >>> from factor_model.portfolio import PortfolioOptimizer
    >>> opt = PortfolioOptimizer(returns_df)
    >>> results = opt.compare_all_strategies()
    >>> opt.plot_comparison(results)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


TRADING_DAYS: int = 252


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimisation."""
    estimation_window: int = 126     # days of history used to estimate parameters
    rebalance_freq: str = "M"        # rebalance frequency
    transaction_cost: float = 0.001  # 10bps per trade
    risk_free_rate: float = 0.02     # annual, for Sharpe calculation
    min_weight: float = 0.0          # minimum weight per stock (0 = long only)
    max_weight: float = 1.0          # maximum weight per stock


class PortfolioOptimizer:
    """
    Walk-forward portfolio optimiser comparing four construction methods.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily excess returns. Must include a "Market" column as benchmark.
    config : OptimizationConfig, optional
        Optimisation parameters.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        config: Optional[OptimizationConfig] = None
    ) -> None:
        self.config = config or OptimizationConfig()

        if "Market" not in returns_df.columns:
            raise ValueError("returns_df must contain a 'Market' column.")

        self.market = returns_df["Market"].copy()
        self.stocks = returns_df.drop(columns=["Market"]).copy()
        self.n_stocks = len(self.stocks.columns)

    # ------------------------------------------------------------------ #
    #  Weight Calculation Methods                                          #
    # ------------------------------------------------------------------ #

    def _equal_weight(self, returns_window: pd.DataFrame) -> np.ndarray:
        """Equal weight: 1/N for each stock."""
        n = returns_window.shape[1]
        return np.ones(n) / n

    def _min_variance(self, returns_window: pd.DataFrame) -> np.ndarray:
        """Minimum variance: minimise portfolio variance."""
        cov = returns_window.cov().values
        n = cov.shape[0]

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(
            fun=lambda w: w @ cov @ w,
            x0=w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000}
        )
        return result.x if result.success else w0

    def _max_sharpe(self, returns_window: pd.DataFrame) -> np.ndarray:
        """Maximum Sharpe ratio portfolio."""
        mu = returns_window.mean().values * TRADING_DAYS
        cov = returns_window.cov().values * TRADING_DAYS
        rf = self.config.risk_free_rate
        n = len(mu)

        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            return -(port_ret - rf) / port_vol if port_vol > 0 else 0.0

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(
            fun=neg_sharpe,
            x0=w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000}
        )
        return result.x if result.success else w0

    def _risk_parity(self, returns_window: pd.DataFrame) -> np.ndarray:
        """
        Risk parity: each stock contributes equally to total portfolio variance.

        Minimises sum of squared differences between actual and target
        risk contributions (equal risk contribution).
        """
        cov = returns_window.cov().values
        n = cov.shape[0]
        target_risk = np.ones(n) / n

        def risk_parity_objective(w):
            port_var = w @ cov @ w
            marginal_risk = cov @ w
            risk_contribution = w * marginal_risk / port_var if port_var > 0 else w
            return np.sum((risk_contribution - target_risk) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.01, self.config.max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(
            fun=risk_parity_objective,
            x0=w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000}
        )
        return result.x if result.success else w0

    # ------------------------------------------------------------------ #
    #  Walk-Forward Backtest                                               #
    # ------------------------------------------------------------------ #

    def _run_strategy(self, weight_fn) -> pd.Series:
        """
        Walk-forward backtest for a given weight function.

        For each rebalance date:
        1. Use the past `estimation_window` days to compute weights
        2. Apply weights to next period's returns
        3. Deduct transaction costs on turnover
        """
        rebalance_dates = self.stocks.resample(self.config.rebalance_freq).last().index
        portfolio_returns = pd.Series(index=self.stocks.index, dtype=float)

        current_weights = np.ones(self.n_stocks) / self.n_stocks
        prev_weights = current_weights.copy()
        dates = self.stocks.index.tolist()

        for i, date in enumerate(dates):
            prev_was_rebal = i > 0 and dates[i - 1] in set(rebalance_dates)

            if i == 0 or prev_was_rebal:
                # Estimate weights using past window
                start_idx = max(0, i - self.config.estimation_window)
                window = self.stocks.iloc[start_idx:i]

                if len(window) >= 20:
                    try:
                        current_weights = weight_fn(window)
                    except Exception:
                        current_weights = np.ones(self.n_stocks) / self.n_stocks

                # Transaction cost on turnover
                turnover = np.sum(np.abs(current_weights - prev_weights))
                tc = turnover * self.config.transaction_cost
                prev_weights = current_weights.copy()
            else:
                tc = 0.0

            daily_ret = (self.stocks.loc[date].values * current_weights).sum()
            portfolio_returns[date] = daily_ret - tc

        return portfolio_returns

    def run_equal_weight(self) -> pd.Series:
        return self._run_strategy(self._equal_weight)

    def run_min_variance(self) -> pd.Series:
        return self._run_strategy(self._min_variance)

    def run_max_sharpe(self) -> pd.Series:
        return self._run_strategy(self._max_sharpe)

    def run_risk_parity(self) -> pd.Series:
        return self._run_strategy(self._risk_parity)

    # ------------------------------------------------------------------ #
    #  Performance Metrics & Comparison                                    #
    # ------------------------------------------------------------------ #

    def compute_metrics(self, returns: pd.Series, name: str) -> dict:
        """Compute full performance metrics for a return series."""
        n = len(returns)
        total_ret = (1 + returns).prod() - 1
        cagr = (1 + total_ret) ** (TRADING_DAYS / n) - 1
        ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
        sharpe = (returns.mean() * TRADING_DAYS) / ann_vol if ann_vol > 0 else 0.0

        downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
        sortino = (returns.mean() * TRADING_DAYS) / downside if downside > 0 else 0.0

        cum = (1 + returns).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        bm_total = (1 + self.market).prod() - 1

        return {
            "Strategy": name,
            "Total Return": f"{total_ret:.2%}",
            "CAGR": f"{cagr:.2%}",
            "Ann. Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.3f}",
            "Sortino Ratio": f"{sortino:.3f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Calmar Ratio": f"{calmar:.3f}",
            "Excess vs Market": f"{total_ret - bm_total:.2%}",
        }

    def compare_all_strategies(self) -> dict[str, pd.Series]:
        """
        Run all four strategies and return returns series.

        Returns
        -------
        dict
            Keys: strategy names, values: daily return Series.
        """
        print("Running Equal Weight...")
        ew = self.run_equal_weight()

        print("Running Minimum Variance...")
        mv = self.run_min_variance()

        print("Running Maximum Sharpe...")
        ms = self.run_max_sharpe()

        print("Running Risk Parity...")
        rp = self.run_risk_parity()

        return {
            "Equal Weight": ew,
            "Min Variance": mv,
            "Max Sharpe": ms,
            "Risk Parity": rp,
        }

    def print_comparison_table(self, strategies: dict[str, pd.Series]) -> pd.DataFrame:
        """Print and return formatted performance comparison table."""
        rows = []
        for name, returns in strategies.items():
            rows.append(self.compute_metrics(returns, name))

        bm_total = (1 + self.market).prod() - 1
        n = len(self.market)
        bm_cagr = (1 + bm_total) ** (TRADING_DAYS / n) - 1
        bm_vol = self.market.std() * np.sqrt(TRADING_DAYS)
        bm_sharpe = (self.market.mean() * TRADING_DAYS) / bm_vol
        bm_cum = (1 + self.market).cumprod()
        bm_dd = ((bm_cum - bm_cum.cummax()) / bm_cum.cummax()).min()
        rows.append({
            "Strategy": "S&P 500 Benchmark",
            "Total Return": f"{bm_total:.2%}",
            "CAGR": f"{bm_cagr:.2%}",
            "Ann. Volatility": f"{bm_vol:.2%}",
            "Sharpe Ratio": f"{bm_sharpe:.3f}",
            "Sortino Ratio": "—",
            "Max Drawdown": f"{bm_dd:.2%}",
            "Calmar Ratio": "—",
            "Excess vs Market": "0.00%",
        })

        df = pd.DataFrame(rows)
        print("\n" + "=" * 90)
        print(df.to_string(index=False))
        print("=" * 90)
        return df

    # ------------------------------------------------------------------ #
    #  Visualisations                                                      #
    # ------------------------------------------------------------------ #

    def plot_comparison(
        self,
        strategies: dict[str, pd.Series],
        save_path: str = None
    ) -> None:
        """Plot cumulative performance of all strategies vs benchmark."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

        colors = {
            "Equal Weight": "#3498db",
            "Min Variance": "#2ecc71",
            "Max Sharpe": "#e67e22",
            "Risk Parity": "#9b59b6",
        }

        ax1 = axes[0]
        for name, returns in strategies.items():
            cum = (1 + returns).cumprod() * 100
            ax1.plot(cum.index, cum, label=name, color=colors.get(name, "gray"), linewidth=2)

        bm_cum = (1 + self.market).cumprod() * 100
        ax1.plot(bm_cum.index, bm_cum, label="S&P 500", color="#7f8c8d",
                 linewidth=1.5, linestyle="--")

        ax1.set_ylabel("Portfolio Value (Base = 100)", fontsize=11)
        ax1.set_title("Portfolio Construction Comparison\nEqual Weight vs Min Variance vs Max Sharpe vs Risk Parity",
                      fontsize=13, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Drawdown for Max Sharpe (best strategy)
        ax2 = axes[1]
        best_returns = strategies["Max Sharpe"]
        cum_best = (1 + best_returns).cumprod()
        drawdown = (cum_best - cum_best.cummax()) / cum_best.cummax() * 100
        ax2.fill_between(drawdown.index, drawdown, 0, color="#e74c3c", alpha=0.5)
        ax2.plot(drawdown.index, drawdown, color="#e74c3c", linewidth=1)
        ax2.set_ylabel("Max Sharpe Drawdown (%)", fontsize=10)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_efficient_frontier(
        self,
        n_portfolios: int = 3000,
        save_path: str = None
    ) -> None:
        """
        Plot Monte Carlo efficient frontier with key portfolios marked.

        Simulates random portfolios to visualise the risk-return tradeoff,
        and marks the Equal Weight, Min Variance, and Max Sharpe portfolios.
        """
        mu = self.stocks.mean().values * TRADING_DAYS
        cov = self.stocks.cov().values * TRADING_DAYS
        n = self.n_stocks

        np.random.seed(42)
        rand_weights = np.random.dirichlet(np.ones(n), size=n_portfolios)
        port_returns = rand_weights @ mu
        port_vols = np.sqrt(np.einsum("ij,jk,ik->i", rand_weights, cov, rand_weights))
        port_sharpes = (port_returns - self.config.risk_free_rate) / port_vols

        fig, ax = plt.subplots(figsize=(11, 7))

        sc = ax.scatter(port_vols, port_returns, c=port_sharpes, cmap="viridis",
                        alpha=0.4, s=8)
        plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

        # Mark key portfolios using full data
        for label, weight_fn, color, marker in [
            ("Equal Weight", self._equal_weight, "blue", "o"),
            ("Min Variance", self._min_variance, "green", "^"),
            ("Max Sharpe", self._max_sharpe, "red", "*"),
            ("Risk Parity", self._risk_parity, "purple", "D"),
        ]:
            try:
                w = weight_fn(self.stocks)
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

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def main() -> None:
    """Run full portfolio optimisation comparison."""
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    data = pd.read_csv(
        os.path.join(data_dir, "merged_excess_returns.csv"),
        index_col=0, parse_dates=True
    )

    config = OptimizationConfig(
        estimation_window=126,
        rebalance_freq="M",
        transaction_cost=0.001,
    )

    opt = PortfolioOptimizer(data, config=config)

    strategies = opt.compare_all_strategies()
    metrics_df = opt.print_comparison_table(strategies)

    metrics_df.to_csv(os.path.join(data_dir, "portfolio_comparison.csv"), index=False)

    opt.plot_comparison(strategies, save_path=os.path.join(plots_dir, "portfolio_comparison.png"))
    opt.plot_efficient_frontier(save_path=os.path.join(plots_dir, "efficient_frontier.png"))

    print(f"\nResults saved to: {data_dir}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
