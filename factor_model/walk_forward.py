"""
Walk-Forward Out-of-Sample Testing
=====================================

Splits historical data into training and test periods to evaluate
whether factor model results and strategy performance hold on
data the model has never seen.

This is the most important validation step in quantitative research.
In-sample results are expected to look good — any model can overfit
to historical data. Out-of-sample performance is what matters.

Methodology:
- Train period : 2023-01-01 → 2024-12-31
- Test period  : 2025-01-01 → 2025-12-31
- Models fitted on train, evaluated on test
- Portfolio strategies run walk-forward to avoid look-ahead bias

Example:
--------
    >>> from factor_model.walk_forward import WalkForwardTest
    >>> wf = WalkForwardTest(data, train_end="2024-12-31", test_start="2025-01-01")
    >>> results = wf.run_all()
    >>> wf.print_summary(results)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


TRADING_DAYS: int = 252


@dataclass
class SplitConfig:
    """Train/test split configuration."""
    train_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    transaction_cost: float = 0.001


class WalkForwardTest:
    """
    Walk-forward out-of-sample backtesting framework.

    Fits all models on the training period and evaluates them
    strictly on the test period — no information from the future
    is used at any point during test evaluation.

    Parameters
    ----------
    data : pd.DataFrame
        Full merged excess returns with stock columns + "Market".
    factors : pd.DataFrame
        Factor data (Mkt_RF, SMB, HML, RMW, CMA, MOM) aligned to same dates.
    config : SplitConfig
        Train/test split parameters.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        factors: pd.DataFrame,
        config: SplitConfig = None
    ) -> None:
        self.config = config or SplitConfig()

        self.market_full = data["Market"]
        self.stocks_full = data.drop(columns=["Market"])
        self.factors_full = factors

        # Split
        self.train_stocks = self.stocks_full[:self.config.train_end]
        self.test_stocks = self.stocks_full[self.config.test_start:]
        self.train_market = self.market_full[:self.config.train_end]
        self.test_market = self.market_full[self.config.test_start:]
        self.train_factors = self.factors_full[:self.config.train_end]
        self.test_factors = self.factors_full[self.config.test_start:]

        print(f"Train period: {self.train_stocks.index[0].date()} → {self.train_stocks.index[-1].date()} ({len(self.train_stocks)} days)")
        print(f"Test period:  {self.test_stocks.index[0].date()} → {self.test_stocks.index[-1].date()} ({len(self.test_stocks)} days)")

    # ------------------------------------------------------------------ #
    #  Model Fitting & Prediction                                          #
    # ------------------------------------------------------------------ #

    def _fit_capm(self, stock: str) -> sm.regression.linear_model.RegressionResultsWrapper:
        X = sm.add_constant(self.train_market)
        return sm.OLS(self.train_stocks[stock], X).fit()

    def _fit_ff5(self, stock: str, factor_cols: list[str]) -> sm.regression.linear_model.RegressionResultsWrapper:
        X = sm.add_constant(self.train_factors[factor_cols])
        return sm.OLS(self.train_stocks[stock], X).fit()

    def _oos_r2(
        self,
        model: sm.regression.linear_model.RegressionResultsWrapper,
        test_y: pd.Series,
        test_X: pd.DataFrame
    ) -> float:
        """
        Out-of-sample R² (also called R²_oos or Campbell-Thompson R²).

        OOS R² = 1 - SS_res_oos / SS_tot_oos
        where SS_tot uses the historical mean return as benchmark.
        A positive OOS R² means the model beats the historical mean.
        """
        X_with_const = sm.add_constant(test_X, has_constant="add")
        predictions = model.predict(X_with_const)
        residuals = test_y - predictions
        ss_res = (residuals ** 2).sum()
        # Benchmark: use training mean as forecast (no model)
        train_mean = model.model.endog.mean()
        ss_tot = ((test_y - train_mean) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def run_model_comparison(self) -> pd.DataFrame:
        """
        Compare in-sample vs out-of-sample R² for CAPM and FF5.

        Returns
        -------
        pd.DataFrame
            Per-stock IS and OOS R² for both models.
        """
        ff5_cols = [c for c in ["Mkt_RF", "SMB", "HML", "RMW", "CMA"] if c in self.train_factors.columns]
        capm_test_X = self.test_market.to_frame()

        rows = []
        for stock in self.stocks_full.columns:
            # CAPM
            capm_model = self._fit_capm(stock)
            capm_is_r2 = capm_model.rsquared
            capm_oos_r2 = self._oos_r2(capm_model, self.test_stocks[stock], capm_test_X)

            # FF5
            ff5_model = self._fit_ff5(stock, ff5_cols)
            ff5_is_r2 = ff5_model.rsquared
            ff5_oos_r2 = self._oos_r2(ff5_model, self.test_stocks[stock], self.test_factors[ff5_cols])

            rows.append({
                "Stock": stock,
                "CAPM_IS_R2": capm_is_r2,
                "CAPM_OOS_R2": capm_oos_r2,
                "FF5_IS_R2": ff5_is_r2,
                "FF5_OOS_R2": ff5_oos_r2,
                "CAPM_Degradation": capm_is_r2 - capm_oos_r2,
                "FF5_Degradation": ff5_is_r2 - ff5_oos_r2,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Strategy Walk-Forward                                               #
    # ------------------------------------------------------------------ #

    def _strategy_returns(
        self,
        stocks: pd.DataFrame,
        weight_fn,
        estimation_window: int = 126
    ) -> pd.Series:
        """Run a strategy walk-forward on the given stock returns."""
        rebalance_dates = set(stocks.resample("M").last().index)
        dates = stocks.index.tolist()
        returns = pd.Series(index=stocks.index, dtype=float)
        current_weights = np.ones(stocks.shape[1]) / stocks.shape[1]
        prev_weights = current_weights.copy()

        for i, date in enumerate(dates):
            prev_was_rebal = i > 0 and dates[i - 1] in rebalance_dates
            if i == 0 or prev_was_rebal:
                start_idx = max(0, i - estimation_window)
                window = stocks.iloc[start_idx:i]
                if len(window) >= 20:
                    try:
                        current_weights = weight_fn(window)
                    except Exception:
                        current_weights = np.ones(stocks.shape[1]) / stocks.shape[1]
                turnover = np.sum(np.abs(current_weights - prev_weights))
                tc = turnover * self.config.transaction_cost
                prev_weights = current_weights.copy()
            else:
                tc = 0.0

            returns[date] = (stocks.loc[date].values * current_weights).sum() - tc

        return returns

    def _equal_weight(self, window: pd.DataFrame) -> np.ndarray:
        return np.ones(window.shape[1]) / window.shape[1]

    def _min_variance(self, window: pd.DataFrame) -> np.ndarray:
        from scipy.optimize import minimize
        cov = window.cov().values
        n = cov.shape[0]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 1.0)] * n
        result = minimize(lambda w: w @ cov @ w, np.ones(n) / n,
                          method="SLSQP", bounds=bounds, constraints=constraints)
        return result.x if result.success else np.ones(n) / n

    def _max_sharpe(self, window: pd.DataFrame) -> np.ndarray:
        from scipy.optimize import minimize
        mu = window.mean().values * TRADING_DAYS
        cov = window.cov().values * TRADING_DAYS
        n = len(mu)
        def neg_sharpe(w):
            r = w @ mu
            v = np.sqrt(w @ cov @ w)
            return -(r - 0.02) / v if v > 0 else 0.0
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 1.0)] * n
        result = minimize(neg_sharpe, np.ones(n) / n,
                          method="SLSQP", bounds=bounds, constraints=constraints)
        return result.x if result.success else np.ones(n) / n

    def run_strategy_oos(self) -> Dict[str, Dict[str, pd.Series]]:
        """
        Run all portfolio strategies on train and test periods separately.

        Returns
        -------
        dict
            {strategy_name: {"train": returns_series, "test": returns_series}}
        """
        strategies = {
            "Equal Weight": self._equal_weight,
            "Min Variance": self._min_variance,
            "Max Sharpe": self._max_sharpe,
        }

        results = {}
        for name, fn in strategies.items():
            print(f"  Running {name}...")
            train_ret = self._strategy_returns(self.train_stocks, fn)
            test_ret = self._strategy_returns(self.test_stocks, fn)
            results[name] = {"train": train_ret, "test": test_ret}

        return results

    def _compute_metrics(self, returns: pd.Series) -> dict:
        n = len(returns)
        if n == 0:
            return {}
        total = (1 + returns).prod() - 1
        cagr = (1 + total) ** (TRADING_DAYS / n) - 1
        vol = returns.std() * np.sqrt(TRADING_DAYS)
        sharpe = (returns.mean() * TRADING_DAYS) / vol if vol > 0 else 0.0
        cum = (1 + returns).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        return {
            "Total Return": f"{total:.2%}",
            "CAGR": f"{cagr:.2%}",
            "Sharpe": f"{sharpe:.3f}",
            "Max Drawdown": f"{max_dd:.2%}",
        }

    def print_summary(self, strategy_results: Dict) -> None:
        """Print IS vs OOS performance for all strategies."""
        print("\n" + "=" * 70)
        print("  WALK-FORWARD: IN-SAMPLE vs OUT-OF-SAMPLE PERFORMANCE")
        print("=" * 70)
        header = f"{'Strategy':<20} {'IS Return':>10} {'IS Sharpe':>10} {'OOS Return':>12} {'OOS Sharpe':>12}"
        print(header)
        print("-" * 70)
        for name, splits in strategy_results.items():
            is_m = self._compute_metrics(splits["train"])
            oos_m = self._compute_metrics(splits["test"])
            print(f"{name:<20} {is_m.get('Total Return','—'):>10} {is_m.get('Sharpe','—'):>10} "
                  f"{oos_m.get('Total Return','—'):>12} {oos_m.get('Sharpe','—'):>12}")
        print("=" * 70)

    # ------------------------------------------------------------------ #
    #  Visualisations                                                      #
    # ------------------------------------------------------------------ #

    def plot_is_vs_oos(
        self,
        strategy_results: Dict,
        save_path: str = None
    ) -> None:
        """
        Plot cumulative performance for each strategy split by IS/OOS period.

        A vertical dashed line marks the train/test boundary.
        """
        colors = {
            "Equal Weight": "#3498db",
            "Min Variance": "#2ecc71",
            "Max Sharpe": "#e67e22",
        }

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for ax, period, title in zip(axes, ["train", "test"], ["In-Sample (2023–2024)", "Out-of-Sample (2025)"]):
            for name, splits in strategy_results.items():
                returns = splits[period]
                cum = (1 + returns).cumprod() * 100
                ax.plot(cum.index, cum, label=name, color=colors.get(name, "gray"), linewidth=2)

            bm = self.train_market if period == "train" else self.test_market
            bm_cum = (1 + bm).cumprod() * 100
            ax.plot(bm_cum.index, bm_cum, label="S&P 500", color="#7f8c8d",
                    linewidth=1.5, linestyle="--")

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylabel("Portfolio Value (Base = 100)", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Walk-Forward Validation: In-Sample vs Out-of-Sample",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_r2_is_vs_oos(
        self,
        model_comparison: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """Bar chart comparing IS vs OOS R² for CAPM and FF5."""
        import numpy as np

        stocks = model_comparison["Stock"].tolist()
        x = np.arange(len(stocks))
        width = 0.2

        fig, ax = plt.subplots(figsize=(13, 6))

        ax.bar(x - 1.5 * width, model_comparison["CAPM_IS_R2"], width,
               label="CAPM In-Sample", color="#2980b9", edgecolor="black")
        ax.bar(x - 0.5 * width, model_comparison["CAPM_OOS_R2"], width,
               label="CAPM Out-of-Sample", color="#aed6f1", edgecolor="black")
        ax.bar(x + 0.5 * width, model_comparison["FF5_IS_R2"], width,
               label="FF5 In-Sample", color="#e67e22", edgecolor="black")
        ax.bar(x + 1.5 * width, model_comparison["FF5_OOS_R2"], width,
               label="FF5 Out-of-Sample", color="#f9c784", edgecolor="black")

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(stocks, fontsize=12)
        ax.set_xlabel("Stock", fontsize=12)
        ax.set_ylabel("R²", fontsize=12)
        ax.set_title("In-Sample vs Out-of-Sample R²: CAPM and FF5",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def main() -> None:
    """Run walk-forward validation pipeline."""
    from factor_model.ff5_regression import get_ff5_factors, get_stock_excess_ff5
    from factor_model.data_collector import get_stock_returns
    from factor_model.momentum import get_momentum_factor

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    symbols = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    start, end = "2023-01-01", "2026-01-01"

    data = pd.read_csv(
        os.path.join(data_dir, "merged_excess_returns.csv"),
        index_col=0, parse_dates=True
    )

    print("Downloading FF5 factors...")
    ff5 = get_ff5_factors(start, end)
    stock_returns = get_stock_returns(symbols, start, end)
    _, factors_df = get_stock_excess_ff5(stock_returns, ff5)

    # Add momentum to factors
    mom = get_momentum_factor(start, end)
    factors_df = factors_df.join(mom, how="left").fillna(0)

    config = SplitConfig(train_end="2024-12-31", test_start="2025-01-01")
    wf = WalkForwardTest(data, factors_df, config=config)

    print("\nRunning model R² comparison (IS vs OOS)...")
    model_comp = wf.run_model_comparison()
    print("\n" + model_comp.to_string(index=False))

    print("\nRunning strategy walk-forward...")
    strategy_results = wf.run_strategy_oos()
    wf.print_summary(strategy_results)

    model_comp.to_csv(os.path.join(data_dir, "walk_forward_r2.csv"), index=False)

    wf.plot_is_vs_oos(strategy_results, save_path=os.path.join(plots_dir, "walk_forward_strategies.png"))
    wf.plot_r2_is_vs_oos(model_comp, save_path=os.path.join(plots_dir, "walk_forward_r2.png"))

    print(f"\nResults saved to: {data_dir}")


if __name__ == "__main__":
    main()
