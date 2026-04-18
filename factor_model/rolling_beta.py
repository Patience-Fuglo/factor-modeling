"""
Rolling Beta Analysis
======================

Beta is not a fixed number — it changes over time as a stock's
relationship with the market evolves through different economic cycles,
earnings regimes, and structural changes.

Rolling beta reveals:
- How market sensitivity shifts during bull vs bear markets
- Whether a stock is becoming more or less aggressive over time
- Structural breaks (e.g. AAPL post-iPhone maturity, XOM during oil crises)

This is important for portfolio risk management — using a static beta
to hedge or size positions can be dangerously wrong if the underlying
relationship has changed.

Example:
--------
    >>> from factor_model.rolling_beta import RollingBetaAnalyzer
    >>> rba = RollingBetaAnalyzer(data, window=60)
    >>> rolling_betas = rba.compute_all()
    >>> rba.plot_rolling_betas(rolling_betas)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


TRADING_DAYS: int = 252


class RollingBetaAnalyzer:
    """
    Computes and visualises time-varying rolling beta for each stock.

    Parameters
    ----------
    data : pd.DataFrame
        Daily excess returns with stock columns + "Market".
    window : int
        Rolling window in trading days (default 60 = ~3 months).
    """

    def __init__(self, data: pd.DataFrame, window: int = 60) -> None:
        if "Market" not in data.columns:
            raise ValueError("data must contain a 'Market' column.")

        self.market = data["Market"]
        self.stocks = data.drop(columns=["Market"])
        self.window = window

    def _rolling_beta(self, stock_returns: pd.Series) -> pd.Series:
        """
        Compute rolling OLS beta for a single stock.

        Uses a vectorised rolling covariance / variance approach
        which is significantly faster than rolling OLS fitting.

        β_t = Cov(R_i, R_m) / Var(R_m)  over trailing `window` days
        """
        rolling_cov = stock_returns.rolling(self.window).cov(self.market)
        rolling_var = self.market.rolling(self.window).var()
        return (rolling_cov / rolling_var).dropna()

    def _rolling_alpha(self, stock_returns: pd.Series, rolling_beta: pd.Series) -> pd.Series:
        """
        Compute rolling alpha given rolling beta.

        α_t = mean(R_i) - β_t × mean(R_m)  over trailing window
        """
        rolling_stock_mean = stock_returns.rolling(self.window).mean()
        rolling_market_mean = self.market.rolling(self.window).mean()
        return rolling_stock_mean - rolling_beta * rolling_market_mean

    def _rolling_r2(self, stock_returns: pd.Series) -> pd.Series:
        """Compute rolling R² (squared correlation)."""
        return stock_returns.rolling(self.window).corr(self.market) ** 2

    def compute_all(self) -> pd.DataFrame:
        """
        Compute rolling beta for all stocks.

        Returns
        -------
        pd.DataFrame
            Rolling beta per stock, one column per stock.
        """
        betas = {}
        for stock in self.stocks.columns:
            betas[stock] = self._rolling_beta(self.stocks[stock])
        return pd.DataFrame(betas).dropna(how="all")

    def compute_full_analysis(self) -> dict[str, pd.DataFrame]:
        """
        Compute rolling beta, alpha, and R² for all stocks.

        Returns
        -------
        dict
            Keys: "beta", "alpha", "r2" — each a DataFrame of rolling values.
        """
        betas, alphas, r2s = {}, {}, {}

        for stock in self.stocks.columns:
            b = self._rolling_beta(self.stocks[stock])
            a = self._rolling_alpha(self.stocks[stock], b)
            r = self._rolling_r2(self.stocks[stock])
            betas[stock] = b
            alphas[stock] = a * TRADING_DAYS  # annualise alpha
            r2s[stock] = r

        return {
            "beta": pd.DataFrame(betas).dropna(how="all"),
            "alpha": pd.DataFrame(alphas).dropna(how="all"),
            "r2": pd.DataFrame(r2s).dropna(how="all"),
        }

    def beta_regime_stats(self, rolling_betas: pd.DataFrame) -> pd.DataFrame:
        """
        Summarise beta statistics across the full period and sub-periods.

        Returns
        -------
        pd.DataFrame
            Mean, std, min, max rolling beta per stock.
        """
        stats = pd.DataFrame({
            "Static Beta (Full Period)": [
                self._rolling_beta(self.stocks[s]).iloc[-1]
                for s in self.stocks.columns
            ],
            "Mean Rolling Beta": rolling_betas.mean(),
            "Std Rolling Beta": rolling_betas.std(),
            "Min Rolling Beta": rolling_betas.min(),
            "Max Rolling Beta": rolling_betas.max(),
            "Beta Range": rolling_betas.max() - rolling_betas.min(),
        }, index=self.stocks.columns)

        stats.index.name = "Stock"
        return stats.round(4)

    # ------------------------------------------------------------------ #
    #  Visualisations                                                      #
    # ------------------------------------------------------------------ #

    def plot_rolling_betas(
        self,
        rolling_betas: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot rolling beta over time for all stocks with market reference line.
        """
        n_stocks = len(self.stocks.columns)
        fig, axes = plt.subplots(n_stocks, 1, figsize=(13, 3 * n_stocks), sharex=True)

        colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#e74c3c"]

        for i, (stock, ax) in enumerate(zip(self.stocks.columns, axes)):
            beta = rolling_betas[stock].dropna()
            ax.plot(beta.index, beta, color=colors[i % len(colors)], linewidth=1.5, label=stock)
            ax.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.6, label="β=1 (Market)")
            ax.axhline(y=beta.mean(), color=colors[i % len(colors)], linestyle=":",
                       linewidth=1, alpha=0.7, label=f"Mean β={beta.mean():.2f}")

            # Shade high-beta periods
            ax.fill_between(beta.index, beta, 1,
                            where=(beta > 1), alpha=0.15, color="#e74c3c", label="Above market")
            ax.fill_between(beta.index, beta, 1,
                            where=(beta < 1), alpha=0.15, color="#2ecc71", label="Below market")

            ax.set_ylabel(f"{stock}\nBeta (β)", fontsize=10)
            ax.legend(fontsize=8, loc="upper right", ncol=4)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(max(0, beta.min() - 0.3), beta.max() + 0.3)

        axes[-1].set_xlabel("Date", fontsize=12)
        fig.suptitle(f"Rolling Beta Analysis ({self.window}-Day Window)",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_beta_distribution(
        self,
        rolling_betas: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of rolling beta for each stock (violin + box plot).
        Shows how much beta varies — a wide distribution = unstable risk profile.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        data_to_plot = [rolling_betas[col].dropna().values for col in rolling_betas.columns]
        positions = range(1, len(data_to_plot) + 1)

        vp = ax.violinplot(data_to_plot, positions=positions, showmedians=True, showextrema=True)

        colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#e74c3c"]
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[i % len(colors)])
            body.set_alpha(0.6)

        ax.axhline(y=1, color="black", linestyle="--", linewidth=1.5, label="β=1 (Market)")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(rolling_betas.columns, fontsize=12)
        ax.set_xlabel("Stock", fontsize=12)
        ax.set_ylabel("Rolling Beta (β)", fontsize=12)
        ax.set_title(f"Beta Distribution — {self.window}-Day Rolling Window\n"
                     f"Width = how much beta varies over time", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_rolling_alpha(
        self,
        rolling_alphas: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot annualised rolling alpha for all stocks."""
        fig, ax = plt.subplots(figsize=(13, 6))

        colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#e74c3c"]
        for i, stock in enumerate(rolling_alphas.columns):
            ax.plot(rolling_alphas.index, rolling_alphas[stock],
                    label=stock, color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)

        ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Annualised Rolling Alpha", fontsize=12)
        ax.set_title(f"Rolling Alpha ({self.window}-Day Window) — Is Outperformance Persistent?",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def main() -> None:
    """Run rolling beta analysis pipeline."""
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    data = pd.read_csv(
        os.path.join(data_dir, "merged_excess_returns.csv"),
        index_col=0, parse_dates=True
    )

    for window in [30, 60, 120]:
        print(f"\nComputing rolling betas (window={window} days)...")
        rba = RollingBetaAnalyzer(data, window=window)
        analysis = rba.compute_full_analysis()

        stats = rba.beta_regime_stats(analysis["beta"])
        print(f"\nBeta statistics (window={window}):")
        print(stats.to_string())

        rba.plot_rolling_betas(
            analysis["beta"],
            save_path=os.path.join(plots_dir, f"rolling_beta_{window}d.png")
        )
        rba.plot_beta_distribution(
            analysis["beta"],
            save_path=os.path.join(plots_dir, f"beta_distribution_{window}d.png")
        )
        rba.plot_rolling_alpha(
            analysis["alpha"],
            save_path=os.path.join(plots_dir, f"rolling_alpha_{window}d.png")
        )

        analysis["beta"].to_csv(os.path.join(data_dir, f"rolling_beta_{window}d.csv"))

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
