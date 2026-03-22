#!/usr/bin/env python3
"""
Interactive Plot Display Script
================================

Run this script directly (not through VS Code terminal) to see interactive plots.

Usage:
    python show_plots.py
    
Or double-click the file in Finder.
"""

import matplotlib
matplotlib.use('macosx')

import matplotlib.pyplot as plt
import pandas as pd
import os

# Load data from data/ directory
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "merged_excess_returns.csv")
data = pd.read_csv(data_path, index_col=0, parse_dates=True)

market = data["Market"]
stocks = data.drop(columns=["Market"])

# Import regression functions
import statsmodels.api as sm

def single_factor_regression(stock_returns, market_returns):
    X = sm.add_constant(market_returns)
    return sm.OLS(stock_returns, X).fit()

# Run regressions
results = []
for stock in stocks.columns:
    model = single_factor_regression(stocks[stock], market)
    results.append({
        "Stock": stock,
        "Alpha": model.params.iloc[0],
        "Beta": model.params.iloc[1],
        "R_squared": model.rsquared,
    })

results_df = pd.DataFrame(results)

print("\n=== CAPM Results ===")
print(results_df.to_string(index=False))

# Plot 1: Beta Comparison
print("\n📊 Showing Beta Comparison plot...")
print("   (Close the window to see next plot)")

fig1, ax1 = plt.subplots(figsize=(10, 6))
colors = ["#2ecc71" if b < 1 else "#e74c3c" for b in results_df["Beta"]]
bars = ax1.bar(results_df["Stock"], results_df["Beta"], color=colors, edgecolor="black")
ax1.axhline(y=1, color="black", linestyle="--", linewidth=1.5, label="Market (β=1)")
ax1.set_xlabel("Stock", fontsize=12)
ax1.set_ylabel("Beta (β)", fontsize=12)
ax1.set_title("CAPM Beta Comparison", fontsize=14)
ax1.legend()
ax1.grid(True, axis="y", alpha=0.3)

for bar, beta in zip(bars, results_df["Beta"]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{beta:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.show(block=True)

# Plot 2-6: Individual CAPM Regression plots
for stock in stocks.columns:
    print(f"\n📈 Showing CAPM regression for {stock}...")
    print("   (Close the window to see next plot)")
    
    model = single_factor_regression(stocks[stock], market)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(market, stocks[stock], alpha=0.5, label="Daily Returns")
    ax.plot(market.sort_values(),
            model.predict(sm.add_constant(market.sort_values())),
            color="red", linewidth=2,
            label=f"CAPM: β={model.params.iloc[1]:.2f}")
    
    ax.set_xlabel("Market Excess Returns", fontsize=12)
    ax.set_ylabel(f"{stock} Excess Returns", fontsize=12)
    ax.set_title(f"CAPM Regression: {stock}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)

print("\n✅ All plots displayed!")
