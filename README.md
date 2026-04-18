# Factor Model Research — CAPM & Fama-French 3-Factor Analysis

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen.svg)](#testing)
[![Code Style](https://img.shields.io/badge/code%20style-PEP8-blue.svg)](https://peps.python.org/pep-0008/)

A quantitative finance research project implementing CAPM and Fama-French 3-factor regression analysis, portfolio backtesting, and automated research report generation across five large-cap U.S. equities.

**Portfolio:** AAPL · MSFT · GOOGL · JPM · XOM  
**Period:** January 2023 – December 2025 (751 trading days)  
**Benchmark:** S&P 500 (^GSPC)

---

## Key Results

| Stock | CAPM Beta | CAPM R² | Annual Alpha | Significant? |
|-------|-----------|---------|--------------|-------------|
| AAPL  | 1.16 | 0.473 | +4.1% | No |
| MSFT  | 1.04 | 0.457 | +4.1% | No |
| GOOGL | 1.19 | 0.352 | +19.5% | No |
| JPM   | 0.92 | 0.362 | +13.5% | No |
| XOM   | 0.45 | 0.090 | −2.4% | No |

**Backtested portfolio (equal-weight top 3, monthly rebalance):**  
Sharpe: 1.37 · CAGR: 31.5% · Max Drawdown: −27.9% · Excess vs Benchmark: +64.2%

---

## Project Structure

```
Factor Modeling/
├── factor_model/
│   ├── data_collector.py     # Yahoo Finance data download & log returns
│   ├── regression.py         # CAPM single-factor OLS regression
│   ├── ff3_collector.py      # Fama-French 3 factors from Kenneth French's library
│   ├── ff3_regression.py     # FF3 multi-factor regression & CAPM vs FF3 comparison
│   └── backtest.py           # Portfolio backtesting engine with performance metrics
├── notebooks/
│   └── capm_analysis.ipynb   # Interactive CAPM analysis notebook
├── reports/
│   ├── generate_report.py    # Auto-generate HTML research report
│   └── capm_ff3_research_report.html
├── data/
│   ├── merged_excess_returns.csv
│   ├── capm_results.csv
│   ├── ff3_results.csv
│   └── model_comparison.csv
├── plots/                    # All generated charts
├── scripts/
│   └── show_plots.py         # Interactive plot viewer (macOS)
└── tests/
    ├── test_data_collector.py
    └── test_regression.py
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Download data & compute excess returns
python -m factor_model.data_collector

# 2. Run CAPM analysis
python -m factor_model.regression --save-plots

# 3. Run Fama-French 3-factor analysis
python -m factor_model.ff3_regression

# 4. Run backtest
python -m factor_model.backtest

# 5. Generate full HTML research report
python reports/generate_report.py
```

---

## Models

### CAPM (Capital Asset Pricing Model)
```
R_i - R_f = α + β(R_m - R_f) + ε
```
Single-factor model explaining stock returns through market exposure alone.

### Fama-French 3-Factor Model
```
R_i - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + ε
```
Extends CAPM with two additional risk factors:
- **SMB** (Small Minus Big) — size premium
- **HML** (High Minus Low) — value premium

---

## Key Findings

1. **No statistically significant alpha** exists in any stock — consistent with efficient large-cap markets
2. **Tech stocks are high-beta** (β > 1) — amplify market moves in both directions
3. **XOM is a structural outlier** — CAPM explains only 9% of its returns; oil prices dominate
4. **FF3 improves R² for all stocks** by capturing size and value dynamics CAPM misses
5. **Portfolio outperformed the benchmark** by 64% over 3 years (in-sample; requires out-of-sample validation)

---

## Performance Metrics

| Metric | Portfolio |
|--------|-----------|
| Total Return | 126.1% |
| CAGR | 31.5% |
| Sharpe Ratio | 1.37 |
| Sortino Ratio | 2.02 |
| Max Drawdown | −27.9% |
| Calmar Ratio | 1.13 |
| Excess vs Benchmark | +64.2% |

> **Note:** Results are in-sample. A walk-forward test on unseen data is required before drawing conclusions about future performance.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | Stock price data |
| `pandas-datareader` | Fama-French factor data |
| `statsmodels` | OLS regression |
| `matplotlib` / `seaborn` | Visualizations |
| `numpy` / `pandas` | Data processing |

---

## Testing

```bash
pytest tests/ -v
```

---

## Author

**Patience Fuglo**  
Quantitative Finance Research

---

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Past performance does not guarantee future results.
