"""
Unit tests for data_collector module.
"""

import pytest
import pandas as pd
import numpy as np
from factor_model.data_collector import (
    get_risk_free_rate,
    calculate_excess_returns,
    merge_data,
    annualized_mean_returns,
    extract_close_series,
    TRADING_DAYS,
)


class TestRiskFreeRate:
    """Tests for risk-free rate calculation."""

    def test_risk_free_rate_positive(self):
        """Risk-free rate should be positive."""
        rf = get_risk_free_rate()
        assert rf > 0

    def test_risk_free_rate_daily(self):
        """Risk-free rate should be a small daily value."""
        rf = get_risk_free_rate()
        assert rf < 0.001  # Less than 0.1% daily

    def test_risk_free_rate_annualized(self):
        """Annualized risk-free rate should be ~2%."""
        rf = get_risk_free_rate()
        annualized = rf * TRADING_DAYS
        assert 0.01 < annualized < 0.05  # Between 1% and 5%


class TestExcessReturns:
    """Tests for excess returns calculation."""

    def test_excess_returns_basic(self):
        """Excess returns should subtract risk-free rate."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.005])
        rf_rate = 0.0001
        excess = calculate_excess_returns(returns, rf_rate)
        
        expected = returns - rf_rate
        pd.testing.assert_series_equal(excess, expected)

    def test_excess_returns_dataframe(self):
        """Should work with DataFrames."""
        returns = pd.DataFrame({
            "AAPL": [0.01, 0.02, -0.01],
            "MSFT": [0.005, 0.015, -0.005]
        })
        rf_rate = 0.0001
        excess = calculate_excess_returns(returns, rf_rate)
        
        assert isinstance(excess, pd.DataFrame)
        assert list(excess.columns) == ["AAPL", "MSFT"]

    def test_excess_returns_empty_raises(self):
        """Should raise error for empty returns."""
        with pytest.raises(ValueError, match="empty"):
            calculate_excess_returns(pd.Series([]), 0.0001)

    def test_excess_returns_none_raises(self):
        """Should raise error for None input."""
        with pytest.raises(ValueError, match="empty"):
            calculate_excess_returns(None, 0.0001)


class TestMergeData:
    """Tests for merging stock and market data."""

    def test_merge_data_basic(self):
        """Should merge on common dates."""
        dates = pd.date_range("2024-01-01", periods=5)
        stock_data = pd.DataFrame(
            {"AAPL": [0.01, 0.02, -0.01, 0.005, 0.015]},
            index=dates
        )
        market_data = pd.DataFrame(
            {"Market": [0.005, 0.01, -0.005, 0.003, 0.008]},
            index=dates
        )
        
        merged = merge_data(stock_data, market_data)
        
        assert len(merged) == 5
        assert "AAPL" in merged.columns
        assert "Market" in merged.columns

    def test_merge_data_partial_overlap(self):
        """Should only keep overlapping dates."""
        dates1 = pd.date_range("2024-01-01", periods=5)
        dates2 = pd.date_range("2024-01-03", periods=5)
        
        stock_data = pd.DataFrame({"AAPL": [0.01] * 5}, index=dates1)
        market_data = pd.DataFrame({"Market": [0.005] * 5}, index=dates2)
        
        merged = merge_data(stock_data, market_data)
        
        assert len(merged) == 3  # Only 3 overlapping days

    def test_merge_data_empty_stock_raises(self):
        """Should raise error for empty stock data."""
        with pytest.raises(ValueError, match="stock_excess is empty"):
            merge_data(pd.DataFrame(), pd.DataFrame({"Market": [0.01]}))

    def test_merge_data_empty_market_raises(self):
        """Should raise error for empty market data."""
        stock_data = pd.DataFrame({"AAPL": [0.01]})
        with pytest.raises(ValueError, match="market_excess is empty"):
            merge_data(stock_data, pd.DataFrame())


class TestAnnualizedReturns:
    """Tests for annualized return calculation."""

    def test_annualized_returns_basic(self):
        """Should multiply mean by trading days."""
        returns = pd.Series([0.001] * 252)  # 0.1% daily
        annualized = annualized_mean_returns(returns)
        
        assert abs(annualized - 0.252) < 0.001  # ~25.2% annual

    def test_annualized_returns_negative(self):
        """Should handle negative returns."""
        returns = pd.Series([-0.001] * 252)
        annualized = annualized_mean_returns(returns)
        
        assert annualized < 0


class TestExtractCloseSeries:
    """Tests for extracting close prices from yfinance data."""

    def test_extract_simple_dataframe(self):
        """Should extract Close from simple DataFrame."""
        data = pd.DataFrame({
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [99, 100],
            "Close": [101, 102]
        })
        
        prices = extract_close_series(data)
        
        assert len(prices) == 2
        assert prices.iloc[0] == 101

    def test_extract_empty_raises(self):
        """Should raise error for empty data."""
        with pytest.raises(ValueError, match="empty"):
            extract_close_series(pd.DataFrame())

    def test_extract_missing_close_raises(self):
        """Should raise error if no Close column."""
        data = pd.DataFrame({"Open": [100, 101], "High": [102, 103]})
        
        with pytest.raises(ValueError, match="Close"):
            extract_close_series(data)
