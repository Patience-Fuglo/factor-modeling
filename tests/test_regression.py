"""
Unit tests for regression module.
"""

import pytest
import pandas as pd
import numpy as np
from factor_model.regression import (
    single_factor_regression,
    extract_results,
    run_all_regressions,
)


class TestSingleFactorRegression:
    """Tests for CAPM regression."""

    @pytest.fixture
    def sample_data(self):
        """Create sample stock and market returns."""
        np.random.seed(42)
        n = 100
        market = np.random.normal(0.0005, 0.01, n)
        # Stock with beta=1.2 and small alpha
        stock = 0.0001 + 1.2 * market + np.random.normal(0, 0.005, n)
        return pd.Series(stock), pd.Series(market)

    def test_regression_returns_model(self, sample_data):
        """Should return a fitted statsmodels result."""
        stock, market = sample_data
        model = single_factor_regression(stock, market)
        
        assert hasattr(model, "params")
        assert hasattr(model, "rsquared")
        assert hasattr(model, "pvalues")

    def test_regression_beta_estimate(self, sample_data):
        """Beta should be close to true value (1.2)."""
        stock, market = sample_data
        model = single_factor_regression(stock, market)
        
        beta = model.params.iloc[1]
        assert 1.0 < beta < 1.4  # Close to 1.2

    def test_regression_has_intercept(self, sample_data):
        """Model should have intercept (alpha)."""
        stock, market = sample_data
        model = single_factor_regression(stock, market)
        
        assert len(model.params) == 2  # const + market


class TestExtractResults:
    """Tests for extracting regression results."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        np.random.seed(42)
        n = 100
        market = np.random.normal(0.0005, 0.01, n)
        stock = 0.0001 + 1.2 * market + np.random.normal(0, 0.005, n)
        return single_factor_regression(pd.Series(stock), pd.Series(market))

    def test_extract_results_keys(self, fitted_model):
        """Should return dict with all required keys."""
        results = extract_results(fitted_model, "AAPL")
        
        required_keys = ["Stock", "Alpha", "Beta", "R_squared", 
                        "Alpha_pvalue", "Beta_pvalue"]
        for key in required_keys:
            assert key in results

    def test_extract_results_stock_name(self, fitted_model):
        """Should include correct stock name."""
        results = extract_results(fitted_model, "TEST_STOCK")
        assert results["Stock"] == "TEST_STOCK"

    def test_extract_results_r_squared_valid(self, fitted_model):
        """R-squared should be between 0 and 1."""
        results = extract_results(fitted_model, "AAPL")
        assert 0 <= results["R_squared"] <= 1

    def test_extract_results_pvalues_valid(self, fitted_model):
        """P-values should be between 0 and 1."""
        results = extract_results(fitted_model, "AAPL")
        assert 0 <= results["Alpha_pvalue"] <= 1
        assert 0 <= results["Beta_pvalue"] <= 1


class TestRunAllRegressions:
    """Tests for running regressions on multiple stocks."""

    @pytest.fixture
    def multi_stock_data(self):
        """Create sample data for multiple stocks."""
        np.random.seed(42)
        n = 100
        market = pd.Series(np.random.normal(0.0005, 0.01, n))
        
        stocks = pd.DataFrame({
            "AAPL": 0.0001 + 1.2 * market + np.random.normal(0, 0.005, n),
            "MSFT": 0.0002 + 1.0 * market + np.random.normal(0, 0.004, n),
            "XOM": 0.0000 + 0.5 * market + np.random.normal(0, 0.006, n),
        })
        return stocks, market

    def test_run_all_returns_dataframe(self, multi_stock_data):
        """Should return a DataFrame."""
        stocks, market = multi_stock_data
        results = run_all_regressions(stocks, market)
        
        assert isinstance(results, pd.DataFrame)

    def test_run_all_correct_rows(self, multi_stock_data):
        """Should have one row per stock."""
        stocks, market = multi_stock_data
        results = run_all_regressions(stocks, market)
        
        assert len(results) == 3

    def test_run_all_stock_names(self, multi_stock_data):
        """Should include all stock names."""
        stocks, market = multi_stock_data
        results = run_all_regressions(stocks, market)
        
        assert set(results["Stock"]) == {"AAPL", "MSFT", "XOM"}

    def test_run_all_beta_ordering(self, multi_stock_data):
        """Betas should reflect true ordering (AAPL > MSFT > XOM)."""
        stocks, market = multi_stock_data
        results = run_all_regressions(stocks, market)
        
        betas = results.set_index("Stock")["Beta"]
        assert betas["AAPL"] > betas["MSFT"] > betas["XOM"]


class TestStatisticalProperties:
    """Tests for statistical properties of regression."""

    def test_perfect_correlation(self):
        """Perfect correlation should give R²=1 and beta=1."""
        market = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])
        stock = market.copy()  # Perfect correlation
        
        model = single_factor_regression(stock, market)
        
        assert model.rsquared > 0.99
        assert abs(model.params.iloc[1] - 1.0) < 0.01  # Beta ≈ 1

    def test_uncorrelated_low_rsquared(self):
        """Uncorrelated data should have low R²."""
        np.random.seed(42)
        market = pd.Series(np.random.normal(0, 0.01, 100))
        stock = pd.Series(np.random.normal(0, 0.01, 100))
        
        model = single_factor_regression(stock, market)
        
        assert model.rsquared < 0.1  # Low R² for uncorrelated
