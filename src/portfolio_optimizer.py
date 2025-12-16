import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns


class PortfolioOptimizer:
    """
    Optimizes portfolio using Mean-Variance Optimization (Max Sharpe Ratio)
    """

    def optimize(self, stock_data, predicted_returns, investment_amount):
        tickers = list(stock_data.keys())

        # Historical prices DataFrame (this is SAFE: values are Series)
        price_df = pd.DataFrame({
            t: stock_data[t]["Close"] for t in tickers
        })

        # Covariance matrix
        cov_matrix = risk_models.sample_cov(price_df)

        # Expected returns as Series (IMPORTANT)
        mu = pd.Series(predicted_returns, index=tickers)

        # Optimization
        ef = EfficientFrontier(mu, cov_matrix)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        # Discrete allocation
        latest_prices = price_df.iloc[-1]
        allocation = {}
        leftover_cash = investment_amount

        for t in tickers:
            amount = investment_amount * cleaned_weights.get(t, 0)
            shares = int(amount // latest_prices[t])
            allocation[t] = shares * latest_prices[t]
            leftover_cash -= allocation[t]

        return {
            "weights": cleaned_weights,        # dict
            "allocation": allocation,          # dict
            "leftover_cash": leftover_cash     # float
        }
