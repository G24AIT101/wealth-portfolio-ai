import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

class PortfolioOptimizer:
    """
    Optimizes portfolio using Mean-Variance Optimization (Max Sharpe Ratio)
    """

    def optimize(self, stock_data, predicted_returns, investment_amount):
        tickers = list(stock_data.keys())

        # Historical prices DataFrame
        price_df = pd.DataFrame({
            t: stock_data[t]["Close"] for t in tickers
        })

        # Covariance matrix
        cov_matrix = risk_models.sample_cov(price_df)

        # Expected returns as Series
        mu = pd.Series(predicted_returns, index=tickers)

        # Optimization
        # weight_bounds=(0, 1) means no short selling
        ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, 1))
        
        try:
            weights = ef.max_sharpe()
        except:
            # Fallback if max_sharpe fails
            print("Warning: Max Sharpe failed, falling back to min volatility.")
            weights = ef.min_volatility()
            
        cleaned_weights = ef.clean_weights()

        # Discrete allocation
        latest_prices = price_df.iloc[-1]
        allocation = {}
        leftover_cash = investment_amount

        for t in tickers:
            weight = cleaned_weights.get(t, 0)
            amount_allocated = investment_amount * weight
            price = latest_prices[t]
            
            # FIX: Calculate Number of Shares (int)
            shares = int(amount_allocated // price)
            
            # Only add to allocation if we buy at least 1 share
            if shares > 0:
                allocation[t] = shares
                cost = shares * price
                leftover_cash -= cost

        return {
            "weights": cleaned_weights,        # dict {ticker: weight}
            "allocation": allocation,          # dict {ticker: num_shares}
            "leftover_cash": leftover_cash     # float
        }