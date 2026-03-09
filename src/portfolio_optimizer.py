import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models

class PortfolioOptimizer:
    """
    Optimizes portfolio using Mean-Variance Optimization (Max Sharpe Ratio)
    """

    def optimize(self, stock_data, predicted_returns, investment_amount):
        # Handle both dict and DataFrame inputs
        if isinstance(stock_data, pd.DataFrame):
            price_df = stock_data.copy()
            tickers = list(price_df.columns)
        else:
            tickers = list(stock_data.keys())
            price_series_list = [stock_data[t]["Close"] for t in tickers]
            price_df = pd.concat(price_series_list, axis=1, keys=tickers)

        # Covariance matrix
        cov_matrix = risk_models.sample_cov(price_df)

        # Expected returns as Series
        mu = pd.Series(predicted_returns, index=tickers)

        # Try max_sharpe
        try:
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, 1))
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
        except Exception as e:
            print(f"Max Sharpe failed: {e}. Trying min volatility...")
            try:
                ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, 1))
                weights = ef.min_volatility()
                cleaned_weights = ef.clean_weights()
            except Exception as e2:
                print(f"Min volatility also failed: {e2}. Using equal weights.")
                # Ultimate fallback: equal weights
                n = len(tickers)
                cleaned_weights = {t: 1/n for t in tickers}

        # Discrete allocation
        latest_prices = price_df.iloc[-1]
        allocation = {}
        leftover_cash = investment_amount

        for t in tickers:
            weight = cleaned_weights.get(t, 0)
            amount_allocated = investment_amount * weight

            price = latest_prices[t]
            if hasattr(price, 'item'):
                price = float(price.item())
            else:
                price = float(price)

            shares = int(amount_allocated // price)
            if shares > 0:
                allocation[t] = shares
                cost = shares * price
                leftover_cash -= cost

        return {
            "weights": cleaned_weights,        # dict {ticker: weight}
            "allocation": allocation,          # dict {ticker: num_shares}
            "leftover_cash": leftover_cash     # float
        }