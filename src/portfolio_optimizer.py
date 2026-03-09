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
            # If DataFrame, columns are tickers, rows are dates
            price_df = stock_data.copy()
            tickers = list(price_df.columns)
        else:
            # If dict, extract Close prices and create DataFrame
            tickers = list(stock_data.keys())
            price_series_list = [stock_data[t]["Close"] for t in tickers]
            price_df = pd.concat(price_series_list, axis=1, keys=tickers)

        # Covariance matrix
        cov_matrix = risk_models.sample_cov(price_df)

        # Expected returns as Series
        mu = pd.Series(predicted_returns, index=tickers)

        # Try max_sharpe with different solvers
        weights = None
        ef = None
        solvers = [None, 'ECOS', 'SCS']  # None defaults to OSQP

        for solver in solvers:
            try:
                ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, 1))
                weights = ef.max_sharpe(solver=solver)
                print(f"Max Sharpe succeeded with solver {solver}")
                break
            except Exception as e:
                print(f"Max Sharpe with solver {solver} failed: {e}")
                continue

        if weights is None:
            # Fallback to min volatility
            print("Max Sharpe failed with all solvers. Trying min volatility...")
            for solver in solvers:
                try:
                    ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, 1))
                    weights = ef.min_volatility(solver=solver)
                    print(f"Min volatility succeeded with solver {solver}")
                    break
                except Exception as e:
                    print(f"Min volatility with solver {solver} failed: {e}")
                    continue

        if weights is None:
            # Ultimate fallback: equal weights
            print("All optimization attempts failed. Using equal weights.")
            n = len(tickers)
            equal_weights = {t: 1/n for t in tickers}
            cleaned_weights = equal_weights
        else:
            cleaned_weights = ef.clean_weights()

        # Discrete allocation
        latest_prices = price_df.iloc[-1]
        allocation = {}
        leftover_cash = investment_amount

        for t in tickers:
            weight = cleaned_weights.get(t, 0)
            amount_allocated = investment_amount * weight

            # Ensure price is a scalar float
            price = latest_prices[t]
            if hasattr(price, 'item'):
                price = float(price.item())
            else:
                price = float(price)

            # Calculate Number of Shares (int)
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