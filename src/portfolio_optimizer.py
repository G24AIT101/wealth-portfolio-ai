import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.discrete_allocation import DiscreteAllocation

class PortfolioOptimizer:
    """
    Uses predicted returns + covariance matrix to produce:
        - Optimal weights (max sharpe)
        - Discrete allocation (number of shares)
    """

    def optimize(self, stock_data_dict, predicted_returns, investment_amount):
        tickers = list(stock_data_dict.keys())

        # 1. Prepare expected returns (from model predictions)
        exp_returns = pd.Series(predicted_returns, index=tickers)

        # 2. Prepare historical data for covariance matrix
        price_df = pd.DataFrame({
            t: stock_data_dict[t]["Close"] for t in tickers
        })

        returns_df = price_df.pct_change().dropna()

        # Covariance matrix
        cov_matrix = CovarianceShrinkage(returns_df).ledoit_wolf()

        # 3. Efficient Frontier Optimization
        ef = EfficientFrontier(exp_returns, cov_matrix)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        # 4. Convert weights to share amounts
        latest_prices = price_df.iloc[-1].to_dict()
        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=investment_amount)
        allocation, leftover = da.lp_portfolio()

        return {
            "weights": cleaned_weights,
            "allocation": allocation,
            "leftover": leftover
        }
