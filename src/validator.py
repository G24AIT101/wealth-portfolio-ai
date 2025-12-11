import numpy as np
import pandas as pd

class Validator:
    """
    Validates correctness of model and portfolio:
        - Direction accuracy
        - RMSE, MAE (from trainer)
        - Compare model portfolio vs equal-weight portfolio
    """

    def backtest_portfolio(self, stock_data_dict, weights):
        tickers = list(stock_data_dict.keys())

        # Extract last 30 days of returns for testing
        price_df = pd.DataFrame({
            t: stock_data_dict[t]["Close"] for t in tickers
        })

        returns_df = price_df.pct_change().dropna()

        # Calculate portfolio daily returns
        w = np.array([weights[t] for t in tickers])
        portfolio_returns = (returns_df * w).sum(axis=1)

        # Compute performance metrics
        cumulative_return = (1 + portfolio_returns).prod() - 1
        avg_daily = portfolio_returns.mean()
        std_daily = portfolio_returns.std()

        sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0

        return {
            "cumulative_return": cumulative_return,
            "avg_daily_return": avg_daily,
            "std_dev": std_daily,
            "sharpe_ratio": sharpe
        }

    def equal_weight_comparison(self, stock_data_dict):
        tickers = list(stock_data_dict.keys())
        equal_weights = {t: 1/len(tickers) for t in tickers}
        return self.backtest_portfolio(stock_data_dict, equal_weights)
