import numpy as np
import pandas as pd

class Validator:
    """
    Validates correctness of model and portfolio.
    """

    def backtest_portfolio(self, stock_data_dict, weights):
        tickers = list(stock_data_dict.keys())

        # FIX: Use pd.concat for robustness against 'scalar' value errors
        price_series_list = [stock_data_dict[t]["Close"] for t in tickers]
        price_df = pd.concat(price_series_list, axis=1, keys=tickers)

        # FIX: Set fill_method=None to avoid FutureWarning in pandas 2.0+
        returns_df = price_df.pct_change(fill_method=None).dropna()

        # FIX: Handle missing keys in weights (clean_weights drops 0-weight stocks)
        # Use .get(t, 0.0) to default to 0 if ticker not in weights
        w = np.array([weights.get(t, 0.0) for t in tickers])
        
        # Calculate portfolio daily returns
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
        if not tickers:
            return {}
        equal_weights = {t: 1/len(tickers) for t in tickers}
        return self.backtest_portfolio(stock_data_dict, equal_weights)