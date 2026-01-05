import numpy as np
import pandas as pd

class Validator:
    """
    Validates portfolio performance and robustness.
    Includes:
    - Sharpe ratio
    - Sortino ratio
    - Volatility
    - Max drawdown
    - Rolling-window backtesting
    """

    # ===============================
    # Core risk metrics
    # ===============================

    def max_drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def sortino_ratio(self, returns, risk_free=0.0):
        downside = returns[returns < 0]
        if downside.std() == 0:
            return 0.0
        return (returns.mean() - risk_free) / downside.std()

    # ===============================
    # Single-period backtest (STEP 2)
    # ===============================

    def backtest_portfolio(self, stock_data_dict, weights):
        tickers = list(stock_data_dict.keys())

        price_series_list = [stock_data_dict[t]["Close"] for t in tickers]
        price_df = pd.concat(price_series_list, axis=1, keys=tickers)

        returns_df = price_df.pct_change(fill_method=None).dropna()

        w = np.array([weights.get(t, 0.0) for t in tickers])
        portfolio_returns = (returns_df * w).sum(axis=1)

        avg_daily = portfolio_returns.mean()
        std_daily = portfolio_returns.std()

        volatility = std_daily * np.sqrt(252)
        sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
        sortino = self.sortino_ratio(portfolio_returns)
        max_dd = self.max_drawdown(portfolio_returns)

        return {
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Volatility": volatility,
            "Max_Drawdown": max_dd
        }

    # ===============================
    # Equal-weight baseline
    # ===============================

    def equal_weight_comparison(self, stock_data_dict):
        tickers = list(stock_data_dict.keys())
        if not tickers:
            return {}
        equal_weights = {t: 1 / len(tickers) for t in tickers}
        return self.backtest_portfolio(stock_data_dict, equal_weights)

    # ===============================
    # Rolling-window backtesting (STEP 4)
    # ===============================

    def rolling_window_backtest(
        self,
        price_df,
        weights_fn,
        train_days=756,   # ~3 years
        test_days=126     # ~6 months
    ):
        """
        price_df: DataFrame of prices (Date x Tickers)
        weights_fn: function that returns portfolio weights
        """

        results = []
        start = 0
        end = train_days + test_days

        while end <= len(price_df):
            train_prices = price_df.iloc[start:start + train_days]
            test_prices = price_df.iloc[start + train_days:end]

            test_returns = test_prices.pct_change(fill_method=None).dropna()

            weights = weights_fn()
            tickers = test_returns.columns
            w = np.array([weights.get(t, 0.0) for t in tickers])

            portfolio_returns = (test_returns * w).sum(axis=1)

            metrics = {
                "Sharpe": (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
                          if portfolio_returns.std() > 0 else 0,
                "Volatility": portfolio_returns.std() * np.sqrt(252),
                "Max_Drawdown": self.max_drawdown(portfolio_returns)
            }

            results.append(metrics)
            start += test_days
            end += test_days

        return pd.DataFrame(results)
