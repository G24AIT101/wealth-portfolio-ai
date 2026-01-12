import numpy as np
import pandas as pd

class Validator:
    """
    Validation utilities:
    - Sharpe
    - Sortino
    - Max Drawdown
    - Rolling-window backtesting
    """

    def sharpe_ratio(self, returns, rf=0.0):
        return (returns.mean() - rf) / returns.std() * np.sqrt(252)

    def sortino_ratio(self, returns, rf=0.0):
        downside = returns[returns < 0]
        if downside.std() == 0:
            return np.nan
        return (returns.mean() - rf) / downside.std() * np.sqrt(252)

    def max_drawdown(self, returns):
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()

    def evaluate_portfolio(self, returns):
        return {
            "Sharpe": self.sharpe_ratio(returns),
            "Sortino": self.sortino_ratio(returns),
            "Volatility": returns.std() * np.sqrt(252),
            "Max_Drawdown": self.max_drawdown(returns)
        }

    def rolling_window_backtest(
        self,
        price_df,
        run_fn,
        train_days=756,   # ~3 years
        test_days=126     # ~6 months
    ):
        """
        run_fn(train_price_df) -> weights dict
        """

        results = []

        for start in range(0, len(price_df) - train_days - test_days, test_days):
            train_prices = price_df.iloc[start:start + train_days]
            test_prices = price_df.iloc[start + train_days:start + train_days + test_days]

            # 🔑 TRAIN & OPTIMIZE ONLY ON WINDOW DATA
            weights = run_fn(train_prices)

            returns = test_prices.pct_change().dropna()
            w = np.array([weights[c] for c in returns.columns])
            portfolio_returns = (returns * w).sum(axis=1)

            metrics = self.evaluate_portfolio(portfolio_returns)
            results.append(metrics)

        return pd.DataFrame(results)
