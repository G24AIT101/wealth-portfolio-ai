import numpy as np
import pandas as pd

class Validator:
    def __init__(self):
        pass
    
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.02):
        """Annualized Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def max_drawdown(returns):
        """Maximum drawdown"""
        if len(returns) == 0:
            return 0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0.02):
        """Sortino ratio using downside deviation"""
        if len(returns) == 0:
            return 0
        
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_std = np.sqrt(252) * downside_returns.std()
        
        if downside_std == 0:
            return 0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        return excess_returns / downside_std
    
    def rolling_window_backtest(
        self,
        full_price_df,
        run_fn,
        train_days=756,   # ~3 years
        test_days=126     # ~6 months
    ):
        results = []

        dates = full_price_df.index

        for start in range(0, len(dates) - train_days - test_days, test_days):
            train_start = dates[start]
            train_end = dates[start + train_days]

            # 🔑 run_fn decides how to fetch + train using dates
            weights = run_fn(train_start, train_end)

            test_prices = full_price_df.loc[
                train_end : dates[start + train_days + test_days]
            ]

            returns = test_prices.pct_change().dropna()
            w = pd.Series(weights)

            portfolio_returns = (returns * w).sum(axis=1)

            results.append({
                "Sharpe": self.sharpe_ratio(portfolio_returns),
                "Volatility": portfolio_returns.std() * np.sqrt(252),
                "Max_Drawdown": self.max_drawdown(portfolio_returns),
                "Sortino": self.sortino_ratio(portfolio_returns)
            })

        return pd.DataFrame(results)