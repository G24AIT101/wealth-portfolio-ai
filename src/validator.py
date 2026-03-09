import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        train_days=756,
        test_days=126
    ):
        """
        Rolling window backtest with turnover tracking.
        
        Returns DataFrame with columns: Sharpe, Volatility, Max_Drawdown, Sortino, Turnover
        """
        results = []
        all_weights = []

        dates = full_price_df.index

        for start in range(0, len(dates) - train_days - test_days, test_days):
            train_start = dates[start]
            train_end   = dates[start + train_days]
            test_end    = dates[start + train_days + test_days]

            weights = run_fn(train_start, train_end)
            all_weights.append(weights)

            test_prices = full_price_df.loc[train_end:test_end]
            returns = test_prices.pct_change().dropna()
            w = pd.Series(weights)

            portfolio_returns = (returns * w).sum(axis=1)

            # Turnover calculation
            turnover = np.nan
            if len(all_weights) > 1:
                prev_weights = pd.Series(all_weights[-2])
                curr_weights = pd.Series(weights)
                all_tickers = prev_weights.index.union(curr_weights.index)
                prev_w = prev_weights.reindex(all_tickers, fill_value=0)
                curr_w = curr_weights.reindex(all_tickers, fill_value=0)
                turnover = (prev_w - curr_w).abs().sum() / 2

            results.append({
                "Sharpe":       self.sharpe_ratio(portfolio_returns),
                "Volatility":   portfolio_returns.std() * np.sqrt(252),
                "Max_Drawdown": self.max_drawdown(portfolio_returns),
                "Sortino":      self.sortino_ratio(portfolio_returns),
                "Turnover":     turnover
            })

        return pd.DataFrame(results)

    @staticmethod
    def plot_wealth_curve(returns_dict, title="Wealth Accumulation", figsize=(10, 6)):
        plt.figure(figsize=figsize)
        for label, returns in returns_dict.items():
            wealth = (1 + returns).cumprod()
            plt.plot(wealth.index, wealth, label=label)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Wealth ($)")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_drawdowns(returns_dict, title="Drawdown Comparison", figsize=(10, 6)):
        plt.figure(figsize=figsize)
        for label, returns in returns_dict.items():
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            plt.fill_between(drawdown.index, drawdown, 0, label=label, alpha=0.3)
            plt.plot(drawdown.index, drawdown, linewidth=1)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.legend()
        plt.grid(True)
        plt.show()