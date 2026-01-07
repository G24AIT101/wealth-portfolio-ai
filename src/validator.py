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
    - Rolling-window backtesting with retraining
    """

    # ===============================
    # Risk metrics
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

        price_df = pd.concat(
            [stock_data_dict[t]["Close"] for t in tickers],
            axis=1,
            keys=tickers
        )

        returns = price_df.pct_change(fill_method=None).dropna()

        w = np.array([weights.get(t, 0.0) for t in tickers])
        portfolio_returns = (returns * w).sum(axis=1)

        sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252) \
                 if portfolio_returns.std() > 0 else 0

        return {
            "Sharpe": sharpe,
            "Sortino": self.sortino_ratio(portfolio_returns),
            "Volatility": portfolio_returns.std() * np.sqrt(252),
            "Max_Drawdown": self.max_drawdown(portfolio_returns)
        }

    # ===============================
    # Rolling-window backtesting (STEP 4 — FIXED)
    # ===============================

    def rolling_window_backtest(
        self,
        price_df,
        run_fn,
        train_days=756,   # ~3 years
        test_days=126     # ~6 months
    ):
        """
        price_df : DataFrame (Date x Tickers)
        run_fn   : function(start_idx, end_idx) -> weights
                   This MUST retrain + re-optimize
        """

        results = []
        start = 0

        while start + train_days + test_days <= len(price_df):
            train_end = start + train_days
            test_end = train_end + test_days

            # Train + optimize INSIDE window
            weights = run_fn(start, train_end)

            test_prices = price_df.iloc[train_end:test_end]
            test_returns = test_prices.pct_change(fill_method=None).dropna()

            w = np.array([weights.get(t, 0.0) for t in test_returns.columns])
            portfolio_returns = (test_returns * w).sum(axis=1)

            results.append({
                "Sharpe": (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
                          if portfolio_returns.std() > 0 else 0,
                "Volatility": portfolio_returns.std() * np.sqrt(252),
                "Max_Drawdown": self.max_drawdown(portfolio_returns)
            })

            start += test_days

        return pd.DataFrame(results)
