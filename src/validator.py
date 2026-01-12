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
