from user_input import UserInput
from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from portfolio_optimizer import PortfolioOptimizer
from validator import Validator
import pandas as pd


class WealthAdvisorAI:
    def __init__(
        self,
        amount,
        risk,
        duration_months,
        feature_mode="baseline",
        external_price_df=None
    ):
        self.user = UserInput(amount, risk, duration_months)
        self.feature_mode = feature_mode
        self.external_price_df = external_price_df

    def run(self):
        print(f"\n=== Running experiment: {self.feature_mode.upper()} FEATURES ===")

        # STEP 1: Get price data
        if self.external_price_df is not None:
            price_df = self.external_price_df.copy()
        else:
            years = self.user.historical_years()
            fetcher = DataFetcher()
            stock_data = fetcher.fetch(years)
            price_df = pd.concat(
                {t: d["Close"] for t, d in stock_data.items()},
                axis=1
            ).dropna()

        # STEP 2: Feature engineering per stock
        fe = FeatureEngineer(feature_mode=self.feature_mode)
        stock_frames = {}

        for ticker in price_df.columns:
            df = pd.DataFrame({"Close": price_df[ticker]})
            df = fe.add_features(df)
            stock_frames[ticker] = df

        # STEP 3: Train model per stock
        trainer = ModelTrainer()
        predicted_returns = []
        
        # Store predictions per stock
        stock_predictions = {}

        for ticker, df in stock_frames.items():
            results = trainer.train(df)
            last_prediction = results["predictions"][-1] if len(results["predictions"]) > 0 else 0
            predicted_returns.append(last_prediction)
            stock_predictions[ticker] = last_prediction

        # STEP 4: Portfolio optimization
        optimizer = PortfolioOptimizer()
        portfolio = optimizer.optimize(
            price_df,
            predicted_returns,
            self.user.amount
        )

        # STEP 5: Backtesting
        validator = Validator()
        
        def backtest_run_fn(train_start, train_end):
            # Filter price data for the training window
            train_prices = price_df.loc[train_start:train_end]
            
            if len(train_prices) < 100:  # Minimum data required
                n_stocks = len(price_df.columns)
                return {ticker: 1/n_stocks for ticker in price_df.columns}
            
            # Re-run feature engineering and prediction for this window
            window_predictions = []
            for ticker in train_prices.columns:
                df = pd.DataFrame({"Close": train_prices[ticker]})
                df = fe.add_features(df)
                
                if len(df) < 10:
                    window_predictions.append(0)
                    continue
                    
                window_results = trainer.train(df)
                last_pred = window_results["predictions"][-1] if len(window_results["predictions"]) > 0 else 0
                window_predictions.append(last_pred)
            
            # Optimize portfolio for this window
            window_portfolio = optimizer.optimize(
                train_prices,
                window_predictions,
                self.user.amount
            )
            
            return window_portfolio["weights"]
        
        backtest_results = validator.rolling_window_backtest(
            price_df,
            backtest_run_fn,
            train_days=756,
            test_days=126
        )

        return portfolio, backtest_results