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

        for ticker, df in stock_frames.items():
            results = trainer.train(df)
            predicted_returns.append(results["predictions"][-1])

        # STEP 4: Portfolio optimization
        optimizer = PortfolioOptimizer()
        portfolio = optimizer.optimize(
            price_df,
            predicted_returns,
            self.user.amount
        )

        return portfolio, None
