from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from portfolio_optimizer import PortfolioOptimizer
from validator import Validator

class WealthAdvisorAI:
    def __init__(self, amount, risk, duration_months, feature_mode="baseline"):
        self.amount = amount
        self.feature_mode = feature_mode
        self.duration_months = duration_months

    def run(self):
        print(f"\n=== Running experiment: {self.feature_mode.upper()} FEATURES ===")

        fetcher = DataFetcher()
        stock_data = fetcher.fetch(years=15)

        fe = FeatureEngineer()
        processed = {
            t: fe.add_features(df, mode=self.feature_mode)
            for t, df in stock_data.items()
        }

        trainer = ModelTrainer()
        predictions = []

        for t, df in processed.items():
            result = trainer.train(df)
            predictions.append(result["predictions"][-1])

        optimizer = PortfolioOptimizer()
        portfolio = optimizer.optimize(stock_data, predictions, self.amount)

        validator = Validator()
        performance = validator.backtest_portfolio(stock_data, portfolio["weights"])

        return portfolio, performance
