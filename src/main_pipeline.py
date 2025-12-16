from user_input import UserInput
from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from portfolio_optimizer import PortfolioOptimizer
from validator import Validator
import pandas as pd


class WealthAdvisorAI:
    def __init__(self, amount, risk, duration_months):
        self.user = UserInput(amount, risk, duration_months)

    def run(self):
        print("\n=== Wealth Advisory AI System Running ===")

        # Step 1: Fetch data
        years = self.user.historical_years()
        fetcher = DataFetcher(self.user.risk)
        stock_data = fetcher.fetch(years)
        print("Data fetched:", list(stock_data.keys()))

        # Step 2: Feature engineering
        fe = FeatureEngineer()
        processed = {t: fe.add_features(df.copy()) for t, df in stock_data.items()}

        # Step 3: Train models + get predicted returns
        trainer = ModelTrainer()
        predicted_returns = []

        print("\nModel Performance Summary:")
        for t, df in processed.items():
            results = trainer.train(df)
            predicted_returns.append(results["predictions"][-1])

            print(f"\n📌 Stock: {t}")
            print("RMSE:", results["rmse"])
            print("MAE:", results["mae"])
            print("Direction accuracy:", results["direction_accuracy"])
            print("Feature importance:", results["feature_importance"])

        # Apply risk factor
        predicted_returns = [r * self.user.risk_factor() for r in predicted_returns]

        # Step 4: Portfolio optimization
        optimizer = PortfolioOptimizer()
        portfolio_result = optimizer.optimize(
            stock_data,
            predicted_returns,
            self.user.amount
        )

        print("\n=== Portfolio Optimization Completed ===")
        print("Weights:", portfolio_result["weights"])
        print("Allocation:", portfolio_result["allocation"])

        # Step 5: Validation & Backtesting
        validator = Validator()
        performance = validator.backtest_portfolio(stock_data, portfolio_result["weights"])
        equal_perf = validator.equal_weight_comparison(stock_data)

        # Convert to DataFrame for safe display & future plotting
        performance_df = pd.DataFrame([performance])
        equal_perf_df = pd.DataFrame([equal_perf])

        print("\n=== Validation Results ===")
        print("Model-based portfolio performance:")
        print(performance_df)

        print("\nEqual-weight portfolio performance:")
        print(equal_perf_df)

        return portfolio_result, performance_df

