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

        # --------------------------------------------------
        # Step 1: Fetch historical stock data
        # --------------------------------------------------
        years = self.user.historical_years()
        fetcher = DataFetcher(self.user.risk)
        stock_data = fetcher.fetch(years)
        
        if not stock_data:
            print("Error: No data fetched. Exiting.")
            return None, None
            
        print("Data fetched:", list(stock_data.keys()))

        # --------------------------------------------------
        # Step 2: Feature Engineering
        # --------------------------------------------------
        fe = FeatureEngineer()
        processed_data = {
            t: fe.add_features(df.copy())
            for t, df in stock_data.items()
        }

        # --------------------------------------------------
        # Step 3: Train models & predict returns
        # --------------------------------------------------
        trainer = ModelTrainer()
        predicted_returns = {}
        feature_cols = ["MA3", "MA6", "Vol3", "Vol6", "Momentum3", "Momentum6"]

        print("\nModel Performance Summary:")
        for t, df in processed_data.items():
            # FIX: Split data into Training (History) and Prediction (Live)
            
            # 1. Training Data: Must have valid targets (drop NaNs)
            train_df = df.dropna()
            
            if len(train_df) < 12:
                print(f"Skipping {t}: Not enough data.")
                continue

            # 2. Live Data: The very last row (has features, but NaN target)
            latest_features = df.iloc[[-1]][feature_cols]

            # Train the model
            results = trainer.train(train_df)
            
            # Predict Future (Next Month) using the trained model on the LATEST data
            future_pred = results["model"].predict(latest_features)[0]
            predicted_returns[t] = future_pred

            print(f"\n📌 Stock: {t}")
            print(f"   RMSE (Test Set): {results['rmse']:.4f}")
            print(f"   Next Month Pred: {future_pred:.2%}")
            print(f"   Top Feature: {max(results['feature_importance'], key=results['feature_importance'].get)}")

        # --------------------------------------------------
        # Apply risk factor (C / M / A)
        # --------------------------------------------------
        risk_factor = self.user.risk_factor()
        print(f"\nApplying Risk Factor: {risk_factor}x")
        
        adjusted_predictions = {
            t: r * risk_factor
            for t, r in predicted_returns.items()
        }

        # --------------------------------------------------
        # Step 4: Portfolio Optimization
        # --------------------------------------------------
        optimizer = PortfolioOptimizer()
        portfolio_result = optimizer.optimize(
            stock_data,
            adjusted_predictions,
            self.user.amount
        )

        print("\n=== Portfolio Optimization Completed ===")
        print("Optimized Weights:")
        for t, w in portfolio_result["weights"].items():
            if w > 0: print(f"  {t}: {w:.1%}")
            
        print("\nFinal Share Allocation (Shopping List):")
        for t, s in portfolio_result["allocation"].items():
             print(f"  {t}: {s} shares")
             
        print(f"Leftover cash: ₹{portfolio_result['leftover_cash']:.2f}")

        # --------------------------------------------------
        # Step 5: Validation & Backtesting
        # --------------------------------------------------
        validator = Validator()
        performance = validator.backtest_portfolio(
            stock_data,
            portfolio_result["weights"]
        )
        equal_perf = validator.equal_weight_comparison(stock_data)

        # Convert dicts to 1-row DataFrames
        performance_df = pd.DataFrame([performance])
        equal_perf_df = pd.DataFrame([equal_perf])

        print("\n=== Validation Results (Historical Backtest) ===")
        print("\nModel-based portfolio performance:")
        print(performance_df.to_string(index=False))

        print("\nEqual-weight portfolio performance:")
        print(equal_perf_df.to_string(index=False))

        return portfolio_result, performance_df

# Example usage for Colab:
# advisor = WealthAdvisorAI(amount=100000, risk="moderate", duration_months=12)
# portfolio, stats = advisor.run()