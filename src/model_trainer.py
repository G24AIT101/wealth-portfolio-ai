import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelTrainer:
    """
    Trains Random Forest model on engineered features.
    Feature-agnostic: works for baseline and risk-aware modes.
    """

    def train(self, df):
        # Clean data
        df_clean = df.dropna()

        # Target
        y = df_clean["target"]

        # Use all available features except target
        X = df_clean.drop(columns=["target"])

        # 90/10 chronological split (no leakage)
        split_idx = int(0.9 * len(df_clean))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Model (same for all experiments)
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Predictions
        preds = model.predict(X_test)

        # Evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        direction_pred = preds > 0
        direction_actual = y_test > 0
        direction_accuracy = np.mean(direction_pred == direction_actual)

        feature_importance = dict(
            zip(X.columns, model.feature_importances_)
        )

        return {
            "model": model,
            "predictions": preds,
            "actual": y_test.values,
            "rmse": rmse,
            "mae": mae,
            "direction_accuracy": direction_accuracy,
            "feature_importance": feature_importance
        }
