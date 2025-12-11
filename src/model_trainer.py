import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelTrainer:
    """
    Trains Random Forest model on engineered features.
    Uses:
        - 90% train
        - 10% test
    Computes:
        - RMSE
        - MAE
        - Direction Accuracy (up/down)
        - Feature Importances
    """

    def train(self, df):
        # Feature columns
        feature_cols = ["MA3", "MA6", "Vol3", "Vol6", "Momentum3", "Momentum6"]
        X = df[feature_cols]
        y = df["target"]

        # 90/10 chronological split (shuffle=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, shuffle=False
        )

        # Model
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Predictions
        preds = model.predict(X_test)

        # Evaluation
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)

        direction_pred = preds > 0
        direction_actual = y_test > 0
        direction_accuracy = np.mean(direction_pred == direction_actual)

        feature_importance = dict(zip(feature_cols, model.feature_importances_))

        return {
            "model": model,
            "predictions": preds,
            "actual": y_test.values,
            "rmse": rmse,
            "mae": mae,
            "direction_accuracy": direction_accuracy,
            "feature_importance": feature_importance
        }
