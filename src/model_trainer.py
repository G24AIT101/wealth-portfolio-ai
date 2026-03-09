import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# For LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

class ModelTrainer:
    """
    Trains ML models on engineered features.
    Supports: Random Forest ('rf'), Gradient Boosting ('gb'), and LSTM ('lstm').
    """
    def __init__(self, model_type="rf"):
        self.model_type = model_type

    def _get_model(self):
        if self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gb":
            return GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        else:
            raise ValueError("Unsupported model_type for traditional ML. Use 'rf' or 'gb'.")

    def train(self, df):
        # Clean data
        df_clean = df.dropna()

        # Target and Features
        y = df_clean["target"]
        X = df_clean.drop(columns=["target"])

        # 90/10 chronological split (Strictly no leakage)
        split_idx = int(0.9 * len(df_clean))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        feature_importance = {}

        # --- LSTM ROUTINE (Requires Strict Scaling) ---
        if self.model_type == "lstm":
            # SCALING: Fit ONLY on X_train to prevent look-ahead bias
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Reshape for LSTM [samples, time steps, features]
            X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
            
            model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, verbose=0)
            preds = model.predict(X_test_reshaped, verbose=0).flatten()
            
        # --- TRADITIONAL ML ROUTINE (RF / GB) ---
        else:
            model = self._get_model()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if hasattr(model, "feature_importances_"):
                feature_importance = dict(zip(X.columns, model.feature_importances_))

        # Evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        direction_pred = preds > 0
        direction_actual = y_test > 0
        direction_accuracy = np.mean(direction_pred == direction_actual)

        return {
            "model": model,
            "predictions": preds,
            "actual": y_test.values,
            "rmse": rmse,
            "mae": mae,
            "direction_accuracy": direction_accuracy,
            "feature_importance": feature_importance
        }