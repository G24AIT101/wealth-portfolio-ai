import numpy as np
import pandas as pd

class FeatureEngineer:
    """
    Creates baseline and risk-aware feature sets.
    """

    def add_features(self, df, mode="baseline"):
        df = df.copy()

        # Returns
        df["Return"] = df["Close"].pct_change()

        # Moving averages
        df["MA3"] = df["Close"].rolling(3).mean()
        df["MA6"] = df["Close"].rolling(6).mean()

        # Momentum
        df["Momentum3"] = df["Close"] / df["Close"].shift(3) - 1
        df["Momentum6"] = df["Close"] / df["Close"].shift(6) - 1

        if mode == "risk_aware":
            # Volatility
            df["Vol3"] = df["Return"].rolling(3).std()
            df["Vol6"] = df["Return"].rolling(6).std()

            # Downside deviation
            downside = df["Return"].copy()
            downside[downside > 0] = 0
            df["DownsideVol"] = downside.rolling(6).std()

            # Drawdown proxy
            rolling_max = df["Close"].rolling(6).max()
            df["Drawdown"] = (df["Close"] - rolling_max) / rolling_max

            # Interaction term
            df["MomVolInteraction"] = df["Momentum3"] * df["Vol3"]

        # Target: next-period return
        df["Target"] = df["Return"].shift(-1)

        df.dropna(inplace=True)
        return df
