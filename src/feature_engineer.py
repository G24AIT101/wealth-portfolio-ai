import numpy as np
import pandas as pd

class FeatureEngineer:
    def __init__(self, feature_mode="baseline"):
        self.feature_mode = feature_mode

    def add_features(self, df):
        df = df.copy()

        df["Return"] = df["Close"].pct_change()
        df["MA3"] = df["Close"].rolling(3).mean()
        df["MA6"] = df["Close"].rolling(6).mean()
        df["Momentum3"] = df["Close"] / df["Close"].shift(3) - 1
        df["Momentum6"] = df["Close"] / df["Close"].shift(6) - 1

        if self.feature_mode == "risk_aware":
            df["Vol3"] = df["Return"].rolling(3).std()
            df["Vol6"] = df["Return"].rolling(6).std()

            downside = df["Return"].copy()
            downside[downside > 0] = 0
            df["DownsideVol"] = downside.rolling(6).std()

            rolling_max = df["Close"].rolling(6).max()
            df["Drawdown"] = (df["Close"] - rolling_max) / rolling_max

            df["MomVolInteraction"] = df["Momentum3"] * df["Vol3"]

        df["target"] = df["Return"].shift(-1)
        df.dropna(inplace=True)
        return df