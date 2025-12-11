import pandas as pd

class FeatureEngineer:
    def add_features(self, df):
        df["return"] = df["Close"].pct_change()

        df["MA3"] = df["Close"].rolling(3).mean()
        df["MA6"] = df["Close"].rolling(6).mean()

        df["Vol3"] = df["return"].rolling(3).std()
        df["Vol6"] = df["return"].rolling(6).std()

        df["Momentum3"] = df["Close"] - df["Close"].shift(3)
        df["Momentum6"] = df["Close"] - df["Close"].shift(6)

        df["target"] = df["return"].shift(-1)
        
        return df.dropna()
