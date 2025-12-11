import yfinance as yf

class DataFetcher:
    def __init__(self, risk_profile):
        self.risk_profile = risk_profile.lower()

    def select_stocks(self):
        if self.risk_profile == "conservative":
            return ["HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS", "^NSEI"]
        elif self.risk_profile == "moderate":
            return ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "^NSEI"]
        else:
            return ["TATAMOTORS.NS", "MARUTI.NS", "RELIANCE.NS", "INFY.NS"]

    def fetch(self, years):
        stocks = self.select_stocks()
        data = {}
        for s in stocks:
            df = yf.download(s, period=f"{years}y")
            data[s] = df
        return data
