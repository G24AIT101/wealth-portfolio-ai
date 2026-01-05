import yfinance as yf

class DataFetcher:
    """
    Fetches historical data for a fixed NIFTY-based universe.
    """

    def __init__(self):
        self.stocks = [
            "RELIANCE.NS",
            "INFY.NS",
            "TCS.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "SBIN.NS",
            "LT.NS",
            "ITC.NS",
            "MARUTI.NS",
            "AXISBANK.NS"
        ]

    def fetch(self, years):
        data = {}
        for stock in self.stocks:
            df = yf.download(stock, period=f"{years}y", auto_adjust=True)
            df.dropna(inplace=True)
            data[stock] = df
        return data
