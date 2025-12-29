import yfinance as yf

class DataFetcher:
    def __init__(self, risk_profile):
        self.risk_profile = risk_profile.lower()

    def select_stocks(self):
        # FIX: Handle short codes (c/m/a) and full names
        # FIX: Replace ^NSEI (Index) with NIFTYBEES.NS (ETF) so it's tradeable
        if self.risk_profile in ["conservative", "c"]:
            return ["HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS", "NIFTYBEES.NS"]
        elif self.risk_profile in ["moderate", "m"]:
            return ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "NIFTYBEES.NS"]
        else:
            # Aggressive or default
            return ["TATAMOTORS.NS", "MARUTI.NS", "RELIANCE.NS", "INFY.NS"]

    def fetch(self, years):
        stocks = self.select_stocks()
        data = {}
        print(f"Fetching data for: {stocks} (Period: {years}y)")
        for s in stocks:
            try:
                # auto_adjust=True handles splits/dividends better
                df = yf.download(s, period=f"{years}y", auto_adjust=True)
                if not df.empty:
                    data[s] = df
                else:
                    print(f"Warning: No data found for {s}")
            except Exception as e:
                print(f"Error fetching {s}: {e}")
        return data