class UserInput:
    def __init__(self, amount, risk, duration_months):
        self.amount = amount
        self.risk = risk.lower()
        self.duration_months = duration_months

    class UserInput:
    def __init__(self, amount, risk, duration_months):
        self.amount = amount
        self.risk = risk.lower()
        self.duration_months = duration_months

    def risk_factor(self):
        mapping = {
            "c": 0.7,
            "conservative": 0.7,
            "m": 1.0,
            "moderate": 1.0,
            "a": 1.3,
            "aggressive": 1.3
        }
        return mapping[self.risk]

    def historical_years(self):
        return max(3, self.duration_months // 12)

