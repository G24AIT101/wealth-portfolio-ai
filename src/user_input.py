class UserInput:
    def __init__(self, amount, risk, duration_months):
        self.amount = amount
        self.risk = risk.lower()
        self.duration_months = duration_months

    def risk_factor(self):
        mapping = {
            "conservative": 0.7,
            "moderate": 1.0,
            "aggressive": 1.3
        }
        return mapping[self.risk]

    def historical_years(self):
        if self.duration_months <= 12:
            return 2
        elif self.duration_months <= 24:
            return 3
        else:
            return 5
