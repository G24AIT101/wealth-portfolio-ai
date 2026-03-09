"""
Wealth Portfolio AI - Test Script
This script runs the wealth advisory AI pipeline with sample inputs.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main_pipeline import WealthAdvisorAI

def main():
    """
    Main function to run the wealth advisory AI with test parameters.
    """
    print("=" * 60)
    print("Wealth Portfolio AI - Test Run")
    print("=" * 60)
    
    # Test parameters
    investment_amount = 100000  # ₹1,00,000
    risk_profile = "moderate"   # Options: "conservative", "moderate", "aggressive"
    duration_months = 24        # 2 years
    
    print(f"\nInvestment Amount: ₹{investment_amount:,}")
    print(f"Risk Profile: {risk_profile}")
    print(f"Duration: {duration_months} months")
    print("\n" + "-" * 60)
    
    # Create and run the advisor
    advisor = WealthAdvisorAI(
        amount=investment_amount,
        risk=risk_profile,
        duration_months=duration_months,
        feature_mode="baseline"  # Options: "baseline" or "risk_aware"
    )
    
    try:
        portfolio, _ = advisor.run()
        
        print("\n" + "=" * 60)
        print("PORTFOLIO ALLOCATION RESULTS")
        print("=" * 60)
        
        print(f"\nPortfolio Weights:")
        for ticker, weight in portfolio["weights"].items():
            if weight > 0:
                print(f"  {ticker}: {weight:.2%}")
        
        print(f"\nStock Allocation (Number of Shares):")
        total_invested = 0
        for ticker, shares in portfolio["allocation"].items():
            print(f"  {ticker}: {shares} shares")
            # Note: We'd need price data to calculate exact amount invested
        
        print(f"\nLeftover Cash: ₹{portfolio['leftover_cash']:,.2f}")
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
