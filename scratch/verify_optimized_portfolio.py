import sys
import os
import json
import numpy as np

# Ensure root directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

def run_tests():
    client = app.test_client()
    
    test_cases = [
        {"capital": 10000000.0, "risk_appetite": "balanced", "selected_tickers": ["FPT", "HPG", "VCB", "VNM", "SSI"]},
        {"capital": 30000000.0, "risk_appetite": "balanced", "selected_tickers": ["FPT", "HPG", "VCB", "VNM", "SSI"]},
        {"capital": 100000000.0, "risk_appetite": "balanced", "selected_tickers": ["FPT", "HPG", "VCB", "VNM", "SSI"]},
        {"capital": 10000000.0, "risk_appetite": "balanced", "selected_tickers": ["ABB", "ASM"]},
    ]
    
    for tc in test_cases:
        capital = tc["capital"]
        print(f"\n==========================================")
        print(f"Testing Capital: {capital:,.0f} VND")
        print(f"==========================================")
        
        response = client.post('/api/portfolio/optimize', json=tc)
        
        if response.status_code != 200:
            print(f"Error! Response status code: {response.status_code}")
            print(response.get_data(as_text=True))
            continue
            
        res_data = response.get_json()
        total_allocated = res_data["total_allocated"]
        cash_left = res_data["cash_left"]
        
        print(f"Expected Return: {res_data['expected_return']}%")
        print(f"Volatility: {res_data['volatility']}%")
        print(f"Sharpe Ratio: {res_data['sharpe_ratio']}")
        print(f"Total Allocated: {total_allocated:,.0f} VND")
        print(f"Cash Left: {cash_left:,.0f} VND")
        
        # Verify safety conditions
        assert total_allocated <= capital, f"FAIL: total_allocated ({total_allocated}) exceeds capital ({capital})!"
        assert cash_left >= 0, f"FAIL: cash_left ({cash_left}) is negative!"
        assert cash_left == capital - total_allocated, "FAIL: cash_left mismatch!"
        
        print("\nHoldings:")
        for h in res_data["holdings"]:
            print(f"  {h['ticker']}: Price={h['price']:,.0f} | Qty={h['quantity']} | Val={h['allocated_amount']:,.0f} VND | Target W={h['target_weight']}% | Actual W={h['actual_weight']}%")
            print(f"    AI Explanation: {h['xai_explanation'][:100]}...")
            
        print("\nPortfolio Narrative:")
        print(res_data["portfolio_narrative"])
        print("\nSUCCESS: All math assertions passed.")

if __name__ == '__main__':
    run_tests()
