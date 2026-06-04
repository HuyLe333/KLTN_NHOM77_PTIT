import numpy as np

def simulate_greedy():
    capital_levels = [10000000, 30000000, 100000000]
    weights = {'FPT': 0.3, 'HPG': 0.25, 'VCB': 0.2, 'VNM': 0.15, 'SSI': 0.1}
    prices = {'FPT': 130000.0, 'HPG': 28000.0, 'VCB': 92000.0, 'VNM': 68000.0, 'SSI': 35000.0}
    
    for capital in capital_levels:
        print(f"\n=== Greedy Simulation for Capital: {capital:,.0f} VND ===")
        holdings_data = []
        total_allocated = 0.0
        
        # Calculate initial floor lots
        for t, w in weights.items():
            price = prices[t]
            target_val = capital * w
            qty = int(np.floor(target_val / price / 100) * 100)
            actual_value = qty * price
            holdings_data.append({
                'ticker': t,
                'price': price,
                'target_weight': w,
                'qty': qty,
                'actual_value': actual_value
            })
            total_allocated += actual_value
            
        # Greedy step: allocate remaining cash
        remaining_cash = capital - total_allocated
        
        while True:
            eligible = []
            for h in holdings_data:
                cost_of_lot = h['price'] * 100
                if cost_of_lot <= remaining_cash:
                    current_weight = h['actual_value'] / capital
                    deficit = h['target_weight'] - current_weight
                    eligible.append((deficit, cost_of_lot, h))
            
            if not eligible:
                break
                
            eligible.sort(key=lambda x: x[0], reverse=True)
            best_choice = eligible[0][2]
            best_choice['qty'] += 100
            best_choice['actual_value'] += best_choice['price'] * 100
            remaining_cash -= best_choice['price'] * 100
            total_allocated += best_choice['price'] * 100
            
        print(f"Total Allocated: {total_allocated:,.0f} VND")
        print(f"Cash Left: {remaining_cash:,.0f} VND ({(remaining_cash/capital*100):.1f}%)")
        for h in holdings_data:
            act_w = (h['actual_value'] / total_allocated * 100) if total_allocated > 0 else 0
            print(f"  {h['ticker']}: Target={h['target_weight']*100:.1f}%, Actual={act_w:.1f}%, Qty={h['qty']}, Value={h['actual_value']:,.0f} VND")

if __name__ == '__main__':
    simulate_greedy()
