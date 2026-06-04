import requests
import json

def test_endpoints():
    base_url = "http://localhost:5000"
    
    print("Testing /api/tickers?threshold_offset=0.0")
    try:
        r = requests.get(f"{base_url}/api/tickers?threshold_offset=0.0")
        if r.status_code == 200:
            tickers = r.json()
            print(f"Loaded {len(tickers)} tickers.")
            # Print sample ticker with medium/low/high predictability
            sample = tickers[0] if tickers else {}
            print(f"Sample ticker: {sample.get('ticker')}")
            print(f"  Accuracy: {sample.get('test_accuracy')}")
            print(f"  Predictability: {sample.get('predictability')}")
        else:
            print(f"Failed with status: {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting /api/tickers?threshold_offset=0.15")
    try:
        r = requests.get(f"{base_url}/api/tickers?threshold_offset=0.15")
        if r.status_code == 200:
            tickers = r.json()
            print(f"Loaded {len(tickers)} tickers after filtering.")
            # Print sample ticker
            if tickers:
                for sample in tickers[:3]:
                    print(f"Ticker: {sample.get('ticker')}")
                    print(f"  Accuracy: {sample.get('test_accuracy')}")
                    print(f"  Predictability: {sample.get('predictability')}")
        else:
            print(f"Failed with status: {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting /api/ticker/FPT?threshold_offset=0.15")
    try:
        r = requests.get(f"{base_url}/api/ticker/FPT?threshold_offset=0.15")
        if r.status_code == 200:
            data = r.json()
            print(f"FPT dynamic details at 0.15 offset:")
            print(f"  Accuracy: {data.get('test_accuracy')}")
            print(f"  Predictability: {data.get('predictability')}")
        else:
            print(f"Failed with status: {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    test_endpoints()
