import requests

res = requests.get("http://127.0.0.1:5000/api/ticker/FPT")
data = res.json()
print("Ticker:", data.get('ticker'))
print("Latest Date:", data.get('latest_date'))
print("Predict Date:", data.get('predict_date'))
print("Probability Up:", data.get('probability_up'))
print("\n--- History Data ---")
for idx, h in enumerate(data.get('history', [])):
    print(f"{idx}: Date={h.get('date')} | Close={h.get('close')} | Prob={h.get('probability_up')}")
