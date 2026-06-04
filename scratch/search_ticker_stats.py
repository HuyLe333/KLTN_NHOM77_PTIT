import os

output_lines = []
for root, dirs, files in os.walk('.'):
    # Skip directories like .git or __pycache__
    if any(p in root for p in ['.git', '__pycache__', '.gemini', 'node_modules']):
        continue
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'ticker_stats.json' in content or 'TICKER_PATH' in content:
                    output_lines.append(f"Found in {filepath}")
            except Exception as e:
                pass

with open('scratch/search_ticker_stats.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done searching.")
