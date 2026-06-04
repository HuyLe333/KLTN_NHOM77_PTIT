import re

files_to_check = ['app.py', 'templates/index.html']
output_lines = []

for filename in files_to_check:
    output_lines.append(f"\nChecking file: {filename}")
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if any(w in line.lower() for w in ['hose', 'hnx', 'upcom', 'sàn', 'exchange', 'thị trường']):
            output_lines.append(f"Line {i+1}: {line.strip()[:120]}")

with open('scratch/search_market_details.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done searching.")
