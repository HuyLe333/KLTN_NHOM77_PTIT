with open('backtest_dashboard/templates/backtest.html', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
output_lines = []
for i, line in enumerate(lines):
    if 'universe' in line.lower() or 'thị trường' in line.lower() or 'market' in line.lower():
        output_lines.append(f"Line {i+1}: {line.strip()[:120]}")

with open('scratch/search_backtest_html.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done searching.")
