with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
output_lines = []
for i, line in enumerate(lines):
    if 'portfolio-ticker-selects' in line:
        output_lines.append(f"Line {i+1}: {line.strip()[:120]}")

with open('scratch/search_portfolio_selects.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done searching.")
