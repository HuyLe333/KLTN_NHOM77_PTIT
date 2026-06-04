with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
output_lines = []
for i, line in enumerate(lines):
    if 'high-pred-ticker' in line or 'low-pred-ticker' in line:
        output_lines.append(f"Line {i+1}: {line.strip()[:120]}")

with open('scratch/search_pred_styles.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done searching.")
