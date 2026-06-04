with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
output_lines = []
for i, line in enumerate(lines):
    if 'onMainThresholdChange' in line or 'threshold' in line.lower():
        output_lines.append(f"Line {i+1}: {line.strip()}")

with open('scratch/search_onMainThresholdChange.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done searching.")
