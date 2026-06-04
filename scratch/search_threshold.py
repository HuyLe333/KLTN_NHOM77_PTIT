with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
output_lines = []
for i, line in enumerate(lines):
    if any(w in line.lower() for w in ['threshold', 'confidence', 'lọc', 'ngưỡng', 'tin cậy']):
        output_lines.append(f"Line {i+1}: {line.strip()[:120]}")

with open('scratch/search_threshold.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done searching.")
