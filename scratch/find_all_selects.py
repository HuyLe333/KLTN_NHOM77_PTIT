with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
output_lines = []
for i, line in enumerate(lines):
    if '<select' in line or 'onchange' in line:
        output_lines.append(f"Line {i+1}: {line.strip()}")

with open('scratch/find_all_selects_output.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done writing selects.")
