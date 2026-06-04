import re

with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
output_lines = []
output_lines.append("Searching for 'universe'...")
for i, line in enumerate(lines):
    if 'universe' in line.lower():
        output_lines.append(f"Line {i+1}: {line.strip()[:100]}")

output_lines.append("\nSearching for '<select' or markets...")
for i, line in enumerate(lines):
    if '<select' in line.lower() or 'change' in line.lower() or 'filter' in line.lower() or 'thị trường' in line.lower() or 'sàn' in line.lower():
        if any(w in line.lower() for w in ['thị trường', 'sàn', 'exchange', 'market', 'hose', 'hnx', 'upcom']):
            output_lines.append(f"Line {i+1}: {line.strip()[:100]}")

with open('scratch/search_index_html_output.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done writing search output.")
