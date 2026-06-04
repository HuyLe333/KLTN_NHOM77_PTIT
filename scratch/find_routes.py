with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

output_lines = []
for i, line in enumerate(lines):
    if '@app.route' in line:
        output_lines.append(f"Line {i+1}: {line.strip()}")
        for j in range(1, 4):
            if i + j < len(lines):
                output_lines.append(f"  {lines[i+j].strip()}")

with open('scratch/find_routes_output.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done writing routes.")
