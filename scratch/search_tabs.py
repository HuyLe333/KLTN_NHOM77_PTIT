with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
output_lines = []
for i, line in enumerate(lines):
    if 'Chọn Mã Cổ Phiếu' in line or 'Top 10 Tín Hiệu' in line or 'Tối Ưu Danh Mục' in line:
        output_lines.append(f"Line {i+1}: {line.strip()[:150]}")

with open('scratch/search_tabs.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done searching tabs.")
