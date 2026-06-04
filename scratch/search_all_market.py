import os

output_lines = []
for root, dirs, files in os.walk('.'):
    if any(p in root for p in ['.git', '__pycache__', '.gemini', 'node_modules']):
        continue
    for file in files:
        if file.endswith('.py') or file.endswith('.html'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'market' in content.lower() or 'thị trường' in content.lower():
                    output_lines.append(f"Found in {filepath}")
            except Exception as e:
                pass

with open('scratch/search_all_market.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done searching.")
