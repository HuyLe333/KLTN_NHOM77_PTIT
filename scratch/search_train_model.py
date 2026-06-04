with open('train_model.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

output_lines = []
for i, line in enumerate(lines):
    if 'ticker_stats' in line or 'ticker_stats.json' in line:
        output_lines.append(f"Line {i+1}: {line.strip()[:120]}")

with open('scratch/search_train_model_output.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(output_lines))
print("Done writing.")
