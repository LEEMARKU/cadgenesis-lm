import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')

# Read the current file
with open('src/cadgenesis/inference/self_correction.py', 'r') as f:
    lines = f.readlines()

# Add debug print after line 218 (tokens = list(initial_tokens))
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if i == 217:  # line 218 (0-indexed: 217) is 'tokens = list(initial_tokens)'
        new_lines.append('\\nprint(f\"DEBUG: tokens at start = {tokens}\")\\n')

# Write back
with open('src/cadgenesis/inference/self_correction.py', 'w') as f:
    f.write(''.join(new_lines))

print('Added debug print at line 219')