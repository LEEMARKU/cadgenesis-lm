import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')

# Read the current file
with open('src/cadgenesis/inference/self_correction.py', 'r') as f:
    content = f.read()

# Add debug print at the start of the correct method
lines = content.split('\n')
new_lines = []
in_correct = False
for i, line in enumerate(lines):
    if line.strip().startswith('def correct'):
        in_correct = True
    if in_correct:
        new_lines.append(line)
        if i == 199:  # line number where def correct starts
            new_lines.append('')
            new_lines.append('# DEBUG: About to enter attempt loop')
            new_lines.append('print(f"DEBUG: attempt={attempt}, tokens={tokens}")')
    else:
        new_lines.append(line)

# Write back
with open('src/cadgenesis/inference/self_correction.py', 'w') as f:
    f.write('\n'.join(new_lines))

print('Added debug prints')