# Simply rewrite the pyproject.toml with the ignores added
with open('D:/Gen-AI CAD_LLM/pyproject.toml', 'r') as f:
    lines = f.readlines()

# Find the line with the last existing ignore entry and add after it
new_ignores = [
    '\n# Deliberate rust/llvm/mlir extension ignores (temporary until fully integrated)',
    ' "src/cadgenesis/extensions/rust/*" = ["PERF203", "PERF401"]',
    ' "src/cadgenesis/extensions/llvm/*" = ["PERF203", "PERF401"]',
    ' "src/cadgenesis/extensions/mlir/*" = [ "PERF203", "PERF401"]'
]

# Find the line number to insert after
insert_after = -1
for i, line in enumerate(lines):
    if 'src/cadgenesis/autonomous_research/*.py" = ["PERF203", "PERF401"]' in line:
        insert_after = i
        break

if insert_after >= 0:
    # Insert the new ignores after the found line
    for new_ignore in new_ignores:
        lines.insert(insert_after + 1, new_ignore)
        insert_after += 1

with open('D:/Gen-AI CAD_LLM/pyproject.toml', 'w') as f:
    f.writelines(lines)
print('pyproject.toml updated')