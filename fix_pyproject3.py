#!/usr/bin/env python
with open('D:/Gen-AI CAD_LLM/pyproject.toml', 'r') as f:
    lines = f.readlines()

# Find the line to insert after
insert_after = -1
for i, line in enumerate(lines):
    if '"src/cadgenesis/autonomous_research/*.py" = ["PERF203", "PERF401"]' in line:
        insert_after = i
        break

if insert_after >= 0:
    # New ignore entries to add
    new_entries = [
        '\n# Deliberate rust/llvm/mlir extension ignores (temporary until fully integrated)',
        ' "src/cadgenesis/extensions/rust/*" = ["PERF203", "PERF401"]',
        ' "src/cadgenesis/extensions/llvm/*" = [ "PERF203", "PERF401"]',
        ' "src/cadgenesis/extensions/mlir/*" = [ "PERF203", "PERF401"]'
    ]
    for entry in new_entries:
        lines.insert(insert_after + 1, entry)
        insert_after += 1

with open('D:/Gen-AI CAD_LLM/pyproject.toml', 'w') as f:
    f.writelines(lines)
print('pyproject.toml updated')
"