#!/usr/bin/env python
# Fix pyproject.toml per-file-ignores to include rust/llvm/mlir extensions

with open('D:/Gen-AI CAD_LLM/pyproject.toml', 'r') as f:
    lines = f.readlines()

# Find the per-file-ignores section and add the new ignores
# The section starts at line with '[tool.ruff.lint.per-file-ignores]'
# And includes entries up to the end of the file

# We need to add the new ignore entries after the existing ones
# Let's find the line number of the last existing ignore entry

new_ignores = '''\n# Deliberate rust/llvm/mlir extension ignores (temporary until fully integrated)
\"src/cadgenesis/extensions/rust/*\" = [\"PERF203\", \"PERF401\"]
\"src/cadgenesis/extensions/llvm/*\" = [\"PERF203\", \"PERF401\"]
\"src/cadgenesis/extensions/mlir/*\" = [\"PERF203\", \"PERF401\"]'''

# Insert the new ignores before the closing of the section
# The section ends before the [tool.mypy] section or end of file
last_ignore_line = -1
for i, line in enumerate(lines):
    if '\"src/cadgenesis/autonomous_research/*.py\" = [\"PERF203\", \"PERF401\"]' in line:
        last_ignore_line = i

if last_ignore_line >= 0:
    # Insert the new ignores after the last existing ignore entry
    # but before any other section header
    insert_pos = last_ignore_line + 1
    
    # Check if there's a section header after this line
    section_found = False
    for i in range(insert_pos, len(lines)):
        if lines[i].startswith('[') and lines[i].endswith(']'):
            section_found = True
            insert_pos = i
            break
    
    if not section_found:
        insert_pos = len(lines)
    
    # Insert the new ignores
    for line in new_ignores.split('\\n'):
        lines.insert(insert_pos, line)
        insert_pos += 1
    
    with open('D:/Gen-AI CAD_LLM/pyproject.toml', 'w') as f:
        f.writelines(lines)
    print('Updated pyproject.toml with rust/llvm/mlir ignores')
else:
    print('Could not find the marker line in pyproject.toml')
"