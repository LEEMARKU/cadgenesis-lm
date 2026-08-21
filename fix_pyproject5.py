#!/usr/bin/env python
with open('D:/Gen-AI CAD_LLM/pyproject.toml', 'r') as f:
    lines = f.readlines()

# Remove the trailing triple-quote lines
while lines and lines[-1].strip() == '\"\"\"':
    lines.pop()

with open('D:/Gen-AI CAD_LLM/pyproject.toml', 'w') as f:
    f.writelines(lines)
print('Fixed')
"