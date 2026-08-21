import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')

# Check distillation modules
import os
distillation_dir = 'D:/Gen-AI CAD_LLM/src/cadgenesis/distillation'
for f in sorted(os.listdir(distillation_dir)):
    if not f.endswith('.py') or f == '__init__.py':
        continue
    path = os.path.join(distillation_dir, f)
    with open(path) as fh:
        content = fh.read()
    # Check for actual code (def or class) outside docstrings
    lines = content.split('\n')
    has_code = False
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if stripped.startswith('def ') or stripped.startswith('class '):
            has_code = True
            break
    # Also check if it's just a docstring
    has_stub = 'This module is a stub' in content.lower()
    print(f'{f}: has_code={has_code}, stub={has_stub}')