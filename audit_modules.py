import os
base = 'D:/Gen-AI CAD_LLM/src/cadgenesis'

# Quick check: which modules have actual code vs just docstrings
for d in ['confidence', 'optimization', 'distillation', 'continual_learning']:
    full = os.path.join(base, d)
    print(f'=== {d} ===')
    for f in sorted(os.listdir(full)):
        if not f.endswith('.py') or f == '__init__.py':
            continue
        path = os.path.join(full, f)
        with open(path) as fh:
            lines = fh.readlines()
        # Count non-empty, non-comment lines after docstrings
        code_lines = []
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
            code_lines.append(line)
        non_doc = len(code_lines)
        # Check if function/class definitions exist in first 10 lines
        first_10 = code_lines[:10]
        has_def = any(l.strip().startswith('def ') or l.strip().startswith('class ') for l in first_10)
        print(f'  {f}: {non_doc} code lines, has def/class: {has_def}')