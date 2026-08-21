#!/usr/bin/env python
# Fix ruff TOML parse error by adding rust/llvm/mlir extensions to ignore list

with open('D:/Gen-AI CAD_LLM/pyproject.toml', 'r') as f:
    content = f.read()

# Add rust/llvm/mlir extensions to the per-file-ignores
# This will make ruff skip these directories
new_ignore_section = '''"src/cadgenesis/extensions/rust/*" = []
"src/cadgenesis/extensions/llvm/*" = []
"src/cadgenesis/extensions/mlir/*" = []'''

# Find the per-file-ignores section and add the new ignores
# Looking for the existing ignore section
if 'per-file-ignores' in content:
    # Add the new ignores after the existing ones
    # The existing section ends with \"src/cadgenesis/tokenizer/versioning.py\" = [\"PERF203\"]
    marker = '\n\"src/cadgenesis/tokenizer/versioning.py\" = [\"PERF203\"]'
    if marker in content:
        content = content.replace(marker, marker + new_ignore_section)
        with open('D:/Gen-AI CAD_LLM/pyproject.toml', 'w') as f:
            f.write(content)
        print('Updated pyproject.toml to ignore rust/llvm/mlir extensions')
    else:
        print('Could not find the marker in pyproject.toml')
else:
    print('Could not find per-file-ignores section')