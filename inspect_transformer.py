import os

path = 'D:/Gen-AI CAD_LLM/src/cadgenesis/transformer'
with open(os.path.join(path, 'transformer.py')) as f:
    content = f.read()

# Print first 100 lines to understand structure
lines = content.split('\n')
for i, line in enumerate(lines[:120], 1):
    print(f'{i}: {line}')