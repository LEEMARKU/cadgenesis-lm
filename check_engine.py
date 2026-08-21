import os

path = 'D:/Gen-AI CAD_LLM/src/cadgenesis/inference/engine.py'
with open(path) as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines[:80], 1):
    print(f'{i}: {line}')