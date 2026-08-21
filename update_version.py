#!/usr/bin/env python
import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')

# Read the __init__.py and update version
with open('D:/Gen-AI CAD_LLM/src/cadgenesis/__init__.py', 'r') as f:
    content = f.read()

# Update version
content = content.replace('__version__ = "6.1.0"', '__version__ = "8.0.0"')

with open('D:/Gen-AI CAD_LLM/src/cadgenesis/__init__.py', 'w') as f:
    f.write(content)

print('Version updated to 8.0.0')