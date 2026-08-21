with open('D:/Gen-AI CAD_LLM/pyproject.toml', 'r') as f:
    content = f.read()

# Add pythonpath to the [tool.pytest.ini_options] section
if 'pythonpath' not in content:
    # Add after the addopts line
    content = content.replace(
        'addopts = "-v --tb=short"',
        'addopts = "-v --tb=short"\npythonpath = ["src"]'
    )
    
with open('D:/Gen-AI CAD_LLM/pyproject.toml', 'w') as f:
    f.write(content)
print('pyproject.toml updated')