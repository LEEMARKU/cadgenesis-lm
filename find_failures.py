import subprocess
import sys

result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests', '-q', '--tb=line'],
    capture_output=True, text=True, cwd='D:/Gen-AI CAD_LLM',
    timeout=120
)

# Show failed/error tests
output = result.stdout
if 'FAILED' in output:
    print('=== FAILED TESTS ===')
    for line in output.split('\n'):
        if 'FAILED' in line:
            print(line)
if 'ERROR' in output:
    print()
    print('=== ERROR TESTS ===')
    for line in output.split('\n'):
        if 'ERROR' in line:
            print(line)
if result.returncode != 0:
    print(f'\nReturn code: {result.returncode}')
else:
    print(f'\nAll tests passed (return code: 0)')