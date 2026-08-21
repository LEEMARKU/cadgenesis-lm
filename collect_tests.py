import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests", "--collect-only"],
    capture_output=True,
    text=True,
    cwd="D:/Gen-AI CAD_LLM",
    timeout=60
)

print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("\nSTDERR:")
    print(result.stderr)
print(f"\nReturn code: {result.returncode}")