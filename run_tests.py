import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests", "-q", "--tb=short"],
    capture_output=True,
    text=True,
    cwd="D:/Gen-AI CAD_LLM"
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")