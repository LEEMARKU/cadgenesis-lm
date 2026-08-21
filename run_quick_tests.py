import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", 
     "tests/tokenizer/", 
     "tests/transformer/test_dynamic_routing.py",
     "tests/platform/test_monitoring.py",
     "-q", "--tb=short"],
    capture_output=True, text=True, cwd="D:/Gen-AI CAD_LLM",
    timeout=120
)

print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("\nSTDERR:")
    print(result.stderr)
print(f"\nReturn code: {result.returncode}")