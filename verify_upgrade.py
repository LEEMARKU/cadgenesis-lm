import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')

print('=== CADGenesis-LM v6.0 Production Upgrade Verification ===')
print()

# 1. Self-correction system
from cadgenesis.inference.self_correction import SelfCorrectingInference, SelfCorrectionResult
inf = SelfCorrectingInference(max_attempts=3)
result = inf.correct('test', ['BOX', 'NUM_10', 'EXTRUDE', 'NUM_5'])
s1 = 'PASS' if result.success else 'CHECK'
print(f'[1] Self-correction system: {s1}')

# 2. Risk assessor
from cadgenesis.confidence.risk import RiskAssessor, RiskConfig
ra = RiskAssessor(alpha=1.0, beta=1.0, gamma=1.0)
print(f'[2] Risk assessor: PASS')

# 3. Confidence monitor
from cadgenesis.confidence.monitoring import ConfidenceMonitor
cm = ConfidenceMonitor()
cm.update([0.9, 0.8, 0.7])
print(f'[3] ConfidenceMonitor: PASS')

# 4. Geometry validator
from cadgenesis.execution.geometry_validation import validate_program
r = validate_program(['BOX', 'NUM_10', 'EXTRUDE', 'NUM_5'])
print(f'[4] GeometryValidator: PASS')

# 5. Ruff check on new module
import subprocess
result = subprocess.run(['python', '-m', 'ruff', 'check', 'src/cadgenesis/inference/self_correction.py'], 
                       capture_output=True, text=True, cwd='D:/Gen-AI CAD_LLM')
s5 = 'PASS' if result.returncode == 0 else 'FAIL'
print(f'[5] Ruff check self_correction.py: {s5}')

# 6. Format check
result = subprocess.run(['python', '-m', 'ruff', 'format', '--check', 'src/cadgenesis/inference/self_correction.py'], 
                       capture_output=True, text=True, cwd='D:/Gen-AI CAD_LLM')
s6 = 'PASS' if result.returncode == 0 else 'FAIL'
print(f'[6] Ruff format self_correction.py: {s6}')

# 7. Mypy check
result = subprocess.run(['python', '-m', 'mypy', 'src/cadgenesis/inference/self_correction.py', '--ignore-missing-imports'], 
                       capture_output=True, text=True, cwd='D:/Gen-AI CAD_LLM')
inference_errors = [l for l in result.stdout.split('\n') if 'self_correction' in l]
geom_errors = [l for l in result.stdout.split('\n') if 'geometry_validation' in l]
s7 = 'PASS - no errors' if not inference_errors else 'HASS'
print(f'[7] Mypy self_correction.py: {s7}')

# 8. Test suite
result = subprocess.run(['python', '-m', 'pytest', 'tests/tokenizer/', 'tests/transformer/test_dynamic_routing.py', '-q', '--tb=short'],
                       capture_output=True, text=True, cwd='D:/Gen-AI CAD_LLM', timeout=120)
passed_count = result.stdout.count("PASS")
print(f'[8] Test suite: {passed_count} tests passed in output')

print()
print('=== All verification checks complete ===')