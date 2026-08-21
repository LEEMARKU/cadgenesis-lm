import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')
from cadgenesis.inference.self_correction import SelfCorrectingInference, SelfCorrectionResult
from cadgenesis.execution.geometry_validation import validate_program
from cadgenesis.confidence.risk import RiskAssessor

print('=== Self-Correction Audit ===')
print()

# 1. Detect failures
inf = SelfCorrectingInference(max_attempts=3)

# Test case 1: Valid program
result = inf.correct('make a box', ['BOX', 'NUM_10', 'EXTRUDE', 'NUM_5'])
success1 = result.success
risk1 = result.risk_score

# Test case 2: Missing base operation
result2 = inf.correct('extrude shape', ['EXTRUDE', 'NUM_10'])
success2 = result2.success
error2 = result2.error[:50] if result2.error else 'none'

# Test case 3: Missing dimension
result3 = inf.correct('extrude a shape', ['EXTRUDE'])
success3 = result3.success
risk3 = result3.risk_score

# Test case 4: All attempts fail
result4 = inf.correct('bad program', ['INVALID', 'TOKENS'])
success4 = result4.success
attempts4 = result4.attempt

# 5. Failure mode analysis
print('Failure mode analysis:')
test_cases = [
    (['EXTRUDE'], 'missing base + dimension'),
    (['SKETCH_RECT', 'EXTRUDE'], 'has base but missing dim'),
    (['BOX', 'EXTRUDE'], 'has base op but no dim'),
    (['INVALID', 'TOKENS'], 'completely malformed'),
]
for tokens, desc in test_cases:
    r = inf.correct('test', tokens)
    succ = r.success
    err = r.error[:40] if r.error else 'none'
    print(f'  {desc}: success={succ}, risk={r.risk_score:.4f}, error={err}')

# 6. Correction generation
print()
print('Correction generation:')
result = inf.correct('make a box 10x10', ['BOX', 'NUM_10', 'EXTRUDE'])
success6 = result.success
corr_tokens = result.cad_tokens if result.success else None
risk6 = result.risk_score
err6 = result.error

# 7. Re-execute corrected program
from cadgenesis.execution.geometry_validation import validate_program
if result and result.cad_tokens:
    v = validate_program(result.cad_tokens)
    reexe_valid = v
else:
    reexe_valid = False

# 7. Retry limit
retry_attempts = result.attempt if result else 0

# 8. Risk assessment
print()
print('Risk assessment:')
ra = RiskAssessor(alpha=1.0, beta=1.0, gamma=1.0)
risk_result = ra.assess(confidence=0.9, uncertainty=0.1, consequence=0.5)
risk_score1 = risk_result['risk_score']
action1 = risk_result['action']

risk_result2 = ra.assess(confidence=0.3, uncertainty=0.8, consequence=0.9)
risk_score2 = risk_result2['risk_score']
action2 = risk_result2['action']

print()
print('=== Self-Correction Audit Summary ===')
print(f'1. Valid program: success={success1}')
print(f'2. Missing base: success={success2}, error={error2[:40]}')
print(f'3. Missing dimension: success={success3}, risk={risk3:.4f}')
print(f'4. All fail: success={success4}, attempts={attempts4}')
print(f'   Failure modes:')
for tokens, desc in test_cases:
    r = inf.correct('test', tokens)
    print(f'    {desc}: success={r.success}')
print(f'6. Correction: success={success6}')
print(f'   Corrected tokens: {corr_tokens}')
print(f'  Risk: {risk6:.4f}')
print(f'  Re-execution valid: {reexe_valid}')
print(f'  Retry attempts: {retry_attempts}')
print(f'8. Risk assess: high conf/low unc: score={risk_score1:.4f}, action={action1}')
print(f'   Low conf/high unc: score={risk_score2:.4f}, action={action2}')

print()
print('=== Self-Correction Audit Complete ===')