import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')
from cadgenesis.inference.self_correction import SelfCorrectingInference, SelfCorrectionResult
from cadgenesis.execution.geometry_validation import validate_program

inf = SelfCorrectingInference(max_attempts=3)
prompt = 'test'
initial_tokens = ['BOX', 'NUM_10', 'EXTRUDE', 'NUM_5']

# Manual trace (identical to the actual correct() method logic)
best_result = None
best_risk = 1.0
tokens = list(initial_tokens)

for attempt in range(1, inf.max_attempts + 1):
    print(f'\\nAttempt {attempt}:')
    print(f'  tokens at start: {tokens}')
    
    if attempt > 1 and best_result is not None:
        repair_tokens = inf._attempt_repair(tokens, prompt, attempt)
        print(f'  repair result: {repair_tokens is not None}')
        if repair_tokens is not None:
            tokens = repair_tokens
            print(f'  tokens after repair: {tokens}')
    
    is_valid, reason = inf._validate_program(tokens)
    print(f'  validate: is_valid={is_valid}, reason={reason}')
    
    risk_score = inf._assess_risk(tokens)
    print(f'  risk_score: {risk_score}')
    
    if is_valid:
        best_result = SelfCorrectionResult(
            success=True, attempt=attempt,
            cad_tokens=list(tokens), cad_text=prompt,
            risk_score=risk_score, error=None)
        print(f'  -> best_result success=True, attempt={attempt}')
        if attempt < inf.max_attempts:
            print(f'  -> continue (not last attempt)')
            continue
    else:
        if best_result is None or risk_score < best_risk:
            best_result = SelfCorrectionResult(
                success=False, attempt=attempt,
                cad_tokens=list(tokens), cad_text=prompt,
                risk_score=risk_score, error=reason)
            best_risk = risk_score
            print(f'  -> best_result success=False, risk={risk_score}')

print(f'\\nManual trace Final: success={best_result.success if best_result else None}')
if best_result:
    print(f'  tokens={best_result.cad_tokens}')
    print(f'  risk={best_result.risk_score}')

# Now call the actual method
result = inf.correct(prompt, initial_tokens)
print(f'\\nActual method result: success={result.success}, attempt={result.attempt}')
print(f'  tokens={result.cad_tokens}, risk={result.risk_score}')
PYEOF