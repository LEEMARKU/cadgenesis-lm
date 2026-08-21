import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')

# More detailed monkey-patch
from cadgenesis.inference import self_correction
original_correct = self_correction.SelfCorrectingInference.correct

def debug_correct(self, prompt, initial_tokens):
    print(f'Starting correct() with max_attempts={self.max_attempts}')
    tokens = list(initial_tokens)
    best_result = None
    best_risk = 1.0
    
    for attempt in range(1, self.max_attempts + 1):
        print(f'  Attempt {attempt}: tokens before = {tokens}')
        
        if attempt > 1 and best_result is not None:
            # Try repair
            repaired = self._attempt_repair(tokens, prompt, attempt)
            print(f'    Repair result: {repaired is not None}')
            if repaired is not None:
                tokens = repaired
                print(f'    Tokens after repair: {tokens}')
        
        is_valid, reason = self._validate_program(tokens)
        print(f'    validate: is_valid={is_valid}, reason={reason}')
        
        risk_score = self._assess_risk(tokens, prompt)
        print(f'    risk_score: {risk_score}')
        
        if is_valid:
            best_result = self_correction.SelfCorrectionResult(
                success=True,
                attempt=attempt,
                cad_tokens=list(tokens),
                cad_text=prompt,
                risk_score=risk_score,
                error=None,
            )
            print(f'    -> Set best_result success=True, attempt={attempt}')
            if attempt < self.max_attempts:
                print(f'    -> continuing (not last attempt)')
                continue
        else:
            if best_result is None or risk_score < best_risk:
                best_result = self_correction.SelfCorrectionResult(
                    success=False,
                    attempt=attempt,
                    cad_tokens=list(tokens),
                    cad_text=prompt,
                    risk_score=risk_score,
                    error=reason,
                )
                best_risk = risk_score
                print(f'    -> Set best_result success=False, risk={risk_score}')
    
    final_success = best_result.success if best_result else 'NONE'
    print(f'Final best_result: success={final_success}')
    return best_result

self_correction.SelfCorrectingInference.correct = debug_correct

from cadgenesis.confidence.risk import RiskConfig
inf = self_correction.SelfCorrectingInference(max_attempts=3)

result = inf.correct('test', ['BOX', 'NUM_10', 'EXTRUDE', 'NUM_5'])
print(f'Final result: success={result.success}')