import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')
from cadgenesis.confidence.risk import RiskAssessor, RiskConfig
from cadgenesis.confidence.monitoring import ConfidenceMonitor

print('=== Confidence/Risk System Audit ===')
print()

# 1. RiskAssessor
ra = RiskAssessor(alpha=1.0, beta=1.0, gamma=1.0)
print('1. RiskAssessor: initialized')

# 2. Risk.assess()
risk = ra.assess(confidence=0.9, uncertainty=0.1, consequence=0.5)
print(f'2. Risk.assess(): risk_score={risk["risk_score"]:.4f}, action={risk["action"]}')

# 3. Risk with different values
risk2 = ra.assess(confidence=0.3, uncertainty=0.8, consequence=0.9)
print(f'4. Risk.assess(low conf/high unc): risk_score={risk2["risk_score"]:.4f}, action={risk2["action"]}')

# 3. RiskConfig
rc = RiskConfig()
print(f'3. RiskConfig: alpha={rc.uncertainty_penalty}, beta={rc.consequence_weight}')

rc2 = RiskConfig(uncertainty_penalty=0.2, consequence_weight=0.3)
ra2 = RiskAssessor(alpha=rc2.uncertainty_penalty, beta=rc2.consequence_weight)
print(f'4. Custom RiskConfig: alpha={rc2.uncertainty_penalty}, beta={rc2.consequence_weight}')

# 5. ConfidenceMonitor
cm = ConfidenceMonitor()
cm.update([0.9, 0.8, 0.7, 0.6, 0.5])
summary = cm.summary()
print(f'5. ConfidenceMonitor: count={summary["count"]}, mean={summary["mean"]:.4f}')

# 6. Update with tensor
import torch
cm2 = ConfidenceMonitor()
cm2.update(torch.tensor([0.95, 0.85, 0.75]))
summary2 = cm2.summary()
print(f'6. ConfidenceMonitor(tensor): count={summary2["count"]}, mean={summary2["mean"]:.4f}')

print()
print('=== Confidence/Risk System Audit Complete ===')