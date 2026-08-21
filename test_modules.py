import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')

results = []

# Test some of the modules without tests
try:
    from cadgenesis.adapters.lora import LoRALinear
    results.append('PASS: cadgenesis.adapters.lora')
except Exception as e:
    results.append(f'FAIL: cadgenesis.adapters.lora - {e}')

try:
    from cadgenesis.adapters.manager import AdapterMetadata
    results.append('PASS: cadgenesis.adapters.manager')
except Exception as e:
    results.append(f'FAIL: cadgenesis.adapters.manager - {e}')

try:
    from cadgenesis.confidence.confidence import ConfidenceEngine
    results.append('PASS: cadgenesis.confidence.confidence')
except Exception as e:
    results.append(f'FAIL: cadgenesis.confidence.confidence - {e}')

try:
    from cadgenesis.serving.api import GenerateRequest
    results.append('PASS: cadgenesis.serving.api')
except Exception as e:
    results.append(f'FAIL: cadgenesis.serving.api - {e}')

try:
    from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
    results.append('PASS: cadgenesis.transformer.geometry_transformer')
except Exception as e:
    results.append(f'FAIL: cadgenesis.transformer.geometry_transformer - {e}')

# Write results
with open('test_results.txt', 'w') as f:
    f.write('\n'.join(results))

print('Test results written to test_results.txt')
for r in results:
    print(r)