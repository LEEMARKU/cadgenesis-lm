import os

path = 'D:/Gen-AI CAD_LLM/src/cadgenesis/transformer'
with open(os.path.join(path, 'transformer.py')) as f:
    content = f.read()

# Check for key features
features = [
    'FlashAttention', 'SDPA', 'RMSNorm', 'SwiGLU', 
    'SparseMoE', 'DynamicRouting', 'GeometryAttention',
    'ConstraintAttention', 'UncertaintyAttention', 'MemoryAttention',
    'MoE', 'NAS', 'AdaptiveHeads'
]

print("Transformer core features:")
for feat in features:
    if feat in content:
        # Find line number
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if feat in line:
                print(f"  {feat}: line {i}")
                break
    else:
        print(f"  {feat}: NOT present")