import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')
import cadgenesis.datasets.cad_program_synth as cps

print('Templates:', list(cps._TEMPLATES.keys()) if hasattr(cps, '_TEMPLATES') else 'N/A')
print('__all__:', cps.__all__[:10])

# Check templates
if hasattr(cps, '_TEMPLATES'):
    for name, t in cps._TEMPLATES.items():
        print(f'Template: {name}')
        desc = t.get('description', 'N/A')
        print(f'  Description: {desc[:80] if desc else "N/A"}')
        tokens = t.get('tokens', 'N/A')
        print(f'  Tokens: {tokens}')
        print()
PYEOF