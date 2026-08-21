import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')
from cadgenesis.datasets.cad_program_synth import build_synthetic_records, write_synthetic_jsonl, NUM_MAX, token_coverage
print(f'NUM_MAX = {NUM_MAX}')

# Generate some records
records = build_synthetic_records(10, seed=42)
print(f'Generated {len(records)} records')
for i in range(len(records)):
    r = records[i]
    txt = r['text']
    cad = r['cad']
    print(f'  {i}: text={txt}, cad={cad}')

# Write to JSONL
path = write_synthetic_jsonl('test_records.jsonl', 5, seed=123, progress=False)
print(f'Wrote to {path}')

# Verify token coverage
tokens = token_coverage(records)
print(f'Unique tokens: {tokens}')