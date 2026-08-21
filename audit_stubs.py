import sys
import os

sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')

# Check all top-level directories for code vs stub status
base = 'D:/Gen-AI CAD_LLM/src/cadgenesis'
stub_count_by_mod = {}
total_py_by_mod = {}

for d in sorted(os.listdir(base)):
    full = os.path.join(base, d)
    if not os.path.isdir(full):
        continue
    py_files = [f for f in os.listdir(full) if f.endswith('.py') and f != '__init__.py']
    if not py_files:
        continue
    total_py_by_mod[d] = len(py_files)
    
    stub_count = 0
    for f in py_files:
        path = os.path.join(full, f)
        try:
            with open(path, encoding='utf-8', errors='replace') as fh:
                content = fh.read()
            has_stub = 'This module is a stub' in content.lower()
            if has_stub:
                stub_count += 1
        except Exception as e:
            print(f'Error reading {path}: {e}')
            stub_count += 1  # count as stub if can't read
    stub_count_by_mod[d] = stub_count

print('=== Stub Module Analysis ===')
print(f'{"Module":<25} {"Total Py":>12} {"Stub":>8} {"Code":>10} {"Status"}')
print('-' * 75)
for mod in sorted(stub_count_by_mod.keys()):
    total = total_py_by_mod[mod]
    stubs = stub_count_by_mod[mod]
    code = total - stubs
    if stubs > 0 and code > 0:
        status = f'{stubs} stub + {code} code'
    elif stubs > 0:
        status = f'{stubs} stub only'
    else:
        status = 'fully implemented'
    print(f'{mod:<25} {total:>12} {stubs:>8} {code:>10} {status}')

print()
total_stubs = sum(stub_count_by_mod.values())
total_code = sum(t - s for t, s in total_py_by_mod.items() if t > 0)
print(f'TOTAL: {total_stubs} modules with stub docs out of {sum(total_py_by_mod.values())} total .py files')
print(f'Modules with code: {total_code}')