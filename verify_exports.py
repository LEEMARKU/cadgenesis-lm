import os
import re

base = 'D:/Gen-AI CAD_LLM/src/cadgenesis'
pkgs = ['cad/benchmarks', 'cli', 'continual_learning', 'optimization', 'serving']

for pkg in pkgs:
    full = os.path.join(base, pkg)
    init = os.path.join(full, '__init__.py')
    with open(init) as f:
        content = f.read()
    
    has_all = '__all__' in content
    has_imports = bool(re.search(r'from\s+\S+\s+import', content))
    
    all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.DOTALL)
    exports = []
    if all_match:
        exports = [x.strip().strip('"').strip("'") for x in all_match.group(1).split(',')]
    
    print(pkg + ':')
    print('  __all__ present: ' + str(has_all))
    print('  has imports/exports: ' + str(has_imports))
    print('  exports (first 5): ' + str(exports[:5]) + ('...' if len(exports) > 5 else ''))
    print()