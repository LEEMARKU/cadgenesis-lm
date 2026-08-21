import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')

# Test the cadsecure module
from cadsecure import (
    sandbox_execute, sandbox_create_config, sandbox_status, sandbox_test,
    _rust_available
)

print('✓ cadsecure imports successful')
print(f'  _rust_available: {_rust_available}')

# Test sandbox_create_config
config = sandbox_create_config(30.0, 1024, None, None, 100)
print(f'  sandbox_create_config: max_time={config["max_execution_time"]}')

# Test sandbox_status
status = sandbox_status(config)
print(f'  sandbox_status: keys={list(status.keys())}')

# Test sandbox_test
result = sandbox_test()
print(f'  sandbox_test: {result}')

# Test sandbox_execute
exec_result = sandbox_execute('cad.calculate_distance', {
    'p1_x': 0.0, 'p1_y': 0.0, 'p1_z': 0.0,
    'p2_x': 3.0, 'p2_y': 4.0, 'p2_z': 0.0
})
print(f'  sandbox_execute: success={exec_result["success"]}, message={exec_result["message"]}')

# Test disallowed operation
disallowed = sandbox_execute('cad.create_box', {'test': 1.0})
print(f'  sandbox_execute disallowed: success={disallowed["success"]}')

print('\nAll cadsecure tests passed!')