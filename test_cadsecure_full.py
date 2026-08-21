import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')

from cadsecure import (
    sandbox_execute, sandbox_create_config, sandbox_status, sandbox_test,
    SandboxConfig, SandboxResult
)

print('=== cadsecure Module Tests ===')
print()

# Test 1: Create config
print('Test 1: Create config')
config = sandbox_create_config(30.0, 1024, None, None, 100)
print(f'  max_execution_time: {config.max_execution_time}')
print(f'  max_memory_mb: {config.max_memory_mb}')
print(f'  allowed_operations: {config.allowed_operations}')
print(f'  disallowed_operations: {config.disallowed_operations}')

# Test 2: Check operation allowed
print()
print('Test 2: Check operation allowed')
print(f'  create_box allowed: {config.is_operation_allowed("cad.create_box")}')
print(f'  create_sphere allowed: {config.is_operation_allowed("cad.create_sphere")}')
print(f'  calculate_distance allowed: {config.is_operation_allowed("cad.calculate_distance")}')
print(f'  unknown_op allowed: {config.is_operation_allowed("unknown_op")}')

# Configure with allowed operations
config_allowed = sandbox_create_config(30.0, 1024, ["cad.create_box"], None, 100)
print(f'  create_box with allowed list: {config_allowed.is_operation_allowed("cad.create_box")}')
print(f'  create_sphere with allowed list: {config_allowed.is_operation_allowed("cad.create_sphere")}')

# Configure with disallowed operations
config_disallowed = sandbox_create_config(30.0, 1024, None, ["cad.create_box"], 100)
print(f'  create_box with disallowed list: {not config_disallowed.is_operation_allowed("cad.create_box")}')
print(f'  create_sphere with disallowed list: {config_disallowed.is_operation_allowed("cad.create_sphere")}')

# Test 3: Sandbox execute - create_box
print()
print('Test 3: sandbox_execute create_box')
result = sandbox_execute("cad.create_box", {"length": 10.0, "width": 5.0, "height": 3.0}, config)
print(f'  success: {result["success"]}')
print(f'  message: {result["message"]}')
print(f'  execution_time: {result["execution_time"]}')
print(f'  memory_used_mb: {result["memory_used_mb"]}')
print(f'  output: {result["output"]}')

# Test 4: Sandbox execute - create_sphere
print()
print('Test 4: sandbox_execute create_sphere')
result = sandbox_execute("cad.create_sphere", {"radius": 5.0}, config)
print(f'  success: {result["success"]}')
print(f'  message: {result["message"]}')

# Test 5: Sandbox execute - calculate_distance
print()
print('Test 5: sandbox_execute calculate_distance')
result = sandbox_execute("cad.calculate_distance", {
    "p1_x": 0.0, "p1_y": 0.0, "p1_z": 0.0,
    "p2_x": 3.0, "p2_y": 4.0, "p2_z": 0.0
}, config)
print(f'  success: {result["success"]}')
print(f'  message: {result["message"]}')
print(f'  output: {result["output"]}')
assert abs(float(result["output"]) - 5.0) < 0.01, "Distance should be 5.0"

# Test 6: Sandbox status
print()
print('Test 6: sandbox_status')
status = sandbox_status(config)
print(f'  status keys: {list(status.keys())}')
print(f'  max_execution_time: {status["max_execution_time"]}')
print(f'  allowed_operations_count: {status["allowed_operations_count"]}')

# Test 7: Sandbox test
print()
print('Test 7: sandbox_test')
test_result = sandbox_test()
print(f'  result: {test_result}')

# Test 8: Disallowed operation
print()
print('Test 8: Disallowed operation')
disallowed_result = sandbox_execute("cad.create_box", {"length": 1.0}, config)
print(f'  disallowed success: {disallowed_result["success"]}')
assert disallowed_result["success"] == False

# Test 9: Unknown operation
print()
print('Test 9: Unknown operation')
unknown_result = sandbox_execute("cad.unknown_op", {}, config)
print(f'  unknown success: {unknown_result["success"]}')
assert unknown_result["success"] == False

# Test 10: SandboxResult class
print()
print('Test 10: SandboxResult class')
result = SandboxResult(True, "Test message", 1.5, 2.5)
print(f'  success: {result.success}')
print(f'  message: {result.message}')
print(f'  execution_time: {result.execution_time}')
print(f'  memory_used_mb: {result.memory_used_mb}')
print(f'  output: {result.output}')
print(f'  repr: {repr(result)}')

print()
print('=== All cadsecure tests passed! ===')