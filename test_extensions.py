from cadgenesis.extensions import check_extensions, check_cpp_extension, check_rust_extension, check_c_extension, check_llvm_extension, check_mlir_extension

print('=== Extension Availability ===')
print(f'C++/CUDA: {check_cpp_extension()}')
print(f'Rust/PyO3: {check_rust_extension()}')
print(f'C/ctypes: {check_c_extension()}')
print(f'LLVM: {check_llvm_extension()}')
print(f'MLIR: {check_mlir_extension()}')

print()
print('Full status:')
status = check_extensions()
for lang, available in status.items():
    print(f'  {lang}: {"Available" if available else "Not built (source prepared)"}')