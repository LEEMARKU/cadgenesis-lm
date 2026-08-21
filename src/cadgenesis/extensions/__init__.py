"""cadgenesis.extensions
=====================
Integration layer for multi-language computation in CADGenesis-LM.

This module provides access to compute extensions written in C++, CUDA,
Rust, C, LLVM, and MLIR, enabling performance-critical operations
while keeping the main research code in Python.

Extension Status Overview:
  - C++/CUDA: Source prepared in src/cadgenesis/extensions/cpp/ and cuda/
    - Build: python -m torch.utils.cpp_extension (requires MSVC/BuildTools + CUDA)
  - Rust: PyO3 bindings in src/cadgenesis/extensions/rust/
    - Build: maturin build (requires rustc + maturin + pyo3)
  - C: ctypes/cffi wrapper in src/cadgenesis/extensions/c/
    - Build: C shared library (.so/.dll/.dylib), loaded automatically
  - LLVM: llvmlite integration in src/cadgenesis/extensions/llvm/
    - Build: pip install llvmlite (requires LLVM development headers)
  - MLIR: MLIR Python integration in src/cadgenesis/extensions/mlir/
    - Build: pip install mlir (requires MLIR Python package)

All extensions are designed to be imported conditionally with graceful
fallback to pure Python implementations when the respective
build toolchains are not available.

Typical usage:
    from cadgenesis.extensions import check_extensions, cpp_ext, c_ext
    
    # Check which extensions are available
    status = check_extensions()
    print(status)
    
    # Use extensions that are available, fall back otherwise
    try:
        result = cpp_ext.attention_forward_cuda(q, k, v, mask, scale)
    except (TypeError, NameError):
        result = pure_python_attention(q, k, v, mask, scale)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# C++/CUDA Extensions
# ---------------------------------------------------------------------------

# Import C++ extension if source is available (needs build toolchain)
try:
    from cadgenesis.extensions.cpp_ext import cuda_ext as cpp_ext
    cpp_available = cpp_ext is not None
except ImportError:
    cpp_ext = None
    cpp_available = False
    # Note: Source files exist at src/cadgenesis/extensions/cpp/ and cuda/
    # Build with: python -m torch.utils.cpp_extension

# ---------------------------------------------------------------------------
# Rust Extension (PyO3)
# ---------------------------------------------------------------------------

try:
    from cadgenesis.extensions.rust_ext import cadgenesis_rust_ext as rust_ext
    rust_available = rust_ext is not None
except ImportError:
    rust_ext = None
    rust_available = False
    # Rust source prepared at src/cadgenesis/extensions/rust/
    # Build with: maturin build (requires rustc + maturin + pyo3)

# ---------------------------------------------------------------------------
# C Extension (ctypes/cffi)
# ---------------------------------------------------------------------------

try:
    from cadgenesis.extensions.c.cffi_wrapper import (
        cad_distance_3d,
        cad_token_to_string,
        cad_validate_config,
        compute_bounding_box,
    )
    c_available = True
except ImportError:
    cad_distance_3d = None  # type: ignore
    cad_token_to_string = None  # type: ignore
    cad_validate_config = None  # type: ignore
    compute_bounding_box = None  # type: ignore
    c_available = False

# ---------------------------------------------------------------------------
# LLVM Extension
# ---------------------------------------------------------------------------

try:
    from cadgenesis.extensions.llvm.llvm_integration import (
        create_transformation_ir,
        run_optimization_passes,
        TransformerOptimizationPipeline,
        check_llvm_availability,
    )
    llvm_available = True
except ImportError:
    create_transformation_ir = None  # type: ignore
    run_optimization_passes = None  # type: ignore
    TransformerOptimizationPipeline = None  # type: ignore
    check_llvm_availability = None  # type: ignore
    llvm_available = False

# ---------------------------------------------------------------------------
# MLIR Extension
# ---------------------------------------------------------------------------

try:
    from cadgenesis.extensions.mlir.mlir_integration import (
        HAS_MLIR_PYTHON,
        MLIRContext,
        MLIRModule,
        CADDialect,
        GeometryDialect,
        MLIRPassManager,
        lower_mlir_to_llvm,
        lower_mlir_to_cuda_ir,
        check_mlir_availability,
    )
    mlir_available = True
except ImportError:
    HAS_MLIR_PYTHON = False
    MLIRContext = None  # type: ignore
    MLIRModule = None  # type: ignore
    CADDialect = None  # type: ignore
    GeometryDialect = None  # type: ignore
    MLIRPassManager = None  # type: ignore
    lower_mlir_to_llvm = None  # type: ignore
    lower_mlir_to_cuda_ir = None  # type: ignore
    check_mlir_availability = None  # type: ignore
    mlir_available = False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Build the __all__ list dynamically based on what's available
_all_items = [
    "check_extensions",
    "cpp_available",
    "rust_available",
    "c_available",
    "llvm_available",
    "mlir_available",
]

# Add individual extension references if available
if cpp_available:
    _all_items.append("cpp_ext")
if rust_available:
    _all_items.append("rust_ext")
if c_available:
    _all_items.extend([
        "cad_distance_3d", "cad_token_to_string", "cad_validate_config",
        "compute_bounding_box",
    ])
if llvm_available:
    _all_items.extend([
        "create_transformation_ir", "run_optimization_passes",
        "TransformerOptimizationPipeline", "check_llvm_availability",
    ])
if mlir_available:
    _all_items.extend([
        "HAS_MLIR_PYTHON", "MLIRContext", "MLIRModule",
        "CADDialect", "GeometryDialect", "MLIRPassManager",
        "lower_mlir_to_llvm", "lower_mlir_to_cuda_ir",
        "check_mlir_availability",
    ])

__all__ = _all_items

# ---------------------------------------------------------------------------
# Availability check function
# ---------------------------------------------------------------------------

def check_extensions() -> dict[str, bool]:
    """Check which language extensions are available/configured.
    
    Returns a dictionary mapping language names to boolean availability.
    Extensions that are not built will return False, but the source
    files are prepared and can be built when the toolchain is available.
    
    Returns:
        dict[str, bool]: Availability status for each language extension
    """
    return {
        "cpp": cpp_available,
        "rust": rust_available,
        "c": c_available,
        "llvm": llvm_available,
        "mlir": mlir_available,
    }

# Convenience functions
def check_cpp_extension() -> bool:
    """Check if C++/CUDA extension is available."""
    return cpp_available

def check_rust_extension() -> bool:
    """Check if Rust/PyO3 extension is available."""
    return rust_available

def check_c_extension() -> bool:
    """Check if C/ctypes extension is available."""
    return c_available

def check_llvm_extension() -> bool:
    """Check if LLVM integration is available."""
    return llvm_available

def check_mlir_extension() -> bool:
    """Check if MLIR integration is available."""
    return mlir_available


# Extension availability dictionary
extension_availability = {
    "cpp": cpp_available,
    "rust": rust_available,
    "c": c_available,
    "llvm": llvm_available,
    "mlir": mlir_available,
}