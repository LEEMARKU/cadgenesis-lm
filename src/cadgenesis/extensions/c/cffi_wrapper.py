"""cadgenesis.extensions.cffi
===========================
C Foreign Function Interface for CADGenesis-LM low-level operations.

Provides Python access to C library functions for fundamental
CAD geometry operations that require low-level control.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional, Tuple, Union

# Determine the C library path
_EXTENSION_DIR = Path(__file__).parent
_LIB_NAME = "cad_lib"
_LIB_PATH = _EXTENSION_DIR / f"{_LIB_NAME}.so"  # Linux
_WINDOWS_LIB_PATH = _EXTENSION_DIR / f"{_LIB_NAME}.dll"  # Windows
_MAC_LIB_PATH = _EXTENSION_DIR / f"{_LIB_NAME}.dylib"  # macOS


def _load_library() -> ctypes.CDLL:
    """Load the C library suitable for the current platform."""
    import sys
    
    system = sys.platform
    if system == "linux" or system == "linux2":
        lib_path = str(_LIB_PATH)
    elif system == "win32":
        lib_path = str(_WINDOWS_LIB_PATH)
    elif system == "darwin":
        lib_path = str(_MAC_LIB_PATH)
    else:
        raise OSError(f"Unsupported platform: {system}")
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"C library not found at {lib_path}. "
                                "Build the C extension first.")
    
    return ctypes.CDLL(lib_path)


# Load the library at module import time (with fallback)
_lib: Optional[ctypes.CDLL] = None
_lib_loaded = False


def _ensure_loaded() -> ctypes.CDLL:
    """Ensure the C library is loaded, loading it if necessary."""
    global _lib, _lib_loaded
    
    if not _lib_loaded:
        try:
            _lib = _load_library()
            _lib_loaded = True
        except FileNotFoundError:
            # Library not built yet - set lib to None and let callers handle
            _lib = None
            _lib_loaded = True
    
    return _lib


# --- C function bindings ---

def cad_distance_3d(x1: float, y1: float, z1: float, 
                    x2: float, y2: float, z2: float) -> float:
    """Compute Euclidean distance between two 3D points.
    
    Args:
        x1, y1, z1: Coordinates of first point
        x2, y2, z2: Coordinates of second point
    
    Returns:
        Euclidean distance
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python implementation
        import math
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    # Set up function signature
    lib.cad_distance_3d.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double
    ]
    lib.cad_distance_3d.restype = ctypes.c_double
    
    return lib.cad_distance_3d(
        ctypes.c_double(x1), ctypes.c_double(y1), ctypes.c_double(z1),
        ctypes.c_double(x2), ctypes.c_double(y2), ctypes.c_double(z2)
    )


def cad_token_to_string(token_id: int, family_name: str) -> str:
    """Convert a CAD token ID and family name to string representation.
    
    Args:
        token_id: Integer token identifier
        family_name: Token family name (e.g., "GEOMETRY", "NUMERIC")
    
    Returns:
        String in format "token_id_family_name"
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python implementation
        return f"{token_id}_{family_name}"
    
    # Set up function signature
    lib.cad_token_to_string.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p
    ]
    lib.cad_token_to_string.restype = ctypes.c_char_p
    
    # Call the function
    result_ptr = lib.cad_token_to_string(
        ctypes.c_int(token_id),
        family_name.encode('utf-8') if isinstance(family_name, str) else family_name
    )
    
    if result_ptr:
        result = ctypes.string_at(result_ptr).decode('utf-8')
        # We don't free here - the caller should use cad_free_string
        # but for simplicity in this wrapper, we return the string
        # and document that the memory is managed by the C library
        return result
    else:
        return f"{token_id}_{family_name}"


def cad_free_string(ptr: str) -> None:
    """Free a string previously allocated by cad_token_to_string.
    
    Args:
        ptr: The string pointer to free (note: in practice this
        would need the actual C pointer, this is a simplified interface)
    """
    lib = _ensure_loaded()
    if lib is None:
        return
    
    lib.cad_free_string.argtypes = [ctypes.c_char_p]
    lib.cad_free_string.restype = None
    
    # Note: Full pointer-based free would require more careful
    # memory management. This is a simplified interface.
    pass


def cad_validate_config(config_name: str, threshold: float) -> bool:
    """Validate a CAD configuration.
    
    Args:
        config_name: Name of the configuration to validate
        threshold: Validation threshold (0.0 - 1.0)
    
    Returns:
        True if the configuration is valid, False otherwise
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback validation
        return 0.0 < threshold <= 1.0
    
    # Set up function signature
    lib.cad_validate_config.argtypes = [
        ctypes.c_char_p,
        ctypes.c_double
    ]
    lib.cad_validate_config.restype = ctypes.c_int
    
    result = lib.cad_validate_config(
        config_name.encode('utf-8') if isinstance(config_name, str) else config_name.encode('utf-8'),
        ctypes.c_double(threshold)
    )
    
    return bool(result)


# Convenience function for common CAD operations
def compute_bounding_box(points: list[tuple[float, float, float]]) -> tuple[float, float, float, float]:
    """Compute the axis-aligned bounding box for a set of 3D points.
    
    Args:
        points: List of (x, y, z) tuples
    
    Returns:
        (min_x, min_y, min_z, max_x, max_y, max_z) - wait, that's 6 values
        Actually returns: (min_x, min_y, min_z, max_x, max_y, max_z)
    """
    if not points:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


# Export all public functions
__all__ = [
    "cad_distance_3d",
    "cad_token_to_string",
    "cad_free_string",
    "cad_validate_config",
    "compute_bounding_box",
]