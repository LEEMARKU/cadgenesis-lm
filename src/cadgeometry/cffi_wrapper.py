"""cadgeometry C FFI Wrapper
==========================
Python ctypes wrapper for CAD geometry C++ kernels.

Provides Python access to C++ geometry operations for CADGenesis-LM.
"""

from __future__ import annotations

import ctypes
import os
import sys

# Determine the C library path
_EXTENSION_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_NAME = "cadgeometry_cffi"
_LIB_PATH_UNIX = os.path.join(_EXTENSION_DIR, f"{_LIB_NAME}.so")
_LIB_PATH_WINDOWS = os.path.join(_EXTENSION_DIR, f"{_LIB_NAME}.dll")
_LIB_PATH_MACOS = os.path.join(_EXTENSION_DIR, f"{_LIB_NAME}.dylib")


class CADGeometryError(Exception):
    """Raised when a CAD geometry operation fails."""
    pass


# ============================================================
# Library Loading
# ============================================================

def _load_library() -> ctypes.CDLL:
    """Load the CAD geometry C library suitable for the current platform."""

    system = sys.platform
    if system == "linux" or system == "linux2":
        lib_path = _LIB_PATH_UNIX
    elif system == "win32":
        lib_path = _LIB_PATH_WINDOWS
    elif system == "darwin":
        lib_path = _LIB_PATH_MACOS
    else:
        raise OSError(f"Unsupported platform: {system}. Cannot load CAD geometry library.")

    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"CAD geometry C library not found at {lib_path}. "
            "Build the C extension first."
        )

    return ctypes.CDLL(lib_path)


# Global library instance
_lib: ctypes.CDLL | None = None
_lib_loaded = False


def _ensure_loaded() -> ctypes.CDLL:
    """Ensure the C library is loaded, loading it if necessary."""
    global _lib, _lib_loaded

    if not _lib_loaded:
        try:
            _lib = _load_library()
            _lib_loaded = True
        except FileNotFoundError:
            _lib = None
            _lib_loaded = True

    return _lib


# ============================================================
# Point3D Functions
# ============================================================

def point3d_create(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Create a 3D point.
    
    Args:
        x, y, z: Coordinates
    
    Returns:
        Tuple of (x, y, z)
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python
        return (x, y, z)

    # Set up function signature
    lib.point3d_create.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.point3d_create.restype = ctypes.c_void_p  # Returns Point3D struct pointer

    # Call the function
    ptr = lib.point3d_create(ctypes.c_double(x), ctypes.c_double(y), ctypes.c_double(z))
    
    # For simplicity, we'll just return the coordinates
    # In a full implementation, would marshal the struct back
    return (x, y, z)


def point3d_add(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> tuple[float, float, float]:
    """Add two 3D points.
    
    Args:
        p1, p2: Points as (x, y, z) tuples
    
    Returns:
        Tuple of the sum point (x, y, z)
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python
        return (p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2])

    # Set up function signature
    lib.point3d_add.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)
    ]
    lib.point3d_add.restype = None

    # Call the function
    p1_arr = (ctypes.c_double * 3)(*p1)
    p2_arr = (ctypes.c_double * 3)(*p2)
    result_arr = (ctypes.c_double * 3)()
    
    lib.point3d_add(p1_arr, p2_arr, result_arr)
    
    return (result_arr[0], result_arr[1], result_arr[2])


def point3d_subtract(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> tuple[float, float, float]:
    """Subtract two 3D points.
    
    Args:
        p1, p2: Points as (x, y, z) tuples
    
    Returns:
        Tuple of the difference point (x, y, z)
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python
        return (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

    # Set up function signature
    lib.point3d_subtract.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)
    ]
    lib.point3d_subtract.restype = None

    # Call the function
    p1_arr = (ctypes.c_double * 3)(*p1)
    p2_arr = (ctypes.c_double * 3)(*p2)
    result_arr = (ctypes.c_double * 3)()
    
    lib.point3d_subtract(p1_arr, p2_arr, result_arr)
    
    return (result_arr[0], result_arr[1], result_arr[2])


def point3d_scale(p: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    """Scale a 3D point by a scalar.
    
    Args:
        p: Point as (x, y, z) tuple
        scalar: Scaling factor
    
    Returns:
        Tuple of the scaled point (x, y, z)
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python
        return (p[0] * scalar, p[1] * scalar, p[2] * scalar)

    # Set up function signature
    lib.point3d_scale.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double)
    ]
    lib.point3d_scale.restype = None

    # Call the function
    p_arr = (ctypes.c_double * 3)(*p)
    result_arr = (ctypes.c_double * 3)()
    
    lib.point3d_scale(p_arr, ctypes.c_double(scalar), result_arr)
    
    return (result_arr[0], result_arr[1], result_arr[2])


def point3d_length(p: tuple[float, float, float]) -> float:
    """Compute the length of a 3D point vector from origin.
    
    Args:
        p: Point as (x, y, z) tuple
    
    Returns:
        Euclidean distance from origin
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python
        return (p[0]**2 + p[1]**2 + p[2]**2) ** 0.5

    # Set up function signature
    lib.point3d_length.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,  # dummy for struct size
        ctypes.POINTER(ctypes.c_double)
    ]
    lib.point3d_length.restype = ctypes.c_double

    # Call the function
    p_arr = (ctypes.c_double * 3)(*p)
    result = lib.point3d_length(p_arr, ctypes.c_int(3), None)
    
    return result


def point3d_dot(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    """Compute dot product of two 3D points.
    
    Args:
        p1, p2: Points as (x, y, z) tuples
    
    Returns:
        Dot product
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python
        return p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]

    # Set up function signature
    lib.point3d_dot.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)
    ]
    lib.point3d_dot.restype = ctypes.c_double

    # Call the function
    p1_arr = (ctypes.c_double * 3)(*p1)
    p2_arr = (ctypes.c_double * 3)(*p2)
    result = lib.point3d_dot(p1_arr, p2_arr, None, None)
    
    return result


def point3d_cross(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> tuple[float, float, float]:
    """Compute cross product of two 3D points.
    
    Args:
        p1, p2: Points as (x, y, z) tuples
    
    Returns:
        Cross product as (x, y, z) tuple
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python
        return (
            p1[1]*p2[2] - p1[2]*p2[1],
            p1[2]*p2[0] - p1[0]*p2[2],
            p1[0]*p2[1] - p1[1]*p2[0]
        )

    # Set up function signature
    lib.point3d_cross.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)
    ]
    lib.point3d_cross.restype = None

    # Call the function
    p1_arr = (ctypes.c_double * 3)(*p1)
    p2_arr = (ctypes.c_double * 3)(*p2)
    result_arr = (ctypes.c_double * 3)()
    
    lib.point3d_cross(p1_arr, p2_arr, result_arr, None)
    
    return (result_arr[0], result_arr[1], result_arr[2])


# ============================================================
# Vector3D Functions
# ============================================================

def vector3d_create(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Create a 3D vector.
    
    Args:
        x, y, z: Vector components
    
    Returns:
        Tuple of (x, y, z)
    """
    lib = _ensure_loaded()
    if lib is None:
        return (x, y, z)
    return (x, y, z)


def vector3d_normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalize a 3D vector.
    
    Args:
        v: Vector as (x, y, z) tuple
    
    Returns:
        Normalized vector (x, y, z)
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python
        length = (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5
        if length < 1e-12:
            return (0.0, 0.0, 0.0)
        return (v[0]/length, v[1]/length, v[2]/length)

    # Set up function signature - normalized returns new vector
    lib.vector3d_normalize.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)
    ]
    lib.vector3d_normalize.restype = None

    # Call the function
    v_arr = (ctypes.c_double * 3)(*v)
    result_arr = (ctypes.c_double * 3)()
    
    lib.vector3d_normalize(v_arr, result_arr)
    
    return (result_arr[0], result_arr[1], result_arr[2])


def vector3d_length(v: tuple[float, float, float]) -> float:
    """Compute vector length.
    
    Args:
        v: Vector as (x, y, z) tuple
    
    Returns:
        Length of the vector
    """
    lib = _ensure_loaded()
    if lib is None:
        return (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5
    
    lib.vector3d_length.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.vector3d_length.restype = ctypes.c_double
    
    v_arr = (ctypes.c_double * 3)(*v)
    result = lib.vector3d_length(v_arr, None)
    
    return result


def vector3d_dot(v1: tuple[float, float, float], v2: tuple[float, float, float]) -> float:
    """Compute dot product of two vectors.
    
    Args:
        v1, v2: Vectors as (x, y, z) tuples
    
    Returns:
        Dot product
    """
    lib = _ensure_loaded()
    if lib is None:
        return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    
    lib.vector3d_dot.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.vector3d_dot.restype = ctypes.c_double
    
    v1_arr = (ctypes.c_double * 3)(*v1)
    v2_arr = (ctypes.c_double * 3)(*v2)
    result = lib.vector3d_dot(v1_arr, v2_arr, None, None)
    
    return result


def vector3d_cross(v1: tuple[float, float, float], v2: tuple[float, float, float]) -> tuple[float, float, float]:
    """Compute cross product of two vectors.
    
    Args:
        v1, v2: Vectors as (x, y, z) tuples
    
    Returns:
        Cross product as (x, y, z) tuple
    """
    lib = _ensure_loaded()
    if lib is None:
        return (
            v1[1]*v2[2] - v1[2]*v2[1],
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0]
        )
    
    lib.vector3d_cross.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.vector3d_cross.restype = None
    
    v1_arr = (ctypes.c_double * 3)(*v1)
    v2_arr = (ctypes.c_double * 3)(*v2)
    result_arr = (ctypes.c_double * 3)()
    
    lib.vector3d_cross(v1_arr, v2_arr, result_arr, None)
    
    return (result_arr[0], result_arr[1], result_arr[2])


# ============================================================
# Box Operations
# ============================================================

def box_create(minX: float, minY: float, minZ: float, maxX: float, maxY: float, maxZ: float) -> dict:
    """Create a bounding box.
    
    Args:
        minX, minY, minZ: Minimum corner coordinates
        maxX, maxY, maxZ: Maximum corner coordinates
    
    Returns:
        Dictionary with box properties
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python
        return {
            "minCorner": (minX, minY, minZ),
            "maxCorner": (maxX, maxY, maxZ),
            "center": ((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2),
            "dimensions": (maxX - minX, maxY - minY, maxZ - minZ)
        }

    # For now, use Python fallback since C library box functions are limited
    # In full implementation, would call lib.box_create etc.
    return {
        "minCorner": (minX, minY, minZ),
        "maxCorner": (maxX, maxY, maxZ),
        "center": ((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2),
        "dimensions": (maxX - minX, maxY - minY, maxZ - minZ)
    }


def box_contains(box: dict, x: float, y: float, z: float) -> bool:
    """Check if a point is inside a bounding box.
    
    Args:
        box: Box dictionary from box_create()
        x, y, z: Point coordinates
    
    Returns:
        True if point is inside the box
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python
        return (box["minCorner"][0] <= x <= box["maxCorner"][0] and
                box["minCorner"][1] <= y <= box["maxCorner"][1] and
                box["minCorner"][2] <= z <= box["maxCorner"][2])
    
    # For now, use Python fallback
    return (box["minCorner"][0] <= x <= box["maxCorner"][0] and
            box["minCorner"][1] <= y <= box["maxCorner"][1] and
            box["minCorner"][2] <= z <= box["maxCorner"][2])


# ============================================================
# Sphere Operations
# ============================================================

def sphere_create(cx: float, cy: float, cz: float, radius: float) -> dict:
    """Create a sphere.
    
    Args:
        cx, cy, cz: Center coordinates
        radius: Sphere radius
    
    Returns:
        Dictionary with sphere properties
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python
        return {
            "center": (cx, cy, cz),
            "radius": radius,
            "contains_origin_dist": (cx**2 + cy**2 + cz**2) ** 0.5 - radius if (cx**2 + cy**2 + cz**2) ** 0.5 > radius else 0
        }
    
    # For now, use Python fallback
    return {
        "center": (cx, cy, cz),
        "radius": radius
    }


def sphere_contains(sphere: dict, x: float, y: float, z: float) -> bool:
    """Check if a point is inside a sphere.
    
    Args:
        sphere: Sphere dictionary from sphere_create()
        x, y, z: Point coordinates
    
    Returns:
        True if point is inside the sphere
    """
    lib = _ensure_loaded()
    if lib is None:
        # Fallback: pure Python
        import math
        dist = math.sqrt((x - sphere["center"][0])**2 + (y - sphere["center"][1])**2 + (z - sphere["center"][2])**2)
        return dist <= sphere["radius"]
    
    # For now, use Python fallback
    import math
    dist = math.sqrt((x - sphere["center"][0])**2 + (y - sphere["center"][1])**2 + (z - sphere["center"][2])**2)
    return dist <= sphere["radius"]


# ============================================================
# Public API
# ============================================================

__all__ = [
    "CADGeometryError",
    "box_contains",
    "box_create",
    "point3d_add",
    "point3d_create",
    "point3d_cross",
    "point3d_dot",
    "point3d_length",
    "point3d_scale",
    "point3d_subtract",
    "sphere_contains",
    "sphere_create",
    "vector3d_create",
    "vector3d_cross",
    "vector3d_dot",
    "vector3d_length",
    "vector3d_normalize"
]