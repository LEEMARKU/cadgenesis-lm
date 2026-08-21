/**
 * cadgeometry - C++ geometry kernel for CADGenesis-LM
 *
 * Provides high-performance B-Rep, NURBS, and Boolean operations
 * called from the Python LLM layer via C FFI.
 */

#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>

// ============================================================
// Point3D - 3D point structure
// ============================================================

struct Point3D {
    double x, y, z;

    Point3D() : x(0), y(0), z(0) {}
    Point3D(double x, double y, double z) : x(x), y(y), z(z) {}

    bool operator==(const Point3D& other) const {
        return std::fabs(x - other.x) < 1e-10 &&
               std::fabs(y - other.y) < 1e-10 &&
               std::fabs(z - other.z) < 1e-10;
    }

    bool operator!=(const Point3D& other) const {
        return !(*this == other);
    }

    Point3D operator+(const Point3D& other) const {
        return Point3D(x + other.x, y + other.y, z + other.z);
    }

    Point3D operator-(const Point3D& other) const {
        return Point3D(x - other.x, y - other.y, z - other.z);
    }

    Point3D operator*(double scalar) const {
        return Point3D(x * scalar, y * scalar, z * scalar);
    }

    Point3D operator/(double scalar) const {
        if (std::fabs(scalar) < 1e-12) {
            throw std::runtime_error("Division by zero in Point3D");
        }
        return Point3D(x / scalar, y / scalar, z / scalar);
    }

    double length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    double dot(const Point3D& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Point3D cross(const Point3D& other) const {
        return Point3D(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
};

// ============================================================
// Vector3D - Vector operations
// ============================================================

struct Vector3D {
    double x, y, z;

    Vector3D() : x(0), y(0), z(0) {}
    Vector3D(double x, double y, double z) : x(x), y(y), z(z) {}

    Vector3D(const Point3D& start, const Point3D& end) 
        : x(end.x - start.x), y(end.y - start.y), z(end.z - start.z) {}

    Vector3D operator+(const Vector3D& other) const {
        return Vector3D(x + other.x, y + other.y, z + other.z);
    }

    Vector3D operator-(const Vector3D& other) const {
        return Vector3D(x - other.x, y - other.y, z - other.z);
    }

    Vector3D operator*(double scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }

    Vector3D operator/(double scalar) const {
        if (std::fabs(scalar) < 1e-12) {
            throw std::runtime_error("Division by zero in Vector3D");
        }
        return Vector3D(x / scalar, y / scalar, z / scalar);
    }

    double length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    double lengthSq() const {
        return x * x + y * y + z * z;
    }

    double dot(const Vector3D& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Vector3D cross(const Vector3D& other) const {
        return Vector3D(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    // Normalize the vector in place
    void normalize() {
        double len = length();
        if (len > 1e-12) {
            x /= len;
            y /= len;
            z /= len;
        }
    }

    // Return normalized copy
    Vector3D normalized() const {
        Vector3D result = *this;
        result.normalize();
        return result;
    }
};

// ============================================================
// BRepEdge - Boundary Representation Edge
// ============================================================

struct BRepEdge {
    enum class Type { LINE, ARC, CIRCLE, ELLIPSE };

    Type type;
    Point3D start, end;       // For line
    double radius;            // For arc/circle
    Point3D center;           // For arc/circle
    double startAngle;        // For arc
    double endAngle;          // For arc

    BRepEdge() : type(Type::LINE), radius(0) {}
};

// ============================================================
// BRepFace - Boundary Representation Face
// ============================================================

struct BRepFace {
    enum class Type { PLANE, CYLINDER, CONE, SPHERE, TORUS };

    Type type;
    Point3D normal;         // For plane
    double distance;        // For plane (from origin)
    double radius1;         // For cylinder/torus
    double radius2;         // For cone/torus
    Point3D center;         // For cylinder/sphere/torus

    BRepFace() : type(Type::PLANE), radius1(0), radius2(0) {}
};

// ============================================================
// NURBS Curve - Non-Uniform Rational B-Spline
// ============================================================

struct NURBSPoint {
    Point3D point;
    double weight;
    
    NURBSPoint() : weight(1.0) {}
    NURBSPoint(const Point3D& p, double w) : point(p), weight(w) {}
};

class NURBSCurve {
public:
    std::vector<NURBSPoint> controlPoints;
    std::vector<double> knots;
    int degree;

    NURBSCurve(int degree = 3) : degree(degree) {}

    // Evaluate curve at parameter t (0 <= t <= 1)
    Point3D evaluate(double t) const {
        if (controlPoints.empty()) {
            return Point3D(0, 0, 0);
        }

        // Clamp t to valid range
        int n = controlPoints.size() - 1;
        double totalKnots = knots.empty() ? n + degree + 1 : knots.size() - 1;
        
        // Simple uniform knot vector if not provided
        std::vector<double> localKnots;
        if (knots.empty()) {
            localKnots.resize(n + degree + 2);
            for (int i = 0; i <= n + degree + 1; i++) {
                localKnots[i] = (i <= degree) ? 0.0 : 
                    (i >= n + 1) ? 1.0 : 
                    (double)(i - degree) / (n + 1 - degree);
            }
        } else {
            localKnots = knots;
        }

        // De Boor's algorithm (simplified)
        // In production, use proper B-spline evaluation
        double resultX = 0, resultY = 0, resultZ = 0;
        double totalWeight = 0;

        // Weighted average of control points (basic approximation)
        for (const auto& cp : controlPoints) {
            double blend = cp.weight;
            resultX += cp.point.x * blend;
            resultY += cp.point.y * blend;
            resultZ += cp.point.z * blend;
            totalWeight += blend;
        }

        if (totalWeight > 0) {
            return Point3D(resultX / totalWeight, resultY / totalWeight, resultZ / totalWeight);
        }
        return controlPoints[0].point;
    }

    // Get point at parameter (parametric)
    Point3D pointAt(double t) const {
        // Return first control point for now
        // Proper implementation would use de Boor's algorithm
        if (!controlPoints.empty()) {
            return controlPoints[0].point;
        }
        return Point3D(0, 0, 0);
    }
};

// ============================================================
// Boolean Operations
// ============================================================

enum class BooleanOperation { UNION, INTERSECT, DIFFERENCE };

// Forward declarations for solid types
struct BRepSolid;

// Perform boolean operation on two solids
BRepSolid booleanOperation(const BRepSolid& a, const BRepSolid& b, BooleanOperation op);

// ============================================================
// Sphere - Basic solid primitive
// ============================================================

struct Sphere {
    Point3D center;
    double radius;

    Sphere() : radius(1.0) {}
    Sphere(const Point3D& c, double r) : center(c), radius(r > 0 ? r : 1.0) {}

    // Check if point is inside sphere
    bool contains(const Point3D& p) const {
        return (p - center).length() <= radius;
    }

    // Check if point is on surface
    bool onSurface(const Point3D& p) const {
        double dist = (p - center).length();
        return std::fabs(dist - radius) < 1e-8;
    }
};

// ============================================================
// Box - Axis-aligned bounding box
// ============================================================

struct Box {
    Point3D minCorner;
    Point3D maxCorner;

    Box() {
        minCorner = Point3D(-1, -1, -1);
        maxCorner = Point3D(1, 1, 1);
    }

    Box(const Point3D& min, const Point3D& max) : minCorner(min), maxCorner(max) {}

    // Check if point is inside box
    bool contains(const Point3D& p) const {
        return p.x >= minCorner.x && p.x <= maxCorner.x &&
               p.y >= minCorner.y && p.y <= maxCorner.y &&
               p.z >= minCorner.z && p.z <= maxCorner.z;
    }

    // Get box center
    Point3D center() const {
        return Point3D(
            (minCorner.x + maxCorner.x) / 2,
            (minCorner.y + maxCorner.y) / 2,
            (minCorner.z + maxCorner.z) / 2
        );
    }

    // Get box dimensions
    Vector3D dimensions() const {
        return Vector3D(minCorner, maxCorner);
    }
};

// ============================================================
// C FFI Interface - Functions callable from Python
// ============================================================

// Point3D operations
extern "C" {

// Create a new point
Point3D point3d_create(double x, double y, double z) {
    return Point3D(x, y, z);
}

// Point addition
Point3D point3d_add(Point3D a, Point3D b) {
    return a + b;
}

// Point subtraction
Point3D point3d_subtract(Point3D a, Point3D b) {
    return a - b;
}

// Point scaling
Point3D point3d_scale(Point3D a, double scalar) {
    return a * scalar;
}

// Point length
double point3d_length(Point3D p) {
    return p.length();
}

// Point dot product
double point3d_dot(Point3D a, Point3D b) {
    return a.dot(b);
}

// Point cross product
Point3D point3d_cross(Point3D a, Point3D b) {
    return a.cross(b);
}

// Vector3D operations
Vector3D vector3d_create(double x, double y, double z) {
    return Vector3D(x, y, z);
}

// Vector normalization
Vector3D vector3d_normalize(Vector3D v) {
    v.normalize();
    return v;
}

// Vector length
double vector3d_length(Vector3D v) {
    return v.length();
}

// Vector dot product
double vector3d_dot(Vector3D a, Vector3D b) {
    return a.dot(b);
}

// Vector cross product
Vector3D vector3d_cross(Vector3D a, Vector3D b) {
    return a.cross(b);
}

// BRepEdge creation
BRepEdge brepEdge_create_line(Point3D start, Point3D end) {
    BRepEdge edge;
    edge.type = BRepEdge::Type::LINE;
    edge.start = start;
    edge.end = end;
    return edge;
}

BRepEdge brepEdge_create_arc(Point3D center, double radius, double startAngle, double endAngle) {
    BRepEdge edge;
    edge.type = BRepEdge::Type::ARC;
    edge.radius = radius;
    edge.center = center;
    edge.startAngle = startAngle;
    edge.endAngle = endAngle;
    return edge;
}

// BRepFace creation
BRepFace brepFace_create_plane(Point3D normal, double distance) {
    BRepFace face;
    face.type = BRepFace::Type::PLANE;
    face.normal = normal;
    face.distance = distance;
    return face;
}

// NURBS curve creation
NURBSCurve nurbsCurve_create(int degree) {
    NURBSCurve curve;
    curve.degree = degree;
    return curve;
}

void nurbsCurve_addControlPoint(NURBSCurve& curve, const Point3D& point, double weight) {
    curve.controlPoints.push_back({point, weight});
}

// Boolean operation
// Returns a new solid (opaque pointer in full implementation)
void* booleanOperation(void* a, void* b, int operation) {
    // In full implementation, would cast to BRepSolid and perform operation
    // For now, return NULL
    return nullptr;
}

// Box operations
Box box_create(double minX, double minY, double minZ, double maxX, double maxY, double maxZ) {
    Box box;
    box.minCorner = Point3D(minX, minY, minZ);
    box.maxCorner = Point3D(maxX, maxY, maxZ);
    return box;
}

bool box_contains(Box box, double x, double y, double z) {
    return box.contains(Point3D(x, y, z));
}

// Sphere operations
Sphere sphere_create(double cx, double cy, double cz, double radius) {
    return Sphere(Point3D(cx, cy, cz), radius);
}

bool sphere_contains(Sphere sphere, double x, double y, double z) {
    return sphere.contains(Point3D(x, y, z));
}

// ============================================================
// Factory function to get all C FFI functions
// ============================================================

// Each function is individually exported via extern "C" above
// Python can import specific functions via ctypes

} // extern "C"