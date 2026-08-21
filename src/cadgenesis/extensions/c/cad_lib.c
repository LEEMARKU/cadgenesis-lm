/**
 * C library for low-level CAD operations in CADGenesis-LM.
 * 
 * Provides fundamental CAD geometry operations that require
 * direct memory access and control, such as:
 * - Basic geometry calculations
 * - Mesh operations
 * - File I/O for CAD formats
 * - Low-level state management
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Basic geometry: compute distance between two 3D points
double cad_distance_3d(double x1, double y1, double z1, double x2, double y2, double z2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dz = z2 - z1;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

// Serialize a simple CAD token to string
// Returns allocated string (caller must free)
char* cad_token_to_string(int token_id, const char* family_name) {
    int needed = snprintf(NULL, 0, "%d_%s", token_id, family_name);
    char* result = (char*)malloc(needed + 1);
    if (result) {
        snprintf(result, needed + 1, "%d_%s", token_id, family_name);
    }
    return result;
}

// Free a string allocated by cad_token_to_string
void cad_free_string(char* str) {
    if (str) {
        free(str);
    }
}

// Check if a CAD configuration is valid
int cad_validate_config(const char* config_name, double threshold) {
    // Simple validation - in real implementation would check
    // against CAD standards and constraints
    return (threshold > 0.0 && threshold <= 1.0) ? 1 : 0;
}