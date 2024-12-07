#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <array>
#include <iomanip>  // For setprecision
#include <chrono>  // For timing
#include <algorithm>  // For min/max

using namespace std;

using Vec3 = std::array<double, 3>;
using Triangle = std::array<Vec3, 3>;
constexpr double EPSILON = 1e-12;  // Higher precision for tolerance check

// Helper function: Cross product
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

// Helper function: Dot product
double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Helper function: Subtraction
Vec3 subtract(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// Möller-Trumbore algorithm to check if ray intersects a triangle
bool rayIntersectsTriangle(const Vec3& origin, const Vec3& direction, const Triangle& tri, double& t, double& u, double& v) {
    const Vec3& v0 = tri[0];
    const Vec3& v1 = tri[1];
    const Vec3& v2 = tri[2];

    Vec3 edge1 = subtract(v1, v0);
    Vec3 edge2 = subtract(v2, v0);

    Vec3 pvec = cross(direction, edge2);
    double det = dot(edge1, pvec);

    // if (det > -1e-8 && det < 1e-8) return false; // Parallel
    if (det > -EPSILON && det < EPSILON) return false; // Parallel check with higher precision

    double invDet = 1.0 / det;
    Vec3 tvec = subtract(origin, v0);

    u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    Vec3 qvec = cross(tvec, edge1);
    v = dot(direction, qvec) * invDet;
    if (v < 0 || u + v > 1) return false;

    t = dot(edge2, qvec) * invDet;
    return true;
}

// Check if two triangles intersect using the Möller-Trumbore algorithm
// bool trianglesIntersect(const Triangle& tri1, const Triangle& tri2) {
//     double t, u, v;

//     // Test all edges of tri1 as rays against tri2
//     for (int i = 0; i < 3; ++i) {
//         Vec3 origin = tri1[i];
//         Vec3 direction = subtract(tri1[(i + 1) % 3], origin);
//         if (rayIntersectsTriangle(origin, direction, tri2, t, u, v)) {
//             return true;
//         }
//     }

//     // Test all edges of tri2 as rays against tri1
//     for (int i = 0; i < 3; ++i) {
//         Vec3 origin = tri2[i];
//         Vec3 direction = subtract(tri2[(i + 1) % 3], origin);
//         if (rayIntersectsTriangle(origin, direction, tri1, t, u, v)) {
//             return true;
//         }
//     }

//     return false;
// }

bool trianglesIntersect(const Triangle& tri1, const Triangle& tri2, double tolerance = 1e-9) {
    // Bounding box check
    Vec3 min1 = {std::min({tri1[0][0], tri1[1][0], tri1[2][0]}),
                 std::min({tri1[0][1], tri1[1][1], tri1[2][1]}),
                 std::min({tri1[0][2], tri1[1][2], tri1[2][2]})};
    Vec3 max1 = {std::max({tri1[0][0], tri1[1][0], tri1[2][0]}),
                 std::max({tri1[0][1], tri1[1][1], tri1[2][1]}),
                 std::max({tri1[0][2], tri1[1][2], tri1[2][2]})};
    Vec3 min2 = {std::min({tri2[0][0], tri2[1][0], tri2[2][0]}),
                 std::min({tri2[0][1], tri2[1][1], tri2[2][1]}),
                 std::min({tri2[0][2], tri2[1][2], tri2[2][2]})};
    Vec3 max2 = {std::max({tri2[0][0], tri2[1][0], tri2[2][0]}),
                 std::max({tri2[0][1], tri2[1][1], tri2[2][1]}),
                 std::max({tri2[0][2], tri2[1][2], tri2[2][2]})};

    if (max1[0] < min2[0] - tolerance || min1[0] > max2[0] + tolerance ||
        max1[1] < min2[1] - tolerance || min1[1] > max2[1] + tolerance ||
        max1[2] < min2[2] - tolerance || min1[2] > max2[2] + tolerance) {
        return false;  // Bounding boxes don't intersect
    }

    double t, u, v;

    // Test all edges of tri1 as rays against tri2
    for (int i = 0; i < 3; ++i) {
        Vec3 origin = tri1[i];
        Vec3 direction = subtract(tri1[(i + 1) % 3], origin);
        if (rayIntersectsTriangle(origin, direction, tri2, t, u, v)) {
            if (t >= 0 && t <= 1 + tolerance) return true;
        }
    }

    // Test all edges of tri2 as rays against tri1
    for (int i = 0; i < 3; ++i) {
        Vec3 origin = tri2[i];
        Vec3 direction = subtract(tri2[(i + 1) % 3], origin);
        if (rayIntersectsTriangle(origin, direction, tri1, t, u, v)) {
            if (t >= 0 && t <= 1 + tolerance) return true;
        }
    }

    return false;
}

// Function to read triangles from an OFF file
std::vector<Triangle> readTrianglesFromOFF(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    std::string header;
    infile >> header;
    if (header != "OFF") {
        std::cerr << "Not a valid OFF file." << std::endl;
        exit(1);
    }

    int numVertices, numFaces, numEdges;
    infile >> numVertices >> numFaces >> numEdges;

    std::vector<Vec3> vertices(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        infile >> vertices[i][0] >> vertices[i][1] >> vertices[i][2];
    }

    std::vector<Triangle> triangles;
    for (int i = 0; i < numFaces; ++i) {
        int vertexCount, v1, v2, v3;
        infile >> vertexCount >> v1 >> v2 >> v3;
        if (vertexCount == 3) {
            triangles.push_back({vertices[v1], vertices[v2], vertices[v3]});
        }
    }

    return triangles;
}

int main() {
    auto start  = std::chrono::high_resolution_clock::now();
    // auto triangles1 = readTrianglesFromOFF("VH_F_renal_pyramid_L_a.off");
    // auto triangles2 = readTrianglesFromOFF("VH_F_renal_pyramid_L_b.off");

    // auto triangles1 = readTrianglesFromOFF("VH_M_splenic_flexure_of_colon.off");
    // auto triangles2 = readTrianglesFromOFF("VH_M_transverse_colon.off");

    auto triangles1 = readTrianglesFromOFF("VH_M_ileocecal_valve.off");
    auto triangles2 = readTrianglesFromOFF("VH_M_transverse_colon.off");



    bool intersect = false;
    for (const auto& tri1 : triangles1) {
        for (const auto& tri2 : triangles2) {
            if (trianglesIntersect(tri1, tri2)) {
                intersect = true;
                // print the intersecting triangles
                std::cout << "Triangle 1: " << tri1[0][0] << " " << tri1[0][1] << " " << tri1[0][2] << std::endl;
                std::cout << "Triangle 2: " << tri2[0][0] << " " << tri2[0][1] << " " << tri2[0][2] << std::endl;
                break;
            }
        }
        if (intersect) break;
    }

    if (intersect) {
        std::cout << "Triangles intersect!" << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cout << "Time taken: " << elapsed.count() << endl;

    } else {
        std::cout << "Triangles do not intersect." << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cout << "Time taken: " << elapsed.count() << endl;
    }

    return 0;
}
