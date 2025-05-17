#include <iostream>
#include <vector>
#include <fstream>
#include <array>
#include <iomanip>
#include <chrono>
#include <algorithm>

using namespace std;

using Vec3 = std::array<double, 3>;
using Triangle = std::array<Vec3, 3>;
constexpr double EPSILON = 1e-12;

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

// Helper function: Vector subtraction
Vec3 subtract(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// Compute the normal of a triangle
Vec3 computeNormal(const Triangle& tri) {
    return cross(subtract(tri[1], tri[0]), subtract(tri[2], tri[0]));
}

// Compute signed distance from a point to a plane
double signedDistance(const Vec3& p, const Vec3& normal, const Vec3& p0) {
    return dot(normal, subtract(p, p0));
}

// **Shen's Algorithm** for Triangle-Triangle Intersection
bool shenIntersection(const Triangle& T1, const Triangle& T2) {
    Vec3 normal1 = computeNormal(T1);
    Vec3 normal2 = computeNormal(T2);

    // Compute signed distances
    double d1 = signedDistance(T2[0], normal1, T1[0]);
    double d2 = signedDistance(T2[1], normal1, T1[0]);
    double d3 = signedDistance(T2[2], normal1, T1[0]);

    double d4 = signedDistance(T1[0], normal2, T2[0]);
    double d5 = signedDistance(T1[1], normal2, T2[0]);
    double d6 = signedDistance(T1[2], normal2, T2[0]);

    // If all points of one triangle are on the same side of the separation plane, return false
    if ((d1 > 0 && d2 > 0 && d3 > 0) || (d1 < 0 && d2 < 0 && d3 < 0)) {
        return false;
    }

    if ((d4 > 0 && d5 > 0 && d6 > 0) || (d4 < 0 && d5 < 0 && d6 < 0)) {
        return false;
    }

    return true;
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

    auto triangles1 = readTrianglesFromOFF("VH_F_vitreous_humor_L.off");
    auto triangles2 = readTrianglesFromOFF("VH_F_vitreous_humor_R.off");

    bool intersect = false;
    for (const auto& tri1 : triangles1) {
        for (const auto& tri2 : triangles2) {
            if (shenIntersection(tri1, tri2)) {
                intersect = true;
                // Print the intersecting triangles
                // std::cout << "Triangle 1: " << tri1[0][0] << " " << tri1[0][1] << " " << tri1[0][2] << std::endl;
                // std::cout << "Triangle 2: " << tri2[0][0] << " " << tri2[0][1] << " " << tri2[0][2] << std::endl;
                // break;
            }
        }
        // if (intersect) break;
    }

    if (intersect) {
        std::cout << "Triangles intersect!" << std::endl;
    } else {
        std::cout << "Triangles do not intersect." << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Time taken: " << elapsed.count() << " seconds" << endl;

    return 0;
}
