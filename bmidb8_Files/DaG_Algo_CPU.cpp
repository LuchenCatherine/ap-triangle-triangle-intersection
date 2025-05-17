#include <iostream>
#include <vector>
#include <fstream>
#include <array>
#include <iomanip>  // For setprecision
#include <chrono>   // For timing
#include <algorithm>  // For min/max

using namespace std;

using Vec3 = std::array<double, 3>;
using Triangle = std::array<Vec3, 3>;
constexpr double EPSILON = 1e-12;  // Higher precision for tolerance check

// Helper function: Compute determinant
double determinant(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    return (a[0] - d[0]) * ((b[1] - d[1]) * (c[2] - d[2]) - (c[1] - d[1]) * (b[2] - d[2])) -
           (a[1] - d[1]) * ((b[0] - d[0]) * (c[2] - d[2]) - (c[0] - d[0]) * (b[2] - d[2])) +
           (a[2] - d[2]) * ((b[0] - d[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - d[1]));
}

// Devillers & Guigue Algorithm for Triangle-Triangle Intersection
bool devillersGuigueIntersection(const Triangle& T1, const Triangle& T2) {
    double d1 = determinant(T2[0], T2[1], T2[2], T1[0]);
    double d2 = determinant(T2[0], T2[1], T2[2], T1[1]);
    double d3 = determinant(T2[0], T2[1], T2[2], T1[2]);

    // If all three determinants have the same sign, triangles do not intersect
    if ((d1 > 0 && d2 > 0 && d3 > 0) || (d1 < 0 && d2 < 0 && d3 < 0)) {
        return false;
    }

    double d4 = determinant(T1[0], T1[1], T1[2], T2[0]);
    double d5 = determinant(T1[0], T1[1], T1[2], T2[1]);
    double d6 = determinant(T1[0], T1[1], T1[2], T2[2]);

    // If all three determinants have the same sign, triangles do not intersect
    if ((d4 > 0 && d5 > 0 && d6 > 0) || (d4 < 0 && d5 < 0 && d6 < 0)) {
        return false;
    }

    return true; // Otherwise, triangles intersect
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
    int count = 0;
    for (const auto& tri1 : triangles1) {
        for (const auto& tri2 : triangles2) {
            if (devillersGuigueIntersection(tri1, tri2)) {
                intersect = true;
                count++;
                // Print the intersecting triangles
                // std::cout << "Triangle 1: " << tri1[0][0] << " " << tri1[0][1] << " " << tri1[0][2] << std::endl;
                // std::cout << "Triangle 2: " << tri2[0][0] << " " << tri2[0][1] << " " << tri2[0][2] << std::endl;
                // break;
            }
        }
        // if (intersect) break;
    }

    if (intersect) {
        // std::cout << "Found " << count << " intersecting triangles." << std::endl;
        std::cout << "Triangles intersect!" << std::endl;
    } else {
        std::cout << "Triangles do not intersect." << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Time taken: " << elapsed.count() << " seconds" << endl;

    return 0;
}
