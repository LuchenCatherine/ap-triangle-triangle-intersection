#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <sstream>
#include <chrono>
#include <algorithm>

using namespace std;

struct Point {
    float x, y, z;

    Point operator-(const Point& p) const {
        return {x - p.x, y - p.y, z - p.z};
    }

    Point cross(const Point& p) const {
        return {
            y * p.z - z * p.y,
            z * p.x - x * p.z,
            x * p.y - y * p.x
        };
    }

    float dot(const Point& p) const {
        return x * p.x + y * p.y + z * p.z;
    }
};

struct Triangle {
    Point p1, p2, p3;
};

float sign(Point p1, Point p2, Point p3) {
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

bool isPointInTriangle(const Point& pt, const Triangle& tri) {
    float d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(pt, tri.p1, tri.p2);
    d2 = sign(pt, tri.p2, tri.p3);
    d3 = sign(pt, tri.p3, tri.p1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

bool edgeIntersectsTriangle(const Point& p1, const Point& p2, const Triangle& tri) {
    const float EPSILON = 1e-5f;
    Point edge1 = tri.p2 - tri.p1;
    Point edge2 = tri.p3 - tri.p1;
    Point h = (p2 - p1).cross(edge2);
    float a = edge1.dot(h);

    if (a > -EPSILON && a < EPSILON) return false;

    float f = 1.0f / a;
    Point s = p1 - tri.p1;
    float u = f * s.dot(h);

    if (u < 0.0f || u > 1.0f) return false;

    Point q = s.cross(edge1);
    float v = f * (p2 - p1).dot(q);

    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * edge2.dot(q);

    return (t > EPSILON && t < 1.0f - EPSILON);
}

bool trianglesIntersect(const Triangle& A, const Triangle& B) {
    // Check if any edge of A intersects B
    if (edgeIntersectsTriangle(A.p1, A.p2, B) ||
        edgeIntersectsTriangle(A.p2, A.p3, B) ||
        edgeIntersectsTriangle(A.p3, A.p1, B)) {
        return true;
    }

    // Check if any edge of B intersects A
    if (edgeIntersectsTriangle(B.p1, B.p2, A) ||
        edgeIntersectsTriangle(B.p2, B.p3, A) ||
        edgeIntersectsTriangle(B.p3, B.p1, A)) {
        return true;
    }

    // Check if a vertex of A is inside B
    if (isPointInTriangle(A.p1, B) || isPointInTriangle(A.p2, B) || isPointInTriangle(A.p3, B)) {
        return true;
    }

    // Check if a vertex of B is inside A
    if (isPointInTriangle(B.p1, A) || isPointInTriangle(B.p2, A) || isPointInTriangle(B.p3, A)) {
        return true;
    }

    return false;
}

std::vector<Triangle> readTrianglesFromOFF(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    std::getline(file, line); // Read the "OFF" line

    int numVertices, numFaces, numEdges;
    file >> numVertices >> numFaces >> numEdges;
    std::vector<Point> vertices(numVertices);
    std::vector<Triangle> triangles;

    // Read vertices
    for (int i = 0; i < numVertices; ++i) {
        file >> vertices[i].x >> vertices[i].y >> vertices[i].z;
    }

    // Read faces
    for (int i = 0; i < numFaces; ++i) {
        int numVerticesInFace;
        file >> numVerticesInFace;
        if (numVerticesInFace != 3) {
            std::cerr << "Non-triangle face encountered, skipping." << std::endl;
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        int index1, index2, index3;
        file >> index1 >> index2 >> index3;
        triangles.push_back({vertices[index1], vertices[index2], vertices[index3]});
    }

    file.close();
    return triangles;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Triangle> trianglesA = readTrianglesFromOFF("./VH_F_renal_pyramid_L_a.off");
    std::vector<Triangle> trianglesB = readTrianglesFromOFF("./VH_F_renal_pyramid_L_b.off");

    bool intersectionFound = false;
    for (const auto& triangleA : trianglesA) {
        for (const auto& triangleB : trianglesB) {
            if (trianglesIntersect(triangleA, triangleB)) {
                std::cout << "Triangles intersect." << std::endl;
                intersectionFound = true;
                break;
            }
        }
        if (intersectionFound) break;
    }

    if (!intersectionFound) {
        std::cout << "No intersections found." << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Time taken: " << elapsed.count() << " seconds" << endl;
    return 0;
}