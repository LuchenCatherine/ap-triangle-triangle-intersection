#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

struct Vec3 {
    double x, y, z;
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator*(double scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
    Vec3 cross(const Vec3& v) const { return Vec3(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x); }
    double dot(const Vec3& v) const { return x*v.x + y*v.y + z*v.z; }
};

// Free function for scalar * Vec3
Vec3 operator*(double scalar, const Vec3& v) {
    return v * scalar;
}

struct Triangle {
    Vec3 v0, v1, v2;
};

std::vector<Triangle> readOFF(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line); // OFF header
    
    int numVertices, numFaces, numEdges;
    file >> numVertices >> numFaces >> numEdges;
    
    std::vector<Vec3> vertices(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        file >> vertices[i].x >> vertices[i].y >> vertices[i].z;
    }
    
    std::vector<Triangle> triangles;
    for (int i = 0; i < numFaces; ++i) {
        int n, v0, v1, v2;
        file >> n >> v0 >> v1 >> v2;
        if (n == 3) {
            triangles.push_back({vertices[v0], vertices[v1], vertices[v2]});
        }
    }
    
    return triangles;
}

bool mollerTrumbore(const Vec3& rayOrigin, const Vec3& rayDirection, const Triangle& triangle, double& t) {
    const double EPSILON = 0.0000001;
    Vec3 edge1 = triangle.v1 - triangle.v0;
    Vec3 edge2 = triangle.v2 - triangle.v0;
    Vec3 h = rayDirection.cross(edge2);
    double a = edge1.dot(h);

    if (a > -EPSILON && a < EPSILON) return false;

    double f = 1.0 / a;
    Vec3 s = rayOrigin - triangle.v0;
    double u = f * s.dot(h);

    if (u < 0.0 || u > 1.0) return false;

    Vec3 q = s.cross(edge1);
    double v = f * rayDirection.dot(q);

    if (v < 0.0 || u + v > 1.0) return false;

    t = f * edge2.dot(q);
    return (t > EPSILON);
}

bool triangleIntersection(const Triangle& t1, const Triangle& t2) {
    const Vec3 edges1[3] = {t1.v1 - t1.v0, t1.v2 - t1.v1, t1.v0 - t1.v2};
    const Vec3 edges2[3] = {t2.v1 - t2.v0, t2.v2 - t2.v1, t2.v0 - t2.v2};

    for (int i = 0; i < 3; ++i) {
        Vec3 rayOrigin = t1.v0 + edges1[i] * 0.5;
        Vec3 rayDirection = edges1[i].cross(edges2[0]);
        
        double t;
        if (mollerTrumbore(rayOrigin, rayDirection, t2, t)) {
            return true;
        }
    }

    return false;
}

int main() {
    std::vector<Triangle> meshA = readOFF("VH_F_renal_pyramid_L_a.off");
    std::vector<Triangle> meshB = readOFF("VH_F_renal_pyramid_L_b.off");

    for (const auto& triangleA : meshA) {
        for (const auto& triangleB : meshB) {
            if (triangleIntersection(triangleA, triangleB)) {
                std::cout << "Intersection found!" << std::endl;
                return 0;
            }
        }
    }

    std::cout << "No intersections found." << std::endl;
    return 0;
}