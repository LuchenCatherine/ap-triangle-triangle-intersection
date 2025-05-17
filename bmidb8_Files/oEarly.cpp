#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <algorithm>

using namespace std;

// Type aliases
using Vec3     = array<double,3>;
using Triangle = array<Vec3,3>;

// Tolerances
constexpr double EPSILON   = 1e-8;
constexpr double TOLERANCE = 1e-6;

// ---------------- CPU helper functions ----------------

inline Vec3 crossD(const Vec3 &a, const Vec3 &b) {
    return {
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    };
}

inline double dotD(const Vec3 &a, const Vec3 &b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline Vec3 subD(const Vec3 &a, const Vec3 &b) {
    return { a[0]-b[0], a[1]-b[1], a[2]-b[2] };
}

// Möller–Trumbore ray-triangle intersection
bool rayIntersectsTriangleD(
    const Vec3 &orig,
    const Vec3 &dir,
    const Triangle &tri,
    double &t, double &u, double &v
) {
    Vec3 edge1 = subD(tri[1], tri[0]);
    Vec3 edge2 = subD(tri[2], tri[0]);
    Vec3 pvec  = crossD(dir, edge2);
    double det = dotD(edge1, pvec);
    if (fabs(det) < EPSILON) return false;  // parallel

    double invDet = 1.0 / det;
    Vec3  tvec   = subD(orig, tri[0]);

    u = dotD(tvec, pvec) * invDet;
    if (u < 0.0 || u > 1.0) return false;

    Vec3 qvec = crossD(tvec, edge1);
    v = dotD(dir, qvec) * invDet;
    if (v < 0.0 || u + v > 1.0) return false;

    t = dotD(edge2, qvec) * invDet;
    return true;
}

// Full triangle-triangle intersection test (no early exit flag bailout)
bool trianglesIntersectD(
    const Triangle &A,
    const Triangle &B
) {
    // AABB overlap
    double minAx = min({A[0][0], A[1][0], A[2][0]});
    double minAy = min({A[0][1], A[1][1], A[2][1]});
    double minAz = min({A[0][2], A[1][2], A[2][2]});
    double maxAx = max({A[0][0], A[1][0], A[2][0]});
    double maxAy = max({A[0][1], A[1][1], A[2][1]});
    double maxAz = max({A[0][2], A[1][2], A[2][2]});

    double minBx = min({B[0][0], B[1][0], B[2][0]});
    double minBy = min({B[0][1], B[1][1], B[2][1]});
    double minBz = min({B[0][2], B[1][2], B[2][2]});
    double maxBx = max({B[0][0], B[1][0], B[2][0]});
    double maxBy = max({B[0][1], B[1][1], B[2][1]});
    double maxBz = max({B[0][2], B[1][2], B[2][2]});

    if (maxAx < minBx - TOLERANCE || minAx > maxBx + TOLERANCE ||
        maxAy < minBy - TOLERANCE || minAy > maxBy + TOLERANCE ||
        maxAz < minBz - TOLERANCE || minAz > maxBz + TOLERANCE) {
        return false;
    }

    // Edge-vs-triangle (6 tests)
    double t,u,v;
    for(int i = 0; i < 3; ++i) {
        Vec3 o = A[i];
        Vec3 d = subD(A[(i+1)%3], o);
        if (rayIntersectsTriangleD(o, d, B, t, u, v) && t >= 0.0 && t <= 1.0+TOLERANCE)
            return true;
    }
    for(int i = 0; i < 3; ++i) {
        Vec3 o = B[i];
        Vec3 d = subD(B[(i+1)%3], o);
        if (rayIntersectsTriangleD(o, d, A, t, u, v) && t >= 0.0 && t <= 1.0+TOLERANCE)
            return true;
    }

    return false;
}

// Simple OFF loader: returns a list of triangles
vector<Triangle> readOFF(const string &file) {
    ifstream in(file);
    if (!in) throw runtime_error("Cannot open OFF file: " + file);

    string hdr;
    in >> hdr;
    if (hdr != "OFF") throw runtime_error("Not an OFF file: " + file);

    int verts, faces, edges;
    in >> verts >> faces >> edges;

    vector<Vec3> V(verts);
    for(int i = 0; i < verts; ++i) {
        in >> V[i][0] >> V[i][1] >> V[i][2];
    }

    vector<Triangle> tris;
    for(int i = 0; i < faces; ++i) {
        int cnt, a, b, c;
        in >> cnt >> a >> b >> c;
        if (cnt == 3) {
            tris.push_back({ V[a], V[b], V[c] });
        }
        // (skips any extra vertices if cnt>3)
        for(int k = 3; k < cnt; ++k) {
            int tmp; in >> tmp;
        }
    }
    return tris;
}

int main() {
    // hard-coded OFF filenames
    const auto fileA = "VH_F_vitreous_humor_L.off";
    const auto fileB = "VH_F_vitreous_humor_R.off";

    // load meshes
    auto A = readOFF(fileA);
    auto B = readOFF(fileB);

    // measure time for full intersection test
    auto t0 = chrono::high_resolution_clock::now();

    bool intersects = false;
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < B.size(); ++j) {
            if (trianglesIntersectD(A[i], B[j])) {
                intersects = true;
                // (we still let all pairs run, to mirror the "no early-exit" CUDA kernel)
            }
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();

    cout << "Execution time (CPU): " << ms << " ms\n";
    cout << (intersects ? "Intersect\n" : "No intersect\n");
    return 0;
}
