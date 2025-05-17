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

// ---------------- CPU‐side math ----------------

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

// Möller–Trumbore ray‐triangle intersection (returns true if ray (orig→orig+dir) hits tri)
bool rayIntersectsTriangle(
    const Vec3   &orig,
    const Vec3   &dir,
    const Triangle &tri,
    double       &t,
    double       &u,
    double       &v
) {
    Vec3 edge1 = subD(tri[1], tri[0]);
    Vec3 edge2 = subD(tri[2], tri[0]);
    Vec3 pvec  = crossD(dir, edge2);
    double det = dotD(edge1, pvec);
    if (fabs(det) < EPSILON) return false;  // parallel or nearly so

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

// Full triangle‐triangle intersection test:  
// 1) AABB overlap (with tolerance)  
// 2) edge‐vs‐triangle tests (3 edges of A against B, then 3 edges of B against A)
bool trianglesIntersect(const Triangle &A, const Triangle &B) {
    // AABB for A
    double minAx = min({A[0][0],A[1][0],A[2][0]});
    double minAy = min({A[0][1],A[1][1],A[2][1]});
    double minAz = min({A[0][2],A[1][2],A[2][2]});
    double maxAx = max({A[0][0],A[1][0],A[2][0]});
    double maxAy = max({A[0][1],A[1][1],A[2][1]});
    double maxAz = max({A[0][2],A[1][2],A[2][2]});
    // AABB for B
    double minBx = min({B[0][0],B[1][0],B[2][0]});
    double minBy = min({B[0][1],B[1][1],B[2][1]});
    double minBz = min({B[0][2],B[1][2],B[2][2]});
    double maxBx = max({B[0][0],B[1][0],B[2][0]});
    double maxBy = max({B[0][1],B[1][1],B[2][1]});
    double maxBz = max({B[0][2],B[1][2],B[2][2]});

    // Quick reject if bounding boxes don’t overlap
    if (maxAx < minBx - TOLERANCE || minAx > maxBx + TOLERANCE ||
        maxAy < minBy - TOLERANCE || minAy > maxBy + TOLERANCE ||
        maxAz < minBz - TOLERANCE || minAz > maxBz + TOLERANCE) {
        return false;
    }

    // Edge‐vs‐triangle
    double t,u,v;
    // edges of A vs triangle B
    for (int i = 0; i < 3; ++i) {
        Vec3 orig = A[i];
        Vec3 dir  = subD(A[(i+1)%3], orig);
        if (rayIntersectsTriangle(orig, dir, B, t, u, v)
            && t >= 0.0 && t <= 1.0 + TOLERANCE) {
            return true;
        }
    }
    // edges of B vs triangle A
    for (int i = 0; i < 3; ++i) {
        Vec3 orig = B[i];
        Vec3 dir  = subD(B[(i+1)%3], orig);
        if (rayIntersectsTriangle(orig, dir, A, t, u, v)
            && t >= 0.0 && t <= 1.0 + TOLERANCE) {
            return true;
        }
    }

    return false;
}

// Simple OFF reader (no command‐line args—filename is hard‐coded below)
vector<Triangle> readOFF(const string &file) {
    ifstream in(file);
    if (!in) throw runtime_error("Cannot open OFF file: " + file);

    string hdr; 
    in >> hdr;
    if (hdr != "OFF") throw runtime_error("Not an OFF file: " + file);

    int verts, faces, edges;
    in >> verts >> faces >> edges;

    vector<Vec3> V(verts);
    for (int i = 0; i < verts; ++i) {
        in >> V[i][0] >> V[i][1] >> V[i][2];
    }

    vector<Triangle> tris;
    tris.reserve(faces);
    for (int i = 0; i < faces; ++i) {
        int cnt, a, b, c;
        in >> cnt >> a >> b >> c;
        if (cnt == 3) {
            tris.push_back({ V[a], V[b], V[c] });
        }
        // skip extra indices if face is a polygon
        for (int k = 3; k < cnt; ++k) {
            int dummy; 
            in >> dummy;
        }
    }
    return tris;
}

int main() {
    // Hard-coded OFF filenames
    const string fileA = "VH_F_vitreous_humor_L.off";
    const string fileB = "VH_F_vitreous_humor_R.off";

    // 1) Load meshes
    auto trisA = readOFF(fileA);
    auto trisB = readOFF(fileB);

    // 2) Brute-force intersection sweep with early exit
    auto t0 = chrono::high_resolution_clock::now();

    bool intersects = false;
    for (size_t i = 0; i < trisA.size() && !intersects; ++i) {
        for (size_t j = 0; j < trisB.size(); ++j) {
            if (trianglesIntersect(trisA[i], trisB[j])) {
                intersects = true;
                break;
            }
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();

    // 3) Report
    if (intersects)
        cout << "Triangles intersect!\n";
    else
        cout << "Triangles do not intersect.\n";

    cout << "Time taken: " << secs << " s\n";
    return 0;
}
