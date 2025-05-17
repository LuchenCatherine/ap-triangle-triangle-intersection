#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <chrono>


// ----------------------------
// Vector‐math macros (3‐component vectors)
// ----------------------------
#define VmV_d(A,B,C)   { (A)[0]=(B)[0]-(C)[0]; (A)[1]=(B)[1]-(C)[1]; (A)[2]=(B)[2]-(C)[2]; }
#define VdotV_d(A,B)   ((A)[0]*(B)[0] + (A)[1]*(B)[1] + (A)[2]*(B)[2])
#define VpVxS_d(A,B,C,s) { (A)[0]=(B)[0]+(C)[0]*(s); (A)[1]=(B)[1]+(C)[1]*(s); (A)[2]=(B)[2]+(C)[2]*(s); }
#define VdistV2_d(A,B) ( ((A)[0]-(B)[0])*((A)[0]-(B)[0]) \
                       + ((A)[1]-(B)[1])*((A)[1]-(B)[1]) \
                       + ((A)[2]-(B)[2])*((A)[2]-(B)[2]) )

inline float Vdist_d(const float *a, const float *b) {
    return std::sqrt(VdistV2_d(a,b));
}

// ----------------------------
// Shortest distance between segments P1–P2 and Q1–Q2
// (ported unchanged from the CUDA version)
// ----------------------------
float TriDist_seg(const float *P1, const float *P2,
                  const float *Q1, const float *Q2)
{
    float u[3], v[3], w[3];
    float a, b, c, d, e, D, sc, sN, sD, tc, tN, tD;
    const float SMALL_NUM = 1e-12f;
    float dP[3], dQ[3];

    VmV_d(u, P2, P1);
    VmV_d(v, Q2, Q1);
    VmV_d(w, P1, Q1);

    a = VdotV_d(u,u);
    b = VdotV_d(u,v);
    c = VdotV_d(v,v);
    d = VdotV_d(u,w);
    e = VdotV_d(v,w);
    D = a*c - b*b;
    sD = D;  // denominator for sc
    tD = D;  // denominator for tc

    // compute the line parameters of the two closest points
    if (D < SMALL_NUM) {
        // almost parallel
        sN = 0.0f;        // force using point P1 on segment S1
        tN = e;
        tD = c;
    } else {
        sN = (b*e - c*d);
        tN = (a*e - b*d);
        if (sN < 0.0f) {
            sN = 0.0f;
            tN = e;
            tD = c;
        } else if (sN > sD) {
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }
    // finally do tc clamp
    if (tN < 0.0f) {
        tN = 0.0f;
        if (-d < 0.0f)      sN = 0.0f;
        else if (-d > a)    sN = sD;
        else { sN = -d; sD = a; }
    } else if (tN > tD) {
        tN = tD;
        if ((-d + b) < 0.0f)      sN = 0;
        else if ((-d + b) > a)    sN = sD;
        else { sN = (-d + b); sD = a; }
    }

    sc = (std::fabs(sN) < SMALL_NUM ? 0.0f : sN / sD);
    tc = (std::fabs(tN) < SMALL_NUM ? 0.0f : tN / tD);

    VpVxS_d(dP, P1, u, sc);  // dP = P1 + u*sc
    VpVxS_d(dQ, Q1, v, tc);  // dQ = Q1 + v*tc

    return Vdist_d(dP, dQ);
}

// ----------------------------
// Minimum distance between two triangles S and T
// (edge‐to‐edge checks)
// ----------------------------
float TriDist_kernel(const float *S, const float *T) {
    float min_dist = FLT_MAX;
    const float *A = S, *B = S + 3, *C = S + 6;
    const float *D = T, *E = T + 3, *F = T + 6;
    float d;

    // all 9 edge‐edge combinations
    d = TriDist_seg(A,B, D,E); if (d<min_dist) min_dist=d;
    d = TriDist_seg(A,B, E,F); if (d<min_dist) min_dist=d;
    d = TriDist_seg(A,B, F,D); if (d<min_dist) min_dist=d;

    d = TriDist_seg(B,C, D,E); if (d<min_dist) min_dist=d;
    d = TriDist_seg(B,C, E,F); if (d<min_dist) min_dist=d;
    d = TriDist_seg(B,C, F,D); if (d<min_dist) min_dist=d;

    d = TriDist_seg(C,A, D,E); if (d<min_dist) min_dist=d;
    d = TriDist_seg(C,A, E,F); if (d<min_dist) min_dist=d;
    d = TriDist_seg(C,A, F,D); if (d<min_dist) min_dist=d;

    return min_dist;
}

// ----------------------------
// Fast axis‐aligned bounding‐box overlap test
// ----------------------------
bool BoundingBoxesOverlap(const float A[9], const float B[9]) {
    for (int i = 0; i < 3; ++i) {
        float minA = std::min({ A[i], A[i+3], A[i+6] });
        float maxA = std::max({ A[i], A[i+3], A[i+6] });
        float minB = std::min({ B[i], B[i+3], B[i+6] });
        float maxB = std::max({ B[i], B[i+3], B[i+6] });
        if (maxA < minB || maxB < minA) return false;
    }
    return true;
}

// ----------------------------
// Load OFF file into a flat triangle list (each triangle = 9 floats)
// ----------------------------
bool loadOFF(const std::string &filename, std::vector<float> &tris) {
    std::ifstream in(filename);
    if (!in) return false;

    // ---- read header (skip blank/comment lines) ----
    std::string token;
    while (in >> token) {
        if (token[0]=='#') {
            // skip rest of this line
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (token == "OFF") break;
        // if it's not OFF, could be stray blank—keep looping
    }
    if (token != "OFF") return false;

    // ---- read counts ----
    int vertexCount, faceCount, edgeCount;
    in >> vertexCount >> faceCount >> edgeCount;
    if (!in || vertexCount<=0 || faceCount<=0) return false;

    // ---- read vertices ----
    std::vector<std::array<float,3>> V(vertexCount);
    for (int i = 0; i < vertexCount; ++i) {
        in >> V[i][0] >> V[i][1] >> V[i][2];
        if (!in) return false;
    }

    // ---- read faces ----
    tris.reserve(tris.size() + faceCount*9);
    for (int i = 0; i < faceCount; ++i) {
        int n, a, b, c;
        in >> n;
        if (!in || n < 3) return false;
        // read exactly three indices (triangulate if necessary)
        in >> a >> b >> c;
        if (!in) return false;
        tris.push_back(V[a][0]); tris.push_back(V[a][1]); tris.push_back(V[a][2]);
        tris.push_back(V[b][0]); tris.push_back(V[b][1]); tris.push_back(V[b][2]);
        tris.push_back(V[c][0]); tris.push_back(V[c][1]); tris.push_back(V[c][2]);
        // skip any extra vertices if n>3
        for (int k = 3; k < n; ++k) {
            int idx; in >> idx;
        }
    }

    return true;
}

// ----------------------------
// Entry point
// ----------------------------
int main() {
    auto start  = std::chrono::high_resolution_clock::now();
    // Hard-coded OFF filenames
    const std::string fileA = "VH_F_vitreous_humor_L.off";
    const std::string fileB = "VH_F_vitreous_humor_R.off";

    // Load meshes
    std::vector<float> meshA, meshB;
    if (!loadOFF(fileA, meshA) || !loadOFF(fileB, meshB)) {
        std::cerr << "Failed to load OFF files.\n";
        return 1;
    }

    const size_t triCountA = meshA.size() / 9;
    const size_t triCountB = meshB.size() / 9;
    bool intersects = false;

    // Brute-force all triangle pairs
    for (size_t i = 0; i < triCountA && !intersects; ++i) {
        const float *tA = &meshA[i*9];
        for (size_t j = 0; j < triCountB; ++j) {
            const float *tB = &meshB[j*9];
            if (!BoundingBoxesOverlap(tA, tB)) continue;
            float d = TriDist_kernel(tA, tB);
            if (d < 1e-6f) {
                intersects = true;
                // break;
            }
        }
    }

    if (intersects)
        std::cout << "🟥 Models INTERSECT.\n";
    else
        std::cout << "✅ Models DO NOT intersect.\n";

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << std::endl;

    return 0;
}
