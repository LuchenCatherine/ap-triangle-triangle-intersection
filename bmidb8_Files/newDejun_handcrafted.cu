
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <float.h>

// ----------------------------
// Vector math macros
// ----------------------------
#define VmV_d(A,B,C)  {(A)[0]=(B)[0]-(C)[0]; (A)[1]=(B)[1]-(C)[1]; (A)[2]=(B)[2]-(C)[2];}
#define VpV_d(A,B,C)  {(A)[0]=(B)[0]+(C)[0]; (A)[1]=(B)[1]+(C)[1]; (A)[2]=(B)[2]+(C)[2];}
#define VcV_d(A,B)    {(A)[0]=(B)[0]; (A)[1]=(B)[1]; (A)[2]=(B)[2];}
#define VdotV_d(A,B)  ((A)[0]*(B)[0] + (A)[1]*(B)[1] + (A)[2]*(B)[2])
#define VxS_d(A,B,s)  {(A)[0]=(B)[0]*(s); (A)[1]=(B)[1]*(s); (A)[2]=(B)[2]*(s);}
#define VpVxS_d(A,B,C,s) {(A)[0]=(B)[0]+(C)[0]*(s); (A)[1]=(B)[1]+(C)[1]*(s); (A)[2]=(B)[2]+(C)[2]*(s);}
#define VcrossV_d(A,B,C) {(A)[0]=(B)[1]*(C)[2]-(B)[2]*(C)[1]; (A)[1]=(B)[2]*(C)[0]-(B)[0]*(C)[2]; (A)[2]=(B)[0]*(C)[1]-(B)[1]*(C)[0];}
#define VdistV2_d(A,B) (((A)[0]-(B)[0])*((A)[0]-(B)[0]) + ((A)[1]-(B)[1])*((A)[1]-(B)[1]) + ((A)[2]-(B)[2])*((A)[2]-(B)[2]))

// ----------------------------
// Triangle distance functions
// ----------------------------
__device__ float Vdist_d(const float *a, const float *b) {
    return sqrtf(VdistV2_d(a, b));
}

__device__ float TriDist_seg(const float *P1, const float *P2, const float *Q1, const float *Q2) {
    float u[3], v[3], w[3], a, b, c, d, e, D, sc, sN, sD, tc, tN, tD;
    float SMALL_NUM = 1e-12;
    float dP[3], dQ[3], dPQ[3];

    VmV_d(u, P2, P1);
    VmV_d(v, Q2, Q1);
    VmV_d(w, P1, Q1);
    a = VdotV_d(u, u);
    b = VdotV_d(u, v);
    c = VdotV_d(v, v);
    d = VdotV_d(u, w);
    e = VdotV_d(v, w);
    D = a * c - b * b;

    sD = D;
    tD = D;

    if (D < SMALL_NUM) {
        sN = 0.0;
        sD = 1.0;
        tN = e;
        tD = c;
    } else {
        sN = (b * e - c * d);
        tN = (a * e - b * d);
        if (sN < 0.0) {
            sN = 0.0;
            tN = e;
            tD = c;
        } else if (sN > sD) {
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }

    if (tN < 0.0) {
        tN = 0.0;
        if (-d < 0.0)
            sN = 0.0;
        else if (-d > a)
            sN = sD;
        else {
            sN = -d;
            sD = a;
        }
    } else if (tN > tD) {
        tN = tD;
        if ((-d + b) < 0.0)
            sN = 0;
        else if ((-d + b) > a)
            sN = sD;
        else {
            sN = (-d + b);
            sD = a;
        }
    }

    sc = (fabsf(sN) < SMALL_NUM ? 0.0 : sN / sD);
    tc = (fabsf(tN) < SMALL_NUM ? 0.0 : tN / tD);

    VpVxS_d(dP, P1, u, sc);
    VpVxS_d(dQ, Q1, v, tc);
    return Vdist_d(dP, dQ);
}

__device__ float TriDist_kernel(const float *S, const float *T) {
    float min_dist = FLT_MAX;

    // Triangle edges
    const float *A = S, *B = S + 3, *C = S + 6;
    const float *D = T, *E = T + 3, *F = T + 6;

    float dist;

    // Edge-edge checks
    dist = TriDist_seg(A, B, D, E); if (dist < min_dist) min_dist = dist;
    dist = TriDist_seg(A, B, E, F); if (dist < min_dist) min_dist = dist;
    dist = TriDist_seg(A, B, F, D); if (dist < min_dist) min_dist = dist;

    dist = TriDist_seg(B, C, D, E); if (dist < min_dist) min_dist = dist;
    dist = TriDist_seg(B, C, E, F); if (dist < min_dist) min_dist = dist;
    dist = TriDist_seg(B, C, F, D); if (dist < min_dist) min_dist = dist;

    dist = TriDist_seg(C, A, D, E); if (dist < min_dist) min_dist = dist;
    dist = TriDist_seg(C, A, E, F); if (dist < min_dist) min_dist = dist;
    dist = TriDist_seg(C, A, F, D); if (dist < min_dist) min_dist = dist;

    return min_dist;
}

// ----------------------------
// Bounding box filter
// ----------------------------
__device__ bool BoundingBoxesOverlap(const float* A, const float* B) {
    for (int i = 0; i < 3; ++i) {
        float minA = fminf(fminf(A[i], A[3+i]), A[6+i]);
        float maxA = fmaxf(fmaxf(A[i], A[3+i]), A[6+i]);
        float minB = fminf(fminf(B[i], B[3+i]), B[6+i]);
        float maxB = fmaxf(fmaxf(B[i], B[3+i]), B[6+i]);
        if (maxA < minB || maxB < minA) return false;
    }
    return true;
}

// ----------------------------
// CUDA kernel
// ----------------------------
__global__ void CheckIntersectionKernel(float *meshA, int triCountA, float *meshB, int triCountB, bool *intersectFlag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= triCountA * triCountB) return;

    int a = idx / triCountB;
    int b = idx % triCountB;

    float *triA = &meshA[a * 9];
    float *triB = &meshB[b * 9];

    if (!BoundingBoxesOverlap(triA, triB)) return;

    float dist = TriDist_kernel(triA, triB);
    if (dist < 1e-6f) {
        *intersectFlag = true;
    }
}

// ----------------------------
// OFF file loader
// ----------------------------
bool loadOFF(const std::string &filename, std::vector<float> &triangles) {
    std::ifstream infile(filename);
    std::string line;
    if (!infile.is_open()) return false;

    int vertexCount = 0, faceCount = 0;
    std::vector<std::vector<float>> vertices;

    while (getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);

        if (line.substr(0, 3) == "OFF") continue;
        if (vertexCount == 0 && faceCount == 0) {
            ss >> vertexCount >> faceCount;
            vertices.reserve(vertexCount);
            continue;
        }

        if (vertices.size() < vertexCount) {
            float x, y, z;
            ss >> x >> y >> z;
            vertices.push_back({x, y, z});
        } else {
            int n, a, b, c;
            ss >> n >> a >> b >> c;
            for (int i : {a, b, c}) {
                triangles.insert(triangles.end(), vertices[i].begin(), vertices[i].end());
            }
        }
    }
    return true;
}

// ----------------------------
// Main function
// ----------------------------
int main(int argc, char **argv) {
    // if (argc != 3) {
    //     std::cerr << "Usage: VH_M_ileocecal_valve.off VH_M_transverse_colon.off\n";
    //     return 1;
    // }

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    std::vector<float> meshA, meshB;
    if (!loadOFF("VH_F_cartilage_of_tertiary_bronchus_L.off", meshA) || !loadOFF("VH_F_cartilage_of_tertiary_bronchus_R.off", meshB)) {
        std::cerr << "Failed to load one or both OFF files.\n";
        return 1;
    }

    int triCountA = meshA.size() / 9;
    int triCountB = meshB.size() / 9;

    float *d_meshA, *d_meshB;
    bool *d_flag, h_flag = false;

    cudaMalloc(&d_meshA, meshA.size() * sizeof(float));
    cudaMalloc(&d_meshB, meshB.size() * sizeof(float));
    cudaMalloc(&d_flag, sizeof(bool));

    cudaMemcpy(d_meshA, meshA.data(), meshA.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_meshB, meshB.data(), meshB.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag, &h_flag, sizeof(bool), cudaMemcpyHostToDevice);

    int totalPairs = triCountA * triCountB;
    int threadsPerBlock = 256;
    int numBlocks = (totalPairs + threadsPerBlock - 1) / threadsPerBlock;

    CheckIntersectionKernel<<<numBlocks, threadsPerBlock>>>(d_meshA, triCountA, d_meshB, triCountB, d_flag);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "⏱️  Intersection check took: " << milliseconds << " ms\n";


    cudaMemcpy(&h_flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost);

    if (h_flag)
        std::cout << "🟥 Models INTERSECT.\n";
    else
        std::cout << "✅ Models DO NOT intersect.\n";

    
    cudaFree(d_meshA);
    cudaFree(d_meshB);
    cudaFree(d_flag);


    return 0;
}
