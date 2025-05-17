
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <float.h>

#include "geometry.h"

// Inject vector macros

// Vector math macros (from cuda_util.cuh)
#define VmV_d(A,B,C)  {(A)[0]=(B)[0]-(C)[0]; (A)[1]=(B)[1]-(C)[1]; (A)[2]=(B)[2]-(C)[2];}
#define VpV_d(A,B,C)  {(A)[0]=(B)[0]+(C)[0]; (A)[1]=(B)[1]+(C)[1]; (A)[2]=(B)[2]+(C)[2];}
#define VcV_d(A,B)    {(A)[0]=(B)[0]; (A)[1]=(B)[1]; (A)[2]=(B)[2];}
#define VdotV_d(A,B)  ((A)[0]*(B)[0] + (A)[1]*(B)[1] + (A)[2]*(B)[2])
#define VxS_d(A,B,s)  {(A)[0]=(B)[0]*(s); (A)[1]=(B)[1]*(s); (A)[2]=(B)[2]*(s);}
#define VpVxS_d(A,B,C,s) {(A)[0]=(B)[0]+(C)[0]*(s); (A)[1]=(B)[1]+(C)[1]*(s); (A)[2]=(B)[2]+(C)[2]*(s);}
#define VcrossV_d(A,B,C) {(A)[0]=(B)[1]*(C)[2]-(B)[2]*(C)[1]; (A)[1]=(B)[2]*(C)[0]-(B)[0]*(C)[2]; (A)[2]=(B)[0]*(C)[1]-(B)[1]*(C)[0];}
#define VdistV2_d(A,B) (((A)[0]-(B)[0])*((A)[0]-(B)[0]) + ((A)[1]-(B)[1])*((A)[1]-(B)[1]) + ((A)[2]-(B)[2])*((A)[2]-(B)[2]))


// Inject distance device code
 __device__
 inline void
 SegPoints(float VEC[3],
       float X[3], float Y[3],             // closest points
       const float P[3], const float A[3], // seg 1 origin, vector
       const float Q[3], const float B[3]) // seg 2 origin, vector
 {
   float T[3], A_dot_A, B_dot_B, A_dot_B, A_dot_T, B_dot_T;
   float TMP[3];
 
   VmV_d(T,Q,P);
   A_dot_A = VdotV_d(A,A);
   B_dot_B = VdotV_d(B,B);
   A_dot_B = VdotV_d(A,B);
   A_dot_T = VdotV_d(A,T);
   B_dot_T = VdotV_d(B,T);
   assert(A_dot_A!=0&&B_dot_B!=0);
 
   // t parameterizes ray P,A
   // u parameterizes ray Q,B
 
   float t,u;
 
   // compute t for the closest point on ray P,A to
   // ray Q,B
 
   float denom = A_dot_A*B_dot_B - A_dot_B*A_dot_B;
   if(denom == 0){
       t = 0;
   }else{
       t = (A_dot_T*B_dot_B - B_dot_T*A_dot_B) / denom;
   }
 __device__
 inline float
 TriDist_seg(const float *S, const float *T,
         bool &shown_disjoint, bool &closest_find){
 
     // closest points
     float P[3];
     float Q[3];
 
     // some temporary vectors
     float V[3];
     float Z[3];
     // Compute vectors along the 6 sides
     float Sv[3][3], Tv[3][3];
 
     VmV_d(Sv[0],S+3,S);
     VmV_d(Sv[1],S+6,S+3);
     VmV_d(Sv[2],S,S+6);
 
     VmV_d(Tv[0],T+3,T);
     VmV_d(Tv[1],T+6,T+3);
     VmV_d(Tv[2],T,T+6);
 
     // For each edge pair, the vector connecting the closest points
     // of the edges defines a slab (parallel planes at head and tail
     // enclose the slab). If we can show that the off-edge vertex of
     // each triangle is outside of the slab, then the closest points
     // of the edges are the closest points for the triangles.
     // Even if these tests fail, it may be helpful to know the closest
     // points found, and whether the triangles were shown disjoint
 
     float mindd = DBL_MAX; // Set first minimum safely high
     float VEC[3];
     for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
             // Find closest points on edges i & j, plus the
             // vector (and distance squared) between these points
             SegPoints(VEC,P,Q,S+i*3,Sv[i],T+j*3,Tv[j]);
             VmV_d(V,Q,P);
             float dd = VdotV_d(V,V);
             if (dd <= mindd){
                 mindd = dd;
 
                 // Verify this closest point pair for the segment pairs with minimum distance
                 VmV_d(Z,S+((i+2)%3)*3,P);
                 float a = VdotV_d(Z,VEC);
                 VmV_d(Z,T+((j+2)%3)*3,Q);
                 float b = VdotV_d(Z,VEC);
 
                 // the closest distance of segment pairs is the closest distance of the two triangles
                 if ((a <= 0) && (b >= 0)) {
                     closest_find = true;
                     return sqrt(mindd);
                 }
 __device__
 inline float
 TriDist_other(const float *S, const float *T, bool &shown_disjoint)
 {
 
     // closest points
     float P[3];
     float Q[3];
 
     // some temporary vectors
     float V[3];
     float Z[3];
     // Compute vectors along the 6 sides
     float Sv[3][3], Tv[3][3];
 
     VmV_d(Sv[0],S+3,S);
     VmV_d(Sv[1],S+6,S+3);
     VmV_d(Sv[2],S,S+6);
 
     VmV_d(Tv[0],T+3,T);
     VmV_d(Tv[1],T+6,T+3);
     VmV_d(Tv[2],T,T+6);
 
     // First check for case 1
 
     float Sn[3], Snl;
     VcrossV_d(Sn,Sv[0],Sv[1]); // Compute normal to S triangle
     Snl = VdotV_d(Sn,Sn);      // Compute square of length of normal
 
     // If cross product is long enough,
 
     if (Snl > 1e-15){
         // Get projection lengths of T points
 
         float Tp[3];
 
         VmV_d(V,S,T);
         Tp[0] = VdotV_d(V,Sn);
 
         VmV_d(V,S,T+3);
         Tp[1] = VdotV_d(V,Sn);
 
         VmV_d(V,S,T+6);
         Tp[2] = VdotV_d(V,Sn);
 
         // If Sn is a separating direction,
         // find point with smallest projection
 
         int point = -1;
         if ((Tp[0] > 0) && (Tp[1] > 0) && (Tp[2] > 0)) {
             if (Tp[0] < Tp[1]) point = 0; else point = 1;
             if (Tp[2] < Tp[point]) point = 2;
         } else if ((Tp[0] < 0) && (Tp[1] < 0) && (Tp[2] < 0)) {
             if (Tp[0] > Tp[1]) point = 0; else point = 1;
             if (Tp[2] > Tp[point]) point = 2;
         }
 __device__
 float
 TriDist_kernel(const float *S, const float *T)
 {
     bool shown_disjoint = false;
     bool closest_find = false;
     float mindd_seg = TriDist_seg(S, T, shown_disjoint, closest_find);
     if(closest_find){// the closest points are one segments, simply return
         return mindd_seg;
     }else{
         // No edge pairs contained the closest points.
         // either:
         // 1. one of the closest points is a vertex, and the
         //    other point is interior to a face.
         // 2. the triangles are overlapping.
         // 3. an edge of one triangle is parallel to the other's face. If
         //    cases 1 and 2 are not true, then the closest points from the 9
         //    edge pairs checks above can be taken as closest points for the
         //    triangles.
         // 4. possibly, the triangles were degenerate.  When the
         //    triangle points are nearly colinear or coincident, one
         //    of above tests might fail even though the edges tested
         //    contain the closest points.
         float mindd_other = TriDist_other(S, T, shown_disjoint);
         if(mindd_other != -1){ // is the case
             return mindd_other;
         }
 __device__ static float atomicMin(float* address, float val) {
     int* address_as_i = (int*) address;
     int old = *address_as_i, assumed;
     do {
         assumed = old;
         old = ::atomicCAS(address_as_i, assumed,
             __float_as_int(::fminf(val, __int_as_float(assumed))));
     } while (assumed != old);
     return __int_as_float(old);
 }

// Bounding box filter (from MT_Algo_GPU)
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

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./newDejun model1.off model2.off\n";
        return 1;
    }

    std::vector<float> meshA, meshB;
    if (!loadOFF(argv[1], meshA) || !loadOFF(argv[2], meshB)) {
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
