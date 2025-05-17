// GPU_TriangleIntersection_NoEarlyExit.cu

#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// Type aliases
using Vec3     = array<double,3>;
using Triangle = array<Vec3,3>;

// Tolerances
constexpr double EPSILON   = 1e-8;
constexpr double TOLERANCE = 1e-6;

// ---------------- Device helper functions ----------------

__device__ Vec3 crossD(const Vec3 &a, const Vec3 &b) {
    return {
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    };
}

__device__ double dotD(const Vec3 &a, const Vec3 &b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ Vec3 subD(const Vec3 &a, const Vec3 &b) {
    return { a[0]-b[0], a[1]-b[1], a[2]-b[2] };
}

__device__ bool rayIntersectsTriangleD(
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

__device__ bool trianglesIntersectD(
    const Triangle &A,
    const Triangle &B
) {
    // AABB overlap test
    double minAx = fmin(fmin(A[0][0], A[1][0]), A[2][0]);
    double minAy = fmin(fmin(A[0][1], A[1][1]), A[2][1]);
    double minAz = fmin(fmin(A[0][2], A[1][2]), A[2][2]);
    double maxAx = fmax(fmax(A[0][0], A[1][0]), A[2][0]);
    double maxAy = fmax(fmax(A[0][1], A[1][1]), A[2][1]);
    double maxAz = fmax(fmax(A[0][2], A[1][2]), A[2][2]);

    double minBx = fmin(fmin(B[0][0], B[1][0]), B[2][0]);
    double minBy = fmin(fmin(B[0][1], B[1][1]), B[2][1]);
    double minBz = fmin(fmin(B[0][2], B[1][2]), B[2][2]);
    double maxBx = fmax(fmax(B[0][0], B[1][0]), B[2][0]);
    double maxBy = fmax(fmax(B[0][1], B[1][1]), B[2][1]);
    double maxBz = fmax(fmax(B[0][2], B[1][2]), B[2][2]);

    if (maxAx < minBx - TOLERANCE || minAx > maxBx + TOLERANCE ||
        maxAy < minBy - TOLERANCE || minAy > maxBy + TOLERANCE ||
        maxAz < minBz - TOLERANCE || minAz > maxBz + TOLERANCE) {
        return false;
    }

    // Edge-vs-triangle tests (6 total)
    double t,u,v;
    for(int i=0;i<3;++i){
        Vec3 o = A[i];
        Vec3 d = subD(A[(i+1)%3], o);
        if(rayIntersectsTriangleD(o,d,B,t,u,v) && t>=0.0 && t<=1.0+TOLERANCE)
            return true;
    }
    for(int i=0;i<3;++i){
        Vec3 o = B[i];
        Vec3 d = subD(B[(i+1)%3], o);
        if(rayIntersectsTriangleD(o,d,A,t,u,v) && t>=0.0 && t<=1.0+TOLERANCE)
            return true;
    }
    return false;
}

// ---------------- Kernel WITHOUT early-exit flag checks ----------------

__global__ void triangleIntersectKernelNoEarlyExit(
    const Triangle *tri1, int n1,
    const Triangle *tri2, int n2,
    int *d_flag
) {
    int idx    = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // **No atomicAdd(d_flag,0)==0 check here**—
    // every thread will process its full range of i,j.
    for(int i=idx; i<n1; i+=stride){
        for(int j=0; j<n2; ++j){
            if(trianglesIntersectD(tri1[i], tri2[j])){
                // still use atomic to set the flag, but no bailout
                atomicExch(d_flag,1);
            }
        }
    }
}

// ---------------- Host utilities ----------------

vector<Triangle> readOFF(const string &file){
    ifstream in{file};
    string hdr; in>>hdr;
    if(hdr!="OFF") throw runtime_error("Not OFF");
    int verts,faces,edges;
    in>>verts>>faces>>edges;
    vector<Vec3> V(verts);
    for(int i=0;i<verts;++i)
        in>>V[i][0]>>V[i][1]>>V[i][2];
    vector<Triangle> out;
    for(int i=0;i<faces;++i){
        int cnt,a,b,c; in>>cnt>>a>>b>>c;
        if(cnt==3) out.push_back({V[a],V[b],V[c]});
    }
    return out;
}

int main(){
    // Load meshes
    auto A = readOFF("VH_F_cartilage_of_tertiary_bronchus_L.off");
    auto B = readOFF("VH_F_cartilage_of_tertiary_bronchus_R.off");

    // Device buffers
    Triangle *d_A, *d_B;
    int *d_flag;
    int   h_flag = 0;

    cudaMalloc(&d_A,    A.size()*sizeof(Triangle));
    cudaMalloc(&d_B,    B.size()*sizeof(Triangle));
    cudaMalloc(&d_flag, sizeof(int));

    cudaMemcpy(d_A,    A.data(), A.size()*sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,    B.data(), B.size()*sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag, &h_flag,    sizeof(int),                 cudaMemcpyHostToDevice);

    // Launch parameters
    int blockSize = 256;
    int numBlocks = (A.size() + blockSize - 1) / blockSize;

    // Time only the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    triangleIntersectKernelNoEarlyExit<<<numBlocks, blockSize>>>(
        d_A, A.size(), d_B, B.size(), d_flag
    );

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Kernel time (no early-exit): " << ms << " ms\n";

    // Read back result
    cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

    cout << (h_flag ? "Intersect\n" : "No intersect\n");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_flag);

    return 0;
}
