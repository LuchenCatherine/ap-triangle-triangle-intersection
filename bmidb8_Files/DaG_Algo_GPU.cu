#include <iostream>
#include <vector>
#include <fstream>
#include <array>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

using Vec3 = array<double, 3>;
using Triangle = array<Vec3, 3>;
constexpr double EPSILON = 1e-12;

// ======================================================================
// Devillers & Guigue Determinant Function (CUDA Device Function)
// ======================================================================
__device__ double determinant(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    return (a[0] - d[0]) * ((b[1] - d[1]) * (c[2] - d[2]) - (c[1] - d[1]) * (b[2] - d[2])) -
           (a[1] - d[1]) * ((b[0] - d[0]) * (c[2] - d[2]) - (c[0] - d[0]) * (b[2] - d[2])) +
           (a[2] - d[2]) * ((b[0] - d[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - d[1]));
}

// ======================================================================
// Devillers & Guigue Triangle-Triangle Intersection (CUDA Device Function)
// ======================================================================
__device__ bool devillersGuigueIntersection(const Triangle& T1, const Triangle& T2) {
    double d1 = determinant(T2[0], T2[1], T2[2], T1[0]);
    double d2 = determinant(T2[0], T2[1], T2[2], T1[1]);
    double d3 = determinant(T2[0], T2[1], T2[2], T1[2]);

    if ((d1 > EPSILON && d2 > EPSILON && d3 > EPSILON) || (d1 < -EPSILON && d2 < -EPSILON && d3 < -EPSILON)) {
        return false;
    }

    double d4 = determinant(T1[0], T1[1], T1[2], T2[0]);
    double d5 = determinant(T1[0], T1[1], T1[2], T2[1]);
    double d6 = determinant(T1[0], T1[1], T1[2], T2[2]);

    if ((d4 > EPSILON && d5 > EPSILON && d6 > EPSILON) || (d4 < -EPSILON && d5 < -EPSILON && d6 < -EPSILON)) {
        return false;
    }

    return true;
}

// ======================================================================
// CUDA Kernel Function
// ======================================================================
__global__ void triangleIntersectionKernel(const Triangle* triangles1, int numTriangles1,
                                           const Triangle* triangles2, int numTriangles2,
                                           int* intersectionFound) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numTriangles1; i += stride) {
        for (int j = 0; j < numTriangles2; ++j) {
            if (devillersGuigueIntersection(triangles1[i], triangles2[j])) {
                atomicExch(intersectionFound, 1); // Use int for atomic operation
            }
        }
    }
}

// ======================================================================
// Function to Read Triangles from an OFF File
// ======================================================================
vector<Triangle> readTrianglesFromOFF(const string& filename) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    string header;
    infile >> header;
    if (header != "OFF") {
        cerr << "Not a valid OFF file." << endl;
        exit(1);
    }

    int numVertices, numFaces, numEdges;
    infile >> numVertices >> numFaces >> numEdges;

    vector<Vec3> vertices(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        infile >> vertices[i][0] >> vertices[i][1] >> vertices[i][2];
    }

    vector<Triangle> triangles;
    triangles.reserve(numFaces);
    for (int i = 0; i < numFaces; ++i) {
        int vertexCount, v1, v2, v3;
        infile >> vertexCount >> v1 >> v2 >> v3;
        if (vertexCount == 3) {
            triangles.push_back({vertices[v1], vertices[v2], vertices[v3]});
        } else {
            for (int skip = 0; skip < vertexCount - 3; skip++) {
                infile >> v1;
            }
        }
    }
    return triangles;
}

// ======================================================================
// Main Function
// ======================================================================
int main() {
    auto start = chrono::high_resolution_clock::now();

    // Read OFF files
    vector<Triangle> triangles1 = readTrianglesFromOFF("VH_F_cartilage_of_tertiary_bronchus_L.off");
    vector<Triangle> triangles2 = readTrianglesFromOFF("VH_F_cartilage_of_tertiary_bronchus_R.off");

    // Allocate GPU memory
    Triangle *d_triangles1, *d_triangles2;
    int *d_intersectionFound; // Use int for atomic operation

    cudaMalloc((void**)&d_triangles1, triangles1.size() * sizeof(Triangle));
    cudaMalloc((void**)&d_triangles2, triangles2.size() * sizeof(Triangle));
    cudaMalloc((void**)&d_intersectionFound, sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_triangles1, triangles1.data(), triangles1.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles2, triangles2.data(), triangles2.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    int intersectionFound = 0; // Use int for atomic operation
    cudaMemcpy(d_intersectionFound, &intersectionFound, sizeof(int), cudaMemcpyHostToDevice);

    // Kernel Launch Configuration
    int blockSize = 256;
    int numBlocks = (triangles1.size() + blockSize - 1) / blockSize;

    // Launch CUDA Kernel
    triangleIntersectionKernel<<<numBlocks, blockSize>>>(d_triangles1, triangles1.size(), d_triangles2, triangles2.size(), d_intersectionFound);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete

    // Copy result back to CPU
    cudaMemcpy(&intersectionFound, d_intersectionFound, sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_triangles1);
    cudaFree(d_triangles2);
    cudaFree(d_intersectionFound);

    // Output result
    if (intersectionFound) {
        cout << "Triangles intersect!" << endl;
    } else {
        cout << "Triangles do not intersect." << endl;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Time taken: " << elapsed.count()*1000 << " milliseconds" << endl;

    return 0;
}