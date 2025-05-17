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

// CUDA Helper Functions
__device__ Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

__device__ double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ Vec3 subtract(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// Compute the normal of a triangle
__device__ Vec3 computeNormal(const Triangle& tri) {
    return cross(subtract(tri[1], tri[0]), subtract(tri[2], tri[0]));
}

// Compute signed distance from a point to a plane
__device__ double signedDistance(const Vec3& p, const Vec3& normal, const Vec3& p0) {
    return dot(normal, subtract(p, p0));
}

// **Shen's Algorithm** for CUDA Parallel Execution
__device__ bool shaneIntersection(const Triangle& T1, const Triangle& T2) {
    Vec3 normal1 = computeNormal(T1);
    Vec3 normal2 = computeNormal(T2);

    // Compute signed distances for plane separation test
    double d1 = signedDistance(T2[0], normal1, T1[0]);
    double d2 = signedDistance(T2[1], normal1, T1[0]);
    double d3 = signedDistance(T2[2], normal1, T1[0]);

    double d4 = signedDistance(T1[0], normal2, T2[0]);
    double d5 = signedDistance(T1[1], normal2, T2[0]);
    double d6 = signedDistance(T1[2], normal2, T2[0]);

    // If all points of one triangle are on the same side of the plane, they do not intersect
    if ((d1 > 0 && d2 > 0 && d3 > 0) || (d1 < 0 && d2 < 0 && d3 < 0)) {
        return false;
    }
    if ((d4 > 0 && d5 > 0 && d6 > 0) || (d4 < 0 && d5 < 0 && d6 < 0)) {
        return false;
    }

    return true;
}

// CUDA Kernel for Triangle-Triangle Intersection
__global__ void triangleIntersectionKernel(const Triangle* triangles1, int numTriangles1,
                                           const Triangle* triangles2, int numTriangles2,
                                           bool* intersectionFound) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numTriangles1 && !(*intersectionFound); i += stride) {
        for (int j = 0; j < numTriangles2 && !(*intersectionFound); ++j) {
            if (shaneIntersection(triangles1[i], triangles2[j])) {
                *intersectionFound = true;
            }
        }
    }
}

// Function to read triangles from an OFF file
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
    auto start = chrono::high_resolution_clock::now();

    auto triangles1 = readTrianglesFromOFF("VH_F_cartilage_of_tertiary_bronchus_L.off");
    auto triangles2 = readTrianglesFromOFF("VH_F_cartilage_of_tertiary_bronchus_R.off");

    // Allocate device memory
    Triangle *d_triangles1, *d_triangles2;
    bool *d_intersectionFound;

    cudaMalloc(&d_triangles1, triangles1.size() * sizeof(Triangle));
    cudaMalloc(&d_triangles2, triangles2.size() * sizeof(Triangle));
    cudaMalloc(&d_intersectionFound, sizeof(bool));

    // Copy data to device
    cudaMemcpy(d_triangles1, triangles1.data(), triangles1.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles2, triangles2.data(), triangles2.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    bool intersectionFound = false;
    cudaMemcpy(d_intersectionFound, &intersectionFound, sizeof(bool), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (triangles1.size() + blockSize - 1) / blockSize;
    triangleIntersectionKernel<<<numBlocks, blockSize>>>(d_triangles1, triangles1.size(),
                                                         d_triangles2, triangles2.size(),
                                                         d_intersectionFound);

    // Copy result back to host
    cudaMemcpy(&intersectionFound, d_intersectionFound, sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_triangles1);
    cudaFree(d_triangles2);
    cudaFree(d_intersectionFound);

    if (intersectionFound) {
        cout << "Triangles intersect!" << endl;
    } else {
        cout << "Triangles do not intersect." << endl;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Time taken: " << elapsed.count() << " seconds" << endl;

    return 0;
}