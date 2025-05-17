#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <array>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

using Vec3 = array<double, 3>;
using Triangle = array<Vec3, 3>;
constexpr double EPSILON = 1e-12;

// Helper functions (implemented as __device__ functions)
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

__device__ bool rayIntersectsTriangle(const Vec3& origin, const Vec3& direction, const Triangle& tri, double& t, double& u, double& v) {
    const Vec3& v0 = tri[0];
    const Vec3& v1 = tri[1];
    const Vec3& v2 = tri[2];

    Vec3 edge1 = subtract(v1, v0);
    Vec3 edge2 = subtract(v2, v0);

    Vec3 pvec = cross(direction, edge2);
    double det = dot(edge1, pvec);

    if (det > -EPSILON && det < EPSILON) return false;

    double invDet = 1.0 / det;
    Vec3 tvec = subtract(origin, v0);

    u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    Vec3 qvec = cross(tvec, edge1);
    v = dot(direction, qvec) * invDet;
    if (v < 0 || u + v > 1) return false;

    t = dot(edge2, qvec) * invDet;
    return true;
}

__device__ bool trianglesIntersect(const Triangle& tri1, const Triangle& tri2, double tolerance = 1e-9) {
    // Bounding box check
    Vec3 min1 = {min(min(tri1[0][0], tri1[1][0]), tri1[2][0]),
                 min(min(tri1[0][1], tri1[1][1]), tri1[2][1]),
                 min(min(tri1[0][2], tri1[1][2]), tri1[2][2])};
    Vec3 max1 = {max(max(tri1[0][0], tri1[1][0]), tri1[2][0]),
                 max(max(tri1[0][1], tri1[1][1]), tri1[2][1]),
                 max(max(tri1[0][2], tri1[1][2]), tri1[2][2])};
    Vec3 min2 = {min(min(tri2[0][0], tri2[1][0]), tri2[2][0]),
                 min(min(tri2[0][1], tri2[1][1]), tri2[2][1]),
                 min(min(tri2[0][2], tri2[1][2]), tri2[2][2])};
    Vec3 max2 = {max(max(tri2[0][0], tri2[1][0]), tri2[2][0]),
                 max(max(tri2[0][1], tri2[1][1]), tri2[2][1]),
                 max(max(tri2[0][2], tri2[1][2]), tri2[2][2])};

    if (max1[0] < min2[0] - tolerance || min1[0] > max2[0] + tolerance ||
        max1[1] < min2[1] - tolerance || min1[1] > max2[1] + tolerance ||
        max1[2] < min2[2] - tolerance || min1[2] > max2[2] + tolerance) {
        return false;
    }

    double t, u, v;

    for (int i = 0; i < 3; ++i) {
        Vec3 origin = tri1[i];
        Vec3 direction = subtract(tri1[(i + 1) % 3], origin);
        if (rayIntersectsTriangle(origin, direction, tri2, t, u, v)) {
            if (t >= 0 && t <= 1 + tolerance) return true;
        }
    }

    for (int i = 0; i < 3; ++i) {
        Vec3 origin = tri2[i];
        Vec3 direction = subtract(tri2[(i + 1) % 3], origin);
        if (rayIntersectsTriangle(origin, direction, tri1, t, u, v)) {
            if (t >= 0 && t <= 1 + tolerance) return true;
        }
    }

    return false;
}

// CUDA kernel for triangle intersection
__global__ void triangleIntersectionKernel(const Triangle* triangles1, int numTriangles1,
                                           const Triangle* triangles2, int numTriangles2,
                                           bool* intersectionFound) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numTriangles1 && !(*intersectionFound); i += stride) {
        for (int j = 0; j < numTriangles2 && !(*intersectionFound); ++j) {
            if (trianglesIntersect(triangles1[i], triangles2[j])) {
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