// New One
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <array>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

using Vec3 = std::array<double, 3>;
using Triangle = std::array<Vec3, 3>;
constexpr double EPSILON = 1e-12;

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

__global__ void checkIntersection(const Triangle* triangles1, int n1, const Triangle* triangles2, int n2, bool* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n1 * n2) return;

    int i = idx / n2;
    int j = idx % n2;

    double t, u, v;
    if (rayIntersectsTriangle(triangles1[i][0], subtract(triangles1[i][1], triangles1[i][0]), triangles2[j], t, u, v) ||
        rayIntersectsTriangle(triangles2[j][0], subtract(triangles2[j][1], triangles2[j][0]), triangles1[i], t, u, v)) {
        *result = true;
    }
}

std::vector<Triangle> readTrianglesFromOFF(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    std::string header;
    infile >> header;
    if (header != "OFF") {
        std::cerr << "Not a valid OFF file." << std::endl;
        exit(1);
    }

    int numVertices, numFaces, numEdges;
    infile >> numVertices >> numFaces >> numEdges;

    std::vector<Vec3> vertices(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        infile >> vertices[i][0] >> vertices[i][1] >> vertices[i][2];
    }

    std::vector<Triangle> triangles;
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
    auto triangles1 = readTrianglesFromOFF("VH_M_ileocecal_valve.off");
    auto triangles2 = readTrianglesFromOFF("VH_M_transverse_colon.off");

    int n1 = triangles1.size();
    int n2 = triangles2.size();

    Triangle* d_triangles1;
    Triangle* d_triangles2;
    bool* d_result;

    cudaMalloc(&d_triangles1, n1 * sizeof(Triangle));
    cudaMalloc(&d_triangles2, n2 * sizeof(Triangle));
    cudaMalloc(&d_result, sizeof(bool));

    cudaMemcpy(d_triangles1, triangles1.data(), n1 * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles2, triangles2.data(), n2 * sizeof(Triangle), cudaMemcpyHostToDevice);

    bool result = false;
    cudaMemcpy(d_result, &result, sizeof(bool), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n1 * n2 + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    checkIntersection<<<blocksPerGrid, threadsPerBlock>>>(d_triangles1, n1, d_triangles2, n2, d_result);

    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (result) {
        std::cout << "Triangles intersect!" << std::endl;
    } else {
        std::cout << "Triangles do not intersect." << std::endl;
    }

    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;

    cudaFree(d_triangles1);
    cudaFree(d_triangles2);
    cudaFree(d_result);

    return 0;
}






// Old one
// %%writefile gp.cu
// #include <iostream>
// #include <vector>
// #include <fstream>
// #include <sstream>
// #include <array>
// #include <iomanip>  // For setprecision
// #include <cuda_runtime.h>
// using namespace std;
// using Vec3 = std::array<double, 3>;
// using Triangle = std::array<Vec3, 3>;
// constexpr double EPSILON = 1e-8;  // Higher precision for tolerance check

// __device__ Vec3 cross(const Vec3& a, const Vec3& b) {
//     return {
//         a[1] * b[2] - a[2] * b[1],
//         a[2] * b[0] - a[0] * b[2],
//         a[0] * b[1] - a[1] * b[0]
//     };
// }

// __device__ double dot(const Vec3& a, const Vec3& b) {
//     return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
// }

// __device__ Vec3 subtract(const Vec3& a, const Vec3& b) {
//     return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
// }

// __device__ bool rayIntersectsTriangle(const Vec3& origin, const Vec3& direction, const Triangle& tri, double& t, double& u, double& v) {
//     const Vec3& v0 = tri[0];
//     const Vec3& v1 = tri[1];
//     const Vec3& v2 = tri[2];

//     Vec3 edge1 = subtract(v1, v0);
//     Vec3 edge2 = subtract(v2, v0);

//     Vec3 pvec = cross(direction, edge2);
//     double det = dot(edge1, pvec);
    
//     if (det > -EPSILON && det < EPSILON) return false;
    
//     double invDet = 1.0f / det;
//     Vec3 tvec = subtract(origin, v0);

//     u = dot(tvec, pvec) * invDet;
//     if (u < 0 || u > 1) return false;

//     Vec3 qvec = cross(tvec, edge1);
//     v = dot(direction, qvec) * invDet;
//     if (v < 0 || u + v > 1) return false;

//     t = dot(edge2, qvec) * invDet;
//     return true;
// }

// __global__ void checkIntersections(const Triangle* triangles1, const Triangle* triangles2, int n1, int n2, bool* intersect) {
//     int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
//     int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

//     if (idx1 < n1 && idx2 < n2) {
//         const Triangle& tri1 = triangles1[idx1];
//         const Triangle& tri2 = triangles2[idx2];
        
//         double t, u, v;
//         for (int i = 0; i < 3; ++i) {
//             Vec3 origin = tri1[i];
//             Vec3 direction = subtract(tri1[(i + 1) % 3], origin);
//             if (rayIntersectsTriangle(origin, direction, tri2, t, u, v)) {
//                 *intersect = true;
//                 return;
//             }
//         }
//         for (int i = 0; i < 3; ++i) {
//             Vec3 origin = tri2[i];
//             Vec3 direction = subtract(tri2[(i + 1) % 3], origin);
//             if (rayIntersectsTriangle(origin, direction, tri1, t, u, v)) {
//                 *intersect = true;
//                 return;
//             }
//         }
//     }
// }

// std::vector<Triangle> readTrianglesFromOFF(const std::string& filename) {
//     std::ifstream infile(filename);
//     if (!infile) {
//         std::cerr << "Error opening file: " << filename << std::endl;
//         exit(1);
//     }

//     std::string header;
//     infile >> header;
//     if (header != "OFF") {
//         std::cerr << "Not a valid OFF file." << std::endl;
//         exit(1);
//     }

//     int numVertices, numFaces, numEdges;
//     infile >> numVertices >> numFaces >> numEdges;

//     std::vector<Vec3> vertices(numVertices);
//     for (int i = 0; i < numVertices; ++i) {
//         infile >> vertices[i][0] >> vertices[i][1] >> vertices[i][2];
//     }

//     std::vector<Triangle> triangles;
//     for (int i = 0; i < numFaces; ++i) {
//         int vertexCount, v1, v2, v3;
//         infile >> vertexCount >> v1 >> v2 >> v3;
//         if (vertexCount == 3) {
//             triangles.push_back({vertices[v1], vertices[v2], vertices[v3]});
//         }
//     }

//     return triangles;
// }

// int main() {
//     auto triangles1 = readTrianglesFromOFF("VH_F_renal_pyramid_L_a.off");
//     auto triangles2 = readTrianglesFromOFF("VH_F_renal_pyramid_L_b.off");

//     Triangle *d_triangles1, *d_triangles2;
//     bool *d_intersect, h_intersect = false;

//     cudaMalloc(&d_triangles1, triangles1.size() * sizeof(Triangle));
//     cudaMalloc(&d_triangles2, triangles2.size() * sizeof(Triangle));
//     cudaMalloc(&d_intersect, sizeof(bool));

//     cudaMemcpy(d_triangles1, triangles1.data(), triangles1.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_triangles2, triangles2.data(), triangles2.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_intersect, &h_intersect, sizeof(bool), cudaMemcpyHostToDevice);

//     dim3 blockSize(16, 16);
//     dim3 gridSize((triangles1.size() + blockSize.x - 1) / blockSize.x, 
//                   (triangles2.size() + blockSize.y - 1) / blockSize.y);

//     // CUDA event creation
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     // Record start time
//     cudaEventRecord(start);

//     // Launch kernel
//     checkIntersections<<<gridSize, blockSize>>>(d_triangles1, d_triangles2, triangles1.size(), triangles2.size(), d_intersect);

//     // Ensure kernel execution completes
//     cudaDeviceSynchronize();

//     // Record stop time
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);

//     // Calculate elapsed time
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);

//     cudaMemcpy(&h_intersect, d_intersect, sizeof(bool), cudaMemcpyDeviceToHost);

//     if (h_intersect) {
//         std::cout << "Triangles intersect!" << std::endl;
//     } else {
//         std::cout << "Triangles do not intersect." << std::endl;
//     }

//     std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

//     // Cleanup
//     cudaFree(d_triangles1);
//     cudaFree(d_triangles2);
//     cudaFree(d_intersect);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     return 0;
// }

