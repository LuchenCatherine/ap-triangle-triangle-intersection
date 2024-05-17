#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cuda_runtime.h>

using namespace std;

struct Point {
    float x, y, z;
};

struct Triangle {
    Point p1, p2, p3;
    Point edges[3];
};

__device__ float determinant(Point a, Point b, Point c) {
    return a.x * (b.y * c.z - b.z * c.y) -
           a.y * (b.x * c.z - b.z * c.x) +
           a.z * (b.x * c.y - b.y * c.x);
}

__device__ bool isLegalIntersection(float gamma, float delta) {
    return gamma >= 0 && gamma <= 1 && delta >= 0 && delta <= 1;
}

__global__ void intersectionTestKernel(Triangle *trianglesA, Triangle *trianglesB, bool *results, int numTrianglesA, int numTrianglesB) {
    extern __shared__ Triangle sharedTriangles[];
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    int idxB = blockIdx.y * blockDim.y + threadIdx.y;

    if (idxA < numTrianglesA && idxB < numTrianglesB) {
        Triangle A = trianglesA[idxA];
        Triangle B = trianglesB[idxB];

        // Load triangle B into shared memory
        sharedTriangles[threadIdx.x] = B;
        __syncthreads();

        // Perform intersection test
        Point beta;
bool intersected = false;
for (int i = 0; i < 3; ++i) {
    Point qi = A.edges[i];
    // Calculate beta.x
    Point A_qi = { A.p1.x, A.p2.x, qi.x };
    float det_A_qi = determinant({A_qi.x, A_qi.y, A_qi.z}, {A_qi.y, A_qi.z, A_qi.x}, {A_qi.z, A_qi.x, A_qi.y});

    Point ri = { qi.x - A.p1.x, qi.y - A.p1.y, qi.z - A.p1.z };
    Point A_ri = { A.p1.x, A.p2.x, ri.x };
    float det_A_ri = determinant({A_ri.x, A_ri.y, A_ri.z}, {A_ri.y, A_ri.z, A_ri.x}, {A_ri.z, A_ri.x, A_ri.y});

    beta.x = -det_A_qi / det_A_ri;

    // Check if beta.x is within [0, 1]
    if (beta.x >= 0 && beta.x <= 1) {
        // Calculate intersection point T
        Point T = { A.p1.x + beta.x * qi.x, A.p1.y + beta.x * qi.y, A.p1.z + beta.x * qi.z };

        // Calculate edge vector t
        Point t = { A.edges[1].x - A.edges[0].x, A.edges[1].y - A.edges[0].y, A.edges[1].z - A.edges[0].z };

        // Loop through edges of triangle B
        for (int j = 0; j < 3; ++j) {
            Point P = B.p1;
            Point p1 = B.edges[j];
            Point p2 = B.edges[(j + 1) % 3];
            Point p3 = { p2.x - p1.x, p2.y - p1.y, p2.z - p1.z }; // edge p3

            // Calculate determinants
            float det_p1 = determinant(P, p1, t);
            float det_p2 = determinant(T, t, p1);
            float det_p3 = determinant(P, p1, p3);
            float det_p4 = determinant(T, t, p3);

            float det_total = determinant(p3, {-t.x, -t.y, -t.z}, {det_p1, det_p2, det_p3});
            float gamma = determinant(Point{det_p2, det_p1, det_p4}, Point{det_p3, det_p4, det_total}, Point{0,0,0}) / det_total;
            float delta = determinant(Point{det_p2, det_p1, det_p3}, Point{det_p3, det_p4, det_total}, Point{0,0,0}) / det_total;

            // Check if intersection is legal
            if (isLegalIntersection(gamma, delta)) {
                intersected = true;
                break;
            }
        }
    }

    if (intersected)
        break; // If intersection found, no need to continue checking other edges
}

// Store result
results[idxA * numTrianglesB + idxB] = intersected;

        }
    }

std::vector<Triangle> readTrianglesFromOFF(const std::string& filename) {
    // Function unchanged
}

int main() {
    std::vector<Triangle> trianglesB = readTrianglesFromOFF("./VH_F_renal_pyramid_L_c.off");
    std::vector<Triangle> trianglesA = readTrianglesFromOFF("./VH_F_renal_pyramid_L_a.off");

    // Allocate memory on GPU for triangles and results
    Triangle *d_trianglesA, *d_trianglesB;
    bool *d_results;
    cudaMalloc(&d_trianglesA, trianglesA.size() * sizeof(Triangle));
    cudaMalloc(&d_trianglesB, trianglesB.size() * sizeof(Triangle));
    cudaMalloc(&d_results, trianglesA.size() * trianglesB.size() * sizeof(bool));

    // Transfer data from CPU to GPU
    cudaMemcpy(d_trianglesA, trianglesA.data(), trianglesA.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trianglesB, trianglesB.data(), trianglesB.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16); // 256 threads per block
    dim3 blocksPerGrid(
        (trianglesA.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (trianglesB.size() + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Calculate shared memory size
    int  = threadsharedMemSizesPerBlock.x * sizeof(Triangle);

    // Launch intersectionTestKernel
    intersectionTestKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_trianglesA, d_trianglesB, d_results, trianglesA.size(), trianglesB.size());

    // Transfer results back to CPU
    bool *results = new bool[trianglesA.size() * trianglesB.size()];
    cudaMemcpy(results, d_results, trianglesA.size() * trianglesB.size() * sizeof(bool), cudaMemcpyDeviceToHost);

    // Check results
    bool intersected = false;
    for (int i = 0; i < trianglesA.size() * trianglesB.size(); ++i) {
        if (results[i]) {
            intersected = true;
            break;
        }
    }

    if (intersected) {
        std::cout << "Triangles intersect." << std::endl;
    } else {
        std::cout << "No intersections found." << std::endl;
    }

    // Free GPU memory
    cudaFree(d_trianglesA);
    cudaFree(d_trianglesB);
    cudaFree(d_results);
    delete[] results;

    return 0;
}
