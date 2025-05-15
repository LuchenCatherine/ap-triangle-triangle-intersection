#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <stdexcept>

struct Vec3 {
    float x, y, z;
};

__host__ __device__ inline Vec3 operator-(const Vec3 &a, const Vec3 &b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}
__host__ __device__ inline Vec3 operator+(const Vec3 &a, const Vec3 &b) {
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}
__host__ __device__ inline Vec3 operator*(const Vec3 &v, float s) {
    return Vec3{v.x * s, v.y * s, v.z * s};
}

__host__ __device__ inline float dot(const Vec3 &a, const Vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}
__host__ __device__ inline float orient3D(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &d) {
    Vec3 ad = a - d, bd = b - d, cd = c - d;
    return dot(ad, cross(bd, cd));
}

__host__ __device__ bool triTriSingle(const Vec3* V, const Vec3* U) {
    constexpr float EPS = 1e-6f;
    for (int ax = 0; ax < 3; ++ax) {
        float minV = 1e30f, maxV = -1e30f, minU = 1e30f, maxU = -1e30f;
        for (int i = 0; i < 3; ++i) {
            float v = ((const float*)&V[i].x)[ax];
            float u = ((const float*)&U[i].x)[ax];
            minV = fminf(minV, v); maxV = fmaxf(maxV, v);
            minU = fminf(minU, u); maxU = fmaxf(maxU, u);
        }
        if (maxV < minU || maxU < minV) return false;
    }

    float du[3], dv[3];
    for (int i = 0; i < 3; ++i) {
        du[i] = orient3D(V[0], V[1], V[2], U[i]);
        dv[i] = orient3D(U[0], U[1], U[2], V[i]);
        if (fabsf(du[i]) < EPS) du[i] = 0;
        if (fabsf(dv[i]) < EPS) dv[i] = 0;
    }
    if ((du[0] > 0 && du[1] > 0 && du[2] > 0) || (du[0] < 0 && du[1] < 0 && du[2] < 0)) return false;
    if ((dv[0] > 0 && dv[1] > 0 && dv[2] > 0) || (dv[0] < 0 && dv[1] < 0 && dv[2] < 0)) return false;
    return true;
}

__global__ void kernelAllPairs(const Vec3* A, const Vec3* B, int nA, int nB, int offset, bool* R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nA * nB;
    int global_idx = offset + idx;
    if (global_idx >= total) return;

    int i = global_idx / nB;
    int j = global_idx % nB;
    R[idx] = triTriSingle(A + i * 3, B + j * 3);
}

std::vector<Vec3> loadOFF(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path);

    std::string hdr; in >> hdr;
    if (hdr != "OFF") throw std::runtime_error(path + " is not OFF");
    size_t nv, nf, ne; in >> nv >> nf >> ne;
    std::vector<Vec3> verts(nv);
    for (auto& v : verts) in >> v.x >> v.y >> v.z;

    std::vector<Vec3> tris;
    for (size_t i = 0; i < nf; ++i) {
        int vc; in >> vc;
        std::vector<size_t> idx(vc);
        for (int k = 0; k < vc; ++k) in >> idx[k];
        for (int k = 1; k + 1 < vc; ++k) {
            tris.push_back(verts[idx[0]]);
            tris.push_back(verts[idx[k]]);
            tris.push_back(verts[idx[k + 1]]);
        }
    }
    return tris;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " meshA.off meshB.off batch\n";
        return 1;
    }

    auto A = loadOFF(argv[1]);
    auto B = loadOFF(argv[2]);
    int nA = A.size() / 3, nB = B.size() / 3;
    size_t total = size_t(nA) * nB;
    int batch = std::stoi(argv[3]);

    Vec3 *dA, *dB; cudaMalloc(&dA, A.size() * sizeof(Vec3)); cudaMemcpy(dA, A.data(), A.size() * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMalloc(&dB, B.size() * sizeof(Vec3)); cudaMemcpy(dB, B.data(), B.size() * sizeof(Vec3), cudaMemcpyHostToDevice);

    bool* dR; cudaMalloc(&dR, batch * sizeof(bool));
    std::vector<uint8_t> Rv(batch);

    int intersect_count = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t done = 0; done < total; done += batch) {
        int curr = std::min<size_t>(batch, total - done);
        int grid = (curr + 255) / 256;
        kernelAllPairs<<<grid, 256>>>(dA, dB, nA, nB, done, dR);
        cudaMemcpy(Rv.data(), dR, curr * sizeof(bool), cudaMemcpyDeviceToHost);
        for (int i = 0; i < curr; ++i) intersect_count += Rv[i];
    }
    auto stop = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    int any = intersect_count > 0 ? 1 : 0;
    double Mpairs_s = double(total) / (ms * 1e3);
    std::cout << ms << "\t" << Mpairs_s << "\t" << any << "\t" << intersect_count << "\n";

    cudaFree(dA); cudaFree(dB); cudaFree(dR);
    return 0;
}
