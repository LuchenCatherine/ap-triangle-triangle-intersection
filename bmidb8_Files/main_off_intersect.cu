// main_off_intersect.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cassert>
#include <cuda_runtime.h>
#include "dejun_kernels.h"

// Simple OFF reader for triangular meshes
bool loadOFF(const std::string &filename,
             std::vector<float> &out_tri_data,
             uint32_t           &out_tri_count)
{
    std::ifstream in(filename);
    if (!in || !(in >> std::ws) || in.peek() == EOF) return false;
    std::string header;
    std::getline(in, header);
    if (header != "OFF") return false;

    uint32_t nVerts=0, nFaces=0, nEdges=0;
    in >> nVerts >> nFaces >> nEdges;
    if (!in) return false;

    std::vector<std::array<float,3>> verts(nVerts);
    for (uint32_t i = 0; i < nVerts; ++i) {
        in >> verts[i][0] >> verts[i][1] >> verts[i][2];
        if (!in) return false;
    }

    out_tri_count = nFaces;
    out_tri_data.reserve(size_t(nFaces) * 9);
    for (uint32_t f = 0; f < nFaces; ++f) {
        uint32_t vcnt, a, b, c;
        in >> vcnt >> a >> b >> c;
        if (!in || vcnt != 3) return false;
        auto &A = verts[a], &B = verts[b], &C = verts[c];
        out_tri_data.insert(out_tri_data.end(), {
            A[0], A[1], A[2],
            B[0], B[1], B[2],
            C[0], C[1], C[2]
        });
    }
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " mesh1.off mesh2.off\n";
        return 1;
    }

    // 1) Load both meshes into a single float buffer
    std::vector<float> tri_data;
    uint32_t nTri1 = 0, nTri2 = 0;
    if (!loadOFF(argv[1], tri_data, nTri1)) {
        std::cerr << "Failed to load: " << argv[1] << "\n";
        return 1;
    }
    uint32_t offset2 = nTri1;
    if (!loadOFF(argv[2], tri_data, nTri2)) {
        std::cerr << "Failed to load: " << argv[2] << "\n";
        return 1;
    }

    // 2) Prepare the single pair descriptor: {offset1, size1, offset2, size2}
    uint32_t offset_size[4] = { 0, nTri1, offset2, nTri2 };

    // 3) Allocate device buffers directly
    float            *d_data;
    uint32_t         *d_os;
    result_container *d_intersect;
    size_t data_bytes   = tri_data.size() * sizeof(float);
    size_t os_bytes     = 4 * sizeof(uint32_t);
    size_t result_bytes = sizeof(result_container);

    cudaMalloc(&d_data,       data_bytes);
    cudaMalloc(&d_os,         os_bytes);
    cudaMalloc(&d_intersect,  result_bytes);

    // 4) Upload data and offsets to device
    cudaMemcpy(d_data,      tri_data.data(), data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_os,        offset_size,      os_bytes,   cudaMemcpyHostToDevice);
    // zero the result container on device
    cudaMemset(d_intersect, 0, result_bytes);

    // 5) Prepare gpu_info and invoke the intersection batch
    gpu_info gpu{ d_data, d_os, d_intersect };
    TriInt_batch_gpu(&gpu,
                     d_data,
                     d_os,
                     /*hausdorff=*/nullptr,
                     d_intersect,
                     /*pair_num=*/1,
                     /*triangle_num=*/nTri1 + nTri2);

    // 6) Copy back the result and print
    result_container host_result;
    cudaMemcpy(&host_result, d_intersect, result_bytes, cudaMemcpyDeviceToHost);
    std::cout << "Meshes "
              << (host_result.intersected ? "DO intersect\n"
                                            : "do NOT intersect\n");

    // 7) Cleanup
    cudaFree(d_data);
    cudaFree(d_os);
    cudaFree(d_intersect);
    return 0;
}
