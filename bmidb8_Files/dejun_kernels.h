// dejun_kernels.h
#ifndef DEJUN_KERNELS_H
#define DEJUN_KERNELS_H

#include <cstdint>

// Structure matching the result format in Dejun.cu
struct result_container {
    bool     intersected;
    float    distance;
    uint32_t p1;
    uint32_t p2;
    // ... add any additional fields if present in Dejun.cu ...
};

// GPU buffer descriptor expected by the intersection kernel
struct gpu_info {
    float            *d_data;       // triangle vertex data
    uint32_t         *d_os;         // offset-size descriptor
    result_container *d_intersect;  // per-pair result
};

// Intersection batch entrypoint defined in Dejun.cu
void TriInt_batch_gpu(
    gpu_info        *gpu,
    const float     *data,
    const uint32_t  *offset_size,
    const float     *hausdorff,
    result_container*u_intersect,
    uint32_t         pair_num,
    uint32_t         triangle_num
);

#endif // DEJUN_KERNELS_H
