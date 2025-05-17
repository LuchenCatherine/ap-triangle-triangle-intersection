// get_compute_capability.cu
#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error getting device count: "
                  << cudaGetErrorString(err) << "\n";
        return 1;
    }
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found.\n";
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        std::cout << "Device " << dev << ": " << prop.name
                  << "\n  Compute capability: "
                  << prop.major << "." << prop.minor << "\n";
    }
    return 0;
}
