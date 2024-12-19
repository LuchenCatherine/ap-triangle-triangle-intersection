# 3D Spatial Database Acceleration with GPUs

## Project Overview
This repository contains the implementation of a GPU-accelerated system for 3D spatial database acceleration, focusing on optimized triangle-triangle intersection tests for collision detection. Inspired by Tomas Müller's methods, the project integrates advanced geometric computations with multi-core CPU and GPU parallelization to achieve real-time performance.

## Key Features
- **Optimized Algorithms**: High-performance implementation of the Möller-Trumbore algorithm, enhanced with bounding box checks and early termination techniques.
- **CUDA Parallelization**: Leveraged GPU parallelism using CUDA to handle billions of triangle pairs in minimal time.
- **High Precision**: Advanced floating-point tolerances address numerical inaccuracies.
- **Applications**: Suitable for diverse domains, including autonomous driving, gaming, and biomedical simulations.

## Results
# A. Performance Metrics
- **Small Meshes**: Achieved up to 7% faster execution than traditional methods on CPUs.
- **Large Meshes**: GPU parallelization resulted in 50-60x speedups compared to CPU-based approaches.
- **Memory Efficiency**: Optimized memory layout reduced data transfer overhead significantly.

# B. Scalability
Successfully processed billions of triangle pairs in under a second, demonstrating exceptional scalability for real-time applications. 

# C. Future Scope
- **Integration with Databases**: Incorporate GPU-accelerated engines into relational database systems like PostgreSQL for seamless query execution.
- **Mesh Compression**: Develop methods to compress large 3D meshes for reduced memory usage and faster processing.
- **Advanced Algorithms**: Explore alternative intersection detection methods to further optimize performance.
- **Application Expansion**: Extend use cases to include VR simulations, robotics, and large-scale scientific modeling.
- **Dynamic Environments**: Adapt the system to handle dynamic 3D scenes with real-time updates.

# Project Contributors
- **Shreyas Habade** (115911132)
- **Sarthak Madaan** (115224027)

# Advisors
- **Dr. Prof. Fusheng Wang**
- ** Ms. Lu Chen**

## References
- Oren Tropp, Ayellet Tal, Ilan Shimshoni: A fast triangle to triangle intersection test for collision detection. Computer Animation Virtual Worlds 17(50), 527–535 (2006).
- Vera Skorkovská, Ivana Kolingerová, and Bedrich Benes: A Simple and Robust Approach to Computation of Meshes Intersection. VISIGRAPP (1: GRAPP), (2018), pp. 175–82.
- Shinji Sakane, Tomohiro Takaki & Takayuki Aoki: Parallel-GPU-accelerated adaptive mesh refinement for three-dimensional phase-field simulation of dendritic growth during solidification of binary alloy. Materials Theory, Volume 6, issue 1, article id 3 (2022).
- Möller, Tomas: A Fast Triangle-Triangle Intersection Test. Journal of Graphics Tools, vol. 2, no. 2, (1997), pp. 25-30.
- Xiao, Lei & Mei, Gang & Cuomo, Salvatore & Xu, Nengxiong: Comparative Investigation of GPU-Accelerated Triangle-Triangle Intersection Algorithms for Collision Detection. Multimedia Tools and Applications. (2022).
- Teng D, Liang Y, Baig F, Kong J, Hoang V, Wang F. 3DPro: Querying Complex Three-Dimensional Data with Progressive Compression and Refinement. Adv Database Technol. 2022 Mar-Apr;25(2):104-117.
- Lu Chen, Dejun Teng, Tian Zhu, Jun Kong, Bruce W. Herr, Andreas Bueckle, Katy Börner, and Fusheng Wang: Real-time spatial registration for 3D human atlas. ACM SIGSPATIAL International Workshop on Analytics for Big Geospatial Data (BigSpatial '22). Association for 
  Computing Machinery, New York, NY, USA, 27–35.
- Villa Real, Lucas & Silva, Bruno: Full Speed Ahead: 3D Spatial Database Acceleration with GPUs. (2018).
