```markdown
#  GPU-Accelerated Triangle-Triangle Intersection

This project performs GPU-based triangle-triangle intersection testing between neuroimaging meshes (e.g., brain surfaces) and whole-body meshes using CUDA. It's designed for **high-throughput collision detection**, especially in **neuroscience simulation and validation tasks**.

## Folder Structure

````
saivaibhavkondapaka_ap_gpu/  
├── src/  
│ └── sai_tri_intersect_gpu.cu # CUDA source code with triangle-triangle intersection logic  
├── build/  
│ ├── sai_tri_gpu # Compiled binary (generated after build)  
│ └── *.tsv # Benchmark result files  
├── meshes/  
│ └── *.off # Input 3D mesh files (OFF format)  
├── run_inter_mesh.sh # Script to batch process mesh pairs  
└── README.md  

````

##  How to Compile

Make sure you have CUDA installed (e.g., with `nvcc` available).

###  Steps

```bash
cd saivaibhavkondapaka_ap_gpu/src
mkdir -p ../build

nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
    -I../src sai_tri_intersect_gpu.cu -o ../build/sai_tri_gpu
````

> Replace `compute_86` and `sm_86` with your GPU's architecture if needed.

---

##  How to Run

Run the compiled program from the `build` folder:

```bash
cd ../build
./sai_tri_gpu ../meshes/MeshA.off ../meshes/MeshB.off 1000000
```

###  Parameters

* `MeshA.off`: Path to first mesh file
* `MeshB.off`: Path to second mesh file
* `1000000`: Batch size (controls how many triangle-pairs are processed at a time)

###  Example

```bash
./sai_tri_gpu ../meshes/VH_F_skin.off ../meshes/VH_F_skin.off 1000000
```

---

##  Output Format

```
<ExecutionTime_ms>    <Mpairs/s>    <Intersect_Flag>    <Intersection_Count>
```

###  Example Output

```
7163.54    20438.7    1    3664085
```

* **Time (ms)**: Total execution time (host-device copy + kernel)
* **Mpairs/s**: Million triangle-pairs checked per second
* **Intersect\_Flag**: 1 if any intersection, else 0
* **Intersection\_Count**: Total intersecting triangle pairs

---

##  Batch Testing Script (Optional)

You can run batch tests using `run_inter_mesh.sh`:

```bash
bash run_inter_mesh.sh
```

It loops through mesh pairs and appends results to a `.tsv` file.

---

##  Benchmarking Results

Result files like `vh_skin_vs_skin_results.tsv` or `inter_mesh_results.tsv` contain:

| MeshA           | MeshB           | FacesA | FacesB | Pairs | Time\_ms | Mpairs\_s | Intersected | Intersect\_Count |
| --------------- | --------------- | ------ | ------ | ----- | -------- | --------- | ----------- | ---------------- |
| VH\_F\_skin.off | VH\_M\_skin.off | ...    | ...    | ...   | ...      | ...       | 0           | 0                |

---

##  Mesh Format (.OFF)

Each `.off` file contains:

```
OFF
<num_vertices> <num_faces> <num_edges>
x y z           # vertex lines
3 i j k         # face lines (triangulated during load)
```

---

##  Features

* Supports large-scale triangle pairs (e.g., >300M pairs)
* Batch-based GPU processing to prevent memory overflow
* Accurate detection via:

  * AABB pruning
  * Plane orientation tests
  * Signed volume method

---

##  Future Optimizations

| Idea                                 | Benefit                                       |
| ------------------------------------ | --------------------------------------------- |
| **Bounding Volume Hierarchy (BVH)**  | Reduces O(n²) → O(n log n) comparisons        |
| **`thrust::reduce` for counting**    | GPU-side intersection counting                |
| **Multi-GPU Scaling**                | Use NCCL to spread batches                    |
| **Warp-level pruning**               | Use `__shfl_sync` to skip unnecessary threads |
| **Persistent kernels / CUDA Graphs** | Lower kernel launch overheads                 |

---

##  Author

**Sai Vaibhav Kondapaka**
GitHub: [vaibhav-1608](https://github.com/vaibhav-1608)

```

---

```
