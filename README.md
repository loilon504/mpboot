# MPBoot & MPBootGPU

MPBoot is a tool for maximum parsimony phylogenetic tree search with ultrafast bootstrap.  
**MPBootGPU** is a GPU-accelerated extension that parallelises tree construction and bootstrap evaluation on NVIDIA GPUs using CUDA.

---

## MPBootGPU (GPU-accelerated)

### Requirements

- NVIDIA GPU with CUDA Compute Capability ≥ 8.0 (tested on A100 sm_80)
- CUDA Toolkit ≥ 12.0
- CMake ≥ 3.21
- GCC / Clang with C++14 support

### Building MPBootGPU on Linux

```bash
git clone https://github.com/loilon504/mpboot.git
mkdir build && cd build
cmake ../mpboot \
    -DUSE_GPU=ON \
    -DIQTREE_FLAGS=avx \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CXX_STANDARD=14 \
    -DCMAKE_CUDA_ARCHITECTURES=80
make -j8
```

> **`-DCMAKE_CUDA_ARCHITECTURES`**: set to match your GPU generation.  
> Common values: `80` (A100), `86` (RTX 3090/A30), `89` (RTX 4090), `90` (H100).  
> You can specify multiple: `-DCMAKE_CUDA_ARCHITECTURES="80;86"`.

The build produces `mpboot-avx` in the `build/` directory.

### GPU Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-use_gpu` | — | Enable GPU acceleration (required for all GPU modes) |
| `-gpu_device <id>` | `0` | CUDA device index to use (`nvidia-smi` to list devices) |
| `-gpu_worker <N>` | `200` | Number of parallel workers in hill-climbing phase (K2 blocks) |
| `-gpu_pool_size <N>` | `20` | Size of the shared candidate-tree pool |
| `-gpu_worker_stop <N>` | `1` | Early-stop threshold multiplier; stop after `N × gpu_worker` rounds without improvement |
| `-numpars <N>` | `100` | Number of initial parsimony trees built in parallel (Phase 1) |
| `-sprdist <d>` | `6` | SPR search radius (number of edges from pruning point) |
| `-cost <file>` | — | Cost matrix file for non-uniform (Sankoff) parsimony |

### Example Usage

**Tree search only (no bootstrap):**
```bash
./mpboot-avx -s alignment.phy -use_gpu -gpu_device 0 -gpu_worker 200
```

**Bootstrap (`-bb 1000`) with recommended settings:**
```bash
./mpboot-avx -s alignment.phy -use_gpu -bb 1000 \
    -gpu_device 0 \
    -gpu_worker 200 \
    -gpu_pool_size 20 \
    -gpu_worker_stop 1 \
    -numpars 200
```

**Non-uniform (Sankoff) cost matrix:**
```bash
./mpboot-avx -s alignment.phy -use_gpu -bb 1000 \
    -gpu_device 0 \
    -gpu_worker 200 \
    -cost matrix.cost
```

**Multi-GPU: run one instance per device using `CUDA_VISIBLE_DEVICES`:**
```bash
CUDA_VISIBLE_DEVICES=0 ./mpboot-avx -s data1.phy -use_gpu -gpu_device 0 &
CUDA_VISIBLE_DEVICES=1 ./mpboot-avx -s data2.phy -use_gpu -gpu_device 0 &
```

---

## MPBoot (CPU)

### Downloading source code

```bash
git clone https://github.com/diepthihoang/mpboot.git
```

### Compiling under Linux

```bash
mkdir build && cd build
cmake ../mpboot -DIQTREE_FLAGS=avx -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
make -j4
```

> Replace `avx` with `sse4` if your CPU does not support AVX.

The compiler generates `mpboot-avx` (or `mpboot` for SSE4).

**Run:**
```bash
./mpboot-avx -s example.phy
./mpboot-avx -s example.phy -bb 1000     # with bootstrap
```

### Compiling under Mac OS X

```bash
mkdir build && cd build
cmake ../mpboot -DIQTREE_FLAGS=avx
make -j4
```

### Compiling under Windows

Requirements: CMake ≥ 3.21, TDM-GCC

```bash
mkdir build && cd build
cmake -G "MinGW Makefiles" -DIQTREE_FLAGS=avx ../mpboot
mingw32-make -j4
```

> Replace `avx` with `sse4` for SSE architecture.  
> Do not use Clang on Windows due to vectorisation conflicts.

---

## MPBoot-MPI

### Downloading source code

```bash
git clone https://github.com/diepthihoang/mpboot.git
git checkout mpboot-mpi-sync    # synchronous version
# or
git checkout mpboot-mpi-async   # asynchronous version
```

### Compiling under Linux

```bash
mkdir build && cd build
cmake ../mpboot -DIQTREE_FLAGS=avx -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx
make -j4
```

**Run with 4 MPI processes:**
```bash
mpirun -np 4 ./mpboot-avx -s example.phy
```

### Compiling under Mac OS X

```bash
mkdir build && cd build
cmake ../mpboot -DIQTREE_FLAGS=avx -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx
make -j4
```

### Compiling under Windows

Requirements: CMake ≥ 3.21, TDM-GCC, MSMPI

```bash
mkdir build && cd build
cmake -G "MinGW Makefiles" -DIQTREE_FLAGS=mpiavx ../mpboot
mingw32-make -j4
```

**Run:**
```bash
mpiexec -n 4 ./mpboot-avx -s example.phy
```
