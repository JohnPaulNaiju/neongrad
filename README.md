# High-Performance Deep Learning Engine | C++20, ARM NEON

A from-scratch deep learning inference and training engine written in pure C++20, featuring hand-optimized GEMM kernels with ARM NEON SIMD intrinsics, cache tiling, and OpenMP multi-core parallelism. **Zero external ML dependencies.**

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                    │
│               nn_train          nn_infer                │
├─────────────────────────────────────────────────────────┤
│                     Core Library (nn_core)              │
│                                                         │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │  Tensor  │  │  GEMM Engine │  │  Neural Network  │  │
│   │  N-D,    │  │  NEON SIMD   │  │  Layers, Loss,   │  │
│   │  Aligned │  │  Cache Tiled │  │  Backprop (WIP)  │  │
│   │  Memory  │  │  OpenMP      │  │                  │  │
│   └──────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────┤
│  Hardware: ARM NEON (128-bit SIMD) │ 4× P-Cores via OMP │
└─────────────────────────────────────────────────────────┘
```

## Optimization Techniques

| Technique | What it does |
|---|---|
| **ARM NEON SIMD** | Processes 4 floats per instruction via 128-bit vector registers (`vfmaq_f32` FMA) |
| **4× Loop Unrolling** | 16 floats per inner iteration — reduces branch overhead by 4× |
| **Cache Tiling** | 64×256 tiles keep working set in L1 cache (192KB on M1 P-core) |
| **Padded Allocation** | Rows/columns padded to tile boundaries — eliminates edge-case branching |
| **ikj Loop Order** | Inner loop sweeps contiguous memory — enables hardware prefetching |
| **`__restrict` Pointers** | Guarantees no pointer aliasing — unlocks compiler optimizations |
| **64-byte Aligned Memory** | `std::aligned_alloc` ensures cache-line aligned loads for SIMD |
| **OpenMP Parallelism** | `#pragma omp parallel for` distributes row-blocks across 4 P-cores |
| **`schedule(dynamic)`** | Dynamic work-stealing for load-balanced multi-core execution |

---

## GEMM Benchmark Results — Apple M1

### Single-Core: Tiled SIMD vs Scalar Baseline

> Scalar baseline compiled with `-fno-vectorize -fno-slp-vectorize` to prevent compiler auto-vectorization.

| Matrix Size | Scalar (GFLOPS) | Tiled SIMD (GFLOPS) | Speedup | Time: Scalar → SIMD |
|---|---|---|---|---|
| 512×512 | 5.91 | 18.26 | **3.09×** | 45.4 ms → 14.7 ms |
| 1024×1024 | 6.06 | 16.34 | **2.70×** | 354.6 ms → 131.4 ms |
| 2048×2048 | 5.88 | 15.76 | **2.68×** | 2.92 s → 1.09 s |
| 4096×4096 | 5.69 | 15.59 | **2.74×** | 24.15 s → 8.82 s |

---

### Multi-Core: Tiled SIMD + OpenMP (4 Performance Cores)

| Matrix Size | 1-Core SIMD (GFLOPS) | 4-Core SIMD (GFLOPS) | Core Scaling | vs Scalar |
|---|---|---|---|---|
| 512×512 | 18.26 | 53.53 | **2.93×** | **9.06× faster** |
| 1024×1024 | 16.34 | 58.14 | **3.56×** | **9.60× faster** |
| 2048×2048 | 15.76 | 62.00 | **3.93×** | **10.54× faster** |
| 4096×4096 | 15.59 | 56.38 | **3.62×** | **9.91× faster** |

---

### Summary

| Metric | Value |
|---|---|
| **Peak throughput (multi-core)** | **62.0 GFLOPS** @ 2048×2048 |
| **Peak throughput (single-core)** | **18.3 GFLOPS** @ 512×512 |
| **Max speedup vs scalar** | **10.54×** @ 2048×2048 |
| **Average SIMD speedup** | **2.80×** (single-core, across all sizes) |
| **Average core scaling** | **3.51×** across 4 P-cores (87.7% parallel efficiency) |
| **% of M1 theoretical peak** | **31% single-core, ~78% multi-core** (4-core peak ≈ 200 GFLOPS) |

---

### PyTorch Reference (Apple AMX Coprocessor)

| Matrix Size | Time (ms) | GFLOPS |
|---|---|---|
| 512×512 | 0.28 | 958.70 |
| 1024×1024 | 5.03 | 426.96 |
| 2048×2048 | 36.28 | 473.59 |
| 4096×4096 | 194.95 | 704.99 |

> PyTorch uses Apple's AMX (Apple Matrix Coprocessor), a dedicated hardware accelerator for matrix operations — not NEON SIMD.
> This engine intentionally uses only CPU NEON vector units for educational purposes and full transparency into the computation pipeline.

---

## Build

```bash
# Debug (with sanitizers)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build

# Release (optimized)
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release

# Run benchmark
./build-release/benchmark_gemm
```

### Requirements

- macOS with Apple Silicon (M1 or later)
- CMake 3.20+
- Clang/LLVM with C++20 support
- OpenMP (`brew install libomp`)

## Tech Stack

`C++20` · `ARM NEON SIMD` · `OpenMP` · `CMake` · `LLDB` · `ASan/UBSan` · `CTest` · `clang-format`