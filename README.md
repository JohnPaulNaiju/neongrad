# High-Performance Deep Learning Engine (scratch) | C++, ARM NEON


## GEMM Benchmark Analysis — Apple M1 (Single Core)

### Performance Comparison

| Matrix Size | Naive ikj (GFLOPS) | Tiled SIMD (GFLOPS) | Speedup | Time Reduction | Naive Time | SIMD Time |
|---|---|---|---|---|---|---|
| 512×512 | 5.91 | 18.26 | **3.09×** | 67.6% faster | 45.4 ms | 14.7 ms |
| 1024×1024 | 6.06 | 16.34 | **2.70×** | 62.9% faster | 354.6 ms | 131.4 ms |
| 2048×2048 | 5.88 | 15.76 | **2.68×** | 62.7% faster | 2.92 s | 1.09 s |
| 4096×4096 | 5.69 | 15.59 | **2.74×** | 63.5% faster | 24.15 s | 8.82 s |

### Key observations

| Metric | Value |
|---|---|
| **Average speedup** | **2.80×** across all sizes |
| **Peak throughput** | **18.26 GFLOPS** (512×512) |
| **Sustained throughput** | **~15.7 GFLOPS** (2048+ — cache-limited) |
| **Naive throughput** | ~5.9 GFLOPS (consistent, memory-bound) |
| **Naive scales down** | 5.91 → 5.69 GFLOPS (512→4096) — cache pressure hurts |
| **SIMD scales down** | 18.26 → 15.59 GFLOPS (512→4096) — tiling limits the damage |

### What the numbers mean

- **Naive stays flat at ~6 GFLOPS** regardless of size — it's bottlenecked by memory bandwidth, not compute. The CPU processes one float per cycle, but spends most time waiting for cache misses.

- **SIMD peaks at 18.3 GFLOPS at 512** — small enough for tiles to fit in L1/L2 comfortably. All 4 NEON lanes firing, minimal cache misses.

- **SIMD drops to ~15.6 at 4096** — even with tiling, some inter-tile cache pressure remains. But the drop is only 15%, vs a theoretical ~5× drop without tiling. **Tiling is protecting the throughput.**

- **M1 theoretical single-core peak**: ~50 GFLOPS (1 P-core × 2 FMA pipes × 4 lanes × ~3.2 GHz × 2 ops). We're at **31-37% of peak** — reasonable for a first implementation. For more speedup like 75% we need to go assembly.

### PyTorch GEMM Benchmark

- Device: cpu
- Threads: 1

| Matrix Size | Time (ms) | GFLOPS
|---|---|---|
| 512x512 | 0.28 | 958.70         
| 1024x1024 | 5.03 | 426.96         
| 2048x2048 | 36.28 | 473.59         
| 4096x4096 | 194.95 | 704.99

> Why is PyTorch faster?
> PyTorch uses Apple's Matrix Coprocessor, AMX which is a pure hardware accelerator for GEMM.
> I can also use AMX get same results, but It is a blackbox and removes my purpose of learning.