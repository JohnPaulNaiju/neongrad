# High-Performance Deep Learning Engine (scratch) | C++, ARM NEON

GEMM benchmark results:

### 1) Matrix size: 512x512

- Tiled SIMD:
    - Average Time: 0.0142 seconds
    - Performance:  18.9293 GFLOPS
- Naive ikj:
    - Average Time: 0.0104 seconds
    - Performance:  25.7916 GFLOPS

### 2) Matrix size: 1024x1024

- Tiled SIMD:
    - Average Time: 0.1228 seconds
    - Performance:  17.4930 GFLOPS
- Naive ikj:
    - Average Time: 0.0854 seconds
    - Performance:  25.1457 GFLOPS

### 3) Matrix size: 2048x2048

- Tiled SIMD:
    - Average Time: 1.2082 seconds
    - Performance:  14.2189 GFLOPS
- Naive ikj:
    - Average Time: 0.8803 seconds
    - Performance:  19.5167 GFLOPS

### 3) Matrix size: 4096x4096

- Tiled SIMD:
    - Average Time: 10.0536 seconds
    - Performance: 13.6707 GFLOPS
- Naive ikj:
    - Average Time: 7.0469 seconds
    - Performance:  19.5036 GFLOPS