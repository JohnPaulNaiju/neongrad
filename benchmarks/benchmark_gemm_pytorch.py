import torch
import time
import os

def run_pytorch_benchmark():
    torch.set_num_threads(1)

    device = torch.device('cpu')
    threads = torch.get_num_threads()
    print(f"--- PyTorch GEMM Benchmark ---")
    print(f"Device: {device}")
    print(f"Threads: {threads}")
    print("-" * 55)

    sizes = [512, 1024, 2048, 4096]
    num_runs = 10

    print(f"{'Matrix Size':<15} | {'Time (ms)':<15} | {'GFLOPS':<15}")
    print("-" * 55)

    for N in sizes:
        A = torch.randn(N, N, dtype=torch.float32, device=device)
        B = torch.randn(N, N, dtype=torch.float32, device=device)

        for _ in range(3):
            _ = torch.matmul(A, B)

        start_time = time.perf_counter()

        for _ in range(num_runs):
            C = torch.matmul(A, B)

        end_time = time.perf_counter()

        avg_time_sec = (end_time - start_time) / num_runs
        avg_time_ms = avg_time_sec * 1000.0

        gflops = (2.0 * (N ** 3)) / avg_time_sec / 1e9

        print(f"{N}x{N:<10} | {avg_time_ms:<15.2f} | {gflops:<15.2f}")

if __name__ == "__main__":

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    run_pytorch_benchmark()