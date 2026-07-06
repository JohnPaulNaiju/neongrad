// Created by John Paul Naiju on 05/07/26.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include "core/tensor.h"
#include "core/tensor_math.h"

void fill_random(Tensor& t) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dis(-1.0f, 1.0f);

    float* ptr = t.data();

    for (std::size_t i = 0; i < t.size(); ++i) {
        ptr[i] = dis(gen);
    }
}

int main() {

    constexpr int size = 512;

    std::cout << "Matrix size: " << size << "x" << size << "\n";

    Tensor tensor1({size, size});
    Tensor tensor2({size, size});

    std::cout << "Filling matrices with random values...\n";

    fill_random(tensor1);
    fill_random(tensor2);

    std::cout << "Warming up cpu...\n";

    Tensor tensor3 = gemm(tensor1, tensor2);
    std::cout << "Sanity check 1 [0]: " << tensor3[0] << "\n";

    std::cout << "+-----------------------------------+\n";

    std::cout << "SIMD GEMM BENCHMARK\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i) {
        tensor3 = gemm(tensor1, tensor2);
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Sanity check 2 [0]: " << tensor3[0] << "\n";

    std::chrono::duration<double> total_duration = end_time - start_time;
    double avg_time_sec = total_duration.count() / 10;

    constexpr double total_ops = 2.0 * size * size * size;

    double gflops = total_ops / (avg_time_sec * 1e9);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Average Time: " << avg_time_sec << " seconds\n";
    std::cout << "Performance:  " << gflops << " GFLOPS\n";
    std::cout << "+-----------------------------------+\n";

    std::cout << "Naive ikj GEMM BENCHMARK\n";

    start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i) {
        tensor3 = gemm_naive(tensor1, tensor2);
    }

    end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Sanity check 3 [0]: " << tensor3[0] << "\n";

    total_duration = end_time - start_time;
    avg_time_sec = total_duration.count() / 10;

    gflops = total_ops / (avg_time_sec * 1e9);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Average Time: " << avg_time_sec << " seconds\n";
    std::cout << "Performance:  " << gflops << " GFLOPS\n";
    std::cout << "+-----------------------------------+\n";

    return 0;
}