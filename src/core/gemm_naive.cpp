// Created by John Paul Naiju on 07/07/26.

#include "core/gemm_naive.h"

//Naive ikj GEMM
Tensor gemm_naive(const Tensor& A, const Tensor& B) {

    const std::size_t m1 = A.shape().rbegin()[1];
    const std::size_t n1 = A.shape().back();
    const std::size_t n2 = B.shape().back();

    Tensor C({m1, n2});

    for (std::size_t i = 0; i < m1; ++i) {
        for (std::size_t k = 0; k < n1; ++k) {

            const float a_ik = A(i, k);

            for (std::size_t j = 0; j < n2; ++j) {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }

    return C;
}