#include "core/tensor_math.h"
#include <cassert>
#include <arm_neon.h>

Tensor gemm_ikj(const Tensor& A, const Tensor& B) {

    const std::size_t m1 = A.shape().rbegin()[1];
    const std::size_t n1 = A.shape().back();
    const std::size_t m2 = B.shape().rbegin()[1];
    const std::size_t n2 = B.shape().back();

    assert(n1 == m2 && "Inner dimensions of matrix do not match");

    Tensor C({m1, n2});

    C.fill_zeros();

    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();

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