#include "core/tensor_math.h"
#include <arm_neon.h>

Tensor gemm(const Tensor& A, const Tensor& B) {

    const std::size_t m1 = A.shape().rbegin()[1];
    const std::size_t n1 = A.shape().back();
    const std::size_t n2 = B.shape().back();

    Tensor C({m1, n2});

    C.fill_zeros();

    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();

    for (std::size_t i = 0; i < m1; ++i) {
        for (std::size_t k = 0; k < n1; ++k) {
            const float a_ik = a_ptr[i * n1 + k];
            const float32x4_t v_a = vdupq_n_f32(a_ik);

            std::size_t j = 0;

            for (; j + 15 < n2; j += 16) {

                const float* b_row = &b_ptr[k * n2 + j];
                float* c_row = &c_ptr[i * n2 + j];

                float32x4_t v_c0 = vld1q_f32(c_row);
                float32x4_t v_c1 = vld1q_f32(c_row + 4);
                float32x4_t v_c2 = vld1q_f32(c_row + 8);
                float32x4_t v_c3 = vld1q_f32(c_row + 12);

                const float32x4_t v_b0 = vld1q_f32(b_row);
                const float32x4_t v_b1 = vld1q_f32(b_row + 4);
                const float32x4_t v_b2 = vld1q_f32(b_row + 8);
                const float32x4_t v_b3 = vld1q_f32(b_row + 12);

                v_c0 = vfmaq_f32(v_c0, v_b0, v_a);
                v_c1 = vfmaq_f32(v_c1, v_b1, v_a);
                v_c2 = vfmaq_f32(v_c2, v_b2, v_a);
                v_c3 = vfmaq_f32(v_c3, v_b3, v_a);

                vst1q_f32(c_row, v_c0);
                vst1q_f32(c_row + 4, v_c1);
                vst1q_f32(c_row + 8, v_c2);
                vst1q_f32(c_row + 12, v_c3);

            }

            for (; j + 3 < n2; j += 4) {
                const float* b_row = &b_ptr[k * n2 + j];
                float* c_row = &c_ptr[i * n2 + j];

                float32x4_t v_c = vld1q_f32(c_row);
                const float32x4_t v_b = vld1q_f32(b_row);

                v_c = vfmaq_f32(v_c, v_b, v_a);

                vst1q_f32(c_row, v_c);
            }

            for (; j < n2; ++j) {
                c_ptr[i * n2 + j] += a_ik * b_ptr[k * n2 + j];
            }
        }
    }

    return C;
}

Tensor gemm_naive(const Tensor& A, const Tensor& B) {

    const std::size_t m1 = A.shape().rbegin()[1];
    const std::size_t n1 = A.shape().back();
    const std::size_t n2 = B.shape().back();

    Tensor C({m1, n2});

    C.fill_zeros();

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