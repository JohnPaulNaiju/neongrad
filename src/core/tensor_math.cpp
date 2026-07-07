#include "core/tensor_math.h"
#include <arm_neon.h>

constexpr std::size_t tile_size = 64;
constexpr std::size_t j_tile_size = 256;

// ikj GEMM with SIMD, Loop Unrolling and Cache Tiling
Tensor gemm(const Tensor& A, const Tensor& B) {

    const std::size_t m1 = A.shape().rbegin()[1];
    const std::size_t n2 = B.shape().back();

    const std::size_t m1_pad = A.padded_shape().rbegin()[1];
    const std::size_t n1_pad = A.padded_shape().back();
    const std::size_t n2_pad = B.padded_shape().back();

    Tensor C({m1, n2});

    const float* __restrict a_ptr = A.data();
    const float* __restrict b_ptr = B.data();
    float* __restrict c_ptr = C.data();

    for (std::size_t i_out = 0; i_out < m1_pad; i_out += tile_size) {
        for (std::size_t j_out = 0; j_out < n2_pad; j_out += j_tile_size) {
            for (std::size_t k_out = 0; k_out < n1_pad; k_out += tile_size) {

                const std::size_t i_end = i_out + tile_size;
                const std::size_t j_end = j_out + j_tile_size;
                const std::size_t k_end = k_out + tile_size;

                for (std::size_t i = i_out; i < i_end; ++i) {
                    for (std::size_t k = k_out; k < k_end; ++k) {

                        const float a_ik = a_ptr[i * n1_pad + k];
                        const float32x4_t v_a = vdupq_n_f32(a_ik);

                        for (std::size_t j = j_out; j < j_end; j += 16) {

                            const float* b_row = &b_ptr[k * n2_pad + j];
                            float* c_row = &c_ptr[i * n2_pad + j];

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
                    }
                }
            }
        }
    }

    return C;
}

// Naive ikj GEMM
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