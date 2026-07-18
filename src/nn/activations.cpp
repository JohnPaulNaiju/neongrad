#include "activations.h"
#include <arm_neon.h>


void ReLU::forward(Tensor& tensor) {
    float* data = tensor.data();
    const std::size_t size = tensor.padded_size();

    const float32x4_t zeros = vdupq_n_f32(0.0f);

    for (std::size_t i = 0; i < size; i += 4) {
        const float32x4_t vals = vld1q_f32(&data[i]);
        const float32x4_t relu_vals = vmaxq_f32(vals, zeros);
        vst1q_f32(&data[i], relu_vals);
    }

    cache_ = tensor;
}

Tensor ReLU::backward(Tensor& tensor) {

}

void Sigmoid::forward(Tensor& tensor) {

}

Tensor Sigmoid::backward(Tensor& tensor) {

}