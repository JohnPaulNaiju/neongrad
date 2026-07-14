#include "nn/layer.h"
#include "core/tensor_math.h"

Dense::Dense(std::size_t input_dim, std::size_t output_dim, std::size_t batch_size) :
    weights_({input_dim, output_dim}),
    bias_({1, output_dim}),
    cache_({batch_size, input_dim})
{
    // initialize the weight and bias matrix
}

Tensor Dense::forward(const Tensor& tensor) {
    cache_ = tensor;
    return gemm(tensor, weights_, bias_);
}