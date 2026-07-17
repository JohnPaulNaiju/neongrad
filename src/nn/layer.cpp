#include "layer.h"
#include "initializer.h"
#include "core/tensor_math.h"


Dense::Dense(std::size_t input_dim, std::size_t output_dim, std::size_t batch_size, InitType init_type) :
    weights_({input_dim, output_dim}),
    bias_({1, output_dim}),
    cache_({batch_size, input_dim})
{
    switch (init_type) {
        case InitType::He:
            init::he(weights_, input_dim);
            break;
        case InitType::Xavier:
            init::xavier(weights_, input_dim, output_dim);
            break;
    }
}

Tensor Dense::forward(const Tensor& tensor) {
    cache_ = tensor;
    return gemm(tensor, weights_, bias_);
}

void Dense::backward() {

}