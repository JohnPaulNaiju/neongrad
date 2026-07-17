#include "initializer.h"
#include <random>
#include <cmath>


namespace {
    std::mt19937& get_generator() {
        static std::mt19937 generator(1337);
        return generator;
    }
}

namespace init {

    void he(Tensor& tensor, const std::size_t fan_in) {
        const auto limit = static_cast<float>(std::sqrt(6.0 / static_cast<double>(fan_in)));
        std::uniform_real_distribution distribution(-limit, limit);

        const std::size_t rows = tensor.shape()[0];
        const std::size_t cols = tensor.shape()[1];

        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                tensor(r, c) = distribution(get_generator());
            }
        }
    }

    void xavier(Tensor& tensor, const std::size_t fan_in, const std::size_t fan_out) {
        const auto limit = static_cast<float>(std::sqrt(6.0 / static_cast<double>(fan_in + fan_out)));
        std::uniform_real_distribution distribution(-limit, limit);

        const std::size_t rows = tensor.shape()[0];
        const std::size_t cols = tensor.shape()[1];

        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                tensor(r, c) = distribution(get_generator());
            }
        }
    }
}