#pragma once

#include "core/tensor.h"

namespace init {
    void he(Tensor& tensor, std::size_t fan_in);
    void xavier(Tensor& tensor, std::size_t fan_in, std::size_t fan_out);
}