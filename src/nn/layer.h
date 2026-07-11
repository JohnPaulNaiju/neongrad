// Created by John Paul Naiju on 07/07/26.

#pragma once

#include "core/tensor.h"

class Dense {

public:

    explicit Dense(std::size_t input_dim, std::size_t output_dim, std::size_t batch_size);

    [[nodiscard]] Tensor forward(const Tensor& tensor);

private:
    Tensor weights_;
    Tensor bias_;
    Tensor cache_;
};