// Created by John Paul Naiju on June 1, 2026

#include "core/tensor.h"
#include <cstdlib>
#include <cstring>

typedef std::vector<size_t> array_t;

Tensor::Tensor(const array_t& shape) {
    shape_ = shape;
    size_ = 1;

    for (const auto i : shape_) size_ *= i;

    strides_.resize(shape_.size());
    std::size_t current_stride = 1;

    for (std::size_t i = shape_.size(); i > 0; --i) {
        strides_[i-1] = current_stride;
        current_stride *= shape_[i-1];
    }

    const std::size_t bytes = size_ * sizeof(float);
    const std::size_t aligned_bytes = ((bytes + 63) / 64) * 64;
    data_ = static_cast<float*>(std::aligned_alloc(64, aligned_bytes));
}

Tensor::~Tensor() {
    std::free(data_);
}

float& Tensor::operator[](const size_t index) {
     return data_[index];
}

float Tensor::operator[](const size_t index) const {
    return data_[index];
}

void Tensor::fill_zeros() const {
    std::memset(data_, 0, size_ * sizeof(float));
}