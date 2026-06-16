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
    if (data_ != nullptr) std::free(data_);
    data_ = nullptr;
}

Tensor::Tensor(const Tensor& tensor) : shape_(tensor.shape_), strides_(tensor.strides_), size_(tensor.size_) {
    const std::size_t bytes = size_ * sizeof(float);
    const std::size_t aligned_bytes = ((bytes + 63) / 64) * 64;
    data_ = static_cast<float*>(std::aligned_alloc(64, aligned_bytes));
    if (size_ > 0) {
        std::memcpy(data_, tensor.data_, size_ * sizeof(float));
    }
}

Tensor& Tensor::operator=(const Tensor& tensor) {
    if (this == &tensor) return *this;
    if (data_ != nullptr) std::free(data_);

    shape_ = tensor.shape_;
    strides_ = tensor.strides_;
    size_ = tensor.size_;

    const std::size_t bytes = size_ * sizeof(float);
    const std::size_t aligned_bytes = ((bytes + 63) / 64) * 64;
    data_ = static_cast<float*>(std::aligned_alloc(64, aligned_bytes));
    if (size_ > 0) {
        std::memcpy(data_, tensor.data_, size_ * sizeof(float));
    }

    return *this;
}

Tensor::Tensor(Tensor&& tensor) noexcept
    : data_(tensor.data_), shape_(tensor.shape_), strides_(tensor.strides_), size_(tensor.size_) {
    tensor.data_ = nullptr;
    tensor.size_ = 0;
    tensor.shape_.clear();
    tensor.strides_.clear();
}

Tensor& Tensor::operator=(Tensor&& tensor) noexcept {
    if (this == &tensor) return *this;
    if (data_ != nullptr) std::free(data_);

    shape_ = std::move(tensor.shape_);
    strides_ = std::move(tensor.strides_);
    size_ = tensor.size_;

    data_ = tensor.data_;

    tensor.data_ = nullptr;
    tensor.size_ = 0;

    return *this;
}

float& Tensor::operator[](const size_t index) {
     return data_[index];
}

float Tensor::operator[](const size_t index) const {
    return data_[index];
}

void Tensor::fill_zeros() {
    std::memset(data_, 0, size_ * sizeof(float));
}