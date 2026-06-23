// Created by John Paul Naiju on June 1, 2026

#include "core/tensor.h"
#include <iostream>

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
    const std::size_t aligned_bytes = (bytes + 63) & ~63;
    data_ = static_cast<float*>(std::aligned_alloc(64, aligned_bytes));
}

Tensor::~Tensor() {
    if (data_ != nullptr) std::free(data_);
    data_ = nullptr;
}

Tensor::Tensor(const Tensor& tensor)
    : shape_(tensor.shape_), strides_(tensor.strides_), size_(tensor.size_) {
    const std::size_t bytes = size_ * sizeof(float);
    const std::size_t aligned_bytes = (bytes + 63) & ~63;
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
    const std::size_t aligned_bytes = (bytes + 63) & ~63;
    data_ = static_cast<float*>(std::aligned_alloc(64, aligned_bytes));
    if (size_ > 0) {
        std::memcpy(data_, tensor.data_, size_ * sizeof(float));
    }

    return *this;
}

Tensor::Tensor(Tensor&& tensor) noexcept
    : data_(tensor.data_),
    shape_(std::move(tensor.shape_)),
    strides_(std::move(tensor.strides_)),
    size_(tensor.size_)
{
    tensor.data_ = nullptr;
    tensor.size_ = 0;
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

void Tensor::fill_zeros() const {
    std::memset(data_, 0, size_ * sizeof(float));
}

void Tensor::print() const {
    std::cout << "Tensor shape: [";
    for (const unsigned long i : shape_) {
        std::cout << i << ", ";
    }
    std::cout << "]\n";

    if (size_ == 0) {
        std::cout << "Empty tensor\n";
        return;
    }

    if (shape_.size() == 1) {
        std::cout << "[";
        for (std::size_t i = 0; i < size_; ++i) {
            std::cout << data_[i] << ", ";
        }
        std::cout << "]\n";
    }else if (shape_.size() == 2) {
        for (std::size_t r = 0; r < shape_[0]; ++r) {
            std::cout << "[";
            for (std::size_t c = 0; c < shape_[1]; ++c) {
                std::cout << data_[r * shape_[1] + c] << ", ";
            }
            std::cout << "]\n";
        }
    }else {
        const std::size_t limit = std::min(size_, static_cast<std::size_t>(10));
        for (std::size_t i = 0; i < limit; ++i) {
            std::cout << data_[i] << ", ";
        }
        if (size_ > 10) {
            std::cout << ", ... , " << data_[size_ - 1];
        }
        std::cout << "]\n";
    }
}