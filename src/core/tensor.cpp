// Created by John Paul Naiju on June 1, 2026

#include "tensor.h"
#include <iostream>

typedef std::vector<std::size_t> array_t;

Tensor::Tensor(const array_t& shape) {
    shape_ = shape;
    size_ = 1;

    for (const auto i : shape_) size_ *= i;

    padded_shape_ = shape_;
    if (padded_shape_.size() >= 2) {
        const std::size_t rank = padded_shape_.size();
        padded_shape_[rank - 2] = (padded_shape_[rank - 2] + ALIGN_ROW - 1) / ALIGN_ROW * ALIGN_ROW;
        padded_shape_[rank - 1] = (padded_shape_[rank - 1] + ALIGN_COL - 1) / ALIGN_COL * ALIGN_COL;
    } else if (padded_shape_.size() == 1) {
        padded_shape_[0] = (padded_shape_[0] + ALIGN_COL - 1) / ALIGN_COL * ALIGN_COL;
    }

    strides_.resize(padded_shape_.size());
    std::size_t current_stride = 1;

    for (std::size_t i = shape_.size(); i > 0; --i) {
        strides_[i-1] = current_stride;
        current_stride *= padded_shape_[i-1];
    }

    padded_size_ = 1;
    for (const auto i : padded_shape_) padded_size_ *= i;

    const std::size_t bytes = padded_size_ * sizeof(float);
    const std::size_t aligned_bytes = (bytes + 63) & ~63;

    auto* raw_ptr = static_cast<float*>(std::aligned_alloc(64, aligned_bytes));
    std::memset(raw_ptr, 0, aligned_bytes);

    data_ = std::shared_ptr<float>(raw_ptr, std::free);
}

float& Tensor::operator[](const size_t index) {
     return data_.get()[index];
}

float Tensor::operator[](const size_t index) const {
    return data_.get()[index];
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
            std::cout << data_.get()[i] << ", ";
        }
        std::cout << "]\n";
    }else if (shape_.size() == 2) {
        for (std::size_t r = 0; r < shape_[0]; ++r) {
            std::cout << "[";
            for (std::size_t c = 0; c < shape_[1]; ++c) {
                std::cout << data_.get()[r * strides_[0] + c] << ", ";
            }
            std::cout << "]\n";
        }
    }else {
        const std::size_t limit = std::min(size_, static_cast<std::size_t>(10));
        for (std::size_t i = 0; i < limit; ++i) {
            std::cout << data_.get()[i] << ", ";
        }
        if (size_ > 10) {
            std::cout << ", ... , " << data_.get()[size_ - 1];
        }
        std::cout << "]\n";
    }
}