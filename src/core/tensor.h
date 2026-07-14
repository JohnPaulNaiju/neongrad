// Created by John Paul Naiju on June 1, 2026

#pragma once

#include <vector>
#include <cstddef>
#include <memory>

typedef std::vector<std::size_t> array_t;

class Tensor {

public:
    static constexpr std::size_t ALIGN_ROW = 64;
    static constexpr std::size_t ALIGN_COL = 256;

    explicit Tensor(const array_t& shape);

    Tensor(const Tensor& tensor) = default;
    Tensor& operator=(const Tensor& tensor) = default;

    Tensor(Tensor&& tensor) noexcept = default;
    Tensor& operator=(Tensor&& tensor) noexcept = default;

    ~Tensor() = default;

    float& operator[](size_t index);
    float operator[](size_t index) const;

    template <typename... Args>
    float& operator()(Args... indices) {
        std::size_t coords[] = { static_cast<std::size_t>(indices)... };

        std::size_t index = 0;

        for (int i = 0; i < static_cast<int>(sizeof...(Args)); ++i) {
            index += coords[i] * strides_[i];
        }

        return data_.get()[index];
    }

    template <typename... Args>
    float operator()(Args... indices) const {
        std::size_t coords[] = { static_cast<std::size_t>(indices)... };

        std::size_t index = 0;

        for (int i = 0; i < static_cast<int>(sizeof...(Args)); ++i) {
            index += coords[i] * strides_[i];
        }

        return data_.get()[index];
    }

    [[nodiscard]] const array_t& shape() const {
        return shape_;
    }

    [[nodiscard]] const array_t& padded_shape() const {
        return padded_shape_;
    }

    [[nodiscard]] const float* data() const {
        return data_.get();
    }

    [[nodiscard]] float* data() {
        return data_.get();
    }

    [[nodiscard]] std::size_t size() const {
        return size_;
    }

    [[nodiscard]] std::size_t ndim() const {
        return shape_.size();
    }

    void print() const;

private:
    std::shared_ptr<float> data_ = nullptr;
    array_t shape_;
    array_t padded_shape_;
    array_t strides_;
    std::size_t size_;
};