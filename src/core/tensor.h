// Created by John Paul Naiju on June 1, 2026

#pragma once

#include <vector>
#include <cstddef>

typedef std::vector<size_t> array_t;

class Tensor {

public:
    explicit Tensor(const array_t& shape);
    ~Tensor();

    Tensor(const Tensor& tensor);
    Tensor& operator=(const Tensor& tensor);

    Tensor(Tensor&& tensor) noexcept;
    Tensor& operator=(Tensor&& tensor) noexcept;

    float& operator[](size_t index);
    float operator[](size_t index) const;

    template <typename... Args>
    float& operator()(Args... indices){
        std::size_t coords[] = { static_cast<std::size_t>(indices)... };

        std::size_t index = 0;

        for (int i = 0; i < static_cast<int>(sizeof...(Args)); ++i) {
            index += coords[i] * strides_[i];
        }

        return data_[index];
    }

    [[nodiscard]] const array_t& shape() const {
        return shape_;
    }

    [[nodiscard]] const float* data() const {
        return data_;
    }

    [[nodiscard]] std::size_t size() const {
        return size_;
    }

    [[nodiscard]] std::size_t ndim() const {
        return shape_.size();
    }

    void fill_zeros() const;

    void print() const;

private:
    float* data_ = nullptr;
    array_t shape_;
    array_t strides_;
    std::size_t size_;
};