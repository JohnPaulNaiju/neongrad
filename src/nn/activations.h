#pragma once

#include "core/tensor.h"

class ReLU {

public:
    void forward(Tensor& tensor);
    Tensor backward(Tensor& tensor);

private:
    Tensor cache_;
};

class Softmax {

public:
    void forward(Tensor& tensor);
    Tensor backward(Tensor& tensor);

private:
    Tensor cache_;
};