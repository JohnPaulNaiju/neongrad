#pragma once

#include "core/tensor.h"

class ReLU {

public:
    Tensor forward(Tensor& tensor);
    Tensor backward(Tensor& tensor);

private:
    Tensor cache_;
};

class Softmax {

public:
    Tensor forward(Tensor& tensor);
    Tensor backward(Tensor& tensor);

private:
    Tensor cache_;
};