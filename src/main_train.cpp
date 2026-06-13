// Created by John Paul Naiju on June 2, 2026

#include <iostream>
#include "core/tensor.h"

int main() {
    Tensor tensor({32, 3, 28, 28});
    tensor[0] = 5.3f;
    std::cout << tensor[0] << "\n" << tensor[1] << std::endl;

    return 0;
}
