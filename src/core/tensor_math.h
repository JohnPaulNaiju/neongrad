#pragma once

#include "core/tensor.h"

Tensor gemm(const Tensor& A, const Tensor& B);
Tensor gemm_naive(const Tensor& A, const Tensor& B);