#include <iostream>
#include <cassert>
#include "core/tensor.h"
#include <cstddef>

int main() {
    {
        const Tensor tensor({3, 4});
        assert(tensor.ndim() == 2);
        assert(tensor.shape()[0] == 3);
        assert(tensor.shape()[1] == 4);
        assert(tensor.size() == 12);

        std::cout << "Test 1 passed ✅ [constructor shape and size] \n";
    }

    {
        const Tensor tensor({8, 8});
        assert(tensor.data() != nullptr);
        const auto addr = reinterpret_cast<std::uintptr_t>(tensor.data());
        assert(addr % 64 == 0);

        std::cout << "Test 2 passed ✅ [memory alignment] \n";
    }

    {
        Tensor tensor({2, 2});
        tensor[0] = 1.0f;
        tensor[1] = 2.0f;
        tensor[2] = 3.0f;
        tensor[3] = 4.0f;

        assert(tensor[0] == 1.0f);
        assert(tensor[1] == 2.0f);
        assert(tensor[2] == 3.0f);
        assert(tensor[3] == 4.0f);

        std::cout << "Test 3 passed ✅ [operator[] read & write] \n";
    }

    {
        Tensor tensor({3, 4});
        tensor(2, 3) = 1.0f;
        assert(tensor[11] == 1.0f);

        std::cout << "Test 4 passed ✅ [operator() accessibility] \n";
    }

    {
        Tensor tensor({2, 2});
        tensor.fill_zeros();
        const float* data = tensor.data();
        for (std::size_t i = 0; i < tensor.size(); i++) {
            assert(data[i] == 0);
        }

        std::cout << "Test 5 passed ✅ [fill zeros] \n";
    }

    {
        Tensor tensor1({2, 2});
        tensor1[0] = 1.0f;
        tensor1[1] = 2.0f;
        tensor1[2] = 3.0f;
        tensor1[3] = 4.0f;

        Tensor tensor2 = tensor1;
        assert(tensor1.size() == tensor2.size());
        assert(tensor2[0] == tensor1[0]);

        tensor2[0] = 99.0f;
        assert(tensor1[0] == 1.0f && "FAIL: Shallow copy!");

        std::cout << "Test 6 passed ✅ [copy constructor] \n";
    }

    {
        Tensor tensor1({2, 2});
        tensor1[0] = 1.0f;

        Tensor tensor2({10, 10});
        tensor2 = tensor1;

        assert(tensor1.size() == tensor2.size());
        assert(tensor2[0] == 1.0f);

        std::cout << "Test 7 passed ✅ [copy assignment] \n";
    }

    {
        Tensor tensor1({2, 2});
        tensor1[0] = 1.0f;
        Tensor tensor2 = std::move(tensor1);

        assert(tensor2.size() == 4);
        assert(tensor2[0] == 1.0f);

        std::cout << "Test 8 passed ✅ [move constructor] \n";
    }

    {
        Tensor tensor1({2, 2});
        tensor1[0] = 1.0f;
        Tensor tensor2({3, 3});

        tensor2 = std::move(tensor1);

        assert(tensor2.size() == 4);
        assert(tensor2[0] == 1.0f);

        std::cout << "Test 8 passed ✅ [move assignment] \n";
    }
    return 0;
}
