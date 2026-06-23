#include <iostream>
#include <cassert>
#include "core/tensor.h"

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
        tensor[4] = 5.0f;
        tensor[5] = 6.0f;
        tensor[6] = 7.0f;
        tensor[7] = 8.0f;
        tensor[8] = 9.0f;
        tensor[9] = 10.0f;
        tensor[10] = 11.0f;
        tensor[11] = 12.0f;
        tensor[12] = 13.0f;
        tensor[13] = 14.0f;
        tensor[14] = 15.0f;
        tensor[15] = 16.0f;

        std::cout << "Test 3 passed ✅ [memory assignment] \n";
    }
    return 0;
}
