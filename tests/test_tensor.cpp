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
        auto addr = reinterpret_cast<std::uintptr_t>(tensor.data());
        assert(addr % 64 == 0);

        std::cout << "Test 2 passed ✅ [memory alignment] \n";
    }
    return 0;
}
