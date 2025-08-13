#ifndef GSPLAT_SYCL_HELPERS_HPP
#define GSPLAT_SYCL_HELPERS_HPP

#include<sycl/sycl.hpp>

template<typename T>
void gpuAtomicAdd(T* ptr, T value) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        protected_ref(*ptr);
    protected_ref.fetch_add(value);
}

#endif //GSPLAT_SYCL_HELPERS_HPP