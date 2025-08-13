#ifndef GSPLAT_SYCL_UTILS
#define GSPLAT_SYCL_UTILS


#include<sycl/sycl.hpp>

template <typename S, uint32_t N>
struct BufferType {
	using type = sycl::marray<S, N>;
    constexpr static bool isVec{false};
};

template <typename S>
struct BufferType<S, 2> {
	using type = sycl::vec<S, 2>;
    constexpr static bool isVec{true};
};

template <typename S>
struct BufferType<S, 3> {
	using type = sycl::vec<S, 3>;
    constexpr static bool isVec{true};
};

template <typename S>
struct BufferType<S, 4> {
	using type = sycl::vec<S, 4>;
    constexpr static bool isVec{true};
};

template <typename S>
struct BufferType<S, 8> {
	using type = sycl::vec<S, 8>;
    constexpr static bool isVec{true};
};

template <typename S>
struct BufferType<S, 16> {
	using type = sycl::vec<S, 16>;
    constexpr static bool isVec{true};
};

template<typename S, uint32_t N>
using BufferType_t = typename BufferType<S,N>::type;

template<typename T>
void readToBuffer(T& dest, const void *source) {
    dest = *(reinterpret_cast< const T *>(source));
}

template<typename T>
void gpuAtomicAdd(T* ptr, T value) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        protected_ref(*ptr);
    protected_ref.fetch_add(value);
}

template<typename T>
void gpuAtomicAddGlobal(T& ref, const T& value) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        protected_ref(ref);
    protected_ref.fetch_add(value);
}

template<typename T>
void gpuAtomicAddLocal(T& ref, const T& value) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::local_space>
        protected_ref(ref);
    protected_ref.fetch_add(value);
}

#endif