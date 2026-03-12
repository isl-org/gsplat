#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>

namespace gsplat::xpu {

//
// Some Macros.
//
#define CHECK_XPU(x) TORCH_CHECK(x.is_xpu(), #x " must be a XPU tensor")
#define CHECK_DEVICE(x, y)                                                     \
    TORCH_CHECK(                                                               \
        x.device() == y.device(), #x " must be on device " + y.device().str()  \
    )
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    {                                                                          \
        CHECK_XPU(x);                                                          \
        CHECK_CONTIGUOUS(x);                                                   \
    }
#define CHECK_INPUT2(x, y)                                                     \
    {                                                                          \
        CHECK_DEVICE(x, y);                                                    \
        CHECK_CONTIGUOUS(x);                                                   \
    }
#define DEVICE_GUARD(_ten) const c10::DeviceGuard device_guard(_ten.device());

//
// Legacy Camera Types
//
enum CameraModelType {
    PINHOLE = 0,
    ORTHO = 1,
    FISHEYE = 2,
    FTHETA = 3,
};

#define GSPLAT_N_THREADS 256
#define N_THREADS_PACKED 256
} // namespace  gsplat::xpu