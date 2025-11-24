#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "kernels/ComputeShFwdKernel.hpp"

namespace gsplat::xpu {

at::Tensor spherical_harmonics_fwd(
    const uint32_t degrees_to_use,
    const at::Tensor dirs,               // [..., 3]
    const at::Tensor coeffs,             // [..., K, 3]
    const at::optional<at::Tensor> masks // [...]
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT2(coeffs, dirs);
    if (masks.has_value()) {
        CHECK_INPUT2(masks.value(), dirs);
    }
    TORCH_CHECK(
        dirs.size(-1) == 3,
        "Input 'dirs' tensor must have the last dimension of size 3."
    );
    TORCH_CHECK(
        coeffs.size(-1) == 3,
        "Input 'coeffs' tensor must have the last dimension of size 3."
    );

    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;

    at::Tensor colors = at::empty_like(dirs);

    if (N == 0) {
        return colors;
    }

    auto &d_queue = at::xpu::getCurrentXPUStream().queue();

    size_t numWorkGrps = (N * 3 + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
    sycl::range<1> localRange(GSPLAT_N_THREADS);
    sycl::range<1> globalRange(GSPLAT_N_THREADS * numWorkGrps);
    sycl::nd_range<1> range(globalRange, localRange);

    auto e = d_queue.submit([&](sycl::handler &cgh) {
        ComputeShFwdKernel<float> kernel(
            N,
            K,
            degrees_to_use,
            reinterpret_cast<vec3<float> *>(dirs.data_ptr<float>()),
            coeffs.data_ptr<float>(),
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            colors.data_ptr<float>()
        );
        cgh.parallel_for(range, kernel);
    });
    e.wait();

    return colors;
}

} // namespace  gsplat::xpu