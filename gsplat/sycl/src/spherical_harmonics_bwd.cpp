#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "kernels/ComputeShBwdKernel.hpp"

namespace gsplat::xpu {

std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(
    const uint32_t K,
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., 3]
    bool compute_v_dirs
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT2(coeffs, dirs);
    CHECK_INPUT2(v_colors, dirs);
    if (masks.has_value()) {
        CHECK_INPUT2(masks.value(), dirs);
    }

    TORCH_CHECK(v_colors.size(-1) == 3, "v_colors must have last dimension 3");
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");

    const uint32_t N = dirs.numel() / 3;

    at::Tensor v_coeffs = at::zeros_like(coeffs);
    at::Tensor v_dirs =
        compute_v_dirs ? at::zeros_like(dirs) : at::empty({0}, dirs.options());

    if (N == 0) {
        return std::make_tuple(v_coeffs, v_dirs);
    }

    auto &d_queue = at::xpu::getCurrentXPUStream().queue();

    size_t numWorkGrps = (N * 3 + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
    sycl::range<1> localRange(GSPLAT_N_THREADS);
    sycl::range<1> globalRange(GSPLAT_N_THREADS * numWorkGrps);
    sycl::nd_range<1> range(globalRange, localRange);

    auto e = d_queue.submit([&](sycl::handler &cgh) {
        ComputeShBwdKernel<float> kernel(
            N,
            K,
            degrees_to_use,
            reinterpret_cast<const vec3<float> *>(dirs.data_ptr<float>()),
            coeffs.data_ptr<float>(),
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            v_colors.data_ptr<float>(),
            v_coeffs.data_ptr<float>(),
            compute_v_dirs ? v_dirs.data_ptr<float>() : nullptr
        );
        cgh.parallel_for(range, kernel);
    });
    e.wait();

    return std::make_tuple(v_coeffs, v_dirs);
}

} // namespace gsplat::xpu