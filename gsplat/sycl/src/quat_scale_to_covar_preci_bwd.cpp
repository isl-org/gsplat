#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "kernels/QuatScaleToCovarPreciBwdKernel.hpp"

namespace gsplat::xpu {
    
std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_bwd(
    const at::Tensor quats,                 // [..., 4]
    const at::Tensor scales,                // [..., 3]
    const bool triu,
    const at::optional<at::Tensor> v_covars, // [..., 3, 3] or [..., 6]
    const at::optional<at::Tensor> v_precis  // [..., 3, 3] or [..., 6]
) {
    CHECK_CONTIGUOUS(quats);
    CHECK_CONTIGUOUS(scales);
    if (v_covars.has_value()) {
        CHECK_CONTIGUOUS(v_covars.value());
    }
    if (v_precis.has_value()) {
        CHECK_CONTIGUOUS(v_precis.value());
    }
    TORCH_CHECK(v_covars.has_value() || v_precis.has_value(), "Must provide gradients for at least one of covars or precis");

    const int64_t N = quats.numel() / 4;
    at::Tensor v_quats = at::empty_like(quats);
    at::Tensor v_scales = at::empty_like(scales);

    if (N == 0) {
        return std::make_tuple(v_quats, v_scales);
    }
    
    auto& d_queue = at::xpu::getCurrentXPUStream().queue();
    
    size_t numWorkGrps = (N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
    sycl::range<1> localRange(GSPLAT_N_THREADS);
    sycl::range<1> globalRange(GSPLAT_N_THREADS * numWorkGrps);
    sycl::nd_range<1> range(globalRange, localRange);

    d_queue.submit(
        [&](sycl::handler& cgh)
        {
            QuatScaleToCovarPreciBwdKernel<float> kernel(
                N,
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                v_covars.has_value() ? v_covars.value().data_ptr<float>() : nullptr,
                v_precis.has_value() ? v_precis.value().data_ptr<float>() : nullptr,
                triu,
                v_scales.data_ptr<float>(),
                v_quats.data_ptr<float>()
            );
            cgh.parallel_for(range, kernel);
        }
    );

    return std::make_tuple(v_quats, v_scales);
}

} // namespace gsplat::xpu