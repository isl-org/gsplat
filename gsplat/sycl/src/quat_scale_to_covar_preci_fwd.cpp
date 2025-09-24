#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "kernels/QuatScaleToCovarPreciFwdKernel.hpp"

namespace gsplat::xpu {
    
std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_fwd(
    const at::Tensor quats,   // [..., 4]
    const at::Tensor scales,  // [..., 3]
    const bool compute_covar,
    const bool compute_preci,
    const bool triu
) {
    CHECK_CONTIGUOUS(quats);
    CHECK_CONTIGUOUS(scales);
    TORCH_CHECK(compute_covar || compute_preci, "Must compute at least one of covar or preci");

    const int64_t N = quats.numel() / 4;
    auto options = quats.options();

    at::Tensor covars;
    at::Tensor precis;

    // Create an output shape that preserves the batch dimensions from the input
    at::DimVector out_shape(quats.sizes().slice(0, quats.dim() - 1));
    if (triu) {
        out_shape.push_back(6);
    } else {
        out_shape.push_back(3);
        out_shape.push_back(3);
    }

    if (compute_covar) {
        covars = at::empty(out_shape, options);
    } else {
        covars = at::empty({0}, options);
    }

    if (compute_preci) {
        precis = at::empty(out_shape, options);
    } else {
        precis = at::empty({0}, options);
    }

    if (N == 0) {
        return std::make_tuple(covars, precis);
    }

    auto& d_queue = at::xpu::getCurrentXPUStream().queue();
    
    size_t numWorkGrps = (N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
    sycl::range<1> localRange(GSPLAT_N_THREADS);
    sycl::range<1> globalRange(GSPLAT_N_THREADS * numWorkGrps);
    sycl::nd_range<1> range(globalRange, localRange);

    d_queue.submit(
        [&](sycl::handler& cgh)
        {
            QuatScaleToCovarPreciFwdKernel<float> kernel(
                N,
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                triu,
                compute_covar ? covars.data_ptr<float>() : nullptr,
                compute_preci ? precis.data_ptr<float>() : nullptr
            );
            cgh.parallel_for(range, kernel);
        }
    );
    
    return std::make_tuple(covars, precis);
}

} // namespace gsplat::xpu