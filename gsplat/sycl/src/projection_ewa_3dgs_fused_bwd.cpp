#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"
#include "kernels/FullyFusedProjectionBwdKernel.hpp"

namespace  gsplat::xpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_fused_bwd(
    // fwd inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [..., C, N, 2]
    const at::Tensor conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> compensations, // [..., C, N] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [..., C, N, 2]
    const at::Tensor v_depths,                      // [..., C, N]
    const at::Tensor v_conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [..., C, N] optional
    const bool viewmats_requires_grad
) {
    // Input validation
    CHECK_CONTIGUOUS(means);
    if (covars.has_value()) CHECK_CONTIGUOUS(covars.value());
    if (quats.has_value()) CHECK_CONTIGUOUS(quats.value());
    if (scales.has_value()) CHECK_CONTIGUOUS(scales.value());
    CHECK_CONTIGUOUS(viewmats);
    CHECK_CONTIGUOUS(Ks);
    CHECK_CONTIGUOUS(radii);
    CHECK_CONTIGUOUS(conics);
    if (compensations.has_value()) CHECK_CONTIGUOUS(compensations.value());
    CHECK_CONTIGUOUS(v_means2d);
    CHECK_CONTIGUOUS(v_depths);
    CHECK_CONTIGUOUS(v_conics);
    if (v_compensations.has_value()) CHECK_CONTIGUOUS(v_compensations.value());

    // Dimensions
    const uint32_t N = means.size(-2);
    const uint32_t C = viewmats.size(-3);
    const uint32_t B = means.numel() / (N * 3);
    const int64_t n_elements = B * C * N;

    // Create gradient tensors, initialized to zero
    at::Tensor v_means = at::zeros_like(means);
    at::Tensor v_covars = covars.has_value() ? at::zeros_like(covars.value()) : at::empty({0}, means.options());
    at::Tensor v_quats = quats.has_value() ? at::zeros_like(quats.value()) : at::empty({0}, means.options());
    at::Tensor v_scales = scales.has_value() ? at::zeros_like(scales.value()) : at::empty({0}, means.options());
    at::Tensor v_viewmats = viewmats_requires_grad ? at::zeros_like(viewmats) : at::empty({0}, means.options());

    if (n_elements > 0) {
        auto& d_queue = at::xpu::getCurrentXPUStream().queue();
        auto num_work_groups = (n_elements + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
        sycl::range<1> local_range(GSPLAT_N_THREADS);
        sycl::range<1> global_range(num_work_groups * GSPLAT_N_THREADS);

        AT_DISPATCH_FLOATING_TYPES(
            means.scalar_type(), "projection_ewa_3dgs_fused_bwd", [&] {
                auto e = d_queue.submit([&](sycl::handler& cgh) {
                    FullyFusedProjectionBwdKernel<scalar_t> kernel(
                        B,
                        C,
                        N,
                        means.data_ptr<scalar_t>(),
                        covars.has_value() ? covars.value().data_ptr<scalar_t>() : nullptr,
                        quats.has_value() ? quats.value().data_ptr<scalar_t>() : nullptr,
                        scales.has_value() ? scales.value().data_ptr<scalar_t>() : nullptr,
                        viewmats.data_ptr<scalar_t>(),
                        Ks.data_ptr<scalar_t>(),
                        image_width,
                        image_height,
                        eps2d,
                        camera_model,
                        radii.data_ptr<int32_t>(),
                        conics.data_ptr<scalar_t>(),
                        compensations.has_value() ? compensations.value().data_ptr<scalar_t>() : nullptr,
                        v_means2d.data_ptr<scalar_t>(),
                        v_depths.data_ptr<scalar_t>(),
                        v_conics.data_ptr<scalar_t>(),
                        v_compensations.has_value() ? v_compensations.value().data_ptr<scalar_t>() : nullptr,
                        v_means.data_ptr<scalar_t>(),
                        covars.has_value() ? v_covars.data_ptr<scalar_t>() : nullptr,
                        quats.has_value() ? v_quats.data_ptr<scalar_t>() : nullptr,
                        scales.has_value() ? v_scales.data_ptr<scalar_t>() : nullptr,
                        viewmats_requires_grad ? v_viewmats.data_ptr<scalar_t>() : nullptr
                    );
                    cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kernel);
                });
                e.wait();
            });
    }

    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}

} // namespace  gsplat::xpu