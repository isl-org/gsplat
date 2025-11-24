#include <c10/xpu/XPUStream.h>

#include "Common.h"
#include "Ops.h"
#include "kernels/PackedProjectionBwdKernel.hpp"

namespace gsplat::xpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_packed_bwd(
    // fwd inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6]
    const at::optional<at::Tensor> quats,  // [..., N, 4]
    const at::optional<at::Tensor> scales, // [..., N, 3]
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor batch_ids,                   // [nnz]
    const at::Tensor camera_ids,                  // [nnz]
    const at::Tensor gaussian_ids,                // [nnz]
    const at::Tensor conics,                      // [nnz, 3]
    const at::optional<at::Tensor> compensations, // [nnz] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [nnz, 2]
    const at::Tensor v_depths,                      // [nnz]
    const at::Tensor v_conics,                      // [nnz, 3]
    const at::optional<at::Tensor> v_compensations, // [nnz] optional
    const bool viewmats_requires_grad,
    const bool sparse_grad
) {
    DEVICE_GUARD(means);
    // Input validation
    CHECK_INPUT(means);
    if (covars.has_value())
        CHECK_INPUT2(covars.value(), means);
    if (quats.has_value())
        CHECK_INPUT2(quats.value(), means);
    if (scales.has_value())
        CHECK_INPUT2(scales.value(), means);
    CHECK_INPUT2(viewmats, means);
    CHECK_INPUT2(Ks, means);
    CHECK_INPUT2(batch_ids, means);
    CHECK_INPUT2(camera_ids, means);
    CHECK_INPUT2(gaussian_ids, means);
    CHECK_INPUT2(conics, means);
    if (compensations.has_value())
        CHECK_INPUT2(compensations.value(), means);
    CHECK_INPUT2(v_means2d, means);
    CHECK_INPUT2(v_depths, means);
    CHECK_INPUT2(v_conics, means);
    if (v_compensations.has_value())
        CHECK_INPUT2(v_compensations.value(), means);

    uint32_t N = means.size(-2);
    uint32_t C = viewmats.size(-3);
    uint32_t B = means.numel() / (N * 3);
    uint32_t nnz = batch_ids.size(0);

    // Allocate output gradient tensors
    at::Tensor v_means, v_covars, v_quats, v_scales, v_viewmats;

    if (sparse_grad) {
        v_means = at::empty({(long)nnz, 3}, means.options());
        if (covars.has_value()) {
            v_covars = at::empty({(long)nnz, 6}, covars.value().options());
        } else {
            v_quats = at::empty({(long)nnz, 4}, quats.value().options());
            v_scales = at::empty({(long)nnz, 3}, scales.value().options());
        }
    } else {
        v_means = at::zeros_like(means);
        if (covars.has_value()) {
            v_covars = at::zeros_like(covars.value());
        } else {
            v_quats = at::zeros_like(quats.value());
            v_scales = at::zeros_like(scales.value());
        }
    }

    if (viewmats_requires_grad) {
        v_viewmats = at::zeros_like(viewmats);
    }

    if (nnz == 0) {
        return std::make_tuple(
            v_means, v_covars, v_quats, v_scales, v_viewmats
        );
    }

    auto &d_queue = at::xpu::getCurrentXPUStream().queue();
    sycl::range<1> local_range(256);
    sycl::range<1> global_range(
        (nnz + local_range[0] - 1) / local_range[0] * local_range[0]
    );
    sycl::nd_range<1> range(global_range, local_range);

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_packed_bwd_kernel",
        [&] {
            PackedProjectionBwdKernel<scalar_t> kernel(
                B,
                C,
                N,
                nnz,
                means.data_ptr<scalar_t>(),
                covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                   : nullptr,
                covars.has_value() ? nullptr
                                   : quats.value().data_ptr<scalar_t>(),
                covars.has_value() ? nullptr
                                   : scales.value().data_ptr<scalar_t>(),
                viewmats.data_ptr<scalar_t>(),
                Ks.data_ptr<scalar_t>(),
                image_width,
                image_height,
                (scalar_t)eps2d,
                camera_model,
                batch_ids.data_ptr<int64_t>(),
                camera_ids.data_ptr<int64_t>(),
                gaussian_ids.data_ptr<int64_t>(),
                conics.data_ptr<scalar_t>(),
                compensations.has_value()
                    ? compensations.value().data_ptr<scalar_t>()
                    : nullptr,
                v_means2d.data_ptr<scalar_t>(),
                v_depths.data_ptr<scalar_t>(),
                v_conics.data_ptr<scalar_t>(),
                v_compensations.has_value()
                    ? v_compensations.value().data_ptr<scalar_t>()
                    : nullptr,
                sparse_grad,
                v_means.data_ptr<scalar_t>(),
                covars.has_value() ? v_covars.data_ptr<scalar_t>() : nullptr,
                covars.has_value() ? nullptr : v_quats.data_ptr<scalar_t>(),
                covars.has_value() ? nullptr : v_scales.data_ptr<scalar_t>(),
                viewmats_requires_grad ? v_viewmats.data_ptr<scalar_t>()
                                       : nullptr
            );
            auto e = d_queue.parallel_for(range, kernel);
            e.wait();
        }
    );

    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}

} // namespace gsplat::xpu