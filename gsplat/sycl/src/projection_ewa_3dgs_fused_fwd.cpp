#include <c10/xpu/XPUStream.h>

#include "Common.h"
#include "Ops.h"
#include "kernels/FullyFusedProjectionFwdKernel.hpp"

namespace gsplat::xpu {

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ewa_3dgs_fused_fwd(
    const at::Tensor means,                   // [..., N, 3]
    const at::optional<at::Tensor> covars,    // [..., N, 6] optional
    const at::optional<at::Tensor> quats,     // [..., N, 4] optional
    const at::optional<at::Tensor> scales,    // [..., N, 3] optional
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats,                // [..., C, 4, 4]
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
) {
    CHECK_CONTIGUOUS(means);
    CHECK_CONTIGUOUS(viewmats);
    CHECK_CONTIGUOUS(Ks);
    if (covars.has_value())
        CHECK_CONTIGUOUS(covars.value());
    if (quats.has_value())
        CHECK_CONTIGUOUS(quats.value());
    if (scales.has_value())
        CHECK_CONTIGUOUS(scales.value());
    if (opacities.has_value())
        CHECK_CONTIGUOUS(opacities.value());

    TORCH_CHECK(
        means.dim() >= 2, "means must have at least 2 dimensions [..., N, 3]"
    );
    TORCH_CHECK(
        viewmats.dim() >= 3,
        "viewmats must have at least 3 dimensions [..., C, 4, 4]"
    );

    const uint32_t N = means.size(-2);          // number of gaussians
    const uint32_t C = viewmats.size(-3);       // number of cameras
    const uint32_t B = means.numel() / (N * 3); // number of batches
    const int64_t n_elements = B * C * N;

    auto options = means.options();
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));

    at::DimVector out_shape_cn = batch_dims;
    out_shape_cn.insert(out_shape_cn.end(), {C, N});

    at::DimVector out_shape_cn2 = batch_dims;
    out_shape_cn2.insert(out_shape_cn2.end(), {C, N, 2});

    at::DimVector out_shape_cn3 = batch_dims;
    out_shape_cn3.insert(out_shape_cn3.end(), {C, N, 3});

    at::Tensor radii = at::empty(out_shape_cn2, options.dtype(at::kInt));
    at::Tensor means2d = at::empty(out_shape_cn2, options);
    at::Tensor depths = at::empty(out_shape_cn, options);
    at::Tensor conics = at::empty(out_shape_cn3, options);
    at::Tensor compensations = at::empty(out_shape_cn, options);

    if (n_elements > 0) {
        auto &d_queue = at::xpu::getCurrentXPUStream().queue();
        const auto dev_id =
            d_queue.get_device().get_info<sycl::info::device::driver_version>();

        auto num_work_groups =
            (n_elements + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
        sycl::range<1> local_range(GSPLAT_N_THREADS);
        sycl::range<1> global_range(num_work_groups * GSPLAT_N_THREADS);

        AT_DISPATCH_FLOATING_TYPES(
            means.scalar_type(),
            "projection_ewa_3dgs_fused_fwd",
            [&] {
                auto e = d_queue.submit([&](sycl::handler &cgh) {
                    FullyFusedProjectionFwdKernel<scalar_t> kernel(
                        B,
                        C,
                        N,
                        means.data_ptr<scalar_t>(),
                        covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                           : nullptr,
                        quats.has_value() ? quats.value().data_ptr<scalar_t>()
                                          : nullptr,
                        scales.has_value() ? scales.value().data_ptr<scalar_t>()
                                           : nullptr,
                        opacities.has_value()
                            ? opacities.value().data_ptr<scalar_t>()
                            : nullptr,
                        viewmats.data_ptr<scalar_t>(),
                        Ks.data_ptr<scalar_t>(),
                        image_width,
                        image_height,
                        eps2d,
                        near_plane,
                        far_plane,
                        radius_clip,
                        camera_model,
                        radii.data_ptr<int32_t>(),
                        means2d.data_ptr<scalar_t>(),
                        depths.data_ptr<scalar_t>(),
                        conics.data_ptr<scalar_t>(),
                        calc_compensations ? compensations.data_ptr<scalar_t>()
                                           : nullptr
                    );
                    cgh.parallel_for(
                        sycl::nd_range<1>(global_range, local_range), kernel
                    );
                });
                e.wait();
            }
        );
    }

    return std::make_tuple(radii, means2d, depths, conics, compensations);
}

} // namespace  gsplat::xpu