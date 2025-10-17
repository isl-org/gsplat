#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"
#include "kernels/Projection2DGSFusedBwdKernel.hpp"

namespace gsplat::xpu {
    
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_2dgs_fused_bwd(
    // fwd inputs
    const at::Tensor means,    // [..., N, 3]
    const at::Tensor quats,    // [..., N, 4]
    const at::Tensor scales,   // [..., N, 3]
    const at::Tensor viewmats, // [..., C, 4, 4]
    const at::Tensor Ks,       // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const at::Tensor radii,          // [..., C, N, 2]
    const at::Tensor ray_transforms, // [..., C, N, 3, 3]
    // grad outputs
    const at::Tensor v_means2d,        // [..., C, N, 2]
    const at::Tensor v_depths,         // [..., C, N]
    const at::Tensor v_normals,        // [..., C, N, 3]
    const at::Tensor v_ray_transforms, // [..., C, N, 3, 3]
    const bool viewmats_requires_grad
) {
    CHECK_CONTIGUOUS(means);
    CHECK_CONTIGUOUS(quats);
    CHECK_CONTIGUOUS(scales);
    CHECK_CONTIGUOUS(viewmats);
    CHECK_CONTIGUOUS(Ks);
    CHECK_CONTIGUOUS(radii);
    CHECK_CONTIGUOUS(ray_transforms);
    CHECK_CONTIGUOUS(v_means2d);
    CHECK_CONTIGUOUS(v_depths);
    CHECK_CONTIGUOUS(v_normals);
    CHECK_CONTIGUOUS(v_ray_transforms);

    TORCH_CHECK(means.dim() >= 2, "means must have at least 2 dimensions [..., N, 3]");
    TORCH_CHECK(quats.dim() >= 2, "quats must have at least 2 dimensions [..., N, 4]");
    TORCH_CHECK(scales.dim() >= 2, "scales must have at least 2 dimensions [..., N, 3]");
    TORCH_CHECK(viewmats.dim() >= 3, "viewmats must have at least 3 dimensions [..., C, 4, 4]");

    const uint32_t N = means.size(-2);          // number of gaussians
    const uint32_t C = viewmats.size(-3);       // number of cameras
    const uint32_t B = means.numel() / (N * 3); // number of batches
    const int64_t n_elements = B * C * N;

    auto options = means.options();

    // Initialize gradient tensors
    at::Tensor v_means = at::zeros_like(means);
    at::Tensor v_quats = at::zeros_like(quats);
    at::Tensor v_scales = at::zeros_like(scales);
    at::Tensor v_viewmats = viewmats_requires_grad ? at::zeros_like(viewmats) : at::Tensor();

    if (n_elements == 0) {
        // Skip kernel launch if there are no elements
        return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
    }

    auto& d_queue = at::xpu::getCurrentXPUStream().queue();
    
    auto num_work_groups = (n_elements + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
    sycl::range<1> local_range(GSPLAT_N_THREADS);
    sycl::range<1> global_range(num_work_groups * GSPLAT_N_THREADS);

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(), "projection_2dgs_fused_bwd", [&] {
            auto e = d_queue.submit([&](sycl::handler& cgh) {
                Projection2DGSFusedBwdKernel<scalar_t> kernel(
                    B,
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    quats.data_ptr<scalar_t>(),
                    scales.data_ptr<scalar_t>(),
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    radii.data_ptr<int32_t>(),
                    ray_transforms.data_ptr<scalar_t>(),
                    v_means2d.data_ptr<scalar_t>(),
                    v_depths.data_ptr<scalar_t>(),
                    v_normals.data_ptr<scalar_t>(),
                    v_ray_transforms.data_ptr<scalar_t>(),
                    v_means.data_ptr<scalar_t>(),
                    v_quats.data_ptr<scalar_t>(),
                    v_scales.data_ptr<scalar_t>(),
                    viewmats_requires_grad ? v_viewmats.data_ptr<scalar_t>() : nullptr
                );
                cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kernel);
            });
            e.wait();
        });

    return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
}

} // namespace gsplat::xpu