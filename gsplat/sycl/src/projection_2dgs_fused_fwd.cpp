#include <c10/xpu/XPUStream.h>

#include "Common.h"
#include "Ops.h"
#include "kernels/Projection2DGSFusedFwdKernel.hpp"

namespace gsplat::xpu {

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_2dgs_fused_fwd(
    const at::Tensor means,    // [..., N, 3]
    const at::Tensor quats,    // [..., N, 4]
    const at::Tensor scales,   // [..., N, 3]
    const at::Tensor viewmats, // [..., C, 4, 4]
    const at::Tensor Ks,       // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip
) {
    CHECK_CONTIGUOUS(means);
    CHECK_CONTIGUOUS(quats);
    CHECK_CONTIGUOUS(scales);
    CHECK_CONTIGUOUS(viewmats);
    CHECK_CONTIGUOUS(Ks);

    TORCH_CHECK(
        means.dim() >= 2, "means must have at least 2 dimensions [..., N, 3]"
    );
    TORCH_CHECK(
        quats.dim() >= 2, "quats must have at least 2 dimensions [..., N, 4]"
    );
    TORCH_CHECK(
        scales.dim() >= 2, "scales must have at least 2 dimensions [..., N, 3]"
    );
    TORCH_CHECK(
        viewmats.dim() >= 3,
        "viewmats must have at least 3 dimensions [..., C, 4, 4]"
    );
    TORCH_CHECK(
        Ks.dim() >= 3, "Ks must have at least 3 dimensions [..., C, 3, 3]"
    );

    const uint32_t N = means.size(-2);          // number of gaussians
    const uint32_t C = viewmats.size(-3);       // number of cameras
    const uint32_t B = means.numel() / (N * 3); // number of batches
    const int64_t n_elements = B * C * N;

    auto options = means.options();
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));

    // Output shape: [..., C, N]
    at::DimVector out_shape_cn = batch_dims;
    out_shape_cn.insert(out_shape_cn.end(), {C, N});

    // Output shape: [..., C, N, 2]
    at::DimVector out_shape_cn2 = batch_dims;
    out_shape_cn2.insert(out_shape_cn2.end(), {C, N, 2});

    // Output shape: [..., C, N, 3]
    at::DimVector out_shape_cn3 = batch_dims;
    out_shape_cn3.insert(out_shape_cn3.end(), {C, N, 3});

    // Output shape: [..., C, N, 3, 3]
    at::DimVector out_shape_cn33 = batch_dims;
    out_shape_cn33.insert(out_shape_cn33.end(), {C, N, 3, 3});

    at::Tensor radii = at::empty(out_shape_cn2, options.dtype(at::kInt));
    at::Tensor means2d = at::empty(out_shape_cn2, options);
    at::Tensor depths = at::empty(out_shape_cn, options);
    at::Tensor ray_transforms = at::empty(out_shape_cn33, options);
    at::Tensor normals = at::empty(out_shape_cn3, options);

    if (n_elements == 0) {
        // Skip kernel launch if there are no elements
        return std::make_tuple(radii, means2d, depths, ray_transforms, normals);
    }

    auto &d_queue = at::xpu::getCurrentXPUStream().queue();

    auto num_work_groups =
        (n_elements + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
    sycl::range<1> local_range(GSPLAT_N_THREADS);
    sycl::range<1> global_range(num_work_groups * GSPLAT_N_THREADS);

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_2dgs_fused_fwd",
        [&] {
            auto e = d_queue.submit([&](sycl::handler &cgh) {
                Projection2DGSFusedFwdKernel<scalar_t> kernel(
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
                    near_plane,
                    far_plane,
                    radius_clip,
                    radii.data_ptr<int32_t>(),
                    means2d.data_ptr<scalar_t>(),
                    depths.data_ptr<scalar_t>(),
                    ray_transforms.data_ptr<scalar_t>(),
                    normals.data_ptr<scalar_t>()
                );
                cgh.parallel_for(
                    sycl::nd_range<1>(global_range, local_range), kernel
                );
            });
            e.wait();
        }
    );

    return std::make_tuple(radii, means2d, depths, ray_transforms, normals);
}

} // namespace gsplat::xpu