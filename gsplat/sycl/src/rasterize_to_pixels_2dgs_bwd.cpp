#include <c10/xpu/XPUStream.h>

#include "Common.h"
#include "Ops.h"
#include "kernels/RasterizeToPixels2DGSBwdKernel.hpp"

namespace gsplat::xpu {

namespace {

template <uint32_t COLOR_DIM>
void launch_rasterize_2dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor &means2d,
    const at::Tensor &ray_transforms,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::Tensor &normals,
    const at::Tensor &densify,
    const at::optional<at::Tensor> &backgrounds,
    const at::optional<at::Tensor> &masks,
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    // forward outputs
    const at::Tensor &render_colors,
    const at::Tensor &render_alphas,
    const at::Tensor &last_ids,
    const at::Tensor &median_ids,
    // gradients of outputs
    const at::Tensor &v_render_colors,
    const at::Tensor &v_render_alphas,
    const at::Tensor &v_render_normals,
    const at::Tensor &v_render_distort,
    const at::Tensor &v_render_median,
    // outputs
    at::optional<at::Tensor> v_means2d_abs,
    at::Tensor &v_means2d,
    at::Tensor &v_ray_transforms,
    at::Tensor &v_colors,
    at::Tensor &v_opacities,
    at::Tensor &v_normals,
    at::Tensor &v_densify
) {
    auto &d_queue = at::xpu::getCurrentXPUStream().queue();

    bool packed = means2d.dim() == 2;
    uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
    uint32_t I = render_alphas.size(0);         // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    if (n_isects == 0) {
        // Skip kernel launch if there are no intersections
        return;
    }

    // Define the execution ranges
    sycl::range<3> localRange{1, tile_size, tile_size};
    sycl::range<3> globalRange{
        I, tile_height * tile_size, tile_width * tile_size
    };
    sycl::nd_range<3> range(globalRange, localRange);

    // Use a fixed chunk size for batching
    uint32_t chunk_size = 128;

    auto e = d_queue.submit([&](sycl::handler &cgh) {
        // Allocate shared memory
        sycl::local_accessor<int32_t, 1> slm_id_batch(chunk_size, cgh);
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_xy_opacity(
            chunk_size, cgh
        );
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_u_Ms(chunk_size, cgh);
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_v_Ms(chunk_size, cgh);
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_w_Ms(chunk_size, cgh);
        sycl::local_accessor<BufferType_t<float, COLOR_DIM>, 1> slm_rgbs(
            chunk_size, cgh
        );
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_normals(
            chunk_size, cgh
        );

        RasterizeToPixels2DGSBwdKernel<COLOR_DIM> kernel(
            I,
            N,
            n_isects,
            packed,
            chunk_size,
            reinterpret_cast<const sycl::vec<float, 2> *>(
                means2d.data_ptr<float>()
            ),
            ray_transforms.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            normals.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_colors.data_ptr<float>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            median_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_render_normals.data_ptr<float>(),
            v_render_distort.data_ptr<float>(),
            v_render_median.data_ptr<float>(),
            v_means2d_abs.has_value()
                ? reinterpret_cast<sycl::vec<float, 2> *>(
                      v_means2d_abs.value().data_ptr<float>()
                  )
                : nullptr,
            reinterpret_cast<sycl::vec<float, 2> *>(v_means2d.data_ptr<float>()
            ),
            v_ray_transforms.data_ptr<float>(),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>(),
            v_normals.data_ptr<float>(),
            v_densify.data_ptr<float>(),
            slm_id_batch,
            slm_xy_opacity,
            slm_u_Ms,
            slm_v_Ms,
            slm_w_Ms,
            slm_rgbs,
            slm_normals
        );

        cgh.parallel_for(range, kernel);
    });
    e.wait();
}

} // anonymous namespace

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rasterize_to_pixels_2dgs_bwd(
    // Gaussian parameters
    const at::Tensor means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,         // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,      // [..., N] or [nnz]
    const at::Tensor normals,        // [..., N, 3] or [nnz, 3]
    const at::Tensor densify,        // [..., N, 2] or [nnz, 2]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks, // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor
        render_colors, // [..., image_height, image_width, channels]
    const at::Tensor render_alphas, // [..., image_height, image_width]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    const at::Tensor median_ids,    // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor
        v_render_colors, // [..., image_height, image_width, channels]
    const at::Tensor v_render_alphas,  // [..., image_height, image_width]
    const at::Tensor v_render_normals, // [..., image_height, image_width, 3]
    const at::Tensor v_render_distort, // [..., image_height, image_width]
    const at::Tensor v_render_median,  // [..., image_height, image_width]
    bool absgrad
) {
    // Check input tensors are contiguous
    CHECK_CONTIGUOUS(means2d);
    CHECK_CONTIGUOUS(ray_transforms);
    CHECK_CONTIGUOUS(colors);
    CHECK_CONTIGUOUS(opacities);
    CHECK_CONTIGUOUS(normals);
    CHECK_CONTIGUOUS(densify);
    CHECK_CONTIGUOUS(tile_offsets);
    CHECK_CONTIGUOUS(flatten_ids);
    CHECK_CONTIGUOUS(render_colors);
    CHECK_CONTIGUOUS(render_alphas);
    CHECK_CONTIGUOUS(last_ids);
    CHECK_CONTIGUOUS(median_ids);
    CHECK_CONTIGUOUS(v_render_colors);
    CHECK_CONTIGUOUS(v_render_alphas);
    CHECK_CONTIGUOUS(v_render_normals);
    CHECK_CONTIGUOUS(v_render_distort);
    CHECK_CONTIGUOUS(v_render_median);
    if (backgrounds.has_value())
        CHECK_CONTIGUOUS(backgrounds.value());
    if (masks.has_value())
        CHECK_CONTIGUOUS(masks.value());

    uint32_t channels = colors.size(-1);

    // Create output tensors
    auto options = means2d.options().dtype(torch::kFloat32);
    at::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = at::zeros_like(means2d, options);
    }
    at::Tensor v_means2d = at::zeros_like(means2d, options);
    at::Tensor v_ray_transforms = at::zeros_like(ray_transforms, options);
    at::Tensor v_colors = at::zeros_like(colors, options);
    at::Tensor v_opacities = at::zeros_like(opacities, options);
    at::Tensor v_normals = at::zeros_like(normals, options);
    at::Tensor v_densify = at::zeros_like(densify, options);

    // Launch kernel with appropriate dimension
#define __GS__CALL_(DIM)                                                       \
    case DIM:                                                                  \
        launch_rasterize_2dgs_bwd_kernel<DIM>(                                 \
            means2d,                                                           \
            ray_transforms,                                                    \
            colors,                                                            \
            opacities,                                                         \
            normals,                                                           \
            densify,                                                           \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            render_colors,                                                     \
            render_alphas,                                                     \
            last_ids,                                                          \
            median_ids,                                                        \
            v_render_colors,                                                   \
            v_render_alphas,                                                   \
            v_render_normals,                                                  \
            v_render_distort,                                                  \
            v_render_median,                                                   \
            absgrad ? c10::optional<at::Tensor>(v_means2d_abs) : c10::nullopt, \
            v_means2d,                                                         \
            v_ray_transforms,                                                  \
            v_colors,                                                          \
            v_opacities,                                                       \
            v_normals,                                                         \
            v_densify                                                          \
        );                                                                     \
        break;

    switch (channels) {
        __GS__CALL_(1);
        __GS__CALL_(2);
        __GS__CALL_(3);
        __GS__CALL_(4);
        __GS__CALL_(5);
        __GS__CALL_(8);
        __GS__CALL_(9);
        __GS__CALL_(16);
        __GS__CALL_(17);
        __GS__CALL_(32);
        __GS__CALL_(33);
        __GS__CALL_(64);
        __GS__CALL_(65);
        __GS__CALL_(128);
        __GS__CALL_(129);
        __GS__CALL_(256);
        __GS__CALL_(257);
        __GS__CALL_(512);
        __GS__CALL_(513);
    default:
        TORCH_CHECK(false, "Unsupported number of channels: ", channels);
    }
#undef __GS__CALL_

    return std::make_tuple(
        v_means2d_abs,
        v_means2d,
        v_ray_transforms,
        v_colors,
        v_opacities,
        v_normals,
        v_densify
    );
}

} // namespace gsplat::xpu
