#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"
#include "kernels/RasterizeToPixels2DGSFwdKernel.hpp"

namespace gsplat::xpu {

namespace {

template <uint32_t COLOR_DIM>
void launch_rasterize_2dgs_kernel(
    // Gaussian parameters
    const at::Tensor& means2d,
    const at::Tensor& ray_transforms,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const at::Tensor& normals,
    const at::optional<at::Tensor>& backgrounds,
    const at::optional<at::Tensor>& masks,
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    // other params
    bool packed,
    uint32_t I,
    uint32_t N,
    uint32_t tile_height,
    uint32_t tile_width,
    uint32_t n_isects,
    // outputs
    at::Tensor& renders,
    at::Tensor& alphas,
    at::Tensor& render_normals,
    at::Tensor& render_distort,
    at::Tensor& render_median,
    at::Tensor& last_ids,
    at::Tensor& median_ids
) {
    auto& d_queue = at::xpu::getCurrentXPUStream().queue();

    // Define the execution ranges
    sycl::range<3> localRange{1, tile_size, tile_size};
    sycl::range<3> globalRange{I, tile_height*tile_size, tile_width*tile_size};
    sycl::nd_range<3> range(globalRange, localRange);

    // Use a fixed chunk size for batching - don't make it constexpr with tile_size
    uint32_t chunk_size = 128; // Fixed size that's similar to what would be used

    auto e = d_queue.submit(
        [&](sycl::handler& cgh)
        {            
            // Allocate shared memory
            sycl::local_accessor<int32_t, 1> slm_id_batch(chunk_size, cgh);
            sycl::local_accessor<sycl::vec<float, 3>, 1> slm_xy_opacity(chunk_size, cgh);
            sycl::local_accessor<sycl::vec<float, 3>, 1> slm_u_Ms(chunk_size, cgh);
            sycl::local_accessor<sycl::vec<float, 3>, 1> slm_v_Ms(chunk_size, cgh);
            sycl::local_accessor<sycl::vec<float, 3>, 1> slm_w_Ms(chunk_size, cgh);

            RasterizeToPixels2DGSFwdKernel<COLOR_DIM> kernel(
                I, N, n_isects, packed, chunk_size,
                reinterpret_cast<const sycl::vec<float, 2>*>(means2d.data_ptr<float>()),
                ray_transforms.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                normals.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                renders.data_ptr<float>(),
                alphas.data_ptr<float>(),
                render_normals.data_ptr<float>(),
                render_distort.data_ptr<float>(),
                render_median.data_ptr<float>(),
                last_ids.data_ptr<int32_t>(),
                median_ids.data_ptr<int32_t>(),
                slm_id_batch, slm_xy_opacity, slm_u_Ms, slm_v_Ms, slm_w_Ms
            );
            
            cgh.parallel_for(range, kernel);
        }
    );
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
rasterize_to_pixels_2dgs_fwd(
    // Gaussian parameters
    const at::Tensor means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,         // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,      // [..., N]  or [nnz]
    const at::Tensor normals,        // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids   // [n_isects]
) {
    // Check input tensors are contiguous
    CHECK_CONTIGUOUS(means2d);
    CHECK_CONTIGUOUS(ray_transforms);
    CHECK_CONTIGUOUS(colors);
    CHECK_CONTIGUOUS(opacities);
    CHECK_CONTIGUOUS(normals);
    CHECK_CONTIGUOUS(tile_offsets);
    CHECK_CONTIGUOUS(flatten_ids);
    if (backgrounds.has_value()) CHECK_CONTIGUOUS(backgrounds.value());
    if (masks.has_value()) CHECK_CONTIGUOUS(masks.value());
    
    // Get dimensions
    bool packed = means2d.dim() == 2;
    uint32_t N = packed ? 0 : means2d.size(-2);   // number of gaussians
    uint32_t I = tile_offsets.size(0);            // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);      // number of intersections
    uint32_t channels = colors.size(-1);          // color dimension
    
    // Create output tensors
    auto options_float = means2d.options().dtype(torch::kFloat32);
    auto options_int = means2d.options().dtype(torch::kInt32);
    
    at::Tensor renders = at::zeros({I, image_height, image_width, channels}, options_float);
    at::Tensor alphas = at::zeros({I, image_height, image_width}, options_float);
    at::Tensor render_normals = at::zeros({I, image_height, image_width, 3}, options_float);
    at::Tensor render_distort = at::zeros({I, image_height, image_width}, options_float);
    at::Tensor render_median = at::zeros({I, image_height, image_width}, options_float);
    at::Tensor last_ids = at::zeros({I, image_height, image_width}, options_int);
    at::Tensor median_ids = at::zeros({I, image_height, image_width}, options_int);

    // Launch kernel with appropriate dimension
#define __GS__CALL_(DIM)                                                           \
    case DIM:                                                                      \
        launch_rasterize_2dgs_kernel<DIM>(                                         \
            means2d, ray_transforms, colors, opacities, normals,                   \
            backgrounds, masks, image_width, image_height, tile_size,              \
            tile_offsets, flatten_ids, packed, I, N, tile_height, tile_width,      \
            n_isects, renders, alphas, render_normals, render_distort,             \
            render_median, last_ids, median_ids                                    \
        );                                                                         \
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
    
    return std::make_tuple(renders, alphas, render_normals, render_distort, 
                          render_median, last_ids, median_ids);
}

} // namespace gsplat::xpu
