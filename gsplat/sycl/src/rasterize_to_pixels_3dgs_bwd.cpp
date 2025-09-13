#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "kernels/RasterizeToPixelsBwdKernel.hpp"

namespace gsplat::xpu {

namespace { 

template <uint32_t COLOR_DIM>
void launch_rasterize_bwd_kernel(
    // Gaussian parameters
    const at::Tensor& means2d,
    const at::Tensor& conics,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const at::optional<at::Tensor>& backgrounds,
    const at::optional<at::Tensor>& masks,
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    // forward outputs
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    // gradients of outputs
    const at::Tensor& v_render_colors,
    const at::Tensor& v_render_alphas,
    // options and derived params
    bool absgrad,
    bool packed,
    uint32_t C,
    uint32_t N,
    uint32_t n_isects,
    uint32_t tile_height,
    uint32_t tile_width,
    // output grads
    at::Tensor& v_means2d,
    at::Tensor& v_conics,
    at::Tensor& v_colors,
    at::Tensor& v_opacities,
    at::Tensor& v_means2d_abs
) {
    if (n_isects == 0) {
        return;
    }

    auto& d_queue = at::xpu::getCurrentXPUStream().queue();
    
    sycl::range<3> localRange{1, tile_size, tile_size};
    sycl::range<3> globalRange{C, tile_height * tile_size, tile_width * tile_size};
    sycl::nd_range<3> range(globalRange, localRange);
        
    auto e = d_queue.submit(
        [&](sycl::handler& cgh)
        {
            constexpr uint32_t CHUNK_SIZE = 256;
            sycl::range<1> slm_range(CHUNK_SIZE);

            sycl::local_accessor<int32_t, 1> slm_flatten_ids(slm_range, cgh);
            sycl::local_accessor<sycl::vec<sycl::half, 2>, 1> slm_means2d(slm_range, cgh);
            sycl::local_accessor<sycl::half, 1> slm_opacities(slm_range, cgh);
            sycl::local_accessor<sycl::vec<sycl::half, 3>, 1> slm_conics(slm_range, cgh);
            sycl::local_accessor<BufferType_t<sycl::half, COLOR_DIM>, 1> slm_color;
            if constexpr(BufferType<float, COLOR_DIM>::isVec && COLOR_DIM <= 4) {
                slm_color = sycl::local_accessor<BufferType_t<sycl::half, COLOR_DIM>, 1>(slm_range, cgh);
            }

            RasterizeToPixelsBwdKernel<COLOR_DIM, CHUNK_SIZE, float, false> kernel(
                C, N, n_isects, packed,
                0, nullptr, // concat_stride, concatenated_data
                reinterpret_cast<const sycl::vec<float, 2>*>(means2d.data_ptr<float>()),
                reinterpret_cast<const vec3<float>*>(conics.data_ptr<float>()),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(),
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? reinterpret_cast<sycl::vec<float, 2>*>(v_means2d_abs.data_ptr<float>()) : nullptr,
                reinterpret_cast<sycl::vec<float, 2>*>(v_means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float>*>(v_conics.data_ptr<float>()),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                slm_flatten_ids, slm_means2d, slm_opacities, slm_conics, slm_color
            );
            cgh.parallel_for(range, kernel);
        }
    );
    e.wait();
}

} // anonymous namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_3dgs_bwd(
    // Gaussian parameters
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    // forward outputs
    const at::Tensor render_alphas,
    const at::Tensor last_ids,
    // gradients of outputs
    const at::Tensor v_render_colors,
    const at::Tensor v_render_alphas,
    // options
    bool absgrad
) {
    CHECK_CONTIGUOUS(means2d);
    CHECK_CONTIGUOUS(conics);
    CHECK_CONTIGUOUS(colors);
    CHECK_CONTIGUOUS(opacities);
    CHECK_CONTIGUOUS(tile_offsets);
    CHECK_CONTIGUOUS(flatten_ids);
    CHECK_CONTIGUOUS(render_alphas);
    CHECK_CONTIGUOUS(last_ids);
    CHECK_CONTIGUOUS(v_render_colors);
    CHECK_CONTIGUOUS(v_render_alphas);
    if (backgrounds.has_value()) CHECK_CONTIGUOUS(backgrounds.value());
    if (masks.has_value()) CHECK_CONTIGUOUS(masks.value());

    // --- Parameter Derivation ---
    const uint32_t COLOR_DIM = colors.size(-1);
    const bool packed = means2d.dim() == 2;
    const uint32_t C = tile_offsets.size(0);
    const uint32_t N = packed ? 0 : means2d.size(1);
    const uint32_t n_isects = flatten_ids.size(0);
    const uint32_t tile_height = tile_offsets.size(1);
    const uint32_t tile_width = tile_offsets.size(2);
    
    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_conics = at::zeros_like(conics);
    at::Tensor v_colors = at::zeros_like(colors);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_means2d_abs = absgrad ? at::zeros_like(means2d) : at::empty({0}, means2d.options());

    
#define __GS_BWD_CALL_(DIM)                                                                                             \
    case DIM:                                                                                                           \
        launch_rasterize_bwd_kernel<DIM>(                                                                               \
            means2d, conics, colors, opacities, backgrounds, masks, image_width, image_height, tile_size,               \
            tile_offsets, flatten_ids, render_alphas, last_ids, v_render_colors, v_render_alphas, absgrad,              \
            packed, C, N, n_isects, tile_height, tile_width,                                                            \
            v_means2d, v_conics, v_colors, v_opacities, v_means2d_abs                                                   \
        );                                                                                                              \
        break;

    switch (COLOR_DIM) {
        __GS_BWD_CALL_(1);
        __GS_BWD_CALL_(2);
        __GS_BWD_CALL_(3);
        __GS_BWD_CALL_(4);
        __GS_BWD_CALL_(5);
        __GS_BWD_CALL_(8);
        __GS_BWD_CALL_(9);
        __GS_BWD_CALL_(16);
        __GS_BWD_CALL_(17);
        __GS_BWD_CALL_(32);
        __GS_BWD_CALL_(33);
        __GS_BWD_CALL_(64);
        __GS_BWD_CALL_(65);
        __GS_BWD_CALL_(128);
        __GS_BWD_CALL_(129);
        __GS_BWD_CALL_(256);
        __GS_BWD_CALL_(257);
        __GS_BWD_CALL_(512);
        __GS_BWD_CALL_(513);
        default:
            TORCH_CHECK(false, "Unsupported number of channels: ", COLOR_DIM);
    }
#undef __GS_BWD_CALL_

    return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities);
}

} // namespace gsplat::xpu