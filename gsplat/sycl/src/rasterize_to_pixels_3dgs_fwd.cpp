#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "kernels/RasterizeToPixelsFwdKernel.hpp"

namespace gsplat::xpu {

namespace {

template <uint32_t COLOR_DIM>
void launch_rasterize_kernel(
    // Gaussian parameters
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::optional<at::Tensor> &backgrounds,
    const at::optional<at::Tensor> &masks,
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    // other params
    bool packed,
    uint32_t C,
    uint32_t N,
    uint32_t tile_height,
    uint32_t tile_width,
    // outputs
    at::Tensor &renders,
    at::Tensor &alphas,
    at::Tensor &last_ids
) {
    auto &d_queue = at::xpu::getCurrentXPUStream().queue();

    sycl::range<3> localRange{1, tile_size, tile_size};
    sycl::range<3> globalRange{
        C, tile_height * tile_size, tile_width * tile_size
    };
    sycl::nd_range<3> range(globalRange, localRange);

    auto e = d_queue.submit([&](sycl::handler &cgh) {
        constexpr uint32_t CHUNK_SIZE = 128;
        sycl::range<1> slm_range(tile_size * tile_size);

        sycl::local_accessor<int32_t, 1> slm_flatten_ids(slm_range, cgh);
        sycl::local_accessor<sycl::vec<float, 2>, 1> slm_means2d(
            slm_range, cgh
        );
        sycl::local_accessor<float, 1> slm_opacities(slm_range, cgh);
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_conics(slm_range, cgh);
        sycl::local_accessor<BufferType_t<float, COLOR_DIM>, 1> slm_color;
        if constexpr (BufferType<float, COLOR_DIM>::isVec && COLOR_DIM <= 4) {
            slm_color = sycl::local_accessor<BufferType_t<float, COLOR_DIM>, 1>(
                slm_range, cgh
            );
        }

        RasterizeToPixelsFwdKernel<COLOR_DIM, CHUNK_SIZE, float, false> kernel(
            C,
            N,
            flatten_ids.size(0),
            packed,
            0,
            nullptr, // concat_stride, concatenated_data
            reinterpret_cast<const sycl::vec<float, 2> *>(
                means2d.data_ptr<float>()
            ),
            reinterpret_cast<const vec3<float> *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
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
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            slm_flatten_ids,
            slm_means2d,
            slm_opacities,
            slm_conics,
            slm_color
        );
        cgh.parallel_for(range, kernel);
    });
    e.wait();
}
} // anonymous namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> rasterize_to_pixels_3dgs_fwd(
    // Gaussian parameters
    const at::Tensor means2d,   // [..., C, N, 2] or [C, N, 2]
    const at::Tensor conics,    // [..., C, N, 3] or [C, N, 3]
    const at::Tensor colors,    // [..., C, N, COLOR_DIM] or [C, N, COLOR_DIM]
    const at::Tensor opacities, // [..., C, N] or [C, N]
    const at::optional<at::Tensor>
        backgrounds, // [..., C, COLOR_DIM] or [C, COLOR_DIM] optional
    const at::optional<at::Tensor>
        masks, // [..., C, image_height, image_width] optional
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT2(conics, means2d);
    CHECK_INPUT2(colors, means2d);
    CHECK_INPUT2(opacities, means2d);
    CHECK_INPUT2(tile_offsets, means2d);
    CHECK_INPUT2(flatten_ids, means2d);
    if (backgrounds.has_value())
        CHECK_INPUT2(backgrounds.value(), means2d);
    if (masks.has_value())
        CHECK_INPUT2(masks.value(), means2d);

    TORCH_CHECK(means2d.dim() >= 2, "means2d must have at least 2 dimensions");
    TORCH_CHECK(colors.dim() >= 2, "colors must have at least 2 dimensions");

    const uint32_t channels = colors.size(-1);
    const bool packed = means2d.dim() == 2;
    const uint32_t C = tile_offsets.size(0);
    const uint32_t N = packed ? 0 : means2d.size(-2);
    const uint32_t tile_height = tile_offsets.size(1);
    const uint32_t tile_width = tile_offsets.size(2);

    auto options_float = means2d.options().dtype(torch::kFloat32);
    auto options_int = means2d.options().dtype(torch::kInt32);
    at::DimVector image_dims(
        tile_offsets.sizes().slice(0, tile_offsets.dim() - 2)
    );

    at::DimVector out_shape_renders = image_dims;
    out_shape_renders.append({image_height, image_width, channels});

    at::DimVector out_shape_alphas = image_dims;
    out_shape_alphas.append({image_height, image_width, 1});

    at::DimVector out_shape_last_ids = image_dims;
    out_shape_last_ids.append({image_height, image_width});

    at::Tensor renders = at::empty(out_shape_renders, options_float);
    at::Tensor alphas = at::empty(out_shape_alphas, options_float);
    at::Tensor last_ids = at::empty(out_shape_last_ids, options_int);

#define __GS__CALL_(DIM)                                                       \
    case DIM:                                                                  \
        launch_rasterize_kernel<DIM>(                                          \
            means2d,                                                           \
            conics,                                                            \
            colors,                                                            \
            opacities,                                                         \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            packed,                                                            \
            C,                                                                 \
            N,                                                                 \
            tile_height,                                                       \
            tile_width,                                                        \
            renders,                                                           \
            alphas,                                                            \
            last_ids                                                           \
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

    return std::make_tuple(renders, alphas, last_ids);
}

} // namespace gsplat::xpu