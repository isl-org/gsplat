 
#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"

namespace  gsplat::xpu {
    
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_3dgs_bwd(
    // Gaussian parameters
    const at::Tensor means,                     // [..., N, 3]
    const at::Tensor quats,                     // [..., N, 4]
    const at::Tensor scales,                    // [..., N, 3]
    const at::Tensor colors,                    // [..., C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., C, 3]
    const at::optional<at::Tensor> masks,       // [..., C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0,      // [..., C, 4, 4]
    const at::optional<at::Tensor>
        viewmats1,                   // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,             // [..., C, 3, 3]
    const CameraModelType camera_model,
    // uncented transform
    const UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const FThetaCameraDistortionParameters ftheta_coeffs, // shared parameters for all cameras
    // intersections
    const at::Tensor tile_offsets,    // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,     // [n_isects]
    // forward outputs
    const at::Tensor render_alphas,   // [..., C, image_height, image_width, 1]
    const at::Tensor last_ids,        // [..., C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., C, image_height, image_width, 3]
    const at::Tensor v_render_alphas  // [..., C, image_height, image_width, 1]
) {
    throw std::runtime_error(std::string(__func__) + " is not implemented");
}

} // namespace  gsplat::xpu