 
#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"

namespace  gsplat::xpu {
    
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ut_3dgs_fused(
    const at::Tensor means,                   // [..., N, 3]
    const at::Tensor quats,                   // [..., N, 4]
    const at::Tensor scales,                  // [..., N, 3]
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor>
        viewmats1,                            // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model,
    // uncented transform
    const UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs,  // [..., C, 4] optional
    const FThetaCameraDistortionParameters ftheta_coeffs // shared parameters for all cameras
) {
    throw std::runtime_error(std::string(__func__) + " is not implemented");
}

} // namespace  gsplat::xpu