 
#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"

namespace  gsplat::xpu {
    
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ewa_3dgs_packed_fwd(
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
    throw std::runtime_error(std::string(__func__) + " is not implemented");
}

} // namespace  gsplat::xpu