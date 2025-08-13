 
#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"

namespace  gsplat::xpu {
    
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_fused_bwd(
    // fwd inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [..., C, N, 2]
    const at::Tensor conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> compensations, // [..., C, N] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [..., C, N, 2]
    const at::Tensor v_depths,                      // [..., C, N]
    const at::Tensor v_conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [..., C, N] optional
    const bool viewmats_requires_grad
) {
    throw std::runtime_error(std::string(__func__) + " is not implemented");
}

} // namespace  gsplat::xpu