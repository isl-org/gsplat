 
#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"

namespace  gsplat::xpu {
    
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
    throw std::runtime_error(std::string(__func__) + " is not implemented");
}

} // namespace  gsplat::xpu