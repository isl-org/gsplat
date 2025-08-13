 
#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"

namespace  gsplat::xpu {
    
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_packed_bwd(
    // fwd inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6]
    const at::optional<at::Tensor> quats,  // [..., N, 4]
    const at::optional<at::Tensor> scales, // [..., N, 3]
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor batch_ids,                   // [nnz]
    const at::Tensor camera_ids,                  // [nnz]
    const at::Tensor gaussian_ids,                // [nnz]
    const at::Tensor conics,                      // [nnz, 3]
    const at::optional<at::Tensor> compensations, // [nnz] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [nnz, 2]
    const at::Tensor v_depths,                      // [nnz]
    const at::Tensor v_conics,                      // [nnz, 3]
    const at::optional<at::Tensor> v_compensations, // [nnz] optional
    const bool viewmats_requires_grad,
    const bool sparse_grad
) {
    throw std::runtime_error(std::string(__func__) + " is not implemented");
}

} // namespace  gsplat::xpu