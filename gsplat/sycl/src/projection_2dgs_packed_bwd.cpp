
#include <c10/xpu/XPUStream.h>

#include "Common.h"
#include "Ops.h"

namespace gsplat::xpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_2dgs_packed_bwd(
    // fwd inputs
    const at::Tensor means,    // [..., N, 3]
    const at::Tensor quats,    // [..., N, 4]
    const at::Tensor scales,   // [..., N, 3]
    const at::Tensor viewmats, // [..., C, 4, 4]
    const at::Tensor Ks,       // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const at::Tensor batch_ids,      // [nnz]
    const at::Tensor camera_ids,     // [nnz]
    const at::Tensor gaussian_ids,   // [nnz]
    const at::Tensor ray_transforms, // [nnz, 3, 3]
    // grad outputs
    const at::Tensor v_means2d,        // [nnz, 2]
    const at::Tensor v_depths,         // [nnz]
    const at::Tensor v_ray_transforms, // [nnz, 3, 3]
    const at::Tensor v_normals,        // [nnz, 3]
    const bool viewmats_requires_grad,
    const bool sparse_grad
) {
    throw std::runtime_error(std::string(__func__) + " is not implemented");
}

} // namespace  gsplat::xpu