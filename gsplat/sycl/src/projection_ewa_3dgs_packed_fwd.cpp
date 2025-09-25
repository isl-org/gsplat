#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"
#include "kernels/PackedProjectionFwdKernel.hpp"

namespace gsplat::xpu {

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
    const CameraModelType camera_model) {
    
    TORCH_CHECK(means.is_contiguous(), "Input 'means' tensor must be contiguous.");
    TORCH_CHECK(viewmats.is_contiguous(), "Input 'viewmats' tensor must be contiguous.");
    TORCH_CHECK(Ks.is_contiguous(), "Input 'Ks' tensor must be contiguous.");
    TORCH_CHECK(means.device().type() == at::kXPU, "Input tensors must be on XPU device.");

    uint32_t N = means.size(-2);
    uint32_t C = viewmats.size(-3);
    uint32_t B = means.numel() / (N * 3);

    uint32_t nrows = B * C;
    uint32_t ncols = N;
    uint32_t blocks_per_row = (ncols + N_THREADS_PACKED - 1) / N_THREADS_PACKED;
    uint32_t n_blocks = nrows * blocks_per_row;

    // Create empty outputs for the case where there's nothing to process
    auto long_opts = means.options().dtype(at::kLong);
    auto int_opts = means.options().dtype(at::kInt);
    auto float_opts = means.options();

    at::Tensor batch_ids = at::empty({0}, long_opts);
    at::Tensor camera_ids = at::empty({0}, long_opts);
    at::Tensor gaussian_ids = at::empty({0}, long_opts);
    at::Tensor radii = at::empty({0, 2}, int_opts);
    at::Tensor means2d = at::empty({0, 2}, float_opts);
    at::Tensor depths = at::empty({0}, float_opts);
    at::Tensor conics = at::empty({0, 3}, float_opts);
    at::Tensor indptr = at::zeros({nrows + 1}, int_opts);
    at::Tensor compensations = at::empty({0}, float_opts);

    if (B == 0 || C == 0 || N == 0) {
        return std::make_tuple(
            batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, indptr, compensations);
    }
    
    // --- Start of Correction ---
    // Changed block_cnts to kLong to satisfy at::cumsum requirements
    at::Tensor block_cnts = at::empty({(long)n_blocks}, long_opts);
    // --- End of Correction ---

    auto& d_queue = at::xpu::getCurrentXPUStream().queue();
    sycl::range<2> local_range(1, N_THREADS_PACKED);
    sycl::range<2> global_range(nrows, blocks_per_row * N_THREADS_PACKED);
    sycl::nd_range<2> range(global_range, local_range);

    // First pass: count visible Gaussians per block
    AT_DISPATCH_FLOATING_TYPES(means.scalar_type(), "projection_ewa_3dgs_packed_fwd_kernel_pass1", [&] {
        d_queue.parallel_for(range, PackedProjectionFwdKernel<scalar_t>(
            B, C, N,
            means.data_ptr<scalar_t>(),
            covars.has_value() ? covars.value().data_ptr<scalar_t>() : nullptr,
            quats.has_value() ? quats.value().data_ptr<scalar_t>() : nullptr,
            scales.has_value() ? scales.value().data_ptr<scalar_t>() : nullptr,
            opacities.has_value() ? opacities.value().data_ptr<scalar_t>() : nullptr,
            viewmats.data_ptr<scalar_t>(),
            Ks.data_ptr<scalar_t>(),
            image_width, image_height,
            (scalar_t)eps2d, (scalar_t)near_plane, (scalar_t)far_plane, (scalar_t)radius_clip,
            camera_model,
            nullptr, // block_accum
            // --- Start of Correction ---
            // The kernel expects int32_t*, but the tensor is int64_t.
            // This cast is safe because the counts per block will not exceed int32_t max.
            (int32_t*)block_cnts.data_ptr<int64_t>(),
            // --- End of Correction ---
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
        )).wait();
    });

    at::Tensor block_accum_inclusive = at::cumsum(block_cnts, 0);
    
    int64_t nnz = 0; // Use int64_t for nnz to match tensor type
    if (n_blocks > 0) {
       nnz = block_accum_inclusive.index({-1}).item<int64_t>();
    }

    if (nnz == 0) {
         return std::make_tuple(
            batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, indptr, compensations);
    }
    
    // --- Start of Correction ---
    // To get an exclusive scan, shift the inclusive scan and prepend a zero.
    // Ensure the types are consistent (kLong).
    at::Tensor block_accum_exclusive = at::cat({at::zeros({1}, long_opts), block_accum_inclusive.slice(0, 0, n_blocks - 1)});
    // The kernel expects int32_t*, so we must convert the exclusive scan result back to kInt.
    at::Tensor block_accum_exclusive_int = block_accum_exclusive.to(at::kInt);
    // --- End of Correction ---


    // Allocate final output tensors
    batch_ids = at::empty({nnz}, long_opts);
    camera_ids = at::empty({nnz}, long_opts);
    gaussian_ids = at::empty({nnz}, long_opts);
    radii = at::empty({nnz, 2}, int_opts);
    means2d = at::empty({nnz, 2}, float_opts);
    depths = at::empty({nnz}, float_opts);
    conics = at::empty({nnz, 3}, float_opts);
    if (calc_compensations) {
        compensations = at::empty({nnz}, float_opts);
    }
    
    // Second pass: write packed data
    AT_DISPATCH_FLOATING_TYPES(means.scalar_type(), "projection_ewa_3dgs_packed_fwd_kernel_pass2", [&] {
        d_queue.parallel_for(range, PackedProjectionFwdKernel<scalar_t>(
            B, C, N,
            means.data_ptr<scalar_t>(),
            covars.has_value() ? covars.value().data_ptr<scalar_t>() : nullptr,
            quats.has_value() ? quats.value().data_ptr<scalar_t>() : nullptr,
            scales.has_value() ? scales.value().data_ptr<scalar_t>() : nullptr,
            opacities.has_value() ? opacities.value().data_ptr<scalar_t>() : nullptr,
            viewmats.data_ptr<scalar_t>(),
            Ks.data_ptr<scalar_t>(),
            image_width, image_height,
            (scalar_t)eps2d, (scalar_t)near_plane, (scalar_t)far_plane, (scalar_t)radius_clip,
            camera_model,
            // --- Start of Correction ---
            block_accum_exclusive_int.data_ptr<int32_t>(),
            // --- End of Correction ---
            nullptr, // block_cnts
            indptr.data_ptr<int32_t>(),
            batch_ids.data_ptr<int64_t>(),
            camera_ids.data_ptr<int64_t>(),
            gaussian_ids.data_ptr<int64_t>(),
            radii.data_ptr<int32_t>(),
            means2d.data_ptr<scalar_t>(),
            depths.data_ptr<scalar_t>(),
            conics.data_ptr<scalar_t>(),
            calc_compensations ? compensations.data_ptr<scalar_t>() : nullptr
        )).wait();
    });

    // Set the last element of indptr
    if (nrows > 0) {
        indptr.index_put_({(long)nrows}, nnz);
    }

    return std::make_tuple(
        batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, indptr, compensations);
}

} // namespace  gsplat::xpu