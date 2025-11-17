#include <c10/xpu/XPUStream.h>

#include <cmath>

#include "Common.h"
#include "Ops.h"
#include "kernels/IsectTilesKernel.hpp"

namespace gsplat::xpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
    const at::Tensor means2d,                    // [..., C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., C, N] or [nnz]
    const at::Tensor depths,                     // [..., C, N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz] -> maps to camera_ids
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,                            // -> maps to C
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool segmented
) {
    CHECK_CONTIGUOUS(means2d);
    CHECK_CONTIGUOUS(radii);
    CHECK_CONTIGUOUS(depths);
    if (image_ids.has_value())
        CHECK_CONTIGUOUS(image_ids.value());
    if (gaussian_ids.has_value())
        CHECK_CONTIGUOUS(gaussian_ids.value());

    const bool packed = segmented;
    const uint32_t C = I;
    uint32_t N = 0;
    uint32_t nnz = 0;
    uint32_t total_elems = 0;

    if (packed) {
        nnz = means2d.size(0);
        total_elems = nnz;
        TORCH_CHECK(
            (image_ids.has_value()) && (gaussian_ids.has_value()),
            "When segmented (packed) is set, image_ids and gaussian_ids "
            "must be provided."
        );
    } else {
        N = means2d.size(-2);
        total_elems = C * N;
    }

    if (total_elems == 0) {
        return std::make_tuple(
            at::empty_like(depths, at::kInt),
            at::empty({0}, at::kLong),
            at::empty({0}, at::kInt)
        );
    }
    auto options = depths.options();
    at::Tensor tiles_per_gauss =
        at::empty_like(depths, options.dtype(at::kInt));
    const uint32_t n_tiles = tile_width * tile_height;
    const uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    const uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    TORCH_CHECK(
        tile_n_bits + cam_n_bits <= 32,
        "Not enough bits to encode camera and tile IDs."
    );

    auto &d_queue = at::xpu::getCurrentXPUStream().queue();
    size_t numWorkGrps =
        (total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
    sycl::range<1> localRange(GSPLAT_N_THREADS);
    sycl::range<1> globalRange(GSPLAT_N_THREADS * numWorkGrps);
    sycl::nd_range<1> range(globalRange, localRange);

    auto e1 = d_queue.submit([&](sycl::handler &cgh) {
        IsectTilesKernel<float> kernel(
            packed,
            C,
            N,
            nnz,
            packed ? image_ids.value().data_ptr<int64_t>() : nullptr,
            packed ? gaussian_ids.value().data_ptr<int64_t>() : nullptr,
            means2d.data_ptr<float>(),
            radii.data_ptr<int32_t>(),
            depths.data_ptr<float>(),
            nullptr, // cum_tiles_per_gauss
            tile_size,
            tile_width,
            tile_height,
            tile_n_bits,
            tiles_per_gauss.data_ptr<int32_t>(),
            nullptr, // isect_ids
            nullptr  // flatten_ids
        );
        cgh.parallel_for(range, kernel);
    });
    e1.wait();

    at::Tensor cum_tiles_per_gauss =
        at::cumsum(tiles_per_gauss.view({-1}), 0, at::kLong);
    int64_t n_isects = 0;
    if (total_elems > 0) {
        n_isects = cum_tiles_per_gauss.slice(0, -1).item<int64_t>();
    }

    at::Tensor isect_ids = at::empty({n_isects}, options.dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, options.dtype(at::kInt));

    if (n_isects > 0) {
        auto e2 = d_queue.submit([&](sycl::handler &cgh) {
            IsectTilesKernel<float> kernel(
                packed,
                C,
                N,
                nnz,
                packed ? image_ids.value().data_ptr<int64_t>() : nullptr,
                packed ? gaussian_ids.value().data_ptr<int64_t>() : nullptr,
                means2d.data_ptr<float>(),
                radii.data_ptr<int32_t>(),
                depths.data_ptr<float>(),
                cum_tiles_per_gauss.data_ptr<int64_t>(),
                tile_size,
                tile_width,
                tile_height,
                tile_n_bits,
                nullptr, // tiles_per_gauss
                isect_ids.data_ptr<int64_t>(),
                flatten_ids.data_ptr<int32_t>()
            );
            cgh.parallel_for(range, kernel);
        });
        e2.wait();
    }

    if (n_isects > 0 && sort) {
        auto [sorted_isect_ids, sort_indices] = at::sort(isect_ids);
        isect_ids = sorted_isect_ids;
        flatten_ids = flatten_ids.index_select(0, sort_indices);
    }

    return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
}

} // namespace gsplat::xpu