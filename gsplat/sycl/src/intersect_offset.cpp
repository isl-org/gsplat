#include <c10/xpu/XPUStream.h>

#include <cmath>

#include "Common.h"
#include "Ops.h"
#include "kernels/IsectOffsetEncodeKernel.hpp"

namespace gsplat::xpu {

at::Tensor intersect_offset(
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t I,
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    DEVICE_GUARD(isect_ids);
    CHECK_INPUT(isect_ids);
    const uint32_t C = I;

    auto options = isect_ids.options().dtype(at::kInt);
    at::Tensor offsets = at::empty({C, tile_height, tile_width}, options);

    const uint32_t n_isects = isect_ids.size(0);

    if (n_isects > 0) {
        const uint32_t n_tiles = tile_width * tile_height;
        const uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;

        auto &d_queue = at::xpu::getCurrentXPUStream().queue();

        size_t numWorkGrps =
            (n_isects + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
        sycl::range<1> localRange(GSPLAT_N_THREADS);
        sycl::range<1> globalRange(GSPLAT_N_THREADS * numWorkGrps);
        sycl::nd_range<1> range(globalRange, localRange);

        auto e = d_queue.submit([&](sycl::handler &cgh) {
            IsectOffsetEncodeKernel kernel(
                n_isects,
                isect_ids.data_ptr<int64_t>(),
                C,
                n_tiles,
                tile_n_bits,
                offsets.data_ptr<int32_t>()
            );
            cgh.parallel_for(range, kernel);
        });
        e.wait();
    } else {
        offsets.fill_(0);
    }

    return offsets;
}

} // namespace gsplat::xpu