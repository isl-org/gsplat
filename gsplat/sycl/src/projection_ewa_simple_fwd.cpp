#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"
#include "kernels/ProjFwdKernel.hpp"

namespace gsplat::xpu {
    
std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_fwd(
    const at::Tensor means,  // [C, N, 3]
    const at::Tensor covars, // [C, N, 3, 3]
    const at::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model
) {
    CHECK_CONTIGUOUS(means);
    CHECK_CONTIGUOUS(covars);
    CHECK_CONTIGUOUS(Ks);
    TORCH_CHECK(means.dim() >= 3, "means must have at least 3 dimensions [..., C, N, 3]");
    TORCH_CHECK(covars.dim() >= 4, "covars must have at least 4 dimensions [..., C, N, 3, 3]");
    TORCH_CHECK(Ks.dim() >= 3, "Ks must have at least 3 dimensions [..., C, 3, 3]");

    const uint32_t C = means.size(-3);
    const uint32_t N = means.size(-2);

    at::Tensor means2d = at::empty({C, N, 2}, means.options());
    at::Tensor covars2d = at::empty({C, N, 2, 2}, covars.options());

    if (C > 0 && N > 0) {
        auto& d_queue = at::xpu::getCurrentXPUStream().queue();
        
        
        size_t numWorkGrps = (C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
        sycl::range<1> localRange(GSPLAT_N_THREADS);
        sycl::range<1> globalRange(GSPLAT_N_THREADS * numWorkGrps);
        sycl::nd_range<1> range(globalRange, localRange);

        auto e = d_queue.submit(
            [&](sycl::handler& cgh)
            {
                ProjFwdKernel<float> kernel(
                    C,
                    N,
                    means.data_ptr<float>(),
                    covars.data_ptr<float>(),
                    Ks.data_ptr<float>(),
                    width,
                    height,
                    camera_model,
                    means2d.data_ptr<float>(),
                    covars2d.data_ptr<float>()
                );
                cgh.parallel_for(range, kernel);
            }
        );
        e.wait();
    }
    
    return std::make_tuple(means2d, covars2d);
}

} // namespace gsplat::xpu