#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"
#include "kernels/ProjBwdKernel.hpp"

namespace gsplat::xpu {

std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_bwd(
    const at::Tensor means,      // [..., C, N, 3]
    const at::Tensor covars,     // [..., C, N, 3, 3]
    const at::Tensor Ks,         // [..., C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const at::Tensor v_means2d,  // [..., C, N, 2]
    const at::Tensor v_covars2d  // [..., C, N, 2, 2]
) {

    CHECK_CONTIGUOUS(means);
    CHECK_CONTIGUOUS(covars);
    CHECK_CONTIGUOUS(Ks);
    CHECK_CONTIGUOUS(v_means2d);
    CHECK_CONTIGUOUS(v_covars2d);


    const uint32_t C = means.size(-3);
    const uint32_t N = means.size(-2);


    at::Tensor v_means = at::empty({C, N, 3}, means.options());
    at::Tensor v_covars = at::empty({C, N, 3, 3}, covars.options());


    if (C > 0 && N > 0) {
        auto& d_queue = at::xpu::getCurrentXPUStream().queue();
        
        
        size_t numWorkGrps = (C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
        sycl::range<1> localRange(GSPLAT_N_THREADS);
        sycl::range<1> globalRange(GSPLAT_N_THREADS * numWorkGrps);
        sycl::nd_range<1> range(globalRange, localRange);

        auto e = d_queue.submit(
            [&](sycl::handler& cgh)
            {
                ProjBwdKernel<float> kernel(
                    C,
                    N,
                    means.data_ptr<float>(),
                    covars.data_ptr<float>(),
                    Ks.data_ptr<float>(),
                    width,
                    height,
                    camera_model,
                    v_means2d.data_ptr<float>(),
                    v_covars2d.data_ptr<float>(),
                    v_means.data_ptr<float>(),
                    v_covars.data_ptr<float>()
                );
                cgh.parallel_for(range, kernel);
            }
        );
        e.wait();
    }
    
    return std::make_tuple(v_means, v_covars);
}

} // namespace gsplat::xpu