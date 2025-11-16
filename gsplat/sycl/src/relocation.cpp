
#include <c10/xpu/XPUStream.h>

#include "Common.h"
#include "Ops.h"
#include "kernels/RelocationKernel.hpp"
#include "utils.hpp"

namespace gsplat::xpu {

std::tuple<at::Tensor, at::Tensor> relocation(
    at::Tensor opacities, // [N]
    at::Tensor scales,    // [N, 3]
    at::Tensor ratios,    // [N]
    at::Tensor binoms,    // [n_max, n_max]
    const int n_max
) {
    if (opacities.size(0) == 0) {
        return std::make_tuple(
            at::empty_like(opacities), at::empty_like(scales)
        );
    }
    at::Tensor new_opacities = at::empty_like(opacities);
    at::Tensor new_scales = at::empty_like(scales);

    AT_DISPATCH_FLOATING_TYPES(opacities.scalar_type(), "relocation", ([&] {
                                   auto &d_queue =
                                       at::xpu::getCurrentXPUStream().queue();
                                   auto e = d_queue.parallel_for(
                                       sycl::range<1>(opacities.size(0)),
                                       kernels::RelocationKernel<scalar_t>(
                                           opacities.data_ptr<scalar_t>(),
                                           scales.data_ptr<scalar_t>(),
                                           ratios.data_ptr<int>(),
                                           binoms.data_ptr<scalar_t>(),
                                           n_max,
                                           new_opacities.data_ptr<scalar_t>(),
                                           new_scales.data_ptr<scalar_t>()
                                       )
                                   );
                                   e.wait();
                               }));

    return std::make_tuple(new_opacities, new_scales);
}

} // namespace gsplat::xpu