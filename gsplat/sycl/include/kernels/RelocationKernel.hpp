#pragma once
#include <sycl/sycl.hpp>

namespace gsplat::xpu::kernels {

template <typename scalar_t> class RelocationKernel {
  private:
    const scalar_t *opacities;
    const scalar_t *scales;
    const int *ratios;
    const scalar_t *binoms;
    const int n_max;
    scalar_t *new_opacities;
    scalar_t *new_scales;

  public:
    RelocationKernel(
        const scalar_t *opacities,
        const scalar_t *scales,
        const int *ratios,
        const scalar_t *binoms,
        const int n_max,
        scalar_t *new_opacities,
        scalar_t *new_scales
    )
        : opacities(opacities), scales(scales), ratios(ratios), binoms(binoms),
          n_max(n_max), new_opacities(new_opacities), new_scales(new_scales) {}

    void operator()(sycl::id<1> item) const {
        int idx = item[0];

        int n_idx = ratios[idx];
        float denom_sum = 0.0f;

        // compute new opacity
        new_opacities[idx] =
            1.0f -
            sycl::pow(1.0f - static_cast<float>(opacities[idx]), 1.0f / n_idx);

        // compute new scale
        for (int i = 1; i <= n_idx; ++i) {
            for (int k = 0; k <= (i - 1); ++k) {
                float bin_coeff = binoms[(i - 1) * n_max + k];
                float term =
                    (sycl::pow(-1.0f, k) /
                     sycl::sqrt(static_cast<float>(k + 1))) *
                    sycl::pow(static_cast<float>(new_opacities[idx]), k + 1);
                denom_sum += (bin_coeff * term);
            }
        }
        float coeff = (opacities[idx] / denom_sum);
        for (int i = 0; i < 3; ++i)
            new_scales[idx * 3 + i] = coeff * scales[idx * 3 + i];
    }
};

} // namespace gsplat::xpu::kernels
