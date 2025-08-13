#ifndef ComputeShFwdKernel_HPP
#define ComputeShFwdKernel_HPP

#include "spherical_harmonics.hpp"

namespace gsplat::xpu {

template<typename T>
struct ComputeShFwdKernel{
    const uint32_t m_N;
    const uint32_t m_K;
    const uint32_t m_degrees_to_use;
    const vec3<T>* m_dirs; // [N, 3]
    const T* m_coeffs;     // [N, K, 3]
    const bool* m_masks;   // [N]
    T* m_colors;            // [N, 3]

    ComputeShFwdKernel(
        const uint32_t N,
        const uint32_t K,
        const uint32_t degrees_to_use,
        const vec3<T>* dirs,
        const T* coeffs,
        const bool* masks,
        T* colors
    ) 
    : m_N(N), m_K(K), 
      m_degrees_to_use(degrees_to_use), m_dirs(dirs), m_coeffs(coeffs), 
      m_masks(masks), m_colors(colors)
    {}

    void operator()(sycl::nd_item<1> work_item)  const
    {
        uint32_t idx = work_item.get_global_id(0);
        if (idx >= m_N * 3) {
            return;
        }
        uint32_t elem_id = idx / 3;
        uint32_t c = idx % 3; // color channel
        if (m_masks != nullptr && !m_masks[elem_id]) {
            return;
        }
        sh_coeffs_to_color_fast(
            m_degrees_to_use,
            c,
            m_dirs[elem_id],
            m_coeffs + elem_id * m_K * 3,
            m_colors + elem_id * 3
        );
    }
};

#endif //ComputeShFwdKernel_HPP

} // namespace  gsplat::xpu