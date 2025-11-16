#ifndef QuatScaleToCovarPreciFwdKernel_HPP
#define QuatScaleToCovarPreciFwdKernel_HPP

#include "quat_scale_to_covar_preci.hpp"

namespace gsplat::xpu {

template <typename T> struct QuatScaleToCovarPreciFwdKernel {

    const uint32_t m_N;
    const T *m_quats;  // [N, 4]
    const T *m_scales; // [N, 3]
    const bool m_triu;
    // outputs
    T *m_covars; // [N, 3, 3] or [N, 6]
    T *m_precis; // [N, 3, 3] or [N, 6]

    QuatScaleToCovarPreciFwdKernel(
        const uint32_t N,
        const T *quats,
        const T *scales,
        const bool triu,
        T *covars,
        T *precis
    )
        : m_N(N), m_quats(quats), m_scales(scales), m_triu(triu),
          m_covars(covars), m_precis(precis) {}

    void operator()(sycl::nd_item<1> work_item) const {
        uint32_t idx = work_item.get_global_id(0);
        if (idx >= m_N) {
            return;
        }

        const T *quats = m_quats + (idx * 4);
        const T *scales = m_scales + (idx * 3);

        mat3<T> covar, preci;
        const vec4<T> quat = glm::make_vec4(quats);
        const vec3<T> scale = glm::make_vec3(scales);
        quat_scale_to_covar_preci(
            quat,
            scale,
            m_covars ? &covar : nullptr,
            m_precis ? &preci : nullptr
        );

        // write to outputs: glm is column-major but we want row-major
        if (m_covars != nullptr) {
            if (m_triu) {
                T *covars = m_covars + (idx * 6);
                covars[0] = T(covar[0][0]);
                covars[1] = T(covar[0][1]);
                covars[2] = T(covar[0][2]);
                covars[3] = T(covar[1][1]);
                covars[4] = T(covar[1][2]);
                covars[5] = T(covar[2][2]);
            } else {
                T *covars = m_covars + (idx * 9);
#pragma unroll
                for (uint32_t i = 0; i < 3; i++) { // rows
#pragma unroll
                    for (uint32_t j = 0; j < 3; j++) { // cols
                        covars[i * 3 + j] = T(covar[j][i]);
                    }
                }
            }
        }

        if (m_precis != nullptr) {
            if (m_triu) {
                T *precis = m_precis + (idx * 6);
                precis[0] = T(preci[0][0]);
                precis[1] = T(preci[0][1]);
                precis[2] = T(preci[0][2]);
                precis[3] = T(preci[1][1]);
                precis[4] = T(preci[1][2]);
                precis[5] = T(preci[2][2]);
            } else {
                T *precis = m_precis + (idx * 9);
#pragma unroll
                for (uint32_t i = 0; i < 3; i++) { // rows
#pragma unroll
                    for (uint32_t j = 0; j < 3; j++) { // cols
                        precis[i * 3 + j] = T(preci[j][i]);
                    }
                }
            }
        }
    }
};
#endif // QuatScaleToCovarPreciFwdKernel_HPP

} // namespace  gsplat::xpu