#ifndef QuatScaleToCovarPreciBwdKernel_HPP
#define QuatScaleToCovarPreciBwdKernel_HPP


#include "quat_scale_to_covar_preci.hpp"

namespace gsplat::xpu {
    
template<typename T>
struct QuatScaleToCovarPreciBwdKernel{

    const uint32_t m_N;
    // fwd inputs
    const T* m_quats;  // [N, 4]
    const T* m_scales; // [N, 3]
    // grad outputs
    const T* m_v_covars; // [N, 3, 3] or [N, 6]
    const T* m_v_precis; // [N, 3, 3] or [N, 6]
    const bool m_triu;
    // grad inputs
    T* m_v_scales; // [N, 3]
    T* m_v_quats;   // [N, 4]

    QuatScaleToCovarPreciBwdKernel(
        const uint32_t N,
        const T* quats,
        const T* scales,
        const T* v_covars,
        const T* v_precis,
        const bool triu,
        T* v_scales,
        T* v_quats
    )
    : m_N(N), m_quats(quats), m_scales(scales), m_v_covars(v_covars), m_v_precis(v_precis),
      m_triu(triu), m_v_scales(v_scales), m_v_quats(v_quats)  
    {}

    void operator()(sycl::nd_item<1> work_item) const 
    {
        uint32_t idx = work_item.get_global_id(0);
        if (idx >= m_N) {
            return;
        }

        T* v_scales = m_v_scales + (idx * 3);
        T* v_quats = m_v_quats + (idx * 4);

        vec4<T> quat = glm::make_vec4(m_quats + (idx * 4));
        vec3<T> scale = glm::make_vec3(m_scales + (idx * 3));
        mat3<T> rotmat = quat_to_rotmat<T>(quat);
        
        vec4<T> v_quat(0.f);
        vec3<T> v_scale(0.f);

        if (m_v_covars != nullptr) {
            // glm is column-major, input is row-major
            mat3<T> v_covar;
            if (m_triu) {
                const T* v_covars = m_v_covars + (idx * 6);
                v_covar = mat3<T>(
                    v_covars[0],
                    v_covars[1] * .5f,
                    v_covars[2] * .5f,
                    v_covars[1] * .5f,
                    v_covars[3],
                    v_covars[4] * .5f,
                    v_covars[2] * .5f,
                    v_covars[4] * .5f,
                    v_covars[5]
                );
            } else {
                const T* v_covars = m_v_covars + (idx * 9);
                mat3<T> v_covar_cast = glm::make_mat3(v_covars);
                v_covar = glm::transpose(v_covar_cast);
            }
            quat_scale_to_covar_vjp<T>(
                quat, scale, rotmat, v_covar, v_quat, v_scale
            );
        }

        if (m_v_precis != nullptr) {
            // glm is column-major, input is row-major
            mat3<T> v_preci;
            if (m_triu) {
                const T* v_precis = m_v_precis + (idx * 6);
                v_preci = mat3<T>(
                    v_precis[0],
                    v_precis[1] * .5f,
                    v_precis[2] * .5f,
                    v_precis[1] * .5f,
                    v_precis[3],
                    v_precis[4] * .5f,
                    v_precis[2] * .5f,
                    v_precis[4] * .5f,
                    v_precis[5]
                );
            } else {
                const T* v_precis = m_v_precis + (idx * 9);
                mat3<T> v_precis_cast = glm::make_mat3(v_precis);
                v_preci = glm::transpose(v_precis_cast);
            }
            quat_scale_to_preci_vjp<T>(
                quat, scale, rotmat, v_preci, v_quat, v_scale
            );
        }

        #pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            v_scales[k] = T(v_scale[k]);
        }
        #pragma unroll
        for (uint32_t k = 0; k < 4; ++k) {
            v_quats[k] = T(v_quat[k]);
        }
    }   

};

#endif //QuatScaleToCovarPreciBwdKernel_HPP

} // namespace  gsplat::xpu