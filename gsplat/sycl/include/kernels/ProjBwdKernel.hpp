#ifndef ProjBwdKernel_HPP
#define ProjBwdKernel_HPP


#include "proj.hpp"
#include "Common.h"

namespace gsplat::xpu {

template<typename T>
struct ProjBwdKernel{

    const uint32_t m_C;
    const uint32_t m_N;
    const T* m_means;  // [C, N, 3]
    const T* m_covars; // [C, N, 3, 3]
    const T* m_Ks;     // [C, 3, 3]
    const uint32_t m_width;
    const uint32_t m_height;
    const CameraModelType m_camera_model; 
    const T* m_v_means2d;  // [C, N, 2]
    const T* m_v_covars2d; // [C, N, 2, 2]
    T* m_v_means;          // [C, N, 3]
    T* m_v_covars;          // [C, N, 3, 3]

    ProjBwdKernel(
        const uint32_t C,
        const uint32_t N,
        const T* means,
        const T* covars,
        const T* Ks,
        const uint32_t width,
        const uint32_t height,
        const CameraModelType camera_model,
        const T* v_means2d,
        const T* v_covars2d,
        T* v_means,
        T* v_covars 
    ) 
    : m_C(C), m_N(N), m_means(means), m_covars(covars), m_Ks(Ks), 
      m_width(width), m_height(height), m_camera_model(camera_model), 
      m_v_means2d(v_means2d), m_v_covars2d(v_covars2d), 
      m_v_means(v_means), m_v_covars(v_covars)
    {}

    void operator()(sycl::nd_item<1> work_item) const {

        uint32_t idx = work_item.get_global_id(0);
        const uint32_t total_gaussians = (work_item.get_group_range(0) * work_item.get_local_range(0));
        if (idx >= total_gaussians) {
            return;
        }
        
        const uint32_t bid = idx / (m_C * m_N); // batch id
        const uint32_t cid = (idx / m_N) % m_C; // camera id

        const T* means = m_means + (idx * 3);
        const T* covars = m_covars + (idx * 9);
        T* v_means = m_v_means + (idx * 3);
        T* v_covars = m_v_covars + (idx * 9);
        // Correctly index Ks using batch and camera id
        const T* Ks = m_Ks + (bid * m_C * 9) + (cid * 9);
        const T* v_means2d = m_v_means2d + (idx * 2);
        const T* v_covars2d = m_v_covars2d + (idx * 4);

        T fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
        mat3<T> v_covar(0.f);
        vec3<T> v_mean(0.f);
        const vec3<T> mean = glm::make_vec3(means);
        const mat3<T> covar = glm::make_mat3(covars);
        const vec2<T> v_mean2d = glm::make_vec2(v_means2d);
        const mat2<T> v_covar2d = glm::make_mat2(v_covars2d);

        switch (m_camera_model) {
            case CameraModelType::PINHOLE: // perspective projection
                persp_proj_vjp<T>(
                    mean,
                    covar,
                    fx,
                    fy,
                    cx,
                    cy,
                    m_width,
                    m_height,
                    glm::transpose(v_covar2d),
                    v_mean2d,
                    v_mean,
                    v_covar
                );
                break;
            case CameraModelType::ORTHO: // orthographic projection
                ortho_proj_vjp<T>(
                    mean,
                    covar,
                    fx,
                    fy,
                    cx,
                    cy,
                    m_width,
                    m_height,
                    glm::transpose(v_covar2d),
                    v_mean2d,
                    v_mean,
                    v_covar
                );
                break;
            case CameraModelType::FISHEYE: // fisheye projection
                fisheye_proj_vjp<T>(
                    mean,
                    covar,
                    fx,
                    fy,
                    cx,
                    cy,
                    m_width,
                    m_height,
                    glm::transpose(v_covar2d),
                    v_mean2d,
                    v_mean,
                    v_covar
                );
                break;
        }
        // write to outputs: glm is column-major but we want row-major
        #pragma unroll
        for (uint32_t i = 0; i < 3; i++) { // rows
            #pragma unroll
            for (uint32_t j = 0; j < 3; j++) { // cols
                v_covars[i * 3 + j] = T(v_covar[j][i]);
            }
        }

        #pragma unroll
        for (uint32_t i = 0; i < 3; i++) {
            v_means[i] = T(v_mean[i]);
        }
    }
};

#endif //ProjBwdKernel_HPP

} // namespace  gsplat::xpu