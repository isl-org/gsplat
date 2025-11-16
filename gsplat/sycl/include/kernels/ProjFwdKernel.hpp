#ifndef ProjFwdKernel_HPP
#define ProjFwdKernel_HPP

#include "Common.h"
#include "proj.hpp"

namespace gsplat::xpu {

template <typename T> struct ProjFwdKernel {

    const uint32_t m_C;
    const uint32_t m_N;
    const T *m_means;  // [C, N, 3]
    const T *m_covars; // [C, N, 3, 3]
    const T *m_Ks;     // [C, 3, 3]
    const uint32_t m_width;
    const uint32_t m_height;
    const CameraModelType m_camera_model;
    T *m_means2d;  // [C, N, 2]
    T *m_covars2d; // [C, N, 2, 2]

    ProjFwdKernel(
        const uint32_t C,
        const uint32_t N,
        const T *means,  // [C, N, 3]
        const T *covars, // [C, N, 3, 3]
        const T *Ks,     // [C, 3, 3]
        const uint32_t width,
        const uint32_t height,
        const CameraModelType camera_model,
        T *means2d, // [C, N, 2]
        T *covars2d // [C, N, 2, 2]
    )
        : m_C(C), m_N(N), m_means(means), m_covars(covars), m_Ks(Ks),
          m_width(width), m_height(height), m_camera_model(camera_model),
          m_means2d(means2d), m_covars2d(covars2d) {}

    void operator()(sycl::nd_item<1> work_item) const {
        uint32_t idx = work_item.get_global_id(0);
        const uint32_t total_gaussians =
            (work_item.get_group_range(0) * work_item.get_local_range(0));
        if (idx >= total_gaussians) {
            return;
        }

        const uint32_t bid = idx / (m_C * m_N); // batch id
        const uint32_t cid = (idx / m_N) % m_C; // camera id

        const T *means = m_means + (idx * 3);
        const T *covars = m_covars + (idx * 9);
        const T *Ks = m_Ks + (bid * m_C * 9) + (cid * 9);

        T *means2d = m_means2d + (idx * 2);
        T *covars2d = m_covars2d + (idx * 4);

        T fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
        mat2<T> covar2d(0.f);
        vec2<T> mean2d(0.f);
        const vec3<T> mean = glm::make_vec3(means);
        const mat3<T> covar = glm::make_mat3(covars);

        switch (m_camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj(
                mean, covar, fx, fy, cx, cy, m_width, m_height, covar2d, mean2d
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj(
                mean, covar, fx, fy, cx, cy, m_width, m_height, covar2d, mean2d
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj(
                mean, covar, fx, fy, cx, cy, m_width, m_height, covar2d, mean2d
            );
            break;
        }

#pragma unroll
        for (uint32_t i = 0; i < 2; i++) { // rows
#pragma unroll
            for (uint32_t j = 0; j < 2; j++) { // cols
                covars2d[i * 2 + j] = T(covar2d[j][i]);
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < 2; i++) {
            means2d[i] = T(mean2d[i]);
        }
    }
};
#endif // ProjFwdKernel_HPP

} // namespace gsplat::xpu