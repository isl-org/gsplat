#pragma once

/****************************************************************************
 * World to Camera Transformation Forward Pass
 * From:
 *https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/world_to_cam_fwd.cu
 ****************************************************************************/

#include "transform.hpp"
#include "types.hpp"

namespace gsplat::xpu {

template <typename T> struct WorldToCamFwdKernel {
    const uint32_t m_C;
    const uint32_t m_N;
    const T *m_means;    // [N, 3]
    const T *m_covars;   // [N, 3, 3]
    const T *m_viewmats; // [C, 4, 4]
    T *m_means_c;        // [C, N, 3]
    T *m_covars_c;       // [C, N, 3, 3]

    WorldToCamFwdKernel(
        const uint32_t C,
        const uint32_t N,
        const T *means,
        const T *covars,
        const T *viewmats,
        T *means_c,
        T *covars_c
    )
        : m_C(C), m_N(N), m_means(means), m_covars(covars),
          m_viewmats(viewmats), m_means_c(means_c), m_covars_c(covars_c) {}

    void operator()(sycl::nd_item<1> work_item) const {
        const int64_t idx = work_item.get_global_id(0);
        if (idx >= m_C * m_N) {
            return;
        }

        const uint32_t cid = idx / m_N; // camera id
        const uint32_t gid = idx % m_N; // gaussian id

        // shift pointers to the current camera and gaussian
        const T *means = m_means + (gid * 3);
        const T *covars = m_covars + (gid * 9);
        const T *viewmats = m_viewmats + (cid * 16);

        // glm is column-major but input is row-major
        const mat3<T> R = mat3<T>(
            viewmats[0],
            viewmats[4],
            viewmats[8], // 1st column
            viewmats[1],
            viewmats[5],
            viewmats[9], // 2nd column
            viewmats[2],
            viewmats[6],
            viewmats[10] // 3rd column
        );

        const vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

        if (m_means_c != nullptr) {
            vec3<T> mean_c;
            const vec3<T> mean = glm::make_vec3(means);
            pos_world_to_cam(R, t, mean, mean_c);
            T *means_c = m_means_c + (idx * 3);
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) { // rows
                means_c[i] = mean_c[i];
            }
        }

        // write to outputs: glm is column-major but we want row-major
        if (m_covars_c != nullptr) {
            mat3<T> covar_c;
            const mat3<T> covar = glm::make_mat3(covars);
            covar_world_to_cam<T>(R, covar, covar_c);
            T *covars_c = m_covars_c + (idx * 9);
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) { // rows
#pragma unroll
                for (uint32_t j = 0; j < 3; j++) { // cols
                    covars_c[i * 3 + j] = T(covar_c[j][i]);
                }
            }
        }
    }
};

} // namespace  gsplat::xpu