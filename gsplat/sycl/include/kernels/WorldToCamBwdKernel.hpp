#ifndef WorldToCamBwdKernel_HPP
#define WorldToCamBwdKernel_HPP

 
#include "types.hpp"
#include "transform.hpp"
#include "utils.hpp"

namespace gsplat::xpu {
    
template <typename T>
struct WorldToCamBwdKernel{
    const uint32_t m_C;
    const uint32_t m_N;
    const T* m_means;      // [N, 3]
    const T* m_covars;     // [N, 3, 3]
    const T* m_viewmats;   // [C, 4, 4]
    const T* m_v_means_c;  // [C, N, 3]
    const T* m_v_covars_c; // [C, N, 3, 3]
    T* m_v_means;          // [N, 3]
    T* m_v_covars;         // [N, 3, 3]
    T* m_v_viewmats;        // [C, 4, 4]

    WorldToCamBwdKernel(
        const uint32_t C,
        const uint32_t N,
        const T* means,
        const T* covars,
        const T* viewmats,
        const T* v_means_c,
        const T* v_covars_c,
        T* v_means,
        T* v_covars,
        T* v_viewmats
    ) : m_C(C), m_N(N),
        m_means(means), m_covars(covars), m_viewmats(viewmats),
        m_v_means_c(v_means_c), m_v_covars_c(v_covars_c),
        m_v_means(v_means), m_v_covars(v_covars), m_v_viewmats(v_viewmats)
    {}

    void operator()(sycl::nd_item<1> work_item)  const
    {
        const uint32_t idx = work_item.get_global_id(0);

        if (idx >= m_C * m_N) {
            return;
        }

        const uint32_t cid = idx / m_N; // camera id
        const uint32_t gid = idx % m_N; // gaussian id

        // shift pointers to the current camera and gaussian
        const T* means = m_means + (gid * 3);
        const T* covars = m_covars + (gid * 9);
        const T* viewmats = m_viewmats + (cid * 16);

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
        
        vec3<T> v_mean(0.f);
        mat3<T> v_covar(0.f);
        mat3<T> v_R(0.f);
        vec3<T> v_t(0.f);

        if (m_v_means_c != nullptr) {
            const vec3<T> v_mean_c = glm::make_vec3(m_v_means_c + (idx * 3));
            const vec3<T> mean = glm::make_vec3(means);
            pos_world_to_cam_vjp<T>(R, t, mean, v_mean_c, v_R, v_t, v_mean);
        }
        if (m_v_covars_c != nullptr) {
            const mat3<T> v_covar_c_t = glm::make_mat3(m_v_covars_c + (idx * 9));
            const mat3<T> v_covar_c = glm::transpose(v_covar_c_t);
            const mat3<T> covar = glm::make_mat3(covars);
            covar_world_to_cam_vjp<T>(R, covar, v_covar_c, v_R, v_covar);
        }

        if (m_v_means != nullptr) {
            T* v_means = m_v_means + (gid * 3);
            #pragma unroll
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd( v_means + i, v_mean[i]);
            }            
        }

        if (m_v_covars != nullptr) {
            T* v_covars = m_v_covars + (gid * 9);
            #pragma unroll
            for (uint32_t i = 0; i < 3; i++) { // rows
                #pragma unroll
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_covars + i * 3 + j, v_covar[j][i]);
                }
            }     
        }

        if (m_v_viewmats != nullptr) {
            T* v_viewmats = m_v_viewmats + cid * 16;
            #pragma unroll 
            for (uint32_t i = 0; i < 3; i++) { // rows
                #pragma unroll
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
};

#endif //WorldToCamBwdKernel_HPP

} // namespace  gsplat::xpu