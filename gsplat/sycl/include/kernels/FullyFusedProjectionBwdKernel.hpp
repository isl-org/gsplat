#ifndef FullyFusedProjectionBwdKernel_HPP
#define FullyFusedProjectionBwdKernel_HPP

#include "proj.hpp"
#include "quat.hpp"
#include "quat_scale_to_covar_preci.hpp"
#include "transform.hpp"
#include "utils.hpp"

namespace gsplat::xpu {

template <typename T> struct FullyFusedProjectionBwdKernel {
    // fwd inputs
    // New: Added B
    const uint32_t m_B;
    const uint32_t m_C;
    const uint32_t m_N;
    const T *m_means;    // [B, N, 3]
    const T *m_covars;   // [B, N, 6] optional
    const T *m_quats;    // [B, N, 4] optional
    const T *m_scales;   // [B, N, 3] optional
    const T *m_viewmats; // [B, C, 4, 4]
    const T *m_Ks;       // [B, C, 3, 3]
    const int32_t m_image_width;
    const int32_t m_image_height;
    const T m_eps2d;
    const CameraModelType m_camera_model;
    // fwd outputs
    // Changed: radii is now [B, C, N, 2]
    const int32_t *m_radii;   // [B, C, N, 2]
    const T *m_conics;        // [B, C, N, 3]
    const T *m_compensations; // [B, C, N] optional
    // grad outputs
    const T *m_v_means2d;       // [B, C, N, 2]
    const T *m_v_depths;        // [B, C, N]
    const T *m_v_conics;        // [B, C, N, 3]
    const T *m_v_compensations; // [B, C, N] optional
    // grad inputs
    T *m_v_means;    // [B, N, 3]
    T *m_v_covars;   // [B, N, 6] optional
    T *m_v_quats;    // [B, N, 4] optional
    T *m_v_scales;   // [B, N, 3] optional
    T *m_v_viewmats; // [B, C, 4, 4] optional

    FullyFusedProjectionBwdKernel(
        // New: Added B
        const uint32_t B,
        const uint32_t C,
        const uint32_t N,
        const T *means,
        const T *covars,
        const T *quats,
        const T *scales,
        const T *viewmats,
        const T *Ks,
        const int32_t image_width,
        const int32_t image_height,
        const T eps2d,
        const CameraModelType camera_model,
        const int32_t *radii,
        const T *conics,
        const T *compensations,
        const T *v_means2d,
        const T *v_depths,
        const T *v_conics,
        const T *v_compensations,
        T *v_means,
        T *v_covars,
        T *v_quats,
        T *v_scales,
        T *v_viewmats
    )
        // New: Added m_B
        : m_B(B), m_C(C), m_N(N), m_means(means), m_covars(covars),
          m_quats(quats), m_scales(scales), m_viewmats(viewmats), m_Ks(Ks),
          m_image_width(image_width), m_image_height(image_height),
          m_eps2d(eps2d), m_camera_model(camera_model), m_radii(radii),
          m_conics(conics), m_compensations(compensations),
          m_v_means2d(v_means2d), m_v_depths(v_depths), m_v_conics(v_conics),
          m_v_compensations(v_compensations), m_v_means(v_means),
          m_v_covars(v_covars), m_v_quats(v_quats), m_v_scales(v_scales),
          m_v_viewmats(v_viewmats) {}

    void operator()(sycl::nd_item<1> work_item) const {
        uint32_t idx = work_item.get_global_id(0);
        // Changed: Updated check to include B and both radii components
        if (idx >= m_B * m_C * m_N ||
            (m_radii[idx * 2] <= 0 || m_radii[idx * 2 + 1] <= 0)) {
            return;
        }

        // Changed: Added bid and updated cid, gid calculation
        const uint32_t bid = idx / (m_C * m_N); // batch id
        const uint32_t cid = (idx / m_N) % m_C; // camera id
        const uint32_t gid = idx % m_N;         // gaussian id

        // Changed: Updated pointer arithmetic to include B
        const T *means = m_means + bid * m_N * 3 + gid * 3;
        const T *viewmats = m_viewmats + bid * m_C * 16 + cid * 16;
        const T *Ks = m_Ks + bid * m_C * 9 + cid * 9;
        const T *conics = m_conics + idx * 3;
        const T *v_means2d = m_v_means2d + idx * 2;
        const T *v_depths = m_v_depths + idx;
        const T *v_conics = m_v_conics + idx * 3;

        // vjp: compute the inverse of the 2d covariance
        mat2<T> covar2d_inv =
            mat2<T>(conics[0], conics[1], conics[1], conics[2]);
        mat2<T> v_covar2d_inv = mat2<T>(
            v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]
        );
        mat2<T> v_covar2d(0.f);
        inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

        if (m_v_compensations != nullptr) {
            // vjp: compensation term
            const T compensation = m_compensations[idx];
            const T v_compensation = m_v_compensations[idx];
            add_blur_vjp(
                m_eps2d, covar2d_inv, compensation, v_compensation, v_covar2d
            );
        }

        // transform Gaussian to camera space
        mat3<T> R = mat3<T>(
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
        vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

        mat3<T> covar;
        vec4<T> quat;
        vec3<T> scale;
        if (m_covars != nullptr) {
            // Changed: Updated pointer arithmetic
            const T *covars = m_covars + bid * m_N * 6 + gid * 6;
            covar = mat3<T>(
                covars[0],
                covars[1],
                covars[2], // 1st column
                covars[1],
                covars[3],
                covars[4], // 2nd column
                covars[2],
                covars[4],
                covars[5] // 3rd column
            );
        } else {
            // compute from quaternions and scales
            // Changed: Updated pointer arithmetic
            quat = glm::make_vec4(m_quats + bid * m_N * 4 + gid * 4);
            scale = glm::make_vec3(m_scales + bid * m_N * 3 + gid * 3);
            quat_scale_to_covar_preci<T>(quat, scale, &covar, nullptr);
        }
        vec3<T> mean_c;
        pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
        mat3<T> covar_c;
        covar_world_to_cam(R, covar, covar_c);

        // vjp: perspective projection
        T fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
        mat3<T> v_covar_c(0.f);
        vec3<T> v_mean_c(0.f);

        switch (m_camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj_vjp<T>(
                mean_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                m_image_width,
                m_image_height,
                v_covar2d,
                glm::make_vec2(v_means2d),
                v_mean_c,
                v_covar_c
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj_vjp<T>(
                mean_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                m_image_width,
                m_image_height,
                v_covar2d,
                glm::make_vec2(v_means2d),
                v_mean_c,
                v_covar_c
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj_vjp<T>(
                mean_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                m_image_width,
                m_image_height,
                v_covar2d,
                glm::make_vec2(v_means2d),
                v_mean_c,
                v_covar_c
            );
            break;
        }

        // add contribution from v_depths
        v_mean_c.z += v_depths[0];

        // vjp: transform Gaussian covariance to camera space
        vec3<T> v_mean(0.f);
        mat3<T> v_covar(0.f);
        mat3<T> v_R(0.f);
        vec3<T> v_t(0.f);
        pos_world_to_cam_vjp(
            R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean
        );
        covar_world_to_cam_vjp(R, covar, v_covar_c, v_R, v_covar);

        if (m_v_means != nullptr) {
            // Changed: Updated pointer arithmetic
            T *v_means = m_v_means + bid * m_N * 3 + gid * 3;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }

        if (m_v_covars != nullptr) {
            // Changed: Updated pointer arithmetic
            T *v_covars = m_v_covars + bid * m_N * 6 + gid * 6;
            gpuAtomicAdd(v_covars, v_covar[0][0]);
            gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
            gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
            gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
            gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
            gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
        } else {
            // Directly output gradients w.r.t. the quaternion and scale
            mat3<T> rotmat = quat_to_rotmat<T>(quat);
            vec4<T> v_quat(0.f);
            vec3<T> v_scale(0.f);
            quat_scale_to_covar_vjp<T>(
                quat, scale, rotmat, v_covar, v_quat, v_scale
            );
            // Changed: Updated pointer arithmetic
            T *v_quats = m_v_quats + bid * m_N * 4 + gid * 4;
            T *v_scales = m_v_scales + bid * m_N * 3 + gid * 3;
            gpuAtomicAdd(v_quats, v_quat[0]);
            gpuAtomicAdd(v_quats + 1, v_quat[1]);
            gpuAtomicAdd(v_quats + 2, v_quat[2]);
            gpuAtomicAdd(v_quats + 3, v_quat[3]);
            gpuAtomicAdd(v_scales, v_scale[0]);
            gpuAtomicAdd(v_scales + 1, v_scale[1]);
            gpuAtomicAdd(v_scales + 2, v_scale[2]);
        }

        if (m_v_viewmats != nullptr) {
            // Changed: Updated pointer arithmetic
            T *v_viewmats = m_v_viewmats + bid * m_C * 16 + cid * 16;
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

#endif // FullyFusedProjectionBwdKernel_HPP

} // namespace  gsplat::xpu