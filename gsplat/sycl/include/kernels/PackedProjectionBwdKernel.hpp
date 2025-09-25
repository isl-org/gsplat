#ifndef PackedProjectionBwdKernel_HPP
#define PackedProjectionBwdKernel_HPP

#include <sycl/sycl.hpp>
#include "utils.hpp"
#include "quat.hpp"
#include "quat_scale_to_covar_preci.hpp"
#include "proj.hpp"
#include "transform.hpp"

namespace gsplat::xpu {

template<typename T>
struct PackedProjectionBwdKernel {
    // fwd inputs
    const uint32_t m_B;
    const uint32_t m_C;
    const uint32_t m_N;
    const uint32_t m_nnz;
    const T* m_means;
    const T* m_covars;
    const T* m_quats;
    const T* m_scales;
    const T* m_viewmats;
    const T* m_Ks;
    const int32_t m_image_width;
    const int32_t m_image_height;
    const T m_eps2d;
    const CameraModelType m_camera_model;
    // fwd outputs (packed)
    const int64_t* m_batch_ids;
    const int64_t* m_camera_ids;
    const int64_t* m_gaussian_ids;
    const T* m_conics;
    const T* m_compensations;
    // grad outputs (packed)
    const T* m_v_means2d;
    const T* m_v_depths;
    const T* m_v_conics;
    const T* m_v_compensations;
    const bool m_sparse_grad;
    // grad inputs
    T* m_v_means;
    T* m_v_covars;
    T* m_v_quats;
    T* m_v_scales;
    T* m_v_viewmats;

    PackedProjectionBwdKernel(
        uint32_t B, uint32_t C, uint32_t N, uint32_t nnz,
        const T* means, const T* covars, const T* quats, const T* scales,
        const T* viewmats, const T* Ks,
        int32_t image_width, int32_t image_height, T eps2d, CameraModelType camera_model,
        const int64_t* batch_ids, const int64_t* camera_ids, const int64_t* gaussian_ids,
        const T* conics, const T* compensations,
        const T* v_means2d, const T* v_depths, const T* v_conics, const T* v_compensations,
        bool sparse_grad,
        T* v_means, T* v_covars, T* v_quats, T* v_scales, T* v_viewmats
    ) : m_B(B), m_C(C), m_N(N), m_nnz(nnz), m_means(means), m_covars(covars), m_quats(quats), m_scales(scales),
        m_viewmats(viewmats), m_Ks(Ks), m_image_width(image_width), m_image_height(image_height), m_eps2d(eps2d),
        m_camera_model(camera_model), m_batch_ids(batch_ids), m_camera_ids(camera_ids), m_gaussian_ids(gaussian_ids),
        m_conics(conics), m_compensations(compensations),
        m_v_means2d(v_means2d), m_v_depths(v_depths), m_v_conics(v_conics), m_v_compensations(v_compensations),
        m_sparse_grad(sparse_grad),
        m_v_means(v_means), m_v_covars(v_covars), m_v_quats(v_quats), m_v_scales(v_scales), m_v_viewmats(v_viewmats)
    {}

    void operator()(sycl::nd_item<1> item) const {
        uint32_t idx = item.get_global_id(0);
        if (idx >= m_nnz) {
            return;
        }

        const int64_t bid = m_batch_ids[idx];
        const int64_t cid = m_camera_ids[idx];
        const int64_t gid = m_gaussian_ids[idx];

        // --- VJP Calculation (same as fused, but with packed inputs) ---

        mat2<T> v_covar2d(0.f);
        {
            const T* conics = m_conics + idx * 3;
            const T* v_conics = m_v_conics + idx * 3;
            mat2<T> covar2d_inv = mat2<T>(conics[0], conics[1], conics[1], conics[2]);
            mat2<T> v_covar2d_inv = mat2<T>(v_conics[0], v_conics[1] * 0.5f, v_conics[1] * 0.5f, v_conics[2]);
            inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

            if (m_v_compensations != nullptr) {
                const T compensation = m_compensations[idx];
                const T v_compensation = m_v_compensations[idx];
                add_blur_vjp(m_eps2d, covar2d_inv, compensation, v_compensation, v_covar2d);
            }
        }
        
        const T* means = m_means + bid * m_N * 3 + gid * 3;
        const T* viewmats = m_viewmats + bid * m_C * 16 + cid * 16;
        const T* Ks = m_Ks + bid * m_C * 9 + cid * 9;

        mat3<T> R(
            viewmats[0], viewmats[4], viewmats[8],
            viewmats[1], viewmats[5], viewmats[9],
            viewmats[2], viewmats[6], viewmats[10]
        );
        vec3<T> t(viewmats[3], viewmats[7], viewmats[11]);
        
        mat3<T> covar;
        vec4<T> quat;
        vec3<T> scale;
        if (m_covars != nullptr) {
            const T* covars = m_covars + bid * m_N * 6 + gid * 6;
            covar = mat3<T>(
                covars[0], covars[1], covars[2],
                covars[1], covars[3], covars[4],
                covars[2], covars[4], covars[5]
            );
        } else {
            quat = glm::make_vec4(m_quats + bid * m_N * 4 + gid * 4);
            scale = glm::make_vec3(m_scales + bid * m_N * 3 + gid * 3);
            quat_scale_to_covar_preci<T>(quat, scale, &covar, nullptr);
        }

        vec3<T> mean_c;
        pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
        mat3<T> covar_c;
        covar_world_to_cam(R, covar, covar_c);

        mat3<T> v_covar_c(0.f);
        vec3<T> v_mean_c(0.f);
        const T* v_means2d = m_v_means2d + idx * 2;
        
        switch (m_camera_model) {
            case CameraModelType::PINHOLE:
                persp_proj_vjp<T>(mean_c, covar_c, Ks[0], Ks[4], Ks[2], Ks[5], m_image_width, m_image_height, v_covar2d, glm::make_vec2(v_means2d), v_mean_c, v_covar_c);
                break;
            case CameraModelType::ORTHO:
                ortho_proj_vjp<T>(mean_c, covar_c, Ks[0], Ks[4], Ks[2], Ks[5], m_image_width, m_image_height, v_covar2d, glm::make_vec2(v_means2d), v_mean_c, v_covar_c);
                break;
            case CameraModelType::FISHEYE:
                fisheye_proj_vjp<T>(mean_c, covar_c, Ks[0], Ks[4], Ks[2], Ks[5], m_image_width, m_image_height, v_covar2d, glm::make_vec2(v_means2d), v_mean_c, v_covar_c);
                break;
        }

        v_mean_c.z += m_v_depths[idx];

        vec3<T> v_mean(0.f);
        mat3<T> v_covar(0.f);
        mat3<T> v_R(0.f);
        vec3<T> v_t(0.f);
        pos_world_to_cam_vjp(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
        covar_world_to_cam_vjp(R, covar, v_covar_c, v_R, v_covar);

        // --- Gradient Accumulation ---

        if (m_sparse_grad) {
            // Write gradients to sparse output tensors (no atomics needed)
            if (m_v_means != nullptr) {
                T* v_means_out = m_v_means + idx * 3;
                v_means_out[0] = v_mean.x; v_means_out[1] = v_mean.y; v_means_out[2] = v_mean.z;
            }
            if (m_v_covars != nullptr) {
                T* v_covars_out = m_v_covars + idx * 6;
                v_covars_out[0] = v_covar[0][0];
                v_covars_out[1] = v_covar[0][1] + v_covar[1][0];
                v_covars_out[2] = v_covar[0][2] + v_covar[2][0];
                v_covars_out[3] = v_covar[1][1];
                v_covars_out[4] = v_covar[1][2] + v_covar[2][1];
                v_covars_out[5] = v_covar[2][2];
            } else {
                mat3<T> rotmat = quat_to_rotmat<T>(quat);
                vec4<T> v_quat(0.f);
                vec3<T> v_scale(0.f);
                quat_scale_to_covar_vjp<T>(quat, scale, rotmat, v_covar, v_quat, v_scale);
                T* v_quats_out = m_v_quats + idx * 4;
                T* v_scales_out = m_v_scales + idx * 3;
                v_quats_out[0] = v_quat.x; v_quats_out[1] = v_quat.y; v_quats_out[2] = v_quat.z; v_quats_out[3] = v_quat.w;
                v_scales_out[0] = v_scale.x; v_scales_out[1] = v_scale.y; v_scales_out[2] = v_scale.z;
            }
        } else {
            // Atomically accumulate gradients into dense tensors
            if (m_v_means != nullptr) {
                T* v_means_out = m_v_means + bid * m_N * 3 + gid * 3;
                for (int i = 0; i < 3; ++i) {
                    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(v_means_out[i]);
                    ref.fetch_add(v_mean[i]);
                }
            }
            if (m_v_covars != nullptr) {
                T* v_covars_out = m_v_covars + bid * m_N * 6 + gid * 6;
                sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>(v_covars_out[0]).fetch_add(v_covar[0][0]);
                sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>(v_covars_out[1]).fetch_add(v_covar[0][1] + v_covar[1][0]);
                sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>(v_covars_out[2]).fetch_add(v_covar[0][2] + v_covar[2][0]);
                sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>(v_covars_out[3]).fetch_add(v_covar[1][1]);
                sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>(v_covars_out[4]).fetch_add(v_covar[1][2] + v_covar[2][1]);
                sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>(v_covars_out[5]).fetch_add(v_covar[2][2]);
            } else {
                mat3<T> rotmat = quat_to_rotmat<T>(quat);
                vec4<T> v_quat(0.f);
                vec3<T> v_scale(0.f);
                quat_scale_to_covar_vjp<T>(quat, scale, rotmat, v_covar, v_quat, v_scale);
                T* v_quats_out = m_v_quats + bid * m_N * 4 + gid * 4;
                T* v_scales_out = m_v_scales + bid * m_N * 3 + gid * 3;
                for (int i = 0; i < 4; ++i) sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>(v_quats_out[i]).fetch_add(v_quat[i]);
                for (int i = 0; i < 3; ++i) sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>(v_scales_out[i]).fetch_add(v_scale[i]);
            }
        }

        // v_viewmats is always dense and requires atomics
        if (m_v_viewmats != nullptr) {
            T* v_viewmats_out = m_v_viewmats + bid * m_C * 16 + cid * 16;
            for (uint32_t i = 0; i < 3; i++) { // rows
                for (uint32_t j = 0; j < 3; j++) { // cols
                    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(v_viewmats_out[i * 4 + j]);
                    ref.fetch_add(v_R[j][i]);
                }
                sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(v_viewmats_out[i * 4 + 3]);
                ref.fetch_add(v_t[i]);
            }
        }
    }
};

} // namespace gsplat::xpu

#endif // PackedProjectionBwdKernel_HPP