#ifndef FullyFusedProjectionFwdKernel_HPP
#define FullyFusedProjectionFwdKernel_HPP


#include "utils.hpp"
#include "quat_scale_to_covar_preci.hpp"
#include "proj.hpp"
#include "transform.hpp"

namespace gsplat::xpu {

template<typename T>
struct FullyFusedProjectionFwdKernel{
    // New: Added B
    const uint32_t m_B;
    const uint32_t m_C;
    const uint32_t m_N;
    const T* m_means;    // [B, N, 3]
    const T* m_covars;   // [B, N, 6] optional
    const T* m_quats;    // [B, N, 4] optional
    const T* m_scales;   // [B, N, 3] optional
    // New: Added opacities
    const T* m_opacities; // [B, N] optional
    const T* m_viewmats; // [B, C, 4, 4]
    const T* m_Ks;       // [B, C, 3, 3]
    const int32_t m_image_width;
    const int32_t m_image_height;
    const T m_eps2d;
    const T m_near_plane;
    const T m_far_plane;
    const T m_radius_clip;
    const CameraModelType m_camera_model;
    // outputs
    // Changed: radii is now [B, C, N, 2]
    int32_t * m_radii;  // [B, C, N, 2]
    T* m_means2d;      // [B, C, N, 2]
    T* m_depths;       // [B, C, N]
    T* m_conics;       // [B, C, N, 3]
    T* m_compensations; // [B, C, N] optional

    FullyFusedProjectionFwdKernel(
        // New: Added B
        const uint32_t B,
        const uint32_t C,
        const uint32_t N,
        const T* means,
        const T* covars,
        const T* quats,
        const T* scales,
        // New: Added opacities
        const T* opacities,
        const T* viewmats,
        const T* Ks,
        const int32_t image_width,
        const int32_t image_height,
        const T eps2d,
        const T near_plane,
        const T far_plane,
        const T radius_clip,
        const CameraModelType camera_model,
        int32_t * radii,
        T* means2d,
        T* depths,
        T* conics,
        T* compensations
    )
    // New: Added m_B and m_opacities
    : m_B(B), m_C(C), m_N(N), m_means(means), m_covars(covars), m_quats(quats), m_scales(scales),
      m_opacities(opacities), m_viewmats(viewmats), m_Ks(Ks), m_image_width(image_width), m_image_height(image_height),
      m_eps2d(eps2d), m_near_plane(near_plane), m_far_plane(far_plane), m_radius_clip(radius_clip),
      m_camera_model(camera_model), m_radii(radii), m_means2d(means2d), m_depths(depths),
      m_conics(conics), m_compensations(compensations)
    {}

    void operator()(sycl::nd_item<1> work_item) const
    {
        uint32_t idx = work_item.get_global_id(0);
        // Changed: Updated upper bound to include B
        if (idx >= m_B * m_C * m_N) {
            return;
        }
        // Changed: Added bid and updated cid, gid calculation
        const uint32_t bid = idx / (m_C * m_N); // batch id
        const uint32_t cid = (idx / m_N) % m_C; // camera id
        const uint32_t gid = idx % m_N; // gaussian id

        // Changed: Updated pointer arithmetic to include B
        const T* means = m_means + bid * m_N * 3 + gid * 3;
        const T* viewmats = m_viewmats + bid * m_C * 16 + cid * 16;
        const T* Ks = m_Ks + bid * m_C * 9 + cid * 9;

        // glm is column-major but input is row-major
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

        // transform Gaussian center to camera space
        vec3<T> mean_c;
        pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
        if (mean_c.z < m_near_plane || mean_c.z > m_far_plane) {
            // Changed: Set both radii to 0
            m_radii[idx * 2] = 0;
            m_radii[idx * 2 + 1] = 0;
            return;
        }

        // transform Gaussian covariance to camera space
        mat3<T> covar;
        if (m_covars != nullptr) {
            // Changed: Updated pointer arithmetic
            const T* covars = m_covars + bid * m_N * 6 + gid * 6;
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
            const T* quats = m_quats + bid * m_N * 4 + gid * 4;
            const T* scales = m_scales + bid * m_N * 3 + gid * 3;
            quat_scale_to_covar_preci<T>(
                glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
            );
        }
        mat3<T> covar_c;
        covar_world_to_cam(R, covar, covar_c);

        // perspective projection
        mat2<T> covar2d;
        vec2<T> mean2d;

        switch (m_camera_model) {
            case CameraModelType::PINHOLE: // perspective projection
                persp_proj<T>(
                    mean_c,
                    covar_c,
                    Ks[0],
                    Ks[4],
                    Ks[2],
                    Ks[5],
                    m_image_width,
                    m_image_height,
                    covar2d,
                    mean2d
                );
                break;
            case CameraModelType::ORTHO: // orthographic projection
                ortho_proj<T>(
                    mean_c,
                    covar_c,
                    Ks[0],
                    Ks[4],
                    Ks[2],
                    Ks[5],
                    m_image_width,
                    m_image_height,
                    covar2d,
                    mean2d
                );
                break;
            case CameraModelType::FISHEYE: // fisheye projection
                fisheye_proj<T>(
                    mean_c,
                    covar_c,
                    Ks[0],
                    Ks[4],
                    Ks[2],
                    Ks[5],
                    m_image_width,
                    m_image_height,
                    covar2d,
                    mean2d
                );
                break;
        }

        T compensation;
        T det = add_blur(m_eps2d, covar2d, compensation);
        if (det <= 0.f) {
            // Changed: Set both radii to 0
            m_radii[idx * 2] = 0;
            m_radii[idx * 2 + 1] = 0;
            return;
        }

        // compute the inverse of the 2d covariance
        mat2<T> covar2d_inv;
        inverse(covar2d, covar2d_inv);

        // New: Opacity-aware bounding box and radius calculation
        const T ALPHA_THRESHOLD = 1.f / 255.f;
        T extend = 3.33f;
        if (m_opacities != nullptr) {
            T opacity = m_opacities[bid * m_N + gid];
            if (m_compensations != nullptr) {
                opacity *= compensation;
            }
            if (opacity < ALPHA_THRESHOLD) {
                m_radii[idx * 2] = 0;
                m_radii[idx * 2 + 1] = 0;
                return;
            }
            extend = sycl::min(extend, sycl::sqrt(2.0f * sycl::log(opacity / ALPHA_THRESHOLD)));
        }

        T radius_x = sycl::ceil(extend * sycl::sqrt(covar2d[0][0]));
        T radius_y = sycl::ceil(extend * sycl::sqrt(covar2d[1][1]));

        if (radius_x <= m_radius_clip && radius_y <= m_radius_clip) {
            m_radii[idx * 2] = 0;
            m_radii[idx * 2 + 1] = 0;
            return;
        }

        // mask out gaussians outside the image region
        if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= m_image_width ||
            mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= m_image_height) {
            m_radii[idx * 2] = 0;
            m_radii[idx * 2 + 1] = 0;
            return;
        }

        // write to outputs
        // Changed: Write radius_x and radius_y
        m_radii[idx * 2] = (int32_t)radius_x;
        m_radii[idx * 2 + 1] = (int32_t)radius_y;
        m_means2d[idx * 2] = mean2d.x;
        m_means2d[idx * 2 + 1] = mean2d.y;
        m_depths[idx] = mean_c.z;
        m_conics[idx * 3] = covar2d_inv[0][0];
        m_conics[idx * 3 + 1] = covar2d_inv[0][1];
        m_conics[idx * 3 + 2] = covar2d_inv[1][1];
        if (m_compensations != nullptr) {
            m_compensations[idx] = compensation;
        }

    }

};
#endif //FullyFusedProjectionFwdKernel_HPP

} // namespace  gsplat::xpu