#pragma once

#include "proj.hpp"
#include "quat_scale_to_covar_preci.hpp"
#include "transform.hpp"
#include "utils.hpp"
#include <sycl/sycl.hpp>

namespace gsplat::xpu {

template <typename T> struct PackedProjectionFwdKernel {
    // Inputs
    const uint32_t m_B;
    const uint32_t m_C;
    const uint32_t m_N;
    const T *m_means;
    const T *m_covars;
    const T *m_quats;
    const T *m_scales;
    const T *m_opacities;
    const T *m_viewmats;
    const T *m_Ks;
    const int32_t m_image_width;
    const int32_t m_image_height;
    const T m_eps2d;
    const T m_near_plane;
    const T m_far_plane;
    const T m_radius_clip;
    const CameraModelType m_camera_model;
    const int64_t *m_block_accum; // Packing helper for the second pass

    // Outputs
    int32_t *m_block_cnts;
    int32_t *m_indptr;
    int64_t *m_batch_ids;
    int64_t *m_camera_ids;
    int64_t *m_gaussian_ids;
    int32_t *m_radii;
    T *m_means2d;
    T *m_depths;
    T *m_conics;
    T *m_compensations;

    PackedProjectionFwdKernel(
        uint32_t B,
        uint32_t C,
        uint32_t N,
        const T *means,
        const T *covars,
        const T *quats,
        const T *scales,
        const T *opacities,
        const T *viewmats,
        const T *Ks,
        int32_t image_width,
        int32_t image_height,
        T eps2d,
        T near_plane,
        T far_plane,
        T radius_clip,
        CameraModelType camera_model,
        const int64_t *block_accum,
        // outputs
        int32_t *block_cnts,
        int32_t *indptr,
        int64_t *batch_ids,
        int64_t *camera_ids,
        int64_t *gaussian_ids,
        int32_t *radii,
        T *means2d,
        T *depths,
        T *conics,
        T *compensations
    )
        : m_B(B), m_C(C), m_N(N), m_means(means), m_covars(covars),
          m_quats(quats), m_scales(scales), m_opacities(opacities),
          m_viewmats(viewmats), m_Ks(Ks), m_image_width(image_width),
          m_image_height(image_height), m_eps2d(eps2d),
          m_near_plane(near_plane), m_far_plane(far_plane),
          m_radius_clip(radius_clip), m_camera_model(camera_model),
          m_block_accum(block_accum), m_block_cnts(block_cnts),
          m_indptr(indptr), m_batch_ids(batch_ids), m_camera_ids(camera_ids),
          m_gaussian_ids(gaussian_ids), m_radii(radii), m_means2d(means2d),
          m_depths(depths), m_conics(conics), m_compensations(compensations) {}

    void operator()(sycl::nd_item<2> item) const {
        auto group = item.get_group();

        sycl::id<2> group_id = item.get_group().get_group_id();
        sycl::range<2> group_range = item.get_group_range();
        sycl::id<2> local_id_2d = item.get_local_id();
        sycl::range<2> local_range = item.get_local_range();

        int32_t blocks_per_row =
            group_range[1]; // Get range of the 2nd dimension

        int32_t row_idx = group_id[0]; // Get group ID of the 1st dimension
        int32_t block_col_idx =
            group_id[1]; // Get group ID of the 2nd dimension
        int32_t block_idx = row_idx * blocks_per_row + block_col_idx;

        int32_t local_id = local_id_2d[1]; // Get local ID of the 2nd dimension
        int32_t col_idx = block_col_idx * local_range[1] + local_id;

        const int32_t bid = row_idx / m_C;
        const int32_t cid = row_idx % m_C;
        const int32_t gid = col_idx;

        bool valid = (bid < m_B) && (cid < m_C) && (gid < m_N);

        // --- Culling logic shared between both passes ---
        vec3<T> mean_c;
        mat3<T> R;
        if (valid) {
            const T *current_means = m_means + bid * m_N * 3 + gid * 3;
            const T *current_viewmats = m_viewmats + bid * m_C * 16 + cid * 16;

            R = mat3<T>(
                current_viewmats[0],
                current_viewmats[4],
                current_viewmats[8],
                current_viewmats[1],
                current_viewmats[5],
                current_viewmats[9],
                current_viewmats[2],
                current_viewmats[6],
                current_viewmats[10]
            );
            vec3<T> t(
                current_viewmats[3], current_viewmats[7], current_viewmats[11]
            );

            pos_world_to_cam(R, t, glm::make_vec3(current_means), mean_c);
            if (mean_c.z < m_near_plane || mean_c.z > m_far_plane) {
                valid = false;
            }
        }

        mat2<T> covar2d;
        vec2<T> mean2d;
        mat2<T> covar2d_inv;
        T compensation;
        if (valid) {
            mat3<T> covar;
            if (m_covars != nullptr) {
                const T *current_covars = m_covars + bid * m_N * 6 + gid * 6;
                covar = mat3<T>(
                    current_covars[0],
                    current_covars[1],
                    current_covars[2],
                    current_covars[1],
                    current_covars[3],
                    current_covars[4],
                    current_covars[2],
                    current_covars[4],
                    current_covars[5]
                );
            } else {
                const T *current_quats = m_quats + bid * m_N * 4 + gid * 4;
                const T *current_scales = m_scales + bid * m_N * 3 + gid * 3;
                quat_scale_to_covar_preci<T>(
                    glm::make_vec4(current_quats),
                    glm::make_vec3(current_scales),
                    &covar,
                    nullptr
                );
            }
            mat3<T> covar_c;
            covar_world_to_cam(R, covar, covar_c);

            const T *current_Ks = m_Ks + bid * m_C * 9 + cid * 9;
            switch (m_camera_model) {
            case CameraModelType::PINHOLE:
                persp_proj<T>(
                    mean_c,
                    covar_c,
                    current_Ks[0],
                    current_Ks[4],
                    current_Ks[2],
                    current_Ks[5],
                    m_image_width,
                    m_image_height,
                    covar2d,
                    mean2d
                );
                break;
            case CameraModelType::ORTHO:
                ortho_proj<T>(
                    mean_c,
                    covar_c,
                    current_Ks[0],
                    current_Ks[4],
                    current_Ks[2],
                    current_Ks[5],
                    m_image_width,
                    m_image_height,
                    covar2d,
                    mean2d
                );
                break;
            case CameraModelType::FISHEYE:
                fisheye_proj<T>(
                    mean_c,
                    covar_c,
                    current_Ks[0],
                    current_Ks[4],
                    current_Ks[2],
                    current_Ks[5],
                    m_image_width,
                    m_image_height,
                    covar2d,
                    mean2d
                );
                break;
            }

            T det = add_blur(m_eps2d, covar2d, compensation);
            if (det <= 0.f) {
                valid = false;
            } else {
                inverse(covar2d, covar2d_inv);
            }
        }

        T radius_x, radius_y;
        if (valid) {
            const T ALPHA_THRESHOLD = 1.f / 255.f;
            T extend = 3.33f;
            if (m_opacities != nullptr) {
                T opacity = m_opacities[bid * m_N + gid];
                if (m_compensations != nullptr) {
                    opacity *= compensation;
                }
                if (opacity < ALPHA_THRESHOLD) {
                    valid = false;
                }
                extend = sycl::fmin(
                    extend,
                    sycl::sqrt(2.0f * sycl::log(opacity / ALPHA_THRESHOLD))
                );
            }

            radius_x = sycl::ceil(extend * sycl::sqrt(covar2d[0][0]));
            radius_y = sycl::ceil(extend * sycl::sqrt(covar2d[1][1]));

            if (radius_x <= m_radius_clip && radius_y <= m_radius_clip) {
                valid = false;
            }

            if (mean2d.x + radius_x <= 0 ||
                mean2d.x - radius_x >= m_image_width ||
                mean2d.y + radius_y <= 0 ||
                mean2d.y - radius_y >= m_image_height) {
                valid = false;
            }
        }

        // --- Pass-specific logic ---

        if (m_block_cnts != nullptr) {
            // First pass: Count visible Gaussians in this block.
            int32_t thread_data = static_cast<int32_t>(valid);
            bool any_valid = sycl::any_of_group(group, valid);
            if (any_valid) {
                // Reduce the count of valid Gaussians across the work-group.
                int32_t aggregate =
                    sycl::reduce_over_group(group, thread_data, sycl::plus<>());
                if (local_id == 0) {
                    m_block_cnts[block_idx] = aggregate;
                }
            } else {
                if (local_id == 0) {
                    m_block_cnts[block_idx] = 0;
                }
            }

        } else {
            // Second pass: Write data for visible Gaussians.
            int64_t thread_data = static_cast<int64_t>(valid);
            bool any_valid = sycl::any_of_group(group, valid);
            if (any_valid) {
                // Perform an exclusive scan to find the local offset for this
                // thread.
                int64_t local_offset = sycl::exclusive_scan_over_group(
                    group, thread_data, sycl::plus<>()
                );

                if (valid) {
                    int64_t global_offset = local_offset;
                    if (block_idx > 0) {
                        global_offset += m_block_accum[block_idx - 1];
                    }

                    // Write to sparse output buffers
                    m_batch_ids[global_offset] = bid;
                    m_camera_ids[global_offset] = cid;
                    m_gaussian_ids[global_offset] = gid;
                    m_radii[global_offset * 2] = (int32_t)radius_x;
                    m_radii[global_offset * 2 + 1] = (int32_t)radius_y;
                    m_means2d[global_offset * 2] = mean2d.x;
                    m_means2d[global_offset * 2 + 1] = mean2d.y;
                    m_depths[global_offset] = mean_c.z;
                    m_conics[global_offset * 3] = covar2d_inv[0][0];
                    m_conics[global_offset * 3 + 1] = covar2d_inv[0][1];
                    m_conics[global_offset * 3 + 2] = covar2d_inv[1][1];
                    if (m_compensations != nullptr) {
                        m_compensations[global_offset] = compensation;
                    }
                }
            }
            // Lane 0 of the first block in each row writes the indptr.
            if (local_id == 0 && block_col_idx == 0) {
                if (row_idx == 0) {
                    m_indptr[0] = 0;
                    // The final count is written by the host after a scan over
                    // block_accum. m_indptr[m_B * m_C] = m_block_accum[m_B *
                    // m_C * blocks_per_row - 1];
                } else {
                    m_indptr[row_idx] = m_block_accum[block_idx - 1];
                }
            }
        }
    }
};

} // namespace gsplat::xpu