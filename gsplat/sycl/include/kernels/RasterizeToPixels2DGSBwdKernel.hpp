#pragma once

#include "Sycl_utils.hpp"
#include "types.hpp"
#include <sycl/sycl.hpp>

namespace gsplat::xpu {

// Constants from the CUDA implementation
constexpr float ALPHA_THRESHOLD = 1.0f / 255.0f;
constexpr float FILTER_INV_SQUARE_2DGS = 2.0f;

template <uint32_t COLOR_DIM> struct RasterizeToPixels2DGSBwdKernel {
    // Number of images, gaussians, and intersections
    const uint32_t m_I;
    const uint32_t m_N;
    const uint32_t m_n_isects;
    const bool m_packed;
    const uint32_t m_chunk_size;

    // Forward pass inputs
    const sycl::vec<float, 2> *m_means2d; // Projected Gaussian means
    const float *m_ray_transforms;        // Transformation matrices
    const float *m_colors;                // Gaussian colors
    const float *m_opacities;             // Gaussian opacities
    const float *m_normals;               // Normals in camera space
    const float *m_backgrounds;           // Background colors
    const bool *m_masks;                  // Tile masks

    // Image and tile dimensions
    const uint32_t m_image_width;
    const uint32_t m_image_height;
    const uint32_t m_tile_size;
    const uint32_t m_tile_width;
    const uint32_t m_tile_height;

    // Intersection data
    const int32_t *m_tile_offsets; // Intersection offsets
    const int32_t *m_flatten_ids;  // Global flatten indices

    // Forward pass outputs
    const float *m_render_colors; // Rendered colors
    const float *m_render_alphas; // Alpha values
    const int32_t *m_last_ids;    // Last Gaussian indices
    const int32_t *m_median_ids;  // Median Gaussian indices

    // Gradients from upstream
    const float *m_v_render_colors;  // Gradients of colors
    const float *m_v_render_alphas;  // Gradients of alphas
    const float *m_v_render_normals; // Gradients of normals
    const float *m_v_render_distort; // Gradients of distortion
    const float *m_v_render_median;  // Gradients of median depth

    // Gradient outputs
    sycl::vec<float, 2>
        *m_v_means2d_abs; // Gradients of means2d (absolute, can be null)
    sycl::vec<float, 2> *m_v_means2d; // Gradients of means2d
    float *m_v_ray_transforms;        // Gradients of ray transforms
    float *m_v_colors;                // Gradients of colors
    float *m_v_opacities;             // Gradients of opacities
    float *m_v_normals;               // Gradients of normals
    float *m_v_densify;               // Densification gradients

    // Shared memory
    sycl::local_accessor<int32_t, 1> m_slm_id_batch;
    sycl::local_accessor<sycl::vec<float, 3>, 1> m_slm_xy_opacity;
    sycl::local_accessor<sycl::vec<float, 3>, 1> m_slm_u_Ms;
    sycl::local_accessor<sycl::vec<float, 3>, 1> m_slm_v_Ms;
    sycl::local_accessor<sycl::vec<float, 3>, 1> m_slm_w_Ms;
    sycl::local_accessor<BufferType_t<float, COLOR_DIM>, 1> m_slm_rgbs;
    sycl::local_accessor<sycl::vec<float, 3>, 1> m_slm_normals;

    RasterizeToPixels2DGSBwdKernel(
        const uint32_t I,
        const uint32_t N,
        const uint32_t n_isects,
        const bool packed,
        const uint32_t chunk_size,
        // Forward inputs
        const sycl::vec<float, 2> *means2d,
        const float *ray_transforms,
        const float *colors,
        const float *opacities,
        const float *normals,
        const float *backgrounds,
        const bool *masks,
        // Image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        // Intersections
        const int32_t *tile_offsets,
        const int32_t *flatten_ids,
        // Forward outputs
        const float *render_colors,
        const float *render_alphas,
        const int32_t *last_ids,
        const int32_t *median_ids,
        // Gradient inputs
        const float *v_render_colors,
        const float *v_render_alphas,
        const float *v_render_normals,
        const float *v_render_distort,
        const float *v_render_median,
        // Gradient outputs
        sycl::vec<float, 2> *v_means2d_abs,
        sycl::vec<float, 2> *v_means2d,
        float *v_ray_transforms,
        float *v_colors,
        float *v_opacities,
        float *v_normals,
        float *v_densify,
        // Shared memory
        sycl::local_accessor<int32_t, 1> slm_id_batch,
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_xy_opacity,
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_u_Ms,
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_v_Ms,
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_w_Ms,
        sycl::local_accessor<BufferType_t<float, COLOR_DIM>, 1> slm_rgbs,
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_normals
    )
        : m_I(I), m_N(N), m_n_isects(n_isects), m_packed(packed),
          m_chunk_size(chunk_size), m_means2d(means2d),
          m_ray_transforms(ray_transforms), m_colors(colors),
          m_opacities(opacities), m_normals(normals),
          m_backgrounds(backgrounds), m_masks(masks),
          m_image_width(image_width), m_image_height(image_height),
          m_tile_size(tile_size), m_tile_width(tile_width),
          m_tile_height(tile_height), m_tile_offsets(tile_offsets),
          m_flatten_ids(flatten_ids), m_render_colors(render_colors),
          m_render_alphas(render_alphas), m_last_ids(last_ids),
          m_median_ids(median_ids), m_v_render_colors(v_render_colors),
          m_v_render_alphas(v_render_alphas),
          m_v_render_normals(v_render_normals),
          m_v_render_distort(v_render_distort),
          m_v_render_median(v_render_median), m_v_means2d_abs(v_means2d_abs),
          m_v_means2d(v_means2d), m_v_ray_transforms(v_ray_transforms),
          m_v_colors(v_colors), m_v_opacities(v_opacities),
          m_v_normals(v_normals), m_v_densify(v_densify),
          m_slm_id_batch(slm_id_batch), m_slm_xy_opacity(slm_xy_opacity),
          m_slm_u_Ms(slm_u_Ms), m_slm_v_Ms(slm_v_Ms), m_slm_w_Ms(slm_w_Ms),
          m_slm_rgbs(slm_rgbs), m_slm_normals(slm_normals) {}

    [[intel::reqd_sub_group_size(16)]]
    void operator()(sycl::nd_item<3> item) const {
        // Map thread and block indices
        uint32_t image_id = item.get_group(0); // Block index x -> image_id
        uint32_t tile_y = item.get_group(1);   // Block index y -> tile_y
        uint32_t tile_x = item.get_group(2);   // Block index z -> tile_x
        uint32_t tile_id = tile_y * m_tile_width + tile_x;

        uint32_t i = tile_y * m_tile_size + item.get_local_id(1); // Pixel y
        uint32_t j = tile_x * m_tile_size + item.get_local_id(2); // Pixel x

        // Get pointers to data for current image
        const int32_t *tile_offsets_ptr =
            m_tile_offsets + image_id * m_tile_height * m_tile_width;
        const float *render_alphas_ptr =
            m_render_alphas + image_id * m_image_height * m_image_width;
        const float *render_colors_ptr =
            m_render_colors +
            image_id * m_image_height * m_image_width * COLOR_DIM;

        const int32_t *last_ids_ptr =
            m_last_ids + image_id * m_image_height * m_image_width;
        const int32_t *median_ids_ptr =
            m_median_ids + image_id * m_image_height * m_image_width;

        const float *v_render_colors_ptr =
            m_v_render_colors +
            image_id * m_image_height * m_image_width * COLOR_DIM;
        const float *v_render_alphas_ptr =
            m_v_render_alphas + image_id * m_image_height * m_image_width;
        const float *v_render_normals_ptr =
            m_v_render_normals + image_id * m_image_height * m_image_width * 3;
        const float *v_render_distort_ptr = nullptr;
        if (m_v_render_distort != nullptr) {
            v_render_distort_ptr =
                m_v_render_distort + image_id * m_image_height * m_image_width;
        }
        const float *v_render_median_ptr =
            m_v_render_median + image_id * m_image_height * m_image_width;

        // Background and mask pointers
        const float *backgrounds_ptr = m_backgrounds;
        if (backgrounds_ptr != nullptr) {
            backgrounds_ptr += image_id * COLOR_DIM;
        }

        const bool *masks_ptr = m_masks;
        if (masks_ptr != nullptr) {
            masks_ptr += image_id * m_tile_height * m_tile_width;
        }

        // If tile is masked, do nothing
        if (masks_ptr != nullptr && !masks_ptr[tile_id]) {
            return;
        }

        // Pixel center coordinates
        const float px = static_cast<float>(j) + 0.5f;
        const float py = static_cast<float>(i) + 0.5f;
        const int32_t pix_id = static_cast<int32_t>(sycl::min(
            static_cast<uint32_t>(i * m_image_width + j),
            static_cast<uint32_t>(m_image_width * m_image_height - 1)
        ));

        // Check if pixel is inside image bounds
        bool inside = (i < m_image_height && j < m_image_width);

        // Find range of gaussians for this tile
        int32_t range_start = tile_offsets_ptr[tile_id];
        int32_t range_end;
        if ((image_id == m_I - 1) &&
            (tile_id == static_cast<int32_t>(m_tile_width * m_tile_height - 1)
            )) {
            range_end = m_n_isects;
        } else {
            range_end = tile_offsets_ptr[tile_id + 1];
        }

        // Calculate number of batches needed
        uint32_t num_batches =
            (range_end - range_start + m_chunk_size - 1) / m_chunk_size;

        // Transmittance after last gaussian
        float T_final = 1.0f - render_alphas_ptr[pix_id];
        float T = T_final;

        // Buffers for accumulating contributions
        float buffer[COLOR_DIM] = {0.0f};
        float buffer_normals[3] = {0.0f};

        // Index of last gaussian that contributed to this pixel
        const int32_t bin_final = inside ? last_ids_ptr[pix_id] : 0;

        // Index of gaussian that contributes to median depth
        const int32_t median_idx = inside ? median_ids_ptr[pix_id] : 0;

        // Get thread rank for shared memory access
        uint32_t tr = item.get_local_linear_id();

        // Load gradients for this pixel
        BufferType_t<float, COLOR_DIM> v_render_c{};
        if (inside) {
            if constexpr (BufferType<float, COLOR_DIM>::isVec &&
                          COLOR_DIM <= 4) {
                v_render_c =
                    *reinterpret_cast<const BufferType_t<float, COLOR_DIM> *>(
                        v_render_colors_ptr + pix_id * COLOR_DIM
                    );
            } else {
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_render_c[k] = v_render_colors_ptr[pix_id * COLOR_DIM + k];
                }
            }
        }

        float v_render_a = inside ? v_render_alphas_ptr[pix_id] : 0.0f;

        sycl::vec<float, 3> v_render_n{0.0f, 0.0f, 0.0f};
        if (inside) {
            v_render_n.x() = v_render_normals_ptr[pix_id * 3];
            v_render_n.y() = v_render_normals_ptr[pix_id * 3 + 1];
            v_render_n.z() = v_render_normals_ptr[pix_id * 3 + 2];
        }

        // Prepare for distortion (if needed)
        float v_distort = 0.0f;
        float accum_d = 0.0f, accum_w = 0.0f;
        float accum_d_buffer = 0.0f, accum_w_buffer = 0.0f,
              distort_buffer = 0.0f;
        if (v_render_distort_ptr != nullptr && inside) {
            v_distort = v_render_distort_ptr[pix_id];
            accum_d_buffer =
                render_colors_ptr[pix_id * COLOR_DIM + COLOR_DIM - 1];
            accum_d = accum_d_buffer;
            accum_w_buffer = render_alphas_ptr[pix_id];
            accum_w = accum_w_buffer;
        }

        // Get median depth gradient
        float v_median = inside ? v_render_median_ptr[pix_id] : 0.0f;

        // Find the maximum final gaussian id in the warp
        int32_t warp_bin_final = sycl::reduce_over_group(
            item.get_sub_group(), bin_final, sycl::maximum<int32_t>()
        );

        // Process batches of gaussians in reverse order (back to front)
        for (int32_t b = 0; b < num_batches; ++b) {
            // Synchronize threads before loading next batch
            item.barrier(sycl::access::fence_space::local_space);

            // Compute batch boundaries
            int32_t batch_end = range_end - 1 - m_chunk_size * b;
            int32_t batch_size =
                sycl::min<int32_t>(m_chunk_size, batch_end + 1 - range_start);

            // Load gaussian data into shared memory (in reverse order)
            int32_t idx = batch_end - tr;

            if (idx >= range_start && tr < m_chunk_size) {
                int32_t g = m_flatten_ids[idx];
                m_slm_id_batch[tr] = g;

                // Load position and opacity
                sycl::vec<float, 2> xy = m_means2d[g];
                float opac = m_opacities[g];
                m_slm_xy_opacity[tr] = sycl::vec<float, 3>(xy[0], xy[1], opac);

                // Load ray transform matrix rows
                m_slm_u_Ms[tr] = sycl::vec<float, 3>(
                    m_ray_transforms[g * 9],
                    m_ray_transforms[g * 9 + 1],
                    m_ray_transforms[g * 9 + 2]
                );
                m_slm_v_Ms[tr] = sycl::vec<float, 3>(
                    m_ray_transforms[g * 9 + 3],
                    m_ray_transforms[g * 9 + 4],
                    m_ray_transforms[g * 9 + 5]
                );
                m_slm_w_Ms[tr] = sycl::vec<float, 3>(
                    m_ray_transforms[g * 9 + 6],
                    m_ray_transforms[g * 9 + 7],
                    m_ray_transforms[g * 9 + 8]
                );

                // Load colors
                if constexpr (BufferType<float, COLOR_DIM>::isVec &&
                              COLOR_DIM <= 4) {
                    m_slm_rgbs[tr] = *reinterpret_cast<
                        const BufferType_t<float, COLOR_DIM> *>(
                        m_colors + g * COLOR_DIM
                    );
                } else {
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        m_slm_rgbs[tr][k] = m_colors[g * COLOR_DIM + k];
                    }
                }

                // Load normals
                m_slm_normals[tr] = sycl::vec<float, 3>(
                    m_normals[g * 3], m_normals[g * 3 + 1], m_normals[g * 3 + 2]
                );
            }

            // Wait for all threads to load data
            item.barrier(sycl::access::fence_space::local_space);

            // Process gaussians in batch from back to front
            for (int32_t t = sycl::max(0, batch_end - warp_bin_final);
                 t < batch_size;
                 ++t) {
                bool valid = inside;
                if (batch_end - t > bin_final) {
                    valid = false;
                }

                // Variables for forward pass calculations
                float alpha = 0.0f, opac = 0.0f, vis = 0.0f;
                float gauss_weight_3d = 0.0f, gauss_weight_2d = 0.0f,
                      gauss_weight = 0.0f;
                sycl::vec<float, 2> s{0.0f, 0.0f}, d{0.0f, 0.0f};
                sycl::vec<float, 3> h_u{0.0f, 0.0f, 0.0f},
                    h_v{0.0f, 0.0f, 0.0f};
                sycl::vec<float, 3> ray_cross{0.0f, 0.0f, 0.0f},
                    w_M{0.0f, 0.0f, 0.0f};

                // Perform forward pass calculations for current gaussian
                if (valid) {
                    // Get gaussian parameters from shared memory
                    sycl::vec<float, 3> xy_opac = m_slm_xy_opacity[t];
                    opac = xy_opac[2];

                    sycl::vec<float, 3> u_M = m_slm_u_Ms[t];
                    sycl::vec<float, 3> v_M = m_slm_v_Ms[t];
                    w_M = m_slm_w_Ms[t];

                    // Calculate homogeneous plane parameters
                    h_u = sycl::vec<float, 3>(
                        px * w_M[0] - u_M[0],
                        px * w_M[1] - u_M[1],
                        px * w_M[2] - u_M[2]
                    );

                    h_v = sycl::vec<float, 3>(
                        py * w_M[0] - v_M[0],
                        py * w_M[1] - v_M[1],
                        py * w_M[2] - v_M[2]
                    );

                    // Compute ray intersection using cross product
                    ray_cross = sycl::cross(h_u, h_v);

                    // Check for valid intersection
                    if (ray_cross[2] == 0.0f) {
                        valid = false;
                    } else {
                        // Project to UV space
                        s = sycl::vec<float, 2>(
                            ray_cross[0] / ray_cross[2],
                            ray_cross[1] / ray_cross[2]
                        );

                        // Calculate 3D gaussian weight
                        gauss_weight_3d = s[0] * s[0] + s[1] * s[1];

                        // Calculate 2D projected gaussian weight
                        d = sycl::vec<float, 2>(
                            xy_opac[0] - px, xy_opac[1] - py
                        );
                        gauss_weight_2d = FILTER_INV_SQUARE_2DGS *
                                          (d[0] * d[0] + d[1] * d[1]);

                        // Use minimum of 3D and 2D weights
                        gauss_weight =
                            sycl::min(gauss_weight_3d, gauss_weight_2d);

                        // Calculate sigma and alpha
                        float sigma = 0.5f * gauss_weight;
                        vis = sycl::exp(-sigma);
                        alpha = sycl::min(0.999f, opac * vis);

                        // Skip if gaussian is transparent
                        if (sigma < 0.0f || alpha < ALPHA_THRESHOLD) {
                            valid = false;
                        }
                    }
                }

                // Skip if no thread in the sub-group has a valid gaussian
                bool any_valid =
                    sycl::any_of_group(item.get_sub_group(), valid);
                if (!any_valid) {
                    continue;
                }

                // Initialize gradient variables
                BufferType_t<float, COLOR_DIM> v_rgb_local{};
                sycl::vec<float, 3> v_normal_local{0.0f, 0.0f, 0.0f};
                sycl::vec<float, 3> v_u_M_local{0.0f, 0.0f, 0.0f};
                sycl::vec<float, 3> v_v_M_local{0.0f, 0.0f, 0.0f};
                sycl::vec<float, 3> v_w_M_local{0.0f, 0.0f, 0.0f};
                sycl::vec<float, 2> v_xy_local{0.0f, 0.0f};
                sycl::vec<float, 2> v_xy_abs_local{0.0f, 0.0f};
                float v_opacity_local = 0.0f;

                if (valid) {
                    // Gradient contribution from median depth
                    if (batch_end - t == median_idx) {
                        v_rgb_local[COLOR_DIM - 1] += v_median;
                    }

                    // Compute the current T for this gaussian
                    float ra = 1.0f / (1.0f - alpha);
                    T *= ra;

                    // Weight for the current gaussian
                    float fac = alpha * T;

                    // Update rgb gradients
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        v_rgb_local[k] += fac * v_render_c[k];
                    }

                    // Calculate alpha gradient
                    float v_alpha = 0.0f;
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        v_alpha += (m_slm_rgbs[t][k] * T - buffer[k] * ra) *
                                   v_render_c[k];
                    }

                    // Update normal gradients
                    for (uint32_t k = 0; k < 3; ++k) {
                        v_normal_local[k] = fac * v_render_n[k];
                    }

                    for (uint32_t k = 0; k < 3; ++k) {
                        v_alpha +=
                            (m_slm_normals[t][k] * T - buffer_normals[k] * ra) *
                            v_render_n[k];
                    }

                    // Gradient contribution from alpha
                    v_alpha += T_final * ra * v_render_a;

                    // Adjust alpha gradients by background color
                    if (backgrounds_ptr != nullptr) {
                        float accum = 0.0f;
                        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                            accum += backgrounds_ptr[k] * v_render_c[k];
                        }
                        v_alpha += -T_final * ra * accum;
                    }

                    // Contribution from distortion
                    if (v_render_distort_ptr != nullptr) {
                        float depth = m_slm_rgbs[t][COLOR_DIM - 1];
                        float dl_dw =
                            2.0f *
                            (2.0f * (depth * accum_w_buffer - accum_d_buffer) +
                             (accum_d - depth * accum_w));
                        v_alpha +=
                            (dl_dw * T - distort_buffer * ra) * v_distort;
                        accum_d_buffer -= fac * depth;
                        accum_w_buffer -= fac;
                        distort_buffer += dl_dw * fac;
                        v_rgb_local[COLOR_DIM - 1] +=
                            2.0f * fac * (2.0f - 2.0f * T - accum_w + fac) *
                            v_distort;
                    }

                    // Calculate geometry-related gradients
                    if (opac * vis <= 0.999f) {
                        float v_depth = 0.0f;
                        float v_G = opac * v_alpha;

                        // Case 1: Ray-primitive intersection used in forward
                        // pass
                        if (gauss_weight_3d <= gauss_weight_2d) {
                            sycl::vec<float, 2> v_s(
                                v_G * -vis * s[0] + v_depth * w_M[0],
                                v_G * -vis * s[1] + v_depth * w_M[1]
                            );

                            // Backward through projective transform
                            sycl::vec<float, 3> v_z_w_M(s[0], s[1], 1.0f);
                            float v_sx_pz = v_s[0] / ray_cross[2];
                            float v_sy_pz = v_s[1] / ray_cross[2];
                            sycl::vec<float, 3> v_ray_cross(
                                v_sx_pz,
                                v_sy_pz,
                                -(v_sx_pz * s[0] + v_sy_pz * s[1])
                            );

                            // Calculate cross products for gradient computation
                            sycl::vec<float, 3> v_h_u =
                                sycl::cross(h_v, v_ray_cross);
                            sycl::vec<float, 3> v_h_v =
                                sycl::cross(v_ray_cross, h_u);

                            // Compute gradients for transformation matrices
                            v_u_M_local = sycl::vec<float, 3>(
                                -v_h_u[0], -v_h_u[1], -v_h_u[2]
                            );
                            v_v_M_local = sycl::vec<float, 3>(
                                -v_h_v[0], -v_h_v[1], -v_h_v[2]
                            );
                            v_w_M_local = sycl::vec<float, 3>(
                                px * v_h_u[0] + py * v_h_v[0] +
                                    v_depth * v_z_w_M[0],
                                px * v_h_u[1] + py * v_h_v[1] +
                                    v_depth * v_z_w_M[1],
                                px * v_h_u[2] + py * v_h_v[2] +
                                    v_depth * v_z_w_M[2]
                            );

                            // Case 2: 2D projected gaussian used in forward
                            // pass
                        } else {
                            float v_G_ddelx =
                                -vis * FILTER_INV_SQUARE_2DGS * d[0];
                            float v_G_ddely =
                                -vis * FILTER_INV_SQUARE_2DGS * d[1];
                            v_xy_local = sycl::vec<float, 2>(
                                v_G * v_G_ddelx, v_G * v_G_ddely
                            );

                            if (m_v_means2d_abs != nullptr) {
                                v_xy_abs_local = sycl::vec<float, 2>(
                                    sycl::fabs(v_xy_local[0]),
                                    sycl::fabs(v_xy_local[1])
                                );
                            }
                        }

                        v_opacity_local = vis * v_alpha;
                    }

                    // Update cumulative buffers
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        buffer[k] += m_slm_rgbs[t][k] * fac;
                    }

                    for (uint32_t k = 0; k < 3; ++k) {
                        buffer_normals[k] += m_slm_normals[t][k] * fac;
                    }
                }

                // Sub-group reduction to sum gradients
                auto sub_group = item.get_sub_group();

                // Reduce RGB gradients
                if constexpr (BufferType<float, COLOR_DIM>::isVec &&
                              COLOR_DIM <= 4) {
                    v_rgb_local = sycl::reduce_over_group(
                        sub_group,
                        v_rgb_local,
                        sycl::plus<BufferType_t<float, COLOR_DIM>>()
                    );
                } else {
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        v_rgb_local[k] = sycl::reduce_over_group(
                            sub_group, v_rgb_local[k], sycl::plus<float>()
                        );
                    }
                }

                // Reduce other gradients
                v_normal_local = sycl::reduce_over_group(
                    sub_group, v_normal_local, sycl::plus<sycl::vec<float, 3>>()
                );
                v_u_M_local = sycl::reduce_over_group(
                    sub_group, v_u_M_local, sycl::plus<sycl::vec<float, 3>>()
                );
                v_v_M_local = sycl::reduce_over_group(
                    sub_group, v_v_M_local, sycl::plus<sycl::vec<float, 3>>()
                );
                v_w_M_local = sycl::reduce_over_group(
                    sub_group, v_w_M_local, sycl::plus<sycl::vec<float, 3>>()
                );
                v_xy_local = sycl::reduce_over_group(
                    sub_group, v_xy_local, sycl::plus<sycl::vec<float, 2>>()
                );
                v_opacity_local = sycl::reduce_over_group(
                    sub_group, v_opacity_local, sycl::plus<float>()
                );

                if (m_v_means2d_abs != nullptr) {
                    v_xy_abs_local = sycl::reduce_over_group(
                        sub_group,
                        v_xy_abs_local,
                        sycl::plus<sycl::vec<float, 2>>()
                    );
                }

                // Write gradients to global memory
                int32_t g = m_slm_id_batch[t];

                if (sub_group.get_local_id() == 0) {
                    // Update color gradients
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        gpuAtomicAddGlobal(
                            m_v_colors[g * COLOR_DIM + k], v_rgb_local[k]
                        );
                    }

                    // Update normal gradients
                    for (uint32_t k = 0; k < 3; ++k) {
                        gpuAtomicAddGlobal(
                            m_v_normals[g * 3 + k], v_normal_local[k]
                        );
                    }

                    // Update ray transform gradients
                    gpuAtomicAddGlobal(
                        m_v_ray_transforms[g * 9], v_u_M_local[0]
                    );
                    gpuAtomicAddGlobal(
                        m_v_ray_transforms[g * 9 + 1], v_u_M_local[1]
                    );
                    gpuAtomicAddGlobal(
                        m_v_ray_transforms[g * 9 + 2], v_u_M_local[2]
                    );
                    gpuAtomicAddGlobal(
                        m_v_ray_transforms[g * 9 + 3], v_v_M_local[0]
                    );
                    gpuAtomicAddGlobal(
                        m_v_ray_transforms[g * 9 + 4], v_v_M_local[1]
                    );
                    gpuAtomicAddGlobal(
                        m_v_ray_transforms[g * 9 + 5], v_v_M_local[2]
                    );
                    gpuAtomicAddGlobal(
                        m_v_ray_transforms[g * 9 + 6], v_w_M_local[0]
                    );
                    gpuAtomicAddGlobal(
                        m_v_ray_transforms[g * 9 + 7], v_w_M_local[1]
                    );
                    gpuAtomicAddGlobal(
                        m_v_ray_transforms[g * 9 + 8], v_w_M_local[2]
                    );

                    // Update means2d gradients
                    gpuAtomicAddGlobal(m_v_means2d[g].x(), v_xy_local[0]);
                    gpuAtomicAddGlobal(m_v_means2d[g].y(), v_xy_local[1]);

                    if (m_v_means2d_abs != nullptr) {
                        gpuAtomicAddGlobal(
                            m_v_means2d_abs[g].x(), v_xy_abs_local[0]
                        );
                        gpuAtomicAddGlobal(
                            m_v_means2d_abs[g].y(), v_xy_abs_local[1]
                        );
                    }

                    // Update opacity gradients
                    gpuAtomicAddGlobal(m_v_opacities[g], v_opacity_local);
                }

                if (valid) {
                    float depth = m_slm_w_Ms[t][2];
                    m_v_densify[g * 2] = m_v_ray_transforms[g * 9 + 2] * depth;
                    m_v_densify[g * 2 + 1] =
                        m_v_ray_transforms[g * 9 + 5] * depth;
                }
            }
        }
    }
};

} // namespace gsplat::xpu
