#ifndef RasterizeToPixelsBwdKernel_HPP
#define RasterizeToPixelsBwdKernel_HPP

#include "gsplat_sycl_utils.hpp"
#include "types.hpp"
#include <algorithm>

namespace gsplat::xpu {

template <uint32_t COLOR_DIM, uint32_t CHUNK_SIZE, typename S, bool CONCAT_DATA>
struct RasterizeToPixelsBwdKernel {
    // Inputs (fwd inputs)
    const uint32_t m_C;
    const uint32_t m_N;
    const uint32_t m_n_isects;
    const bool m_packed;
    const uint32_t m_concat_stride;
    const S *m_concatenated_data;
    const sycl::vec<S, 2> *m_means2d; // [C, N, 2] or [nnz, 2]
    const vec3<S> *m_conics;          // [C, N, 3] or [nnz, 3]
    const S *m_colors;                // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *m_opacities;             // [C, N] or [nnz]
    const S *m_backgrounds;           // [C, COLOR_DIM] or [nnz, COLOR_DIM]
    const bool *m_masks;              // [C, tile_height, tile_width]
    const uint32_t m_image_width;
    const uint32_t m_image_height;
    const uint32_t m_tile_size;
    const uint32_t m_tile_width;
    const uint32_t m_tile_height;
    const int32_t *m_tile_offsets; // [C, tile_height, tile_width]
    const int32_t *m_flatten_ids;  // [n_isects]

    // Forward outputs
    const S *m_render_alphas;  // [C, image_height, image_width]
    const int32_t *m_last_ids; // [C, image_height, image_width]

    // Gradients from downstream (grad outputs)
    const S *m_v_render_colors; // [C, image_height, image_width, COLOR_DIM]
    const S *m_v_render_alphas; // [C, image_height, image_width]

    // Gradients to be accumulated (grad inputs)
    sycl::vec<S, 2> *m_v_means2d_abs; // [C, N, 2] or [nnz, 2] (can be nullptr)
    sycl::vec<S, 2> *m_v_means2d;     // [C, N, 2] or [nnz, 2]
    vec3<S> *m_v_conics;              // [C, N, 3] or [nnz, 3]
    S *m_v_colors;                    // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    S *m_v_opacities;                 // [C, N] or [nnz]

    sycl::local_accessor<int32_t, 1> m_slm_flatten_ids;
    sycl::local_accessor<sycl::vec<sycl::half, 2>, 1> m_slm_means2d;
    sycl::local_accessor<sycl::half, 1> m_slm_opacities;
    sycl::local_accessor<sycl::vec<sycl::half, 3>, 1> m_slm_conics;
    sycl::local_accessor<BufferType_t<sycl::half, COLOR_DIM>, 1> m_slm_colors;

    RasterizeToPixelsBwdKernel(
        const uint32_t C,
        const uint32_t N,
        const uint32_t n_isects,
        const bool packed,
        const uint32_t concat_stride,
        const S *concatenated_data,
        const sycl::vec<S, 2> *means2d,
        const vec3<S> *conics,
        const S *colors,
        const S *opacities,
        const S *backgrounds,
        const bool *masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const int32_t *tile_offsets,
        const int32_t *flatten_ids,
        const S *render_alphas,
        const int32_t *last_ids,
        const S *v_render_colors,
        const S *v_render_alphas,
        sycl::vec<S, 2> *v_means2d_abs,
        sycl::vec<S, 2> *v_means2d,
        vec3<S> *v_conics,
        S *v_colors,
        S *v_opacities,
        sycl::local_accessor<int32_t, 1> slm_flatten_ids,
        sycl::local_accessor<sycl::vec<sycl::half, 2>, 1> slm_means2d,
        sycl::local_accessor<sycl::half, 1> slm_opacities,
        sycl::local_accessor<sycl::vec<sycl::half, 3>, 1> slm_conics,
        sycl::local_accessor<BufferType_t<sycl::half, COLOR_DIM>, 1> slm_colors
    )

        : m_C(C), m_N(N), m_n_isects(n_isects), m_packed(packed),
          m_concat_stride(concat_stride),
          m_concatenated_data(concatenated_data), m_means2d(means2d),
          m_conics(conics), m_colors(colors), m_opacities(opacities),
          m_backgrounds(backgrounds), m_masks(masks),
          m_image_width(image_width), m_image_height(image_height),
          m_tile_size(tile_size), m_tile_width(tile_width),
          m_tile_height(tile_height), m_tile_offsets(tile_offsets),
          m_flatten_ids(flatten_ids), m_render_alphas(render_alphas),
          m_last_ids(last_ids), m_v_render_colors(v_render_colors),
          m_v_render_alphas(v_render_alphas), m_v_means2d_abs(v_means2d_abs),
          m_v_means2d(v_means2d), m_v_conics(v_conics), m_v_colors(v_colors),
          m_v_opacities(v_opacities), m_slm_flatten_ids(slm_flatten_ids),
          m_slm_means2d(slm_means2d), m_slm_opacities(slm_opacities),
          m_slm_conics(slm_conics), m_slm_colors(slm_colors) {}

    [[intel::reqd_sub_group_size(16)]]
    void operator()(sycl::nd_item<3> work_item) const {
        // Compute camera and tile indices (each work-group corresponds to a
        // tile)
        const uint32_t camera_id = work_item.get_group(0);
        const uint32_t tile_y = work_item.get_group(1);
        const uint32_t tile_x = work_item.get_group(2);
        const int32_t tile_id = tile_y * m_tile_width + tile_x;

        // Each work-work_item covers one pixel within the tile.
        const uint32_t i = tile_y * m_tile_size + work_item.get_local_id(1);
        const uint32_t j = tile_x * m_tile_size + work_item.get_local_id(2);
        // Clamp pixel index to valid range.
        const int32_t pix_id = sycl::min(
            static_cast<int32_t>(i * m_image_width + j),
            static_cast<int32_t>(m_image_width * m_image_height - 1)
        );

        // Adjust pointers to the current camera.
        const int32_t *tile_offsets_ptr =
            m_tile_offsets + camera_id * m_tile_height * m_tile_width;

        const int32_t range_start = tile_offsets_ptr[tile_id];
        int32_t range_end;
        if ((camera_id == m_C - 1) &&
            (tile_id == static_cast<int32_t>(m_tile_width * m_tile_height - 1)
            )) {
            range_end = m_n_isects;
        } else {
            range_end = tile_offsets_ptr[tile_id + 1];
        }

        const S *render_alphas_ptr =
            m_render_alphas + camera_id * m_image_height * m_image_width;
        const int32_t *last_ids_ptr =
            m_last_ids + camera_id * m_image_height * m_image_width;
        const S *v_render_colors_ptr =
            m_v_render_colors +
            camera_id * m_image_height * m_image_width * COLOR_DIM;
        const S *v_render_alphas_ptr =
            m_v_render_alphas + camera_id * m_image_height * m_image_width;
        const S *backgrounds_ptr = m_backgrounds;
        if (backgrounds_ptr != nullptr) {
            backgrounds_ptr += camera_id * COLOR_DIM;
        }
        const bool *masks_ptr = m_masks;
        if (masks_ptr != nullptr) {
            masks_ptr += camera_id * m_tile_height * m_tile_width;
        }

        // If a mask exists and this tile is not active, do nothing.
        if (masks_ptr != nullptr && !masks_ptr[tile_id]) {
            return;
        }

        // Compute the pixel’s center.
        const S px = static_cast<S>(j) + static_cast<S>(0.5);
        const S py = static_cast<S>(i) + static_cast<S>(0.5);
        const bool inside = (i < m_image_height && j < m_image_width);

        // In the forward pass T_final = 1 - render_alphas.
        const S T_final = static_cast<S>(1.0) - render_alphas_ptr[pix_id];
        S T = T_final;
        // Buffer to accumulate contributions (one per channel).
        BufferType_t<S, COLOR_DIM> buffer{};
        // The index of the last gaussian that contributed (if inside).
        const int32_t bin_final = inside ? last_ids_ptr[pix_id] : 0;

        // Load the pixel’s downstream gradients.
        BufferType_t<S, COLOR_DIM> v_render_c;
        readToBuffer(v_render_c, v_render_colors_ptr + pix_id * COLOR_DIM);

        const S v_render_a = v_render_alphas_ptr[pix_id];

        int32_t numGaussians = range_end - range_start;
        int32_t batchSize = CHUNK_SIZE;
        int32_t numBatches = (numGaussians + batchSize - 1) / batchSize;

        const size_t threadRank = work_item.get_local_linear_id(
        ); // given that range in 0th dimension is 1

        for (int32_t b = numBatches - 1; b >= 0; b--) {

            work_item.barrier(sycl::access::fence_space::local_space);

            int32_t batchStart = b * batchSize + range_start;
            int32_t numel = sycl::min(batchSize, range_end - batchStart);
            int32_t batchEnd = batchStart + numel;

            int32_t loadIdx = batchStart + threadRank;
            int32_t g_thread = -1;
            if (loadIdx < range_end && threadRank < CHUNK_SIZE) {
                int32_t g = m_flatten_ids[loadIdx];
                g_thread = g;
                m_slm_flatten_ids[threadRank] = g;

                if constexpr (CONCAT_DATA) {
                    const S *data = m_concatenated_data + g * m_concat_stride;

                    if constexpr (COLOR_DIM == 3) {
                        auto temp =
                            *(reinterpret_cast<const sycl::vec<S, 8> *>(data));
                        auto temp16 = temp.template convert<
                            sycl::half,
                            sycl::rounding_mode::automatic>();
                        m_slm_means2d[threadRank] = {temp[0], temp[1]};
                        m_slm_conics[threadRank] = {temp[2], temp[3], temp[4]};
                        m_slm_colors[threadRank] = {temp[5], temp[6], temp[7]};
                    } else {
                        auto xy =
                            *(reinterpret_cast<const sycl::vec<S, 2> *>(data));
                        m_slm_means2d[threadRank] = xy.template convert<
                            sycl::half,
                            sycl::rounding_mode::automatic>();

                        auto conic = *(
                            reinterpret_cast<const sycl::vec<S, 3> *>(data + 2)
                        );
                        m_slm_conics[threadRank] = conic.template convert<
                            sycl::half,
                            sycl::rounding_mode::automatic>();

                        if constexpr (BufferType<S, COLOR_DIM>::isVec &&
                                      COLOR_DIM <= 4) {
                            auto color = *(reinterpret_cast<
                                           const BufferType_t<S, COLOR_DIM> *>(
                                data + 2 + 3
                            ));
                            m_slm_colors[threadRank] = color.template convert<
                                sycl::half,
                                sycl::rounding_mode::automatic>();
                            ;
                        }
                    }
                    m_slm_opacities[threadRank] =
                        static_cast<sycl::half>(*(data + 2 + 3 + COLOR_DIM));

                } else {
                    m_slm_means2d[threadRank] =
                        m_means2d[g]
                            .template convert<
                                sycl::half,
                                sycl::rounding_mode::automatic>();

                    m_slm_opacities[threadRank] =
                        static_cast<sycl::half>(m_opacities[g]);
                    auto temp = *(
                        reinterpret_cast<const sycl::vec<S, 3> *>(m_conics + g)
                    );

                    m_slm_conics[threadRank] = temp.template convert<
                        sycl::half,
                        sycl::rounding_mode::automatic>();

                    if constexpr (BufferType<S, COLOR_DIM>::isVec &&
                                  COLOR_DIM <= 4) {
                        auto temp2 = *(reinterpret_cast<
                                       const BufferType_t<S, COLOR_DIM> *>(
                            m_colors + g * COLOR_DIM
                        ));
                        m_slm_colors[threadRank] = temp2.template convert<
                            sycl::half,
                            sycl::rounding_mode::automatic>();
                    }
                }
            }

            work_item.barrier(sycl::access::fence_space::local_space);

            for (int32_t idx = numel - 1; idx >= 0; idx--) {
                // Only process gaussians that actually contributed in the
                // forward pass.

                bool toProcess{true};
                if (idx + batchStart > bin_final)
                    toProcess = false;

                const int32_t g = m_slm_flatten_ids[idx];

                // Load forward parameters.
                sycl::vec<S, 2> xy =
                    m_slm_means2d[idx]
                        .template convert<S, sycl::rounding_mode::automatic>();
                const S opac = static_cast<S>(m_slm_opacities[idx]);
                auto conic = m_slm_conics[idx]
                                 .convert<S, sycl::rounding_mode::automatic>();

                BufferType_t<S, COLOR_DIM> rgb;
                if constexpr (BufferType<S, COLOR_DIM>::isVec &&
                              COLOR_DIM <= 4) {
                    rgb = m_slm_colors[idx]
                              .template convert<
                                  S,
                                  sycl::rounding_mode::automatic>();
                } else {
                    if constexpr (CONCAT_DATA) {
                        readToBuffer(
                            rgb,
                            m_concatenated_data + g * m_concat_stride + 2 + 3
                        );
                    } else {
                        readToBuffer(rgb, m_colors + g * COLOR_DIM);
                    }
                }

                // Compute distance from pixel center.
                sycl::vec<S, 2> delta = {xy.x() - px, xy.y() - py};
                S sigma =
                    static_cast<S>(0.5) * (conic.x() * delta.x() * delta.x() +
                                           conic.z() * delta.y() * delta.y()) +
                    conic.y() * delta.x() * delta.y();
                S vis = sycl::exp(-sigma);
                S alpha = sycl::min(static_cast<S>(0.999), opac * vis);
                if (sigma < static_cast<S>(0.0) ||
                    alpha < static_cast<S>(1.0 / 255.0))
                    toProcess = false;

                BufferType_t<S, COLOR_DIM> v_rgb_local{};
                sycl::vec<S, 3> v_conic_local{};
                sycl::vec<S, 2> v_xy_local{};
                sycl::vec<S, 2> v_xy_abs_local{};
                S v_opacity_local{0.0};

                if (toProcess) {

                    // Compute reciprocal factor and update T.
                    const S ra =
                        static_cast<S>(1.0) / (static_cast<S>(1.0) - alpha);
                    T *= ra;
                    const S fac = alpha * T;

                    // Compute gradient contribution from color.

                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        v_rgb_local[k] = fac * v_render_c[k];
                    }

                    // Compute partial derivative of alpha.
                    S v_alpha = static_cast<S>(0.0);
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        v_alpha +=
                            (rgb[k] * T - buffer[k] * ra) * v_render_c[k];
                    }
                    v_alpha += T_final * ra * v_render_a;
                    if (backgrounds_ptr != nullptr) {
                        S accum = static_cast<S>(0.0);
                        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                            accum += backgrounds_ptr[k] * v_render_c[k];
                        }
                        v_alpha += -T_final * ra * accum;
                    }

                    if (opac * vis <= static_cast<S>(0.999)) {
                        const S v_sigma = -opac * vis * v_alpha;
                        v_conic_local[0] = static_cast<S>(0.5) * v_sigma *
                                           delta.x() * delta.x();
                        v_conic_local[1] = v_sigma * delta.x() * delta.y();
                        v_conic_local[2] = static_cast<S>(0.5) * v_sigma *
                                           delta.y() * delta.y();
                        v_xy_local[0] = v_sigma * (conic.x() * delta.x() +
                                                   conic.y() * delta.y());
                        v_xy_local[1] = v_sigma * (conic.y() * delta.x() +
                                                   conic.z() * delta.y());
                        if (m_v_means2d_abs != nullptr) {
                            v_xy_abs_local[0] = std::abs(v_xy_local[0]);
                            v_xy_abs_local[1] = std::abs(v_xy_local[1]);
                        }
                        v_opacity_local = vis * v_alpha;
                    }

                    // Update the buffer.
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        buffer[k] += rgb[k] * fac;
                    }
                }

                BufferType_t<S, COLOR_DIM> local_color;
                if constexpr (BufferType<S, COLOR_DIM>::isVec) {
                    local_color = sycl::reduce_over_group(
                        work_item.get_group(),
                        v_rgb_local,
                        sycl::plus<BufferType_t<S, COLOR_DIM>>()
                    );
                } else {
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        local_color[k] = sycl::reduce_over_group(
                            work_item.get_group(),
                            v_rgb_local[k],
                            sycl::plus<S>()
                        );
                    }
                }

                S local_opacity = sycl::reduce_over_group(
                    work_item.get_group(), v_opacity_local, sycl::plus<S>()
                );
                auto local_conic = sycl::reduce_over_group(
                    work_item.get_group(),
                    v_conic_local,
                    sycl::plus<sycl::vec<S, 3>>()
                );
                auto local_mean = sycl::reduce_over_group(
                    work_item.get_group(),
                    v_xy_local,
                    sycl::plus<sycl::vec<S, 2>>()
                );

                sycl::vec<S, 2> local_mean_abs;
                if (m_v_means2d_abs != nullptr) {
                    local_mean_abs = sycl::reduce_over_group(
                        work_item.get_group(),
                        v_xy_abs_local,
                        sycl::plus<sycl::vec<S, 2>>()
                    );
                }

                if (threadRank == idx) {
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        gpuAtomicAddGlobal(
                            m_v_colors[g * COLOR_DIM + k], local_color[k]
                        );
                    }
                    gpuAtomicAddGlobal(m_v_opacities[g], local_opacity);
                    gpuAtomicAddGlobal(m_v_conics[g].x, local_conic[0]);
                    gpuAtomicAddGlobal(m_v_conics[g].y, local_conic[1]);
                    gpuAtomicAddGlobal(m_v_conics[g].z, local_conic[2]);
                    gpuAtomicAddGlobal(m_v_means2d[g].x(), local_mean[0]);
                    gpuAtomicAddGlobal(m_v_means2d[g].y(), local_mean[1]);
                    if (m_v_means2d_abs != nullptr) {
                        gpuAtomicAddGlobal(
                            m_v_means2d_abs[g].x(), local_mean_abs[0]
                        );
                        gpuAtomicAddGlobal(
                            m_v_means2d_abs[g].y(), local_mean_abs[1]
                        );
                    }
                }
            }
        }
    }
};

#endif // RasterizeToPixelsBwdKernel_HPP

} // namespace  gsplat::xpu