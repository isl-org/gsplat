#ifndef RASTERIZE_TO_PIXELS_2DGS_FWD_KERNEL_HPP
#define RASTERIZE_TO_PIXELS_2DGS_FWD_KERNEL_HPP

#include <sycl/sycl.hpp>
#include "types.hpp"
#include "gsplat_sycl_utils.hpp"

namespace gsplat::xpu {

// Constants from the CUDA implementation
constexpr float ALPHA_THRESHOLD = 1.0f / 255.0f;
constexpr float FILTER_INV_SQUARE_2DGS = 2.0f;

template <uint32_t COLOR_DIM>
struct RasterizeToPixels2DGSFwdKernel {
    const uint32_t m_I;         // number of images
    const uint32_t m_N;         // number of gaussians
    const uint32_t m_n_isects;  // number of intersections
    const bool m_packed;        // whether tensors are packed
    const uint32_t m_chunk_size; // chunk size for batch processing
    
    const sycl::vec<float, 2>* m_means2d;    // Projected Gaussian means
    const float* m_ray_transforms;           // Transformation matrices
    const float* m_colors;                   // Gaussian colors
    const float* m_opacities;                // Gaussian opacities
    const float* m_normals;                  // Normals in camera space
    const float* m_backgrounds;              // Background colors
    const bool* m_masks;                     // Tile masks
    
    const uint32_t m_image_width;
    const uint32_t m_image_height;
    const uint32_t m_tile_size;
    const uint32_t m_tile_width;
    const uint32_t m_tile_height;
    
    const int32_t* m_tile_offsets;  // Intersection offsets
    const int32_t* m_flatten_ids;   // Global flatten indices
    
    float* m_render_colors;      // Output rendered colors
    float* m_render_alphas;      // Output alpha values
    float* m_render_normals;     // Output rendered normals
    float* m_render_distort;     // Output distortion values
    float* m_render_median;      // Output median depth values
    int32_t* m_last_ids;         // Output indices of last Gaussians
    int32_t* m_median_ids;       // Output indices of median Gaussians
    
    // Shared memory accessors
    sycl::local_accessor<int32_t, 1> m_slm_id_batch;
    sycl::local_accessor<sycl::vec<float, 3>, 1> m_slm_xy_opacity;
    sycl::local_accessor<sycl::vec<float, 3>, 1> m_slm_u_Ms;
    sycl::local_accessor<sycl::vec<float, 3>, 1> m_slm_v_Ms;
    sycl::local_accessor<sycl::vec<float, 3>, 1> m_slm_w_Ms;
    
    RasterizeToPixels2DGSFwdKernel(
        const uint32_t I,
        const uint32_t N,
        const uint32_t n_isects,
        const bool packed,
        const uint32_t chunk_size,
        const sycl::vec<float, 2>* means2d,
        const float* ray_transforms,
        const float* colors,
        const float* opacities,
        const float* normals,
        const float* backgrounds,
        const bool* masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        float* render_colors,
        float* render_alphas,
        float* render_normals,
        float* render_distort,
        float* render_median,
        int32_t* last_ids,
        int32_t* median_ids,
        sycl::local_accessor<int32_t, 1> slm_id_batch,
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_xy_opacity,
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_u_Ms,
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_v_Ms,
        sycl::local_accessor<sycl::vec<float, 3>, 1> slm_w_Ms
    ) : 
        m_I(I), m_N(N), m_n_isects(n_isects), m_packed(packed), m_chunk_size(chunk_size),
        m_means2d(means2d), m_ray_transforms(ray_transforms), 
        m_colors(colors), m_opacities(opacities), m_normals(normals),
        m_backgrounds(backgrounds), m_masks(masks),
        m_image_width(image_width), m_image_height(image_height),
        m_tile_size(tile_size), m_tile_width(tile_width), m_tile_height(tile_height),
        m_tile_offsets(tile_offsets), m_flatten_ids(flatten_ids),
        m_render_colors(render_colors), m_render_alphas(render_alphas),
        m_render_normals(render_normals), m_render_distort(render_distort),
        m_render_median(render_median), m_last_ids(last_ids), m_median_ids(median_ids),
        m_slm_id_batch(slm_id_batch), m_slm_xy_opacity(slm_xy_opacity),
        m_slm_u_Ms(slm_u_Ms), m_slm_v_Ms(slm_v_Ms), m_slm_w_Ms(slm_w_Ms)
    {}

    [[intel::reqd_sub_group_size(16)]]
    void operator()(sycl::nd_item<3> item) const {
        // Map thread and block indices to image, tile, and pixel coordinates
        int32_t image_id = item.get_group(0);  // Block index x -> image_id
        int32_t tile_y = item.get_group(1);    // Block index y -> tile_y
        int32_t tile_x = item.get_group(2);    // Block index z -> tile_x
        int32_t tile_id = tile_y * m_tile_width + tile_x;
        
        uint32_t i = tile_y * m_tile_size + item.get_local_id(1);  // Pixel y
        uint32_t j = tile_x * m_tile_size + item.get_local_id(2);  // Pixel x
        
        // Get pointers to data for current image
        const int32_t* tile_offsets_ptr = m_tile_offsets + image_id * m_tile_height * m_tile_width;
        float* render_colors_ptr = m_render_colors + image_id * m_image_height * m_image_width * COLOR_DIM;
        float* render_alphas_ptr = m_render_alphas + image_id * m_image_height * m_image_width;
        int32_t* last_ids_ptr = m_last_ids + image_id * m_image_height * m_image_width;
        float* render_normals_ptr = m_render_normals + image_id * m_image_height * m_image_width * 3;
        float* render_distort_ptr = m_render_distort + image_id * m_image_height * m_image_width;
        float* render_median_ptr = m_render_median + image_id * m_image_height * m_image_width;
        int32_t* median_ids_ptr = m_median_ids + image_id * m_image_height * m_image_width;
        
        // Background and mask pointers
        const float* backgrounds_ptr = m_backgrounds;
        if (backgrounds_ptr != nullptr) {
            backgrounds_ptr += image_id * COLOR_DIM;
        }
        
        const bool* masks_ptr = m_masks;
        if (masks_ptr != nullptr) {
            masks_ptr += image_id * m_tile_height * m_tile_width;
        }
        
        // Find pixel center
        float px = static_cast<float>(j) + 0.5f;
        float py = static_cast<float>(i) + 0.5f;
        int32_t pix_id = i * m_image_width + j;
        
        // Check if pixel is inside image bounds
        bool inside = (i < m_image_height && j < m_image_width);
        bool done = !inside;
        
        // Handle masked tiles
        if (masks_ptr != nullptr && inside && !masks_ptr[tile_id]) {
            // Render background for masked tiles
            if (inside) {
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    render_colors_ptr[pix_id * COLOR_DIM + k] = 
                        backgrounds_ptr == nullptr ? 0.0f : backgrounds_ptr[k];
                }
            }
            return;
        }
        
        // Get range of gaussians for this tile
        int32_t range_start = tile_offsets_ptr[tile_id];
        int32_t range_end = 
            (image_id == m_I - 1) && (tile_id == static_cast<int32_t>(m_tile_width * m_tile_height - 1))
                ? m_n_isects
                : tile_offsets_ptr[tile_id + 1];
        
        // Calculate number of batches needed
        uint32_t num_batches = (range_end - range_start + m_chunk_size - 1) / m_chunk_size;
        
        // Initialize rendering accumulators
        float T = 1.0f;  // Transmittance
        BufferType_t<float, COLOR_DIM> pix_out{};  // Accumulated color
        float normal_out[3] = {0.0f};  // Accumulated normal
        uint32_t cur_idx = 0;          // Current index
        float distort = 0.0f;          // Distortion
        float accum_vis_depth = 0.0f;  // Accumulated visibility * depth
        float median_depth = 0.0f;     // Median depth
        uint32_t median_idx = 0;       // Median index
        
        // Get thread rank for shared memory access
        uint32_t tr = item.get_local_id(1) * m_tile_size + item.get_local_id(2);
        
        // Process batches of gaussians
        for (uint32_t b = 0; b < num_batches; ++b) {
            // Synchronize threads
            item.barrier(sycl::access::fence_space::local_space);
            
            // Each thread loads one gaussian
            uint32_t batch_start = range_start + m_chunk_size * b;
            uint32_t idx = batch_start + tr;
            
            if (tr < m_chunk_size && idx < range_end) {
                // Get gaussian index
                int32_t g = m_flatten_ids[idx];
                m_slm_id_batch[tr] = g;
                
                // Load gaussian parameters
                sycl::vec<float, 2> xy = m_means2d[g];
                float opac = m_opacities[g];
                m_slm_xy_opacity[tr] = sycl::vec<float, 3>(xy[0], xy[1], opac);
                
                // Load ray transformation matrix rows
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
            }
            
            // Wait for all threads to load data
            item.barrier(sycl::access::fence_space::local_space);
            
            // Manual check for all threads done (instead of CUDA's __syncthreads_count)
            // In SYCL, we have to use barrier synchronization and local variables for this
            
            // Process gaussians in the current batch
            uint32_t batch_size = sycl::min(m_chunk_size, range_end - batch_start);
            for (uint32_t t = 0; t < batch_size && !done; ++t) {
                // Get gaussian parameters from shared memory
                const sycl::vec<float, 3> xy_opac = m_slm_xy_opacity[t];
                const float opac = xy_opac[2];
                
                // Get transformation matrix rows
                const sycl::vec<float, 3> u_M = m_slm_u_Ms[t];
                const sycl::vec<float, 3> v_M = m_slm_v_Ms[t];
                const sycl::vec<float, 3> w_M = m_slm_w_Ms[t];
                
                // Calculate homogeneous plane parameters
                // h_u = px * w_M - u_M
                sycl::vec<float, 3> h_u(
                    px * w_M[0] - u_M[0],
                    px * w_M[1] - u_M[1],
                    px * w_M[2] - u_M[2]
                );
                
                // h_v = py * w_M - v_M
                sycl::vec<float, 3> h_v(
                    py * w_M[0] - v_M[0],
                    py * w_M[1] - v_M[1],
                    py * w_M[2] - v_M[2]
                );
                
                // Compute intersection using cross product
                // ray_cross = h_u × h_v
                sycl::vec<float, 3> ray_cross(
                    h_u[1] * h_v[2] - h_u[2] * h_v[1],
                    h_u[2] * h_v[0] - h_u[0] * h_v[2],
                    h_u[0] * h_v[1] - h_u[1] * h_v[0]
                );
                
                if (ray_cross[2] == 0.0f) {
                    continue;
                }
                
                // Project to UV space
                // s = [ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z]
                sycl::vec<float, 2> s(
                    ray_cross[0] / ray_cross[2],
                    ray_cross[1] / ray_cross[2]
                );
                
                // Calculate gaussian weight in 3D
                // gauss_weight_3d = s.x * s.x + s.y * s.y
                float gauss_weight_3d = s[0] * s[0] + s[1] * s[1];
                
                // Calculate projected gaussian weight in 2D
                // d = [xy_opac.x - px, xy_opac.y - py]
                sycl::vec<float, 2> d(
                    xy_opac[0] - px,
                    xy_opac[1] - py
                );
                // gauss_weight_2d = FILTER_INV_SQUARE_2DGS * (d.x * d.x + d.y * d.y)
                float gauss_weight_2d = FILTER_INV_SQUARE_2DGS * (d[0] * d[0] + d[1] * d[1]);
                
                // Use minimum of 3D and 2D gaussian weights
                // gauss_weight = min(gauss_weight_3d, gauss_weight_2d)
                float gauss_weight = sycl::min(gauss_weight_3d, gauss_weight_2d);
                
                // Calculate sigma and alpha
                float sigma = 0.5f * gauss_weight;
                float alpha = sycl::min(0.999f, opac * sycl::exp(-sigma));
                
                // Skip transparent gaussians
                if (sigma < 0.0f || alpha < ALPHA_THRESHOLD) {
                    continue;
                }
                
                // Calculate next transmittance
                float next_T = T * (1.0f - alpha);
                if (next_T <= 1e-4f) {
                    done = true;
                    break;
                }
                
                // Perform volumetric rendering
                int32_t g = m_slm_id_batch[t];
                float vis = alpha * T;
                
                // Accumulate color
                if constexpr(BufferType<float, COLOR_DIM>::isVec && COLOR_DIM <= 4) {
                    const auto* c_ptr = reinterpret_cast<const BufferType_t<float, COLOR_DIM>*>(m_colors + g * COLOR_DIM);
                    pix_out += (*c_ptr) * vis;
                } else {
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        pix_out[k] += m_colors[g * COLOR_DIM + k] * vis;
                    }
                }
                
                // Accumulate normal
                const float* n_ptr = m_normals + g * 3;
                for (uint32_t k = 0; k < 3; ++k) {
                    normal_out[k] += n_ptr[k] * vis;
                }
                
                // Calculate distortion if needed
                if (m_render_distort != nullptr) {
                    const float depth = m_colors[g * COLOR_DIM + COLOR_DIM - 1];
                    const float distort_bi_0 = vis * depth * (1.0f - T);
                    const float distort_bi_1 = vis * accum_vis_depth;
                    distort += 2.0f * (distort_bi_0 - distort_bi_1);
                    accum_vis_depth += vis * depth;
                }
                
                // Track median depth
                if (T > 0.5f) {
                    median_depth = m_colors[g * COLOR_DIM + COLOR_DIM - 1];
                    median_idx = batch_start + t;
                }
                
                cur_idx = batch_start + t;
                T = next_T;
            }
        }
        
        // Write results if pixel is inside the image
        if (inside) {
            // Store alpha (1 - transmittance)
            render_alphas_ptr[pix_id] = 1.0f - T;
            
            // Store color (accumulated + background * transmittance)
            if (backgrounds_ptr == nullptr) {
                // No background
                if constexpr(BufferType<float, COLOR_DIM>::isVec && COLOR_DIM <= 4) {
                    *reinterpret_cast<BufferType_t<float, COLOR_DIM>*>(render_colors_ptr + pix_id * COLOR_DIM) = pix_out;
                } else {
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        render_colors_ptr[pix_id * COLOR_DIM + k] = pix_out[k];
                    }
                }
            } else {
                // With background
                if constexpr(BufferType<float, COLOR_DIM>::isVec && COLOR_DIM <= 4) {
                    BufferType_t<float, COLOR_DIM> bg;
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        bg[k] = backgrounds_ptr[k];
                    }
                    *reinterpret_cast<BufferType_t<float, COLOR_DIM>*>(render_colors_ptr + pix_id * COLOR_DIM) = 
                        pix_out + bg * T;
                } else {
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        render_colors_ptr[pix_id * COLOR_DIM + k] = pix_out[k] + T * backgrounds_ptr[k];
                    }
                }
            }
            
            // Store normal
            for (uint32_t k = 0; k < 3; ++k) {
                render_normals_ptr[pix_id * 3 + k] = normal_out[k];
            }
            
            // Store last gaussian index
            last_ids_ptr[pix_id] = static_cast<int32_t>(cur_idx);
            
            // Store distortion if needed
            if (m_render_distort != nullptr) {
                render_distort_ptr[pix_id] = distort;
            }
            
            // Store median depth and index
            render_median_ptr[pix_id] = median_depth;
            median_ids_ptr[pix_id] = static_cast<int32_t>(median_idx);
        }
    }
};

} // namespace gsplat::xpu

#endif // RASTERIZE_TO_PIXELS_2DGS_FWD_KERNEL_HPP
