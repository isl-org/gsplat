#ifndef RasterizeToPixelsFwdKernel_HPP
#define RasterizeToPixelsFwdKernel_HPP

#include "types.hpp"    
#include "gsplat_sycl_utils.hpp"

namespace gsplat::xpu {
     
template <uint32_t COLOR_DIM, uint32_t CHUNK_SIZE, typename S, bool CONCAT_DATA>
struct RasterizeToPixelsFwdKernel{
     const uint32_t m_C;
     const uint32_t m_N;
     const uint32_t m_n_isects;
     const bool m_packed;
     const uint32_t m_concat_stride;
     const S* m_concatenated_data;
     const sycl::vec<S, 2>* m_means2d; // [C, N, 2] or [nnz, 2] // <<< TYPE CHANGED
     const vec3<S>* m_conics;  // [C, N, 3] or [nnz, 3] // <<< TYPE CHANGED
     const S* m_colors;          // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
     const S* m_opacities;   // [C, N] or [nnz]
     const S* m_backgrounds; // [C, COLOR_DIM]
     const bool* m_masks;     // [C, tile_height, tile_width]
     const uint32_t m_image_width;
     const uint32_t m_image_height;
     const uint32_t m_tile_size;
     const uint32_t m_tile_width;
     const uint32_t m_tile_height;
     const int32_t* m_tile_offsets; // [C, tile_height, tile_width]
     const int32_t* m_flatten_ids;  // [n_isects]
     S* m_render_colors; // [C, image_height, image_width, COLOR_DIM]
     S* m_render_alphas; // [C, image_height, image_width, 1]
     int32_t* m_last_ids; // [C, image_height, image_width]
     sycl::local_accessor<int32_t, 1> m_slm_flatten_ids;
	sycl::local_accessor<sycl::vec<S, 2>, 1> m_slm_means2d;
	sycl::local_accessor<S, 1> m_slm_opacities;
	sycl::local_accessor<sycl::vec<S, 3>, 1> m_slm_conics;
	sycl::local_accessor<BufferType_t<S, COLOR_DIM>, 1> m_slm_colors;

     RasterizeToPixelsFwdKernel(
          const uint32_t C,
          const uint32_t N,
          const uint32_t n_isects,
          const bool packed,
          const uint32_t concat_stride,
          const S* concatenated_data,
          const sycl::vec<S, 2>* means2d, // [C, N, 2] or [nnz, 2] // <<< TYPE CHANGED
          const vec3<S>* conics,  // [C, N, 3] or [nnz, 3] // <<< TYPE CHANGED
          const S* colors,          // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
          const S* opacities,   // [C, N] or [nnz]
          const S* backgrounds, // [C, COLOR_DIM]
          const bool* masks,     // [C, tile_height, tile_width]
          const uint32_t image_width,
          const uint32_t image_height,
          const uint32_t tile_size,
          const uint32_t tile_width,
          const uint32_t tile_height,
          const int32_t* tile_offsets, // [C, tile_height, tile_width]
          const int32_t* flatten_ids,  // [n_isects]
          S* render_colors, // [C, image_height, image_width, COLOR_DIM]
          S* render_alphas, // [C, image_height, image_width, 1]
          int32_t* last_ids, // [C, image_height, image_width]
          sycl::local_accessor<int32_t, 1> slm_flatten_ids,
		sycl::local_accessor<sycl::vec<S, 2>, 1> slm_means2d,
		sycl::local_accessor<S, 1> slm_opacities,
		sycl::local_accessor<sycl::vec<S, 3>, 1> slm_conics,
		sycl::local_accessor<BufferType_t<S, COLOR_DIM>, 1> slm_colors
     )

     : m_C(C), m_N(N), m_n_isects(n_isects), m_packed(packed), 
       m_concat_stride(concat_stride), m_concatenated_data(concatenated_data), m_means2d(means2d),
       m_conics(conics), m_colors(colors), m_opacities(opacities), m_backgrounds(backgrounds),
       m_masks(masks), m_image_width(image_width), m_image_height(image_height),
       m_tile_size(tile_size), m_tile_width(tile_width), m_tile_height(tile_height),
       m_tile_offsets(tile_offsets), m_flatten_ids(flatten_ids), m_render_colors(render_colors),
       m_render_alphas(render_alphas), m_last_ids(last_ids),
       m_slm_flatten_ids(slm_flatten_ids), m_slm_means2d(slm_means2d), m_slm_opacities(slm_opacities),
       m_slm_conics(slm_conics), m_slm_colors(slm_colors)
     {}

     [[intel::reqd_sub_group_size(16)]]
     void operator()(sycl::nd_item<3> work_item) const {

          const uint32_t camera_id = work_item.get_group(0);  // [0, C)
          const uint32_t tile_y    = work_item.get_group(1);  // [0, tile_height)
          const uint32_t tile_x    = work_item.get_group(2);  // [0, tile_width)
          const int32_t tile_id    = tile_y * m_tile_width + tile_x;

          const int32_t* tile_offsets_ptr = m_tile_offsets + camera_id * m_tile_height * m_tile_width;

          const int32_t range_start = tile_offsets_ptr[tile_id];
          int32_t range_end = 0;

          if ((camera_id == m_C - 1) && (tile_id == static_cast<int32_t>(m_tile_width * m_tile_height - 1))) {
               range_end = m_n_isects;
          } else {
               range_end = tile_offsets_ptr[tile_id + 1];
          }

          S* render_colors_ptr = m_render_colors + camera_id * m_image_height * m_image_width * COLOR_DIM;
          S* render_alphas_ptr = m_render_alphas + camera_id * m_image_height * m_image_width;
          int32_t* last_ids_ptr = m_last_ids + camera_id * m_image_height * m_image_width;
  
          BufferType_t<S, COLOR_DIM> backgroundColor{};
          if (m_backgrounds != nullptr) {
               readToBuffer(backgroundColor, m_backgrounds + camera_id * COLOR_DIM);
          }
          const bool* masks_ptr = m_masks;
          if (masks_ptr != nullptr) {
               masks_ptr += camera_id * m_tile_height * m_tile_width;
          }
          
          // Local range is {1, tile_size, tile_size} so that:
          //   local_id(1) in [0, tile_size), local_id(2) in [0, tile_size)
          const uint32_t i = tile_y * m_tile_size + work_item.get_local_id(1);
          const uint32_t j = tile_x * m_tile_size + work_item.get_local_id(2);
          const int32_t pix_id = i * m_image_width + j;
          // Compute pixel center
          bool inside = (i < m_image_height && j < m_image_width);
          bool done = !inside;

          // If a mask exists and the tile is marked false, output background color immediately.
          if (masks_ptr != nullptr && inside && !masks_ptr[tile_id]) {
               for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    render_colors_ptr[pix_id * COLOR_DIM + k] = backgroundColor[k];
               }
               return;
          }

          // Initialize transmittance and pixel accumulator.
          S T = static_cast<S>(1.0);
          
          BufferType_t<S, COLOR_DIM> pix_out{};
          
          int32_t cur_idx = 0;

          int32_t numGaussians = range_end - range_start;
          int32_t batchSize = CHUNK_SIZE;
          int32_t numBatches = (numGaussians + batchSize - 1) / batchSize;

          const size_t localId_y = work_item.get_local_id(1);
          const size_t localId_x = work_item.get_local_id(2);
          const size_t groupWidth = work_item.get_local_range(2);
          const size_t threadRank = localId_y * groupWidth + localId_x; // given that range in 0th dimension is 1

          // Compute pixel coordinates: each work-item covers one pixel inside the tile.
          const S px = static_cast<S>(j) + static_cast<S>(0.5);
          const S py = static_cast<S>(i) + static_cast<S>(0.5);

          for(uint32_t b = 0; b < numBatches; b++){

               work_item.barrier(sycl::access::fence_space::local_space);

               int32_t batchStart = b*batchSize + range_start;
               int32_t idx = batchStart + threadRank;

               if( idx < range_end && threadRank < CHUNK_SIZE) {

                    int32_t g = m_flatten_ids[idx];
                    m_slm_flatten_ids[threadRank] = g;

                    if constexpr( CONCAT_DATA) {
                         const S* data = m_concatenated_data + g*m_concat_stride;
                         if constexpr (COLOR_DIM == 3){
						// means(2) + conics(3) + colors(3) + opac(1)
                              const S* data = m_concatenated_data + g*m_concat_stride;
						auto temp = *(reinterpret_cast<const sycl::vec<S,8>*>(data) );
						
						m_slm_means2d[threadRank] = {temp[0], temp[1]};
						m_slm_conics[threadRank] =  {temp[2], temp[3], temp[4]};
						m_slm_colors[threadRank] =  {temp[5], temp[6], temp[7]};
                              m_slm_opacities[threadRank] = *(data + 2 + 3 + COLOR_DIM);
                         } else {
						if constexpr(BufferType<S, COLOR_DIM>::isVec && COLOR_DIM == 4){
                                   // means(2) + conics(3) + colors(4) + opac(1)
                                   auto temp1 = *(reinterpret_cast<const sycl::vec<S,8>*>(data) );
                                   auto temp2 = *(reinterpret_cast<const sycl::vec<S,2>*>(data + 8) );
							m_slm_means2d[threadRank] = {temp1[0], temp1[1]};
						     m_slm_conics[threadRank] =  {temp1[2], temp1[3], temp1[4]};
						     m_slm_colors[threadRank] =  {temp1[5], temp1[6], temp1[7], temp2[0]};
                                   m_slm_opacities[threadRank] = temp2[1];

						} else {
                                   m_slm_means2d[threadRank] = *(reinterpret_cast<const sycl::vec<S,2>*>(data) );
						     m_slm_conics[threadRank] = *(reinterpret_cast<const sycl::vec<S,3>*>(data+2) );
                                   m_slm_colors[threadRank] = *( reinterpret_cast<const BufferType_t<S, COLOR_DIM>*>(data + 2 + 3) );
                                   m_slm_opacities[threadRank] = *(data + 2 + 3 + COLOR_DIM);
                              }
					}

                    } else {
                         m_slm_means2d[threadRank] = m_means2d[g];
                         m_slm_opacities[threadRank] = m_opacities[g];           
                         m_slm_conics[threadRank] = *(reinterpret_cast<const sycl::vec<S,3>*>(m_conics + g) );
                         if constexpr(BufferType<S, COLOR_DIM>::isVec && COLOR_DIM <= 4){
                              m_slm_colors[threadRank] = *( reinterpret_cast<const BufferType_t<S, COLOR_DIM>*>(m_colors + g * COLOR_DIM) );
                         }
                    }
               }

               work_item.barrier(sycl::access::fence_space::local_space);

               int32_t rangeDiff = range_end - batchStart;
               int32_t endSize = (rangeDiff < batchSize) ? rangeDiff : batchSize;

               for(int i = 0; i < endSize && (!done); i++){

                    int32_t g = m_slm_flatten_ids[i];
                    const sycl::vec<S, 2> xy = m_slm_means2d[i];
                    const S opac             = m_slm_opacities[i];
                    const auto conic = m_slm_conics[i];

                    sycl::vec<S, 2> delta = {xy[0] - px, xy[1] - py};
                    S sigma = static_cast<S>(0.5) *
                              (conic.x() * delta.x() * delta.x() + conic.z() * delta.y() * delta.y()) + 
                              conic.y() * delta.x() * delta.y();


                    S alpha = sycl::min(static_cast<S>(0.999), opac * sycl::exp(-sigma));

                    if (sigma < static_cast<S>(0.0) || alpha < static_cast<S>(1.0 / 255.0))
                         continue;

                    S next_T = T * (static_cast<S>(1.0) - alpha);
                    if (next_T <= static_cast<S>(1e-4)) {
                         done = true;
                         break; 
                    }


                    const S vis = alpha * T;
                    
                    BufferType_t<S, COLOR_DIM> currColor;
                    if constexpr(BufferType<S, COLOR_DIM>::isVec && COLOR_DIM <= 4){
					currColor = m_slm_colors[i];
				} else {
					if constexpr(CONCAT_DATA) {
						readToBuffer(currColor, m_concatenated_data + g*m_concat_stride + 2 + 3);
					} else {
						readToBuffer(currColor, m_colors + g * COLOR_DIM);
					}
				}

                    pix_out += currColor * vis;

                    cur_idx = batchStart + i;
                    T = next_T;
               }
          }

          // Write out results if the pixel is within the image.
          if (inside) {

               render_alphas_ptr[pix_id] = static_cast<S>(1.0) - T;
               last_ids_ptr[pix_id] = cur_idx;

               S* current_pixel_color_ptr_base = render_colors_ptr + pix_id * COLOR_DIM;
               auto* current_pixel_color_ptr = reinterpret_cast<BufferType_t<S, COLOR_DIM>*>(current_pixel_color_ptr_base);

               #pragma unroll
               for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    current_pixel_color_ptr[0][k] = pix_out[k] + T * backgroundColor[k];
               }
          }
     }
};

#endif //RasterizeToPixelsFwdKernel_HPP

} // namespace  gsplat::xpu