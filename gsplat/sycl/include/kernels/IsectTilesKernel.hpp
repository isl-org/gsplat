#ifndef IsectTilesKernel_HPP
#define IsectTilesKernel_HPP

 
#include "types.hpp"
#include "transform.hpp"
#include "utils.hpp"
#include <algorithm>

namespace gsplat::xpu {

struct uint2 {
    uint32_t x;
    uint32_t y;
};

template<typename T>
struct IsectTilesKernel {
    const bool m_packed;
    const uint32_t m_C;
    const uint32_t m_N;
    const uint32_t m_nnz;
    const int64_t* m_camera_ids;   // [nnz] optional
    const int64_t* m_gaussian_ids; // [nnz] optional
    const T* m_means2d;                   // [C, N, 2] or [nnz, 2]
    const int32_t* m_radii;               // [C, N] or [nnz]
    const T* m_depths;                    // [C, N] or [nnz]
    const int64_t* m_cum_tiles_per_gauss; // [C, N] or [nnz]
    const uint32_t m_tile_size;
    const uint32_t m_tile_width;
    const uint32_t m_tile_height;
    const uint32_t m_tile_n_bits;
    int32_t* m_tiles_per_gauss; // [C, N] or [nnz]
    int64_t* m_isect_ids;       // [n_isects]
    int32_t* m_flatten_ids;      // [n_isects]

    IsectTilesKernel(
        const bool packed,
        const uint32_t C,
        const uint32_t N,
        const uint32_t nnz,
        const int64_t* camera_ids,
        const int64_t* gaussian_ids,
        const T* means2d,
        const int32_t* radii,
        const T* depths,
        const int64_t* cum_tiles_per_gauss,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const uint32_t tile_n_bits,
        int32_t* tiles_per_gauss,
        int64_t* isect_ids,
        int32_t* flatten_ids
    ) : 
        m_packed(packed),
        m_C(C),
        m_N(N),
        m_nnz(nnz),
        m_camera_ids(camera_ids),
        m_gaussian_ids(gaussian_ids),
        m_means2d(means2d),
        m_radii(radii),
        m_depths(depths),
        m_cum_tiles_per_gauss(cum_tiles_per_gauss),
        m_tile_size(tile_size),
        m_tile_width(tile_width),
        m_tile_height(tile_height),
        m_tile_n_bits(tile_n_bits),
        m_tiles_per_gauss(tiles_per_gauss),
        m_isect_ids(isect_ids),
        m_flatten_ids(flatten_ids)
    {}

    void operator()(sycl::nd_item<1> work_item)  const {
        uint32_t idx = work_item.get_global_id(0);

        bool first_pass = m_cum_tiles_per_gauss == nullptr;
        if (idx >= (m_packed ? m_nnz : m_C * m_N)) {
            return;
        }

        const T radius = m_radii[idx];
        if (radius <= 0) {
            if (first_pass) {
                m_tiles_per_gauss[idx] = 0;
            }
            return;
        }

        vec2<T> mean2d = glm::make_vec2(m_means2d + 2 * idx);

        T tile_radius = radius / static_cast<T>(m_tile_size);
        T tile_x = mean2d.x / static_cast<T>(m_tile_size);
        T tile_y = mean2d.y / static_cast<T>(m_tile_size);

        uint2 tile_min, tile_max;
        tile_min.x = sycl::min( sycl::max((uint32_t)0, (uint32_t)sycl::floor(tile_x - tile_radius)), m_tile_width);
        tile_min.y = sycl::min( sycl::max((uint32_t)0, (uint32_t)sycl::floor(tile_y - tile_radius)), m_tile_height);

        tile_max.x = sycl::min( sycl::max((uint32_t)0, (uint32_t)sycl::ceil(tile_x + tile_radius)), m_tile_width);
        tile_max.y = sycl::min( sycl::max((uint32_t)0, (uint32_t)sycl::ceil(tile_y + tile_radius)), m_tile_height);

        if (first_pass) {
            // first pass only writes out tiles_per_gauss
            m_tiles_per_gauss[idx] = static_cast<int32_t>(
                (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
            );
            return;
        }

        int64_t cid; // camera id
        if (m_packed) {
            // parallelize over nnz
            cid = m_camera_ids[idx];
            // gid = gaussian_ids[idx];
        } else {
            // parallelize over C * N
            cid = idx / m_N;
            // gid = idx % N;
        }

        const int64_t cid_enc = cid << (32 + m_tile_n_bits);

        int64_t depth_id_enc = (int64_t) * (int32_t *)&(m_depths[idx]);
        int64_t cur_idx = (idx == 0) ? 0 : m_cum_tiles_per_gauss[idx - 1];
        for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
            for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
                int64_t tile_id = i * m_tile_width + j;
                // e.g. tile_n_bits = 22:
                // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
                m_isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
                // the flatten index in [C * N] or [nnz]
                m_flatten_ids[cur_idx] = static_cast<int32_t>(idx);
                ++cur_idx;
            }
        }
    }
};

#endif //IsectTilesKernel_HPP

} // namespace  gsplat::xpu