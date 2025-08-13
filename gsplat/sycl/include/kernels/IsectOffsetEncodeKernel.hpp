#ifndef IsectOffsetEncodeKernel_HPP
#define IsectOffsetEncodeKernel_HPP

namespace gsplat::xpu {

struct IsectOffsetEncodeKernel {

    const uint32_t m_n_isects;
    const int64_t* m_isect_ids;
    const uint32_t m_C;
    const uint32_t m_n_tiles;
    const uint32_t m_tile_n_bits;
    int32_t* m_offsets; //[C, n_tiles]

    IsectOffsetEncodeKernel(
        const uint32_t n_isects,
        const int64_t* isect_ids,
        const uint32_t C,
        const uint32_t n_tiles,
        const uint32_t tile_n_bits,
        int32_t* offsets
    ) : 
    m_n_isects(n_isects),
    m_isect_ids(isect_ids),
    m_C(C),
    m_n_tiles(n_tiles),
    m_tile_n_bits(tile_n_bits),
    m_offsets(offsets)
    {}

    void operator()(sycl::nd_item<1> work_item)  const {
        uint32_t idx = work_item.get_global_id(0);

        if (idx >= m_n_isects)
            return;

        int64_t isect_id_curr = m_isect_ids[idx] >> 32;
        int64_t cid_curr = isect_id_curr >> m_tile_n_bits;
        int64_t tid_curr = isect_id_curr & ((1 << m_tile_n_bits) - 1);
        int64_t id_curr = cid_curr * m_n_tiles + tid_curr;

        if (idx == 0) {
            // write out the offsets until the first valid tile (inclusive)
            for (uint32_t i = 0; i < id_curr + 1; ++i)
                m_offsets[i] = static_cast<int32_t>(idx);
        }
        if (idx == m_n_isects - 1) {
            // write out the rest of the offsets
            for (uint32_t i = id_curr + 1; i < m_C * m_n_tiles; ++i)
                m_offsets[i] = static_cast<int32_t>(m_n_isects);
        }

        if (idx > 0) {
            // visit the current and previous isect_id and check if the (cid,
            // tile_id) pair changes.
            int64_t isect_id_prev = m_isect_ids[idx - 1] >> 32; // shift out the depth
            if (isect_id_prev == isect_id_curr)
                return;
    
            // write out the offsets between the previous and current tiles
            int64_t cid_prev = isect_id_prev >> m_tile_n_bits;
            int64_t tid_prev = isect_id_prev & ((1 << m_tile_n_bits) - 1);
            int64_t id_prev = cid_prev * m_n_tiles + tid_prev;
            for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
                m_offsets[i] = static_cast<int32_t>(idx);
        }
    }
};

#endif 

} // namespace  gsplat::xpu