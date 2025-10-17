#ifndef Projection2DGSFusedBwdKernel_HPP
#define Projection2DGSFusedBwdKernel_HPP

#include "utils.hpp"
#include "quat_scale_to_covar_preci.hpp"
#include "transform.hpp"

namespace gsplat::xpu {

template<typename T>
inline T sum(vec3<T> a) { return a.x + a.y + a.z; }

template<typename T>
inline void compute_ray_transforms_aabb_vjp(
    const T *ray_transforms,
    const T *v_means2d,
    const vec3<T> v_normals,
    const mat3<T> W,
    const mat3<T> P,
    const vec3<T> cam_pos,
    const vec3<T> mean_w,
    const vec3<T> mean_c,
    const vec4<T> quat,
    const vec2<T> scale,
    mat3<T> &_v_ray_transforms,
    vec4<T> &v_quat,
    vec2<T> &v_scale,
    vec3<T> &v_mean,
    mat3<T> &v_R,
    vec3<T> &v_t
) {
    if (v_means2d[0] != 0 || v_means2d[1] != 0) {
        const T distance = ray_transforms[6] * ray_transforms[6] +
                               ray_transforms[7] * ray_transforms[7] -
                               ray_transforms[8] * ray_transforms[8];
        const T f = T(1) / (distance);
        const T dpx_dT00 = f * ray_transforms[6];
        const T dpx_dT01 = f * ray_transforms[7];
        const T dpx_dT02 = -f * ray_transforms[8];
        const T dpy_dT10 = f * ray_transforms[6];
        const T dpy_dT11 = f * ray_transforms[7];
        const T dpy_dT12 = -f * ray_transforms[8];
        const T dpx_dd = -f * f * (ray_transforms[0] * ray_transforms[6] + ray_transforms[1] * ray_transforms[7] - ray_transforms[2] * ray_transforms[8]);
        const T dpx_dT30 = ray_transforms[0] * f + T(2) * dpx_dd * ray_transforms[6];
        const T dpx_dT31 = ray_transforms[1] * f + T(2) * dpx_dd * ray_transforms[7];
        const T dpx_dT32 = -ray_transforms[2] * f - T(2) * dpx_dd * ray_transforms[8];
        const T dpy_dd = -f * f * (ray_transforms[3] * ray_transforms[6] + ray_transforms[4] * ray_transforms[7] - ray_transforms[5] * ray_transforms[8]);
        const T dpy_dT30 = ray_transforms[3] * f + T(2) * dpy_dd * ray_transforms[6];
        const T dpy_dT31 = ray_transforms[4] * f + T(2) * dpy_dd * ray_transforms[7];
        const T dpy_dT32 = -ray_transforms[5] * f - T(2) * dpy_dd * ray_transforms[8];

        _v_ray_transforms[0][0] += v_means2d[0] * dpx_dT00;
        _v_ray_transforms[0][1] += v_means2d[0] * dpx_dT01;
        _v_ray_transforms[0][2] += v_means2d[0] * dpx_dT02;
        _v_ray_transforms[1][0] += v_means2d[1] * dpy_dT10;
        _v_ray_transforms[1][1] += v_means2d[1] * dpy_dT11;
        _v_ray_transforms[1][2] += v_means2d[1] * dpy_dT12;
        _v_ray_transforms[2][0] +=
            v_means2d[0] * dpx_dT30 + v_means2d[1] * dpy_dT30;
        _v_ray_transforms[2][1] +=
            v_means2d[0] * dpx_dT31 + v_means2d[1] * dpy_dT31;
        _v_ray_transforms[2][2] +=
            v_means2d[0] * dpx_dT32 + v_means2d[1] * dpy_dT32;
    }

    mat3<T> R = quat_to_rotmat(quat);
    mat3<T> v_M = P * glm::transpose(_v_ray_transforms);
    mat3<T> W_t = glm::transpose(W);
    mat3<T> v_RS = W_t * v_M;
    vec3<T> v_tn = W_t * v_normals;

    // dual visible
    vec3<T> tn = W * R[2];
    T cos = glm::dot(-tn, mean_c);
    T multiplier = cos > T(0) ? T(1) : T(-1);
    v_tn *= multiplier;

    mat3<T> v_Rot = mat3<T>(v_RS[0] * scale[0], v_RS[1] * scale[1], v_tn);

    quat_to_rotmat_vjp(quat, v_Rot, v_quat);
    v_scale[0] += glm::dot(v_RS[0], R[0]);
    v_scale[1] += glm::dot(v_RS[1], R[1]);

    v_mean += v_RS[2];

    v_R += glm::outerProduct(v_M[2], mean_w);

    mat3<T> RS = quat_to_rotmat(quat) * 
        mat3<T>(scale[0], T(0.0), T(0.0), T(0.0), scale[1], T(0.0), T(0.0), T(0.0), T(1.0));
    mat3<T> v_RS_cam = mat3<T>(v_M[0], v_M[1], v_normals * multiplier);
    
    v_R += v_RS_cam * glm::transpose(RS);
    v_t += v_M[2];
}

template<typename T>
struct Projection2DGSFusedBwdKernel {
    // fwd inputs
    const uint32_t m_B;
    const uint32_t m_C;
    const uint32_t m_N;
    const T* m_means;          // [B, N, 3]
    const T* m_quats;          // [B, N, 4]
    const T* m_scales;         // [B, N, 3]
    const T* m_viewmats;       // [B, C, 4, 4]
    const T* m_Ks;             // [B, C, 3, 3]
    const uint32_t m_image_width;
    const uint32_t m_image_height;
    // fwd outputs
    const int32_t* m_radii;          // [B, C, N, 2]
    const T* m_ray_transforms;       // [B, C, N, 3, 3]
    // grad outputs
    const T* m_v_means2d;            // [B, C, N, 2]
    const T* m_v_depths;             // [B, C, N]
    const T* m_v_normals;            // [B, C, N, 3]
    const T* m_v_ray_transforms;     // [B, C, N, 3, 3]
    // grad inputs
    T* m_v_means;                    // [B, N, 3]
    T* m_v_quats;                    // [B, N, 4]
    T* m_v_scales;                   // [B, N, 3]
    T* m_v_viewmats;                 // [B, C, 4, 4]

    Projection2DGSFusedBwdKernel(
        const uint32_t B,
        const uint32_t C,
        const uint32_t N,
        const T* means,
        const T* quats,
        const T* scales,
        const T* viewmats,
        const T* Ks,
        const uint32_t image_width,
        const uint32_t image_height,
        const int32_t* radii,
        const T* ray_transforms,
        const T* v_means2d,
        const T* v_depths,
        const T* v_normals,
        const T* v_ray_transforms,
        T* v_means,
        T* v_quats,
        T* v_scales,
        T* v_viewmats
    )
    : m_B(B), m_C(C), m_N(N), m_means(means), m_quats(quats), m_scales(scales),
      m_viewmats(viewmats), m_Ks(Ks), m_image_width(image_width), m_image_height(image_height),
      m_radii(radii), m_ray_transforms(ray_transforms),
      m_v_means2d(v_means2d), m_v_depths(v_depths), m_v_normals(v_normals),
      m_v_ray_transforms(v_ray_transforms),
      m_v_means(v_means), m_v_quats(v_quats), m_v_scales(v_scales), m_v_viewmats(v_viewmats)
    {}

    void operator()(sycl::nd_item<1> work_item) const
    {
        uint32_t idx = work_item.get_global_id(0);
        
        if (idx >= m_B * m_C * m_N) {
            return;
        }

        // Check if radii are valid
        if (m_radii[idx * 2] <= 0 || m_radii[idx * 2 + 1] <= 0) {
            return;
        }
        
        const uint32_t bid = idx / (m_C * m_N); // batch id
        const uint32_t cid = (idx / m_N) % m_C; // camera id
        const uint32_t gid = idx % m_N;         // gaussian id

        // Shift pointers to current camera and gaussian
        const T* means = m_means + bid * m_N * 3 + gid * 3;
        const T* viewmats = m_viewmats + bid * m_C * 16 + cid * 16;
        const T* Ks = m_Ks + bid * m_C * 9 + cid * 9;

        const T* ray_transforms = m_ray_transforms + idx * 9;

        const T* v_means2d = m_v_means2d + idx * 2;
        const T* v_depths = m_v_depths + idx;
        const T* v_normals = m_v_normals + idx * 3;
        const T* v_ray_transforms = m_v_ray_transforms + idx * 9;

        // Transform Gaussian to camera space
        mat3<T> R = mat3<T>(
            viewmats[0], viewmats[4], viewmats[8],   // 1st column
            viewmats[1], viewmats[5], viewmats[9],   // 2nd column
            viewmats[2], viewmats[6], viewmats[10]   // 3rd column
        );
        vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);
        
        vec3<T> mean_w = vec3<T>(means[0], means[1], means[2]);
        vec3<T> mean_c;
        pos_world_to_cam(R, t, mean_w, mean_c);

        const T* quats_ptr = m_quats + bid * m_N * 4 + gid * 4;
        const T* scales_ptr = m_scales + bid * m_N * 3 + gid * 3;
        
        vec4<T> quat = vec4<T>(quats_ptr[0], quats_ptr[1], quats_ptr[2], quats_ptr[3]);
        vec2<T> scale = vec2<T>(scales_ptr[0], scales_ptr[1]);

        mat3<T> P = mat3<T>(
            Ks[0], T(0.0), Ks[2],
            T(0.0), Ks[4], Ks[5],
            T(0.0), T(0.0), T(1.0)
        );

        mat3<T> _v_ray_transforms = mat3<T>(
            v_ray_transforms[0], v_ray_transforms[1], v_ray_transforms[2],
            v_ray_transforms[3], v_ray_transforms[4], v_ray_transforms[5],
            v_ray_transforms[6], v_ray_transforms[7], v_ray_transforms[8]
        );

        // Add depth gradient to the last element
        _v_ray_transforms[2][2] += v_depths[0];

        vec3<T> v_normal = vec3<T>(v_normals[0], v_normals[1], v_normals[2]);

        vec3<T> v_mean = vec3<T>(T(0.0));
        vec2<T> v_scale = vec2<T>(T(0.0));
        vec4<T> v_quat = vec4<T>(T(0.0));
        mat3<T> v_R = mat3<T>(T(0.0));
        vec3<T> v_t = vec3<T>(T(0.0));

        // Compute gradients using VJP
        compute_ray_transforms_aabb_vjp<T>(
            ray_transforms,
            v_means2d,
            v_normal,
            R,
            P,
            t,
            mean_w,
            mean_c,
            quat,
            scale,
            _v_ray_transforms,
            v_quat,
            v_scale,
            v_mean,
            v_R,
            v_t
        );

        // Write out results with atomic additions
        if (m_v_means != nullptr) {
            T* v_means_out = m_v_means + bid * m_N * 3 + gid * 3;
            gpuAtomicAdd(v_means_out, v_mean.x);
            gpuAtomicAdd(v_means_out + 1, v_mean.y);
            gpuAtomicAdd(v_means_out + 2, v_mean.z);
        }

        // Gradients w.r.t. quaternion and scale
        T* v_quats_out = m_v_quats + bid * m_N * 4 + gid * 4;
        T* v_scales_out = m_v_scales + bid * m_N * 3 + gid * 3;
        
        gpuAtomicAdd(v_quats_out, v_quat.x);
        gpuAtomicAdd(v_quats_out + 1, v_quat.y);
        gpuAtomicAdd(v_quats_out + 2, v_quat.z);
        gpuAtomicAdd(v_quats_out + 3, v_quat.w);
        
        gpuAtomicAdd(v_scales_out, v_scale.x);
        gpuAtomicAdd(v_scales_out + 1, v_scale.y);

        if (m_v_viewmats != nullptr) {
            T* v_viewmats_out = m_v_viewmats + bid * m_C * 16 + cid * 16;
            
            // Write rotation gradients (column-major to row-major)
            for (uint32_t i = 0; i < 3; i++) {
                for (uint32_t j = 0; j < 3; j++) {
                    gpuAtomicAdd(v_viewmats_out + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats_out + i * 4 + 3, v_t[i]);
            }
        }
    }
};

} // namespace gsplat::xpu

#endif // Projection2DGSFusedBwdKernel_HPP