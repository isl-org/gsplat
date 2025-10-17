#ifndef Projection2DGSFusedFwdKernel_HPP
#define Projection2DGSFusedFwdKernel_HPP

#include "utils.hpp"
#include "quat_scale_to_covar_preci.hpp"
#include "transform.hpp"

namespace gsplat::xpu {

template<typename T>
struct Projection2DGSFusedFwdKernel {
    const uint32_t m_B;
    const uint32_t m_C;
    const uint32_t m_N;
    const T* m_means;          // [B, N, 3]
    const T* m_quats;          // [B, N, 4]
    const T* m_scales;         // [B, N, 3]
    const T* m_viewmats;       // [B, C, 4, 4]
    const T* m_Ks;             // [B, C, 3, 3]
    const int32_t m_image_width;
    const int32_t m_image_height;
    const T m_near_plane;
    const T m_far_plane;
    const T m_radius_clip;
    // outputs
    int32_t* m_radii;          // [B, C, N, 2]
    T* m_means2d;              // [B, C, N, 2]
    T* m_depths;               // [B, C, N]
    T* m_ray_transforms;       // [B, C, N, 3, 3]
    T* m_normals;              // [B, C, N, 3]

    Projection2DGSFusedFwdKernel(
        const uint32_t B,
        const uint32_t C,
        const uint32_t N,
        const T* means,
        const T* quats,
        const T* scales,
        const T* viewmats,
        const T* Ks,
        const int32_t image_width,
        const int32_t image_height,
        const T near_plane,
        const T far_plane,
        const T radius_clip,
        int32_t* radii,
        T* means2d,
        T* depths,
        T* ray_transforms,
        T* normals
    )
    : m_B(B), m_C(C), m_N(N), m_means(means), m_quats(quats), m_scales(scales),
      m_viewmats(viewmats), m_Ks(Ks), m_image_width(image_width), m_image_height(image_height),
      m_near_plane(near_plane), m_far_plane(far_plane), m_radius_clip(radius_clip),
      m_radii(radii), m_means2d(means2d), m_depths(depths),
      m_ray_transforms(ray_transforms), m_normals(normals)
    {}

    void operator()(sycl::nd_item<1> work_item) const
    {
        uint32_t idx = work_item.get_global_id(0);
        
        if (idx >= m_B * m_C * m_N) {
            return;
        }
        
        const uint32_t bid = idx / (m_C * m_N); // batch id
        const uint32_t cid = (idx / m_N) % m_C; // camera id
        const uint32_t gid = idx % m_N;         // gaussian id

        // Load data and construct pointers
        const T* means = m_means + bid * m_N * 3 + gid * 3;
        const T* viewmats = m_viewmats + bid * m_C * 16 + cid * 16;
        const T* Ks = m_Ks + bid * m_C * 9 + cid * 9;

        // glm is column-major but input is row-major
        // Rotation component of the camera (explicit transpose)
        mat3<T> R = mat3<T>(
            viewmats[0], viewmats[4], viewmats[8],   // 1st column
            viewmats[1], viewmats[5], viewmats[9],   // 2nd column
            viewmats[2], viewmats[6], viewmats[10]   // 3rd column
        );
        
        // Translation component of the camera
        vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

        // Transform Gaussian center to camera space
        vec3<T> mean_c;
        pos_world_to_cam(R, t, vec3<T>(means[0], means[1], means[2]), mean_c);

        // Return if primitive is outside valid depth range
        if (mean_c.z <= m_near_plane || mean_c.z >= m_far_plane) {
            m_radii[idx * 2] = 0;
            m_radii[idx * 2 + 1] = 0;
            return;
        }

        const T* quats = m_quats + bid * m_N * 4 + gid * 4;
        const T* scales = m_scales + bid * m_N * 3 + gid * 3;

        // Build rotation matrix from quaternion (quat_to_rotmat returns a mat3)
        mat3<T> rot_mat = quat_to_rotmat(vec4<T>(quats[0], quats[1], quats[2], quats[3]));

        // Build scale matrix (only x and y for 2D, z is 1)
        mat3<T> scale_mat = mat3<T>(
            scales[0], T(0.0), T(0.0),
            T(0.0), scales[1], T(0.0),
            T(0.0), T(0.0), T(1.0)
        );

        // RS_camera = R * quat_to_rotmat * scale_mat
        mat3<T> RS_camera = R * rot_mat * scale_mat;

        // WH = [RS_camera[0], RS_camera[1], mean_c]
        mat3<T> WH = mat3<T>(
            RS_camera[0][0], RS_camera[1][0], mean_c.x,
            RS_camera[0][1], RS_camera[1][1], mean_c.y,
            RS_camera[0][2], RS_camera[1][2], mean_c.z
        );

        // Projective transformation matrix: Camera -> Screen
        // K^T in column-major order
        mat3<T> world_2_pix = mat3<T>(
            Ks[0], T(0.0), Ks[2],
            T(0.0), Ks[4], Ks[5],
            T(0.0), T(0.0), T(1.0)
        );

        // M = (WH)^T * K^T
        mat3<T> M = glm::transpose(WH) * world_2_pix;

        // Compute AABB
        const vec3<T> M0 = vec3<T>(M[0][0], M[0][1], M[0][2]); // first row of KWH
        const vec3<T> M1 = vec3<T>(M[1][0], M[1][1], M[1][2]); // second row of KWH
        const vec3<T> M2 = vec3<T>(M[2][0], M[2][1], M[2][2]); // third row of KWH

        const vec3<T> temp_point = vec3<T>(T(1.0), T(1.0), T(-1.0));

        // Algebraic manipulation for computing mean and radius
        const T distance = dot(temp_point * M2, M2);

        // Ignore ill-conditioned primitives
        if (distance == T(0.0)) {
            m_radii[idx * 2] = 0;
            m_radii[idx * 2 + 1] = 0;
            return;
        }

        const vec3<T> f = (T(1.0) / distance) * temp_point;
        const vec2<T> mean2d = vec2<T>(
            dot(f * M0, M2),
            dot(f * M1, M2)
        );

        const vec2<T> temp = vec2<T>(
            dot(f * M0, M0),
            dot(f * M1, M1)
        );
        const vec2<T> half_extend = mean2d * mean2d - temp;

        const T radius_x = sycl::ceil(T(3.33) * sycl::sqrt(sycl::max(T(1e-4), half_extend.x)));
        const T radius_y = sycl::ceil(T(3.33) * sycl::sqrt(sycl::max(T(1e-4), half_extend.y)));

        if (radius_x <= m_radius_clip && radius_y <= m_radius_clip) {
            m_radii[idx * 2] = 0;
            m_radii[idx * 2 + 1] = 0;
            return;
        }

        // Culling: mask out gaussians outside the image region
        if (mean2d.x + radius_x <= T(0) || mean2d.x - radius_x >= m_image_width ||
            mean2d.y + radius_y <= T(0) || mean2d.y - radius_y >= m_image_height) {
            m_radii[idx * 2] = 0;
            m_radii[idx * 2 + 1] = 0;
            return;
        }

        // Compute normals (dual visible)
        vec3<T> normal = vec3<T>(RS_camera[2][0], RS_camera[2][1], RS_camera[2][2]);
        
        // Flip normal if it is pointing away from the camera
        T multiplier = dot(-normal, mean_c) > T(0) ? T(1.0) : T(-1.0);
        normal *= multiplier;

        // Write to outputs
        m_radii[idx * 2] = (int32_t)radius_x;
        m_radii[idx * 2 + 1] = (int32_t)radius_y;
        m_means2d[idx * 2] = mean2d.x;
        m_means2d[idx * 2 + 1] = mean2d.y;
        m_depths[idx] = mean_c.z;

        // Store ray transforms (row major KWH)
        m_ray_transforms[idx * 9] = M0.x;
        m_ray_transforms[idx * 9 + 1] = M0.y;
        m_ray_transforms[idx * 9 + 2] = M0.z;
        m_ray_transforms[idx * 9 + 3] = M1.x;
        m_ray_transforms[idx * 9 + 4] = M1.y;
        m_ray_transforms[idx * 9 + 5] = M1.z;
        m_ray_transforms[idx * 9 + 6] = M2.x;
        m_ray_transforms[idx * 9 + 7] = M2.y;
        m_ray_transforms[idx * 9 + 8] = M2.z;

        // Store primitive normals
        m_normals[idx * 3] = normal.x;
        m_normals[idx * 3 + 1] = normal.y;
        m_normals[idx * 3 + 2] = normal.z;
    }
};

} // namespace gsplat::xpu

#endif // Projection2DGSFusedFwdKernel_HPP