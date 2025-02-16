#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyMetal &bsdf) const
{
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
        dot(vertex.geometric_normal, dir_out) < 0)
    {
        // No light below the surface
        return make_zero_spectrum();
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0)
    {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 half_vector = normalize(dir_in + dir_out);

    Real h_dot_out = dot(half_vector, dir_out);
    Spectrum F_m = schlick_fresnel(base_color, h_dot_out);

    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real alpha_min = 0.0001;
    Real alpha_x = std::fmax(alpha_min, roughness * roughness / aspect);
    Real alpha_y = std::fmax(alpha_min, roughness * roughness * aspect);
    Spectrum hl = Vector3(dot(half_vector, frame[0]), dot(half_vector, frame[1]), dot(half_vector, frame[2]));

    Real D_m;
    {
        Real t1 = 1 / (c_PI * alpha_x * alpha_y);
        Real t2 = (hl.x * hl.x) / (alpha_x * alpha_x);
        Real t3 = (hl.y * hl.y) / (alpha_y * alpha_y);
        Real t4 = hl.z * hl.z;
        Real t5 = (t2 + t3 + t4) * (t2 + t3 + t4);
        D_m = t1 / t5;
    }

    Spectrum wl_in = to_local(frame, dir_in);
    Spectrum wl_out = to_local(frame, dir_out);
    Real G_m = disney_metal_smith_masking_G(wl_in, alpha_x, alpha_y) * disney_metal_smith_masking_G(wl_out, alpha_x, alpha_y);

    Real n_dot_in = dot(frame.n, dir_in);

    return F_m * D_m * G_m / (4 * n_dot_in);
}

Real pdf_sample_bsdf_op::operator()(const DisneyMetal &bsdf) const
{
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
        dot(vertex.geometric_normal, dir_out) < 0)
    {
        // No light below the surface
        return 0;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0)
    {
        frame = -frame;
    }
    // Homework 1: implement this!
    Vector3 half_vector = normalize(dir_in + dir_out);

    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real alpha_min = 0.0001;
    Real alpha_x = std::fmax(alpha_min, roughness * roughness / aspect);
    Real alpha_y = std::fmax(alpha_min, roughness * roughness * aspect);
    Spectrum hl = Vector3(dot(half_vector, frame[0]), dot(half_vector, frame[1]), dot(half_vector, frame[2]));

    Real D_m;
    {
        Real t1 = 1 / (c_PI * alpha_x * alpha_y);
        Real t2 = (hl.x * hl.x) / (alpha_x * alpha_x);
        Real t3 = (hl.y * hl.y) / (alpha_y * alpha_y);
        Real t4 = hl.z * hl.z;
        Real t5 = (t2 + t3 + t4) * (t2 + t3 + t4);
        D_m = t1 / t5;
    }

    Spectrum wl_in = to_local(frame, dir_in);
    Real G_in = disney_metal_smith_masking_G(wl_in, alpha_x, alpha_y);

    Real n_dot_in = dot(frame.n, dir_in);

    return D_m * G_in / (4 * n_dot_in);
}

std::optional<BSDFSampleRecord>
sample_bsdf_op::operator()(const DisneyMetal &bsdf) const
{
    if (dot(vertex.geometric_normal, dir_in) < 0)
    {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0)
    {
        frame = -frame;
    }
    // Homework 1: implement this!
    // visible normal sampling
    // refer roughplastic
    Vector3 local_dir_in = to_local(frame, dir_in);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha = roughness * roughness;
    Vector3 local_normal = sample_visible_normals(local_dir_in, alpha, rnd_param_uv);

    // Transform the micro normal to world space
    Vector3 half_vector = to_world(frame, local_normal);
    // Reflect over the world space normal
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
        reflected,
        Real(1.5) /* eta */, roughness /* roughness */
    };
}

TextureSpectrum get_texture_op::operator()(const DisneyMetal &bsdf) const
{
    return bsdf.base_color;
}

