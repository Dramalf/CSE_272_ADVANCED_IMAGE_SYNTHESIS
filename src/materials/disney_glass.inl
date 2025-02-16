#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyGlass &bsdf) const
{
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                       dot(vertex.geometric_normal, dir_out) >
                   0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0)
    {
        frame = -frame;
    }
    // Homework 1: implement this!
    (void)reflect; // silence unuse warning, remove this when implementing hw
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometric_normal, dir_in) >= 0 ? bsdf.eta : 1 / bsdf.eta;

     Vector3 half_vector;
    if (reflect) {
        half_vector = normalize(dir_in + dir_out);
    } else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }
    Real h_dot_out = dot(half_vector, dir_out);
    Real h_dot_in = dot(half_vector, dir_in);
    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real alpha_min = 0.0001;
    Real alpha_x = std::fmax(alpha_min, roughness * roughness / aspect);
    Real alpha_y = std::fmax(alpha_min, roughness * roughness * aspect);
    Spectrum hl = Vector3(dot(half_vector, frame[0]), dot(half_vector, frame[1]), dot(half_vector, frame[2]));
    Real D_g;
    {
        Real t1 = 1 / (c_PI * alpha_x * alpha_y);
        Real t2 = (hl.x * hl.x) / (alpha_x * alpha_x);
        Real t3 = (hl.y * hl.y) / (alpha_y * alpha_y);
        Real t4 = hl.z * hl.z;
        Real t5 = (t2 + t3 + t4) * (t2 + t3 + t4);
        D_g = t1 / t5;
    }

    Spectrum wl_in = to_local(frame, dir_in);
    Spectrum wl_out = to_local(frame, dir_out);
    Real G_g = disney_metal_smith_masking_G(wl_in, alpha_x, alpha_y) * disney_metal_smith_masking_G(wl_out, alpha_x, alpha_y);
    Real R_s = (h_dot_in - h_dot_out * eta) / (h_dot_in + eta * h_dot_out);
    Real R_p = (eta * h_dot_in - h_dot_out) / (eta * h_dot_in + h_dot_out);

    Real F_g = 0.5 * (R_s * R_s + R_p * R_p);

    if (reflect)
    {
        return base_color * F_g * D_g * G_g / (4 * dot(frame.n, dir_in));
    }
    else
    {
        return sqrt(base_color) * (1 - F_g) * D_g * G_g * abs(h_dot_in * h_dot_out) / abs(dot(frame.n, dir_in) * pow(h_dot_in + eta * h_dot_out, 2));
    }
}

Real pdf_sample_bsdf_op::operator()(const DisneyGlass &bsdf) const
{
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                       dot(vertex.geometric_normal, dir_out) >
                   0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0)
    {
        frame = -frame;
    }
    // HW1
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    assert(eta > 0);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 half_vector;
    if (reflect)
    {
        half_vector = normalize(dir_in + dir_out);
    }
    else
    {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0)
    {
        half_vector = -half_vector;
    }

    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    // We sample the visible normals, also we use F to determine
    // whether to sample reflection or refraction
    // so PDF ~ F * D * G_in for reflection, PDF ~ (1 - F) * D * G_in for refraction.
    Real h_dot_in = dot(half_vector, dir_in);
    Real h_dot_out = dot(half_vector, dir_out);
    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real alpha_min = 0.0001;
    Real alpha_x = std::fmax(alpha_min, roughness * roughness / aspect);
    Real alpha_y = std::fmax(alpha_min, roughness * roughness * aspect);
    Spectrum hl = Vector3(dot(half_vector, frame[0]), dot(half_vector, frame[1]), dot(half_vector, frame[2]));

    Real D_g;
    {
        Real t1 = 1 / (c_PI * alpha_x * alpha_y);
        Real t2 = (hl.x * hl.x) / (alpha_x * alpha_x);
        Real t3 = (hl.y * hl.y) / (alpha_y * alpha_y);
        Real t4 = hl.z * hl.z;
        Real t5 = (t2 + t3 + t4) * (t2 + t3 + t4);
        D_g = t1 / t5;
    }
    Real R_s = (h_dot_in - h_dot_out * eta) / (h_dot_in + eta * h_dot_out);
    Real R_p = (eta * h_dot_in - h_dot_out) / (eta * h_dot_in + h_dot_out);

    Real F_g = 0.5 * (R_s * R_s + R_p * R_p);
    if (reflect)
    {
        return (F_g * D_g) / (4 * fabs(dot(frame.n, dir_in)));
    }
    else
    {
        Real h_dot_out = dot(half_vector, dir_out);
        Real sqrt_denom = h_dot_in + eta * h_dot_out;
        Real dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
        return (1 - F_g) * D_g * fabs(dh_dout * h_dot_in / dot(frame.n, dir_in));
    }
}

std::optional<BSDFSampleRecord>
sample_bsdf_op::operator()(const DisneyGlass &bsdf) const
{
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0)
    {
        frame = -frame;
    }
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    // Sample a micro normal and transform it to world space -- this is our half-vector.
    Real alpha = roughness * roughness;
    Vector3 local_dir_in = to_local(frame, dir_in);
    Vector3 local_micro_normal =
        sample_visible_normals(local_dir_in, alpha, rnd_param_uv);

    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0)
    {
        half_vector = -half_vector;
    }

    // Now we need to decide whether to reflect or refract.
    // We do this using the Fresnel term.
    Real h_dot_in = dot(half_vector, dir_in);
    Real F = fresnel_dielectric(h_dot_in, eta);

    if (rnd_param_w <= F)
    {
        // Reflection
        Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
        // set eta to 0 since we are not transmitting
        return BSDFSampleRecord{reflected, Real(0) /* eta */, roughness};
    }
    else
    {
        // Refraction
        // https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
        // (note that our eta is eta2 / eta1, and l = -dir_in)
        auto refract_dir = [](Spectrum dir_in, Spectrum half_vector, Real eta)
        {
            Real cos_theta_1 = dot(dir_in, half_vector);
            Real sin2_theta_2 = (1.0 - cos_theta_1 * cos_theta_1) / eta / eta;
            Real cos_theta_2 = sqrt(1.0 - sin2_theta_2);
            return normalize(-dir_in / eta + (cos_theta_1 / eta - cos_theta_2) * half_vector);
        };
        Vector3 refracted = refract_dir(dir_in, half_vector, eta);
        return BSDFSampleRecord{refracted, eta, roughness};
    }
}

TextureSpectrum get_texture_op::operator()(const DisneyGlass &bsdf) const
{
    return bsdf.base_color;
}
