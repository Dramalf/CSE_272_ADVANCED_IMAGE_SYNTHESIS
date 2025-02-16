#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyClearcoat &bsdf) const
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

    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Real eta = 1.5;
    // Real R0 = (eta - 1) * (eta - 1) / (eta + 1) / (eta + 1);
    Real R0 = 0.04;
    Real alpha_g = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Vector3 half_vector = normalize(dir_in + dir_out);
    Real h_dot_out = dot(half_vector, dir_out);
    Spectrum hl = to_local(frame, half_vector);

    Real D_c;
    {
        Real t1 = alpha_g * alpha_g - 1;
        Real t2 = c_PI * log(alpha_g * alpha_g);
        D_c = t1 / (t2 * (1 + t1 * hl.z * hl.z));
    }
    Real F_c = schlick_fresnel(R0, h_dot_out);
    Spectrum wl_in = to_local(frame, dir_in);
    Spectrum wl_out = to_local(frame, dir_out);
    Real G_c = disney_metal_smith_masking_G(wl_in, 0.25, 0.25) * disney_metal_smith_masking_G(wl_out, 0.25, 0.25);
    Spectrum base_color = Vector3(1.0, 1.0, 1.0);
    return base_color * F_c * D_c * G_c / (4 * dot(frame.n, dir_in));
}

Real pdf_sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const
{
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
        dot(vertex.geometric_normal, dir_out) < 0)
    {
        // No light below the surface
        return 0;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;

    // Homework 1: implement this!
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);

    Real alpha_g = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Vector3 half_vector = normalize(dir_in + dir_out);
    Spectrum hl = to_local(frame, half_vector);

    Real D_c;
    {
        Real t1 = alpha_g * alpha_g - 1;
        Real t2 = c_PI * log(alpha_g * alpha_g);
        D_c = t1 / (t2 * (1 + t1 * hl.z * hl.z));
    }
    Spectrum wl_in = to_local(frame, dir_in);
    Spectrum wl_out = to_local(frame, dir_out);
    Real G_c = disney_metal_smith_masking_G(wl_in, 0.25, 0.25) * disney_metal_smith_masking_G(wl_out, 0.25, 0.25);
    return D_c * G_c / (4 * dot(frame.n, dir_in));
}

std::optional<BSDFSampleRecord>
sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const
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
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Real h_elevation = acos(sqrt((1 - pow(pow(alpha, 2), 1 - rnd_param_uv[0])) / (1 - pow(alpha, 2))));
    Real h_azimuth = 2 * c_PI * rnd_param_uv[1];
    auto hl = Vector3(sin(h_elevation) * cos(h_azimuth), sin(h_elevation) * sin(h_azimuth), cos(h_elevation));
    Vector3 half_vector = to_world(frame, hl);
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
        reflected,
        Real(0) /* eta */, clearcoat_gloss /* roughness */
    };
}

TextureSpectrum get_texture_op::operator()(const DisneyClearcoat &bsdf) const
{
    return make_constant_spectrum_texture(make_zero_spectrum());
}