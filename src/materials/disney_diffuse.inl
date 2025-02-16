#include "lajolla.h"

Spectrum eval_op::operator()(const DisneyDiffuse &bsdf) const
{
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
        dot(vertex.geometric_normal, dir_out) < 0)
    {
        // No light below the surface
        return make_zero_spectrum();
    }

    Frame frame = vertex.shading_frame;
    Real ndin = dot(frame.n, dir_in);
    Real ndout = dot(frame.n, dir_out);
    if (ndin <= 0 || ndout <= 0)
    {
        return make_zero_spectrum();
    }

    // Homework 1: implement this!
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);

    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);

    Real internal_media_absorption = 1.0 / (ndin + ndout) - 0.5;

    // Flip half-vector if it's below surface
    //     We use the standard formula
    // for −→hr , except that we modulate it by the sign of (i · n) so
    // that our equations will work for directions on either side of
    // the surface (i.e. front or back).
    Vector3 half_vector = normalize(dir_in + dir_out);

    if (dot(half_vector, frame.n) < 0)
    {
        half_vector = -half_vector;
    }
    Real h_dot_out = dot(half_vector, dir_out);

    Real F_SS90 = roughness * h_dot_out * h_dot_out;

    // Good question! They are actually both correct.
    // The F_D term (they call it "Diffuse Fresnel") in the Disney BSDF is
    // designed to be 1 at normal incident angle (cos_theta = 0)
    // and to be F_D90 at grazing angle (theta = 90 degree).
    // On the other hand, the standard Schlick approximation is
    // designed to be F0 at normal incident angle,
    // and 1 at grazing angle. Hence the difference.
    auto disney_diffuse_schlick_fresnel = [](Real F0, Real cosTheta)
    {
        return 1.0 + (F0 - 1.0) * pow(1.0 - cosTheta, 5);
    };

    Spectrum f_subsurface = 1.25 * base_color * (disney_diffuse_schlick_fresnel(F_SS90, ndin) * disney_diffuse_schlick_fresnel(F_SS90, ndout) * internal_media_absorption + 0.5) * ndout / c_PI;
    Real F_D90 = 0.5 + 2.0 * F_SS90;
    Spectrum f_baseDiffuse = base_color * disney_diffuse_schlick_fresnel(F_D90, ndin) * disney_diffuse_schlick_fresnel(F_D90, ndout) * ndout / c_PI;
    Spectrum f_diffuse = (1.0 - subsurface) * f_baseDiffuse + subsurface * f_subsurface;
    return f_diffuse;
}

Real pdf_sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const
{
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
        dot(vertex.geometric_normal, dir_out) < 0)
    {
        // No light below the surface
        return 0;
    }

    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0)
    {
        return 0;
    }

    // Homework 1: implement this!
    return fmax(dot(frame.n, dir_out), Real(0)) / c_PI;
}

std::optional<BSDFSampleRecord> sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const
{
    if (dot(vertex.geometric_normal, dir_in) < 0)
    {
        // No light below the surface
        return {};
    }

    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0)
    {
        return {};
    }

    // Homework 1: implement this!
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    return BSDFSampleRecord{
        to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
        Real(0) /* eta */, Real(roughness) /* roughness */};
}

TextureSpectrum get_texture_op::operator()(const DisneyDiffuse &bsdf) const
{
    return bsdf.base_color;
}
