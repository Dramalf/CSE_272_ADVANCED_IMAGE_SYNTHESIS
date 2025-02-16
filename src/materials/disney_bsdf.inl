#include "../microfacet.h"


Spectrum eval_op::operator()(const DisneyBSDF &bsdf) const
{
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                       dot(vertex.geometric_normal, dir_out) >
                   0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    // Homework 1: implement this!
    if (!reflect || dot(frame.n, dir_in) <= 0)
    {
        return (*this)(DisneyGlass{bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta});
    }
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_tint = eval(bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    DisneyDiffuse diffuse_bsdf = {bsdf.base_color, bsdf.roughness, bsdf.subsurface};
    DisneyClearcoat clearcoat_bsdf = {bsdf.clearcoat_gloss};
    DisneySheen sheen_bsdf = {bsdf.base_color, bsdf.sheen_tint};
    DisneyGlass glass_bsdf = {bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta};

    Spectrum metal_color;

    {
        Real L = luminance(base_color);
        Spectrum C_tint = (L > 0.0) ? (base_color / L) : Spectrum(1.0, 1.0, 1.0);
        Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

        Spectrum K_s = (1 - specular_tint) + specular_tint * C_tint;
        metal_color = specular * pow((eta - 1) / (eta + 1), 2) * (1 - metallic) * K_s + metallic * base_color;
    }

    DisneyMetal metal_bsdf = {make_constant_spectrum_texture(metal_color), bsdf.roughness, bsdf.anisotropic};
    bool inside_object = dot(vertex.shading_frame.n, dir_out) <= 0;
    Spectrum f_diffuse = inside_object ? Vector3{0, 0, 0} : this->operator()(diffuse_bsdf);
    Spectrum f_clearcoat = inside_object ? Vector3{0, 0, 0} : this->operator()(clearcoat_bsdf);
    Spectrum f_sheen = inside_object ? Vector3{0, 0, 0} : this->operator()(sheen_bsdf);
    Spectrum f_glass = this->operator()(glass_bsdf);
    Spectrum f_metal = inside_object ? Vector3{0, 0, 0} : this->operator()(metal_bsdf);
    std::vector<Real> weights;
    calculate_bsdf_weights(weights, specular, specular_transmission, metallic, clearcoat, sheen);
    Real w_diffuse = weights[0];
    Real w_sheen = weights[1];
    Real w_metal = weights[2];
    Real w_clearcoat = weights[3];
    Real w_glass = weights[4];
    return w_diffuse * f_diffuse + w_sheen * f_sheen + w_metal * f_metal + w_clearcoat * f_clearcoat + w_glass * f_glass;
}

Real pdf_sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const
{
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                       dot(vertex.geometric_normal, dir_out) >
                   0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;

    // Homework 1: implement this!
    if (!reflect || dot(frame.n, dir_in) <= 0)
    {
        return (*this)(DisneyGlass{bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta});
    }
    Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    DisneyDiffuse diffuse_bsdf = {bsdf.base_color, bsdf.roughness, bsdf.subsurface};
    DisneyClearcoat clearcoat_bsdf = {bsdf.clearcoat_gloss};
    DisneyGlass glass_bsdf = {bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta};
    DisneyMetal metal_bsdf = {bsdf.base_color, bsdf.roughness, bsdf.anisotropic};
    DisneySheen sheen_bsdf = {bsdf.base_color, bsdf.sheen_tint};
    std::vector<Real> weights;
    calculate_bsdf_weights(weights, specular, specular_transmission, metallic, clearcoat, sheen);

    Real w_diffuse = weights[0];
    Real w_sheen = weights[1];
    Real w_metal = weights[2];
    Real w_clearcoat = weights[3];
    Real w_glass = weights[4];

    Real total_w = w_diffuse + w_metal + w_glass + w_clearcoat + w_sheen;

    Real diffuse_bsdf_pdf = (*this)(diffuse_bsdf);
    Real sheen_bsdf_pdf = (*this)(sheen_bsdf);
    Real metal_bsdf_pdf = (*this)(metal_bsdf);
    Real glass_bsdf_pdf = (*this)(glass_bsdf);
    Real clearcoat_bsdf_pdf = (*this)(clearcoat_bsdf);

    return (diffuse_bsdf_pdf * w_diffuse + sheen_bsdf_pdf * w_sheen + metal_bsdf_pdf * w_metal + glass_bsdf_pdf * w_glass + clearcoat_bsdf_pdf * w_clearcoat) / total_w;
}

std::optional<BSDFSampleRecord>
sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const
{
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    // if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0)
    // {
    //     frame = -frame;
    // }
    // Homework 1: implement this!
    if (dot(frame.n, dir_in) <= 0)
    {
        return (*this)(DisneyGlass{bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta});
    }
    Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);

    std::vector<Real> weights;
    calculate_bsdf_weights(weights, specular, specular_transmission, metallic, clearcoat, sheen);

    Real w_diffuse = weights[0];
    Real w_metal = weights[2];
    Real w_clearcoat = weights[3];
    Real w_glass = weights[4];

    w_metal += w_diffuse;
    w_glass += w_metal;
    w_clearcoat += w_glass;
    Real p = rnd_param_w * w_clearcoat;
    if (p <= w_diffuse)
    {
        // diffuse
        return (*this)(DisneyDiffuse{bsdf.base_color, bsdf.roughness, bsdf.subsurface});
    }
    else if (p <= w_metal)
    {
        // metal
        return (*this)(DisneyMetal{bsdf.base_color, bsdf.roughness, bsdf.anisotropic});
    }
    else if (p <= w_glass)
    {
        // glass
        return (*this)(DisneyGlass{bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta});
    }
    else
    {
        // clearcoat
        return (*this)(DisneyClearcoat{bsdf.clearcoat_gloss});
    }
    return {};
}

TextureSpectrum get_texture_op::operator()(const DisneyBSDF &bsdf) const
{
    return bsdf.base_color;
}