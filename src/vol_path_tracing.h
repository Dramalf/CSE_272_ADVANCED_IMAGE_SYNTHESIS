#pragma once

int update_medium_id(const PathVertex &vertex, const Ray &ray, int medium_id)
{
    if (vertex.interior_medium_id != vertex.exterior_medium_id)
    {
        bool is_hit_inside = dot(ray.dir, vertex.geometric_normal) < 0;
        medium_id = is_hit_inside ? vertex.interior_medium_id : vertex.exterior_medium_id;
    }
    return medium_id;
}

Spectrum next_event_estimation(const Scene &scene, Ray ray, int current_medium_id, int bounces, pcg32_state &rng, const Material *mat, PathVertex original_vertex_)
{
    Vector2 rnd_param{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
    Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);
    int light_id = sample_light(scene, light_w);
    const Light &light = scene.lights[light_id];
    // the second parameter is no longer the vertex position, but the point in the medium
    PointAndNormal point_on_light = sample_point_on_light(light, ray.org, light_uv, shape_w, scene);
    Vector3 p_prime = point_on_light.position;
    Vector3 p = ray.org;
    Spectrum T_light = make_const_spectrum(1);
    int shadow_medium_id = current_medium_id;
    int shadow_bounces = 0;
    Vector3 original_p = p;
    Spectrum original_ray_dir = ray.dir;
    Spectrum p_trans_dir = make_const_spectrum(1);
    Spectrum dir_light = normalize(p_prime - p);

    while (true)
    {
        Ray shadow_ray = Ray{p, dir_light, get_shadow_epsilon(scene), (1 - get_shadow_epsilon(scene)) * distance(p_prime, p)};

        std::optional<PathVertex> vertex_ = intersect(scene, shadow_ray);
        Real next_t = distance(p, p_prime);

        if (vertex_)
        {
            next_t = distance(vertex_->position, p);
        }
        if (shadow_medium_id != -1)
        {
            Medium medium = scene.media[shadow_medium_id];
            Spectrum sigma_s = get_sigma_s(medium, ray.org);
            Spectrum sigma_a = get_sigma_a(medium, ray.org);
            Spectrum sigma_t = sigma_s + sigma_a; // get_majorant(medium, ray);
            T_light *= exp(-sigma_t * next_t);
            p_trans_dir *= exp(-sigma_t * next_t);
        }
        if (!vertex_)
        {
            break;
        }
        else
        {
            if (vertex_->material_id >= 0)
            {
                return make_zero_spectrum();
            }
            shadow_bounces++;
            if (scene.options.max_depth != -1 && bounces + shadow_bounces + 1 >= scene.options.max_depth)
            {
                return make_zero_spectrum();
            }
        }
        shadow_medium_id = update_medium_id(*vertex_, shadow_ray, shadow_medium_id);
        p += next_t * shadow_ray.dir;
    }

    if (T_light[0] > 0)
    {
        Spectrum dir_light = normalize(p_prime - original_p);
        Real G = max(-dot(dir_light, point_on_light.normal), Real(0)) / distance_squared(p_prime, original_p);
        Spectrum Le = emission(light, -dir_light, Real(0), point_on_light, scene);
        Real pdf_nee = light_pmf(scene, light_id) *
                       pdf_point_on_light(light, point_on_light, original_p, scene);
        Real pdf_scatter = G * p_trans_dir[0];
        Spectrum f;

        if (mat)
        {
            pdf_scatter *= pdf_sample_bsdf(*mat, original_ray_dir, dir_light, original_vertex_, scene.texture_pool);
            f = eval(*mat, -original_ray_dir, dir_light, original_vertex_, scene.texture_pool);
        }
        else
        {
            Medium medium = scene.media[current_medium_id];
            PhaseFunction phase_function = get_phase_function(medium);
            f = eval(phase_function, -original_ray_dir, dir_light);
            pdf_scatter *= pdf_sample_phase(phase_function, -original_ray_dir, dir_light);
        }
        Spectrum contrib = T_light * Le * f * G / pdf_nee;
        Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_scatter * pdf_scatter);
        return w * contrib;
    }

    return make_zero_spectrum();
}
Spectrum next_event_estimation_final(const Scene &scene, Ray ray, int current_medium_id, int bounces, pcg32_state &rng, const Material *mat, std::optional<PathVertex> original_vertex_)
{
    Vector2 rnd_param{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
    Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);
    int light_id = sample_light(scene, light_w);
    const Light &light = scene.lights[light_id];
    PointAndNormal point_on_light = sample_point_on_light(light, ray.org, light_uv, shape_w, scene);
    Vector3 p_prime = point_on_light.position;
    Vector3 p = ray.org;
    Spectrum T_light = make_const_spectrum(1);
    Spectrum p_trans_dir = make_const_spectrum(1);
    Spectrum p_trans_nee = make_const_spectrum(1);
    Spectrum dir_light = normalize(p_prime - p);
    int shadow_medium_id = current_medium_id;
    int shadow_bounces = 0;
    while (true)
    {
        // actually the direction of shadow ray doesn't change
        Ray shadow_ray{p, dir_light, get_shadow_epsilon(scene),
                       (1 - get_shadow_epsilon(scene)) *
                           distance(p_prime, p)};
        std::optional<PathVertex> vertex_ = intersect(scene, shadow_ray);
        Real next_t = distance(p, p_prime);
        if (!vertex_)
        {
            break;
        }
        next_t = distance(p, vertex_->position);
        if (shadow_medium_id != -1)
        {
            Spectrum majorant = get_majorant(scene.media[shadow_medium_id], shadow_ray);
            Real u = next_pcg32_real<Real>(rng);
            int channel = std::clamp(int(u * 3), 0, 2);
            Real accum_t = 0;
            int iteration = 0;
            while (true)
            {

                if (majorant[channel] <= 0 || iteration >= scene.options.max_null_collisions)
                {
                    break;
                };
                Real t = -log(1 - next_pcg32_real<Real>(rng)) / majorant[channel];
                Real dt = next_t - accum_t;
                accum_t = min(accum_t + t, next_t);
                if (t < dt)
                {
                    // didn’t hit the surface, so this is a null-scattering event
                    Spectrum sigma_a = get_sigma_a(scene.media[shadow_medium_id], shadow_ray.org + accum_t * shadow_ray.dir);
                    Spectrum sigma_s = get_sigma_s(scene.media[shadow_medium_id], shadow_ray.org + accum_t * shadow_ray.dir);
                    Spectrum sigma_t = sigma_a + sigma_s;
                    Spectrum sigma_n = majorant - sigma_t;
                    T_light *= exp(-majorant * t) * sigma_n / max(majorant);
                    p_trans_nee *= exp(-majorant * t) * majorant / max(majorant);
                    Spectrum real_prob = sigma_t / majorant;
                    p_trans_dir *= exp(-majorant * t) * majorant * (1 - real_prob) / max(majorant);
                    if (max(T_light) <= 0)
                    {
                        break;
                    }
                }
                else
                {
                    // hit the surface
                    T_light *= exp(-majorant * dt);
                    p_trans_nee *= exp(-majorant * dt);
                    p_trans_dir *= exp(-majorant * dt);
                    break;
                }
                iteration++;
            }
        }

        if (vertex_->material_id >= 0)
        {
            return make_zero_spectrum();
        }
        shadow_bounces += 1;
        if ((scene.options.max_depth != -1) && (bounces + shadow_bounces + 1 >= scene.options.max_depth))
            return make_zero_spectrum();
        shadow_medium_id = update_medium_id(*vertex_, shadow_ray, shadow_medium_id);
        p = p + next_t * dir_light;
    }
    if (max(T_light) > 0)
    {
        // Compute T_light * G * rho * L & pdf_nee
        Real G = max(-dot(dir_light, point_on_light.normal), Real(0)) / distance_squared(p_prime, ray.org);
        Spectrum Le = emission(light, -dir_light, Real(0), point_on_light, scene);
        Spectrum pdf_nee = p_trans_nee * light_pmf(scene, light_id) * pdf_point_on_light(light, point_on_light, ray.org, scene);
        Spectrum f = make_zero_spectrum();
        Spectrum pdf_scatter = G * p_trans_dir;
        if (!mat)
        {
            PhaseFunction phase = get_phase_function(scene.media[current_medium_id]);
            f = eval(phase, -ray.dir, dir_light);
            pdf_scatter *= pdf_sample_phase(phase, -ray.dir, dir_light);
        }
        else
        {
            PathVertex vertex = *original_vertex_;
            const Material &mat = scene.materials[vertex.material_id];
            f = eval(mat, -ray.dir, dir_light, vertex, scene.texture_pool);
            pdf_scatter *= pdf_sample_bsdf(mat, -ray.dir, dir_light, vertex, scene.texture_pool);
        }
        Spectrum contrib = T_light * G * f * Le / avg(pdf_nee);
        if (pdf_scatter[0] <= 0 || pdf_scatter[1] <= 0 || pdf_scatter[2] <= 0)
        {
            return make_zero_spectrum();
        }
        Spectrum w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_scatter * pdf_scatter);
        return contrib * w;
    }
    return make_zero_spectrum();
}

// The simplest volumetric renderer:
// single absorption only homogeneous volume
// only handle directly visible light sources
Spectrum vol_path_tracing_1(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng)
{
    // Homework 2: implememt this!
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(Real(0), Real(0));

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    if (!vertex_)
    {
        // Hit background. Account for the environment map if needed.

        return make_zero_spectrum();
    }
    PathVertex vertex = *vertex_;
    Spectrum radiance = make_zero_spectrum();

    if (is_light(scene.shapes[vertex.shape_id]))
    {
        bool is_hit_inside = dot(ray.dir, vertex.geometric_normal) < 0;
        int medium_id = is_hit_inside ? vertex.interior_medium_id : vertex.exterior_medium_id;
        // The Mediums are stored in scene.media which you can access through scene.media[medium_id].
        Medium medium = scene.media[medium_id];
        Spectrum sigma_a = get_sigma_a(medium, vertex.position);
        Spectrum Le = emission(vertex, -ray.dir, scene);
        Real t_hit = distance(ray.org, vertex.position);
        radiance = Le * exp(-sigma_a * t_hit);
    }
    return radiance;
}

// The second simplest volumetric renderer:
// single monochromatic homogeneous volume with single scattering,
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_2(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng)
{
    // Homework 2: implememt this!
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(Real(0), Real(0));

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    Real t_hit = infinity<Real>();
    PathVertex vertex;
    Medium medium = scene.media[scene.camera.medium_id];
    if (vertex_)
    {
        vertex = *vertex_;
        t_hit = distance(vertex.position, ray.org);
        bool is_hit_inside = dot(vertex.shading_frame.n, vertex.geometric_normal) < 0;
        int medium_id = is_hit_inside ? vertex.interior_medium_id : vertex.exterior_medium_id;
        // The Mediums are stored in scene.media which you can access through scene.media[medium_id].
        medium = scene.media[medium_id];
    }

    Spectrum sigma_a = get_sigma_a(medium, Vector3(0.0, 0.0, 0.0));
    Spectrum sigma_s = get_sigma_s(medium, Vector3(0.0, 0.0, 0.0));
    Spectrum sigma_t = sigma_a + sigma_s;
    Real u = next_pcg32_real<Real>(rng);
    Real t = -log(1 - u) / sigma_t[0];
    // hit nothing, the energy comes from the medium
    if (t < t_hit)
    {
        Vector3 trans_pdf = exp(-sigma_t * t) * sigma_t;
        Vector3 transmittance = exp(-sigma_t * t);
        Vector3 p = ray.org + t * ray.dir;

        // First, we sample a point on the light source.
        // We do this by first picking a light source, then pick a point on it.
        Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
        Real light_w = next_pcg32_real<Real>(rng);
        Real shape_w = next_pcg32_real<Real>(rng);
        int light_id = sample_light(scene, light_w);
        const Light &light = scene.lights[light_id];
        // the second parameter is no longer the vertex position, but the point in the medium
        PointAndNormal point_on_light = sample_point_on_light(light, p, light_uv, shape_w, scene);
        // the above code is randmoly sampling a point on the light source and we need to compute the contribution

        // Let's first deal with C1 = G * f * L.
        // Let's first compute G.
        Real G = 0;
        // dir_light = normalize(point_on_light.position - vertex.position);
        // dir_light is no longer the above, because the actual hit point is p, it is shorter than the vertex position
        Vector3 dir_light = normalize(point_on_light.position - p);
        // If the point on light is occluded, G is 0. So we need to test for occlusion.
        // To avoid self intersection, we need to set the tnear of the ray
        // to a small "epsilon". We set the epsilon to be a small constant times the
        // scale of the scene, which we can obtain through the get_shadow_epsilon() function.
        Ray shadow_ray{p, dir_light, get_shadow_epsilon(scene), (1 - get_shadow_epsilon(scene)) * distance(point_on_light.position, p)};
        if (!occluded(scene, shadow_ray))
        {
            // geometry term is cosine at v_{i+1} divided by distance squared
            // this can be derived by the infinitesimal area of a surface projected on
            // a unit sphere -- it's the Jacobian between the area measure and the solid angle
            // measure.
            G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                distance_squared(point_on_light.position, p);
        }
        // Before we proceed, we first compute the probability density p1(v1)
        // The probability density for light sampling to sample our point is
        // just the probability of sampling a light times the probability of sampling a point
        Real L_s1_pdf = light_pmf(scene, light_id) *
                        pdf_point_on_light(light, point_on_light, p, scene);
        if (L_s1_pdf <= 0 || G <= 0)
        {
            return make_zero_spectrum();
        }
        Spectrum rho = eval(get_phase_function(medium), dir_light, -ray.dir);
        Spectrum Le = emission(light, -dir_light, Real(0), point_on_light, scene);
        Spectrum p_t_greater_than_t_hit = exp(-distance(point_on_light.position, p) * sigma_t);
        Spectrum L_s1_estimate = rho * Le * G * p_t_greater_than_t_hit;
        return (transmittance / trans_pdf) * sigma_s * (L_s1_estimate / L_s1_pdf);
    }
    // or we hit the light source.
    else if (is_light(scene.shapes[vertex.shape_id]))
    {
        Vector3 trans_pdf = exp(-sigma_t * t_hit);
        Vector3 transmittance = exp(-sigma_t * t_hit);
        return transmittance / trans_pdf * emission(vertex, -ray.dir, scene);
    }

    return make_zero_spectrum();
}

// The third volumetric renderer (not so simple anymore):
// multiple monochromatic homogeneous volumes with multiple scattering
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_3(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng)
{
    // Homework 2: implememt this!
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(Real(0), Real(0));
    int current_medium_id = scene.camera.medium_id;
    Spectrum current_path_throughput = make_const_spectrum(1);
    Spectrum radiance = make_zero_spectrum();
    int bounces = 0;
    while (true)
    {
        bool scatter = false;
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        PathVertex vertex;
        if (vertex_)
        {
            vertex = *vertex_;
        }
        Spectrum transmittance = make_const_spectrum(1);
        Spectrum trans_pdf = make_const_spectrum(1);

        if (current_medium_id != -1)
        {
            // sample t s.t. p(t) ~ exp(-sigma_t * t)
            // compute transmittance and trans_pdf
            // if t < t_hit, set scatter = True
            // ...
            const Medium &medium = scene.media[current_medium_id];
            Spectrum sigma_s = get_sigma_s(medium, ray.org);
            Spectrum sigma_a = get_sigma_a(medium, ray.org);
            Spectrum sigma_t = sigma_s + sigma_a;

            Real u = next_pcg32_real<Real>(rng);
            Real t = -log(1 - u) / sigma_t[0];
            Real t_hit = t_hit = infinity<Real>();
            if (vertex_)
            {
                t_hit = distance(vertex.position, ray.org);
            }

            if (t < t_hit)
            {
                scatter = true;
                trans_pdf = exp(-sigma_t * t) * sigma_t;
                transmittance = exp(-sigma_t * t);
                ray.org = ray.org + t * ray.dir;
            }
            else
            {
                trans_pdf = exp(-sigma_t * t_hit);
                transmittance = exp(-sigma_t * t_hit);
                ray.org = ray.org + t_hit * (1 + get_intersection_epsilon(scene)) * ray.dir;
            }
        }
        else if (vertex_)
        {
            ray.org = (*vertex_).position;
        }
        current_path_throughput *= (transmittance / trans_pdf);
        if (vertex_ && !scatter && is_light(scene.shapes[vertex.shape_id]))
        {
            // reach a light source, include emission
            radiance += current_path_throughput * emission(vertex, -ray.dir, scene);
        }

        if (bounces == scene.options.max_depth - 1 && scene.options.max_depth != -1)
        {
            break;
        }
        if (!scatter && vertex_ && vertex.material_id == -1)
        {
            current_medium_id = update_medium_id(vertex, ray, current_medium_id);
            bounces++;

            continue;
        }
        // sample next direct & update path throughput
        if (scatter)
        {
            Medium medium = scene.media[current_medium_id];
            Spectrum sigma_s = get_sigma_s(medium, ray.org);
            Vector2 rnd_param{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            PhaseFunction phase_function = get_phase_function(medium);
            std::optional<Spectrum> next_dir_ = sample_phase_function(phase_function, -ray.dir, rnd_param);
            Spectrum rho = eval(phase_function, -ray.dir, *next_dir_);
            Real dir_pdf = pdf_sample_phase(phase_function, -ray.dir, *next_dir_);
            if (dir_pdf <= 0)
            {
                break;
            }
            current_path_throughput *= rho / dir_pdf * sigma_s;
            ray.dir = *next_dir_;
        }
        else
        {
            // Hit a surface -- don’t need to deal with this yet
            break;
        }
        Real rr_prob = 1;
        if (bounces >= scene.options.rr_depth)
        {
            rr_prob = min(max(current_path_throughput), Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob)
            {
                // Terminate the path
                break;
            }
            else
            {
                current_path_throughput /= rr_prob;
            }
        }
        bounces++;
    }
    return radiance;
}

// The fourth volumetric renderer:
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
Spectrum vol_path_tracing_4(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng)
{
    // Homework 2: implememt this!
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(Real(0), Real(0));
    int current_medium_id = scene.camera.medium_id;
    Spectrum current_path_throughput = make_const_spectrum(1);
    Spectrum radiance = make_zero_spectrum();
    int bounces = 0;
    bool never_scatter = true;
    Real dir_pdf = 0;
    Vector3 nee_p_cache = Vector3{0, 0, 0};
    Spectrum multi_trans_pdf = Spectrum{1, 1, 1};

    while (true)
    {
        bool scatter = false;
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        PathVertex vertex;
        if (vertex_)
        {
            vertex = *vertex_;
        }
        Spectrum transmittance = make_const_spectrum(1);
        Spectrum trans_pdf = make_const_spectrum(1);

        if (current_medium_id != -1)
        {
            // sample t s.t. p(t) ~ exp(-sigma_t * t)
            // compute transmittance and trans_pdf
            // if t < t_hit, set scatter = True
            // ...
            const Medium &medium = scene.media[current_medium_id];
            Spectrum sigma_s = get_sigma_s(medium, ray.org);
            Spectrum sigma_a = get_sigma_a(medium, ray.org);
            Spectrum sigma_t = sigma_s + sigma_a; // get_majorant(medium, ray);

            Real u = next_pcg32_real<Real>(rng);
            Real t = -log(1 - u) / sigma_t[0];
            Real t_hit = t_hit = infinity<Real>();
            if (vertex_)
            {
                t_hit = distance(vertex.position, ray.org);
            }

            if (t < t_hit)
            {
                scatter = true;
                trans_pdf = exp(-sigma_t * t) * sigma_t;
                transmittance = exp(-sigma_t * t);
                ray.org = ray.org + t * ray.dir;
            }
            else
            {
                trans_pdf = exp(-sigma_t * t_hit);
                transmittance = exp(-sigma_t * t_hit);
                ray.org = vertex_->position + ray.dir * get_intersection_epsilon(scene);
            }
        }
        else if (vertex_)
        {
            ray.org = vertex_->position + ray.dir * get_intersection_epsilon(scene);
        }
        else
        {
            break;
        }
        multi_trans_pdf *= trans_pdf;
        current_path_throughput *= (transmittance / trans_pdf);
        if (vertex_ && !scatter && is_light(scene.shapes[vertex.shape_id]))
        {
            Spectrum Le = emission(vertex, -ray.dir, scene);

            // reach a light source, include emission
            if (never_scatter)
            {
                radiance += current_path_throughput * Le;
            }
            else
            {
                int light_id = get_area_light_id(scene.shapes[vertex.shape_id]);
                const Light &light = scene.lights[light_id];
                PointAndNormal light_point{vertex.position, vertex.geometric_normal};
                Real pdf_nee = pdf_point_on_light(light, light_point, nee_p_cache, scene);
                pdf_nee = light_pmf(scene, light_id) * pdf_nee;

                Real G = max(-dot(normalize(light_point.position - nee_p_cache), light_point.normal), Real(0)) /
                         distance_squared(vertex.position, nee_p_cache);
                Spectrum dir_pdf_ = dir_pdf * multi_trans_pdf * G;

                Vector3 w = (dir_pdf_ * dir_pdf_) / (dir_pdf_ * dir_pdf_ + pdf_nee * pdf_nee);

                radiance += current_path_throughput * Le * w;
            }
        }

        if (bounces == scene.options.max_depth - 1 && scene.options.max_depth != -1)
        {
            break;
        }
        // Pass through last medium without scattering and hit nothing, continue to go through this direction
        if (!scatter && vertex_ && vertex.material_id == -1)
        {
            current_medium_id = update_medium_id(vertex, ray, current_medium_id);
            bounces++;
            continue;
        }
        // sample next direct & update path throughput
        if (scatter)
        {
            never_scatter = false;
            // we need to calculate the nee contribution
            // nee only happens when scattering, calculate the nee contribution at this position
            // and then choose a new direction for next iteration
            Spectrum nee = next_event_estimation(scene, ray, current_medium_id, bounces, rng, nullptr, *vertex_);
            Medium medium = scene.media[current_medium_id];

            Spectrum sigma_s = get_sigma_s(medium, ray.org);
            radiance += current_path_throughput * nee * sigma_s;

            PhaseFunction phase_function = get_phase_function(medium);
            Vector2 rnd_param{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            std::optional<Spectrum> next_dir_ = sample_phase_function(phase_function, -ray.dir, rnd_param);
            // cache the ray.org
            nee_p_cache = ray.org;
            // We skip the surface hit cases, so we need to use dir_pdf to balance the throughput
            dir_pdf = pdf_sample_phase(phase_function, -ray.dir, *next_dir_);
            if (dir_pdf <= 0)
            {
                break;
            }
            Spectrum rho = eval(phase_function, -ray.dir, *next_dir_);
            current_path_throughput *= (rho / dir_pdf) * sigma_s;
            // update ray.dir
            ray.dir = *next_dir_;
            // Scattered and need to recompute the multi_trans_pdf for new path
            multi_trans_pdf = make_const_spectrum(1);
        }
        else
        {
            // Hit a surface -- don’t need to deal with this yet
            break;
        }
        Real rr_prob = 1;
        if (bounces >= scene.options.rr_depth)
        {
            rr_prob = min(max(current_path_throughput), Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob)
            {
                // Terminate the path
                break;
            }
            else
            {
                current_path_throughput /= rr_prob;
            }
        }
        bounces++;
    }
    return radiance;
}

// The fifth volumetric renderer:
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing_5(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng)
{
    // Homework 2: implememt this!
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(Real(0), Real(0));
    int current_medium_id = scene.camera.medium_id;
    Spectrum current_path_throughput = make_const_spectrum(1);
    Spectrum radiance = make_zero_spectrum();
    int bounces = 0;
    bool never_scatter = true;
    Real dir_pdf = 0;
    Vector3 nee_p_cache = Vector3{0, 0, 0};
    Spectrum multi_trans_pdf = Spectrum{1, 1, 1};
    Real eta_scale = Real(1);

    while (true)
    {
        bool scatter = false;
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        PathVertex vertex;
        if (!vertex_)
        {
            break;
        }
        vertex = *vertex_;
        Spectrum transmittance = make_const_spectrum(1);
        Spectrum trans_pdf = make_const_spectrum(1);

        if (current_medium_id != -1)
        {
            // sample t s.t. p(t) ~ exp(-sigma_t * t)
            // compute transmittance and trans_pdf
            // if t < t_hit, set scatter = True
            // ...
            const Medium &medium = scene.media[current_medium_id];
            Spectrum sigma_s = get_sigma_s(medium, ray.org);
            Spectrum sigma_a = get_sigma_a(medium, ray.org);
            Spectrum sigma_t = sigma_s + sigma_a; // get_majorant(medium, ray);

            Real u = next_pcg32_real<Real>(rng);
            Real t = -log(1 - u) / sigma_t[0];
            Real t_hit = t_hit = infinity<Real>();
            if (vertex_)
            {
                t_hit = distance(vertex.position, ray.org);
            }

            if (t < t_hit)
            {
                scatter = true;
                trans_pdf = exp(-sigma_t * t) * sigma_t;
                transmittance = exp(-sigma_t * t);
                ray.org = ray.org + t * ray.dir;
            }
            else
            {
                trans_pdf = exp(-sigma_t * t_hit);
                transmittance = exp(-sigma_t * t_hit);
                ray.org = vertex_->position + ray.dir * get_intersection_epsilon(scene);
            }
        }
        else if (vertex_)
        {
            ray.org = (*vertex_).position + ray.dir * get_intersection_epsilon(scene);
        }

        multi_trans_pdf *= trans_pdf;
        current_path_throughput *= (transmittance / trans_pdf);
        if (vertex_ && !scatter && is_light(scene.shapes[vertex.shape_id]))
        {
            Spectrum Le = emission(vertex, -ray.dir, scene);

            // reach a light source, include emission
            if (never_scatter)
            {
                radiance += current_path_throughput * Le;
            }
            else
            {
                int light_id = get_area_light_id(scene.shapes[vertex.shape_id]);
                const Light &light = scene.lights[light_id];
                PointAndNormal light_point{vertex.position, vertex.geometric_normal};
                Real pdf_nee = pdf_point_on_light(light, light_point, nee_p_cache, scene);
                pdf_nee = light_pmf(scene, light_id) * pdf_nee;

                Real G = max(-dot(normalize(light_point.position - nee_p_cache), light_point.normal), Real(0)) /
                         distance_squared(vertex.position, nee_p_cache);
                Spectrum dir_pdf_ = dir_pdf * multi_trans_pdf * G;

                Vector3 w = (dir_pdf_ * dir_pdf_) / (dir_pdf_ * dir_pdf_ + pdf_nee * pdf_nee);

                radiance += current_path_throughput * Le * w;
            }
        }

        if (bounces == scene.options.max_depth - 1 && scene.options.max_depth != -1)
        {
            break;
        }
        // Pass through last medium without scattering and hit nothing, continue to go through this direction
        if (!scatter && vertex_ && vertex.material_id == -1)
        {
            current_medium_id = update_medium_id(vertex, ray, current_medium_id);
            bounces++;
            continue;
        }
        // sample next direct & update path throughput
        if (scatter)
        {
            never_scatter = false;
            // we need to calculate the nee contribution
            // nee only happens when scattering, calculate the nee contribution at this position
            // and then choose a new direction for next iteration
            Spectrum nee = next_event_estimation(scene, ray, current_medium_id, bounces, rng, nullptr, *vertex_);
            Medium medium = scene.media[current_medium_id];

            Spectrum sigma_s = get_sigma_s(medium, ray.org);
            radiance += current_path_throughput * nee * sigma_s;

            PhaseFunction phase_function = get_phase_function(medium);
            Vector2 rnd_param{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            std::optional<Spectrum> next_dir_ = sample_phase_function(phase_function, -ray.dir, rnd_param);
            // cache the ray.org
            nee_p_cache = ray.org;
            // We skip the surface hit cases, so we need to use dir_pdf to balance the throughput
            dir_pdf = pdf_sample_phase(phase_function, -ray.dir, *next_dir_);
            if (dir_pdf <= 0)
            {
                break;
            }
            Spectrum rho = eval(phase_function, -ray.dir, *next_dir_);
            current_path_throughput *= (rho / dir_pdf) * sigma_s;
            // update ray.dir
            ray.dir = *next_dir_;
            // Scattered and need to recompute the multi_trans_pdf for new path
            multi_trans_pdf = make_const_spectrum(1);
        }
        else if (vertex_)
        {
            PathVertex vertex = *vertex_;
            nee_p_cache = vertex.position;
            never_scatter = false;
            const Material &mat = scene.materials[vertex.material_id];
            Spectrum nee = next_event_estimation(scene, ray, current_medium_id, bounces, rng, &mat, *vertex_);
            radiance += current_path_throughput * nee;
            Vector3 dir_view = -ray.dir;
            Vector2 bsdf_rnd_param_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};

            Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
            std::optional<BSDFSampleRecord> bsdf_sample_ =
                sample_bsdf(mat,
                            dir_view,
                            vertex,
                            scene.texture_pool,
                            bsdf_rnd_param_uv,
                            bsdf_rnd_param_w);
            if (!bsdf_sample_)
            {
                // BSDF sampling failed. Abort the loop.
                break;
            }
            const BSDFSampleRecord &bsdf_sample = *bsdf_sample_;
            Vector3 dir_bsdf = bsdf_sample.dir_out;
            // Update ray differentials & eta_scale
            if (bsdf_sample.eta == 0)
            {
                ray_diff.spread = reflect(ray_diff, vertex.mean_curvature, bsdf_sample.roughness);
            }
            else
            {
                ray_diff.spread = refract(ray_diff, vertex.mean_curvature, bsdf_sample.eta, bsdf_sample.roughness);
                eta_scale /= (bsdf_sample.eta * bsdf_sample.eta);
                current_medium_id = update_medium_id(vertex, ray, current_medium_id);
            }

            // Trace a ray towards bsdf_dir. Note that again we have
            // to have an "epsilon" tnear to prevent self intersection.
            Ray bsdf_ray{vertex.position, dir_bsdf, get_intersection_epsilon(scene), infinity<Real>()};

            Spectrum f = eval(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
            Real p2 = pdf_sample_bsdf(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
            if (p2 <= 0)
            {
                break;
            }
            current_path_throughput *= f / p2;
            dir_pdf = p2;
            ray = bsdf_ray;
            multi_trans_pdf = make_const_spectrum(1);
        }
        Real rr_prob = 1;
        if (bounces >= scene.options.rr_depth)
        {
            rr_prob = min(max((1 / eta_scale) * current_path_throughput), Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob)
            {
                // Terminate the path
                break;
            }
            else
            {
                current_path_throughput /= rr_prob;
            }
        }
        bounces++;
    }
    return radiance;
}

// The final volumetric renderer:
// multiple chromatic heterogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing(const Scene &scene,
                          int x, int y, /* pixel coordinates */
                          pcg32_state &rng)
{
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(Real(0), Real(0));
    int current_medium_id = scene.camera.medium_id;
    Spectrum current_path_throughput = make_const_spectrum(1);
    Spectrum radiance = make_zero_spectrum();
    int bounces = 0;
    bool never_scatter = true;
    Real dir_pdf = 0;
    Vector3 nee_p_cache = Vector3{0, 0, 0};
    Spectrum multi_trans_pdf = make_const_spectrum(1);
    Real eta_scale = Real(1);

    while (true)
    {

        bool scatter = false;
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        Spectrum transmittance = make_const_spectrum(1);
        Spectrum trans_dir_pdf = make_const_spectrum(1);
        Spectrum trans_nee_pdf = make_const_spectrum(1);
        Real t_hit = infinity<Real>();
        if (vertex_)
        {
            t_hit = distance(vertex_->position, ray.org);
        }
        if (current_medium_id != -1)
        {

            Real u = next_pcg32_real<Real>(rng);
            int channel = std::clamp(int(u * 3), 0, 2);
            Real accum_t = 0;
            int iteration = 0;
            Spectrum majorant = get_majorant(scene.media[current_medium_id], ray);

            while (true)
            {
                if (majorant[channel] <= 0 || iteration >= scene.options.max_null_collisions)
                {
                    break;
                }
                Real t = -log(1 - next_pcg32_real<Real>(rng)) / majorant[channel];
                Real dt = t_hit - accum_t;
                accum_t = min(accum_t + t, t_hit);
                if (t < dt)
                { // haven’t reached the surface, sample from real/fake particle events
                    Spectrum sigma_a = get_sigma_a(scene.media[current_medium_id], ray.org + accum_t * ray.dir);
                    Spectrum sigma_s = get_sigma_s(scene.media[current_medium_id], ray.org + accum_t * ray.dir);
                    Spectrum sigma_t = sigma_a + sigma_s;
                    Spectrum sigma_n = majorant - sigma_t;
                    Spectrum real_prob = sigma_t / majorant;
                    if (next_pcg32_real<Real>(rng) < real_prob[channel])
                    {
                        // hit a "real" particle
                        scatter = true;
                        never_scatter = false;
                        ray.org = ray.org + accum_t * ray.dir;
                        transmittance *= exp(-majorant * t) / max(majorant);
                        trans_dir_pdf *= exp(-majorant * t) * majorant * real_prob / max(majorant);
                        // don’t need to account for trans_nee_pdf since we scatter
                        break;
                    }
                    else
                    {
                        // hit a "fake" particle
                        transmittance *= exp(-majorant * t) * sigma_n / max(majorant);
                        trans_dir_pdf *= exp(-majorant * t) * majorant * (1 - real_prob) / max(majorant);
                        trans_nee_pdf *= exp(-majorant * t) * majorant / max(majorant);
                    }
                }
                else
                {
                    // reach the surface
                    // before going forward the approximation virtual distance, the ray already hit something real
                    ray.org = ray.org + t_hit * ray.dir;
                    transmittance *= exp(-majorant * dt);
                    trans_dir_pdf *= exp(-majorant * dt);
                    trans_nee_pdf *= exp(-majorant * dt);
                    break;
                }
                iteration += 1;
            }
        }
        else if (vertex_)
        {
            ray.org = vertex_->position;
        }
        multi_trans_pdf *= trans_dir_pdf;
        current_path_throughput *= (transmittance / avg(trans_dir_pdf));
        // If we reach a surface and didn’t scatter, include the emission.
        if (!scatter && vertex_ && is_light(scene.shapes[vertex_->shape_id]))
        {
            if (never_scatter)
            {
                // This is the only way we can see the light source, so
                // we don’t need multiple importance sampling.
                radiance += current_path_throughput * emission(*vertex_, -ray.dir, scene);
            }
            else
            {
                // Need to account for next event estimation
                PointAndNormal light_point{vertex_->position, vertex_->geometric_normal};
                // Note that pdf_nee needs to account for the path vertex that issued
                // next event estimation potentially many bounces ago.
                // The vertex position is stored in nee_p_cache.
                int light_id = get_area_light_id(scene.shapes[vertex_->shape_id]);
                Spectrum pdf_nee = trans_nee_pdf * pdf_point_on_light(scene.lights[light_id], light_point, nee_p_cache, scene);
                // The PDF for sampling the light source using phase function sampling + transmittance sampling
                // The directional sampling pdf was cached in dir_pdf in solid angle measure.
                // The transmittance sampling pdf was cached in multi_trans_pdf.
                Real G = max(-dot(ray.dir, light_point.normal), Real(0)) / distance_squared(light_point.position, nee_p_cache);
                Spectrum dir_pdf_ = dir_pdf * multi_trans_pdf * G;
                Spectrum w = (dir_pdf_ * dir_pdf_) / (dir_pdf_ * dir_pdf_ + pdf_nee * pdf_nee);
                radiance += current_path_throughput * emission(*vertex_, -ray.dir, scene) * w;
            }
        }

        if ((bounces >= scene.options.max_depth - 1) && (scene.options.max_depth != -1))
        {
            break;
        }
        if (!scatter && vertex_ && vertex_->material_id == -1)
        {
            // index-matching interface, skip through it
            //  Sometimes we will hit surfaces that have no materials assigned,
            // For these surfaces, we need to pass through them.
            // Passing through an index-matched surface counts as one bounce.
            current_medium_id = update_medium_id(*vertex_, ray, current_medium_id);
            ray = Ray{(*vertex_).position, ray.dir, get_intersection_epsilon(scene), infinity<Real>()};
            bounces += 1;
            continue;
        }
        // sample next direct & update path throughput
        if (scatter)
        {
            never_scatter = false;
            nee_p_cache = ray.org;
            PhaseFunction phase = get_phase_function(scene.media[current_medium_id]);
            Vector2 rand_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            std::optional<Vector3> next_dir_ = sample_phase_function(phase, -ray.dir, rand_uv);
            // put next_event_estimation() in our previous code and include its contribution. Remember
            // to multiply the result of next event estimation with the transmittance from the previous path vertex to p
            // and σs(p).
            Spectrum nee = next_event_estimation_final(scene, ray, current_medium_id, bounces, rng, nullptr, *vertex_);
            Spectrum sigma_s = get_sigma_s(scene.media[current_medium_id], ray.org);
            radiance += current_path_throughput * nee * sigma_s;

            // update ray.dir
            dir_pdf = pdf_sample_phase(phase, -ray.dir, *next_dir_);
            if (dir_pdf <= 0)
            {
                break;
            }
            current_path_throughput *= (eval(phase, -ray.dir, *next_dir_) / dir_pdf) * sigma_s;
            ray.dir = *next_dir_;
            multi_trans_pdf = make_const_spectrum(1);
        }
        else if (vertex_)
        {
            // consider the case where we hit a surface and include the BSDF sampling and evaluation.
            nee_p_cache = vertex_->position;
            never_scatter = false;
            Material mat = scene.materials[vertex_->material_id];

            Spectrum nee = next_event_estimation_final(scene, ray, current_medium_id, bounces, rng, &mat, *vertex_);
            radiance += current_path_throughput * nee;

            PathVertex vertex = *vertex_;
            Vector3 dir_view = -ray.dir;
            Vector2 bsdf_rnd_param_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
            std::optional<BSDFSampleRecord> bsdf_sample_ =
                sample_bsdf(mat,
                            dir_view,
                            vertex,
                            scene.texture_pool,
                            bsdf_rnd_param_uv,
                            bsdf_rnd_param_w);
            if (!bsdf_sample_)
            {
                // BSDF sampling failed. Abort the loop.
                break;
            }
            const BSDFSampleRecord &bsdf_sample = *bsdf_sample_;
            Vector3 dir_bsdf = bsdf_sample.dir_out;
            // Update ray differentials & eta_scale
            if (bsdf_sample.eta == 0)
            {
                ray_diff.spread = reflect(ray_diff, vertex.mean_curvature, bsdf_sample.roughness);
            }
            else
            {
                ray_diff.spread = refract(ray_diff, vertex.mean_curvature, bsdf_sample.eta, bsdf_sample.roughness);
                eta_scale /= (bsdf_sample.eta * bsdf_sample.eta);
                current_medium_id = update_medium_id(vertex, ray, current_medium_id);
            }
            // Trace a ray towards bsdf_dir. Note that again we have
            // to have an "epsilon" tnear to prevent self intersection.
            dir_pdf = pdf_sample_bsdf(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
            if (dir_pdf <= 0)
            {
                // Numerical issue -- we generated some invalid rays.
                break;
            }
            Spectrum bsdf_f = eval(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
            current_path_throughput *= bsdf_f / dir_pdf;
            ray.dir = dir_bsdf;
            multi_trans_pdf = make_const_spectrum(1);
        }
        Real rr_prob = 1;
        if (bounces >= scene.options.rr_depth)
        {
            rr_prob = min(max((1 / eta_scale) * current_path_throughput), (Real)0.95);
            if (next_pcg32_real<Real>(rng) > rr_prob)
            {
                break;
            }
            else
            {
                current_path_throughput /= rr_prob;
            }
        }
        bounces += 1;
    }
    return radiance;
}