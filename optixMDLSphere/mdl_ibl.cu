/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "random.h"
#include <mi/neuraylib/target_code_types.h>
#include "shared_structs.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace optix;

// ray and geometry
rtDeclareVariable(optix::Ray,          ray,              rtCurrentRay, );
rtDeclareVariable(float3,              shading_normal,   attribute shading_normal, );
rtDeclareVariable(float3,              geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3,              shading_tangent,  attribute shading_tangent, );
rtDeclareVariable(float3,              texcoord,         attribute texcoord, );
rtDeclareVariable(PerRayData_radiance, prd_radiance,     rtPayload, );

rtDeclareVariable(int, mdl_test_type, , );


// environment and environment light importance sampling
rtTextureSampler<float4, 2> env_texture;
rtBuffer<Env_accel, 1>      env_accel;
rtDeclareVariable(uint2, env_size, , );

// MDL BSDF programs
rtDeclareVariable( rtCallableProgramId<mi::neuraylib::Bsdf_init_function>,     mdl_bsdf_init, , );
rtDeclareVariable( rtCallableProgramId<mi::neuraylib::Bsdf_sample_function>,   mdl_bsdf_sample, , );
rtDeclareVariable( rtCallableProgramId<mi::neuraylib::Bsdf_evaluate_function>, mdl_bsdf_evaluate, , );
rtDeclareVariable( rtCallableProgramId<mi::neuraylib::Bsdf_pdf_function>,      mdl_bsdf_pdf, , );


__device__ const float identity[16] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f };

__device__ const mi::neuraylib::Resource_data res_data = {
    NULL,
    NULL
};


// direction to environment map coordinates
__device__ __inline__ float2 environment_coords(const float3 &dir)
{
    const float u = atan2f(dir.z, dir.x) * (float)(0.5 / M_PI) + 0.5f;
    const float v = acosf(fmax(fminf(dir.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI);
    return make_float2(u, v);
}

// importance sample the environment
__device__ __inline__ float3 environment_sample(
    float3 &dir,
    float  &pdf,
    const  float3 xi)
{
    // importance sample an envmap pixel using an alias map
    const unsigned int size = env_size.x * env_size.y;
    const unsigned int idx = min((unsigned int)(xi.x * (float)size), size - 1);
    unsigned int env_idx;
    float xi_y = xi.y;
    if (xi_y < env_accel[idx].q) {
        env_idx = idx ;
        xi_y /= env_accel[idx].q;

    } else {
        env_idx = env_accel[idx].alias;
        xi_y = (xi_y - env_accel[idx].q) / (1.0f - env_accel[idx].q);
    }

    const unsigned int py = env_idx / env_size.x;
    const unsigned int px = env_idx % env_size.x;
    pdf = env_accel[env_idx].pdf;

    // uniformly sample spherical area of pixel
    const float u = (float)(px + xi_y) / (float)env_size.x;
    const float phi = u * (float)(2.0 * M_PI) - (float)M_PI;
    float sin_phi, cos_phi;
    sincosf(phi, &sin_phi, &cos_phi);
    const float step_theta = (float)M_PI / (float)env_size.y;
    const float theta0 = (float)(py) * step_theta;
    const float cos_theta = cosf(theta0) * (1.0f - xi.z) + cosf(theta0 + step_theta) * xi.z;
    const float theta = acosf(cos_theta);
    const float sin_theta = sinf(theta);
    dir = make_float3(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);

    // lookup filtered value
    const float v = theta * (float)(1.0 / M_PI);
    const float4 t = tex2D(env_texture, u, v);
    return make_float3(t.x, t.y, t.z) / pdf;
}

// evaluate the environment
__device__ __inline__ float3 environment_eval(
    float &pdf,
    const float3 &dir)
{
    const float2 uv = environment_coords(dir);
    const unsigned int x = min((unsigned int)(uv.x * (float)env_size.x), env_size.x - 1);
    const unsigned int y = min((unsigned int)(uv.y * (float)env_size.y), env_size.y - 1);

    pdf = env_accel[y * env_size.x + x].pdf;
    const float4 t = tex2D(env_texture, uv.x, uv.y);
    return make_float3(t.x, t.y, t.z);
}


// computes direct lighting using BSDF and environment importance sampling (combined with MIS)
RT_PROGRAM void closest_hit_radiance()
{
    const float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 tangent_u = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_tangent));
    const float3 tangent_v = cross(normal, tangent_u);

    float3 text_coords = texcoord;      // taking an address of an attribute is not supported

    prd_radiance.result = make_float3(0.0f);

    // setup state
    float4 texture_results[16];
    mi::neuraylib::Shading_state_material state;
    state.normal = normal;
    state.geom_normal = normal;
    state.position = normal;
    state.animation_time = 0.0f;
    state.text_coords = &text_coords;
    state.tangent_u = &tangent_u;
    state.tangent_v = &tangent_v;
    state.text_results = texture_results;
    state.ro_data_segment = NULL;
    state.world_to_object = (float4 *)&identity;
    state.object_to_world = (float4 *)&identity;
    state.object_id = 0;

    mdl_bsdf_init(&state, &res_data, NULL, NULL);

    // put the BSDF data structs into a union to reduce number of memory writes
    union
    {
        mi::neuraylib::Bsdf_sample_data sample;
        mi::neuraylib::Bsdf_evaluate_data evaluate;
        mi::neuraylib::Bsdf_pdf_data pdf;
    } data;

    const float3 dir_out = normalize(-ray.direction);

    // set fields common among the union structs
    data.sample.ior1 = make_float3(1.0f);
    data.sample.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
    data.sample.k1 = dir_out;

    // importance sample environment light
    if (mdl_test_type != MDL_TEST_SAMPLE)
    {
        const float xi0 = rnd(prd_radiance.rnd_seed);
        const float xi1 = rnd(prd_radiance.rnd_seed);
        const float xi2 = rnd(prd_radiance.rnd_seed);

        float pdf;
        float3 dir;
        const float3 f = environment_sample(dir, pdf, make_float3(xi0, xi1, xi2));

        const float cos_theta = dot(dir, normal);
        if (cos_theta > 0.0f && pdf > 0.0f) {
            data.evaluate.k2 = dir;

            mdl_bsdf_evaluate(&data.evaluate, &state, &res_data, NULL, NULL);

            const float mis_weight = (mdl_test_type == MDL_TEST_EVAL)
                ? 1.0f : pdf / (pdf + data.evaluate.pdf);
            prd_radiance.result += f * data.evaluate.bsdf * mis_weight;
        }
    }

    // importance sample BSDF
    if (mdl_test_type != MDL_TEST_EVAL)
    {
        const float xi0 = rnd(prd_radiance.rnd_seed);
        const float xi1 = rnd(prd_radiance.rnd_seed);
        const float xi2 = rnd(prd_radiance.rnd_seed);

        data.sample.xi = make_float3(xi0, xi1, xi2);

        mdl_bsdf_sample(&data.sample, &state, &res_data, NULL, NULL);

        if (data.sample.event_type != mi::neuraylib::BSDF_EVENT_ABSORB)
        {
            const float3 bsdf_over_pdf = data.sample.bsdf_over_pdf;

            float pdf_env;
            const float3 f = environment_eval(pdf_env, data.sample.k2);

            float pdf_bsdf;
            if (mdl_test_type == MDL_TEST_MIS_PDF) {
                const float3 k2 = data.sample.k2;
                data.pdf.k2 = k2;
                mdl_bsdf_pdf(&data.pdf, &state, &res_data, NULL, NULL);
                pdf_bsdf = data.pdf.pdf;
            }
            else pdf_bsdf = data.sample.pdf;

            const bool is_specular =
                (data.sample.event_type & mi::neuraylib::BSDF_EVENT_SPECULAR) != 0;
            if (is_specular || pdf_bsdf > 0.0f) {
                const float mis_weight = is_specular ||
                    (mdl_test_type == MDL_TEST_SAMPLE) ? 1.0f : pdf_bsdf / (pdf_env + pdf_bsdf);
                prd_radiance.result += f * bsdf_over_pdf * mis_weight;
            }
        }
    }

    // make sure we don't return infinite values
    if (!isfinite(prd_radiance.result.x) ||
        !isfinite(prd_radiance.result.y) ||
        !isfinite(prd_radiance.result.z))
    {
        prd_radiance.result = make_float3(0);
    }
}


RT_PROGRAM void miss()
{
    const float2 uv = environment_coords(ray.direction);
    const float4 t = tex2D(env_texture, uv.x,  uv.y);
    prd_radiance.result = make_float3(t.x, t.y, t.z);
}


