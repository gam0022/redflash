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

#include <optixu/optixu_math_namespace.h>
#include <common.h>
#include "redflash.h"
#include "random.h"

using namespace optix;

struct PerRayData_pathtrace
{
    float3 radiance;
    float3 attenuation;

    float3 origin;
    float3 direction;

    float pdf;
    float3 wo;

    unsigned int seed;
    int depth;
    int done;
};

struct PerRayData_pathtrace_shadow
{
    bool inShadow;
};

// Scene wide variables
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );



//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  sample_at_once, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );
rtDeclareVariable(unsigned int, max_depth, , );

rtBuffer<float4, 2>              output_buffer;
//rtBuffer<ParallelogramLight>     lights;

rtDeclareVariable(int, sysNumberOfLights, , );
rtBuffer<LightParameter> sysLightParameters;


RT_PROGRAM void pathtrace_camera()
{
    size_t2 screen = output_buffer.size();
    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number);
    float3 result = make_float3(0.0f);

    for(int i = 0; i < sample_at_once; i++)
    {
        float2 subpixel_jitter = frame_number == 0 ? make_float2(0.0f) : make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);
        float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
        float3 ray_origin = eye;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);

        // Initialze per-ray data
        PerRayData_pathtrace prd;
        prd.radiance = make_float3(0.0f);
        prd.attenuation = make_float3(1.0f);
        prd.done = false;
        prd.seed = seed;
        prd.depth = 0;

        // Each iteration is a segment of the ray path.  The closest hit will
        // return new segments to be traced here.
        for(;;)
        {
            Ray ray = make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
            prd.wo = -ray.direction;
            rtTrace(top_object, ray, prd);

            if (prd.done || prd.depth >= max_depth)
            {
                break;
            }

            // Russian roulette termination 
            /*if(prd.depth >= rr_begin_depth)
            {
                float pcont = fmaxf(prd.attenuation);
                if(rnd(prd.seed) >= pcont)
                    break;
                prd.attenuation /= pcont;
            }*/

            prd.depth++;

            // Update ray data for the next path segment
            ray_origin = prd.origin;
            ray_direction = prd.direction;
        }

        result += prd.radiance;
        seed = prd.seed;
    }

    //
    // Update the output buffer
    //
    float3 pixel_color = result / (float)sample_at_once;

    if (frame_number > 1)
    {
        float a = 1.0f / (float)frame_number;
        float3 old_color = make_float3(output_buffer[launch_index]);
        output_buffer[launch_index] = make_float4( lerp( old_color, pixel_color, a ), 1.0f );
    }
    else
    {
        output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
    }
}


//-----------------------------------------------------------------------------
//
//  closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, emission_color, , );
rtDeclareVariable(float3, albedo_color, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

RT_PROGRAM void light_closest_hit()
{
    current_prd.radiance += emission_color * current_prd.attenuation;
    current_prd.done = true;
}


//-----------------------------------------------------------------------------
//
//  bsdf
//
//-----------------------------------------------------------------------------

RT_CALLABLE_PROGRAM void diffuse_Pdf(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 n = state.ffnormal;
    float3 L = prd.direction;

    float pdfDiff = abs(dot(L, n))* (1.0f / M_PIf);

    prd.pdf = pdfDiff;

}

RT_CALLABLE_PROGRAM void diffuse_Sample(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 N = state.ffnormal;

    float3 dir;

    float r1 = rnd(prd.seed);
    float r2 = rnd(prd.seed);

    optix::Onb onb(N);

    cosine_sample_hemisphere(r1, r2, dir);
    onb.inverse_transform(dir);

    prd.direction = dir;
}


RT_CALLABLE_PROGRAM float3 diffuse_Eval(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 N = state.ffnormal;
    float3 V = prd.wo;
    float3 L = prd.direction;

    float NDotL = dot(N, L);
    float NDotV = dot(N, V);
    if (NDotL <= 0.0f || NDotV <= 0.0f) return make_float3(0.0f);

    float3 out = (1.0f / M_PIf) * mat.albedo;

    return out * clamp(dot(N, L), 0.0f, 1.0f);
}

RT_FUNCTION float sqr(float x) { return x * x; }

RT_FUNCTION float SchlickFresnel(float u)
{
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2*m; // pow(m,5)
}

RT_FUNCTION float GTR1(float NDotH, float a)
{
    if (a >= 1.0f) return (1.0f / M_PIf);
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f)*NDotH*NDotH;
    return (a2 - 1.0f) / (M_PIf*logf(a2)*t);
}

RT_FUNCTION float GTR2(float NDotH, float a)
{
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f)*NDotH*NDotH;
    return a2 / (M_PIf * t*t);
}

RT_FUNCTION float smithG_GGX(float NDotv, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NDotv * NDotv;
    return 1.0f / (NDotv + sqrtf(a + b - a * b));
}


/*
    http://simon-kallweit.me/rendercompo2015/
*/
RT_CALLABLE_PROGRAM void Pdf(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 n = state.ffnormal;
    float3 V = prd.wo;
    float3 L = prd.direction;

    float specularAlpha = max(0.001f, mat.roughness);
    float clearcoatAlpha = lerp(0.1f, 0.001f, mat.clearcoatGloss);

    float diffuseRatio = 0.5f * (1.f - mat.metallic);
    float specularRatio = 1.f - diffuseRatio;

    float3 half = normalize(L + V);

    float cosTheta = abs(dot(half, n));
    float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
    float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

    // calculate diffuse and specular pdfs and mix ratio
    float ratio = 1.0f / (1.0f + mat.clearcoat);
    float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * abs(dot(L, half)));
    float pdfDiff = abs(dot(L, n))* (1.0f / M_PIf);

    // weight pdfs according to ratios
    prd.pdf = diffuseRatio * pdfDiff + specularRatio * pdfSpec;
}

/*
    https://learnopengl.com/PBR/IBL/Specular-IBL
*/

RT_CALLABLE_PROGRAM void Sample(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 N = state.ffnormal;
    float3 V = prd.wo;

    float3 dir;

    float probability = rnd(prd.seed);
    float diffuseRatio = 0.5f * (1.0f - mat.metallic);

    float r1 = rnd(prd.seed);
    float r2 = rnd(prd.seed);

    optix::Onb onb(N); // basis

    if (probability < diffuseRatio) // sample diffuse
    {
        cosine_sample_hemisphere(r1, r2, dir);
        onb.inverse_transform(dir);
    }
    else
    {
        float a = max(0.001f, mat.roughness);

        float phi = r1 * 2.0f * M_PIf;

        float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a*a - 1.0f) *r2));
        float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
        float sinPhi = sinf(phi);
        float cosPhi = cosf(phi);

        float3 half = make_float3(sinTheta*cosPhi, sinTheta*sinPhi, cosTheta);
        onb.inverse_transform(half);

        dir = 2.0f*dot(V, half)*half - V; //reflection vector

    }
    prd.direction = dir;
}


RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 N = state.ffnormal;
    float3 V = prd.wo;
    float3 L = prd.direction;

    float NDotL = dot(N, L);
    float NDotV = dot(N, V);
    if (NDotL <= 0.0f || NDotV <= 0.0f) return make_float3(0.0f);

    float3 H = normalize(L + V);
    float NDotH = dot(N, H);
    float LDotH = dot(L, H);

    float3 Cdlin = mat.albedo;
    float Cdlum = 0.3f*Cdlin.x + 0.6f*Cdlin.y + 0.1f*Cdlin.z; // luminance approx.

    float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
    float3 Cspec0 = lerp(mat.specular*0.08f*lerp(make_float3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);
    float3 Csheen = lerp(make_float3(1.0f), Ctint, mat.sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
    float Fd90 = 0.5f + 2.0f * LDotH*LDotH * mat.roughness;
    float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotrokPic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LDotH * LDotH*mat.roughness;
    float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.0f / (NDotL + NDotV) - 0.5f) + 0.5f);

    // specular
    //float aspect = sqrt(1-mat.anisotrokPic*.9);
    //float ax = Max(.001f, sqr(mat.roughness)/aspect);
    //float ay = Max(.001f, sqr(mat.roughness)*aspect);
    //float Ds = GTR2_aniso(NDotH, Dot(H, X), Dot(H, Y), ax, ay);

    float a = max(0.001f, mat.roughness);
    float Ds = GTR2(NDotH, a);
    float FH = SchlickFresnel(LDotH);
    float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
    float roughg = sqr(mat.roughness*0.5f + 0.5f);
    float Gs = smithG_GGX(NDotL, roughg) * smithG_GGX(NDotV, roughg);

    // sheen
    float3 Fsheen = FH * mat.sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, mat.clearcoatGloss));
    float Fr = lerp(0.04f, 1.0f, FH);
    float Gr = smithG_GGX(NDotL, 0.25f) * smithG_GGX(NDotV, 0.25f);

    float3 out = ((1.0f / M_PIf) * lerp(Fd, ss, mat.subsurface)*Cdlin + Fsheen)
        * (1.0f - mat.metallic)
        + Gs * Fs*Ds + 0.25f*mat.clearcoat*Gr*Fr*Dr;

    return out * clamp(dot(N, L), 0.0f, 1.0f);
}

static __host__ __device__ __inline__ float powerHeuristic(float a, float b)
{
    float t = a * a;
    return t / (b*b + t);
}

RT_FUNCTION float3 UniformSampleSphere(float u1, float u2)
{
    float z = 1.f - 2.f * u1;
    float r = sqrtf(max(0.f, 1.f - z * z));
    float phi = 2.f * M_PIf * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);

    return make_float3(x, y, z);
}

RT_CALLABLE_PROGRAM void sphere_sample(LightParameter &light, PerRayData_pathtrace &prd, LightSample &sample)
{
    const float r1 = rnd(prd.seed);
    const float r2 = rnd(prd.seed);
    sample.surfacePos = light.position + UniformSampleSphere(r1, r2) * light.radius;
    sample.normal = normalize(sample.surfacePos - light.position);
    sample.emission = light.emission * sysNumberOfLights;
}

RT_FUNCTION float3 DirectLight(MaterialParameter &mat, State &state)
{
    float3 L = make_float3(0.0f);

    //Pick a light to sample
    int index = optix::clamp(static_cast<int>(floorf(rnd(current_prd.seed) * sysNumberOfLights)), 0, sysNumberOfLights - 1);
    LightParameter light = sysLightParameters[index];
    LightSample lightSample;

    // float3 surfacePos = state.fhp;
    float3 surfacePos = state.hitpoint;
    float3 surfaceNormal = state.ffnormal;

    // sysLightSample[light.lightType](light, current_prd, lightSample);
    sphere_sample(light, current_prd, lightSample);

    float3 lightDir = lightSample.surfacePos - surfacePos;
    float lightDist = length(lightDir);
    float lightDistSq = lightDist * lightDist;
    lightDir /= sqrtf(lightDistSq);

    if (dot(lightDir, surfaceNormal) <= 0.0f || dot(lightDir, lightSample.normal) >= 0.0f)
        return L;

    PerRayData_pathtrace_shadow prd_shadow;
    prd_shadow.inShadow = false;
    optix::Ray shadowRay = optix::make_Ray(surfacePos, lightDir, 1, scene_epsilon, lightDist - scene_epsilon);
    rtTrace(top_object, shadowRay, prd_shadow);

    if (!prd_shadow.inShadow)
    {
        float NdotL = dot(lightSample.normal, -lightDir);
        if (NdotL > 0.0f)
        {
            float lightPdf = lightDistSq / (light.area * NdotL);
            current_prd.direction = lightDir;

            // sysBRDFPdf[programId](mat, state, current_prd);
            Pdf(mat, state, current_prd);
            // float3 f = sysBRDFEval[programId](mat, state, current_prd);
            float3 f = Eval(mat, state, current_prd);

            // if (NdotL > 0.0f)
            L = powerHeuristic(lightPdf, current_prd.pdf) * current_prd.attenuation * f * lightSample.emission / max(0.001f, lightPdf);
        }
    }

    return L;
}

RT_PROGRAM void closest_hit()
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    float3 hitpoint = ray.origin + t_hit * ray.direction + ffnormal * scene_epsilon;

    State state;
    state.hitpoint = hitpoint;
    state.normal = world_shading_normal;
    state.ffnormal = ffnormal;

    current_prd.wo = -ray.direction;

    // FIXME: Sample‚É‚à‚Á‚Ä‚¢‚­
    current_prd.origin = hitpoint;
    
    current_prd.radiance += emission_color * current_prd.attenuation;

    MaterialParameter mat;
    mat.albedo = albedo_color;
    mat.metallic = 0.8f;
    mat.roughness = 0.05f;

    // Direct light Sampling
    if (/*!prd.specularBounce && */ current_prd.depth < max_depth)
    {
        current_prd.radiance += DirectLight(mat, state);
    }

    // BRDF Sampling
    Sample(mat, state, current_prd);
    Pdf(mat, state, current_prd);
    float3 f = Eval(mat, state, current_prd);

    if (current_prd.pdf > 0.0f)
    {
        current_prd.attenuation *= f / current_prd.pdf;
    }
    else
    {
        current_prd.done = true;
    }
}


//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow()
{
    current_prd_shadow.inShadow = true;
    rtTerminateRay();
}


//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_float4(bad_color, 1.0f);
}


//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void envmap_miss()
{
    float theta = atan2f(ray.direction.x, ray.direction.z);
    float phi = M_PIf * 0.5f - acosf(ray.direction.y);
    float u = (theta + M_PIf) * (0.5f * M_1_PIf);
    float v = 0.5f * (1.0f + sin(phi));
    current_prd.radiance += make_float3(tex2D(envmap, u, v)) * current_prd.attenuation;
    current_prd.done = true;
}