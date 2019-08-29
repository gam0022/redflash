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
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<ParallelogramLight>     lights;


RT_PROGRAM void pathtrace_camera()
{
    size_t2 screen = output_buffer.size();

    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

    float2 jitter_scale = inv_screen / sqrt_num_samples;
    unsigned int samples_per_pixel = sqrt_num_samples*sqrt_num_samples;
    float3 result = make_float3(0.0f);

    unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame_number);
    do 
    {
        //
        // Sample pixel using jittering
        //
        unsigned int x = samples_per_pixel%sqrt_num_samples;
        unsigned int y = samples_per_pixel/sqrt_num_samples;
        float2 jitter = make_float2(x-rnd(seed), y-rnd(seed));
        float2 d = pixel + jitter*jitter_scale;
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
        for(int i = 0; i < 10; i++)
        {
            Ray ray = make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
            rtTrace(top_object, ray, prd);

            if(prd.done)
            {
                // We have hit the background or a luminaire
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
    } while (--samples_per_pixel);

    //
    // Update the output buffer
    //
    float3 pixel_color = result/(sqrt_num_samples*sqrt_num_samples);

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
//  Emissive surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        emission_color, , );

RT_PROGRAM void light_closest_hit()
{
    current_prd.radiance += emission_color * current_prd.attenuation;
    current_prd.done = true;
}


//-----------------------------------------------------------------------------
//
//  Lambertian surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,     diffuse_color, , );
rtDeclareVariable(float3,     geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3,     shading_normal,   attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray,              rtCurrentRay, );
rtDeclareVariable(float,      t_hit,            rtIntersectionDistance, );


/*RT_CALLABLE_PROGRAM void Pdf(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 n = state.ffnormal;
    float3 L = prd.direction;

    float pdfDiff = abs(dot(L, n))* (1.0f / M_PIf);

    prd.pdf = pdfDiff;

}

RT_CALLABLE_PROGRAM void Sample(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
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


RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 N = state.ffnormal;
    float3 V = prd.wo;
    float3 L = prd.direction;

    float NDotL = dot(N, L);
    float NDotV = dot(N, V);
    if (NDotL <= 0.0f || NDotV <= 0.0f) return make_float3(0.0f);

    float3 out = (1.0f / M_PIf) * mat.albedo;

    return out * clamp(dot(N, L), 0.0f, 1.0f);
}*/


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

/*RT_FUNCTION float3 DirectLight(MaterialParameter &mat, State &state)
{
    float3 L = make_float3(0.0f);

    //Pick a light to sample
    int index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * sysNumberOfLights)), 0, sysNumberOfLights - 1);
    LightParameter light = sysLightParameters[index];
    LightSample lightSample;

    float3 surfacePos = state.fhp;
    float3 surfaceNormal = state.ffnormal;

    sysLightSample[light.lightType](light, prd, lightSample);

    float3 lightDir = lightSample.surfacePos - surfacePos;
    float lightDist = length(lightDir);
    float lightDistSq = lightDist * lightDist;
    lightDir /= sqrtf(lightDistSq);

    if (dot(lightDir, surfaceNormal) <= 0.0f || dot(lightDir, lightSample.normal) >= 0.0f)
        return L;

    PerRayData_shadow prd_shadow;
    prd_shadow.inShadow = false;
    optix::Ray shadowRay = optix::make_Ray(surfacePos, lightDir, 1, scene_epsilon, lightDist - scene_epsilon);
    rtTrace(top_object, shadowRay, prd_shadow);

    if (!prd_shadow.inShadow)
    {
        float NdotL = dot(lightSample.normal, -lightDir);
        float lightPdf = lightDistSq / (light.area * NdotL);

        prd.direction = lightDir;

        sysBRDFPdf[programId](mat, state, prd);
        float3 f = sysBRDFEval[programId](mat, state, prd);

        L = powerHeuristic(lightPdf, prd.pdf) * prd.throughput * f * lightSample.emission / max(0.001f, lightPdf);
    }

    return L;
}*/

float3 DirectLightParallelogram(MaterialParameter &mat, State &state)
{
    unsigned int num_lights = lights.size();
    float3 L = make_float3(0.0f);
    float3 surfaceNormal = state.ffnormal;

    for(int i = 0; i < num_lights; ++i)
    {
        // Choose random point on light
        ParallelogramLight lightSample = lights[i];
        const float z1 = rnd(current_prd.seed);
        const float z2 = rnd(current_prd.seed);
        const float3 light_pos = lightSample.corner + lightSample.v1 * z1 + lightSample.v2 * z2;

        // Calculate properties of light sample (for area based pdf)
        float3 lightDir = light_pos - current_prd.origin;
        float lightDist = length(lightDir);
        float lightDistSq = lightDist * lightDist;
        lightDir /= sqrtf(lightDistSq);

        // cast shadow ray
        if (dot(lightDir, surfaceNormal) >= 0.0f && dot(lightDir, lightSample.normal) <= 0.0f)
        {
            PerRayData_pathtrace_shadow shadow_prd;
            shadow_prd.inShadow = false;
            // Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
            Ray shadow_ray = make_Ray(current_prd.origin, lightDir, SHADOW_RAY_TYPE, scene_epsilon, lightDist - scene_epsilon);
            rtTrace(top_object, shadow_ray, shadow_prd);

            if(!shadow_prd.inShadow)
            {
                const float A = length(cross(lightSample.v1, lightSample.v2));
                float NdotL = dot(lightSample.normal, -lightDir);
                float lightPdf = lightDistSq / (A * NdotL);

                current_prd.direction = lightDir;

                Pdf/*sysBRDFPdf[programId]*/(mat, state, current_prd);
                float3 f = Eval/*sysBRDFEval[programId]*/(mat, state, current_prd);

                L = powerHeuristic(lightPdf, current_prd.pdf) * current_prd.attenuation * f * lightSample.emission / max(0.001f, lightPdf);
            }
        }
    }

    return L;
}


RT_PROGRAM void closest_hit()
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    float3 hitpoint = ray.origin + t_hit * ray.direction + world_geometric_normal * scene_epsilon * 100.0;

    State state;
    state.normal = world_shading_normal;
    state.ffnormal = ffnormal;

    current_prd.wo = -ray.direction;
    current_prd.origin = hitpoint;
    
    current_prd.radiance += emission_color * current_prd.attenuation;

    MaterialParameter mat;
    mat.albedo = diffuse_color;
    mat.metallic = 1.0f;
    mat.roughness = 0.0f;

    // Direct light Sampling
    if (true/*!prd.specularBounce && prd.depth < max_depth*/)
    {
        current_prd.radiance += DirectLightParallelogram(mat, state);
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
    current_prd.radiance = make_float3(tex2D(envmap, u, v));
    current_prd.done = true;
}


//-----------------------------------------------------------------------------
//
//  Raymarching
//
//-----------------------------------------------------------------------------

#include <optix_world.h>

rtDeclareVariable(float3, center, , );
rtDeclareVariable(float3, size, , );
rtDeclareVariable(int, lgt_instance, , ) = {0};
rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(int, lgt_idx, attribute lgt_idx, );

float dMenger(float3 z0, float3 offset, float scale) {
    float4 z = make_float4(z0, 1.0);
    for (int n = 0; n < 4; n++) {
        // z = abs(z);
        z.x = abs(z.x);
        z.y = abs(z.y);
        z.z = abs(z.z);
        z.w = abs(z.w);

        // if (z.x < z.y) z.xy = z.yx;
        if (z.x < z.y)
        {
            float x = z.x;
            z.x = z.y;
            z.y = x;
        }

        // if (z.x < z.z) z.xz = z.zx;
        if (z.x < z.z)
        {
            float x = z.x;
            z.x = z.z;
            z.z = x;
        }

        // if (z.y < z.z) z.yz = z.zy;
        if (z.y < z.z)
        {
            float y = z.y;
            z.y = z.z;
            z.z = y;
        }

        z *= scale;
        // z.xyz -= offset * (scale - 1.0);
        z.x -= offset.x * (scale - 1.0);
        z.y -= offset.y * (scale - 1.0);
        z.z -= offset.z * (scale - 1.0);

        if (z.z < -0.5 * offset.z * (scale - 1.0))
            z.z += offset.z * (scale - 1.0);
    }
    // return (length(max(abs(z.xyz) - make_float3(1.0, 1.0, 1.0), 0.0)) - 0.05) / z.w;
    return (length(make_float3(max(abs(z.x) - 1.0, 0.0), max(abs(z.y) - 1.0, 0.0), max(abs(z.z) - 1.0, 0.0))) - 0.05) / z.w;
}

float3 get_xyz(float4 p)
{
    return make_float3(p.x, p.y, p.z);
}

// not work...
void set_xyz(float4 &a, float3 b)
{
    a.x = b.x;
    a.y = b.y;
    a.x = b.z;
}

float dMandelFast(float3 p, float scale, int n) {
    float4 q0 = make_float4(p, 1.);
    float4 q = q0;

    for (int i = 0; i < n; i++) {
        // q.xyz = clamp(q.xyz, -1.0, 1.0) * 2.0 - q.xyz;
        // set_xyz(q, clamp(get_xyz(q), -1.0, 1.0) * 2.0 - get_xyz(q));
        float4 tmp = clamp(q, -1.0, 1.0) * 2.0 - q;
        q.x = tmp.x;
        q.y = tmp.y;
        q.z = tmp.z;

        // q = q * scale / clamp( dot( q.xyz, q.xyz ), 0.3, 1.0 ) + q0;
        float3 q_xyz = get_xyz(q);
        q = q * scale / clamp(dot(q_xyz, q_xyz), 0.3, 1.0) + q0;
    }

    // return length( q.xyz ) / abs( q.w );
    return length(get_xyz(q)) / abs(q.w);
}

float map(float3 p)
{
    //return length(p - center) - 100.0;

    float scale = 100 * 0.3;
    // f((p - position) / scale) * scale;
    // return dMenger((p - center) / scale, make_float3(1.23, 1.65, 1.45), 2.56) * scale;
    // return dMenger((p - center) / scale, make_float3(1, 1, 1), 3.1) * scale;
    return dMandelFast((p - center) / scale, 2.76, 20) * scale;
}

#define calcNormal(p, dFunc, eps) normalize(\
    make_float3( eps, -eps, -eps) * dFunc(p + make_float3( eps, -eps, -eps)) + \
    make_float3(-eps, -eps,  eps) * dFunc(p + make_float3(-eps, -eps,  eps)) + \
    make_float3(-eps,  eps, -eps) * dFunc(p + make_float3(-eps,  eps, -eps)) + \
    make_float3( eps,  eps,  eps) * dFunc(p + make_float3( eps,  eps,  eps)))

float3 calcNormalBasic(float3 p, float eps)
{
    return normalize(make_float3(
        map(p + make_float3(eps, 0.0, 0.0)) - map(p + make_float3(-eps, 0.0, 0.0)),
        map(p + make_float3(0.0, eps, 0.0)) - map(p + make_float3(0.0, -eps, 0.0)),
        map(p + make_float3(0.0, 0.0, eps)) - map(p + make_float3(0.0, 0.0, -eps))
    ));
}

RT_PROGRAM void intersect(int primIdx)
{
    const float EPS = scene_epsilon;
    float t = ray.tmin, d = 0.0;
    float3 p = ray.origin;

    for (int i = 0; i < 300; i++)
    {
        p = ray.origin + t * ray.direction;
        d = map(p);
        t += d;
        if (abs(d) < EPS || t > ray.tmax)
        {
            break;
        }
    }

    if (t < ray.tmax && rtPotentialIntersection(t))
    {
        shading_normal = geometric_normal = calcNormal(p, map, scene_epsilon);
        texcoord = make_float3(p.x, p.y, 0);
        lgt_idx = lgt_instance;
        rtReportIntersection(0);
    }
}

RT_PROGRAM void bounds(int, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = center - size;
    aabb->m_max = center + size;
}