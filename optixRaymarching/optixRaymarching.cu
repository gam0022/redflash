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
#include "optixRaymarching.h"
#include "random.h"

using namespace optix;

struct PerRayData_pathtrace
{
    float3 result;
    float3 radiance;
    float3 attenuation;
    float3 origin;
    float3 direction;
    unsigned int seed;
    int depth;
    int countEmitted;
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
        prd.result = make_float3(0.f);
        prd.attenuation = make_float3(1.f);
        prd.countEmitted = true;
        prd.done = false;
        prd.seed = seed;
        prd.depth = 0;

        // Each iteration is a segment of the ray path.  The closest hit will
        // return new segments to be traced here.
        for(;;)
        {
            Ray ray = make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon * 10.0, RT_DEFAULT_MAX);
            rtTrace(top_object, ray, prd);

            if(prd.done)
            {
                // We have hit the background or a luminaire
                prd.result += prd.radiance * prd.attenuation;
                break;
            }

            // Russian roulette termination 
            if(prd.depth >= rr_begin_depth)
            {
                float pcont = fmaxf(prd.attenuation);
                if(rnd(prd.seed) >= pcont)
                    break;
                prd.attenuation /= pcont;
            }

            prd.depth++;
            prd.result += prd.radiance * prd.attenuation;

            // Update ray data for the next path segment
            ray_origin = prd.origin;
            ray_direction = prd.direction;
        }

        result += prd.result;
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

RT_PROGRAM void diffuseEmitter()
{
    current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(0.f);
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


RT_PROGRAM void diffuse()
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    float3 hitpoint = ray.origin + t_hit * ray.direction + ffnormal * scene_epsilon * 10.0;

    //
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //
    current_prd.origin = hitpoint;

    float z1=rnd(current_prd.seed);
    float z2=rnd(current_prd.seed);
    float3 p;
    cosine_sample_hemisphere(z1, z2, p);
    optix::Onb onb( ffnormal );
    onb.inverse_transform( p );
    current_prd.direction = p;

    // NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
    // with cosine density.
    current_prd.attenuation = current_prd.attenuation * diffuse_color;
    current_prd.countEmitted = false;

    //
    // Next event estimation (compute direct lighting).
    //
    unsigned int num_lights = lights.size();
    float3 result = make_float3(0.0f);

    for(int i = 0; i < num_lights; ++i)
    {
        // Choose random point on light
        ParallelogramLight light = lights[i];
        const float z1 = rnd(current_prd.seed);
        const float z2 = rnd(current_prd.seed);
        const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

        // Calculate properties of light sample (for area based pdf)
        const float  Ldist = length(light_pos - hitpoint);
        const float3 L     = normalize(light_pos - hitpoint);
        const float  nDl   = dot( ffnormal, L );
        const float  LnDl  = dot( light.normal, L );

        // cast shadow ray
        if ( nDl > 0.0f && LnDl > 0.0f )
        {
            PerRayData_pathtrace_shadow shadow_prd;
            shadow_prd.inShadow = false;
            // Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
            Ray shadow_ray = make_Ray( hitpoint, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist - scene_epsilon );
            rtTrace(top_object, shadow_ray, shadow_prd);

            if(!shadow_prd.inShadow)
            {
                const float A = length(cross(light.v1, light.v2));
                // convert area based pdf to solid angle
                const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
                result += light.emission * weight;
            }
        }
    }

    current_prd.radiance = result;
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

rtDeclareVariable(float3, bg_color, , );

RT_PROGRAM void miss()
{
    current_prd.radiance = bg_color;
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

float map(float3 p)
{
    //return length(p - center) - 100.0;

    float scale = 100 * 0.25;
    // f((p - position) / scale) * scale;
    // return dMenger((p - center) / scale, make_float3(1.23, 1.65, 1.45), 2.56) * scale;
    return dMenger((p - center) / scale, make_float3(1, 1, 1), 3.1) * scale;
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
    float t = 0.0, d = 1e100;
    float3 p = ray.origin;

    for (int i = 0; i < 100; i++)
    {
        d = map(p);
        t += d;
        p = ray.origin + t * ray.direction;
        if (abs(d) < EPS)
        {
            break;
        }
    }

    if (abs(d) < EPS && rtPotentialIntersection(t))
    {
        shading_normal = geometric_normal = calcNormal(p, map, scene_epsilon * 0.1);
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


//
// Environment map background
//
rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void envmap_miss()
{
    float theta = atan2f( ray.direction.x, ray.direction.z );
    float phi   = M_PIf * 0.5f -  acosf( ray.direction.y );
    float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
    float v     = 0.5f * ( 1.0f + sin(phi) );
    current_prd.radiance = make_float3( tex2D(envmap, u, v) );
    current_prd.done = true;
}