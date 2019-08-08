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
#include "common.h"

struct PerRayData_radiance
{
    float3 result;
    float importance;
    int depth;
};

struct PerRayData_shadow
{
    float3 attenuation;
};

using namespace optix;

// buffers that contains the light definitions
rtBuffer<BasicLight>        lights;

rtDeclareVariable(float,            scene_epsilon, , );

// position and color of the particle that has been hit
rtDeclareVariable(float3,   particle_position,  attribute particle_position, );
rtDeclareVariable(float3,   particle_color,     attribute particle_color, );

rtDeclareVariable(float,                t_hit,          rtIntersectionDistance, );
rtDeclareVariable(optix::Ray,           ray,            rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance,  prd,            rtPayload, );
rtDeclareVariable(PerRayData_shadow,    prd_shadow,     rtPayload, );
rtDeclareVariable(rtObject,             top_shadower, , );

rtDeclareVariable(int,      flat_shaded, , );     // true/false


RT_PROGRAM void any_hit()
{
    // this material is opaque, so it fully attenuates all shadow rays
    prd_shadow.attenuation = optix::make_float3( 0.0f );
    rtTerminateRay();
}


RT_PROGRAM void closest_hit()
{
    if ( flat_shaded ) {
        prd.result = particle_color;
        return;
    }

    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 shading_normal = normalize( hit_point - particle_position );
    float3 result = make_float3( 0.0f );

    // simple Lambertian model, the particle color is used as diffuse color

    // compute direct lighting
    unsigned int num_lights = lights.size();
    for( int i = 0; i < num_lights; ++i ) {
        BasicLight light = lights[i];
        float Ldist = optix::length( light.pos - hit_point );
        float3 L = optix::normalize( light.pos - hit_point );
        float nDl = optix::dot( shading_normal, L );

        // cast shadow ray
        float3 light_attenuation = make_float3( static_cast<float>( nDl > 0.0f ) );
        if ( nDl > 0.0f && light.casts_shadow ) {
            PerRayData_shadow shadow_prd;
            shadow_prd.attenuation = make_float3( 1.0f );
            optix::Ray shadow_ray = optix::make_Ray( hit_point, L,
                SHADOW_RAY_TYPE, scene_epsilon, Ldist );
            rtTrace( top_shadower, shadow_ray, shadow_prd );
            light_attenuation = shadow_prd.attenuation;
        }

        // If not completely shadowed, light the hit point
        if( fmaxf( light_attenuation ) > 0.0f ) {
            float3 Lc = light.color * light_attenuation;
            result += particle_color * nDl * Lc;
        }
    }

    // pass the color back up the tree
    prd.result = result;
}

