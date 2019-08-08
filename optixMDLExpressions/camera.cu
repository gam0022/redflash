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

#include <optix_world.h>
#include "common.h"
#include "helpers.h"

using namespace optix;

struct PerRayData_radiance
{
    float3 result;
    float  importance;
    int    depth;
};

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtBuffer<uchar4, 2>              output_buffer;
rtDeclareVariable(rtObject,      top_object, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

rtDeclareVariable(float, tonemap_scale, , );

// #define TIME_VIEW

static __device__ inline float3 powf(float3 a, float exp)
{
    return make_float3(powf(a.x, exp), powf(a.y, exp), powf(a.z, exp));
}

// exposure + simple Reinhard tonemapper + gamma
__device__ __inline__ uchar4 display(float3 val)
{
    val *= tonemap_scale;
    const float burn_out = 0.1f;
    val *= (1.0f + val * burn_out) / (1.0f + val);
    return make_color(
        fminf(
            powf( fmaxf( val, make_float3( 0.0f ) ), (float)(1.0 / 2.2) ),
            make_float3( 1.0f ) ) );
}

RT_PROGRAM void pinhole_camera()
{
#ifdef TIME_VIEW
    clock_t t0 = clock();
#endif
    const float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
    const float3 ray_origin = eye;
    const float3 ray_direction = normalize(d.x*U + d.y*V + W);

    optix::Ray ray = optix::make_Ray(ray_origin, ray_direction,
        RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);

    PerRayData_radiance prd;
    prd.result = make_float3(0.0f);
    prd.importance = 1.f;
    prd.depth = 0;

    rtTrace(top_object, ray, prd);

#ifdef TIME_VIEW
    clock_t t1 = clock();

    float expected_fps   = 1.0f;
    float pixel_time     = ( t1 - t0 ) * time_view_scale * expected_fps;
    output_buffer[launch_index] = make_color( make_float3( pixel_time ) );
#else
    output_buffer[launch_index] = display(prd.result);
#endif
}

RT_PROGRAM void exception()
{
    const unsigned int code = rtGetExceptionCode();
    rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
    output_buffer[launch_index] = make_uchar4(255, 0, 255, 255);
}
