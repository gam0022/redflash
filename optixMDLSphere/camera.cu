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
#include <common.h>

#include "random.h"
#include "shared_structs.h"

using namespace optix;

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtBuffer<float4, 2>              accum_buffer;
rtBuffer<uchar4, 2>              output_buffer;
rtDeclareVariable(rtObject,      top_object, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
//rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

rtDeclareVariable(float, tonemap_scale, , );

rtDeclareVariable(unsigned int,  frame_number, , );

// exposure + simple Reinhard tonemapper + gamma
__device__ __inline__ uchar4 display(float4 val)
{
    val *= tonemap_scale;
    const float burn_out = 0.1f;
    val.x *= (1.0f + val.x * burn_out) / (1.0f + val.x);
    val.y *= (1.0f + val.y * burn_out) / (1.0f + val.y);
    val.z *= (1.0f + val.z * burn_out) / (1.0f + val.z);
    return make_uchar4(
        (unsigned char)(255.0 * fminf(powf(fmaxf(val.z, 0.0f), (float)(1.0 / 2.2)), 1.0f)),
        (unsigned char)(255.0 * fminf(powf(fmaxf(val.y, 0.0f), (float)(1.0 / 2.2)), 1.0f)),
        (unsigned char)(255.0 * fminf(powf(fmaxf(val.x, 0.0f), (float)(1.0 / 2.2)), 1.0f)),
        255);
}

RT_PROGRAM void pinhole_camera()
{
    const size_t2 screen = output_buffer.size();
    unsigned int rnd_seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number + 1);

    const float jitter_x = rnd(rnd_seed);
    const float jitter_y = rnd(rnd_seed);
    const float2 jitter = make_float2(jitter_x, jitter_y);
    const float2 d = (make_float2(launch_index) + jitter) / make_float2(launch_dim) * 2.f - 1.f;
    const float3 ray_origin = eye;
    const float3 ray_direction = normalize(d.x*U + d.y*V + W);

    optix::Ray ray = optix::make_Ray(ray_origin, ray_direction,
        RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);

    PerRayData_radiance prd;
    prd.result = make_float3(0.0f);
    prd.rnd_seed = rnd_seed;
    rtTrace(top_object, ray, prd);

    // accumulate result
    const float4 result = make_float4(prd.result, 1.0f);
    if (frame_number > 0) {
        const float4 prev_result = accum_buffer[launch_index];
        accum_buffer[launch_index] += (result - prev_result) / (float)(frame_number + 1);
    } else
        accum_buffer[launch_index] = result;

    output_buffer[launch_index] = display(accum_buffer[launch_index]);
}

RT_PROGRAM void exception()
{
    const unsigned int code = rtGetExceptionCode();
    rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
    output_buffer[launch_index] = make_uchar4(255, 0, 255, 255);
}
