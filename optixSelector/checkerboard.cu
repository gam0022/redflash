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
#include <common.h>

using namespace optix;

rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(int,          max_depth, , );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, isect_dist, rtIntersectionDistance, );

rtDeclareVariable(float3,       tile_size, , ); 
rtDeclareVariable(float3,       tile_color_dark, , );
rtDeclareVariable(float3,       tile_color_light, , );
rtDeclareVariable(float3,       light_direction, , );

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

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// -----------------------------------------------------------------------------

RT_PROGRAM void closest_hit_radiance()
{
  const float3 h = ray.origin + isect_dist * ray.direction;
  const float3 l = normalize(light_direction);
  const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

  const float3 p = h / tile_size;
  float3 result = (static_cast<int>( floorf(p.x) + floorf(p.y) + floorf(p.z) ) & 1) ?
                  tile_color_light : tile_color_dark;

  const float cos_theta = dot(l, n);
  if (cos_theta > 0.01f)
  {
    optix::Ray shadow_ray = optix::make_Ray( h, l, SHADOW_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX );
    PerRayData_shadow shadow_prd;
    shadow_prd.attenuation = make_float3(1.0f);
  
    rtTrace( top_object, shadow_ray, shadow_prd );
  
    result *= 0.25f + (cos_theta * shadow_prd.attenuation * 0.75f);
  }
  else
    result *= 0.25f;

  prd_radiance.result = result;
}

// -----------------------------------------------------------------------------

RT_PROGRAM void any_hit_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0.0f);
  rtTerminateRay();
}
