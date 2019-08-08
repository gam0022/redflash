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

using namespace optix;

rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 

rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(rtObject,          top_shadower, , );
rtBuffer<BasicLight>                 lights;

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

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );


RT_PROGRAM void any_hit_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);

  rtTerminateRay();
}

RT_PROGRAM void closest_hit_radiance()
{
  
  const float3 hit_point = ray.origin + t_hit * ray.direction;

  const float3 world_shading_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  const float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  const float3 Kd = world_shading_normal*0.5f + make_float3(0.5f);

  float3 result = make_float3( 0.0f );
  unsigned int num_lights = lights.size();
  for(int i = 0; i < num_lights; ++i) {
      BasicLight light = lights[i];
      float3 L = optix::normalize(light.pos - hit_point);
      float Ldist = optix::length( L );
      float NdotL = optix::dot( ffnormal, L );

      // cast shadow ray
      if ( NdotL > 0.0f ) {
          float3 light_attenuation = make_float3( 1.0f );
          if ( light.casts_shadow ) {
              PerRayData_shadow shadow_prd;
              shadow_prd.attenuation = make_float3( 1.0f );
              optix::Ray shadow_ray = optix::make_Ray( hit_point, L, 
                  SHADOW_RAY_TYPE, scene_epsilon, Ldist );
              rtTrace(top_shadower, shadow_ray, shadow_prd);
              light_attenuation = shadow_prd.attenuation;
          }
          result += Kd * light_attenuation * light.color;
      }
  }

  prd_radiance.result = result;
}

