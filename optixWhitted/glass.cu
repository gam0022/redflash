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
#include "helpers.h"

using namespace optix;

rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(int,          max_depth, , );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtDeclareVariable(float,        importance_cutoff, , );
rtDeclareVariable(float3,       cutoff_color, , );
rtDeclareVariable(float,        fresnel_exponent, , );
rtDeclareVariable(float,        fresnel_minimum, , );
rtDeclareVariable(float,        fresnel_maximum, , );
rtDeclareVariable(float,        refraction_index, , );
rtDeclareVariable(int,          refraction_maxdepth, , );
rtDeclareVariable(int,          reflection_maxdepth, , );
rtDeclareVariable(float3,       refraction_color, , );
rtDeclareVariable(float3,       reflection_color, , );
rtDeclareVariable(float3,       extinction_constant, , );

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
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

// -----------------------------------------------------------------------------

static __device__ __inline__ float3 TraceRay(float3 origin, float3 direction, int depth, float importance )
{
  optix::Ray ray = optix::make_Ray( origin, direction, RADIANCE_RAY_TYPE, 0.0f, RT_DEFAULT_MAX );
  PerRayData_radiance prd;
  prd.depth = depth;
  prd.importance = importance;

  rtTrace( top_object, ray, prd );
  return prd.result;
}

static __device__ __inline__ float3 exp( const float3& x )
{
  return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

// -----------------------------------------------------------------------------

RT_PROGRAM void closest_hit_radiance()
{
  // intersection vectors
  const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal
  const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);
  const float3 bhp = rtTransformPoint(RT_OBJECT_TO_WORLD, back_hit_point);
  const float3 i = ray.direction;                                            // incident direction
        float3 t;                                                            // transmission direction
        float3 r;                                                            // reflection direction

  float reflection = 1.0f;
  float3 result = make_float3(0.0f);
  
  const int depth = prd_radiance.depth;

  float3 beer_attenuation;
  if(dot(n, ray.direction) > 0) {
    // Beer's law attenuation
    beer_attenuation = exp(extinction_constant * t_hit);
  } else {
    beer_attenuation = make_float3(1);
  }

  // refraction
  if (depth < min(refraction_maxdepth, max_depth))
  {
    if ( refract(t, i, n, refraction_index) )
    {
      // check for external or internal reflection
      float cos_theta = dot(i, n);
      if (cos_theta < 0.0f)
        cos_theta = -cos_theta;
      else
        cos_theta = dot(t, n);

      reflection = fresnel_schlick(cos_theta, fresnel_exponent, fresnel_minimum, fresnel_maximum);

      float importance = prd_radiance.importance * (1.0f-reflection) * optix::luminance( refraction_color * beer_attenuation );
      float3 color = cutoff_color;
      if ( importance > importance_cutoff ) {
        color = TraceRay(bhp, t, depth+1, importance);
      }
      result += (1.0f - reflection) * refraction_color * color;
    }
    // else TIR
  } // else reflection==1 so refraction has 0 weight

  // reflection
  float3 color = cutoff_color;
  if (depth < min(reflection_maxdepth, max_depth))
  {
    r = reflect(i, n);
  
    float importance = prd_radiance.importance * reflection * optix::luminance( reflection_color * beer_attenuation );
    if ( importance > importance_cutoff ) {
      color = TraceRay( fhp, r, depth+1, importance );
    }
  }
  result += reflection * reflection_color * color;

  result = result * beer_attenuation;

  prd_radiance.result = result;
}

// -----------------------------------------------------------------------------

//
// Attenuates shadow rays for shadowing transparent objects
//
rtDeclareVariable(float3, shadow_attenuation, , );

RT_PROGRAM void any_hit_shadow()
{
  float3 world_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float nDi = fabs(dot(world_normal, ray.direction));

  prd_shadow.attenuation *= 1-fresnel_schlick(nDi, 5, 1-shadow_attenuation, make_float3(1));
  if(optix::luminance(prd_shadow.attenuation) < importance_cutoff)
    rtTerminateRay();
  else
    rtIgnoreIntersection();
}
