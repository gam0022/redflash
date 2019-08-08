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
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtDeclareVariable(float3,  center, , );
rtDeclareVariable(float,   radius1, , );
rtDeclareVariable(float,   radius2, , );
rtDeclareVariable(float,   scene_epsilon, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, ); 
rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

RT_PROGRAM void intersect(int primIdx)
{
  float3 O = ray.origin - center;
  float3 D = ray.direction;

  float b = dot(O, D);
  float O_dot_O = dot(O, O);
  float sqr_radius2 = radius2*radius2;

  // check if we are outside of outer sphere
  if ( O_dot_O > sqr_radius2 + scene_epsilon ) { 
    float c = O_dot_O - sqr_radius2; 
    float root = b*b-c;
    if ( root > 0.0f ) {
      float t = -b - sqrtf( root );
      if( rtPotentialIntersection( t ) ) {
        shading_normal = geometric_normal = ( O + t*D ) / radius2;
        float3 hit_p  = ray.origin +t*ray.direction;
        float3 offset = normalize( shading_normal )*scene_epsilon;
        front_hit_point = hit_p + offset;
        back_hit_point = hit_p  - offset;
        rtReportIntersection( 0 );
      } 
    }

    // else we are inside of the outer sphere
  } else {

    float c = O_dot_O - radius1*radius1;
    float root = b*b-c;
    if ( root > 0.0f ) {
      float t = -b - sqrtf( root );
      // do we hit inner sphere from between spheres? 
      if( rtPotentialIntersection( t ) ) {
        shading_normal = geometric_normal = ( O + t*D )/(-radius1);
        float3 hit_p  = ray.origin +t*ray.direction;
        float3 offset = normalize( shading_normal )*scene_epsilon;
        front_hit_point = hit_p - offset;
        back_hit_point  = hit_p  + offset;
        rtReportIntersection( 0 );
      } else { 
        t = -b + sqrtf( root );
        // do we hit inner sphere from within both spheres?
        if( rtPotentialIntersection( t ) ) {
          shading_normal = geometric_normal = ( O + t*D )/(-radius1);
          float3 hit_p  = ray.origin +t*ray.direction;
          float3 offset = normalize( shading_normal )*scene_epsilon;
          front_hit_point = hit_p + offset;
          back_hit_point = hit_p  - offset;
          rtReportIntersection( 0 );
        } else {
          c = O_dot_O - sqr_radius2;
          root = b*b-c;
          t = -b + sqrtf( root );
          // do we hit outer sphere from between spheres?
          if( rtPotentialIntersection( t ) ) {
            shading_normal = geometric_normal = ( O + t*D )/radius2;
            float3 hit_p  = ray.origin +t*ray.direction;
            float3 offset = normalize( shading_normal )*scene_epsilon;
            front_hit_point = hit_p - offset;
            back_hit_point = hit_p  + offset;
            rtReportIntersection( 0 );
          }
        }
      }
    } else { 
      c = O_dot_O - sqr_radius2;
      root = b*b-c;
      float t = -b + sqrtf( root );
      // do we hit outer sphere from between spheres? 
      if( rtPotentialIntersection( t ) ) {
        shading_normal = geometric_normal = ( O + t*D )/radius2;
        float3 hit_p  = ray.origin +t*ray.direction;
        float3 offset = normalize( shading_normal )*scene_epsilon;
        front_hit_point = hit_p - offset;
        back_hit_point = hit_p  + offset;
        rtReportIntersection( 0 );
      }
    }
  }
}


RT_PROGRAM void bounds (int, optix::Aabb* aabb)
{
  float3 rad = make_float3( max(radius1,radius2) );
  aabb->m_min = center - rad;
  aabb->m_max = center + rad;
}
