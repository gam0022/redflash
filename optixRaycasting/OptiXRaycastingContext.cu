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

#include "Common.h"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

//
// OptiX programs for raycasting context
//


rtBuffer<float3> vertex_buffer;     
rtBuffer<int3>   index_buffer;
rtBuffer<float2> texcoord_buffer;  // per vertex, indexed with index_buffer

rtDeclareVariable( Hit, hit_attr, attribute hit_attr, ); 

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


RT_PROGRAM void intersect( int primIdx )
{
    const int3 v_idx = index_buffer[ primIdx ];

    const float3 p0 = vertex_buffer[ v_idx.x ];
    const float3 p1 = vertex_buffer[ v_idx.y ];
    const float3 p2 = vertex_buffer[ v_idx.z ];

    // Intersect ray with triangle
    float3 normal;
    float  t, beta, gamma;
    if( intersect_triangle( ray, p0, p1, p2, normal, t, beta, gamma ) )
    {
        if(  rtPotentialIntersection( t ) )
        {
            Hit h;
            h.t = t;
            h.triId = primIdx;
            h.u = beta;
            h.v = gamma;
            h.geom_normal = optix::normalize( normal );

            if ( texcoord_buffer.size() == 0 ) {
              h.texcoord = optix::make_float2( 0.0f, 0.0f ); 
            } else {
              const float2 t0 = texcoord_buffer[ v_idx.x ];
              const float2 t1 = texcoord_buffer[ v_idx.y ];
              const float2 t2 = texcoord_buffer[ v_idx.z ];
              h.texcoord = t1*beta + t2*gamma + t0*(1.0f-beta-gamma);
            }

            hit_attr = h;

            rtReportIntersection( /*material index*/ 0 );
        }
    }
}


//------------------------------------------------------------------------------
//
// Bounds program
//
//------------------------------------------------------------------------------

RT_PROGRAM void bounds( int primIdx, float result[6] )
{
    const int3 v_idx = index_buffer[ primIdx ];

    const float3 p0 = vertex_buffer[ v_idx.x ];
    const float3 p1 = vertex_buffer[ v_idx.y ];
    const float3 p2 = vertex_buffer[ v_idx.z ];

    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = fminf( fminf( p0, p1), p2 );
    aabb->m_max = fmaxf( fmaxf( p0, p1), p2 );
}


//------------------------------------------------------------------------------
//
// Hit program copies hit attribute into hit PRD 
//
//------------------------------------------------------------------------------

rtDeclareVariable( Hit, hit_prd, rtPayload, );

RT_PROGRAM void closest_hit()
{
    hit_prd = hit_attr;
}


//------------------------------------------------------------------------------
//
// Any-hit program masks geometry with a texture
//
//------------------------------------------------------------------------------

rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> mask_sampler;

RT_PROGRAM void any_hit()
{
    float4 mask = tex2D( mask_sampler, hit_attr.texcoord.x, hit_attr.texcoord.y );
    if ( mask.x < 0.5f ) {
      rtIgnoreIntersection(); // make surface transparent
    }
}



//------------------------------------------------------------------------------
//
// Ray generation
//
//------------------------------------------------------------------------------

rtDeclareVariable(unsigned int, launch_index, rtLaunchIndex, );

rtDeclareVariable(rtObject, top_object, , );

rtBuffer<Hit, 1>  hits;
rtBuffer<Ray, 1>  rays;


RT_PROGRAM void ray_gen()
{
    Hit hit_prd;
    hit_prd.t           = -1.0f;
    hit_prd.triId       = -1;
    hit_prd.u           = 0.0f;
    hit_prd.v           = 0.0f;
    hit_prd.geom_normal = optix::make_float3(1, 0, 0);

    Ray ray = rays[launch_index];
    rtTrace( top_object,
             optix::make_Ray( ray.origin, ray.dir, 0, ray.tmin, ray.tmax ),
             hit_prd );

    hits[ launch_index ] = hit_prd;
}

//------------------------------------------------------------------------------
//
// Exception program for debugging only
//
//------------------------------------------------------------------------------


RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d)\n", code, launch_index );
  Hit hit_prd;
  hit_prd.t           = -1.0f;
  hit_prd.triId       = -1;
  hit_prd.u           = 0.0f;
  hit_prd.v           = 0.0f;
  hit_prd.geom_normal = optix::make_float3(1, 0, 0);
  hits[ launch_index ] = hit_prd;
}

