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


struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );


//------------------------------------------------------------------------------
//
//  Callable programs 
//
//------------------------------------------------------------------------------


RT_CALLABLE_PROGRAM float3 shade_from_normal( const float3 normal )
{
  return optix::normalize( normal ) * 0.5f + 0.5f;
}

RT_CALLABLE_PROGRAM float3 shade_from_ray( const optix::Ray ray )
{
  return optix::normalize( ray.direction ) * 0.5f + 0.5f;
}


//------------------------------------------------------------------------------
//
//  Closest hit program which calls a callable program to perform shading
//
//------------------------------------------------------------------------------


rtCallableProgram( float3, shade_normal, ( const float3 ) );
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, ); 

RT_PROGRAM void closest_hit_radiance()
{
  prd_radiance.result = shade_normal( geometric_normal );
}


//------------------------------------------------------------------------------
//
//  Miss program which calls a callable program to perform shading
//
//------------------------------------------------------------------------------


rtBuffer<rtCallableProgramId< float3( optix::Ray )> > shade_ray;
rtDeclareVariable( optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void miss()
{
  prd_radiance.result = shade_ray[0]( ray );
}



