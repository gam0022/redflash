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
#include "phong.h" 

using namespace optix;

rtDeclareVariable(float3,       Kd1, , );
rtDeclareVariable(float3,       Kd2, , );
rtDeclareVariable(float3,       Ka1, , );
rtDeclareVariable(float3,       Ka2, , );
rtDeclareVariable(float3,       Ks1, , );
rtDeclareVariable(float3,       Ks2, , );
rtDeclareVariable(float3,       Kr1, , );
rtDeclareVariable(float3,       Kr2, , );
rtDeclareVariable(float,        phong_exp1, , );
rtDeclareVariable(float,        phong_exp2, , );
rtDeclareVariable(float3,       inv_checker_size, , );  // Inverse checker height, width and depth in texture space

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 


RT_PROGRAM void any_hit_shadow()
{
  phongShadowed();
}


RT_PROGRAM void closest_hit_radiance()
{
  float3 Kd, Ka, Ks, Kr;
  float  phong_exp;

  float3 t  = texcoord * inv_checker_size;
  t.x = floorf(t.x);
  t.y = floorf(t.y);
  t.z = floorf(t.z);

  int which_check = ( static_cast<int>( t.x ) +
                      static_cast<int>( t.y ) +
                      static_cast<int>( t.z ) ) & 1;

  if ( which_check ) {
    Kd = Kd1; Ka = Ka1; Ks = Ks1; Kr = Kr1; phong_exp = phong_exp1;
  } else {
    Kd = Kd2; Ka = Ka2; Ks = Ks2; Kr = Kr2; phong_exp = phong_exp2;
  }

  float3 world_shading_normal   = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
  float3 ffnormal  = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
  phongShade( Kd, Ka, Ks, Kr, phong_exp, ffnormal );
}
