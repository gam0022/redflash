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

#ifndef __simple_prime_common_h__
#define __simple_prime_common_h__


#include <optix_prime/optix_prime.h>
#include <putil/Buffer.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <Mesh.h>

//------------------------------------------------------------------------------
struct SimpleMatrix4x3
{
  float f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11;
  SimpleMatrix4x3( 
    float  v0=0, float  v1=0, float  v2=0, float v3=0,
    float  v4=0, float  v5=0, float  v6=0, float v7=0,
    float  v8=0, float  v9=0, float v10=0, float v11=0 ) 
    : f0(v0), f1(v1),   f2(v2),   f3(v3)
    , f4(v4), f5(v5),   f6(v6),   f7(v7)  
    , f8(v8), f9(v9), f10(v10), f11(v11) 
  {}
};

//------------------------------------------------------------------------------
// 
//  Ray and hit structures for query input and output
//
struct Ray
{
  static const RTPbufferformat format = RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX;

  float3 origin;
  float  tmin;
  float3 dir;
  float  tmax;
};

struct Hit
{
  static const RTPbufferformat format = RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V;

  float t;
  int   triId;
  float u;
  float v;
};

struct HitInstancing
{
  static const RTPbufferformat format = RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V;

  float t;
  int   triId;
  int   instId;
  float u;
  float v;
};

//------------------------------------------------------------------------------
// A wrapper that provides more convenient return types
class PrimeMesh : public Mesh
{
public:
  float3  getBBoxMin()       { return ptr_to_float3( bbox_min ); }
  float3  getBBoxMax()       { return ptr_to_float3( bbox_max ); }
  int3*   getVertexIndices() { return reinterpret_cast<int3*>( tri_indices ); }
  float3* getVertexData()    { return reinterpret_cast<float3*>( positions );  }

private:
  float3 ptr_to_float3( const float* v ) { return make_float3( v[0], v[1], v[2] ); }
};

//------------------------------------------------------------------------------
// Generate rays in an orthographic view frustum.
void createRaysOrtho( Buffer<Ray>& raysBuffer, int width, int* height,
  const float3& bbmin, const float3& bbmax, float margin, unsigned rayMask=0, int yOffset=0, int yStride=1 );

//------------------------------------------------------------------------------
// Generate rays in a perspective view frustum.
void createRaysPersp( Buffer<Ray>& raysBuffer, int width, int height, 
  const float3& eye, const float3& lookAt, const float vfov=60.0f );

//------------------------------------------------------------------------------
// Offset ray origins.
void translateRays( Buffer<Ray>& raysBuffer, const float3& offset );

//------------------------------------------------------------------------------
// Perform simple shading via normal visualization.
void shadeHits( std::vector<float3>& image, Buffer<Hit>& hitsBuffer, PrimeMesh& mesh );
void shadeHits( std::vector<float3>& image, Buffer<HitInstancing>& hitsBuffer, std::vector<int>& modelIds, std::vector<PrimeMesh>& meshes, float3 eye, std::vector<SimpleMatrix4x3>& invTransforms );

//------------------------------------------------------------------------------
// Write PPM image to a file.
void writePpm( const char* filename, const float* image, int width, int height );

//------------------------------------------------------------------------------
// Resets all devices
void resetAllDevices();


#endif // __simple_prime_common_h__
