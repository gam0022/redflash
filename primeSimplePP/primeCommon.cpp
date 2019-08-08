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

#include "primeCommon.h"
#include <math.h>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

//------------------------------------------------------------------------------
//
//  External functions from simplePrimeKernels.cu
//
extern "C" 
{
  void translateRaysOnDevice(float4* rays, size_t count, float3 offset );
  void createRaysOrthoOnDevice( float4* rays, 
                                int width, int height,
                                float x0, float y0,
                                float z,
                                float dx, float dy,
                                int yOffset, int yStride, unsigned rayMask );
  void createRaysPerspOnDevice( float4* rays, 
                                int width, int height,
                                float3 eye,
                                float3 U, float3 V, float3 W );
}

//------------------------------------------------------------------------------
// ceiling( x/y ) where x and y are integers
inline int idivCeil( int x, int y ) { return (x + y - 1)/y; }

//------------------------------------------------------------------------------
inline float __int_as_float  (int val)
{ 
  union {float f; int i;} var;
  var.i = val;
  return var.f;
}

//------------------------------------------------------------------------------
void createRaysOrtho( Buffer<Ray>& raysBuffer, int width, int* height,
  const float3& bbmin, const float3& bbmax, float margin, unsigned rayMask, int yOffset, int yStride )
{
  float3 bbspan = bbmax - bbmin;
  
  // set height according to aspect ratio of bounding box    
  *height = (int)(width * bbspan.y / bbspan.x);

  float dx = bbspan.x * (1 + 2*margin) / width;
  float dy = bbspan.y * (1 + 2*margin) / *height;
  float x0 = bbmin.x - bbspan.x*margin + dx/2;
  float y0 = bbmin.y - bbspan.y*margin + dy/2;
  float z = bbmin.z - std::max(bbspan.z,1.0f)*.001f;
  int rows = idivCeil( (*height - yOffset), yStride );
  raysBuffer.alloc( width * rows );

  if( raysBuffer.type() == RTP_BUFFER_TYPE_HOST )
  {
    Ray* rays = raysBuffer.ptr();
    float y = y0 + dy*yOffset;
    size_t idx = 0;
    for( int iy = yOffset; iy < *height; iy += yStride )
    {
      float x = x0;
      for( int ix = 0; ix < width; ix++ )
      {
        float tminOrMask = 0.0f;
        if( rayMask ) 
          tminOrMask = __int_as_float( rayMask );
        
        Ray r = { make_float3(x,y,z), tminOrMask, make_float3(0,0,1), 1e34f };
        rays[idx++] = r;
        x += dx;
      }
      y += dy*yStride;
    }  
  }
  else if( raysBuffer.type() == RTP_BUFFER_TYPE_CUDA_LINEAR )
  {    
    createRaysOrthoOnDevice( (float4*)raysBuffer.ptr(), width, *height, x0, y0, z, dx, dy, yOffset, yStride, rayMask );
  }
}

//------------------------------------------------------------------------------
// Compute a left-handed coordinate frame for the camera
void computeUVW( float3 eye, float3 lookat, float3 up, float3& U, float3& V, float3& W )
{
  W = optix::normalize( lookat - eye );
  U = optix::normalize( optix::cross( W, up ) );
  V = optix::cross( U, W );
}

//------------------------------------------------------------------------------
void createRaysPersp( Buffer<Ray>& raysBuffer, int width, int height, const float3& eye, const float3& lookAt, const float vfov /* = 60.0f */ )
{
  float aspectRatio  = float(width)/height;
  float vScale = float( tan( vfov/2 * M_PI/180 ) );
  float uScale = vScale * aspectRatio;

  float3 U, V, W;
  computeUVW( eye, lookAt, make_float3( 0.0f, 1.0f, 0.0f ), U, V, W );
  U *= uScale;
  V *= vScale;

  raysBuffer.alloc( width * height );
  if( raysBuffer.type() == RTP_BUFFER_TYPE_HOST )
  {
    Ray* rays = raysBuffer.ptr();

    int idx=0;
    for( int h=0; h < height; h++ ) 
    {
      float v = float(h)/height * 2.0f - 1.0f;
      for( int w=0; w < width; w++ ) 
      {
        float u = float(w)/width * 2.0f - 1.0f;
        float3 dir = optix::normalize(u*U + v*V + W);
        Ray r = { eye, 0.0f, dir, 1e34f };
        rays[idx++] = r;
      }
    }
  }
  else if( raysBuffer.type() == RTP_BUFFER_TYPE_CUDA_LINEAR )
  {
    createRaysPerspOnDevice( (float4*)raysBuffer.ptr(), width, height, eye, U, V, W );
  }
}

//------------------------------------------------------------------------------
void translateRays( Buffer<Ray>& raysBuffer, const float3& offset )
{
  if( raysBuffer.type() == RTP_BUFFER_TYPE_HOST )
  {
    Ray* rays = raysBuffer.ptr();
    for( size_t r=0; r < raysBuffer.count(); r++ )
      rays[r].origin = rays[r].origin + offset;
  }
  else if( raysBuffer.type() == RTP_BUFFER_TYPE_CUDA_LINEAR )
  {
    translateRaysOnDevice( (float4*)raysBuffer.ptr(), raysBuffer.count(), offset );
  }
}

//------------------------------------------------------------------------------
void shadeHits( std::vector<float3>& image, Buffer<Hit>& hitsBuffer, PrimeMesh& mesh )
{
  float3 backgroundColor = { 0.2f, 0.2f, 0.2f };

  int3* indices = mesh.getVertexIndices();
  float3* vertices = mesh.getVertexData();
  const Hit* hits = hitsBuffer.hostPtr();
  for( size_t i=0; i < hitsBuffer.count(); i++ )
  {
    if( hits[i].t < 0.0f )
    {
      image[i] = backgroundColor;
    }
    else
    {
      int3 tri  = indices[hits[i].triId];
      float3 v0 = vertices[tri.x];
      float3 v1 = vertices[tri.y];
      float3 v2 = vertices[tri.z];
      float3 e0 = v1-v0;
      float3 e1 = v2-v0;
      float3 n = optix::normalize( optix::cross( e0, e1 ) );

      image[i] = 0.5f*n + make_float3( 0.5f, 0.5f, 0.5f ); 
    }
  }
}

//------------------------------------------------------------------------------
float3 transformPoint(const SimpleMatrix4x3& M, const float3& p)
{
  return make_float3(
    M.f0*p.x + M.f1*p.y + M.f2*p.z  + M.f3,
    M.f4*p.x + M.f5*p.y + M.f6*p.z  + M.f7,
    M.f8*p.x + M.f9*p.y + M.f10*p.z + M.f11 );
}

//------------------------------------------------------------------------------
// Transform normal by inverse transpose of transform matrix M
float3 transformNormal( const SimpleMatrix4x3& Minv, const float3& n)
{
  return make_float3(
    Minv.f0*n.x + Minv.f4*n.y + Minv.f8*n.z,
    Minv.f1*n.x + Minv.f5*n.y + Minv.f9*n.z,
    Minv.f2*n.x + Minv.f6*n.y + Minv.f10*n.z );  
}

//------------------------------------------------------------------------------
void shadeHits( std::vector<float3>& image, Buffer<HitInstancing>& hitsBuffer, std::vector<int>& modelIds, std::vector<PrimeMesh>& models, float3 eye, std::vector<SimpleMatrix4x3>& invTransforms )
{
  float3 backgroundColor = { 1.0f, 1.0f, 1.0f };

  const HitInstancing* hits = hitsBuffer.hostPtr();
  for( size_t i=0; i < hitsBuffer.count(); ++i )
  {
    if( hits[i].t < 0.0f )
    {
      image[i] = backgroundColor;
    }
    else
    {
      int modelId = modelIds[hits[i].instId];
      PrimeMesh& mesh = models[modelId];
      int3* indices = mesh.getVertexIndices();
      float3* vertices = mesh.getVertexData();
      SimpleMatrix4x3& Minv = invTransforms[hits[i].instId];

      // Compute normal in object space
      int3 tri  = indices[hits[i].triId];
      float3 v0 = vertices[tri.x];
      float3 v1 = vertices[tri.y];
      float3 v2 = vertices[tri.z];
      float3 e0 = v1-v0;
      float3 e1 = v2-v0;
      float3  n = optix::cross( e0, e1 ); // save normalization for later

      // Flip normal if facing away from eye
      float3 eyeO = transformPoint( Minv, eye );
      float3 dir = v0 - eyeO;
      if( optix::dot(n, dir) > 0 )
        n = -n;

      // Transform to world space
      n = optix::normalize( transformNormal( Minv, n ) );     
      
      // Compute color
      image[i] = 0.5f*n + make_float3( 0.5f, 0.5f, 0.5f ); 
    }
  }
}
//------------------------------------------------------------------------------
void writePpm( const char* filename, const float* image, int width, int height )
{
  std::ofstream out( filename, std::ios::out | std::ios::binary );
  if( !out ) 
  {
    std::cerr << "Cannot open file " << filename << "'" << std::endl;
    return;
  }

  out << "P6\n" << width << " " << height << "\n255" << std::endl;
  for( int y=height-1; y >= 0; --y ) // flip vertically
  {  
    for( int x = 0; x < width*3; ++x ) 
    {
      float val = image[y*width*3 + x];
      unsigned char cval = val < 0.0f ? 0u : val > 1.0f ? 255u : static_cast<unsigned char>( val*255.0f );
      out.put( cval );
    }
  }
   
  std::cout << "Wrote file " << filename << std::endl;
}

//------------------------------------------------------------------------------
void resetAllDevices()
{
  int deviceCount;
  CHK_CUDA( cudaGetDeviceCount( &deviceCount ) );
  for( int i=0; i < deviceCount; ++i )
  {
    CHK_CUDA( cudaSetDevice(i) );
    CHK_CUDA( cudaDeviceReset() );
  }
}


