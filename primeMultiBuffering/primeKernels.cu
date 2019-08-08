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

#include <cuda_runtime.h>
#include <optixu/optixu_math_namespace.h>
#include <stdio.h>

//------------------------------------------------------------------------------
// Return ceil(x/y) for integers x and y
inline int idivCeil( int x, int y )
{
  return (x + y-1)/y;
}

//------------------------------------------------------------------------------
__global__ void createRaysOrthoKernel(float4* rays, int width, int height, float x0, float y0, float z, float dx, float dy, unsigned rayMask )
{
  int rayx = threadIdx.x + blockIdx.x*blockDim.x;
  int rayy = threadIdx.y + blockIdx.y*blockDim.y;
  if( rayx >= width || rayy >= height )
    return;

  float tMinOrMask = 0.0f;
  if( rayMask ) 
    tMinOrMask = __int_as_float( rayMask );
  
  int idx = rayx + rayy*width;
  rays[2*idx+0] = make_float4( x0+rayx*dx, y0+rayy*dy, z, tMinOrMask );  // origin, tmin
  rays[2*idx+1] = make_float4( 0, 0, 1, 1e34f ); // dir, tmax
}

//------------------------------------------------------------------------------
extern "C" void createRaysOrthoOnDevice( float4* rays, int width, int height, float x0, float y0, float z, float dx, float dy, int yOffset, int yStride, unsigned rayMask )
{
  int rows = idivCeil( (height-yOffset), yStride );
  dim3 blockSize( 32, 16 );
  dim3 gridSize( idivCeil( width, blockSize.x ), idivCeil( rows, blockSize.y ) );
  createRaysOrthoKernel<<<gridSize,blockSize>>>( rays, width, rows, x0, y0+dy*yOffset, z, dx, dy*yStride, rayMask );
}

//------------------------------------------------------------------------------
__global__ void createRaysPerspKernel(float4* rays, int width, int height, float3 eye, float3 U, float3 V, float3 W  )
{
  int rayx = threadIdx.x + blockIdx.x*blockDim.x;
  int rayy = threadIdx.y + blockIdx.y*blockDim.y;
  if( rayx >= width || rayy >= height )
    return;

  int idx = rayx + rayy*width;

  float2 d = make_float2((float)rayx, (float)rayy) / make_float2((float)width, (float)height) * 2.f - 1.f;

  float3 ray_origin = eye;
  float3 ray_direction = optix::normalize(d.x*U + d.y*V + W);

  rays[2*idx+0] = make_float4( ray_origin.x, ray_origin.y, ray_origin.z, 0.0f ); // origin, tmin
  rays[2*idx+1] = make_float4( ray_direction.x, ray_direction.y, ray_direction.z, 1e34f ); // dir, tmax
}

//------------------------------------------------------------------------------
extern "C" void createRaysPerspOnDevice( float4* rays, int width, int height, float3 eye, float3 U, float3 V, float3 W )
{
  dim3 blockSize( 32, 16 );
  dim3 gridSize( idivCeil( width, blockSize.x ), idivCeil( height, blockSize.y ) );
  createRaysPerspKernel<<<gridSize,blockSize>>>(rays, width, height, eye, U, V, W );
}

//------------------------------------------------------------------------------
__global__ void translateRaysKernel(float4* rays, int count, float3 offset)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if( idx >= count )
    return;

  float4 prev = rays[2*idx+0];
  rays[2*idx+0] = make_float4( prev.x + offset.x, prev.y + offset.y, prev.z + offset.z, prev.w );  // origin, tmin
}

//------------------------------------------------------------------------------
extern "C" void translateRaysOnDevice(float4* rays, size_t count, float3 offset)
{
  int blockSize = 512;
  int blockCount = idivCeil((int)count, blockSize);
  translateRaysKernel<<<blockCount,blockSize>>>( rays, (int)count, offset );
}
