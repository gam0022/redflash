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

#pragma once

#include <optixu/optixpp_namespace.h>

#include <stdint.h>

// Forward decls
struct Ray;
struct Hit;


// A wrapper around OptiX for pure raycasting queries as in the Prime API.

// Geometry buffers are created from host pointers, similar to RTP_BUFFER_TYPE_HOST buffers in Prime.
// Ray and hit buffers are created directly from CUDA device pointers, similar to RTP_BUFFER_TYPE_CUDA_LINEAR in Prime,
// which avoids a copy and allows pre and post processing in CUDA.

class OptiXRaycastingContext
{
public:
  OptiXRaycastingContext();
  virtual ~OptiXRaycastingContext();

  int getCudaDeviceOrdinal() const {
    return m_cuda_device_ordinal;
  }

  // host pointers
  void setTriangles( int num_triangles, int32_t* indices, int num_vertices, float* positions, float* texcoords );

  // device pointers
  void setRaysDevicePointer( const Ray* rays, size_t n );
  void setHitsDevicePointer( Hit* hits, size_t n );

  // optional mask
  void setMask( const char* texture_filename );

  // Note: Rays and hits can be read from device pointers as soon as this function returns.
  void execute();

private:
  optix::Context m_context;
  int m_cuda_device_ordinal;
  int m_optix_device_ordinal;

  optix::Buffer m_indices;
  optix::Buffer m_positions;
  optix::Buffer m_texcoords;
  optix::Geometry m_geometry;
  optix::Material m_material;
  optix::Buffer m_rays;
  optix::Buffer m_hits; 
  
};

