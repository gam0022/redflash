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

//-----------------------------------------------------------------------------
//
//  This sample uses OptiX as a replacement for Prime, to compute hits only.  Compare to primeSimplePP.
//  Shading and ray generation are done with separate CUDA kernels and interop.
//  Also supports an optional mask texture for geometry transparency (the hole in the default fish model).
//
//-----------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "Common.h"
#include "OptiXRaycastingContext.h"
#include "optixRaycastingKernels.h"

#include <optixu/optixu_math_namespace.h>
#include <sutil.h>
#include <Mesh.h>

#include <cstdlib>
#include <fstream>
#include <iostream>


void printUsageAndExit( const char* argv0 )
{
  std::cerr
  << "Usage  : " << argv0 << " [options]\n"
  << "App options:\n"
  << "  -h  | --help                               Print this usage message\n"
  << "  -m  | --mesh <mesh_file>                   Model to be rendered\n"
  << "        --mask <ppm_file>                    Mask texture (optional)\n"
  << "  -w  | --width <number>                     Output image width\n"
  << std::endl;
  
  exit(1);
}


void writePPM( const char* filename, const float* image, int width, int height )
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


int main( int argc, char** argv )
{
  std::string objFilename;
  std::string maskFilename;
  int width = 640;

  // parse arguments
  for ( int i = 1; i < argc; ++i ) 
  { 
    std::string arg( argv[i] );
    if( arg == "-h" || arg == "--help" ) 
    {
      printUsageAndExit( argv[0] ); 
    } 
    else if( (arg == "-m" || arg == "--mesh") && i+1 < argc ) 
    {
      objFilename = argv[++i];
    } 
    else if ( (arg == "--mask") && i+1 < argc )
    {
      maskFilename = argv[++i];
    }
    else if( (arg == "-w" || arg == "--width") && i+1 < argc ) 
    {
      width = atoi(argv[++i]);
    } 
    else 
    {
      std::cerr << "Bad option: '" << arg << "'" << std::endl;
      printUsageAndExit( argv[0] );
    }
  }

  // Set default scene with mask if user did not specify scene
  if (objFilename.empty()) {
    objFilename = std::string( sutil::samplesDir() ) + "/data/fish.obj";
    if (maskFilename.empty()) { 
      maskFilename = std::string( sutil::samplesDir() ) + "/data/fish_mask.ppm";
    }
  }

  try {

    //
    // Create Context
    //

    OptiXRaycastingContext context;


    //
    // Create model on host
    //

    std::cerr << "Loading model: " << objFilename << std::endl;
    HostMesh model( objFilename );
    context.setTriangles( model.num_triangles, model.tri_indices, model.num_vertices, model.positions,
      model.has_texcoords ? model.texcoords : NULL );

    //
    // Create CUDA buffers for rays and hits
    //

    const optix::float3& bbox_min = *reinterpret_cast<const optix::float3*>(model.bbox_min);
    const optix::float3& bbox_max = *reinterpret_cast<const optix::float3*>(model.bbox_max);
    const optix::float3 bbox_span = bbox_max - bbox_min;
    const int height = static_cast<int>(width * bbox_span.y / bbox_span.x);

    cudaError_t err = cudaSetDevice( context.getCudaDeviceOrdinal() );
    if( err != cudaSuccess )
    {
      printf( "cudaSetDevice failed (%s): %s\n", cudaGetErrorName( err ), cudaGetErrorString( err ) );
      exit( 1 );
    }

    // Populate rays using CUDA kernel
    Ray* rays_d = NULL;
    err = cudaMalloc( &rays_d, sizeof(Ray)*width*height );
    if( err != cudaSuccess )
    {
      printf( "cudaMalloc failed (%s): %s\n", cudaGetErrorName( err ), cudaGetErrorString( err ) );
      exit( 1 );
    }

    createRaysOrthoOnDevice( rays_d, width, height, bbox_min, bbox_max, 0.05f );
    err = cudaGetLastError();
    if( err != cudaSuccess )
    {
      printf( "Error while creating rays on device (%s): %s\n", cudaGetErrorName( err ), cudaGetErrorString( err ) );
      exit( 1 );
    }

    context.setRaysDevicePointer( rays_d, size_t(width*height) );

    Hit* hits_d = NULL;
    err = cudaMalloc( &hits_d, sizeof(Hit)*width*height );
    if( err != cudaSuccess )
    {
      printf( "cudaMalloc failed (%s): %s\n", cudaGetErrorName( err ), cudaGetErrorString( err ) );
      exit( 1 );
    }
    context.setHitsDevicePointer( hits_d, size_t(width*height) );

    //
    // Optional mask
    //

    if( !maskFilename.empty() )
    {
      std::cerr << "Loading mask: " << maskFilename << std::endl;
      context.setMask( maskFilename.c_str() );
    }

    //
    // Execute query.  Includes time to compile OptiX programs and upload model buffers.
    //

    context.execute();

    //
    // Shade the hit results to create image
    //

    optix::float3* image_d = NULL;
    err = cudaMalloc( &image_d, width*height*sizeof(optix::float3));
    if( err != cudaSuccess )
    {
      printf( "cudaMalloc failed (%s): %s\n", cudaGetErrorName( err ), cudaGetErrorString( err ) );
      exit( 1 );
    }

    std::vector<float3> image_h( width*height );

    shadeHitsOnDevice( image_d, width*height, hits_d );
    err = cudaGetLastError();
    if( err != cudaSuccess )
    {
      printf( "Error while shading hits on device (%s): %s\n", cudaGetErrorName( err ), cudaGetErrorString( err ) );
      exit( 1 );
    }

    err = cudaMemcpy( &image_h[0], image_d, (size_t)(width*height)*sizeof(optix::float3), cudaMemcpyDeviceToHost );
    if( err != cudaSuccess )
    {
      printf( "cudaMemcpy failed (%s): %s\n", cudaGetErrorName( err ), cudaGetErrorString( err ) );
      exit( 1 );
    }

    writePPM( "output.ppm", &image_h[0].x, width, height );

    //
    // Re-execute query with different rays
    //

    translateRaysOnDevice( rays_d, width*height, bbox_span * optix::make_float3(0.2f, 0, 0) );
    err = cudaGetLastError();
    if( err != cudaSuccess )
    {
      printf( "Error while translating rays on device (%s): %s\n", cudaGetErrorName( err ), cudaGetErrorString( err ) );
      exit( 1 );
    }

    context.execute();

    shadeHitsOnDevice( image_d, width*height, hits_d );
    err = cudaGetLastError();
    if( err != cudaSuccess )
    {
      printf( "Error while shading hits on device (%s): %s\n", cudaGetErrorName( err ), cudaGetErrorString( err ) );
      exit( 1 );
    }

    err = cudaMemcpy( &image_h[0], image_d, (size_t)(width*height)*sizeof(optix::float3), cudaMemcpyDeviceToHost );
    if( err != cudaSuccess )
    {
      printf( "cudaMemcpy failed (%s): %s\n", cudaGetErrorName( err ), cudaGetErrorString( err ) );
      exit( 1 );
    }

    writePPM( "outputTranslated.ppm", &image_h[0].x, width, height );

    // Clean up
    cudaFree( rays_d );
    cudaFree( hits_d );
    cudaFree( image_d );
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  return 0;
}

