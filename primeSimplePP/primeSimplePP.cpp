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
//  Minimal OptiX Prime C++ wrapper usage demonstration  
//
//-----------------------------------------------------------------------------

#include <primeCommon.h>
#include <optix_prime/optix_primepp.h>
#include <optixu/optixu_math_namespace.h>
#include <sutil.h>


using namespace optix::prime;


//------------------------------------------------------------------------------
void printUsageAndExit( const char* argv0 )
{
  std::cerr
  << "Usage  : " << argv0 << " [options]\n"
  << "App options:\n"
  << "  -h  | --help                               Print this usage message\n"
  << "  -o  | --obj <obj_file>                     Specify model to be rendered\n"
  << "  -c  | --context [cpu|(cuda)]               Specify context type. Default is cuda\n"
  << "  -b  | --buffer [(host)|cuda]               Specify buffer type. Default is host\n"
  << "  -w  | --width <number>                     Specify output image width\n"
  << std::endl;
  
  exit(1);
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
  // set defaults
  RTPcontexttype contextType = RTP_CONTEXT_TYPE_CUDA;
  RTPbuffertype bufferType = RTP_BUFFER_TYPE_HOST;
  std::string objFilename = std::string( sutil::samplesDir() ) + "/data/cow.obj";
  int width = 640;
  int height = 0;

  // parse arguments
  for ( int i = 1; i < argc; ++i ) 
  { 
    std::string arg( argv[i] );
    if( arg == "-h" || arg == "--help" ) 
    {
      printUsageAndExit( argv[0] ); 
    } 
    else if( (arg == "-o" || arg == "--obj") && i+1 < argc ) 
    {
      objFilename = argv[++i];
    } 
    else if( ( arg == "-c" || arg == "--context" ) && i+1 < argc )
    {
      std::string param( argv[++i] );
      if( param == "cpu" )
        contextType = RTP_CONTEXT_TYPE_CPU;
      else if( param == "cuda" )
        contextType = RTP_CONTEXT_TYPE_CUDA;
      else
        printUsageAndExit( argv[0] );
    } 
    else if( ( arg == "-b" || arg == "--buffer" ) && i+1 < argc )
    {
      std::string param( argv[++i] );
      if( param == "host" )
        bufferType = RTP_BUFFER_TYPE_HOST;
      else if( param == "cuda" )
        bufferType = RTP_BUFFER_TYPE_CUDA_LINEAR;
      else
        printUsageAndExit( argv[0] );
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

  try {
    //
    // Create Context
    //
    Context context = Context::create(contextType);
    if (contextType == RTP_CONTEXT_TYPE_CPU) {
      std::cerr << "Using cpu context\n";
    } else {
      unsigned int device = 0;
      context->setCudaDeviceNumbers(1, &device);
      std::cerr << "Using cuda context\n";
    }

    //
    // Create the Model object
    //
    PrimeMesh mesh;
    loadMesh( objFilename, mesh );

    Model model = context->createModel();
    model->setTriangles(mesh.num_triangles, RTP_BUFFER_TYPE_HOST, mesh.getVertexIndices(), 
                        mesh.num_vertices,  RTP_BUFFER_TYPE_HOST, mesh.getVertexData() );
    model->update( 0 );

    //
    // Create buffers for rays and hits
    //
    Buffer<Ray> rays( 0, bufferType, LOCKED );
    createRaysOrtho( rays, width, &height, mesh.getBBoxMin(), mesh.getBBoxMax(), 0.05f );
    Buffer<Hit> hits( rays.count(), bufferType, LOCKED );

    //
    // Execute query
    //
    Query query = model->createQuery(RTP_QUERY_TYPE_CLOSEST);
    query->setRays( rays.count(), Ray::format, rays.type(), rays.ptr() );
    query->setHits( hits.count(), Hit::format, hits.type(), hits.ptr() );
    query->execute( 0 );

    //
    // Shade the hit results to create image
    //
    std::vector<float3> image( width * height );
    shadeHits( image, hits, mesh );
    writePpm( "output.ppm", &image[0].x, width, height );

    //
    // Re-execute query with different rays
    //
    float3 extents = mesh.getBBoxMax() - mesh.getBBoxMin();
    translateRays( rays, extents * make_float3(0.2f, 0, 0) );
    query->execute( 0 );
    shadeHits( image, hits, mesh );
    freeMesh( mesh );
    writePpm( "outputTranslated.ppm", &image[0].x, width, height );

  } 
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    exit(1);
  }
}
