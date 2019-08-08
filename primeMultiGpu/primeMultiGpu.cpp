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
//  Minimal demonstration of handling multiple GPUs by allocating an
//  OptiX Prime context for each.
//
//-----------------------------------------------------------------------------

#include <primeCommon.h>
#include <optix_prime/optix_primepp.h>
#include <optixu/optixu_math_namespace.h>
#include <sutil.h>
#include <memory.h>

using namespace optix::prime;

//------------------------------------------------------------------------------
// This class allocates a context and necessary buffer for each device. Optix 
// Prime handles multi-GPU configurations automatically, but ray and hit
// buffers must come from a single location. This means that data needs to be
// copied between devices over the PCIe bus, which imposes a limit on
// performance. When each device has its own context and buffers, maximum 
// scalability can be achieved.
class MultiGpuManager
{
public:
  MultiGpuManager() 
    : m_width(0)
    , m_height(0)
    , m_rays_d(0)
    , m_hits_d(0)    
  {}

  ~MultiGpuManager()
  {
    delete[] m_rays_d;
    delete[] m_hits_d;
  }

  void init()
  {
    int deviceCount = 0;
    CHK_CUDA( cudaGetDeviceCount( &deviceCount ) );
    for( unsigned i=0; i < (unsigned)deviceCount; ++i )
    {
      m_contexts.push_back( Context::create( RTP_CONTEXT_TYPE_CUDA ) );
      m_contexts[i]->setCudaDeviceNumbers( 1, &i );
      m_models.push_back( m_contexts[i]->createModel() );
      m_queries.push_back( m_models[i]->createQuery( RTP_QUERY_TYPE_CLOSEST ));
    }
    delete[] m_rays_d;
    delete[] m_hits_d;
    m_rays_d = new Buffer<Ray>[deviceCount];
    m_hits_d = new Buffer<Hit>[deviceCount];
  }

  void createModel( int numTriangles, int3* indices, int numVertices, float3* vertices )
  {
    // Create the model in the first context
    m_models[0]->setTriangles( numTriangles, RTP_BUFFER_TYPE_HOST,  indices,
                               numVertices,  RTP_BUFFER_TYPE_HOST,  vertices );
    m_models[0]->update( RTP_MODEL_HINT_ASYNC );

    // Copy the models to the other contexts. The copy is performed 
    // asynchronously. Alternatively the context for each device could update
    // the model in parallel.
    for( size_t i=1; i < m_models.size(); ++i )
      m_models[i]->copy( m_models[0] );

    // Ensure that all computation on models is complete so that indices and 
    // vertices buffers can be safely modified. It is not necessary to finish 
    // the copies because they do not use the buffers. 
    m_models[0]->finish();
  }

  void createRaysOrtho( int width, int* height,
     const float3& bbmin, const float3& bbmax, float margin )
  {
    m_width = width;
    for( size_t i=0; i < m_contexts.size(); i++ )
    {
      CHK_CUDA( cudaSetDevice(int(i)) );

      // Use a simple load balancing scheme that distributes rows from the image
      // round-robin between the devices. This assumes that the devices have
      // roughly equal computational power.
      m_rays_d[i].alloc( 0, RTP_BUFFER_TYPE_CUDA_LINEAR ); // set buffer type
      ::createRaysOrtho( m_rays_d[i], m_width, &m_height, bbmin, bbmax, margin, 0, int(i), int(m_contexts.size()) );

      m_hits_d[i].alloc( m_rays_d[i].count(), RTP_BUFFER_TYPE_CUDA_LINEAR );
    }
    *height = m_height;
    m_hits_h.alloc( m_width*m_height, RTP_BUFFER_TYPE_HOST, UNLOCKED );
    m_temp_h.alloc( m_width*m_height, RTP_BUFFER_TYPE_HOST, LOCKED );
  }

  void translateRays( const float3& offset )
  {
    for( size_t i=0; i < m_contexts.size(); i++ )
    {
      CHK_CUDA( cudaSetDevice(int(i)) );
      ::translateRays( m_rays_d[i], offset );
    }
  }

  // Returns a pointer to the internal hit buffer. Do not delete it.
  Buffer<Hit>* queryExecute()
  {   
    std::vector<size_t> tempOffset( m_contexts.size() );
    for( size_t i=0; i < m_contexts.size(); ++i )
    {
      CHK_CUDA( cudaSetDevice(int(i)) );
      m_queries[i]->setRays( m_rays_d[i].count(), Ray::format, m_rays_d[i].type(), m_rays_d[i].ptr() );
      m_queries[i]->setHits( m_hits_d[i].count(), Hit::format, m_hits_d[i].type(), m_hits_d[i].ptr() );

      // Execute the query asynchronously on each context so that they can 
      // execute in parallel. 
      m_queries[i]->execute( RTP_QUERY_HINT_ASYNC );

      // We need to send the hits back to a temp buffer on the host to merge the results from each
      // context. We could have configured the query to place the hits directly in
      // the temp buffer. However, we manually copy the results back here to 
      // demonstrate adding extra asynchronous operations after the query. To 
      // ensure that all asynchronous operations have completed we will need to use
      // CUDA to synchronize. 
      if( i > 0 ) 
        tempOffset[i] = tempOffset[i-1] + m_hits_d[i-1].count();
      CHK_CUDA( cudaMemcpyAsync( m_temp_h.ptr()+tempOffset[i], m_hits_d[i].ptr(), m_hits_d[i].sizeInBytes(), cudaMemcpyDeviceToHost ) );
    }

    // Synchronize the default stream on all contexts.
    for( size_t i=0; i < m_contexts.size(); ++i )
    {
      CHK_CUDA( cudaSetDevice(int(i)) );
      CHK_CUDA( cudaStreamSynchronize( 0 ) );
    }

    // Merge results
    size_t i=0;
    for( int y=0; y < m_height; ++y )
    {
      memcpy( m_hits_h.ptr() + y*m_width, m_temp_h.ptr() + tempOffset[i], m_width*sizeof(Hit));
      tempOffset[i] += m_width;
      if( ++i >= m_contexts.size() )
        i = 0;
    }
    
    return &m_hits_h;
  }

private:
  int m_width;            // image width
  int m_height;           // image height
  Buffer<Hit>   m_hits_h; // hits on the host 
  Buffer<Hit>   m_temp_h; // temp host buffer for merging hits from multiple GPUs
  Buffer<Ray>*  m_rays_d; // array of ray buffers per device
  Buffer<Hit>*  m_hits_d; // array of hit buffers per device

  // Per-device API objects
  std::vector<Context> m_contexts;
  std::vector<Model>   m_models;
  std::vector<Query>   m_queries;
};

//------------------------------------------------------------------------------
void printUsageAndExit( const char* argv0 )
{
  std::cerr
  << "Usage  : " << argv0 << " [options]\n"
  << "App options:\n"
  << "  -h  | --help                               Print this usage message\n"
  << "  -o  | --obj <obj_file>                     Specify model to be rendered\n"
  << "  -w  | --width <number>                     Specify output image width\n"
  << std::endl;
  
  exit(1);
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
  // set defaults
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
    PrimeMesh mesh;
    loadMesh( objFilename, mesh );

    MultiGpuManager manager;
    manager.init();
    manager.createModel( mesh.num_triangles, mesh.getVertexIndices(),  
                         mesh.num_vertices,  mesh.getVertexData() );
    manager.createRaysOrtho( width, &height, mesh.getBBoxMin(), mesh.getBBoxMax(), 0.05f );
    Buffer<Hit>* hits = manager.queryExecute();

    //
    // Shade the hit results to create image.
    //
    std::vector<float3> image( width * height );
    shadeHits( image, *hits, mesh );
    writePpm( "output.ppm", &image[0].x, width, height );

    //
    // Re-execute query with different rays.
    //
    float3 extents = mesh.getBBoxMax() - mesh.getBBoxMin();
    manager.translateRays( extents * make_float3(0.2f, 0, 0) );
    hits = manager.queryExecute();
    shadeHits( image, *hits, mesh );
    freeMesh( mesh );
    writePpm( "outputTranslated.ppm", &image[0].x, width, height );
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    exit(1);
  }
}
