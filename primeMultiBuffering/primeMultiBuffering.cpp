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
//  Minimal demonstration of using multi-buffering with Optix Prime. Multi-buffering
//  serves two purposes:
//     1) Overlap CPU computation with operations on the GPU
//     2) Conserve page-locked host memory. Larger computations can be staged
//        through a smaller amount page-locked memory
//
//-----------------------------------------------------------------------------

#include <primeCommon.h>
#include <optix_prime/optix_primepp.h>
#include <sutil.h>
#include <memory.h>
#include <algorithm>

using namespace optix::prime;

//------------------------------------------------------------------------------
size_t getBufferSize( RTPbufferformat format, size_t count )
{
  switch( format )
  {
  case RTP_BUFFER_FORMAT_INDICES_INT3:
    return sizeof(int3)*count;
  case RTP_BUFFER_FORMAT_VERTEX_FLOAT3:
    return sizeof(float3)*count;
  case RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION:        
    return 2*sizeof(float3)*count;
  case RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX:
    return 2*sizeof(float4)*count;
  case RTP_BUFFER_FORMAT_HIT_BITMASK:
    return ((count + 31)/32)*sizeof(int);
  case RTP_BUFFER_FORMAT_HIT_T:
    return sizeof(float)*count;
  case RTP_BUFFER_FORMAT_HIT_T_TRIID:
    return (sizeof(float) + sizeof(int))*count;
  case RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V:
    return (3*sizeof(float) + sizeof(int))*count;
  default:
    std::cerr << "Unknown format\n";
    exit(1);
  }

  return 0;
}

//------------------------------------------------------------------------------
class MultiBufferQueryManager
{
public:
  struct BufferSet
  {
    void* rays;
    void* hits;
    size_t maxCount;
    size_t resultCount;
    void* userData;
  };

  MultiBufferQueryManager( int numBufferSets, size_t maxCount, 
     RTPbufferformat rayFormat, RTPbufferformat hitFormat,
     Model model, RTPquerytype queryType )
    : m_rayFormat(rayFormat)
    , m_hitFormat(hitFormat)
    , m_pendingCount(0)
    , m_pos(0)
    , m_lastPos(-1)
  {
    m_bufferSets.resize( numBufferSets );
    m_queries.resize( numBufferSets );
    m_queryFinished.resize( numBufferSets, true );
    size_t rayBufferSize = getBufferSize( m_rayFormat, maxCount );
    size_t hitBufferSize = getBufferSize( m_hitFormat, maxCount );
    for( int i=0; i < numBufferSets; ++i )
    {
      m_queries[i] = model->createQuery( queryType );
      m_bufferSets[i].rays = new char[rayBufferSize];
      m_bufferSets[i].hits = new char[hitBufferSize];
      m_bufferSets[i].maxCount = maxCount;
      m_bufferSets[i].resultCount = 0;
      m_bufferSets[i].userData = 0;

      rtpHostBufferLock( m_bufferSets[i].rays, rayBufferSize );
      rtpHostBufferLock( m_bufferSets[i].hits, hitBufferSize );
    }
  }

  ~MultiBufferQueryManager()
  {
    for( int i=0; i < (int)m_bufferSets.size(); ++i )
    {
      if( !m_queryFinished[i] )
        m_queries[i]->finish();

      rtpHostBufferUnlock( m_bufferSets[i].rays );
      rtpHostBufferUnlock( m_bufferSets[i].hits );
      delete[] (char*)m_bufferSets[i].rays;
      delete[] (char*)m_bufferSets[i].hits;
    }
  }

  // Get a set of ray and hit buffers. If all BufferSets are in use with 
  // pending queries this function will block until a query is finished. If
  // bufferSet->resultCount > 0 then  the BufferSet contains results from a 
  // previous query. bufferSet->userData can be used to tag the results with
  // information pertaining to a particular query. After processing
  // previous results, if any, the rays buffer can filled with new data and a
  // new query issued with executeQueryOnLastBufferSet().
  //
  // Returns true if other BufferSets with results are pending.
  bool getBufferSet( BufferSet* bufferSet )
  {
    finishQuery( m_pos );
    *bufferSet = m_bufferSets[m_pos];
    m_lastPos = m_pos;
    if( ++m_pos >= (int)m_bufferSets.size() )
      m_pos = 0;

    return m_pendingCount > 0;
  }


  // Execute query using ray buffer returned with the last call to getBufferSet().
  void executeQueryOnLastBufferSet( size_t count, void* userData )
  {
    if( m_lastPos < 0 )
    {
      std::cerr << "Call getBufferSet() first\n";
      return;
    }

    finishQuery( m_lastPos );

    BufferSet& bs = m_bufferSets[m_lastPos];
    bs.resultCount = count;

    Query& query = m_queries[m_lastPos];
    query->setRays( count, m_rayFormat, RTP_BUFFER_TYPE_HOST, bs.rays );
    query->setHits( count, m_hitFormat, RTP_BUFFER_TYPE_HOST, bs.hits );
    query->execute( RTP_QUERY_HINT_ASYNC );

    m_queryFinished[m_lastPos] = false;
    m_pendingCount++;
  }

private:
  RTPbufferformat m_rayFormat;
  RTPbufferformat m_hitFormat;
  std::vector<BufferSet> m_bufferSets;
  std::vector<Query>    m_queries;
  std::vector<bool>     m_queryFinished;
  int m_pendingCount;
  int m_pos;
  int m_lastPos;

  void finishQuery( int pos )
  {
    if( !m_queryFinished[pos] )
    {
      m_queries[pos]->finish();
      m_queryFinished[pos] = true;
      m_pendingCount--;
    }
  }
};

//------------------------------------------------------------------------------
void printUsageAndExit( const char* argv0 )
{
  std::cerr
  << "Usage  : " << argv0 << " [options]\n"
  << "App options:\n"
  << "  -h  | --help                Print this usage message\n"
  << "  -o  | --obj <obj_file>      Specify model to be rendered\n"
  << "  -b  | --buffers <number>    Number of buffer sets. Default is 2.\n"
  << "  -c  | --count <number>      Max count for each buffer. Default is 65536.\n"
  << "  -w  | --width <number>      Specify output image width\n"
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
  int numBufferSets = 2;
  size_t maxCount = 64*1024;

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
    else if( (arg == "-b" || arg == "--buffers") && i+1 < argc ) 
    {
      numBufferSets = atoi(argv[++i]);
    } 
    else if( (arg == "-c" || arg == "--count") && i+1 < argc ) 
    {
      maxCount = (size_t)atoi(argv[++i]);
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
    Context context = Context::create(RTP_CONTEXT_TYPE_CUDA);
    unsigned int device = 0;
    context->setCudaDeviceNumbers(1, &device);

    //
    // Create the Model object
    //
    PrimeMesh mesh;
    loadMesh( objFilename, mesh );

    Model model = context->createModel();
    model->setTriangles( mesh.num_triangles, RTP_BUFFER_TYPE_HOST, mesh.getVertexIndices(), 
                         mesh.num_vertices,  RTP_BUFFER_TYPE_HOST, mesh.positions );
    model->update( 0 );

    //
    // Create buffers for rays and hits
    //
    Buffer<Ray> rays( 0, RTP_BUFFER_TYPE_HOST );
    createRaysOrtho( rays, width, &height, mesh.getBBoxMin(), mesh.getBBoxMax(), 0.05f );
    Buffer<Hit> hits( rays.count(), RTP_BUFFER_TYPE_HOST );
    
    // 
    // Execute queries with multi-buffering to stage data through page-locked host
    // memory
    // 
    MultiBufferQueryManager manager( numBufferSets, maxCount, Ray::format, Hit::format,
      model, RTP_QUERY_TYPE_CLOSEST );
    size_t count = 0;
    size_t finishedCount = 0;
    while( finishedCount < hits.count() )
    {
      MultiBufferQueryManager::BufferSet bufferSet;
      manager.getBufferSet( &bufferSet );
      
      // process results
      if( bufferSet.resultCount > 0 )
      {
        memcpy( hits.ptr()+finishedCount, bufferSet.hits, bufferSet.resultCount*sizeof(Hit) );
        finishedCount += bufferSet.resultCount;
      }

      // prepare query
      if( count < rays.count() )
      {
        size_t queryCount = std::min( bufferSet.maxCount, rays.count()-count );
        memcpy( bufferSet.rays, rays.ptr()+count, queryCount*sizeof(Ray) );
        manager.executeQueryOnLastBufferSet( queryCount, 0 );
        count += queryCount;
      }
    }

    //
    // Shade the hit results to create image.
    //
    std::vector<float3> image( width * height );
    shadeHits( image, hits, mesh );

    freeMesh( mesh );

    writePpm( "output.ppm", &image[0].x, width, height );

  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    exit(1);
  }
}
