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
//  OptiX Prime instancing usage demonstration
//
//-----------------------------------------------------------------------------

#include <primeCommon.h>
#include <optix_prime/optix_primepp.h>
#include <optixu/optixu_math_namespace.h>
#include <sutil.h>
#include <cuda/random.h>

using namespace optix::prime;
using optix::fmaxf;

namespace {
  float3 make_float3( const float* a )
  {
    return ::make_float3( a[0], a[1], a[2] );
  }
}

//------------------------------------------------------------------------------
inline float genRndFloat(float from, float to)
{
  static unsigned int seed = 1984u;
  return from + (to - from)*rnd(seed);
}

//------------------------------------------------------------------------------
inline int genRndInt(int to)
{
  static unsigned int seed = 42u;
  return lcg(seed) % to;
}

//------------------------------------------------------------------------------
void createTransform(SimpleMatrix4x3& transform, SimpleMatrix4x3& invTransform, float range, float scale, float scaleVariance = 0.3f)
{
  // uniform scale   
  float sx, sy, sz;
  sx = sy = sz = scale * genRndFloat(1-scaleVariance/2, 1+scaleVariance/2);

  // rotation
  float angle = genRndFloat(0, float(2*M_PI));
  float c = cosf(angle);
  float s = sinf(angle);

  // translation
  float tx = genRndFloat(-range, range);
  float tz = genRndFloat(-range, range);

  transform = SimpleMatrix4x3(
    c*sx,   0, -s*sx,  tx,
       0,  sy,     0,   0,
    s*sz,   0,  c*sz,  tz
  );

  invTransform = SimpleMatrix4x3(
     c/sx,    0,  s/sx,  -(c*tx + s*tz)/sx,
        0, 1/sy,     0,                  0,
    -s/sz,    0,  c/sz, -(-s*tx + c*tz)/sz
  );
}


//------------------------------------------------------------------------------
void printUsageAndExit(const char* argv0)
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -o  | --obj <obj_file>                     Specify model to be rendered\n"
    << "  -c  | --context [cpu|(cuda)]               Specify context type. Default is cuda\n"
    << "  -b  | --buffer [(host)|cuda]               Specify buffer type. Default is host\n"
    << "  -i  | --num-instances <num_instances>      Specify the number of instances to be rendered. Must be > 0\n"
    << std::endl;

  exit(1);
}


//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
  // set defaults
  RTPcontexttype contextType = RTP_CONTEXT_TYPE_CUDA;
  RTPbuffertype bufferType = RTP_BUFFER_TYPE_HOST;
  std::string objFilename;
  int width = 1024;
  int height = 768;
  int numInstances = 10000;
  float sceneSize = 1;

  // parse arguments
  for (int i = 1; i < argc; ++i)
  {
    std::string arg(argv[i]);
    if (arg == "-h" || arg == "--help")
    {
      printUsageAndExit(argv[0]);
    }
    else if ((arg == "-c" || arg == "--context") && i + 1 < argc)
    {
      std::string param(argv[++i]);
      if (param == "cpu")
        contextType = RTP_CONTEXT_TYPE_CPU;
      else if (param == "cuda")
        contextType = RTP_CONTEXT_TYPE_CUDA;
      else
        printUsageAndExit(argv[0]);
    }
    else if (arg == "-i" || arg == "--num-instances") {
      if (i == argc - 1)
        printUsageAndExit(argv[0]);
      numInstances = atoi(argv[++i]);
      if (numInstances <= 0)
        printUsageAndExit(argv[0]);
    }
    else if (arg == "-o" || arg == "--obj") {
      if (i == argc - 1)
        printUsageAndExit(argv[0]);
      objFilename = std::string(argv[++i]);
    }
    else if ((arg == "-b" || arg == "--buffer") && i + 1 < argc)
    {
      std::string param(argv[++i]);
      if (param == "host")
        bufferType = RTP_BUFFER_TYPE_HOST;
      else if (param == "cuda")
        bufferType = RTP_BUFFER_TYPE_CUDA_LINEAR;
      else
        printUsageAndExit(argv[0]);
    }
    else
    {
      std::cerr << "Bad option: '" << arg << "'" << std::endl;
      printUsageAndExit(argv[0]);
    }
  }

  try {
    //
    // Create Prime context
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
    // Load meshes
    //
    std::vector<PrimeMesh> meshes;
    std::vector<float> baseScale; 
    if (objFilename.empty())
    {
      // Load grass and daisies
      std::string filenameBase = std::string(sutil::samplesDir()) + "/data/";
      const std::string filenames[] = 
      {
        "daisy1.obj",
        "daisy2.obj",
        "daisy3.obj",
        "daisy4.obj",
        "daisy5.obj",
        "daisy6.obj",
        "grass1.obj",
        "grass2.obj",
        "grass3.obj",
        "grass4.obj",
        "grass5.obj",
        "grass6.obj"
      };
      size_t numFiles = sizeof(filenames) / sizeof(filenames[0]);
      meshes.resize(numFiles);
      baseScale.resize(numFiles, 1.0f);
      for (size_t i = 0; i < numFiles; ++i)
      {
        loadMesh( filenameBase + filenames[i], meshes[i] );
        if (filenames[i].substr(0,5) == "daisy")
          baseScale[i] *= 1.5f; // make daisies taller
      }
    }
    else 
    {
      // Load the specified model
      baseScale.resize(1, 1.0f);
      meshes.resize(1);
      loadMesh( objFilename, meshes[0] );
    }

    // Adjust scale factor based on size of the mesh.
    float spacing = float(2 * sceneSize / 30);
    for( size_t i = 0; i < baseScale.size(); ++i )
    {
      float3 dim = make_float3( meshes[i].bbox_max ) - 
                   make_float3( meshes[i].bbox_min );
      float maxDim = fmaxf(fmaxf(dim.x,dim.y),dim.z);
      baseScale[i] *= spacing / maxDim;
    }

    // 
    // Create models from meshes
    //
    std::vector<Model> models(meshes.size());
    for (size_t i = 0; i < models.size(); ++i)
    {
      models[i] = context->createModel();
      models[i]->setTriangles(meshes[i].num_triangles, RTP_BUFFER_TYPE_HOST, meshes[i].tri_indices, 
                              meshes[i].num_vertices,  RTP_BUFFER_TYPE_HOST, meshes[i].positions );
      models[i]->update(0);      
    }

    //
    // Create a list of random instances
    //
    std::vector<RTPmodel> instances(numInstances);
    std::vector<int> modelIds(numInstances);
    std::vector<SimpleMatrix4x3> transforms(numInstances);
    std::vector<SimpleMatrix4x3> invTransforms(numInstances);
    for (int i = 0; i < numInstances; ++i)
    {
      int modelId = genRndInt((int)models.size());
      modelIds[i]  = modelId;
      instances[i] = models[modelId]->getRTPmodel();
      createTransform(transforms[i], invTransforms[i], sceneSize, baseScale[modelId]);
    }

    //
    // Assemble instances into a single scene
    //
    Model scene = context->createModel();
    scene->setInstances(numInstances, RTP_BUFFER_TYPE_HOST, &instances[0], 
                        RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3, RTP_BUFFER_TYPE_HOST, &transforms[0] );
    scene->update(0);

    //
    // Create buffers for rays and hits
    //
    float3 center = make_float3(0, 0, 0);
    float3 eye = sceneSize/3 * make_float3(1.0f, 0.5f, 1.0f);
    Buffer<Ray> rays(0, bufferType, LOCKED);
    createRaysPersp(rays, width, height, eye, center, 90.0f);
    Buffer<HitInstancing> hits(rays.count(), RTP_BUFFER_TYPE_HOST);

    //
    // Execute query
    //
    Query query = scene->createQuery(RTP_QUERY_TYPE_CLOSEST);
    query->setRays(rays.count(), Ray::format, rays.type(), rays.ptr());
    query->setHits(hits.count(), HitInstancing::format, hits.type(), hits.ptr());
    query->execute(0);

    //
    // Shade the hit results to create image
    //
    std::vector<float3> image(width * height);
    shadeHits(image, hits, modelIds, meshes, eye, invTransforms );
    writePpm("output.ppm", &image[0].x, width, height);

    for( int i = 0; i < static_cast<int>( meshes.size() ); ++i )
        freeMesh( meshes[i] );
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    exit(1);
  }
}
