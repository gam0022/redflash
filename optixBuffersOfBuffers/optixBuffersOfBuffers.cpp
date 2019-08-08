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
//  optixBuffersOfBuffers.cpp - simple test for nested buffers 
//
//-----------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <sutil.h>
#include "common.h"
#include "random.h"
#include <iostream>
#include <cstdlib>
#include <cstring>

using namespace optix;

const char* const SAMPLE_NAME = "optixBuffersOfBuffers";

int width  = 512;
int height = 512;

const unsigned int MAX_BUFFER_WIDTH  = 64u;
const unsigned int MAX_BUFFER_HEIGHT = 32u;

//-----------------------------------------------------------------------------
// 
// Helpers 
//
//-----------------------------------------------------------------------------

float randf() 
{
  static unsigned int seed = 123u;
  return rnd( seed );
}


// Create an randomly sized image buffer with randomly colored dots in it
Buffer createRandomBuffer(
    Context      context,
    unsigned int max_width,
    unsigned int max_height )
{
  const float scale = randf();
  const unsigned width  = max(static_cast<unsigned>(max_width*scale), 1u);
  const unsigned height = max(static_cast<unsigned>(max_height*scale), 1u);

  Buffer buffer = context->createBuffer(
      RT_BUFFER_INPUT,
      RT_FORMAT_UNSIGNED_BYTE4,
      width,
      height );
  uchar4* data = reinterpret_cast<uchar4*>( buffer->map() );

  // generate some random dots
  float red   = randf();
  float blue  = randf();
  float green = randf();
  for( unsigned int i = 0; i < height*width; ++i )
  {
    if( randf() < 0.1f )
    {
      data[i].x = (unsigned char)( red  * 255.0f );  // R
      data[i].y = (unsigned char)( green* 255.0f );  // G
      data[i].z = (unsigned char)( blue * 255.0f );  // B
      data[i].w = 255;                               // A
    }
    else
    {
      data[i].x = 255; // R
      data[i].y = 255; // G
      data[i].z = 255; // B
      data[i].w = 0;   // A
    }
  }

  buffer->unmap();

  return buffer;
}

void setBufferIds( const std::vector<optix::Buffer>& buffers,
                   optix::Buffer top_level_buffer )
{
  top_level_buffer->setSize( buffers.size() );
  int* data = reinterpret_cast<int*>( top_level_buffer->map() );
  for( unsigned i = 0; i < buffers.size(); ++i )
    data[i] = buffers[i]->getId();
  top_level_buffer->unmap();
}

Context createContext() 
{
  // Set up context
  Context context = Context::create();
  context->setRayTypeCount( 2 );
  context->setEntryPointCount( 1 );
  context->setStackSize( 1200 );
  context->setMaxTraceDepth( 2 );

  context["max_depth"]->setInt( 5 );
  context["scene_epsilon"]->setFloat( 1.e-4f );

  Variable output_buffer = context["output_buffer"];
  Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height );
  output_buffer->set(buffer);

  // Camera
  float3 cam_eye = { 2.0f, 1.5f, -2.0f };
  float3 lookat  = { 0.0f, 1.2f,  0.0f };
  float3 up      = { 0.0f, 1.0f, 0.0f };
  float  hfov    = 60.0f;
  float  aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
  float3 camera_u, camera_v, camera_w;
  sutil::calculateCameraVariables(
          cam_eye, lookat, up, hfov, aspect_ratio,
          camera_u, camera_v, camera_w );

  context["eye"]->setFloat( cam_eye );
  context["U"]->setFloat( camera_u );
  context["V"]->setFloat( camera_v );
  context["W"]->setFloat( camera_w );

  // Ray generation program
  const char *ptx = sutil::getPtxString( SAMPLE_NAME, "pinhole_camera.cu" );
  Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
  context->setRayGenerationProgram( 0, ray_gen_program );

  // Exception program
  Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
  context->setExceptionProgram( 0, exception_program );
  context["bad_color"]->setFloat( 1.0f, 1.0f, 0.0f );

  // Miss program
  context->setMissProgram( 0, context->createProgramFromPTXString( sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" ), "miss" ) );
  context["bg_color"]->setFloat( 
      make_float3(108.0f/255.0f, 166.0f/255.0f, 205.0f/255.0f) * 0.5f );

  return context;
}

void createScene( Context context, unsigned num_buffers )
{
  // Sphere
  Geometry sphere = context->createGeometry();
  sphere->setPrimitiveCount( 1u );
  const char *ptx = sutil::getPtxString( SAMPLE_NAME, "sphere_texcoord.cu" );
  sphere->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
  sphere->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
  sphere["sphere"      ]->setFloat( 0.0f, 1.2f, 0.0f, 1.0f );
  sphere["matrix_row_0"]->setFloat( 1.0f, 0.0f, 0.0f );
  sphere["matrix_row_1"]->setFloat( 0.0f, 1.0f, 0.0f );
  sphere["matrix_row_2"]->setFloat( 0.0f, 0.0f, 1.0f );

  // Floor
  Geometry parallelogram = context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
  parallelogram->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
  parallelogram->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
  float3 anchor = make_float3( -20.0f, 0.01f, 20.0f);
  float3 v1 = make_float3( 40, 0, 0);
  float3 v2 = make_float3( 0, 0, -40);
  float3 normal = cross( v1, v2 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );
  float4 plane = make_float4( normal, d );
  parallelogram["plane"]->setFloat( plane );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["anchor"]->setFloat( anchor );

  // Sphere material
  ptx = sutil::getPtxString( SAMPLE_NAME, "optixBuffersOfBuffers.cu" );
  Program bob_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
  Program bob_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
  Material sphere_matl = context->createMaterial();
  sphere_matl->setClosestHitProgram( 0, bob_ch );
  sphere_matl->setAnyHitProgram( 1, bob_ah );
  
  std::vector<optix::Buffer> buffers( num_buffers );
  for( unsigned i = 0u; i < buffers.size(); ++i ) {
    buffers[i] = createRandomBuffer( context, MAX_BUFFER_WIDTH, MAX_BUFFER_HEIGHT );
  }

  optix::Buffer top_level_buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_BUFFER_ID );
  setBufferIds( buffers, top_level_buffer );
  sphere_matl["Kd_layers"]->set( top_level_buffer );

  // Floor material
  ptx = sutil::getPtxString( SAMPLE_NAME, "phong.cu" );
  Program phong_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
  Program phong_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
  Material floor_matl = context->createMaterial();
  floor_matl->setClosestHitProgram( 0, phong_ch);
  floor_matl->setAnyHitProgram( 1, phong_ah );

  floor_matl["Kd"           ]->setFloat( 0.7f, 0.7f, 0.7f );
  floor_matl["Ka"           ]->setFloat( 1.0f, 1.0f, 1.0f );
  floor_matl["Kr"           ]->setFloat( 0.0f, 0.0f, 0.0f );
  floor_matl["phong_exp"    ]->setFloat( 1.0f );
  
  // Place geometry into hierarchy
  std::vector<GeometryInstance> gis;
  gis.push_back( context->createGeometryInstance( sphere, &sphere_matl, &sphere_matl+1 ) );
  gis.push_back( context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 ) );

  GeometryGroup geometrygroup = context->createGeometryGroup();
  geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
  for( unsigned i = 0; i < gis.size(); ++i ) {
    geometrygroup->setChild( i, gis[i] );
  }
  geometrygroup->setAcceleration( context->createAcceleration("NoAccel") );

  context["top_object"]->set( geometrygroup );
  context["top_shadower"]->set( geometrygroup );

  // Setup lights
  context["ambient_light_color"]->setFloat(0.1f,0.1f,0.1f);
  BasicLight lights[] = { 
    { { 0.0f, 8.0f, -5.0f }, { .4f, .4f, .4f }, 1 },
    { { 5.0f, 8.0f,  0.0f }, { .4f, .4f, .4f }, 1 }
  };

  Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(BasicLight));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  context["lights"]->set(light_buffer);
}


void printUsageAndExit( const std::string& argv0 )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "Options:\n"
    << "  --help | -h             Print this usage message\n"
    << "  --num-buffers | -n      Set number of buffers (defaults to 4)\n"
    << "  --file | -f <filename>  Specify file for image output\n"
    << "  --dim=<width>x<height>  Set image dimensions; defaults to " << width << "x" << height << "\n";

  exit(1);
}

int main( int argc, char** argv )
{

  std::string outfile;
  unsigned num_buffers = 4u;

  for(int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if(arg == "-n" || arg == "--num-buffers") {
      if( i < argc-1 ) {
        num_buffers = atoi( argv[++i] );
      } else {
        printUsageAndExit( argv[0] );
      }
    } else if ( arg.substr( 0, 6 ) == "--dim=" ) {
      std::string dims_arg = arg.substr(6);
      sutil::parseDimensions( dims_arg.c_str(), width, height );
    } else if( arg == "--file" || arg == "-f" ) {
      if( i < argc-1 ) {
        outfile = argv[++i];
      } else {
        printUsageAndExit( argv[0] );
      }
    } else if( arg == "-h" || arg == "--help" ) {
      printUsageAndExit( argv[0] );
    } else {
      std::cerr << "Unknown option '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( outfile.empty() ) {
      sutil::initGlut( &argc, argv );
  }

  try {
    
    Context context = createContext();
    createScene( context, num_buffers );

    context->validate();
    context->launch( 0, width, height );

    // DisplayImage
    if( outfile.empty() ) {
      sutil::displayBufferGlut( argv[0], context["output_buffer"]->getBuffer() );
    } else {
      sutil::displayBufferPPM( outfile.c_str(), context["output_buffer"]->getBuffer() );
    }

  return 0;

  } SUTIL_CATCH( 0 ) 
}

