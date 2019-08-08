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
//  optixConsole.cpp - Rendered image appears on the console as ASCII art with no GL or 
//                third party libraries.
//
//-----------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include "common.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sutil.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixConsole";

//-----------------------------------------------------------------------------
//
// Manta Scene
//
//-----------------------------------------------------------------------------

class ConsoleScene
{
public:
  void   initScene();
  void   trace();
  void   cleanUp();
  Buffer getOutputBuffer();
  void createOutputBuffer();

  void display( std::ostream& ofs );

  void createGeometry();

private:
  Context m_context;

  void calculateCameraParameters(float3 eye, float3 lookat, float3 up, float vfov, float hfov,
    float3& U, float3& V, float3& W);

  static unsigned int WIDTH;
  static unsigned int HEIGHT;
};

unsigned int ConsoleScene::WIDTH  = 48u*2u;
unsigned int ConsoleScene::HEIGHT = 32u*2u;

void ConsoleScene::initScene()
{
  try {
    // Setup state
    m_context = Context::create();
    m_context->setRayTypeCount( 2 );
    m_context->setEntryPointCount( 1 );
    m_context->setStackSize(1040);

    m_context["max_depth"]->setInt( 5 );
    m_context["scene_epsilon"]->setFloat( 1.e-4f );
    createOutputBuffer();

    float3 eye    = make_float3( 3.0f, 2.0f, -3.0f );
    float3 lookat = make_float3( 0.0f, 0.3f,  0.0f );
    float3 up     = make_float3( 0.0f, 1.0f,  0.0f );
    float  vfov   = 60.0f;
    float  hfov   = 60.0f;
    float3 U,V,W;
    // Get the U,V,W vectors from the input camera paramters.
    calculateCameraParameters( eye, lookat, up, vfov, hfov, U, V, W);

    // Declare camera variables.  The values do not matter, they will be overwritten in trace.
    m_context["eye"]->setFloat( eye );
    m_context["U"]->setFloat( U );
    m_context["V"]->setFloat( V );
    m_context["W"]->setFloat( W );

    // Ray gen program
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "pinhole_camera.cu" );
    Program ray_gen_program = m_context->createProgramFromPTXString( ptx, "pinhole_camera" );
    m_context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception
    Program exception_program = m_context->createProgramFromPTXString( ptx, "exception" );
    m_context->setExceptionProgram( 0, exception_program );
    m_context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    m_context->setMissProgram( 0, m_context->createProgramFromPTXString( sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" ), "miss" ) );
    m_context["bg_color"]->setFloat( make_float3(108.0f/255.0f, 166.0f/255.0f, 205.0f/255.0f) * 0.5f );

    // Setup lights
    m_context["ambient_light_color"]->setFloat(0.0f, 0.0f, 0.0f);
    // Lights buffer
    BasicLight lights[] = {
      { make_float3( -60.0f,  30.0f, -120.0f ), make_float3( 0.2f, 0.2f, 0.25f ), 1, 0 },
      { make_float3( -60.0f,   0.0f,  120.0f ), make_float3( 0.1f, 0.1f, 0.10f ), 1, 0 },
      { make_float3(  60.0f,  60.0f,   60.0f ), make_float3( 0.7f, 0.7f, 0.65f ), 1, 0 }
    };

    Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(BasicLight));
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    m_context["lights"]->set(light_buffer);

    // Create scene geom
    createGeometry();

    // Finalize
    m_context->validate();

  } catch( Exception& e ){
    std::cerr << "OptiX error: " << e.getErrorString();
    exit(2);
  }
}

void ConsoleScene::calculateCameraParameters( float3 eye, float3 lookat, float3 up, float vfov, float hfov,
                                             float3& camera_u, float3& camera_v, float3& camera_w )
{
  float3 lookdir = lookat-eye;  // do not normalize lookdir -- implies focal length
  float lookdir_len = length( lookdir );
  up = normalize(up);
  camera_u = normalize( cross(lookdir, up) );
  camera_v = normalize( cross(camera_u, lookdir) );
  float ulen = lookdir_len * tanf( 0.5f * hfov * M_PIf / 180.0f );
  camera_u *= ulen;                                
  float vlen = lookdir_len * tanf( 0.5f * vfov * M_PIf / 180.0f );
  camera_v *= vlen;
  camera_w = lookdir;
}

Buffer ConsoleScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void ConsoleScene::trace()
{
  Buffer buffer = getOutputBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  m_context->launch( 0,
                    static_cast<unsigned int>(buffer_width),
                    static_cast<unsigned int>(buffer_height)
                    );
}

void ConsoleScene::createOutputBuffer()
{
  m_context["output_buffer"]->set(m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT ) );
}


void ConsoleScene::cleanUp()
{
  m_context->destroy();
}


void ConsoleScene::createGeometry()
{
  // Sphere
  Geometry sphere = m_context->createGeometry();
  sphere->setPrimitiveCount( 1u );
  const char *ptx = sutil::getPtxString( SAMPLE_NAME, "sphere.cu" );
  sphere->setBoundingBoxProgram( m_context->createProgramFromPTXString( ptx, "bounds" ) );
  sphere->setIntersectionProgram( m_context->createProgramFromPTXString( ptx, "intersect" ) );
  sphere["sphere"]->setFloat( 0.0f, 1.2f, 0.0f, 1.0f );

  // Floor
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
  parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXString( ptx, "bounds" ) );
  parallelogram->setIntersectionProgram( m_context->createProgramFromPTXString( ptx, "intersect" ) );
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

  // Phong programs
  ptx = sutil::getPtxString( SAMPLE_NAME, "phong.cu" );
  Program phong_ch = m_context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
  Program phong_ah = m_context->createProgramFromPTXString( ptx, "any_hit_shadow" );

  // Sphere material
  Material sphere_matl = m_context->createMaterial();
  sphere_matl->setClosestHitProgram( 0, phong_ch );
  sphere_matl->setAnyHitProgram( 1, phong_ah );

  sphere_matl["Ka"]->setFloat(0,0,0);
  sphere_matl["Kd"]->setFloat(.6f, 0, 0);
  sphere_matl["Ks"]->setFloat(.6f, .6f, .6f);
  sphere_matl["phong_exp"]->setFloat(32);
  sphere_matl["Kr"]->setFloat(.4f, .4f, .4f);

  // Floor material
  ptx = sutil::getPtxString( SAMPLE_NAME, "checker.cu" );
  Program check_ch    = m_context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
  Program check_ah    = m_context->createProgramFromPTXString( ptx, "any_hit_shadow" );
  Material floor_matl = m_context->createMaterial();
  floor_matl->setClosestHitProgram( 0, check_ch );
  floor_matl->setAnyHitProgram( 1, check_ah );

  floor_matl["Kd1"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["Ka1"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["Ks1"]->setFloat( 0.5f, 0.5f, 0.5f);
  floor_matl["Kr1"]->setFloat( 0.5f, 0.5f, 0.5f);
  floor_matl["phong_exp1"]->setFloat( 32.0f );

  floor_matl["Kd2"]->setFloat( 0.6f, 0.6f, 0.6f);
  floor_matl["Ka2"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["Ks2"]->setFloat( 0.5f, 0.5f, 0.5f);
  floor_matl["Kr2"]->setFloat( 0.2f, 0.2f, 0.2f);
  floor_matl["phong_exp2"]->setFloat( 32.0f );

  floor_matl["inv_checker_size"]->setFloat( 40.0f, 40.0f, 1.0f );

  // Place geometry into hierarchy
  std::vector<GeometryInstance> gis;
  gis.push_back( m_context->createGeometryInstance( sphere,        &sphere_matl, &sphere_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 ) );

  GeometryGroup geometrygroup = m_context->createGeometryGroup();
  geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
  geometrygroup->setChild( 0, gis[0] );
  geometrygroup->setChild( 1, gis[1] );
  geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel") );

  m_context["top_object"]->set( geometrygroup );
  m_context["top_shadower"]->set( geometrygroup );
}

void ConsoleScene::display( std::ostream& output_stream )
{

  Buffer buffer = getOutputBuffer();

  uchar4* data = static_cast<uchar4*>(buffer->map());
  if (!data) {
    std::cerr << "Can't map output buffer\n";
    exit(2);
  }
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );
  std::ostringstream out;

  char lumchar[] = { ' ', '.', ',', ';', '!', 'o', '&', '8', '#', '@' };
  for(RTsize y = 0; y < buffer_height; ++y) {
    uchar4* row = data+((buffer_height-y-1)*buffer_width);
    for(RTsize x = 0; x < buffer_width; ++x) {
      uchar4 ucolor = row[x];
      float3 color = make_float3(static_cast<float>(ucolor.x),
                                 static_cast<float>(ucolor.y), 
                                 static_cast<float>(ucolor.z))
                     /make_float3(256.0f);
      float lum = color.x*0.3f + color.y*0.6f + color.z*0.1f;
      out << lumchar[static_cast<int>(lum*10)];
    }
    out << "\n";
  }
  output_stream << out.str();
  buffer->unmap();
}

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help    Print this usage message\n"
    << "  -f  | --file    Save frame to text file, defaults to stdout\n"                          
    << std::endl;

  if ( doExit ) exit(1);
}

int main( int argc, char** argv )
{
  std::ostream* outp = &std::cout;
  std::ofstream fout;
  for(int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (0) {
    } else if( arg == "-h" || arg == "--help" ) {
      printUsageAndExit(argv[0]);
    } else if( arg == "-f" || arg == "--file" ) {
      if( i == argc-1 )
      {
        std::cerr << "Option '" << arg << "' requires additional argument.\n";
        printUsageAndExit( argv[0] );
      }
      fout.open( argv[++i], std::ofstream::out );
      outp = &fout;
    } else {
      std::cerr << "Unknown option '" << arg << "'\n";
      printUsageAndExit(argv[0]);
    }
  }

  ConsoleScene scene;
  scene.initScene();
  scene.trace();
  scene.display( *outp );
  scene.cleanUp();

  return 0;
}
