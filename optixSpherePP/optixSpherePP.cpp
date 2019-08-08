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
//  optixSpherePP.cpp -- Renders sphere with a normal shader using CPP optix wrapper. 
//
//-----------------------------------------------------------------------------


#include <iostream>
#include <cstdlib>
#include <optixu/optixpp_namespace.h>
#include <sutil.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixSpherePP";

int width  = 1024;
int height = 768;

Context  createContext();
Material createMaterial( Context context );
Geometry createGeometry( Context context );
void     createInstance( Context context, Geometry sphere, Material material );
void     printUsageAndExit( const std::string& argv0 );


int main( int argc, char** argv )
{
  // Process command line options
  std::string outfile;
  
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else if( arg == "--file" || arg == "-f" ) {
      if( i < argc-1 ) {
        outfile = argv[++i];
      } else {
        printUsageAndExit( argv[0] );
      }
    } else if ( arg.substr( 0, 6 ) == "--dim=" ) {
      std::string dims_arg = arg.substr(6);
      sutil::parseDimensions( dims_arg.c_str(), width, height );
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  sutil::initGlut( &argc, argv );

  try
  {
    // Setup state
    Context  context  = createContext();
    Geometry sphere   = createGeometry( context );
    Material material = createMaterial( context );
    createInstance( context, sphere, material );

    // Run
    context->validate();
    context->launch( 0, width, height );

    // DisplayImage
    if( outfile.empty() ) {
        sutil::displayBufferGlut( argv[0], context["output_buffer"]->getBuffer() );
    } else {
        sutil::displayBufferPPM( outfile.c_str(), context["output_buffer"]->getBuffer() );
    }

    // Clean up
    context->destroy();

  } catch( Exception& e ){
      sutil::reportErrorMessage( e.getErrorString().c_str() );
    exit(1);
  }
  return( 0 );
}


Context createContext() 
{
  // Set up context
  Context context = Context::create();
  context->setRayTypeCount( 1 );
  context->setEntryPointCount( 1 );

  context["scene_epsilon"]->setFloat( 1.e-4f );

  Variable output_buffer = context["output_buffer"];
  Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height );
  output_buffer->set(buffer);

  // Ray generation program
  const char *ptx = sutil::getPtxString( SAMPLE_NAME, "pinhole_camera.cu" );
  Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
  context->setRayGenerationProgram( 0, ray_gen_program );

  float3 cam_eye = { 0.0f, 0.0f, 5.0f };
  float3 lookat  = { 0.0f, 0.0f, 0.0f };
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

  // Exception program
  Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
  context->setExceptionProgram( 0, exception_program );
  context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

  // Miss program
  context->setMissProgram( 0, context->createProgramFromPTXString( sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" ), "miss" ) );
  context["bg_color"]->setFloat( 0.3f, 0.1f, 0.2f );

  return context;
}


Geometry createGeometry( Context context )
{
  Geometry sphere = context->createGeometry();
  sphere->setPrimitiveCount( 1u );
  const char *ptx = sutil::getPtxString( SAMPLE_NAME, "sphere.cu" );
  sphere->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
  sphere->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
  sphere["sphere"]->setFloat( 0, 0, 0, 1.5 );
  return sphere;
}


Material createMaterial( Context context )
{
  const char *ptx = sutil::getPtxString( SAMPLE_NAME, "normal_shader.cu" );
  Program chp = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );

  Material matl = context->createMaterial();
  matl->setClosestHitProgram( 0, chp );
  return matl;
}


void createInstance( Context context, Geometry sphere, Material material )
{
  // Create geometry instance
  GeometryInstance gi = context->createGeometryInstance();
  gi->setMaterialCount( 1 );
  gi->setGeometry( sphere );
  gi->setMaterial( 0, material );

  // Create geometry group
  GeometryGroup geometrygroup = context->createGeometryGroup();
  geometrygroup->setChildCount( 1 );
  geometrygroup->setChild( 0, gi );
  geometrygroup->setAcceleration( context->createAcceleration("NoAccel") );

  context["top_object"]->set( geometrygroup );
}


void printUsageAndExit( const std::string& argv0 )
{ 
  std::cerr << "Usage  : " << argv0 << " [options]\n"
            << "Options: --help | -h             Print this usage message\n"
            << "         --file | -f <filename>  Specify file for image output\n"
            << "         --dim=<width>x<height>  Set image dimensions; defaults to 1024x768\n";
  exit(1); 
} 

