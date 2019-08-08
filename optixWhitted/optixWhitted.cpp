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
// optixWhitted: whitted's original sphere scene
//
//-----------------------------------------------------------------------------

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#  include <GL/wglew.h>
#  include <GL/freeglut.h>
#  else
#  include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "common.h"
#include <Arcball.h>

#include <cstring>
#include <iostream>
#include <stdint.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixWhitted";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context      context;
uint32_t     width  = 768u;
uint32_t     height = 768u;
bool         use_pbo = true;

// Camera state
float3       camera_up;
float3       camera_lookat;
float3       camera_eye;
Matrix4x4    camera_rotate;
bool         camera_dirty = true;  // Do camera params need to be copied to OptiX context
sutil::Arcball arcball;

// Mouse state
int2       mouse_prev_pos;
int        mouse_button;


//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void createGeometry();
void setupCamera();
void setupLights();
void updateCamera();
void glutInitialize( int* argc, char** argv );
void glutRun();

void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutMousePress( int button, int state, int x, int y );
void glutMouseMotion( int x, int y);
void glutResize( int w, int h );


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}


void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
    atexit( destroyContext );
#endif
}


void createContext()
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 2800 );
    context->setMaxTraceDepth( 12 );

    // Note: high max depth for reflection and refraction through glass
    context["max_depth"]->setInt( 10 );
    context["frame"]->setUint( 0u );
    context["scene_epsilon"]->setFloat( 1.e-4f );
    context["ambient_light_color"]->setFloat( 0.4f, 0.4f, 0.4f );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo );
    context["output_buffer"]->set( buffer );

    // Accumulation buffer.  This scene has a lot of high frequency detail and
    // benefits from accumulation of samples.
    Buffer accum_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT4, width, height );
    context["accum_buffer"]->set( accum_buffer );

    // Ray generation program
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "accum_camera.cu" );
    Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    context->setMissProgram( 0, context->createProgramFromPTXString( sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" ), "miss" ) );
    context["bg_color"]->setFloat( 0.34f, 0.55f, 0.85f );
}


void createGeometry()
{
    // Create glass sphere geometry
    Geometry glass_sphere = context->createGeometry();
    glass_sphere->setPrimitiveCount( 1u );

    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "sphere_shell.cu" );
    glass_sphere->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    glass_sphere->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    glass_sphere["center"]->setFloat( 4.0f, 2.3f, -4.0f );
    glass_sphere["radius1"]->setFloat( 0.96f );
    glass_sphere["radius2"]->setFloat( 1.0f );

    // Metal sphere geometry
    Geometry metal_sphere = context->createGeometry();
    metal_sphere->setPrimitiveCount( 1u );
    ptx = sutil::getPtxString( SAMPLE_NAME, "sphere.cu" );
    metal_sphere->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    metal_sphere->setIntersectionProgram( context->createProgramFromPTXString( ptx, "robust_intersect" ) );
    metal_sphere["sphere"]->setFloat( 2.0f, 1.5f, -2.5f, 1.0f );

    // Floor geometry
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );
    ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
    parallelogram->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    parallelogram->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    float3 anchor = make_float3( -16.0f, 0.01f, -8.0f );
    float3 v1 = make_float3( 32.0f, 0.0f, 0.0f );
    float3 v2 = make_float3( 0.0f, 0.0f, 16.0f );
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


    // Glass material
    ptx = sutil::getPtxString( SAMPLE_NAME, "glass.cu" );
    Program glass_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program glass_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
    Material glass_matl = context->createMaterial();
    glass_matl->setClosestHitProgram( 0, glass_ch );
    glass_matl->setAnyHitProgram( 1, glass_ah );

    glass_matl["importance_cutoff"]->setFloat( 1e-2f );
    glass_matl["cutoff_color"]->setFloat( 0.034f, 0.055f, 0.085f );
    glass_matl["fresnel_exponent"]->setFloat( 3.0f );
    glass_matl["fresnel_minimum"]->setFloat( 0.1f );
    glass_matl["fresnel_maximum"]->setFloat( 1.0f );
    glass_matl["refraction_index"]->setFloat( 1.4f );
    glass_matl["refraction_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    glass_matl["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    glass_matl["refraction_maxdepth"]->setInt( 10 );
    glass_matl["reflection_maxdepth"]->setInt( 5 );
    const float3 extinction = make_float3(.83f, .83f, .83f);
    glass_matl["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );
    glass_matl["shadow_attenuation"]->setFloat( 0.6f, 0.6f, 0.6f );

    // Metal material
    ptx = sutil::getPtxString( SAMPLE_NAME, "phong.cu" );
    Program phong_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program phong_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
    Material metal_matl = context->createMaterial();
    metal_matl->setClosestHitProgram( 0, phong_ch );
    metal_matl->setAnyHitProgram( 1, phong_ah );
    metal_matl["Ka"]->setFloat( 0.2f, 0.5f, 0.5f );
    metal_matl["Kd"]->setFloat( 0.2f, 0.7f, 0.8f );
    metal_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
    metal_matl["phong_exp"]->setFloat( 64 );
    metal_matl["Kr"]->setFloat( 0.5f,  0.5f,  0.5f);

    // Checker material for floor
    ptx = sutil::getPtxString( SAMPLE_NAME, "checker.cu" );
    Program check_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program check_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
    Material floor_matl = context->createMaterial();
    floor_matl->setClosestHitProgram( 0, check_ch );
    floor_matl->setAnyHitProgram( 1, check_ah );

    floor_matl["Kd1"]->setFloat( 0.8f, 0.3f, 0.15f);
    floor_matl["Ka1"]->setFloat( 0.8f, 0.3f, 0.15f);
    floor_matl["Ks1"]->setFloat( 0.0f, 0.0f, 0.0f);
    floor_matl["Kd2"]->setFloat( 0.9f, 0.85f, 0.05f);
    floor_matl["Ka2"]->setFloat( 0.9f, 0.85f, 0.05f);
    floor_matl["Ks2"]->setFloat( 0.0f, 0.0f, 0.0f);
    floor_matl["inv_checker_size"]->setFloat( 32.0f, 16.0f, 1.0f );
    floor_matl["phong_exp1"]->setFloat( 0.0f );
    floor_matl["phong_exp2"]->setFloat( 0.0f );
    floor_matl["Kr1"]->setFloat( 0.0f, 0.0f, 0.0f);
    floor_matl["Kr2"]->setFloat( 0.0f, 0.0f, 0.0f);

    // Create GIs for each piece of geometry
    std::vector<GeometryInstance> gis;
    gis.push_back( context->createGeometryInstance( glass_sphere, &glass_matl, &glass_matl+1 ) );
    gis.push_back( context->createGeometryInstance( metal_sphere,  &metal_matl,  &metal_matl+1 ) );
    gis.push_back( context->createGeometryInstance( parallelogram, &floor_matl,  &floor_matl+1 ) );

    // Place all in group
    GeometryGroup geometrygroup = context->createGeometryGroup();
    geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
    geometrygroup->setChild( 0, gis[0] );
    geometrygroup->setChild( 1, gis[1] );
    geometrygroup->setChild( 2, gis[2] );
    geometrygroup->setAcceleration( context->createAcceleration("NoAccel") );

    context["top_object"]->set( geometrygroup );
    context["top_shadower"]->set( geometrygroup );

}


void setupCamera()
{
    camera_eye    = make_float3( 8.0f, 2.0f, -4.0f );
    camera_lookat = make_float3( 4.0f, 2.3f, -4.0f );
    camera_up   = make_float3( 0.0f, 1.0f,  0.0f );

    camera_rotate  = Matrix4x4::identity();
    camera_dirty = true;
}


void setupLights()
{

    BasicLight lights[] = {
        { make_float3( 60.0f, 40.0f, 0.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
    };

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( BasicLight ) );
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    context[ "lights" ]->set( light_buffer );
}


void updateCamera()
{
    const float vfov  = 60.0f;
    const float aspect_ratio = static_cast<float>(width) /
        static_cast<float>(height);

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, /*fov_is_vertical*/ true );

    const Matrix4x4 frame = Matrix4x4::fromBasis(
            normalize( camera_u ),
            normalize( camera_v ),
            normalize( -camera_w ),
            camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans   = frame*camera_rotate*camera_rotate*frame_inv;

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    context["eye"]->setFloat( camera_eye );
    context["U"  ]->setFloat( camera_u );
    context["V"  ]->setFloat( camera_v );
    context["W"  ]->setFloat( camera_w );

    camera_dirty = false;
}


void glutInitialize( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( SAMPLE_NAME );
    glutHideWindow();
}


void glutRun()
{
    // Initialize GL state
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1 );
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, width, height);

    glutShowWindow();
    glutReshapeWindow( width, height);

    // register glut callbacks
    glutDisplayFunc( glutDisplay );
    glutIdleFunc( glutDisplay );
    glutReshapeFunc( glutResize );
    glutKeyboardFunc( glutKeyboardPress );
    glutMouseFunc( glutMousePress );
    glutMotionFunc( glutMouseMotion );

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
    static unsigned int accumulation_frame = 0;
    if( camera_dirty ) {
        updateCamera();
        accumulation_frame = 0;
    }

    context["frame"]->setUint( accumulation_frame++ );
    context->launch( 0, width, height );

    sutil::displayBufferGL( getOutputBuffer() );

    {
        static unsigned frame_count = 0;
        sutil::displayFps( frame_count++ );
    }

    glutSwapBuffers();
}


void glutKeyboardPress( unsigned char k, int x, int y )
{
    switch( k )
    {
        case( 'q' ):
        case( 27 ): // ESC
        {
            destroyContext();
            exit(0);
        }
        case( 's' ):
        {
            const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
            break;
        }
    }
}


void glutMousePress( int button, int state, int x, int y )
{
    if( state == GLUT_DOWN )
    {
        mouse_button = button;
        mouse_prev_pos = make_int2( x, y );
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion( int x, int y)
{
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
        const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
                         static_cast<float>( width );
        const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
                         static_cast<float>( height );
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float scale = fminf( dmax, 0.9f );
        camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
        camera_dirty = true;
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to   = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x   / width, to.y   / height };

        camera_rotate = arcball.rotate( b, a );
        camera_dirty = true;
    }

    mouse_prev_pos = make_int2( x, y );
}


void glutResize( int w, int h )
{
    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    camera_dirty = true;

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    sutil::resizeBuffer( context[ "accum_buffer" ]->getBuffer(), width, height );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glViewport(0, 0, width, height);
    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help         Print this usage message and exit.\n"
        "  -f | --file         Save single frame to file and exit.\n"
        "  -n | --nopbo        Disable GL interop for display buffer.\n"
        "App Keystrokes:\n"
        "  q  Quit\n"
        "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
        << std::endl;

    exit(1);
}

int main( int argc, char** argv )
{
    std::string out_file;
    for( int i=1; i<argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "-f" || arg == "--file"  )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            out_file = argv[++i];
        }
        else if( arg == "-n" || arg == "--nopbo"  )
        {
            use_pbo = false;
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        createContext();
        createGeometry();
        setupCamera();
        setupLights();

        context->validate();

        if ( out_file.empty() )
        {
            glutRun();
        }
        else
        {
            updateCamera();
            context->launch( 0, width, height );
            sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
            destroyContext();
        }
        return 0;
    }
    SUTIL_CATCH( context->get() )
}

