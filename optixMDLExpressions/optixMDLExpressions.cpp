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
//  Render a sphere and a plane with a MDL expressions
//
//-----------------------------------------------------------------------------

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include <Arcball.h>
#include <HDRLoader.h>

#include <mdl_wrapper.h>
#define Mdl_helper Mdl_wrapper

using namespace optix;

#include "common.h"


const char* const SAMPLE_NAME = "optixMDLExpressions";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context  context;
int      width  = 1024;
int      height = 768;

// MDL state
Mdl_helper *mdl_helper = NULL;

struct MDL_material_programs
{
    Program tint;
    Program shading_normal;
};

std::vector<MDL_material_programs> mdl_material_programs;

std::vector<Program> mdl_environment_programs;

Program  miss_program;
size_t   cur_env_idx = 0;

Material sphere_material;
std::vector<size_t> sphere_mat_idxs;
size_t   sphere_cur_mat_idx = 0;

Material floor_material;
std::vector<size_t> floor_mat_idxs;
size_t   floor_cur_mat_idx = 2;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
bool           camera_changed = true;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;


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
    if( mdl_helper )
    {
        delete mdl_helper;
        mdl_helper = NULL;
    }
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


Context createContext()
{
    // Set up context
    Context context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 1200 );

    context["max_depth"]->setInt( 5 );
    context["scene_epsilon"]->setFloat( 1.e-4f );
    context["tonemap_scale"]->setFloat( 1.0f );

    // uchar display buffer
    Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height );
    context["output_buffer"]->set( buffer );

    // Ray generation program
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "camera.cu" );
    Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );

    // Miss program
    ptx = sutil::getPtxString( SAMPLE_NAME, "mdl_material.cu" );
    miss_program = context->createProgramFromPTXString( ptx, "miss" );
    context->setMissProgram( 0, miss_program );

    return context;
}


void setupCamera()
{
    camera_eye    = make_float3( 10.0f, 0.5f, 0.0f );
    camera_lookat = make_float3(  0.0f, 0.0f, 0.0f );
    camera_up     = make_float3(  0.0f, 1.0f, 0.0f );

    camera_rotate  = Matrix4x4::identity();
}


void updateCamera()
{
    if (!camera_changed) return;
    camera_changed = false;

    const float fov  = 35.0f;
    const float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
        camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
        camera_u, camera_v, camera_w, /*fov_is_vertical*/ true );

    const Matrix4x4 frame = Matrix4x4::fromBasis(
        normalize( camera_u ),
        normalize( camera_v ),
        normalize( -camera_w ),
        camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans     = frame*camera_rotate*camera_rotate*frame_inv;

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
        camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
        camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    context["eye"]->setFloat( camera_eye );
    context["U"  ]->setFloat( camera_u );
    context["V"  ]->setFloat( camera_v );
    context["W"  ]->setFloat( camera_w );
}


Geometry createSphere()
{
    Geometry sphere = context->createGeometry();
    sphere->setPrimitiveCount( 1u );

    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "sphere_texcoord_tangent.cu" );
    sphere->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    sphere->setIntersectionProgram(
        context->createProgramFromPTXString( ptx, "robust_intersect" ) );
    sphere["sphere"      ]->setFloat( 0, 0, 0, 1.5 );
    sphere["matrix_row_0"]->setFloat( 1.0f, 0.0f, 0.0f );
    sphere["matrix_row_1"]->setFloat( 0.0f, 1.0f, 0.0f );
    sphere["matrix_row_2"]->setFloat( 0.0f, 0.0f, 1.0f );
    return sphere;
}


Geometry createParallelogram()
{
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );

    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram_tangent.cu" );
    parallelogram->setBoundingBoxProgram(
        context->createProgramFromPTXString( ptx, "bounds" ) );
    parallelogram->setIntersectionProgram(
        context->createProgramFromPTXString( ptx, "intersect" ) );
    float3 anchor = make_float3( -20.0f, -2.0f, 20.0f );
    float3 v1 = make_float3( 40, 0, 0 );
    float3 v2 = make_float3( 0, 0, -40 );
    float3 normal = cross( v1, v2 );
    normal = normalize( normal );
    float d = dot( normal, anchor );
    v1 *= 1.0f / dot( v1, v1 );
    v2 *= 1.0f / dot( v2, v2 );
    float4 plane = make_float4( normal, d );
    parallelogram["plane"]->setFloat( plane );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );
    parallelogram["anchor"]->setFloat( anchor );

    return parallelogram;
}


GeometryInstance createInstance( Geometry geometry, Material material )
{
    GeometryInstance gi = context->createGeometryInstance();
    gi->setGeometry( geometry );
    gi->addMaterial( material );
    return gi;
}


size_t addMDLMaterial(const std::string &module_name, const std::string &material_name)
{
    MDL_material_programs progs;

    progs.tint = mdl_helper->compile_expression(
        module_name,
        material_name,
        "surface.scattering.tint" );

    progs.shading_normal = mdl_helper->compile_expression(
        module_name,
        material_name,
        "geometry.normal" );

    // Force the programs to be bindless to avoid long compilation times, when first used
    progs.tint->getId();
    progs.shading_normal->getId();

    mdl_material_programs.push_back( progs );
    return mdl_material_programs.size() - 1;
}


size_t addMDLEnvironment( const std::string &module_name, const std::string &function_name )
{
    Program env_prog = mdl_helper->compile_environment( module_name, function_name );

    // Force the program to be bindless to avoid long compilation times, when first used
    env_prog->getId();

    mdl_environment_programs.push_back( env_prog );
    return mdl_environment_programs.size() - 1;
}


void initMDL()
{
    mdl_helper = new Mdl_helper( context );
    mdl_helper->add_module_path(
        std::string( sutil::samplesDir() ) + "/data" );
    mdl_helper->set_mdl_textures_ptx_string(
        sutil::getPtxString( SAMPLE_NAME, "mdl_textures.cu" ) );

    // Initialize MDL material programs
    size_t m1 = addMDLMaterial( "optixMDLExpressions", "M_textured" );
    size_t m2 = addMDLMaterial( "optixMDLExpressions", "M_checker" );
    size_t m3 = addMDLMaterial( "optixMDLExpressions", "M_bump_worley" );
    size_t m4 = addMDLMaterial( "optixMDLExpressions", "M_bump_worley_small" );
    size_t m5 = addMDLMaterial( "optixMDLExpressions", "M_bump_texture" );

    // Choose available materials for the sphere and the floor
    sphere_mat_idxs.push_back( m1 );
    sphere_mat_idxs.push_back( m2 );

    floor_mat_idxs.push_back( m1 );
    floor_mat_idxs.push_back( m2 );
    floor_mat_idxs.push_back( m3 );
    floor_mat_idxs.push_back( m4 );
    floor_mat_idxs.push_back( m5 );

    // Initialize MDL environment programs
    addMDLEnvironment( "optixMDLExpressions", "sun_and_sky(texture_2d)" );
    addMDLEnvironment( "optixMDLExpressions", "perez_sun_and_sky()" );
    addMDLEnvironment( "optixMDLExpressions", "cubemap(texture_cube)" );
}


void applyMDLMaterial( Material material, size_t material_idx )
{
    material["mdl_expr"]->setProgramId( mdl_material_programs[material_idx].tint );
    material["mdl_shading_normal_expr"]->setProgramId(
        mdl_material_programs[material_idx].shading_normal );
}


Material createMDLMaterial( size_t material_idx )
{
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "mdl_material.cu" );
    Program chp = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program ahp = context->createProgramFromPTXString( ptx, "any_hit_shadow" );

    Material matl = context->createMaterial();
    matl->setClosestHitProgram( 0, chp );
    matl->setAnyHitProgram( 1, ahp );

    applyMDLMaterial( matl, material_idx );

    return matl;
}


void applyMDLEnvironment( size_t environment_idx )
{
    miss_program["mdl_env_expr"]->setProgramId( mdl_environment_programs[environment_idx] );
}


void createScene()
{
    // Choose initial environment
    applyMDLEnvironment( cur_env_idx );

    // Setup geometry
    std::vector<GeometryInstance> gis;

    // Sphere
    Geometry sphere = createSphere();
    sphere_material = createMDLMaterial( sphere_mat_idxs[sphere_cur_mat_idx] );
    sphere_material["reflection_coefficient"]->setFloat( 0.01f );
    gis.push_back( createInstance( sphere, sphere_material ) );

    // Floor
    Geometry parallelogram = createParallelogram();
    floor_material = createMDLMaterial( floor_mat_idxs[floor_cur_mat_idx] );
    floor_material["reflection_coefficient"]->setFloat( 0.f );
    gis.push_back( createInstance( parallelogram, floor_material ) );


    // Create geometry and shadow group
    GeometryGroup geometry_group = context->createGeometryGroup( gis.begin(), gis.end() );
    geometry_group->setAcceleration( context->createAcceleration( "NoAccel" ) );
    context["top_object"]->set( geometry_group );
    context["top_shadower"]->set( geometry_group );


    // Setup lights
    context["ambient_light_color"]->setFloat( 0.2f, 0.2f, 0.2f );

    BasicLight lights[] = {
        { { 0.0f, 8.0f, -5.0f }, { .6f, .1f, .1f }, 1 },
        { { 5.0f, 8.0f,  0.0f }, { .1f, .6f, .1f }, 1 },
        { { 5.0f, 2.0f, -5.0f }, { .2f, .2f, .2f }, 1 }
    };

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( BasicLight ));
    light_buffer->setSize( sizeof( lights ) / sizeof( lights[0] ) );
    memcpy( light_buffer->map(), lights, sizeof( lights ) );
    light_buffer->unmap();

    context["lights"]->set( light_buffer );
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void displayGlut()
{
    updateCamera();
    context->launch( 0, width, height );

    sutil::displayBufferGL( getOutputBuffer() );

    {
        static unsigned frame_count = 0;
        sutil::displayFps( frame_count++ );
    }

    glutSwapBuffers();
}


void keyboardPressGlut( unsigned char k, int x, int y )
{
    switch( k )
    {
        case 'q':
        case 27: // ESC
            destroyContext();
            exit(0);

        case 'e':
            cur_env_idx = (cur_env_idx + 1) % mdl_environment_programs.size();
            applyMDLEnvironment( cur_env_idx );
            break;

        case 'M':
            sphere_cur_mat_idx = (sphere_cur_mat_idx + 1) % sphere_mat_idxs.size();
            applyMDLMaterial( sphere_material, sphere_mat_idxs[sphere_cur_mat_idx] );
            break;

        case 'm':
            floor_cur_mat_idx = (floor_cur_mat_idx + 1) % floor_mat_idxs.size();
            applyMDLMaterial( floor_material, floor_mat_idxs[floor_cur_mat_idx] );
            break;
    }
}


void mousePressGlut( int button, int state, int x, int y )
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


void mouseMotionGlut( int x, int y )
{
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
        const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
            static_cast<float>( width );
        const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
            static_cast<float>( height );
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float scale = std::min<float>( dmax, 0.9f );
        camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
        camera_changed = true;
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
        camera_changed = true;
    }

    mouse_prev_pos = make_int2( x, y );
}


void resizeGlut( int w, int h )
{
    if ( w == (int)width && h == (int)height ) return;

    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    camera_changed = true;  // force updating the camera to adjust to new aspect ratio

    sutil::resizeBuffer( getOutputBuffer(), width, height );

    glViewport( 0, 0, width, height );

    glutPostRedisplay();
}


void initializeGlut( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( "MDL material expressions - press M, m, e to change the materials" );
    glutHideWindow();
}


void runGlut()
{
    // Initialize GL state
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( 0, 1, 0, 1, -1, 1 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glViewport( 0, 0, width, height );

    glutShowWindow();
    glutReshapeWindow( width, height );

    // register glut callbacks
    glutDisplayFunc( displayGlut );
    glutIdleFunc( displayGlut );
    glutReshapeFunc( resizeGlut );
    glutKeyboardFunc( keyboardPressGlut );
    glutMouseFunc( mousePressGlut );
    glutMotionFunc( mouseMotionGlut );

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n"
        "App Options:\n"
        "  -h | --help             Print this usage message\n"
        "  -f | --file <filename>  Save single frame to file and exit.\n"
        "\n"
        "App Keystrokes:\n"
        "  e Switch to the next MDL environment\n"
        "  m Switch to the next floor MDL material\n"
        "  M Switch to the next sphere MDL material\n"
        << std::endl;
    exit(1);
}


int main( int argc, char** argv )
{
    std::string out_file;

    for( int i = 1; i < argc; ++i )
    {
        std::string arg( argv[i] );
        if ( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "-f" || arg == "--file" )
        {
            if( i + 1 >= argc )
            {
                std::cerr << "Option '" << arg << "' requires an additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            out_file = argv[++i];
        }
        else
        {
            std::cerr << "Unknown option: '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        initializeGlut( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        // Setup state & scene
        context = createContext();
        initMDL();
        setupCamera();
        createScene();

        context->validate();

        if( out_file.empty() )
        {
            runGlut();
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
