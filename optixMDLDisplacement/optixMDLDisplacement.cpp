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
// optixMDLDisplacement: Renders a model with displacement mapping using
//                       MDL materials.
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

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "common.h"
#include <Arcball.h>
#include <OptiXMesh.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdint.h>

#include <mdl_wrapper.h>
#define Mdl_helper Mdl_wrapper

using namespace optix;

const char* const SAMPLE_NAME = "optixMDLDisplacement";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context        context;
uint32_t       width  = 1024u;
uint32_t       height = 768u;
bool           use_pbo = true;
optix::Aabb    aabb;

// MDL state
Mdl_helper *mdl_helper = NULL;

struct MDL_material_programs
{
    Program tint;
    Program displace;
};

std::vector<MDL_material_programs> mdl_material_programs;

// Only used when USE_MDL_DISPLACEMENT is commented out in geometry_programs.cu.
std::vector<Program> non_mdl_displace_programs;

Material mesh_material;
Program mesh_intersect_prog;
size_t mesh_cur_mat_idx = 0;

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

    context["scene_epsilon"    ]->setFloat( 1.e-4f );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo );
    context["output_buffer"]->set( buffer );

    // Ray generation program
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "pinhole_camera.cu" );
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


size_t addMDLMaterial( const std::string &module_name, const std::string &material_name )
{
    MDL_material_programs progs;

    progs.tint = mdl_helper->compile_expression(
        module_name,
        material_name,
        "surface.scattering.tint" );

    progs.displace = mdl_helper->compile_expression(
        module_name,
        material_name,
        "geometry.displacement" );

    // Force the programs to be bindless to avoid long compilation times, when first used
    progs.tint->getId();
    progs.displace->getId();

    mdl_material_programs.push_back( progs );
    return mdl_material_programs.size() - 1;
}


void initMDL()
{
    mdl_helper = new Mdl_helper( context );
    mdl_helper->add_module_path(
        std::string( sutil::samplesDir() ) + "/data" );
    mdl_helper->set_mdl_textures_ptx_string(
        sutil::getPtxString( SAMPLE_NAME, "mdl_textures.cu" ) );

    addMDLMaterial("optixMDLDisplacement", "Fish");
    addMDLMaterial("optixMDLDisplacement", "FishRings");

    // Also prepare displacement programs for non-mdl case
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "geometry_programs.cu" );
    non_mdl_displace_programs.push_back(
        context->createProgramFromPTXString( ptx, "displace_fish" ) );
    non_mdl_displace_programs.push_back(
        context->createProgramFromPTXString( ptx, "displace_fish_rings" ) );

    // Force the programs to be bindless to avoid long compilation times, when first used
    for( size_t i = 0, n = non_mdl_displace_programs.size(); i < n; ++i )
        non_mdl_displace_programs[i]->getId();
}


void applyMDLMaterial( Material material, Program intersect_prog, size_t material_idx )
{
    material["mdl_expr"]->setProgramId( mdl_material_programs[material_idx].tint );
    intersect_prog["mdl_displace_expr"]->setProgramId(
        mdl_material_programs[material_idx].displace );

    if( material_idx < non_mdl_displace_programs.size() )
        intersect_prog["non_mdl_displace"]->setProgramId(
            non_mdl_displace_programs[material_idx] );
}


Material createMDLMaterial( Program intersect_prog, size_t material_idx )
{
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "materials.cu" );
    Program chp = context->createProgramFromPTXString( ptx, "mdl_material_apply" );

    Material matl = context->createMaterial();
    matl->setClosestHitProgram( 0, chp );

    applyMDLMaterial( matl, intersect_prog, material_idx );

    return matl;
}


bool loadMesh( const std::string& filename )
{
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "geometry_programs.cu" );

    OptiXMesh mesh;
    mesh.context = context;
    mesh.intersection = context->createProgramFromPTXString( ptx, "mesh_intersect" );
    mesh.bounds       = context->createProgramFromPTXString( ptx, "mesh_bounds" );
    mesh.material     = createMDLMaterial( mesh.intersection, 0 );

    // Note: The Triangle API does not support custom intersection or bounds
    //       programs, so we must disable it.
    mesh.use_tri_api  = false;

    mesh_material = mesh.material;
    mesh_intersect_prog = mesh.intersection;

    loadMesh( filename, mesh );

    aabb.set( mesh.bbox_min, mesh.bbox_max );

    optix::Buffer normal_buffer = mesh.geom_instance["normal_buffer"]->getBuffer();
    RTsize num_normals = 0;
    normal_buffer->getSize(num_normals);
    if( num_normals == 0 )
    {
        std::cerr << "Error: The mesh \"" << filename
            << "\" cannot be used, because it does not contain vertex normals." << std::endl;
        return false;
    }


    // Note: Acceleration structure builders that use splitting like "Trbvh" or "Sbvh"
    //       don't work with displacement.

    GeometryGroup geometry_group = context->createGeometryGroup();
    geometry_group->addChild( mesh.geom_instance );
    geometry_group->setAcceleration( context->createAcceleration( "Bvh" ) );
    context[ "top_object"   ]->set( geometry_group );
    context[ "top_shadower" ]->set( geometry_group );

    return true;
}


void setupLights()
{
    const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components

    BasicLight lights[] = {
        { make_float3( -0.5f,  0.25f, -1.0f ), make_float3( 0.2f, 0.2f, 0.25f ), 0, 0 },
        { make_float3( -0.5f,  0.0f ,  1.0f ), make_float3( 0.1f, 0.1f, 0.10f ), 0, 0 },
        { make_float3(  0.5f,  0.5f ,  0.5f ), make_float3( 0.7f, 0.7f, 0.65f ), 1, 0 }
    };
    lights[0].pos *= max_dim * 10.0f;
    lights[1].pos *= max_dim * 10.0f;
    lights[2].pos *= max_dim * 10.0f;

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( BasicLight ) );
    light_buffer->setSize( sizeof( lights ) / sizeof( lights[0] ) );
    memcpy( light_buffer->map(), lights, sizeof( lights ) );
    light_buffer->unmap();

    context[ "lights" ]->set( light_buffer );
}


void setupCamera()
{
    camera_eye    = aabb.m_min;
    float3 size   = aabb.m_max - aabb.m_min;
    camera_eye   += size * make_float3(0.25f, 4.f, 1.f);

    camera_lookat = aabb.center();
    camera_up     = make_float3( 0.0f, 0.0f, 1.0f );

    camera_rotate = Matrix4x4::identity();
}


void updateCamera()
{
    if (!camera_changed) return;
    camera_changed = false;

    const float vfov = 35.0f;
    const float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);

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
    const Matrix4x4 trans     = frame*camera_rotate*camera_rotate*frame_inv;

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

        case 's':
        {
            const std::string outputImage = std::string( SAMPLE_NAME ) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
            break;
        }

        case 'm':
            mesh_cur_mat_idx = (mesh_cur_mat_idx + 1) % mdl_material_programs.size();
            applyMDLMaterial( mesh_material, mesh_intersect_prog, mesh_cur_mat_idx );
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


void mouseMotionGlut( int x, int y)
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
        camera_changed = true;
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
        const float2 from = { static_cast<float>( mouse_prev_pos.x ),
                              static_cast<float>( mouse_prev_pos.y ) };
        const float2 to   = { static_cast<float>( x ),
                              static_cast<float>( y ) };

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

    glViewport(0, 0, width, height);

    glutPostRedisplay();
}


void initializeGlut( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( "MDL displacement - press m to change the material" );
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
    glutReshapeWindow( width, height);

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
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help               Print this usage message and exit.\n"
        "  -f | --file <filename>    Save single frame to file and exit.\n"
        "  -n | --nopbo              Disable GL interop for display buffer.\n"
        "  -m | --mesh <mesh_file>   Specify path to mesh to be loaded.\n"
        "App Keystrokes:\n"
        "  q  Quit\n"
        "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
        "  m  Select next material for the object\n"
        << std::endl;

    exit(1);
}


int main( int argc, char** argv )
 {
    std::string out_file;
    std::string mesh_file = std::string( sutil::samplesDir() ) + "/data/Kumanomi.obj";

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "-f" || arg == "--file"  )
        {
            if( i + 1 >= argc )
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
        else if( arg == "-m" || arg == "--mesh" )
        {
            if( i + 1 >= argc )
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            mesh_file = argv[++i];
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
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
        createContext();
        initMDL();
        if( !loadMesh( mesh_file ) )
            return 1;
        setupCamera();
        setupLights();

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
