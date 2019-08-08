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
//  Render a sphere with an MDL material and environment light
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include <Arcball.h>
#include <HDRLoader.h>

using namespace optix;

#include "shared_structs.h"

#include <mdl_wrapper.h>
#define Mdl_helper Mdl_wrapper


const char* const SAMPLE_NAME = "optixMDLSphere";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context context;
int     width        = 1024;
int     height       = 768;
int     frame_number = 0;

Mdl_helper *mdl_helper = NULL;

std::vector<Mdl_BSDF_program_group> mdl_material_programs;

Material sphere_material;
size_t   sphere_cur_mat_idx = 0;

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
    if ( mdl_helper )
    {
        delete mdl_helper;
        mdl_helper = NULL;
    }
    if ( context )
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
    context->setRayTypeCount( 1 );
    context->setEntryPointCount( 1 );

    context["scene_epsilon"]->setFloat( 1.e-4f );

    // uchar display buffer
    {
        Variable output_buffer = context["output_buffer"];
        Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height );
        output_buffer->set( buffer );
    }

    // float accumulation buffer
    {
        Variable accum_buffer = context["accum_buffer"];
        Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height );
        accum_buffer->set( buffer );
    }

    // Ray generation program
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "camera.cu" );
    Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );

    // Miss program
    ptx = sutil::getPtxString( SAMPLE_NAME, "mdl_ibl.cu" );
    context->setMissProgram( 0, context->createProgramFromPTXString( ptx, "miss" ) );

    return context;
}


void setupCamera()
{
    camera_eye    = make_float3( 0.0f, 0.0f, 5.0f );
    camera_lookat = make_float3( 0.0f, 0.0f, 0.0f );
    camera_up     = make_float3( 0.0f, 1.0f, 0.0f );

    camera_rotate  = Matrix4x4::identity();
}


void updateCamera()
{
    if ( camera_changed )
        frame_number = 0;  // reset accumulation
    context[ "frame_number" ]->setUint( frame_number++ );

    if ( !camera_changed ) return;

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

    context[ "eye"]->setFloat( camera_eye );
    context[ "U"  ]->setFloat( camera_u );
    context[ "V"  ]->setFloat( camera_v );
    context[ "W"  ]->setFloat( camera_w );
}


// Helper function to extract the module name from a fully-qualified material name.
std::string get_module_name( const std::string& material_name )
{
    std::size_t p = material_name.rfind( "::" );
    return material_name.substr( 0, p );
}


// Helper function to extract the material name from a fully-qualified material name.
std::string get_material_name( const std::string& material_name )
{
    std::size_t p = material_name.rfind( "::" );
    if( p == std::string::npos )
        return material_name;
    return material_name.substr( p + 2, material_name.size() - p );
}


size_t addMDLMaterial( const std::string &full_material_name )
{
    Mdl_BSDF_program_group bsdf_progs = mdl_helper->compile_df(
        get_module_name( full_material_name ),
        get_material_name( full_material_name ),
        "surface.scattering" );

    // Force the programs to be bindless to avoid long compilation times, when first used
    bsdf_progs.init_prog->getId();
    bsdf_progs.sample_prog->getId();
    bsdf_progs.evaluate_prog->getId();
    bsdf_progs.pdf_prog->getId();

    mdl_material_programs.push_back( bsdf_progs );
    return mdl_material_programs.size() - 1;
}


void applyMDLMaterial( Material material, size_t material_idx )
{
    Mdl_BSDF_program_group &bsdf_progs = mdl_material_programs[material_idx];

    material["mdl_bsdf_init"    ]->setProgramId( bsdf_progs.init_prog );
    material["mdl_bsdf_sample"  ]->setProgramId( bsdf_progs.sample_prog );
    material["mdl_bsdf_evaluate"]->setProgramId( bsdf_progs.evaluate_prog );
    material["mdl_bsdf_pdf"     ]->setProgramId( bsdf_progs.pdf_prog );
}


void initMDL( const std::vector<std::string> &material_names )
{
    mdl_helper = new Mdl_helper(
        context,
        /*mdl_textures_ptx_path=*/ "",
        /*module_path=*/ std::string( sutil::samplesDir() ) + "/data",
        /*num_texture_spaces=*/ 1,
        /*num_texture_results=*/ 16 );
    mdl_helper->set_mdl_textures_ptx_string(
        sutil::getPtxString( SAMPLE_NAME, "mdl_textures.cu" ) );

    for( size_t i = 0, n = material_names.size(); i < n; ++i )
        addMDLMaterial( material_names[i] );
}


Geometry createSphere()
{
    Geometry sphere = context->createGeometry();
    sphere->setPrimitiveCount( 1u );
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "sphere_texcoord_tangent.cu" );
    sphere->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    sphere->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    sphere["sphere"]->setFloat( 0, 0, 0, 1.5 );
    return sphere;
}


Material createMaterial( size_t material_idx )
{
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "mdl_ibl.cu" );
    Program chp = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );

    Material matl = context->createMaterial();
    matl->setClosestHitProgram( 0, chp );

    applyMDLMaterial( matl, material_idx );

    return matl;
}


GeometryInstance createInstance( Geometry geometry, Material material )
{
    GeometryInstance gi = context->createGeometryInstance();
    gi->setGeometry( geometry );
    gi->addMaterial( material );
    return gi;
}


void createScene()
{
    // Setup geometry

    // Sphere
    Geometry sphere = createSphere();
    sphere_material = createMaterial( sphere_cur_mat_idx );
    GeometryInstance inst = createInstance( sphere, sphere_material );

    // Create geometry group
    GeometryGroup geometry_group = context->createGeometryGroup();
    geometry_group->addChild(inst);
    geometry_group->setAcceleration( context->createAcceleration( "NoAccel" ) );

    context["top_object"]->set( geometry_group );
}


// helper for createEnvironment()
static float build_alias_map(
    const float *data,
    const unsigned int size,
    Env_accel *accel )
{
    // create qs (normalized)
    float sum = 0.0f;
    for( unsigned int i = 0; i < size; ++i )
        sum += data[i];

    for( unsigned int i = 0; i < size; ++i )
        accel[i].q = (float)((float)size * data[i] / sum);

    // create partition table
    unsigned int *partition_table = (unsigned int *)malloc( size * sizeof(unsigned int) );
    unsigned int s = 0u, large = size;
    for( unsigned int i = 0; i < size; ++i )
        partition_table[(accel[i].q < 1.0f) ? (s++) : (--large)] = accel[i].alias = i;

    // create alias map
    for( s = 0; s < large && large < size; ++s )
    {
        const unsigned int j = partition_table[s], k = partition_table[large];
        accel[j].alias = k;
        accel[k].q += accel[j].q - 1.0f;
        large = (accel[k].q < 1.0f) ? (large + 1u) : large;
    }

    free( partition_table );

    return sum;
}


void createEnvironment( const std::string &filename )
{
    // read data
    HDRLoader hdr( filename );
    if( hdr.failed() )
    {
        std::cerr << "failed to load HDR environment from file \"" << filename << "\"\n";
        exit(1);
    }
    const unsigned int rx = hdr.width();
    const unsigned int ry = hdr.height();
    context["env_size"]->setUint( rx, ry );
    const float *pixels = hdr.raster(); // float4

    // create importance sampling acceleration
    Buffer env_accel_buffer = context->createBuffer( RT_BUFFER_INPUT );
    env_accel_buffer->setFormat( RT_FORMAT_USER );
    env_accel_buffer->setElementSize( sizeof(Env_accel) );
    env_accel_buffer->setSize( rx * ry );

    Env_accel *env_accel = static_cast<Env_accel *>( env_accel_buffer->map() );
    float *importance_data = (float *)malloc( rx * ry * sizeof(float) );
    float cos_theta0 = 1.0f;
    const float step_phi = (float)(2.0 * M_PI) / (float)rx;
    const float step_theta = (float)M_PI / (float)ry;
    for( unsigned int y = 0; y < ry; ++y )
    {
        const float theta1 = (float)(y + 1) * step_theta;
        const float cos_theta1 = cos(theta1);
        const float area = (cos_theta0 - cos_theta1) * step_phi;
        cos_theta0 = cos_theta1;

        for( unsigned int x = 0; x < rx; ++x )
        {
            const unsigned int idx = y * rx + x;
            const unsigned int idx4 =  idx * 4;
            importance_data[idx] = area * std::max( pixels[idx4], std::max( pixels[idx4 + 1], pixels[idx4 + 2] ) );
        }
    }
    const float inv_env_integral = 1.0f / build_alias_map(importance_data, rx * ry, env_accel);
    free( importance_data );
    for( unsigned int i = 0; i < rx * ry; ++i )
    {
        const unsigned int idx4 = i * 4;
        env_accel[i].pdf = std::max( pixels[idx4], std::max( pixels[idx4 + 1], pixels[idx4 + 2] ) )* inv_env_integral;
    }
    env_accel_buffer->unmap();
    context["env_accel"]->setBuffer( env_accel_buffer );

    // create pixel data buffer
    Buffer pixel_buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, rx, ry );
    void *buffer_data = pixel_buffer->map();
    std::memcpy( buffer_data, pixels, rx * ry * sizeof(float) * 4 );
    pixel_buffer->unmap();

    // create texture sampler
    TextureSampler sampler = context->createTextureSampler();
    sampler->setWrapMode( 0, RT_WRAP_REPEAT );
    sampler->setWrapMode( 1, RT_WRAP_CLAMP_TO_EDGE ); // don't sample beyond the poles of the environment sphere
    sampler->setWrapMode( 2, RT_WRAP_REPEAT );
    sampler->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
    sampler->setMaxAnisotropy( 1.0f );
    sampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );
    sampler->setBuffer( pixel_buffer );
    context["env_texture"]->setTextureSampler( sampler );
}


void initializeGlut( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( "MDL material sphere" );
    glutHideWindow();
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

        case 'm':
            sphere_cur_mat_idx = (sphere_cur_mat_idx + 1) % mdl_material_programs.size();
            applyMDLMaterial( sphere_material, sphere_cur_mat_idx );
            frame_number = 0;  // reset accumulation
            break;

        case '0':
        case '1':
        case '2':
        case '3':
            context["mdl_test_type"]->setInt( int(k - '0') );
            frame_number = 0;  // reset accumulation
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
    if( w == (int)width && h == (int)height ) return;

    frame_number = 0;

    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    camera_changed = true;  // force updating the camera to adjust to new aspect ratio

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    sutil::resizeBuffer( context["accum_buffer"]->getBuffer(), width, height );

    glViewport( 0, 0, width, height );

    glutPostRedisplay();
}


void runGlut()
{
    // Initialize GL state
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1 );

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

static void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options] [<material_name1> ...]\n"
        "Options: --help | -h             Print this usage message\n"
        "         --file | -f <filename>  Specify PPM file for image output (default: none for progressive display)\n"
        "         --spp                   Specify samples per pixel to render (default: 1024)\n"
        "         --exposure              Specify exposure (default: 0)\n"
        "         --hdr <filename>        Specify HDR file as environment (default: environment.hdr)\n"
        "         --mdl-test-type <type>  0: evaluate, 1: sample, 2: mis, 3: mis+pdf (default: 2)\n"
        "         --dim=<width>x<height>  Set image dimensions; defaults to 1024x768\n"
        "\n"
        "Default material: example_df::df_material\n"
        "\n"
        "App Keystrokes:\n"
        "  m Switch to the next MDL material\n"
        "  0 Set MDL test type to \"evaluate\"\n"
        "  1 Set MDL test type to \"sample\"\n"
        "  2 Set MDL test type to \"mis\"\n"
        "  3 Set MDL test type to \"mis+pdf\"\n"
        << std::endl;
    exit(1);
}


int main( int argc, char** argv )
{
    // Process command line options
    std::string outfile;
    std::string hdrfile = std::string( sutil::samplesDir() ) + "/data/environment.hdr";
    std::vector<std::string> material_names;
    float exposure_scale = 1.0f;
    int mdl_test_type = MDL_TEST_MIS;
    int num_samples = 1024;

    for( int i = 1; i < argc; ++i ) {
        std::string arg( argv[i] );
        if( !arg.empty() && arg[0] == '-') {
            #define string_arg( arg ) \
                if( i < argc - 1 ) arg = argv[++i]; \
                else printUsageAndExit( argv[0] );

            if( arg == "--help" || arg == "-h" ) {
                printUsageAndExit( argv[0] );
            } else if( arg == "--file" || arg == "-f" ) {
                string_arg(outfile);
            } else if( arg == "--exposure" ) {
                std::string exposure;
                float exposure_value;
                string_arg( exposure );
                if( sscanf( exposure.c_str(), "%f", &exposure_value ) != 1 ) {
                    std::cerr << "Invalid value for --exposure: " << exposure << std::endl;
                    exit(1);
                }
                exposure_scale = powf( 2.0f, exposure_value );
            } else if( arg == "--spp" ) {
                std::string spp;
                int spp_value;
                string_arg( spp );
                if( sscanf( spp.c_str(), "%d", &spp_value ) != 1 || spp_value <= 0 ) {
                    std::cerr << "Invalid value for --spp: " << spp << std::endl;
                    exit(1);
                }
                num_samples = spp_value;
            } else if( arg == "--hdr" ) {
                string_arg( hdrfile );
            } else if( arg == "--mdl-test-type" ) {
                std::string test_type;
                int test_type_value;
                string_arg( test_type );
                if( sscanf( test_type.c_str(), "%d", &test_type_value ) != 1 ||
                        (unsigned)test_type_value >= MDL_TEST_COUNT ) {
                    std::cerr << "Invalid value for --mdl-test-type: " << test_type << std::endl;
                    exit(1);
                }
                mdl_test_type = test_type_value;
            } else if( arg.substr( 0, 6 ) == "--dim=" ) {
                std::string dims_arg = arg.substr( 6 );
                sutil::parseDimensions( dims_arg.c_str(), width, height );
            } else {
                std::cerr << "Unknown option: '" << arg << "'" << std::endl;
                printUsageAndExit( argv[0] );
            }

            #undef string_arg
        } else {
            material_names.push_back(arg);
        }
    }

    // No material specified? Use default material
    if( material_names.empty() )
        material_names.push_back("::example_df::df_material");

    try
    {
        initializeGlut( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        // Setup state & scene
        context = createContext();
        initMDL( material_names );
        setupCamera();
        createScene();
        createEnvironment( hdrfile );

        context["mdl_test_type"]->setInt( mdl_test_type );
        context["tonemap_scale"]->setFloat( exposure_scale );

        context->validate();

        if( outfile.empty() )
        {
            runGlut();
        }
        else
        {
            updateCamera();
            for( int s = 0; s < num_samples; ++s ) {
                std::cout << "rendering sample " << s << " of " << num_samples << "\n";
                context[ "frame_number" ]->setUint( s );
                context->launch( 0, width, height );
            }
            sutil::displayBufferPPM( outfile.c_str(), context["output_buffer"]->getBuffer() );
        }

        // Clean up
        destroyContext();
        return 0;
    }
    SUTIL_CATCH( context->get() )
}
