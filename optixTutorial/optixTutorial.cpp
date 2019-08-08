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
//  tutorial
//
//-----------------------------------------------------------------------------

// 0 - normal shader
// 1 - lambertian
// 2 - specular
// 3 - shadows
// 4 - reflections
// 5 - miss
// 6 - schlick
// 7 - procedural texture on floor
// 8 - LGRustyMetal
// 9 - intersection
// 10 - anyhit
// 11 - camera



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
#include "random.h"
#include <Arcball.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdint.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixTutorial";

static float rand_range(float min, float max)
{
    static unsigned int seed = 0u;
    return min + (max - min) * rnd(seed);
}


//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context      context;
uint32_t     width  = 1080u;
uint32_t     height = 720;
bool         use_pbo = true;

std::string  texture_path;
const char*  tutorial_ptx;
int          tutorial_number = 10;

// Camera state
float3       camera_up;
float3       camera_lookat;
float3       camera_eye;
Matrix4x4    camera_rotate;
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
    context->setStackSize( 4640 );
    if( tutorial_number < 8 )
        context->setMaxTraceDepth( 5 );
    else
        context->setMaxTraceDepth( 31 );

    // Note: high max depth for reflection and refraction through glass
    context["max_depth"]->setInt( 100 );
    context["scene_epsilon"]->setFloat( 1.e-4f );
    context["importance_cutoff"]->setFloat( 0.01f );
    context["ambient_light_color"]->setFloat( 0.31f, 0.33f, 0.28f );

    // Output buffer
    // First allocate the memory for the GL buffer, then attach it to OptiX.
    GLuint vbo = 0;
    glGenBuffers( 1, &vbo );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glBufferData( GL_ARRAY_BUFFER, 4 * width * height, 0, GL_STREAM_DRAW);
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo );
    context["output_buffer"]->set( buffer );


    // Ray generation program
    const std::string camera_name = tutorial_number >= 11 ? "env_camera" : "pinhole_camera";
    Program ray_gen_program = context->createProgramFromPTXString( tutorial_ptx, camera_name );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( tutorial_ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    const std::string miss_name = tutorial_number >= 5 ? "envmap_miss" : "miss";
    context->setMissProgram( 0, context->createProgramFromPTXString( tutorial_ptx, miss_name ) );
    const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
    const std::string texpath = texture_path + "/" + std::string( "CedarCity.hdr" );
    context["envmap"]->setTextureSampler( sutil::loadTexture( context, texpath, default_color) );
    context["bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) );

    // 3D solid noise buffer, 1 float channel, all entries in the range [0.0, 1.0].

    const int tex_width  = 64;
    const int tex_height = 64;
    const int tex_depth  = 64;
    Buffer noiseBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, tex_width, tex_height, tex_depth);
    float *tex_data = (float *) noiseBuffer->map();

    // Random noise in range [0, 1]
    for (int i = tex_width * tex_height * tex_depth;  i > 0; i--) {
        // One channel 3D noise in [0.0, 1.0] range.
        *tex_data++ = rand_range(0.0f, 1.0f);
    }
    noiseBuffer->unmap(); 


    // Noise texture sampler
    TextureSampler noiseSampler = context->createTextureSampler();

    noiseSampler->setWrapMode(0, RT_WRAP_REPEAT);
    noiseSampler->setWrapMode(1, RT_WRAP_REPEAT);
    noiseSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    noiseSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    noiseSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    noiseSampler->setMaxAnisotropy(1.0f);
    noiseSampler->setMipLevelCount(1);
    noiseSampler->setArraySize(1);
    noiseSampler->setBuffer(0, 0, noiseBuffer);

    context["noise_texture"]->setTextureSampler(noiseSampler);
}

float4 make_plane( float3 n, float3 p )
{
    n = normalize(n);
    float d = -dot(n, p);
    return make_float4( n, d );
}

void createGeometry()
{
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "box.cu" );
    Program box_bounds    = context->createProgramFromPTXString( ptx, "box_bounds" );
    Program box_intersect = context->createProgramFromPTXString( ptx, "box_intersect" );

    // Create box
    Geometry box = context->createGeometry();
    box->setPrimitiveCount( 1u );
    box->setBoundingBoxProgram( box_bounds );
    box->setIntersectionProgram( box_intersect );
    box["boxmin"]->setFloat( -2.0f, 0.0f, -2.0f );
    box["boxmax"]->setFloat(  2.0f, 7.0f,  2.0f );

    // Create chull
    Geometry chull = 0;
    if( tutorial_number >= 9){
        chull = context->createGeometry();
        chull->setPrimitiveCount( 1u );
        chull->setBoundingBoxProgram( context->createProgramFromPTXString( tutorial_ptx, "chull_bounds" ) );
        chull->setIntersectionProgram( context->createProgramFromPTXString( tutorial_ptx, "chull_intersect" ) );
        Buffer plane_buffer = context->createBuffer(RT_BUFFER_INPUT);
        plane_buffer->setFormat(RT_FORMAT_FLOAT4);
        int nsides = 6; 
        plane_buffer->setSize( nsides + 2 );
        float4* chplane = (float4*)plane_buffer->map();
        float radius = 1;
        float3 xlate = make_float3(-1.4f, 0, -3.7f);

        for(int i = 0; i < nsides; i++){
            float angle = float(i)/float(nsides) * M_PIf * 2.0f;
            float x = cos(angle);
            float y = sin(angle);
            chplane[i] = make_plane( make_float3(x, 0, y), make_float3(x*radius, 0, y*radius) + xlate);
        }
        float min = 0.02f;
        float max = 3.5f;
        chplane[nsides + 0] = make_plane( make_float3(0, -1, 0), make_float3(0, min, 0) + xlate);
        float angle = 5.f/nsides * M_PIf * 2;
        float pitch = 0.7f;
        float ytopOffset = (radius / pitch) / cos(M_PIf / nsides);
        chplane[nsides + 1] = make_plane( make_float3(cos(angle), pitch, sin(angle)), make_float3(0, max, 0) + xlate);
        plane_buffer->unmap();
        chull["planes"]->setBuffer(plane_buffer);
        float radoffset = radius / cos(M_PIf / nsides);
        chull["chull_bbmin"]->setFloat(-radoffset + xlate.x, min + xlate.y, -radoffset + xlate.z);
        chull["chull_bbmax"]->setFloat( radoffset + xlate.x, max + xlate.y + ytopOffset,  radoffset + xlate.z);
    }

    // Floor geometry
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );
    ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
    parallelogram->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    parallelogram->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    float3 anchor = make_float3( -64.0f, 0.01f, -64.0f );
    float3 v1 = make_float3( 128.0f, 0.0f, 0.0f );
    float3 v2 = make_float3( 0.0f, 0.0f, 128.0f );
    float3 normal = cross( v2, v1 );
    normal = normalize( normal );
    float d = dot( normal, anchor );
    v1 *= 1.0f/dot( v1, v1 );
    v2 *= 1.0f/dot( v2, v2 );
    float4 plane = make_float4( normal, d );
    parallelogram["plane"]->setFloat( plane );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );
    parallelogram["anchor"]->setFloat( anchor );

    // Materials
    std::string box_chname;
    if(tutorial_number >= 8){
        box_chname = "box_closest_hit_radiance";
    } else if(tutorial_number >= 3){
        box_chname = "closest_hit_radiance3";
    } else if(tutorial_number >= 2){
        box_chname = "closest_hit_radiance2";
    } else if(tutorial_number >= 1){
        box_chname = "closest_hit_radiance1";
    } else {
        box_chname = "closest_hit_radiance0";
    }

    Material box_matl = context->createMaterial();
    Program box_ch = context->createProgramFromPTXString( tutorial_ptx, box_chname.c_str() );
    box_matl->setClosestHitProgram( 0, box_ch );
    if( tutorial_number >= 3) {
        Program box_ah = context->createProgramFromPTXString( tutorial_ptx, "any_hit_shadow" );
        box_matl->setAnyHitProgram( 1, box_ah );
    }
    box_matl["Ka"]->setFloat( 0.3f, 0.3f, 0.3f );
    box_matl["Kd"]->setFloat( 0.6f, 0.7f, 0.8f );
    box_matl["Ks"]->setFloat( 0.8f, 0.9f, 0.8f );
    box_matl["phong_exp"]->setFloat( 88 );
    box_matl["reflectivity_n"]->setFloat( 0.2f, 0.2f, 0.2f );

    std::string floor_chname;
    if(tutorial_number >= 7){
        floor_chname = "floor_closest_hit_radiance";
    } else if(tutorial_number >= 6){
        floor_chname = "floor_closest_hit_radiance5";
    } else if(tutorial_number >= 4){
        floor_chname = "floor_closest_hit_radiance4";
    } else if(tutorial_number >= 3){
        floor_chname = "closest_hit_radiance3";
    } else if(tutorial_number >= 2){
        floor_chname = "closest_hit_radiance2";
    } else if(tutorial_number >= 1){
        floor_chname = "closest_hit_radiance1";
    } else {
        floor_chname = "closest_hit_radiance0";
    }

    Material floor_matl = context->createMaterial();
    Program floor_ch = context->createProgramFromPTXString( tutorial_ptx, floor_chname.c_str() );
    floor_matl->setClosestHitProgram( 0, floor_ch );
    if(tutorial_number >= 3) {
        Program floor_ah = context->createProgramFromPTXString( tutorial_ptx, "any_hit_shadow" );
        floor_matl->setAnyHitProgram( 1, floor_ah );
    }
    floor_matl["Ka"]->setFloat( 0.3f, 0.3f, 0.1f );
    floor_matl["Kd"]->setFloat( 194/255.f*.6f, 186/255.f*.6f, 151/255.f*.6f );
    floor_matl["Ks"]->setFloat( 0.4f, 0.4f, 0.4f );
    floor_matl["reflectivity"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["reflectivity_n"]->setFloat( 0.05f, 0.05f, 0.05f );
    floor_matl["phong_exp"]->setFloat( 88 );
    floor_matl["tile_v0"]->setFloat( 0.25f, 0, .15f );
    floor_matl["tile_v1"]->setFloat( -.15f, 0, 0.25f );
    floor_matl["crack_color"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["crack_width"]->setFloat( 0.02f );

    // Glass material
    Material glass_matl;
    if( chull.get() ) {
        Program glass_ch = context->createProgramFromPTXString( tutorial_ptx, "glass_closest_hit_radiance" );
        const std::string glass_ahname = tutorial_number >= 10 ? "glass_any_hit_shadow" : "any_hit_shadow";
        Program glass_ah = context->createProgramFromPTXString( tutorial_ptx, glass_ahname.c_str() );
        glass_matl = context->createMaterial();
        glass_matl->setClosestHitProgram( 0, glass_ch );
        glass_matl->setAnyHitProgram( 1, glass_ah );

        glass_matl["importance_cutoff"]->setFloat( 1e-2f );
        glass_matl["cutoff_color"]->setFloat( 0.34f, 0.55f, 0.85f );
        glass_matl["fresnel_exponent"]->setFloat( 3.0f );
        glass_matl["fresnel_minimum"]->setFloat( 0.1f );
        glass_matl["fresnel_maximum"]->setFloat( 1.0f );
        glass_matl["refraction_index"]->setFloat( 1.4f );
        glass_matl["refraction_color"]->setFloat( 1.0f, 1.0f, 1.0f );
        glass_matl["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
        glass_matl["refraction_maxdepth"]->setInt( 100 );
        glass_matl["reflection_maxdepth"]->setInt( 100 );
        float3 extinction = make_float3(.80f, .89f, .75f);
        glass_matl["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );
        glass_matl["shadow_attenuation"]->setFloat( 0.4f, 0.7f, 0.4f );
    }

    // Create GIs for each piece of geometry
    std::vector<GeometryInstance> gis;
    gis.push_back( context->createGeometryInstance( box, &box_matl, &box_matl+1 ) );
    gis.push_back( context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 ) );
    if(chull.get())
        gis.push_back( context->createGeometryInstance( chull, &glass_matl, &glass_matl+1 ) );

    // Place all in group
    GeometryGroup geometrygroup = context->createGeometryGroup();
    geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
    geometrygroup->setChild( 0, gis[0] );
    geometrygroup->setChild( 1, gis[1] );
    if(chull.get()) {
        geometrygroup->setChild( 2, gis[2] );
    }
    geometrygroup->setAcceleration( context->createAcceleration("Trbvh") );
    //geometrygroup->setAcceleration( context->createAcceleration("NoAccel") );

    context["top_object"]->set( geometrygroup );
    context["top_shadower"]->set( geometrygroup );

}


void setupCamera()
{
    camera_eye    = make_float3( 7.0f, 9.2f, -6.0f );
    camera_lookat = make_float3( 0.0f, 4.0f,  0.0f );
    camera_up     = make_float3( 0.0f, 1.0f,  0.0f );

    camera_rotate  = Matrix4x4::identity();
}


void setupLights()
{

    BasicLight lights[] = { 
        { make_float3( -5.0f, 60.0f, -16.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
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
    const float vfov = 60.0f;
    const float aspect_ratio = static_cast<float>(width) /
                               static_cast<float>(height);

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

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
    updateCamera();

    context->launch( 0, width, height );

    Buffer buffer = getOutputBuffer();
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
    }

    mouse_prev_pos = make_int2( x, y );
}


void glutResize( int w, int h )
{
    if ( w == (int)width && h == (int)height ) return;

    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    sutil::resizeBuffer( getOutputBuffer(), width, height );

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
        "  -T | --tutorial-number <num>              Specify tutorial number\n"
        "  -t | --texture-path <path>                Specify path to texture directory\n"
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
        else if ( arg == "-f" || arg == "--file" )
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
        else if ( arg == "-t" || arg == "--texture-path" )
        {
            if ( i == argc-1 ) {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            texture_path = argv[++i];
        }
        else if ( arg == "-T" || arg == "--tutorial-number" )
        {
            if ( i == argc-1 ) {
                printUsageAndExit( argv[0] );
            }
            tutorial_number = atoi(argv[++i]);
            if ( tutorial_number < 0 || tutorial_number > 11 ) {
                std::cerr << "Tutorial number (" << tutorial_number << ") is out of range [0..11]\n";
                printUsageAndExit( argv[0] );
            }
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    if( texture_path.empty() ) {
        texture_path = std::string( sutil::samplesDir() ) + "/data";
    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        // load the ptx source associated with tutorial number
        std::stringstream ss;
        ss << "tutorial" << tutorial_number << ".cu";
        std::string tutorial_ptx_path = ss.str();
        tutorial_ptx = sutil::getPtxString( SAMPLE_NAME, tutorial_ptx_path.c_str() );

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

