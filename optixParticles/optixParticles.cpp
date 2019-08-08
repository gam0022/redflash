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
// optixParticles: dynamic particle  generator
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

#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdint.h>
#include <map>

using namespace optix;

const char* const SAMPLE_NAME = "optixParticles";

struct ParticlesBuffer
{
    Buffer      positions;
    Buffer      velocities;
    Buffer      colors;
    Buffer      radii;
};

struct ParticleFrameData {
    std::vector<float3> positions;
    std::vector<float3> velocities;
    std::vector<float3> colors;
    std::vector<float>  radii;
    float3 bbox_min;
    float3 bbox_max;
};

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context         context;
uint32_t        width  = 1024u;
uint32_t        height = 768u;
bool            use_pbo = true;
bool            shade_particles = false;
bool            play = false;
unsigned int    iterations_per_animation_frame = 1;
float           motion_blur = -1.0f;
optix::Aabb     aabb;

// Camera state
float3          camera_up;
float3          camera_lookat;
float3          camera_eye;
Matrix4x4       camera_rotate;
sutil::Arcball  arcball;

// Mouse state
int2            mouse_prev_pos;
int             mouse_button;

// Particles frame state
std::string     particles_file_base;
int             current_particle_frame = 1;
int             max_particle_frames = 25;

// Accumulation frame
unsigned int    accumulation_frame = 0;

// Buffers, Geometry and GeometryGroup for the particle set
ParticlesBuffer buffers;
Geometry        geometry;
GeometryGroup   geometry_group;

//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

struct UsageReportLogger;

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext( int usage_report_level, UsageReportLogger* logger );
void loadMesh( const std::string& filename );
void setupCamera();
void setupLights();
void updateCamera();
void glutInitialize( int* argc, char** argv );
void glutRun();

void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutMousePress( int button, int state, int x, int y );
void glutMouseMotion( int x, int y );
void glutArrowPress( int k, int x, int y );
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


struct UsageReportLogger
{
  void log( int lvl, const char* tag, const char* msg )
  {
    std::cout << "[" << lvl << "][" << std::left << std::setw( 12 ) << tag << "] " << msg;
  }
};


// Static callback
void usageReportCallback( int lvl, const char* tag, const char* msg, void* cbdata )
{
    // Route messages to a C++ object (the "logger"), as a real app might do.
    // We could have printed them directly in this simple case.

    UsageReportLogger* logger = reinterpret_cast<UsageReportLogger*>( cbdata );
    logger->log( lvl, tag, msg );
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


void setParticlesBaseName( const std::string &particles_file )
{
    // gets the base name by stripping the suffix and the number, if there is no number
    // then there will be no sequence
    bool found_sequence = false;
    std::string::size_type last_dot_pos = particles_file.rfind( "." );
    if ( last_dot_pos != std::string::npos ) {
        // looks now for a possible second dot in the name
        std::string first_part = particles_file.substr( 0, last_dot_pos );
        std::string::size_type second_dot_pos = first_part.rfind( "." );

        if ( second_dot_pos != std::string::npos ) {
            std::string frame_str = particles_file.substr( second_dot_pos+1, last_dot_pos-second_dot_pos-1 );
            current_particle_frame = atoi( frame_str.c_str() );
            particles_file_base = particles_file.substr( 0, second_dot_pos );
            found_sequence = true;
        }
    }

    if ( !found_sequence ) {
        // either there are no dots in the name or there is just one, it won't be a sequence
        current_particle_frame = 0;
        particles_file_base = particles_file;
    }
}


void createContext( int usage_report_level, UsageReportLogger* logger )
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    if( usage_report_level > 0 )
    {
        context->setUsageReportCallback( usageReportCallback, usage_report_level, logger );
    }

    context["scene_epsilon"    ]->setFloat( 3.e-3f );

    Buffer output_buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo );
    context["output_buffer"]->set( output_buffer );

    Buffer accum_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT4, width, height );
    context["accum_buffer"]->set( accum_buffer );

    if( motion_blur > 0.0f )
    {
        // Timeview buffers (they are used by the camera program in "accum_camera_mblur.cu", while
        // the camera program in "accum_camera.cu" enables timeviewing by means of a define constant).
        // These buffers are needed by the camera program but they are not used in this example
        Buffer buf = context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, 2 );
        context[ "timeview_min_max" ]->set( buf );

        context[ "time_range" ]->setFloat( 0.0f, 1.0f );
    }

    // Ray generation program
    const char *ptx;
    if( motion_blur > 0.0f )
    {   // motion blur enabled
        ptx = sutil::getPtxString( SAMPLE_NAME, "accum_camera_mblur.cu" );
    }
    else
    {   // no motion blur
        ptx = sutil::getPtxString( SAMPLE_NAME, "accum_camera.cu" );
    }
    Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );

    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    context->setMissProgram( 0, context->createProgramFromPTXString( sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" ), "miss" ) );
    if ( shade_particles ) // Lambert-shaded, light background
        context["bg_color"]->setFloat( 0.34f, 0.55f, 0.85f );
    else // flat shaded, dark background
        context["bg_color"]->setFloat( 0.07f, 0.11f, 0.17f );
}


static inline float parseFloat( const char *&token )
{
  token += strspn( token, " \t" );
  float f = (float) atof( token );
  token += strcspn( token, " \t\r" );
  return f;
}


static inline float3 get_min(
    const float3 &v1,
    const float3 &v2)
{
    float3 result;
    result.x = std::min<float>( v1.x, v2.x );
    result.y = std::min<float>( v1.y, v2.y );
    result.z = std::min<float>( v1.z, v2.z );
    return result;
}


static inline float3 get_max(
    const float3 &v1,
    const float3 &v2)
{
    float3 result;
    result.x = std::max<float>( v1.x, v2.x );
    result.y = std::max<float>( v1.y, v2.y );
    result.z = std::max<float>( v1.z, v2.z );
    return result;
}


static void fillBuffers(
    const std::vector<float3> &positions,
    const std::vector<float3> &velocities,
    const std::vector<float3> &colors,
    const std::vector<float>  &radii)
{
    buffers.positions->setSize( positions.size() );
    float *pos = reinterpret_cast<float*> ( buffers.positions->map() );
    for ( int i=0, index = 0; i<static_cast<int>(positions.size()); ++i ) {
        float3 p = positions[i];

        pos[index++] = p.x;
        pos[index++] = p.y;
        pos[index++] = p.z;
    }
    buffers.positions->unmap();

    buffers.velocities->setSize( velocities.size() );
    float *vel = reinterpret_cast<float*> ( buffers.velocities->map() );
    for ( int i=0, index = 0; i<static_cast<int>(velocities.size()); ++i ) {
        float3 v = velocities[i];

        vel[index++] = v.x;
        vel[index++] = v.y;
        vel[index++] = v.z;
    }
    buffers.velocities->unmap();

    buffers.colors->setSize( colors.size() );
    float *col = reinterpret_cast<float*> ( buffers.colors->map() );
    for ( int i=0, index = 0; i<static_cast<int>(colors.size()); ++i ) {
        float3 c = colors[i];

        col[index++] = c.x;
        col[index++] = c.y;
        col[index++] = c.z;
    }
    buffers.colors->unmap();

    buffers.radii->setSize( radii.size() );
    float *rad = reinterpret_cast<float*> ( buffers.radii->map() );
    for ( int i=0; i<static_cast<int>(radii.size()); ++i ) {
        rad[i] = radii[i];
    }
    buffers.radii->unmap();
}


void createMaterialPrograms(
    Context context,
    Program &closest_hit,
    Program &any_hit)
{
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "particles_material.cu" );

    if( !closest_hit )
        closest_hit = context->createProgramFromPTXString( ptx, "closest_hit" );
    if( !any_hit )
        any_hit     = context->createProgramFromPTXString( ptx, "any_hit" );
}


Material createOptiXMaterial(
    Context context,
    Program closest_hit,
    Program any_hit)
{
    // this material does not have parameters attached, it is a simple model that uses the color
    // of the particles to shade them flat or as a Lambertian material where the diffuse color
    // is the particles' color
    Material mat = context->createMaterial();
    mat->setClosestHitProgram( 0u, closest_hit );
    mat->setAnyHitProgram( 1u, any_hit ) ;

    mat[ "flat_shaded" ]->setInt( shade_particles ? 0 : 1 );

    return mat;
}


Program createBoundingBoxProgram( Context context )
{
    bool has_motion = ( motion_blur > 0.0f );
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "particles_geometry.cu" );
    return context->createProgramFromPTXString( ptx, has_motion
        ? "particle_bounds_motion" : "particle_bounds" );
}


Program createIntersectionProgram( Context context )
{
    bool has_motion = ( motion_blur > 0.0f );
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "particles_geometry.cu" );
    return context->createProgramFromPTXString( ptx, has_motion
        ? "particle_intersect_motion" : "particle_intersect" );
}

// loads up the particles file corresponding to the current frame (if it is a sequence)
void loadParticles()
{

    // caching to avoid reloading the particles with every frame
    // however, we still refill the buffers and rebuild the acceleration structure
    static std::map<int, ParticleFrameData> dataCache;

    std::map<int, ParticleFrameData>::iterator cacheIt = dataCache.find(current_particle_frame);
    if(cacheIt == dataCache.end())
    {
        ParticleFrameData newCacheEntry;

        std::vector<float3>& positions = newCacheEntry.positions;
        std::vector<float3>& velocities = newCacheEntry.velocities;
        std::vector<float3>& colors = newCacheEntry.colors;
        std::vector<float>&  radii = newCacheEntry.radii;

        std::string filename = particles_file_base;

        if ( current_particle_frame > 0 ) {
            std::ostringstream s;
            s << current_particle_frame;

            if ( current_particle_frame < 10 )
                filename += ".000" + s.str() + ".txt";
            else
                filename += ".00" + s.str() + ".txt";
        }
        else
            filename = particles_file_base;


        std::ifstream ifs( filename.c_str() );

        float3 bbox_min, bbox_max;
        bbox_min.x = bbox_min.y = bbox_min.z = 1e16f;
        bbox_max.x = bbox_max.y = bbox_max.z = -1e16f;

        int maxchars = 8192;
        std::vector<char> buf(static_cast<size_t>(maxchars)); // Alloc enough size

        while ( ifs.peek() != -1 ) {
            ifs.getline( &buf[0], maxchars );

            std::string linebuf(&buf[0]);

            // Trim newline '\r\n' or '\n'
            if ( linebuf.size() > 0 ) {
                if ( linebuf[linebuf.size() - 1] == '\n' )
                    linebuf.erase(linebuf.size() - 1);
            }

            if ( linebuf.size() > 0 ) {
                if ( linebuf[linebuf.size() - 1] == '\r' )
                    linebuf.erase( linebuf.size() - 1 );
            }

            // Skip if empty line.
            if ( linebuf.empty() ) {
                continue;
            }

            // Skip leading space.
            const char *token = linebuf.c_str();
            token += strspn( token, " \t" );

            assert( token );
            if ( token[0] == '\0' )
                continue; // empty line

            if ( token[0] == '#' )
                continue; // comment line

            // meaningful line here. The expected format is: position, velocity, color and radius

            // position
            float x  = parseFloat( token );
            float y  = parseFloat( token );
            float z  = parseFloat( token );

            // velocity
            float vx = parseFloat( token );
            float vy = parseFloat( token );
            float vz = parseFloat( token );

            // color
            float r  = parseFloat( token );
            float g  = parseFloat( token );
            float b  = parseFloat( token );

            // radius
            float rd = parseFloat( token );

            positions.push_back( make_float3( x, y, z ) );
            velocities.push_back( make_float3( vx, vy, vz ) );
            colors.push_back( make_float3( r, g, b ) );
            radii.push_back( rd );

            // updates the bounding box with the bounding box of the current particle
            const float3 p_min = make_float3( x - rd, y - rd, z - rd );
            const float3 p_max = make_float3( x + rd, y + rd, z + rd );

            bbox_min = get_min( bbox_min, p_min );
            bbox_max = get_max( bbox_max, p_max );
        }

        newCacheEntry.bbox_min = bbox_min;
        newCacheEntry.bbox_max = bbox_max;
        cacheIt = dataCache.insert(std::make_pair(current_particle_frame, newCacheEntry)).first;
    }

    ParticleFrameData& cacheEntry = cacheIt->second;

    // all vectors have the same size
    const int num_particles = (int) cacheEntry.positions.size();

    geometry->setPrimitiveCount( num_particles );

    // fills up the buffers
    fillBuffers( cacheEntry.positions, cacheEntry.velocities, cacheEntry.colors, cacheEntry.radii );

    // the bounding box will actually be used only for the first frame
    aabb.set( cacheEntry.bbox_min, cacheEntry.bbox_max );

    // builds the BVH (or re-builds it if already existing)
    Acceleration accel = geometry_group->getAcceleration();
    accel->markDirty();
}


void setupParticles()
{
    // the buffers will be set to the right size at a later stage
    buffers.positions  = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0 );
    buffers.velocities = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0 );
    buffers.colors     = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0 );
    buffers.radii      = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT,  0 );

    geometry = context->createGeometry();
    geometry[ "positions_buffer"  ]->setBuffer( buffers.positions );
    geometry[ "velocities_buffer" ]->setBuffer( buffers.velocities );
    geometry[ "colors_buffer"     ]->setBuffer( buffers.colors );
    geometry[ "radii_buffer"      ]->setBuffer( buffers.radii );
    geometry[ "motion_blur"       ]->setFloat(  motion_blur );
    geometry->setBoundingBoxProgram  ( createBoundingBoxProgram( context ) );
    geometry->setIntersectionProgram ( createIntersectionProgram( context ) );

    if( motion_blur > 0.0f )
        geometry->setMotionSteps( 2 );

    Program closest_hit, any_hit;
    createMaterialPrograms( context, closest_hit, any_hit );

    std::vector<Material> optix_materials;
    optix_materials.push_back( createOptiXMaterial(
        context,
        closest_hit,
        any_hit ) );

    GeometryInstance geom_instance = context->createGeometryInstance(
        geometry,
        optix_materials.begin(),
        optix_materials.end() );

    geometry_group = context->createGeometryGroup();
    geometry_group->addChild( geom_instance );
    geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
    context[ "top_object"   ]->set( geometry_group );
    context[ "top_shadower" ]->set( geometry_group );
}

void setupCamera()
{
    const float max_dim = fmaxf( aabb.extent( 0 ), aabb.extent( 1 ) ); // max of x, y components

    camera_eye    = aabb.center() + make_float3( 0.0f, 0.0f, max_dim*1.85f );
    camera_lookat = aabb.center();
    camera_up     = make_float3( 0.0f, 1.0f, 0.0f );

    camera_rotate  = Matrix4x4::identity();
}


void setupLights()
{
    const float max_dim = fmaxf( aabb.extent( 0 ), aabb.extent (1 ) ); // max of x, y components

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
    light_buffer->setSize( sizeof( lights )/sizeof( lights[0] ) );
    memcpy( light_buffer->map(), lights, sizeof( lights ) );
    light_buffer->unmap();

    context[ "lights" ]->set( light_buffer );
}


void updateCamera()
{
    const float vfov = 35.0f;
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
    const Matrix4x4 trans     = frame*camera_rotate*camera_rotate*frame_inv;

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    context[ "eye" ]->setFloat( camera_eye );
    context[ "U"   ]->setFloat( camera_u );
    context[ "V"   ]->setFloat( camera_v );
    context[ "W"   ]->setFloat( camera_w );
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
    glutSpecialFunc( glutArrowPress );

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
    if(play && accumulation_frame >= iterations_per_animation_frame)
    {
        current_particle_frame++;
        if(current_particle_frame > max_particle_frames)
            current_particle_frame = 1;
        loadParticles();
        accumulation_frame = 0;
    }

    updateCamera();

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
        case ('p'):
        {
            play = !play;
            std::cout << (play ? "play animation" : "pause animation") << std::endl;
            break;
        }
        case ('m'):
        {
            iterations_per_animation_frame++;
            std::cout << "iterations per animation frame: " << iterations_per_animation_frame << std::endl;
            break;
        }
        case ('n'):
        {
            iterations_per_animation_frame--;
            if(iterations_per_animation_frame == 0)
                iterations_per_animation_frame = 1;
            std::cout << "iterations per animation frame: " << iterations_per_animation_frame << std::endl;
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


void glutMouseMotion( int x, int y )
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

        accumulation_frame = 0;
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

        accumulation_frame = 0;
    }

    mouse_prev_pos = make_int2( x, y );
}


void glutArrowPress( int k, int x, int y )
{
    // does nothing if single frame
    if( current_particle_frame == 0 )
        return;

    int prevFrame = current_particle_frame;

    switch( k )
    {
        case GLUT_KEY_LEFT:
        {
            current_particle_frame--;
            if( current_particle_frame==0 )
                current_particle_frame = max_particle_frames;
            break;
        }

        case GLUT_KEY_RIGHT:
        {
            current_particle_frame++;
            if( current_particle_frame > max_particle_frames )
                current_particle_frame = 1;
            break;
        }
    }

    if ( prevFrame != current_particle_frame ) {
        std::cerr << "Loading frame " << current_particle_frame << std::endl;
        loadParticles();
        accumulation_frame = 0;
    }
}

void glutResize( int w, int h )
{
    if( w == (int)width && h == (int)height ) return;

    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    sutil::resizeBuffer( getOutputBuffer(), width, height );

    Buffer accum_buffer = context[ "accum_buffer" ]->getBuffer();
    accum_buffer->setSize( width, height );

    accumulation_frame = 0;

    glViewport( 0, 0, width, height);

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
        "  -h | --help                         Print this usage message and exit.\n"
        "  -f | --file                         Save single frame to file and exit.\n"
        "  -n | --nopbo                        Disable GL interop for display buffer.\n"
        "  -p | --particles <particles_file>   Specify path to particles file to be loaded.\n"
        "  -s | --shade                        Shade the particles with a Lambert material.\n"
        "  -m | --motionblur [F]               Enables motion blur, with an optional extent.\n"
        "  -r | --report <LEVEL>               Enable usage reporting and report level [1-3].\n"
        "App Keystrokes:\n"
        "  p  Play/pause particle animation\n"
        "  m  Increase number of path tracing iterations per animation frame during animation\n"
        "  n  Decrease number of path tracing iterations per animation frame during animation\n"
        "  q  Quit\n"
        "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
        "  Left/right arrow keys to move in particles' animation\n"
        << std::endl;

    exit(1);
}

int main( int argc, char** argv )
 {
    std::string out_file;
    std::string particles_file = std::string( sutil::samplesDir() ) + "/data/particles/particles.0001.txt";
    int usage_report_level = 0;
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
        else if( arg == "-p" || arg == "--particles" )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            particles_file = argv[++i];
        }
        else if( arg == "-s" || arg == "--shade"  )
        {
            shade_particles = true;
        }
        else if( arg == "-m" || arg == "--motionblur"  )
        {
            motion_blur = 1.0f;
            if( i < argc-1 ) {
                // checks if there is an extra value for the extent of motion blur
                const std::string next_option( argv[i + 1] );
                if ( next_option[0] != '-' ) {
                    motion_blur = (float) atof( next_option.c_str() );
                    i++;
                }
            }
        }
        else if( arg == "-r" || arg == "--report" )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            usage_report_level = atoi( argv[++i] );
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

        UsageReportLogger logger;
        createContext( usage_report_level, &logger );
        setupParticles();
        setParticlesBaseName( particles_file );
        loadParticles();
        setupCamera();
        setupLights();

        context->validate();

        if( out_file.empty() )
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

