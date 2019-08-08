/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
// optixGeometryTriangles: Demonstrates the GeometryTriangles API.
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

const char* const SAMPLE_NAME = "optixGeometryTriangles";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context  context;
uint32_t width   = 768u;
uint32_t height  = 768u;
bool     use_pbo = true;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
bool           camera_dirty = true;  // Do camera params need to be copied to OptiX context
sutil::Arcball arcball;

// Mouse state
int2 mouse_prev_pos;
int  mouse_button;

// Materials
Material phong_matl;
Material tex_matl;
Material checker_matl;
Material normal_matl;
Material bary_matl;

// Geometry
GeometryInstance tri_gi;
GeometryInstance sphere_gi;
GeometryInstance plane_gi;


//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void createMaterials();
GeometryGroup createGeometry();
GeometryGroup createGeometryTriangles();
void setupScene();
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

struct Tetrahedron
{
    float3   vertices [12];
    float3   normals  [12];
    float2   texcoords[12];
    unsigned indices  [12];

    Tetrahedron( const float H, const float3 trans )
    {
        const float a = ( 3.0f * H ) / sqrtf( 6.0f ); // Side length
        const float d = a * sqrtf( 3.0f ) / 6.0f;     // Offset for base vertices from apex

        // There are only four vertex positions, but we will duplicate vertices
        // instead of sharing them among faces.
        const float3 v0 = trans + make_float3( 0.0f, 0, H - d );
        const float3 v1 = trans + make_float3( a / 2.0f, 0, -d );
        const float3 v2 = trans + make_float3( -a / 2.0f, 0, -d );
        const float3 v3 = trans + make_float3( 0.0f, H, 0.0f );

        // Bottom face
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;

        // Duplicate the face normals across the vertices.
        float3 n = optix::normalize( optix::cross( v2-v0, v1-v0 ) );
        normals[0] = n;
        normals[1] = n;
        normals[2] = n;

        texcoords[0] = make_float2( 0.5f, 1.0f );
        texcoords[1] = make_float2( 1.0f, 0.0f );
        texcoords[2] = make_float2( 0.0f, 0.0f );

        // Left face
        vertices[3] = v3;
        vertices[4] = v2;
        vertices[5] = v0;

        n = optix::normalize( optix::cross( v2-v3, v0-v3 ) );
        normals[3] = n;
        normals[4] = n;
        normals[5] = n;

        texcoords[3] = make_float2( 0.5f, 1.0f );
        texcoords[4] = make_float2( 0.0f, 0.0f );
        texcoords[5] = make_float2( 1.0f, 0.0f );

        // Right face
        vertices[6] = v3;
        vertices[7] = v0;
        vertices[8] = v1;

        n = optix::normalize( optix::cross( v0-v3, v1-v3 ) );
        normals[6] = n;
        normals[7] = n;
        normals[8] = n;

        texcoords[6] = make_float2( 0.5f, 1.0f );
        texcoords[7] = make_float2( 0.0f, 0.0f );
        texcoords[8] = make_float2( 1.0f, 0.0f );

        // Back face
        vertices[9]  = v3;
        vertices[10] = v1;
        vertices[11] = v2;

        n = optix::normalize( optix::cross( v1-v3, v2-v3 ) );
        normals[9]  = n;
        normals[10] = n;
        normals[11] = n;

        texcoords[9]  = make_float2( 0.5f, 1.0f );
        texcoords[10] = make_float2( 0.0f, 0.0f );
        texcoords[11] = make_float2( 1.0f, 0.0f );

        for( int i = 0; i < 12; ++i )
            indices[i]  = i;
    }
};


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

    // Accumulation buffer.
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


void createMaterials()
{
    // Normal shader material
    const char* ptx    = sutil::getPtxString( SAMPLE_NAME, "normal_shader.cu" );
    Program     tri_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program     tri_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );

    normal_matl = context->createMaterial();
    normal_matl->setClosestHitProgram( 0, tri_ch );
    normal_matl->setAnyHitProgram( 1, tri_ah );

    // Metal material
    ptx              = sutil::getPtxString( SAMPLE_NAME, "phong.cu" );
    Program phong_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program phong_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );

    phong_matl = context->createMaterial();
    phong_matl->setClosestHitProgram( 0, phong_ch );
    phong_matl->setAnyHitProgram( 1, phong_ah );
    phong_matl["Ka"]->setFloat( 0.2f, 0.5f, 0.5f );
    phong_matl["Kd"]->setFloat( 0.2f, 0.7f, 0.8f );
    phong_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
    phong_matl["phong_exp"]->setFloat( 64 );
    phong_matl["Kr"]->setFloat( 0.5f, 0.5f, 0.5f );

    // Texture material
    ptx            = sutil::getPtxString( SAMPLE_NAME, "phong.cu" );
    Program tex_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance_textured" );
    Program tex_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );

    tex_matl = context->createMaterial();
    tex_matl->setClosestHitProgram( 0, tex_ch );
    tex_matl->setAnyHitProgram( 1, tex_ah );

    tex_matl["Kd_map"]->setTextureSampler(
        sutil::loadTexture( context, std::string( sutil::samplesDir() ) + "/data/nvidia_logo.ppm",
                            optix::make_float3(0.0f, 0.0f, 1.0f) ) );

    // Checker material
    ptx              = sutil::getPtxString( SAMPLE_NAME, "checker.cu" );
    Program check_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program check_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );

    checker_matl = context->createMaterial();
    checker_matl->setClosestHitProgram( 0, check_ch );
    checker_matl->setAnyHitProgram( 1, check_ah );

    checker_matl["Kd1"]->setFloat( 0.5f, 0.9f, 0.4f );
    checker_matl["Ka1"]->setFloat( 0.5f, 0.9f, 0.4f );
    checker_matl["Ks1"]->setFloat( 0.0f, 0.0f, 0.0f );
    checker_matl["Kd2"]->setFloat( 0.9f, 0.4f, 0.2f );
    checker_matl["Ka2"]->setFloat( 0.9f, 0.4f, 0.2f );
    checker_matl["Ks2"]->setFloat( 0.0f, 0.0f, 0.0f );
    checker_matl["inv_checker_size"]->setFloat( 16.0f, 16.0f, 1.0f );
    checker_matl["phong_exp1"]->setFloat( 0.0f );
    checker_matl["phong_exp2"]->setFloat( 0.0f );
    checker_matl["Kr1"]->setFloat( 0.0f, 0.0f, 0.0f );
    checker_matl["Kr2"]->setFloat( 0.0f, 0.0f, 0.0f );

    // Barycentric material
    ptx             = sutil::getPtxString( SAMPLE_NAME, "optixGeometryTriangles.cu" );
    Program bary_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program bary_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );

    bary_matl = context->createMaterial();
    bary_matl->setClosestHitProgram( 0, bary_ch );
    bary_matl->setAnyHitProgram( 1, bary_ah );
}


GeometryGroup createGeometryTriangles()
{
    // Create a tetrahedron using four triangular faces.  First We will create
    // vertex and index buffers for the faces, and then create a
    // GeometryTriangles object.
    const unsigned num_faces    = 4;
    const unsigned num_vertices = num_faces * 3;

    // Define a regular tetrahedron of height 2, translated 1.5 units from the origin.
    Tetrahedron tet( 2.0f, make_float3( 1.5f, 0.0f, 0.0f ) );

    // Create Buffers for the triangle vertices, normals, texture coordinates, and indices.
    Buffer vertex_buffer   = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices );
    Buffer normal_buffer   = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices );
    Buffer texcoord_buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, num_vertices );
    Buffer index_buffer    = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, num_faces );

    // Copy the tetrahedron geometry into the device Buffers.
    memcpy( vertex_buffer->map(), tet.vertices, sizeof( tet.vertices ) );
    memcpy( normal_buffer->map(), tet.normals, sizeof( tet.normals ) );
    memcpy( texcoord_buffer->map(), tet.texcoords, sizeof( tet.texcoords ) );
    memcpy( index_buffer->map(), tet.indices, sizeof( tet.indices ) );

    vertex_buffer->unmap();
    normal_buffer->unmap();
    texcoord_buffer->unmap();
    index_buffer->unmap();

    // Create a GeometryTriangles object.
    optix::GeometryTriangles geom_tri = context->createGeometryTriangles();

    geom_tri->setPrimitiveCount( num_faces );
    geom_tri->setTriangleIndices( index_buffer, RT_FORMAT_UNSIGNED_INT3 );
    geom_tri->setVertices( num_vertices, vertex_buffer, RT_FORMAT_FLOAT3 );
    geom_tri->setBuildFlags( RTgeometrybuildflags( 0 ) );

    // Set an attribute program for the GeometryTriangles, which will compute
    // things like normals and texture coordinates based on the barycentric
    // coordindates of the intersection.
    const char* ptx = sutil::getPtxString( SAMPLE_NAME, "optixGeometryTriangles.cu" );
    geom_tri->setAttributeProgram( context->createProgramFromPTXString( ptx, "triangle_attributes" ) );

    geom_tri["index_buffer"]->setBuffer( index_buffer );
    geom_tri["vertex_buffer"]->setBuffer( vertex_buffer );
    geom_tri["normal_buffer"]->setBuffer( normal_buffer );
    geom_tri["texcoord_buffer"]->setBuffer( texcoord_buffer );

    // Bind a Material to the GeometryTriangles.  Materials can be shared
    // between GeometryTriangles objects and other Geometry types, as long as
    // all of the attributes needed by the attached hit programs are produced in
    // the attribute program.
    tri_gi = context->createGeometryInstance( geom_tri, phong_matl );

    GeometryGroup tri_gg = context->createGeometryGroup();
    tri_gg->addChild( tri_gi );
    tri_gg->setAcceleration( context->createAcceleration( "Trbvh" ) );

    return tri_gg;
}


GeometryGroup createGeometry()
{
    // Sphere geometry
    Geometry sphere = context->createGeometry();
    sphere->setPrimitiveCount( 1u );

    const char* ptx = sutil::getPtxString( SAMPLE_NAME, "sphere.cu" );
    sphere->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    sphere->setIntersectionProgram( context->createProgramFromPTXString( ptx, "robust_intersect" ) );
    sphere["sphere"]->setFloat( -1.5f, 1.0f, 0.0f, 1.0f );

    // Parallelogram geometry for ground plane.
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );

    ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
    parallelogram->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    parallelogram->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );

    float3 anchor = make_float3( -8.0f, -1e-3f, -8.0f );
    float3 v1     = make_float3( 16.0f, 0.0f, 0.0f );
    float3 v2     = make_float3( 0.0f, 0.0f, 16.0f );
    float3 normal = normalize( cross( v1, v2 ) );

    float d = dot( normal, anchor );
    v1 *= 1.0f / dot( v1, v1 );
    v2 *= 1.0f / dot( v2, v2 );

    float4 plane = make_float4( normal, d );
    parallelogram["plane"]->setFloat( plane );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );
    parallelogram["anchor"]->setFloat( anchor );

    // Greate GIs to bind Materials to the Geometry objects.
    sphere_gi = context->createGeometryInstance( sphere, &phong_matl, &phong_matl + 1 );
    plane_gi  = context->createGeometryInstance( parallelogram, &checker_matl, &checker_matl + 1 );

    // Create a GeometryGroup for the non-GeometryTriangles objects.
    GeometryGroup gg = context->createGeometryGroup();
    gg->addChild( sphere_gi );
    gg->addChild( plane_gi );
    gg->setAcceleration( context->createAcceleration( "Trbvh" ) );

    return gg;
}


void setupScene()
{
    // Create a GeometryGroup for the GeometryTriangles instances and a separate
    // GeometryGroup for all other primitives.
    GeometryGroup tri_gg = createGeometryTriangles();
    GeometryGroup gg     = createGeometry();

    // Create a top-level Group to contain the two GeometryGroups.
    Group top_group = context->createGroup();
    top_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
    top_group->addChild( gg );
    top_group->addChild( tri_gg );

    context["top_object"]->set( top_group );
    context["top_shadower"]->set( top_group );
}


void setupCamera()
{
    camera_eye    = make_float3( -2.0f, 4.0f, 10.0f );
    camera_lookat = make_float3( 0.0f, 1.0f, 0.0f );
    camera_up     = make_float3( 0.0f, 1.0f, 0.0f );
    camera_rotate = Matrix4x4::identity();
    camera_dirty  = true;
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
    const float vfov  = 30.0f;
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
        case( 'm' ):
        {
            // Cycle through a few different materials.
            static int count = 0;
            if( ++count == 1 )
                tri_gi->setMaterial( 0, normal_matl );
            else if( count == 2 )
                tri_gi->setMaterial( 0, bary_matl );
            else if( count == 3 )
                tri_gi->setMaterial( 0, tex_matl );
            else if( count == 4 )
                tri_gi->setMaterial( 0, checker_matl );
            else
                tri_gi->setMaterial( 0, phong_matl );

            count %= 5;
            camera_dirty = true;
            break;
        }
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
        "  m  Toggle the material on the GeometryTriangles object.\n"
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
        createMaterials();
        setupScene();
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

