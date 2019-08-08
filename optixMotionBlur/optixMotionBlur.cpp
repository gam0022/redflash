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
// optixMotionBlur: simple demonstration of motion blur API 
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
#include <optixu/optixu_quaternion_namespace.h>

#include <sutil.h>
#include "common.h"
#include <Arcball.h>
#include <OptiXMesh.h>

#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdint.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixMotionBlur";
// globals
bool do_timeview = false;

enum MotionType {
  MOTION_TYPE_VERTEX,
  MOTION_TYPE_MATRIX,
  MOTION_TYPE_SRT
};

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context        context;
uint32_t       width  = 1024u;
uint32_t       height = 768u;
bool           use_pbo = true;
bool           use_tri_api = false;
optix::Aabb    aabb;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
sutil::Arcball arcball;
unsigned int   accumulation_frame = 0u;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;


//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

std::string ptxPath( const std::string& cuda_file );
Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext( int usage_report_level );
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

std::string ptxPath( const std::string& cuda_file )
{
    return
        std::string(sutil::samplesPTXDir()) +
        "/" + std::string(SAMPLE_NAME) + "_generated_" +
        cuda_file +
        ".ptx";
}


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

void printToStdout( int lvl, const char* tag, const char* msg, void* /*cbdata*/ )
{
    std::cout << "[" << lvl << "][" << std::left << std::setw( 12 ) << tag << "] " << msg;
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


void createContext( int usage_report_level )
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 2 );   // raygen, timeview
    if( usage_report_level > 0 )
    {
        context->setUsageReportCallback( printToStdout, usage_report_level, NULL );
    }

    context["scene_epsilon"    ]->setFloat( 1.e-4f );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo );
    context["output_buffer"]->set( buffer );
    
    Buffer accum_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT4, width, height );
    context["accum_buffer"]->set( accum_buffer );

    // Timeview buffers
    { 
      Buffer buf = context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, 2 );
      context["timeview_min_max"]->set( buf );
      context["do_timeview"]->setInt( int(do_timeview) );

      
      std::string colormap_file = std::string( sutil::samplesDir() ) + "/data/colormap.ppm";
      optix::TextureSampler sampler = sutil::loadTexture( context, colormap_file, optix::make_float3(1.0f, 1.0f, 1.0f) );
      sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
      context["colormap_sampler"]->set( sampler );
    }

    // Ray generation program
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "accum_camera_mblur.cu" );
    Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
    context["time_range"]->setFloat( 0.0f, 1.0f );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Timeview program
    Program timeview_program = context->createProgramFromPTXString( ptx, "colormap" );
    context->setRayGenerationProgram( 1, timeview_program );

    // Miss program
    ptx = sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" );
    context->setMissProgram( 0, context->createProgramFromPTXString( ptx, "miss" ) );
    context["bg_color"]->setFloat( 0.1f, 0.1f, 0.1f );
}

// Build a matrix by lerping S,R,T components separately
static optix::Matrix4x4 lerp_srt( const float3& scale, const float4& rotation, const float3& translation, float t )
{
    const float3 rotation_axis = normalize(make_float3( rotation.x, rotation.y, rotation.z ));
    const float rotation_angle = rotation.w * M_PIf / 180.0f;

    optix::Matrix4x4 xform = optix::Matrix4x4::translate( t*translation ) *
                             optix::Matrix4x4::rotate( t*rotation_angle, rotation_axis ) *
                             optix::Matrix4x4::scale( (1.0f-t) + t*scale );
    return xform;
}

// Build an interpolated key by lerping S,R,T components separately.
// Use quaternions for rotation. Set rotation pivot in key.
static void lerp_srt_to_key( float t_key[], 
                             const float3& scale, 
                             const float4& rotation, const float3& pivot,
                             const float3& translation, 
                             float t )
{
    // scale
    t_key[0] = (1.f - t) + t * scale.x;
    t_key[1] = t_key[2] = t_key[3] = 0.f;
    t_key[4] = (1.f - t) + t * scale.y;
    t_key[5] = t_key[6] = 0.f;
    t_key[7] = (1.f - t) + t * scale.z;
    t_key[8] = 0.f;

    // pivot
    t_key[3] = pivot.x;
    t_key[6] = pivot.y;
    t_key[8] = pivot.z;

    // rotation

    // use nlerp for interpolation between id = (0,0,0,1) and q.
    Quaternion q( make_float3(rotation.x, rotation.y, rotation.z), rotation.w );
    if (fabsf(t - 1.f) > 1.e-6f) {
        Quaternion id;
        q = nlerp( id, q, t );
    }
    t_key[9]  = q.m_q.x;
    t_key[10] = q.m_q.y;
    t_key[11] = q.m_q.z;
    t_key[12] = q.m_q.w;

    // translate
    t_key[13] = t * translation.x;
    t_key[14] = t * translation.y;
    t_key[15] = t * translation.z;
}

static float3 eval_srt_key(float key[], float3 p)
{
    // scale, TBD shear
    float3 ps = make_float3( key[0] * p.x, key[4] * p.y, key[7] * p.z );

    // pivot point for rotation, before rotation translate by -pivot
    const float3 pivot = make_float3( key[3], key[6], key[8] );
    ps = ps - pivot;

    // rotation as quaternion
    const optix::Quaternion q(key[9], key[10], key[11], key[12]);
    float3 psr = q * ps;

    // translate back by pivot
    psr = psr + pivot;

    // lerp translate
    const float3 translate = make_float3(key[13], key[14], key[15]);

    float3 psrt = translate + psr;

    return psrt;
}

static void srt_identity( float *key, const float3& pivot  )
{
    // set identity key for SRT
    // S
    key[ 0] = 1.f; 
    key[ 1] = 0.f; 
    key[ 2] = 0.f; 
    key[ 3] = pivot.x;
    key[ 4] = 1.f; 
    key[ 5] = 0.f; 
    key[ 6] = pivot.y;
    key[ 7] = 1.f; 
    key[ 8] = pivot.z;
    // R (quaternion representation)
    key[ 9] = 0.f; 
    key[10] = 0.f; 
    key[11] = 0.f; 
    key[12] = 1.f;
    // T
    key[13] = 0.f; 
    key[14] = 0.f; 
    key[15] = 0.f;
}

void createGeometry( optix::Program closest_hit,
                     optix::Program any_hit, 
                     const std::string& builder, 
                     const std::string& mesh_filename, 
                     const MotionType motion_type,
                     const bool do_ground,
                     int motion_steps, 
                     float2 motion_range, 
                     RTmotionbordermode motion_border_modes[2],
                     const float3& scale, 
                     const float4& rotation, 
                     const float3& pivot,
                     const float3& translation )
                     
{
    // Mesh geometry
    OptiXMesh mesh;
    mesh.context = context;
    mesh.closest_hit = closest_hit;
    mesh.any_hit = any_hit;
    mesh.use_tri_api = use_tri_api;

    if( motion_type == MOTION_TYPE_VERTEX )
    {
        // vertex blur enabled
        const char* ptx = sutil::getPtxString( SAMPLE_NAME, "triangle_mesh_mblur.cu" );
        mesh.intersection = context->createProgramFromPTXString( ptx,"intersect" );
        mesh.intersection[ "motion_range" ]->setFloat( motion_range.x, motion_range.y );
        mesh.bounds       = context->createProgramFromPTXString( ptx,"bounds" );
    }
    loadMesh( mesh_filename, mesh ); 

    aabb.set( mesh.bbox_min, mesh.bbox_max );

    const float maxExtent = aabb.maxExtent();
    
    if( motion_type == MOTION_TYPE_VERTEX )
    {
        GeometryInstance gi = mesh.geom_instance;
        Buffer v0 = gi[ "vertex_buffer" ]->getBuffer();
        Buffer n0 = gi[ "normal_buffer" ]->getBuffer();
        gi->removeVariable( gi[ "vertex_buffer" ] );
        gi->removeVariable( gi[ "normal_buffer" ] );

        RTsize num_verts, num_norms;
        v0->getSize( num_verts );
        n0->getSize( num_norms );
        assert( num_verts == num_norms || num_norms == 0 );

        Buffer vbs = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_BUFFER_ID, motion_steps );
        gi[ "vertex_buffers" ]->set( vbs );
        int* vbs_data = reinterpret_cast<int*>( vbs->map() );
        vbs_data[0] = v0->getId();

        // Transform vertex positions at each motion step
        const float3* v0_data = reinterpret_cast<const float3*>( v0->map() );
        for ( int motion_index = 1; motion_index < motion_steps; motion_index++ )
        {
            const float t = float(motion_index)/(motion_steps-1);
            const optix::Matrix4x4 xform = lerp_srt( scale, rotation, maxExtent*translation, t );

            Buffer verts = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_verts );
            float3* verts_data = reinterpret_cast<float3*>( verts->map() );
            for( size_t i = 0; i < num_verts; ++i ) {
                verts_data[i] = make_float3( xform*make_float4( v0_data[i], 1.0f ) );
                aabb.include(verts_data[i]);
            }
            verts->unmap();
            vbs_data[motion_index] = verts->getId();
        }
        v0->unmap();
        vbs->unmap();

        // Transform normals
        {
            Buffer nbs = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_BUFFER_ID, 0 );
            gi[ "normal_buffers" ]->set( nbs );
            if( num_norms != 0 )
            {
                nbs->setSize( motion_steps );
                int* nbs_data = reinterpret_cast<int*>( nbs->map() );
                const float3* n0_data = reinterpret_cast<const float3*>( n0->map() );
                nbs_data[0] = n0->getId();

                for ( int motion_index = 0; motion_index < motion_steps; ++motion_index )
                {
                    const float t = float(motion_index)/(motion_steps-1);
                    const optix::Matrix4x4 xform = lerp_srt( scale, rotation, maxExtent*translation, t );
                    optix::Matrix4x4 xform_inv_trans = xform.inverse().transpose();

                    Buffer normals = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_norms );
                    float3* normals_data = reinterpret_cast<float3*>( normals->map() );
                    for( size_t i = 0; i < num_norms; ++i )
                        normals_data[i] = make_float3( xform_inv_trans*make_float4( n0_data[i], 0.0f ) );
                    normals->unmap();

                    nbs_data[motion_index] = normals->getId();
                }
                
                n0->unmap();
                nbs->unmap();
            }

        }

        if( use_tri_api )
        {
            GeometryTriangles geom_tri = mesh.geom_instance->getGeometryTriangles();
            geom_tri->setMotionSteps( motion_steps ); 
            geom_tri->setMotionRange( motion_range.x, motion_range.y );
            geom_tri->setMotionBorderMode( motion_border_modes[0], motion_border_modes[1] );
        }
        else
        {
            Geometry geom = mesh.geom_instance->getGeometry();
            geom->setMotionSteps( motion_steps ); 
            geom->setMotionRange( motion_range.x, motion_range.y );
            geom->setMotionBorderMode( motion_border_modes[0], motion_border_modes[1] );
        }
    }

    // Ground geometry.
    GeometryInstance ground_gi;
    if ( do_ground )
    {
        Geometry ground_geom = context->createGeometry();
        ground_geom->setPrimitiveCount( 1u );
        const char* pgram_ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
        ground_geom->setBoundingBoxProgram( context->createProgramFromPTXString( pgram_ptx, "bounds" ) );
        ground_geom->setIntersectionProgram( context->createProgramFromPTXString( pgram_ptx,"intersect" ) );

        const float  extent = aabb.maxExtent();
        const float3 center = aabb.center();
        float3 anchor = make_float3( center.x - 2.0f*extent, aabb.m_min.y, center.z + 2.0f*extent);
        float3 v1 = make_float3( 4.0f*extent, 0.0f, 0.0f);
        float3 v2 = make_float3( 0.0f, 0.0f, -4.0f*extent);
        float3 normal = cross( v1, v2 );
        normal = normalize( normal );
        float d = dot( normal, anchor );
        v1 *= 1.0f/dot( v1, v1 );
        v2 *= 1.0f/dot( v2, v2 );
        float4 plane = make_float4( normal, d );
        ground_geom["plane"]->setFloat( plane.x, plane.y, plane.z, plane.w );
        ground_geom["v1"]->setFloat( v1.x, v1.y, v1.z );
        ground_geom["v2"]->setFloat( v2.x, v2.y, v2.z );
        ground_geom["anchor"]->setFloat( anchor.x, anchor.y, anchor.z );

        Material ground_matl = context->createMaterial();
        ground_matl->setClosestHitProgram( 0, closest_hit );
        ground_matl->setAnyHitProgram( 1, any_hit );

        ground_gi = context->createGeometryInstance( ground_geom, &ground_matl, &ground_matl+1 );
    }

    if ( motion_type != MOTION_TYPE_VERTEX )
    {
        Transform mxform = context->createTransform();
        mxform->setMotionRange( motion_range.x, motion_range.y );
        mxform->setMotionBorderMode( motion_border_modes[0], motion_border_modes[1] );

        const bool use_srt = motion_type == MOTION_TYPE_SRT;
        const int ksize = use_srt ? 16 : 12;
        std::vector<float> keys( ksize * motion_steps );
        optix::Matrix4x4 x0 = optix::Matrix4x4::identity();
        if ( use_srt )
            srt_identity( x0.getData(), pivot );
        memcpy( &keys[0], x0.getData(), ksize*sizeof(float) );

        // Map mesh vertices to host, for computing bounding box below
        GeometryInstance geom = mesh.geom_instance;
        Buffer v0 = geom[ "vertex_buffer" ]->getBuffer();
        const float3* v0_data = reinterpret_cast<const float3*>( v0->map() );
        RTsize num_verts;
        v0->getSize( num_verts );

        if ( !use_srt )
        {
            for ( int motion_index = 1; motion_index < motion_steps; ++motion_index )
            {
                const float t = float(motion_index)/(motion_steps - 1);

                const optix::Matrix4x4 xform = lerp_srt( scale, rotation, maxExtent*translation, t );
                memcpy( &keys[0] + 12 * (motion_index), xform.getData(), 12 * sizeof(float) );

                // Include transformed vertices in bounding box
                for ( size_t i = 0; i < num_verts; ++i ) {
                    float3 v = make_float3( xform*make_float4( v0_data[i], 1.0f ) );
                    aabb.include( v );
                }
            }
        }
        else
        {
            for ( int motion_index = 1; motion_index < motion_steps; ++motion_index)
            {
                const float t = float(motion_index)/(motion_steps - 1);

                float t_key[16];
                lerp_srt_to_key( t_key, scale, rotation, pivot, maxExtent*translation, t );
                memcpy( &keys[0] + 16 * (motion_index), t_key, 16 * sizeof(float) );

                // Include transformed vertices in bounding box
                for ( size_t i = 0; i < num_verts; ++i ) {
                    float3 v = eval_srt_key( t_key, v0_data[i] );
                    aabb.include( v );
                }
            }
        }

        v0->unmap();

        mxform->setMotionKeys( motion_steps, use_srt ? RT_MOTIONKEYTYPE_SRT_FLOAT16 : RT_MOTIONKEYTYPE_MATRIX_FLOAT12, &keys[0] );
        
        // Geometry groups
        Group top_group = context->createGroup();
        top_group->setAcceleration( context->createAcceleration( builder ) );

        GeometryGroup geometry_group0 = context->createGeometryGroup();
        geometry_group0->addChild( mesh.geom_instance );
        geometry_group0->setAcceleration( context->createAcceleration( builder ) );
        mxform->setChild( geometry_group0 );
        top_group->addChild( mxform );

        if ( do_ground )
        {
            GeometryGroup geometry_group1 = context->createGeometryGroup();
            geometry_group1->addChild( ground_gi );
            geometry_group1->setAcceleration( context->createAcceleration( "NoAccel" ) );
            top_group->addChild( geometry_group1 );
        }

        context[ "top_object"   ]->set( top_group ); 
        context[ "top_shadower" ]->set( top_group ); 
    }
    else
    {
        // Geometry group
        GeometryGroup geometry_group = context->createGeometryGroup();
        geometry_group->addChild( mesh.geom_instance );
        if ( do_ground ) geometry_group->addChild( ground_gi );
        geometry_group->setAcceleration( context->createAcceleration( builder ) );

        // Insert a top group, to test the bounding box of the lower-level accel
        Group top_group = context->createGroup();

        top_group->setAcceleration( context->createAcceleration( builder ) );
        top_group->addChild( geometry_group );

        context[ "top_object"   ]->set( top_group ); 
        context[ "top_shadower" ]->set( top_group ); 
    }

}

  
void setupCamera()
{
    const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components

    camera_eye    = aabb.center() + make_float3( 0.0f, 0.0f, max_dim*1.5f ); 
    camera_lookat = aabb.center(); 
    camera_up     = make_float3( 0.0f, 1.0f, 0.0f );

    camera_rotate  = Matrix4x4::identity();
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
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
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

    if ( do_timeview )
    {
      // Clear timeview buffer
      Buffer timeview_min_max = context["timeview_min_max"]->getBuffer();
      float* data = (float*)timeview_min_max->map();  // reinterpret as positive float.
      data[0] = std::numeric_limits<float>::max();
      data[1] = 0.0f;
      timeview_min_max->unmap();
    }
    
    context["frame"]->setUint( accumulation_frame++ );
    context->launch( 0, width, height );

    // colormap
    if ( do_timeview )
      context->launch( 1, width, height );

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
        case( 't' ):
        {
            do_timeview = !do_timeview;
            context["do_timeview"]->setInt( int(do_timeview) );
            accumulation_frame = 0;
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


void glutResize( int w, int h )
{
    if ( w == (int)width && h == (int)height ) return;

    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    sutil::resizeBuffer( context[ "accum_buffer" ]->getBuffer(), width, height );

    accumulation_frame = 0;
    
    glViewport(0, 0, width, height);                                               

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( )
{
    std::cerr << "\nUsage: " << SAMPLE_NAME << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help                         Print this usage message and exit.\n"
        "  -b | --build <name>                 Acceleration structure builder.  Default Trbvh.\n"
        "  -f | --file                         Save single frame to file and exit.\n"
        "  -g | --ground                       Toggle ground plane (default off).\n"
        "  -l | --lights                       Toggle lighting with shadows (default off).\n"
        "  -m | --mesh <mesh_file>             Specify path to mesh to be loaded.\n"
        "  -n | --nopbo                        Disable GL interop for display buffer.\n"
        "  -r | --report <LEVEL>               Enable usage reporting and report level [1-3].\n"
        "  -t | --motion-type   <type>         Type of motion: \"vertex\", \"matrix\", or \"srt\".  Default \"matrix\".\n"
        "       --scale         <sx sy sz>     Scale. Default [1 1 1]\n"
        "       --rotate        <rx ry rz rw>  Rotation as axis and angle (in degrees). Default no rotation.\n"
        "       --pivot         <px py pz>     Rotation about this pivot point. Default origin.\n"
        "       --translate     <tx ty tz>     Translation in units of scene bounding box max extent.  Default [0.1 0 0]\n"
        "       --motion-steps  <n>            Number of motion steps for vertex positions or transforms.  Default 2\n"
        "       --motion-range  <t0 t1>        Motion range for geometry or transforms.  Default [0 1], which is the shutter range.\n"
        "       --motion-border <mode mode>    Motion border modes (\"clamp\" or \"vanish\") for geometry or transforms.  Defaults to \"clamp clamp\".\n"
        "       --triangle-api                 Enable the Triangle API to use built-in triangle intersection (default off).\n"
        "App Keystrokes:\n"
        "  q  Quit\n" 
        "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
        "  t  Toggle timeview mode\n"
        << std::endl;

    exit(1);
}

void checkargs( int argc, const std::string& arg, int index, int numargs=1 )
{
    if( index + numargs >= argc )
    {
        if ( numargs == 1 )
        {
            std::cerr << "Option '" << arg << "' requires additional argument.\n";
        } else {
            std::cerr << "Option '" << arg << "' requires " << numargs << " additional arguments.\n";
        }
        printUsageAndExit();
    }
}

float3 readfloat3( char** argv, int index )
{
    float3 v = make_float3(0.0f);
    if ( sscanf( argv[index+1], "%f", &v.x ) == 1 && 
         sscanf( argv[index+2], "%f", &v.y ) == 1 &&
         sscanf( argv[index+3], "%f", &v.z ) == 1 )
        return v;

    std::cerr << "Could not parse 3 float arguments for " << argv[index] << "\n";
    printUsageAndExit();
    return v;
}

float4 readfloat4( char** argv, int index )
{
    float4 v = make_float4(0.0f);
    if ( sscanf( argv[index+1], "%f", &v.x ) != 1 ||
         sscanf( argv[index+2], "%f", &v.y ) != 1 ||
         sscanf( argv[index+3], "%f", &v.z ) != 1 ||
         sscanf( argv[index+4], "%f", &v.w ) != 1 )
    {
        std::cerr << "Could not parse 4 float arguments for " << argv[index] << "\n";
        printUsageAndExit();
    }
    return v;
}

RTmotionbordermode parseBorderMode( const std::string& arg )
{
    RTmotionbordermode mode = RT_MOTIONBORDERMODE_CLAMP;
    if ( arg == "vanish" )
        mode = RT_MOTIONBORDERMODE_VANISH;
    else if ( arg == "clamp" )
        mode = RT_MOTIONBORDERMODE_CLAMP;
    else
    {
        std::cerr << "Could not parse motion border mode: " << arg << ", must be \'clamp\' or \'vanish\'\n";
        printUsageAndExit();
    }
    return mode;
}

MotionType parseMotionType( const std::string& arg )
{
    MotionType motion_type = MOTION_TYPE_MATRIX;
    if ( arg == "vertex" )
        motion_type = MOTION_TYPE_VERTEX;
    else if ( arg == "matrix" )
        motion_type = MOTION_TYPE_MATRIX;
    else if ( arg == "srt" )
        motion_type = MOTION_TYPE_SRT;
    else {
        std::cerr << "Could not parse motion type: " << arg << ", must be \'vertex\', \'matrix\', or \'srt\'\n";
        printUsageAndExit();
    }
    return motion_type;
}

int main( int argc, char** argv )
 {
    std::string builder = "Trbvh";
    std::string out_file;
    std::string mesh_file = std::string( sutil::samplesDir() ) + "/data/cow.obj";
    int usage_report_level = 0;
    MotionType motion_type = MOTION_TYPE_MATRIX;
    bool do_ground = false;
    bool do_lights = false;
    int motion_steps = 2;
    float2 motion_range = make_float2( 0.0f, 1.0f );
    RTmotionbordermode motion_border_modes[] = { RT_MOTIONBORDERMODE_CLAMP, RT_MOTIONBORDERMODE_CLAMP };
    float3 translation = make_float3( 0.0f );
    float3 pivot = make_float3(0.0f);
    float3 scale = make_float3( 1.0f );
    float4 rotation = make_float4( 1.0f, 0.0f, 0.0f, 0.0f );  // axis,angle
    bool have_animation = false;
    for( int i=1; i<argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit();
        }
        else if( arg == "-b" || arg == "--build"  )
        {
            checkargs( argc, arg, i );
            builder = argv[++i];
        }
        else if( arg == "-f" || arg == "--file"  )
        {
            checkargs( argc, arg, i );
            out_file = argv[++i];
        }
        else if( arg == "-g" || arg == "--ground"  )
        {
            do_ground = true;
        }
        else if( arg == "-l" || arg == "--lights"  )
        {
            do_lights = true;
        }
        else if( arg == "-n" || arg == "--nopbo"  )
        {
            use_pbo = false;
        }
        else if( arg == "-m" || arg == "--mesh" )
        {
            checkargs( argc, arg, i );
            mesh_file = argv[++i];
        }
        else if( arg == "-r" || arg == "--report" )
        {
            checkargs( argc, arg, i );
            usage_report_level = atoi( argv[++i] );
        }
        else if( arg == "-t" || arg == "--motion-type" )
        {
            checkargs( argc, arg, i );
            motion_type = parseMotionType( argv[++i] );
        }
        else if( arg == "--scale" )
        {
            checkargs( argc, arg, i, 3 );
            scale = readfloat3( argv, i );
            have_animation = true;
            i += 3;
        }
        else if( arg == "--rotate" )
        {
            checkargs( argc, arg, i, 4 );
            rotation = readfloat4( argv, i );  // axis, angle
            have_animation = true;
            i += 4;
        }
        else if( arg == "--translate" )
        {
            checkargs( argc, arg, i, 3 );
            translation = readfloat3( argv, i );
            have_animation = true;
            i += 3;
        }
        else if( arg == "--motion-steps" )
        {
            checkargs( argc, arg, i, 1 );
            motion_steps = atoi(argv[++i]);
            if ( motion_steps < 2 )
            {
                std::cerr << "motion-steps must be >= 2." << "\n";
                printUsageAndExit();
            }
        }
        else if( arg == "--motion-range" )
        {
            checkargs( argc, arg, i, 2 );
            motion_range.x = float(atof(argv[++i]));
            motion_range.y = float(atof(argv[++i]));
            if ( motion_range.y < motion_range.x )
            {
                std::cerr << "Invalid motion range: " << motion_range.x << " " << motion_range.y << "\n";
                printUsageAndExit();
            }
        }
        else if( arg == "--motion-border" )
        {
            checkargs( argc, arg, i, 2 );
            motion_border_modes[0] = parseBorderMode( argv[i+1] );
            motion_border_modes[1] = parseBorderMode( argv[i+2] );
            i += 2;
        }
        else if (arg == "--pivot")
        {
            checkargs(argc, arg, i, 3);
            pivot = readfloat3(argv, i);
            i += 3;
        }
        else if (arg == "--triangle-api")
        {
            use_tri_api = true;
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit();
        }
    }

    // Sanity checks
    if ( !have_animation )
    {
        std::cerr << "No animation specified; using default translation of (0.1, 0, 0) times scene scale\n";
        translation = make_float3( 0.1f, 0.0f, 0.0f );
    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        createContext( usage_report_level );

        const char* material_ptx = sutil::getPtxString( SAMPLE_NAME, do_lights ? "normal_shader_with_shadows.cu" : "normal_shader.cu" );
        optix::Program closest_hit = context->createProgramFromPTXString( material_ptx, "closest_hit_radiance" );
        optix::Program any_hit = context->createProgramFromPTXString( material_ptx, "any_hit_shadow" );

        createGeometry( closest_hit, any_hit, builder, mesh_file, motion_type, do_ground, motion_steps, motion_range, motion_border_modes, 
                        scale, rotation, pivot, translation );
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
            for (unsigned i = 0; i < 100; ++i )
            {
              context["frame"]->setUint( i );
              context->launch( 0, width, height );
            }
            sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
            destroyContext();
        }
        return 0;
    }
    SUTIL_CATCH( context->get() )
}

