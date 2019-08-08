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
// optixDynamicGeometry: compares accel structures for a dynamic scene.
// A multi level BVH allows faster updates at the cost of slower ray
// intersections vs. a single level BVH.
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
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include <sutil.h>
#include "common.h"
#include "random.h"
#include <Arcball.h>
#include <OptiXMesh.h>

#include <cassert>
#include <string>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixDynamicGeometry";
const float MOVE_ANIMATION_TIME = 2.0f;

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context        context;
uint32_t       width  = 1024u;
uint32_t       height = 1024u;
bool           use_pbo = true;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

// Accel layout
class DynamicLayout;
DynamicLayout* layout = NULL;


//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void setupCamera( const optix::Aabb& aabb );
void setupLights( const optix::Aabb& aabb );
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
// Layout classes 
//
//------------------------------------------------------------------------------

float3 getRandomPosition( int child_idx, int position_idx  )
{
  unsigned seed = tea<4>( child_idx, position_idx );
  const float f0 = rnd( seed ) - 0.5f;
  const float f1 = rnd( seed ) - 0.5f;
  const float f2 = rnd( seed ) - 0.5f;
  return make_float3( f0, f1, f2 );
}

enum LayoutType
{
  SEPARATE_ACCELS = 0,
  REBUILD_LAYOUT,
  HYBRID_LAYOUT
};

class DynamicLayout
{
public:
  virtual ~DynamicLayout() {}
  virtual void createGeometry( Context ctx, const std::string& filename, int num_meshes ) = 0;
  virtual Aabb getSceneBBox() const = 0;
  virtual void triggerGeometryMove() = 0;
  virtual void updateGeometry() = 0;
  virtual void resetGeometry() = 0;
};

//------------------------------------------------------------------------------
//
// Single BVH layout:
//
//------------------------------------------------------------------------------

class RebuildLayout : public DynamicLayout
{
public:
  RebuildLayout( const std::string& builder, bool print_timing );

  void createGeometry( Context ctx, const std::string& filename, int num_meshes );
  Aabb getSceneBBox()const;
  void triggerGeometryMove();
  void updateGeometry();
  void resetGeometry();

private:
  struct Mesh
  {
    float3    start_pos;
    float3    end_pos;
    float3    last_pos;
    double    move_start_time;
    Buffer    vertices;
  };

  Aabb                    m_aabb;
  std::vector<Mesh>       m_meshes;
  int                     m_num_moved_meshes;
  GeometryGroup           m_top_object;
  std::string             m_builder;
  bool                    m_print_timing;
};


RebuildLayout::RebuildLayout ( const std::string& builder, bool print_timing )
  : m_num_moved_meshes( 0 ), m_builder( builder ), m_print_timing( print_timing )
{
}


void RebuildLayout::createGeometry( Context ctx, const std::string& filename, int num_meshes )
{
  std::cerr << "Creating geometry ... ";

  Acceleration accel = ctx->createAcceleration( m_builder.c_str() );

  m_top_object = ctx->createGeometryGroup();                           
  m_top_object->setAcceleration( accel ); 

  assert( m_meshes.size() == 0 );
  for( int i = 0; i < num_meshes; ++i )
  {
    float3 pos0 = getRandomPosition( i, 0 );
    float3 pos1 = getRandomPosition( i, 1 );

    OptiXMesh omesh;
    omesh.context = ctx;
    if( i ==  0 )
    {
      loadMesh( filename, omesh ); 
      m_aabb = Aabb( omesh.bbox_min, omesh.bbox_max );
    }
    else
    {
      loadMesh( filename, omesh, Matrix4x4::translate( pos0 ) ); 
    }
    m_top_object->addChild( omesh.geom_instance );

    Mesh mesh;
    mesh.start_pos = mesh.last_pos = pos0;
    mesh.end_pos   = pos1;
    mesh.move_start_time = 0.0;
    mesh.vertices  = omesh.geom_instance->getGeometry()->queryVariable( "vertex_buffer" )->getBuffer();
    assert( mesh.vertices );
    m_meshes.push_back( mesh );
  }

  ctx[ "top_object"   ]->set( m_top_object );                            
  ctx[ "top_shadower" ]->set( m_top_object );       

  std::cerr << "done" << std::endl;
}


Aabb RebuildLayout::getSceneBBox()const
{
  return m_aabb;
}


void RebuildLayout::triggerGeometryMove()
{
  if( m_num_moved_meshes < static_cast<int>( m_meshes.size() ) )
  {
    const double cur_time = sutil::currentTime();
    m_meshes[m_num_moved_meshes].move_start_time = cur_time;
    ++m_num_moved_meshes;
  }
}


void RebuildLayout::updateGeometry()
{
  double t0 = sutil::currentTime();
  
  bool meshes_have_moved = false;
  assert( m_num_moved_meshes <= static_cast<int>( m_meshes.size() ) );
  for( int i = 0; i < m_num_moved_meshes; ++i )
  {
    Mesh& mesh = m_meshes[i];
    const double elapsed_time = t0 - mesh.move_start_time;
    if( elapsed_time >= MOVE_ANIMATION_TIME )
      continue;

    const float t       = static_cast<float>(elapsed_time / MOVE_ANIMATION_TIME);
    const float3 pos    = lerp( mesh.start_pos, mesh.end_pos, t );
    const float3 offset = pos - mesh.last_pos; 
    mesh.last_pos       = pos;

    float3* verts = static_cast<float3*>( mesh.vertices->map( 0, RT_BUFFER_MAP_WRITE ) );
    RTsize num_verts;
    mesh.vertices->getSize( num_verts );
    for( int i = 0; i < static_cast<int>( num_verts ); ++i )
      verts[i] += offset;
    mesh.vertices->unmap();

    meshes_have_moved = true;
  }

  double t1 = sutil::currentTime();

  if( meshes_have_moved )
  {
    if( m_print_timing )
    {
      std::cerr << "Geometry transform time: "
                << std::fixed << std::setw( 7 ) << std::setprecision( 2 ) << ( t1-t0 )*1000.0 << "ms" << std::endl;
    }
    t0 = sutil::currentTime();
    m_top_object->getAcceleration()->markDirty();
    m_top_object->getContext()->launch( 0, 0, 0 );
    t1 = sutil::currentTime();
    if( m_print_timing )
    {
      std::cerr << "Accel rebuild time     : "
                << std::fixed << std::setw( 7 ) << std::setprecision( 2 ) << ( t1-t0 )*1000.0 << "ms" << std::endl;
    }
  }

}


void RebuildLayout::resetGeometry()
{
    if( m_num_moved_meshes == 0 )
        return;
    assert( m_num_moved_meshes <= static_cast<int>(m_meshes.size()) );
    for( int i = 0; i < m_num_moved_meshes; ++i )
    {
        Mesh& mesh = m_meshes[i];
        const float3 offset = mesh.last_pos - mesh.start_pos;
        mesh.last_pos = mesh.start_pos;
        mesh.move_start_time = 0.0;

        float3* verts = static_cast<float3*>(mesh.vertices->map( 0, RT_BUFFER_MAP_WRITE ));
        RTsize num_verts;
        mesh.vertices->getSize( num_verts );
        for( int i = 0; i < static_cast<int>(num_verts); ++i )
            verts[i] -= offset;
        mesh.vertices->unmap();
    }

    m_num_moved_meshes = 0;
    m_top_object->getAcceleration()->markDirty();
}


//------------------------------------------------------------------------------
//
// Multi BVH layout:
//
//------------------------------------------------------------------------------

class SeparateAccelsLayout : public DynamicLayout
{
public:
  SeparateAccelsLayout( const std::string& builder, bool print_timing );

  void createGeometry( Context ctx, const std::string& filename, int num_meshes );
  Aabb getSceneBBox()const;
  void triggerGeometryMove();
  void updateGeometry();
  void resetGeometry();

private:
  struct Mesh
  {
    Transform xform;
    float3    start_pos;
    float3    end_pos;
    double    move_start_time;
  };

  Aabb                    m_aabb;
  std::vector<Mesh>       m_meshes;
  int                     m_num_moved_meshes;
  Group                   m_top_object;
  std::string             m_builder;
  bool                    m_print_timing;
};

  
SeparateAccelsLayout::SeparateAccelsLayout( const std::string& builder, bool print_timing )
  : m_num_moved_meshes( 0 ), m_builder( builder ), m_print_timing( print_timing )
{
}


void SeparateAccelsLayout::createGeometry( Context ctx, const std::string& filename, int num_meshes )
{
  std::cerr << "Creating geometry ... ";

  m_top_object = ctx->createGroup();
  m_top_object->setAcceleration( ctx->createAcceleration( m_builder.c_str() ) );

  assert( m_meshes.size() == 0 );
  for( int i = 0; i < num_meshes; ++i )
  {
    
    OptiXMesh omesh;
    omesh.context = ctx;
    loadMesh( filename, omesh ); 
    m_aabb = Aabb( omesh.bbox_min, omesh.bbox_max );
    
    GeometryGroup geometry_group = ctx->createGeometryGroup();
    geometry_group->setAcceleration( ctx->createAcceleration( m_builder.c_str() ) );
    geometry_group->addChild( omesh.geom_instance );

    float3 pos0 = getRandomPosition( i, 0 );
    float3 pos1 = getRandomPosition( i, 1 );

    Transform xform = ctx->createTransform();
    if ( i == 0 ) {
      xform->setMatrix( false, Matrix4x4::identity().getData(), 0 );
    } else {
      xform->setMatrix( false, Matrix4x4::translate( pos0 ).getData(), 0 );
    }
    xform->setChild( geometry_group );
    m_top_object->addChild( xform );

    Mesh mesh;
    mesh.xform     = xform;
    mesh.start_pos = pos0;
    mesh.end_pos   = pos1;
    mesh.move_start_time = 0.0;
    m_meshes.push_back( mesh );
  }

  ctx[ "top_object"   ]->set( m_top_object );                            
  ctx[ "top_shadower" ]->set( m_top_object );       


  std::cerr << "done" << std::endl;;
}


Aabb SeparateAccelsLayout::getSceneBBox()const
{
  return m_aabb; 
}

  
void SeparateAccelsLayout::triggerGeometryMove()
{

  if( m_num_moved_meshes < static_cast<int>( m_meshes.size() ) )
  {
    const double cur_time = sutil::currentTime();
    m_meshes[m_num_moved_meshes].move_start_time = cur_time;
    ++m_num_moved_meshes;
  }
}

void SeparateAccelsLayout::updateGeometry()
{
  double t0 = sutil::currentTime();
  
  bool meshes_have_moved = false;
  assert( m_num_moved_meshes <= static_cast<int>( m_meshes.size() ) );
  for( int i = 0; i < m_num_moved_meshes; ++i )
  {
    Mesh mesh = m_meshes[i];
    const double elapsed_time = t0 - mesh.move_start_time;
    if( elapsed_time >= MOVE_ANIMATION_TIME )
      continue;

    const float t    = static_cast<float>(elapsed_time / MOVE_ANIMATION_TIME);
    const float3 pos = lerp( mesh.start_pos, mesh.end_pos, t );
    mesh.xform->setMatrix( false, Matrix4x4::translate( pos ).getData(), 0 );
    meshes_have_moved = true;
  }

  double t1 = sutil::currentTime();

  if( meshes_have_moved )
  {
    if( m_print_timing )
    {
      std::cerr << "Geometry transform time: "
                << std::fixed << std::setw( 7 ) << std::setprecision( 2 ) << ( t1-t0 )*1000.0 << "ms" << std::endl;
    }
    t0 = sutil::currentTime();
    m_top_object->getAcceleration()->markDirty();
    m_top_object->getContext()->launch( 0, 0, 0 );
    t1 = sutil::currentTime();
    if( m_print_timing )
    {
      std::cerr << "Accel rebuild time     : "
                << std::fixed << std::setw( 7 ) << std::setprecision( 2 ) << ( t1-t0 )*1000.0 << "ms" << std::endl;
    }
  }

}


void SeparateAccelsLayout::resetGeometry()
{
  if( m_num_moved_meshes == 0 )
    return;

  assert( m_num_moved_meshes <= static_cast<int>( m_meshes.size() ) );
  for( int i = 0; i < m_num_moved_meshes; ++i )
  {
    Mesh mesh = m_meshes[i];
    mesh.xform->setMatrix( false, Matrix4x4::translate( mesh.start_pos ).getData(), 0 );
  }
    
  m_num_moved_meshes = 0;
  m_top_object->getAcceleration()->markDirty();
}


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
    context->setStackSize( 1600 );

    context["scene_epsilon"      ]->setFloat( 1.e-4f );
    context["max_depth"          ]->setInt( 5 );

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
    Program miss_program = context->createProgramFromPTXString( sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" ), "miss" );
    context->setMissProgram( 0, miss_program );
    context["bg_color"]->setFloat(  0.1f, 0.7f, 0.7f );
}

  
void setupCamera( const optix::Aabb& aabb )
{
    camera_eye    = aabb.center() + make_float3( 0.0f, 0.0f, 2.05f );
    camera_lookat = aabb.center(); 
    camera_up     = make_float3( 0.0f, 1.0f, 0.0f );

    camera_rotate  = Matrix4x4::identity();
}


void setupLights( const optix::Aabb& /*aabb*/ )
{
    // Lights buffer
    BasicLight lights[] = {
        { make_float3( 60.0f, 60.0f, 60.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
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
    const float vfov  = 45.0f;
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

    layout->updateGeometry();

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
        case( 'r' ):
        {
            layout->resetGeometry();
            break;
        }
        case( ' ' ):
        {
            layout->triggerGeometryMove();
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
        "  -h | --help               Print this usage message and exit.\n"
        "  -f | --file               Save single frame to file and exit.\n"
        "  -n | --nopbo              Disable GL interop for display buffer.\n"
        "  -m | --mesh <mesh_file>   Specify path to mesh to be loaded.\n"
        "  -x | --multi-accel        Turn on multi-acceleration mode (default)\n"
        "  -r | --single-accel       Turn on single-acceleration mode\n"
        "  -t | --print-timing       Print acceleration structure update/rebuild times\n"
        "App Keystrokes:\n"
        "  q      Quit\n" 
        "  s      Save image to '" << SAMPLE_NAME << ".ppm'\n"
        "  space  Translate a randomly chosen mesh\n"
        "  r      Reset scene\n"
        << std::endl;

    exit(1);
}

int main( int argc, char** argv )
 {
    std::string out_file;
    std::string mesh_file = std::string( sutil::samplesDir() ) + "/data/cow.obj";
    LayoutType layout_type = SEPARATE_ACCELS;
    bool print_timing = false;
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
        else if( arg == "-m" || arg == "--mesh" )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            mesh_file = argv[++i];
        }
        else if( arg == "-x" || arg == "--multi-accel" )
        {
            layout_type = SEPARATE_ACCELS;
        }
        else if( arg == "-r" || arg == "--single-accel" )
        {
            layout_type = REBUILD_LAYOUT;
        }
        else if( arg == "-t" || arg == "--print-timing" )
        {
            print_timing = true;
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
        const std::string builder = "Trbvh";
        if( layout_type == SEPARATE_ACCELS )
        {
            std::cerr << "Using multi-acceleration mode\n";
            layout = new SeparateAccelsLayout( builder, print_timing );
        }
        else if( layout_type == REBUILD_LAYOUT )
        {
            std::cerr << "Using single-acceleration mode\n";
            layout = new RebuildLayout( builder, print_timing );
        }
        else
        {
            std::cerr << "WARNING: Unsupported layout requested.  Defaulting to SeparateAccels.\n";
            layout = new SeparateAccelsLayout( builder, print_timing );
        }

        createContext();
        layout->createGeometry( context, mesh_file, 200 );
        const optix::Aabb aabb = layout->getSceneBBox();

        setupCamera( aabb );
        setupLights( aabb );

        std::cerr << "Validating ... ";
        context->validate();
        std::cerr << "done" << std::endl;;

        std::cerr << "Preprocessing scene ... ";
        const double t0 = sutil::currentTime();
        context->launch( 0, 0, 0 );
        const double t1 = sutil::currentTime();
        std::cerr << "done (" << t1-t0 << "sec )" << std::endl;

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

