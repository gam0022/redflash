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
// optixDenoiser: simple interactive path tracer with denoising 
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

#include "optixDenoiser.h"
#include <sutil.h>
#include <Arcball.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <string>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <cstdlib>
#include <iomanip>

using namespace optix;

const char* const SAMPLE_NAME = "optixDenoiser";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context        context = 0;
int            width  = 512;
int            height = 512;
bool           use_pbo = true;

bool           denoiser_perf_mode = false;
int            denoiser_perf_iter = 1;

int            frame_number = 1;
int            sqrt_num_samples = 2;
int            rr_begin_depth = 1;
Program        pgram_intersection = 0;
Program        pgram_bounding_box = 0;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
bool           camera_changed = true;
bool           postprocessing_needs_init = true;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

// Post-processing
CommandList commandListWithDenoiser;
CommandList commandListWithoutDenoiser;
PostprocessingStage tonemapStage;
PostprocessingStage denoiserStage;
Buffer denoisedBuffer;
Buffer emptyBuffer;
Buffer trainingDataBuffer;

// number of frames that show the original image before switching on denoising
int numNonDenoisedFrames = 4;

// Defines the amount of the original image that is blended with the denoised result
// ranging from 0.0 to 1.0
float denoiseBlend = 0.f;

// Defines which buffer to show.
// 0 - denoised 1 - original, 2 - tonemapped, 3 - albedo, 4 - normal
int showBuffer = 0;

// The denoiser mode.
// 0 - RGB only, 1 - RGB + albedo, 2 - RGB + albedo + normals
int denoiseMode = 0;

// The path to the training data file set with -t or empty
std::string training_file;

// The path to the second training data file set with -t2 or empty
std::string training_file_2;

// Toggles between using custom training data (if set) or the built in training data.
bool useCustomTrainingData = true;

// Toggles the custom data between the one specified with -t1 and -t2, if available.
bool useFirstTrainingDataPath = true;

// Contains info for the currently shown buffer
std::string bufferInfo;

//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void loadGeometry();
void setupCamera();
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


void loadTrainingFile(const std::string& path)
{
    if (path.length() == 0)
    {
        trainingDataBuffer->setSize(0);
        return;
    }

    using namespace std;
    ifstream fin(path.c_str(), ios::in | ios::ate | ios::binary);
    if (fin.fail())
    {
        fprintf(stderr, "Failed to load training file %s\n", path.c_str());
        return;
    }
    size_t size = static_cast<size_t>(fin.tellg());

    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "Failed to load training file %s\n", path.c_str());
        return;
    }

    trainingDataBuffer->setSize(size);

    char* data = reinterpret_cast<char*>(trainingDataBuffer->map());

    const bool ok = fread(data, 1, size, fp) == size;
    fclose(fp);

    trainingDataBuffer->unmap();

    if (!ok)
    {
        fprintf(stderr, "Failed to load training file %s\n", path.c_str());
        trainingDataBuffer->setSize(0);
    }
}


Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}

Buffer getTonemappedBuffer()
{
    return context[ "tonemapped_buffer" ]->getBuffer();
}

Buffer getAlbedoBuffer()
{
    return context["input_albedo_buffer"]->getBuffer();
}

Buffer getNormalBuffer()
{
    return context["input_normal_buffer"]->getBuffer();
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
#endif
}

void convertNormalsToColors(
  Buffer& normalBuffer)
{
    float* data = reinterpret_cast<float*>(normalBuffer->map());

    RTsize width, height;
    normalBuffer->getSize(width, height);

    RTsize size = width * height;
    for (size_t i = 0; i < size; ++i)
    {
      const float r = *(data + 3*i);
      const float g = *(data + 3*i + 1);
      const float b = *(data + 3*i + 2);

      *(data + 3*i) = std::abs(r);
      *(data + 3*i + 1) = std::abs(g);
      *(data + 3*i + 2) = std::abs(b);
    }

    normalBuffer->unmap();
}


void setMaterial(
        GeometryInstance& gi,
        Material material,
        const std::string& color_name,
        const float3& color)
{
    gi->addMaterial(material);
    gi[color_name]->setFloat(color);
}


GeometryInstance createParallelogram(
        const float3& anchor,
        const float3& offset1,
        const float3& offset2)
{
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );
    parallelogram->setIntersectionProgram( pgram_intersection );
    parallelogram->setBoundingBoxProgram( pgram_bounding_box );

    float3 normal = normalize( cross( offset1, offset2 ) );
    float d = dot( normal, anchor );
    float4 plane = make_float4( normal, d );

    float3 v1 = offset1 / dot( offset1, offset1 );
    float3 v2 = offset2 / dot( offset2, offset2 );

    parallelogram["plane"]->setFloat( plane );
    parallelogram["anchor"]->setFloat( anchor );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );

    GeometryInstance gi = context->createGeometryInstance();
    gi->setGeometry(parallelogram);
    return gi;
}

void denoiserReportCallback(int lvl, const char* tag, const char* msg, void* cbdata)
{
    if (std::string("DLDENOISER") == tag)
        std::cout << "[" << std::left << std::setw(12) << tag << "] " << msg;
    else if (std::string("POSTPROCESSING") == tag && denoiser_perf_mode)
        std::cout << "[" << std::left << std::setw(12) << tag << "] " << msg;

}

void createContext()
{
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 1800 );

    context[ "scene_epsilon"                  ]->setFloat( 1.e-3f );
    context[ "rr_begin_depth"                 ]->setUint( rr_begin_depth );

    context->setUsageReportCallback(denoiserReportCallback, 2, NULL);

    Buffer renderBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["output_buffer"]->set(renderBuffer);
    Buffer tonemappedBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["tonemapped_buffer"]->set(tonemappedBuffer); 
    Buffer albedoBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["input_albedo_buffer"]->set(albedoBuffer);

    // The normal buffer use float4 for performance reasons, the fourth channel will be ignored.
    Buffer normalBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["input_normal_buffer"]->set(normalBuffer);

    denoisedBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    emptyBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, 0, 0);
    trainingDataBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, 0);

    // Setup programs
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "optixDenoiser.cu" );
    context->setRayGenerationProgram( 0, context->createProgramFromPTXString( ptx, "pathtrace_camera" ) );
    context->setExceptionProgram( 0, context->createProgramFromPTXString( ptx, "exception" ) );
    context->setMissProgram( 0, context->createProgramFromPTXString( ptx, "miss" ) );

    context[ "sqrt_num_samples" ]->setUint( sqrt_num_samples );
    context[ "bad_color"        ]->setFloat( 1000000.0f, 0.0f, 1000000.0f ); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
    context[ "bg_color"         ]->setFloat( make_float3(0.0f) );
}


void loadGeometry()
{
    // Light buffer
    ParallelogramLight light;
    light.corner   = make_float3( 343.0f, 548.6f, 227.0f);
    light.v1       = make_float3( -130.0f, 0.0f, 0.0f);
    light.v2       = make_float3( 0.0f, 0.0f, 105.0f);
    light.normal   = normalize( cross(light.v1, light.v2) );
    light.emission = make_float3( 340.0f, 190.0f, 100.0f );

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( ParallelogramLight ) );
    light_buffer->setSize( 1u );
    memcpy( light_buffer->map(), &light, sizeof( light ) );
    light_buffer->unmap();
    context["lights"]->setBuffer( light_buffer );


    // Set up material
    Material diffuse = context->createMaterial();
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "optixDenoiser.cu" );
    Program diffuse_ch = context->createProgramFromPTXString( ptx, "diffuse" );
    Program diffuse_ah = context->createProgramFromPTXString( ptx, "shadow" );
    diffuse->setClosestHitProgram( 0, diffuse_ch );
    diffuse->setAnyHitProgram( 1, diffuse_ah );

    Material diffuse_light = context->createMaterial();
    Program diffuse_em = context->createProgramFromPTXString( ptx, "diffuseEmitter" );
    diffuse_light->setClosestHitProgram( 0, diffuse_em );

    // Set up parallelogram programs
    ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
    pgram_bounding_box = context->createProgramFromPTXString( ptx, "bounds" );
    pgram_intersection = context->createProgramFromPTXString( ptx, "intersect" );

    // create geometry instances
    std::vector<GeometryInstance> gis;

    const float3 white = make_float3( 0.8f, 0.8f, 0.8f );
    const float3 green = make_float3( 0.05f, 0.8f, 0.05f );
    const float3 red   = make_float3( 0.8f, 0.05f, 0.05f );
    const float3 light_em = make_float3( 340.0f, 190.0f, 100.0f );

    // Floor
    gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 0.0f ),
                                        make_float3( 0.0f, 0.0f, 559.2f ),
                                        make_float3( 556.0f, 0.0f, 0.0f ) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);

    // Ceiling
    gis.push_back( createParallelogram( make_float3( 0.0f, 548.8f, 0.0f ),
                                        make_float3( 556.0f, 0.0f, 0.0f ),
                                        make_float3( 0.0f, 0.0f, 559.2f ) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);

    // Back wall
    gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 559.2f),
                                        make_float3( 0.0f, 548.8f, 0.0f),
                                        make_float3( 556.0f, 0.0f, 0.0f) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);

    // Right wall
    gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 0.0f ),
                                        make_float3( 0.0f, 548.8f, 0.0f ),
                                        make_float3( 0.0f, 0.0f, 559.2f ) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", green);

    // Left wall
    gis.push_back( createParallelogram( make_float3( 556.0f, 0.0f, 0.0f ),
                                        make_float3( 0.0f, 0.0f, 559.2f ),
                                        make_float3( 0.0f, 548.8f, 0.0f ) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", red);

    // Short block
    gis.push_back( createParallelogram( make_float3( 130.0f, 165.0f, 65.0f),
                                        make_float3( -48.0f, 0.0f, 160.0f),
                                        make_float3( 160.0f, 0.0f, 49.0f) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back( createParallelogram( make_float3( 290.0f, 0.0f, 114.0f),
                                        make_float3( 0.0f, 165.0f, 0.0f),
                                        make_float3( -50.0f, 0.0f, 158.0f) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back( createParallelogram( make_float3( 130.0f, 0.0f, 65.0f),
                                        make_float3( 0.0f, 165.0f, 0.0f),
                                        make_float3( 160.0f, 0.0f, 49.0f) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back( createParallelogram( make_float3( 82.0f, 0.0f, 225.0f),
                                        make_float3( 0.0f, 165.0f, 0.0f),
                                        make_float3( 48.0f, 0.0f, -160.0f) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back( createParallelogram( make_float3( 240.0f, 0.0f, 272.0f),
                                        make_float3( 0.0f, 165.0f, 0.0f),
                                        make_float3( -158.0f, 0.0f, -47.0f) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);

    // Tall block
    gis.push_back( createParallelogram( make_float3( 423.0f, 330.0f, 247.0f),
                                        make_float3( -158.0f, 0.0f, 49.0f),
                                        make_float3( 49.0f, 0.0f, 159.0f) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back( createParallelogram( make_float3( 423.0f, 0.0f, 247.0f),
                                        make_float3( 0.0f, 330.0f, 0.0f),
                                        make_float3( 49.0f, 0.0f, 159.0f) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back( createParallelogram( make_float3( 472.0f, 0.0f, 406.0f),
                                        make_float3( 0.0f, 330.0f, 0.0f),
                                        make_float3( -158.0f, 0.0f, 50.0f) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back( createParallelogram( make_float3( 314.0f, 0.0f, 456.0f),
                                        make_float3( 0.0f, 330.0f, 0.0f),
                                        make_float3( -49.0f, 0.0f, -160.0f) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back( createParallelogram( make_float3( 265.0f, 0.0f, 296.0f),
                                        make_float3( 0.0f, 330.0f, 0.0f),
                                        make_float3( 158.0f, 0.0f, -49.0f) ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);

    // Create shadow group (no light)
    GeometryGroup shadow_group = context->createGeometryGroup(gis.begin(), gis.end());
    shadow_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
    context["top_shadower"]->set( shadow_group );

    // Light
    gis.push_back( createParallelogram( make_float3( 343.0f, 548.6f, 227.0f),
                                        make_float3( -130.0f, 0.0f, 0.0f),
                                        make_float3( 0.0f, 0.0f, 105.0f) ) );
    setMaterial(gis.back(), diffuse_light, "emission_color", light_em);

    // Create geometry group
    GeometryGroup geometry_group = context->createGeometryGroup(gis.begin(), gis.end());
    geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
    context["top_object"]->set( geometry_group );
}

  
void setupCamera()
{
    camera_eye    = make_float3( 278.0f, 273.0f, -900.0f );
    camera_lookat = make_float3( 278.0f, 273.0f,    0.0f );
    camera_up     = make_float3(   0.0f,   1.0f,    0.0f );

    camera_rotate  = Matrix4x4::identity();
}


void updateCamera()
{
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

    if( camera_changed ) // reset accumulation
        frame_number = 1;
    camera_changed = false;

    context[ "frame_number" ]->setUint( frame_number );
    context[ "eye"]->setFloat( camera_eye );
    context[ "U"  ]->setFloat( camera_u );
    context[ "V"  ]->setFloat( camera_v );
    context[ "W"  ]->setFloat( camera_w );

    const Matrix4x4 current_frame_inv = Matrix4x4::fromBasis( 
            normalize( camera_u ),
            normalize( camera_v ),
            normalize( -camera_w ),
            camera_lookat).inverse();
    Matrix3x3 normal_matrix = make_matrix3x3(current_frame_inv);

    context[ "normal_matrix"  ]->setMatrix3x3fv(false,normal_matrix.getData());
}


void setupPostprocessing()
{

    if (!tonemapStage)
    {
        // create stages only once: they will be reused in several command lists without being re-created
        tonemapStage = context->createBuiltinPostProcessingStage("TonemapperSimple");
        denoiserStage = context->createBuiltinPostProcessingStage("DLDenoiser");
        if (trainingDataBuffer)
        {
            Variable trainingBuff = denoiserStage->declareVariable("training_data_buffer");
            trainingBuff->set(trainingDataBuffer);
        }

        tonemapStage->declareVariable("input_buffer")->set(getOutputBuffer());
        tonemapStage->declareVariable("output_buffer")->set(getTonemappedBuffer());
        tonemapStage->declareVariable("exposure")->setFloat(0.25f);
        tonemapStage->declareVariable("gamma")->setFloat(2.2f);

        denoiserStage->declareVariable("input_buffer")->set(getTonemappedBuffer());
        denoiserStage->declareVariable("output_buffer")->set(denoisedBuffer);
        denoiserStage->declareVariable("blend")->setFloat(denoiseBlend);
        denoiserStage->declareVariable("input_albedo_buffer");
        denoiserStage->declareVariable("input_normal_buffer");
    }

    if (commandListWithDenoiser) 
    {
        commandListWithDenoiser->destroy();
        commandListWithoutDenoiser->destroy();
    }

    // Create two command lists with two postprocessing topologies we want:
    // One with the denoiser stage, one without. Note that both share the same
    // tonemap stage.

    commandListWithDenoiser = context->createCommandList();
    commandListWithDenoiser->appendLaunch(0, width, height);
    commandListWithDenoiser->appendPostprocessingStage(tonemapStage, width, height);
    commandListWithDenoiser->appendPostprocessingStage(denoiserStage, width, height);
    commandListWithDenoiser->finalize();

    commandListWithoutDenoiser = context->createCommandList();
    commandListWithoutDenoiser->appendLaunch(0, width, height);
    commandListWithoutDenoiser->appendPostprocessingStage(tonemapStage, width, height);
    commandListWithoutDenoiser->finalize();

    postprocessing_needs_init = false;
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

    if (postprocessing_needs_init)
    {
        setupPostprocessing();
    }

    Variable(denoiserStage->queryVariable("blend"))->setFloat(denoiseBlend);

    bool isEarlyFrame = (frame_number <= numNonDenoisedFrames);
    if (isEarlyFrame)
    {
        commandListWithoutDenoiser->execute();
    }
    else
    {
        commandListWithDenoiser->execute();
    }

    switch (showBuffer)
    {
    case 1:
    {
        bufferInfo = "Original";
        sutil::displayBufferGL(getOutputBuffer());
        break;
    }
    case 2:
    {
        bufferInfo = "Tonemapped";
        // gamma correction already applied by tone mapper, avoid doing it twice
        sutil::displayBufferGL(getTonemappedBuffer(), BUFFER_PIXEL_FORMAT_DEFAULT, true);
        break;
    }
    case 3:
    {
        bufferInfo = "Albedo";
        sutil::displayBufferGL(getAlbedoBuffer());
        break;
    }
    case 4:
    {
        bufferInfo = "Normals";
        Buffer normalBuffer = getNormalBuffer();
        //convertNormalsToColors(normalBuffer);
        sutil::displayBufferGL(normalBuffer);
        break;
    }
    default:
        switch (denoiseMode)
        {
            case 0:
            {
                bufferInfo = "Denoised";
                break;
            }
            case 1:
            {
                bufferInfo = "Denoised (albedo)";
                break;
            }
            case 2:
            {
                bufferInfo = "Denoised (albedo+normals)";
                break;
            }
        }
        if (isEarlyFrame)
        {
            bufferInfo = "Tonemapped (early frame non-denoised)";
            // gamma correction already applied by tone mapper, avoid doing it twice
            sutil::displayBufferGL(getTonemappedBuffer(), BUFFER_PIXEL_FORMAT_DEFAULT, true);
        }
        else
        {
            RTsize trainingSize = 0;
            trainingDataBuffer->getSize(trainingSize);
            if (useCustomTrainingData && trainingSize > 0)
            {
                if (useFirstTrainingDataPath)
                    bufferInfo += " Custom data";
                else
                    bufferInfo += " Custom data 2";
            }

            // gamma correction already applied by tone mapper, avoid doing it twice
            sutil::displayBufferGL(denoisedBuffer, BUFFER_PIXEL_FORMAT_DEFAULT, true);
        }

    }

    {
        static unsigned frame_count = 0;
        sutil::displayFps( frame_count++ );
    }

    sutil::displayText(bufferInfo.c_str(), 140, 10);
    char str[64];
    sprintf(str, "#%d", frame_number);
    sutil::displayText(str, (float)width - 50, (float)height - 20);

    frame_number++;

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
            Buffer buff;
            bool disableSrgbConversion = true;
            switch (showBuffer)
            {
                case 0:
                {
                    buff = denoisedBuffer;
                    break;
                }
                case 1:
                {
                    disableSrgbConversion = false;
                    buff = getOutputBuffer();
                    break;
                }
                case 2:
                {
                    buff = getTonemappedBuffer();
                    break;
                }
                case 3:
                {
                    disableSrgbConversion = false;
                    buff = getAlbedoBuffer();
                    break;
                }
                case 4:
                {
                    disableSrgbConversion = false;
                    buff = getNormalBuffer();
                    break;
                }
            }

            const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), buff, disableSrgbConversion );
            break;
        }
        case('d'):
        {
            showBuffer = 0;
            break;
        }
        case('o'):
        {
            showBuffer = 1;
            break;
        }
        case('t'):
        {
            showBuffer = 2;
            break;
        }
        case('a'):
        {
            showBuffer = 3;
            break;
        }
        case('n'):
        {
            showBuffer = 4;
            break;
        }
        case('m'):
        {
            ++denoiseMode;
            if (denoiseMode > 2) denoiseMode = 0;
            switch (denoiseMode)
            {
                case 0:
                {
                    Variable albedoBuffer = denoiserStage->queryVariable("input_albedo_buffer");
                    albedoBuffer->set(emptyBuffer);
                    Variable normalBuffer = denoiserStage->queryVariable("input_normal_buffer");
                    normalBuffer->set(emptyBuffer);
                    break;
                }
                case 1:
                {
                    Variable albedoBuffer = denoiserStage->queryVariable("input_albedo_buffer");
                    albedoBuffer->set(getAlbedoBuffer());
                    break;
                }
                case 2:
                {
                  Variable normalBuffer = denoiserStage->queryVariable("input_normal_buffer");
                  normalBuffer->set(getNormalBuffer());
                  break;
                }
            }
            break;
        }
        case('0'):
        {
            denoiseBlend = 0.f;
            break;
        }
        case('1'):
        {
            denoiseBlend = 0.1f;
            break;
        }
        case('2'):
        {
            denoiseBlend = 0.2f;
            break;
        }
        case('3'):
        {
            denoiseBlend = 0.3f;
            break;
        }
        case('4'):
        {
            denoiseBlend = 0.4f;
            break;
        }
        case('5'):
        {
            denoiseBlend = 0.5f;
            break;
        }
        case('6'):
        {
            denoiseBlend = 0.6f;
            break;
        }
        case('7'):
        {
            denoiseBlend = 0.7f;
            break;
        }
        case('8'):
        {
            denoiseBlend = 0.8f;
            break;
        }
        case('9'):
        {
            denoiseBlend = 0.9f;
            break;
        }
        case('c'):
        {
            useCustomTrainingData = !useCustomTrainingData;
            Variable trainingBuff = denoiserStage->queryVariable("training_data_buffer");
            if (trainingBuff)
            {
                if (useCustomTrainingData)
                    trainingBuff->setBuffer(trainingDataBuffer);
                else
                    trainingBuff->setBuffer(emptyBuffer);
            }
            break;
        }
        case('z'):
        {
            useFirstTrainingDataPath = !useFirstTrainingDataPath;
            if (useFirstTrainingDataPath)
            {
                if (training_file.length() == 0)
                    useFirstTrainingDataPath = false;
                else
                    loadTrainingFile(training_file);
            }
            else
            {
                if (training_file_2.length() == 0)
                    useFirstTrainingDataPath = true;
                else
                    loadTrainingFile(training_file_2);
            }
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


void glutResize( int w, int h )
{
    if ( w == (int)width && h == (int)height ) return;

    camera_changed = true;

    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    sutil::resizeBuffer( getTonemappedBuffer(), width, height );
    sutil::resizeBuffer( getAlbedoBuffer(), width, height );
    sutil::resizeBuffer( getNormalBuffer(), width, height );
    sutil::resizeBuffer( denoisedBuffer, width, height );

    glViewport(0, 0, width, height);                                               

    postprocessing_needs_init = true;

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
        "  -h  | --help                   Print this usage message and exit.\n"
        "  -f  | --file <path>            Save single frame to file and exit.\n"
        "  -d  | --dim=<width>x<height>   Set image dimensions. Defaults to 512x512\n"
        "  -b  | --blend <blend>          The blend factor in percent (0-100). Defaults to 0.\n"
        "  -m  | --denoise_mode <mode>    0: rgb buffer only, 1: rgb + albedo, 2: rgb + albedo + normals. Defaults to 0.\n"
        "  -p  | --perf <iter>            Renders iter frames, outputs post processing usage reports to stdout, and then exits.\n"
        "  -n  | --nopbo                  Disable GL interop for display buffer.\n"
        "  -t  | --training_file <path>   Specify an optional denoising training data file.\n"
        "  -t2 | --training_file_2 <path> Specify an optional second denoising training data file.\n"
        "App Keystrokes:\n"
        "  q  Quit\n" 
        "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
        "  o  Show original image.\n"
        "  t  Show tone-mapped image.\n"
        "  d  Show denoised image.\n"
        "  a  Show albedo buffer.\n"
        "  n  Show (color representation of) normal buffer.\n"
        "  m  Cycle through rgb only/albedo/albedo+normals denoising mode.\n"
        "  c  Toggle custom training data and built in training data.\n"
        "  z  Toggle custom training data between the one specified by -t and -t2.\n"
        " 0-9 Set amount of blending with original image from 0% to 90%.\n"
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
        else if (arg.find("-d") == 0 || arg.find("--dim") == 0)
        {
            size_t index = arg.find_first_of('=');
            if(index == std::string::npos)
            {
                std::cerr << "Option '" << arg << " is malformed. Please use the syntax -d | --dim=<width>x<height>.\n";
                printUsageAndExit(argv[0]);
            }
            std::string dim = arg.substr(index+1);
            try
            {
                sutil::parseDimensions(dim.c_str(), width, height);
            }
            catch (Exception e)
            {
                std::cerr << "Option '" << arg << " is malformed. Please use the syntax -d | --dim=<width>x<height>.\n";
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg == "-b" || arg == "--blend")
        {
            if (i == argc-1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            int denoiseBlendPercent = atoi(argv[++i]);
            if (denoiseBlendPercent < 0) denoiseBlendPercent = 0;
            if (denoiseBlendPercent > 100) denoiseBlendPercent = 100;
            denoiseBlend = denoiseBlendPercent/100.f;
        }
        else if (arg == "-m" || arg == "--denoise_mode")
        {
            if (i == argc-1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            denoiseMode = atoi(argv[++i]);
            if( denoiseMode<0 || denoiseMode > 2)
            {
                std::cerr << "Option '" << arg << "' must be 0, 1, or 2.\n";
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg == "-p" || arg == "--perf")
        {
            if (i == argc-1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            denoiser_perf_mode = true;
            denoiser_perf_iter = atoi(argv[++i]);
        }
        else if( arg == "-n" || arg == "--nopbo"  )
        {
            use_pbo = false;
        }
        else if (arg == "-t" || arg == "--training_file")
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            training_file = argv[++i];
        }
        else if( arg == "-t2" || arg == "--training_file2" )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            training_file_2 = argv[++i];
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

        if (training_file.length() == 0 && training_file_2.length() != 0)
            useFirstTrainingDataPath = false;

        if (useFirstTrainingDataPath)
            loadTrainingFile(training_file);
        else
            loadTrainingFile(training_file_2);

        setupCamera();
        loadGeometry();
        

        context->validate();

        if (denoiser_perf_mode)
        {
            setupPostprocessing();
            updateCamera();
            Variable(denoiserStage->queryVariable("blend"))->setFloat(denoiseBlend);
         
            if(denoiseMode > 0)
            {
                Variable albedoBuffer = denoiserStage->queryVariable("input_albedo_buffer");
                albedoBuffer->set(getAlbedoBuffer());
            }

            if(denoiseMode > 1)
            {
                Variable normalBuffer = denoiserStage->queryVariable("input_normal_buffer");
                normalBuffer->set(getNormalBuffer());
            }

            for (int i=0; i<denoiser_perf_iter; i++)
            {
                commandListWithDenoiser->execute();
            }

            destroyContext();
        }
        else if ( out_file.empty() )
        {
            glutRun();
        }
        else
        {
            setupPostprocessing();
            updateCamera();
            Variable(denoiserStage->queryVariable("blend"))->setFloat(denoiseBlend);
            commandListWithDenoiser->execute();
            sutil::displayBufferPPM( out_file.c_str(), denoisedBuffer);
            destroyContext();
        }
        
        return 0;
    }
    SUTIL_CATCH( context->get() )
}

