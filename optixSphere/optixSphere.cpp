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

/*
 *  optixSphere.cpp -- Renders sphere with a normal shader 
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <optix.h>
#include <sutil.h>

const char* const SAMPLE_NAME = "optixSphere";

int width = 1024;
int height = 768;

void createContext( RTcontext* context, RTbuffer* buffer );
void createGeometry( RTcontext context, RTgeometry* sphere );
void createMaterial( RTcontext context, RTmaterial* material );
void createInstance( RTcontext context, RTgeometry sphere, RTmaterial material );
void printUsageAndExit( const char* argv0 );


int main(int argc, char* argv[])
{
    RTcontext context = 0;
    try
    {
        /* Primary RTAPI objects */
        RTbuffer            output_buffer_obj;
        RTgeometry          sphere;
        RTmaterial          material;

        char outfile[512];
        int i;

        outfile[0] = '\0';

        for( i = 1; i < argc; ++i ) {
            if( strcmp( argv[i], "--help" ) == 0 || strcmp( argv[i], "-h" ) == 0 ) {
                printUsageAndExit( argv[0] );
            } else if( strcmp( argv[i], "--file" ) == 0 || strcmp( argv[i], "-f" ) == 0 ) {
                if( i < argc-1 ) {
                    strcpy( outfile, argv[++i] );
                } else {
                    printUsageAndExit( argv[0] );
                }
            } else if ( strncmp( argv[i], "--dim=", 6 ) == 0 ) {
                const char *dims_arg = &argv[i][6];
                sutil::parseDimensions( dims_arg, width, height );
            } else {
                fprintf( stderr, "Unknown option '%s'\n", argv[i] );
                printUsageAndExit( argv[0] );
            }
        }

        /* Process command line args */
        if( strlen( outfile ) == 0 ) {
            sutil::initGlut(&argc, argv);
        }

        /* Setup state */
        createContext( &context, &output_buffer_obj );
        createGeometry( context, &sphere );
        createMaterial( context, &material);
        createInstance( context, sphere, material );

        /* Run */
        RT_CHECK_ERROR( rtContextValidate( context ) );
        RT_CHECK_ERROR( rtContextLaunch2D( context, 0, width, height ) );

        /* Display image */
        if( strlen( outfile ) == 0 ) {
            sutil::displayBufferGlut( argv[0], output_buffer_obj );
        } else {
            sutil::displayBufferPPM( outfile, output_buffer_obj );
        }

        /* Clean up */
        RT_CHECK_ERROR( rtContextDestroy( context ) );
        return( 0 );

    } SUTIL_CATCH( context )
}

void createContext( RTcontext* context, RTbuffer* output_buffer_obj )
{
    RTprogram  ray_gen_program;
    RTprogram  exception_program;
    RTprogram  miss_program;
    RTvariable output_buffer;
    RTvariable epsilon;

    /* variables for ray gen program */
    RTvariable eye;
    RTvariable U;
    RTvariable V;
    RTvariable W;
    RTvariable badcolor;

    /* viewing params */
    float hfov, aspect_ratio; 

    /* variables for miss program */
    RTvariable bg_color;

    /* Setup context */
    RT_CHECK_ERROR( rtContextCreate( context ) );
    RT_CHECK_ERROR( rtContextSetRayTypeCount( *context, 1 ) );
    RT_CHECK_ERROR( rtContextSetEntryPointCount( *context, 1 ) );

    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "output_buffer" , &output_buffer) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "scene_epsilon" , &epsilon) );

    RT_CHECK_ERROR( rtVariableSet1f( epsilon, 1.e-4f ) );

    /* Render result buffer */
    RT_CHECK_ERROR( rtBufferCreate( *context, RT_BUFFER_OUTPUT, output_buffer_obj ) );
    RT_CHECK_ERROR( rtBufferSetFormat( *output_buffer_obj, RT_FORMAT_UNSIGNED_BYTE4 ) );
    RT_CHECK_ERROR( rtBufferSetSize2D( *output_buffer_obj, width, height ) );
    RT_CHECK_ERROR( rtVariableSetObject( output_buffer, *output_buffer_obj ) );

    /* Ray generation program */
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "pinhole_camera.cu" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( *context, ptx, "pinhole_camera", &ray_gen_program ) );
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram( *context, 0, ray_gen_program ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "eye" , &eye) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "U" , &U) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "V" , &V) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "W" , &W) );

    optix::float3 cam_eye = { 0.0f, 0.0f, 5.0f };
    optix::float3 lookat  = { 0.0f, 0.0f, 0.0f };
    optix::float3 up      = { 0.0f, 1.0f, 0.0f };
    hfov      = 60.0f;
    aspect_ratio = (float)width/(float)height;
    optix::float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            cam_eye, lookat, up, hfov, aspect_ratio,
            camera_u, camera_v, camera_w );

    RT_CHECK_ERROR( rtVariableSet3fv( eye, &cam_eye.x ) );
    RT_CHECK_ERROR( rtVariableSet3fv( U, &camera_u.x ) );
    RT_CHECK_ERROR( rtVariableSet3fv( V, &camera_v.x ) );
    RT_CHECK_ERROR( rtVariableSet3fv( W, &camera_w.x ) );

    /* Exception program */
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "bad_color" , &badcolor) );
    RT_CHECK_ERROR( rtVariableSet3f( badcolor, 1.0f, 0.0f, 1.0f ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( *context, ptx, "exception", &exception_program ) );
    RT_CHECK_ERROR( rtContextSetExceptionProgram( *context, 0, exception_program ) );

    /* Miss program */
    ptx = sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( *context, ptx, "miss", &miss_program ) );
    RT_CHECK_ERROR( rtProgramDeclareVariable( miss_program, "bg_color" , &bg_color) );
    RT_CHECK_ERROR( rtVariableSet3f( bg_color, .3f, 0.1f, 0.2f ) );
    RT_CHECK_ERROR( rtContextSetMissProgram( *context, 0, miss_program ) );
}


void createGeometry( RTcontext context, RTgeometry* sphere )
{
    RTprogram  intersection_program;
    RTprogram  bounding_box_program;
    RTvariable s;
    float sphere_loc[4] =  {0, 0, 0, 1.5};

    RT_CHECK_ERROR( rtGeometryCreate( context, sphere ) );
    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( *sphere, 1u ) );

    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "sphere.cu" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "bounds", &bounding_box_program) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( *sphere, bounding_box_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "intersect", &intersection_program) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( *sphere, intersection_program ) );

    RT_CHECK_ERROR( rtGeometryDeclareVariable( *sphere, "sphere" , &s) );
    RT_CHECK_ERROR( rtVariableSet4fv( s, &sphere_loc[0] ) );
}


void createMaterial( RTcontext context, RTmaterial* material )
{
    RTprogram chp;

    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "normal_shader.cu" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "closest_hit_radiance", &chp ) );

    RT_CHECK_ERROR( rtMaterialCreate( context, material ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( *material, 0, chp) );
}


void createInstance( RTcontext context, RTgeometry sphere, RTmaterial material )
{
    RTgeometrygroup geometrygroup;
    RTvariable      top_object;
    RTacceleration  acceleration;
    RTgeometryinstance instance;

    /* Create geometry instance */
    RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &instance ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( instance, sphere ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( instance, 1 ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, 0, material ) );

    /* Create geometry group */
    RT_CHECK_ERROR( rtAccelerationCreate( context, &acceleration ) );
    RT_CHECK_ERROR( rtAccelerationSetBuilder( acceleration, "NoAccel" ) );
    RT_CHECK_ERROR( rtGeometryGroupCreate( context, &geometrygroup ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChildCount( geometrygroup, 1 ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( geometrygroup, 0, instance ) );
    RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( geometrygroup, acceleration ) );

    RT_CHECK_ERROR( rtContextDeclareVariable( context, "top_object", &top_object ) );
    RT_CHECK_ERROR( rtVariableSetObject( top_object, geometrygroup ) );
}


void printUsageAndExit( const char* argv0 )
{
    fprintf( stderr, "Usage  : %s [options]\n", argv0 );
    fprintf( stderr, "Options: --help | -h             Print this usage message\n" );
    fprintf( stderr, "         --file | -f <filename>  Specify file for image output\n" );
    fprintf( stderr, "         --dim=<width>x<height>  Set image dimensions; defaults to 1024x768\n" );
    exit(1);
}
