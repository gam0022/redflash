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
//  Sample demonstrating various instancing cases:
//    - multiple geometry instances referencing the same geometry object
//    - multiple geometry instances referencing the same material
//    - multiple transforms referencing the same geometry group
//    - sharing of acceleration structures between different geometry groups and
//      between higher level groups
//
//-----------------------------------------------------------------------------
 


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <sutil.h>
#include "common.h"

#include <algorithm> // max
#include <iostream>
#include <string>
#include <vector>
#include <cmath> // fmod, fabs
#include <cstring> // memcpy
#include <cstdlib> // exit
#include <cassert>

using namespace optix;
using namespace std;

const char* const SAMPLE_NAME = "optixInstancing";

int width  = 1024;
int height = 720;
bool test_scale;

enum {
    OBJECT_GROUND = 0,
    OBJECT_INSTANCE = 1
};

Context createContext();
void createMaterials( Context context, vector<Material>& materials );
void createGeometries( Context context, vector<Geometry>& geometries );
void createNodeGraph( Context context, const std::vector<Geometry>& geometries,
                      const std::vector<Material>& materials,
                      bool no_accel );
void printUsageAndExit( const std::string& argv0 );

 

int main(int argc, char* argv[])
{
    // Process command line options
    std::string outfile;
    bool no_accel = false;
    test_scale = true;
    for ( int i = 1; i < argc; i++ ) {
        std::string arg( argv[i] );
        if ( arg == "--no-scale" || arg == "-n" ) {
            test_scale = false;
        } else if ( arg == "--no-accel" || arg == "-na") {
            no_accel = true;
        } else if( arg == "--file" || arg == "-f" ) {
            if( i < argc-1 ) {
                outfile = argv[++i];
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( arg == "--help" || arg == "-h" ) {
            printUsageAndExit( argv[0] );
        } else if ( arg.substr( 0, 6 ) == "--dim=" ) {
            std::string dims_arg = arg.substr(6);
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        } else {
            std::cerr << "Unknown option: '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try {
        /* Process command line args */
        if( outfile.empty() ) {
            sutil::initGlut(&argc, argv);
        }

        vector<Material> materials;
        vector<Geometry> geometries;
        Context context = createContext();
        createMaterials( context, materials );
        createGeometries( context, geometries );
        createNodeGraph( context, geometries, materials, no_accel );

        context->validate();
        context->launch( 0, width, height );
        if( outfile.empty() ) {
            sutil::displayBufferGlut( argv[0], context["output_buffer"]->getBuffer() );
        } else {
            sutil::displayBufferPPM( outfile.c_str(), context["output_buffer"]->getBuffer() );
        }
        context->destroy();
    } catch( Exception& e ) {
        sutil::reportErrorMessage( e.getErrorString().c_str() );
        exit( 1 );
    }
    return 0;
}


Context createContext()
{
    // Context.
    Context context = Context::create();
    context->setEntryPointCount( 1 );
    context->setRayTypeCount( 2 );
    context->setStackSize( 1760 );

    context["scene_epsilon"]->setFloat( 1.e-3f );
    context["max_depth"]->setInt( 2 );
    context["ambient_light_color"]->setFloat( 1.0f, 1.0f, 1.0f );

    // Output buffer.
    Variable output_buffer = context["output_buffer"];
    Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height );
    output_buffer->set(buffer);

    // Ray generation program.
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "pinhole_camera.cu" );
    Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program.
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program.
    context->setMissProgram( 0, context->createProgramFromPTXString( sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" ), "miss" ) );
    context["bg_color"]->setFloat( 0.462f, 0.4f, 0.925f );

    // Light variables.
    BasicLight light = { make_float3(12.0f, 10.0f, 4.0f), make_float3(1.0f, 1.0f, 1.0f), 1 };

    Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(BasicLight));
    light_buffer->setSize( 1 );
    memcpy(light_buffer->map(), &light, sizeof(light));
    light_buffer->unmap();
    context[ "lights" ]->set( light_buffer );

    // Camera variables.
    float3 eye = make_float3(15, 10, 15);
    float3 lookat = make_float3(0.0f, 0.3f, 0.0f);
    float3 up = make_float3(0, 1, 0);
    float hfov = 60;
    float3 lookdir = normalize(lookat-eye);
    float3 camera_u = cross(lookdir, up);
    float3 camera_v = cross(camera_u, lookdir);
    float ulen = tanf(hfov/2.0f*M_PIf/180.0f);
    camera_u = normalize(camera_u);
    camera_u *= ulen;
    float aspect_ratio = static_cast<float>(width)/static_cast<float>(height);
    float vlen = ulen/aspect_ratio;
    camera_v = normalize(camera_v);
    camera_v *= vlen;
    context["eye"]->setFloat( eye.x, eye.y, eye.z );
    context["U"]->setFloat( camera_u.x, camera_u.y, camera_u.z );
    context["V"]->setFloat( camera_v.x, camera_v.y, camera_v.z );
    context["W"]->setFloat( lookdir.x, lookdir.y, lookdir.z );

    return context;
}


Material createBaseMaterial( Context context, const char *ptx )
{
    Material material = context->createMaterial();
    Program  radiance_program = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program  shadow_program   = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
    material->setClosestHitProgram( 0, radiance_program );
    material->setAnyHitProgram( 1, shadow_program );
    material["Ks"]->setFloat( 0.4f, 0.4f, 0.4f );
    material["phong_exp"]->setFloat( 200.0f );
    return material;
}


void createMaterials( Context context, vector<Material>& materials )
{
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "phong.cu" );

    Material groundMat = createBaseMaterial( context, ptx );
    groundMat["Kd"]->setFloat( 0.7f, 0.7f, 0.7f );
    groundMat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
    groundMat["Kr"]->setFloat( 0.0f, 0.0f, 0.0f );
    materials.push_back( groundMat );

    Material instanceMat = createBaseMaterial( context, ptx );
    instanceMat["Kd"]->setFloat( 1.0f, 1.0f, 1.0f );
    instanceMat["Ka"]->setFloat( 0.2f, 0.2f, 0.2f );
    instanceMat["Kr"]->setFloat( 0.1f, 0.1f, 0.1f );
    materials.push_back( instanceMat );
}


Geometry createBaseGeometry( Context context, const char *ptx_filename )
{
    Geometry geometry = context->createGeometry();
    geometry->setPrimitiveCount( 1u );
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, ptx_filename );
    geometry->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    geometry->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    return geometry;
}


void createGeometries( Context context, std::vector<Geometry>& geometries )
{
    // Ground geometry.
    Geometry ground_geom = createBaseGeometry( context, "parallelogram.cu" );
    const float3 anchor = make_float3( -1, 0, 1);
    float3 v1 = make_float3( 2, 0, 0);
    float3 v2 = make_float3( 0, 0, -2);
    const float3 normal = normalize( cross( v1, v2 ) );
    const float d = dot( normal, anchor );
    v1 *= 1.0f/dot( v1, v1 );
    v2 *= 1.0f/dot( v2, v2 );
    const float4 plane = make_float4( normal, d );
    ground_geom["plane"]->setFloat( plane.x, plane.y, plane.z, plane.w );
    ground_geom["v1"]->setFloat( v1.x, v1.y, v1.z );
    ground_geom["v2"]->setFloat( v2.x, v2.y, v2.z );
    ground_geom["anchor"]->setFloat( anchor.x, anchor.y, anchor.z );
    geometries.push_back( ground_geom );

    // Sphere geometry.
    Geometry sphere_geom = createBaseGeometry( context, "sphere.cu" );
    sphere_geom["sphere"]->setFloat( 0.0f, 1.0f, 0.0f, 1.0f );
    geometries.push_back( sphere_geom );
}


void rgbToHls( float red, float green, float blue, float &hue, float &lightness, float &saturation )
{
    const float c_max = std::max( red, std::max( green, blue ) );
    const float c_min = std::min( red, std::min( green, blue ) );
    const float delta = c_max - c_min;
    lightness = (c_max + c_min)*0.5f;
    saturation = delta < 1.0e-6f ? 0.0f : delta / (1.0f - std::fabs( c_max + c_min - 1.0f ));
    if( delta < 1.0e-6f ) {
        hue = 0.0f;
    } else if( c_max == red ) {
        hue = 60.0f*std::fmod( (green - blue) / delta, 6.0f );
    } else if( c_max == green ) {
        hue = 60.0f*((blue - red) / delta + 2.0f);
    } else if( c_max == blue ) {
        hue = 60.0f*((red - green) / delta + 4.0f);
    }
}


void hlsToRgb( float hue, float lightness, float saturation, float &red, float &green, float &blue )
{
    const float c = (1.0f - std::fabs( 2.0f*lightness - 1.0f ))*saturation;
    const float x = c*(1.0f - std::fabs( std::fmod( hue / 60.0f, 2.0f ) - 1.0f ));
    const float m = lightness - c/2.0f;
    if (hue < 60.0f) {
        red = c;
        green = x;
        blue = 0;
    } else if (hue < 120.0f) {
        red = x;
        green = c;
        blue = 0;
    } else if (hue < 180.0f) {
        red = 0;
        green = c;
        blue = x;
    } else if (hue < 240.0f) {
        red = 0;
        green = x;
        blue = c;
    } else if (hue < 300.0f) {
        red = x;
        green = 0;
        blue = c;
    } else {
        red = c;
        green = 0;
        blue = x;
    }
    red += m;
    green += m;
    blue += m;
}


void hueShiftKd( GeometryInstance sphere, float Kd_red, float Kd_green, float Kd_blue, unsigned int col )
{
    // shift hue by 65 degrees per column
    float hue, lightness, saturation;
    rgbToHls( Kd_red, Kd_green, Kd_blue, hue, lightness, saturation );
    hue = std::fmod( hue + 65.0f*col, 360.0f );
    hlsToRgb( hue, lightness, saturation, Kd_red, Kd_green, Kd_blue );

    sphere["Kd"]->setFloat( Kd_red, Kd_green, Kd_blue );
}


void setInstanceMaterialParameters( GeometryInstance sphere, unsigned row, unsigned col )
{
    switch( row )
    {
    case 0:
        hueShiftKd( sphere, 0.5f, 0.1f, 0.1f, col );
        break;
    case 1:
        hueShiftKd( sphere, 0.1f, 0.5f, 0.1f, col );
        break;
    case 2:
        hueShiftKd( sphere, 0.1f, 0.1f, 0.5f, col );
        break;
    case 3:
        hueShiftKd( sphere, 0.5f, 0.5f, 0.1f, col );
        break;
    default: break;
    }
}


void createNodeGraph( Context context, const vector<Geometry>& geometries, const vector<Material>& materials, bool no_accel )
{
    const unsigned int num_cols = 4U;
    const unsigned int num_rows = 4U;

    Group top_group = context->createGroup();;
    top_group->setAcceleration( context->createAcceleration( no_accel ? "NoAccel" : "Trbvh" ) );
    top_group->setChildCount( num_cols * num_rows );

    // Since instances share the same Geometry, they can also share the same Acceleration
    Acceleration accel = context->createAcceleration( no_accel ? "NoAccel" : "Trbvh" );

    for( unsigned int row = 0; row < num_rows; ++row ) {
        for( unsigned int col = 0; col < num_cols; ++col ) {
            GeometryGroup gg = context->createGeometryGroup();
            gg->setAcceleration( accel );
            gg->setChildCount( 2 );

            GeometryInstance sphere = context->createGeometryInstance();
            sphere->setGeometry( geometries[OBJECT_INSTANCE] );
            sphere->setMaterialCount( 1 );
            sphere->setMaterial( 0, materials[OBJECT_INSTANCE] );
            setInstanceMaterialParameters( sphere, row, col );
            gg->setChild( OBJECT_INSTANCE, sphere );

            GeometryInstance ground = context->createGeometryInstance();
            ground->setGeometry( geometries[OBJECT_GROUND] );
            ground->setMaterialCount( 1 );
            ground->setMaterial( 0, materials[OBJECT_GROUND] );
            gg->setChild( OBJECT_GROUND, ground );

            Transform transform = context->createTransform();
            float m[16] = {
                1.0f, 0.0f, 0.0f, row * 3.0f - 1.5f,
                0.0f, 1.0f, 0.0f, 0.5f,
                0.0f, 0.0f, 1.0f, col * 2.5f - 1.5f,
                0.0f, 0.0f, 0.0f, 1.0f
            };
            if ( test_scale ) {
                m[5] = 1.0f - col*0.25f;
            }
            transform->setMatrix( false, m, NULL );
            transform->setChild( gg );

            top_group->setChild( row*num_rows + col, transform );
        }
    }

    // Attach to context
    context["top_object"]->set( top_group );
    context["top_shadower"]->set( top_group );
}


void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n"
        << "Options: --help     | -h             Print this usage message\n"
        << "         --file     | -f <filename>  Specify file for image output\n"
        << "       : --no-scale | -n             Turn off testing of scale transformations\n" 
        << "         --dim=<width>x<height>      Set image dimensions; defaults to 1024x720\n";
    exit(1); 
}   

