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
 *  A simple selector sample.
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <sutil.h>
#include <iostream>
#include <string>
#include <cstdlib>

using namespace optix;

const char* const SAMPLE_NAME = "optixSelector";

int width  = 1024;
int height = 768;

const unsigned NUM_RAY_TYPES = 2;
const unsigned STACK_SIZE    = 1024;

// Scene setup functions
void create_context( RTcontext* context, RTbuffer* output_buffer );
void create_materials( RTcontext context, RTmaterial material[] );
void create_geometry( RTcontext context, RTmaterial material[] );

// Helper functions
void               makeMaterialPrograms  ( RTcontext context, RTmaterial material, const char *filename, const char *ch_program_name, const char *ah_program_name );
RTvariable         makeMaterialVariable1i( RTcontext context, RTmaterial material, const char *name, int i1 );
RTvariable         makeMaterialVariable1f( RTcontext context, RTmaterial material, const char *name, float f1 );
RTvariable         makeMaterialVariable3f( RTcontext context, RTmaterial material, const char *name, float f1, float f2, float f3 );
RTgeometry         makeGeometry          ( RTcontext context, unsigned int primitives);
void               makeGeometryPrograms  ( RTcontext context, RTgeometry geometry, const char *filename, const char *is_program_name, const char *bb_program_name );
RTvariable         makeGeometryVariable4f( RTcontext context, RTgeometry geometry, const char *name, float f1, float f2, float f3, float f4 );
RTgeometryinstance makeGeometryInstance  ( RTcontext context, RTgeometry geometry, RTmaterial material );
RTgeometrygroup    makeGeometryGroup     ( RTcontext context, RTgeometryinstance instance, RTacceleration acceleration );
RTacceleration     makeAcceleration      ( RTcontext context, const char *builder );

void               printUsageAndExit( const std::string& argv0 );

// -----------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    RTcontext  context;
    RTbuffer   output_buffer;
    RTmaterial material[2];

    // Process command line options
    std::string outfile;
    for ( int i = 1; i < argc; ++i ) {
        std::string arg( argv[i] );
        if ( arg == "--help" || arg == "-h" ) {
            printUsageAndExit( argv[0] );
        } else if( arg == "--file" || arg == "-f" ) {
            if( i < argc-1 ) {
                outfile = argv[++i];
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( arg.substr( 0, 6 ) == "--dim=" ) {
            std::string dims_arg = arg.substr(6);
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        } else {
            std::cerr << "Unknown option: '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    sutil::initGlut(&argc, argv);

    create_context( &context, &output_buffer );
    create_materials( context, material );
    create_geometry( context, material );

    RT_CHECK_ERROR( rtContextValidate(context) );
    RT_CHECK_ERROR( rtContextLaunch2D(context,0,width,height) );
    if( outfile.empty() ) {
        sutil::displayBufferGlut( argv[0], output_buffer );
    } else {
        sutil::displayBufferPPM( outfile.c_str(), output_buffer );
    }

    RT_CHECK_ERROR( rtContextDestroy(context) );
    return 0;
}

// -----------------------------------------------------------------------------

void create_context( RTcontext* context, RTbuffer* output_buffer )
{
    // Context
    RT_CHECK_ERROR( rtContextCreate(context) );
    RT_CHECK_ERROR( rtContextSetEntryPointCount(*context, 1) );
    RT_CHECK_ERROR( rtContextSetRayTypeCount(*context, NUM_RAY_TYPES) );
    RT_CHECK_ERROR( rtContextSetStackSize(*context, STACK_SIZE) );

    // Output buffer
    RT_CHECK_ERROR( rtBufferCreate(*context,RT_BUFFER_OUTPUT,output_buffer) );
    RT_CHECK_ERROR( rtBufferSetFormat(*output_buffer,RT_FORMAT_UNSIGNED_BYTE4) );
    RT_CHECK_ERROR( rtBufferSetSize2D(*output_buffer,width,height) );

    // Ray generation program
    RTprogram raygen_program;
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "pinhole_camera.cu" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( *context, ptx, "pinhole_camera", &raygen_program ) );
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram(*context,0,raygen_program) );

    // Exception program
    RTprogram exception_program;
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( *context, ptx, "exception", &exception_program ) );
    RT_CHECK_ERROR( rtContextSetExceptionProgram(*context,0,exception_program) );

    // Miss program
    RTprogram miss_program;
    ptx = sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( *context, ptx, "miss", &miss_program ) );
    RT_CHECK_ERROR( rtContextSetMissProgram(*context,0,miss_program) );

    // System variables
    RTvariable var_output_buffer;
    RTvariable var_scene_epsilon;
    RTvariable var_max_depth;
    RTvariable var_badcolor;

    RT_CHECK_ERROR( rtContextDeclareVariable(*context,"output_buffer",&var_output_buffer) );
    RT_CHECK_ERROR( rtContextDeclareVariable(*context,"scene_epsilon",&var_scene_epsilon) );
    RT_CHECK_ERROR( rtContextDeclareVariable(*context,"max_depth",&var_max_depth) );
    RT_CHECK_ERROR( rtContextDeclareVariable(*context,"bad_color",&var_badcolor) );

    RT_CHECK_ERROR( rtVariableSetObject(var_output_buffer,*output_buffer) );
    RT_CHECK_ERROR( rtVariableSet1f(var_scene_epsilon,1e-3f) );
    RT_CHECK_ERROR( rtVariableSet1i(var_max_depth,10) );
    RT_CHECK_ERROR( rtVariableSet3f(var_badcolor,0.0f,1.0f,1.0f) );

    // Image background variables
    RTvariable var_bgcolor;
    RT_CHECK_ERROR( rtContextDeclareVariable(*context,"bg_color",&var_bgcolor) );
    RT_CHECK_ERROR( rtVariableSet3f(var_bgcolor,0.34f,0.55f,0.85f) );

    // Camera variables
    RTvariable var_eye;
    RTvariable var_U;
    RTvariable var_V;
    RTvariable var_W;

    RT_CHECK_ERROR( rtContextDeclareVariable(*context,"eye",&var_eye) );
    RT_CHECK_ERROR( rtContextDeclareVariable(*context,"U",&var_U) );
    RT_CHECK_ERROR( rtContextDeclareVariable(*context,"V",&var_V) );
    RT_CHECK_ERROR( rtContextDeclareVariable(*context,"W",&var_W) );

    float3 eye = make_float3(0.0f, 0.0f, 3.0f);
    float3 lookat = make_float3(0.0f, 0.0f, 0.0f);

    float3 up = make_float3(0, 1, 0);
    float hfov = 60;
    float3 lookdir = normalize(lookat-eye);
    float3 camera_u = cross(lookdir, up);
    float3 camera_v = cross(camera_u, lookdir);
    float ulen = tanf(hfov/2.0f*M_PIf/180.0f);
    camera_u = normalize(camera_u);
    camera_u *= ulen;
    float aspect_ratio = (float)width/(float)height;
    float vlen = ulen/aspect_ratio;
    camera_v = normalize(camera_v);
    camera_v *= vlen;

    RT_CHECK_ERROR( rtVariableSet3f(var_eye,eye.x,eye.y,eye.z) );
    RT_CHECK_ERROR( rtVariableSet3f(var_U,camera_u.x,camera_u.y,camera_u.z) );
    RT_CHECK_ERROR( rtVariableSet3f(var_V,camera_v.x,camera_v.y,camera_v.z) );
    RT_CHECK_ERROR( rtVariableSet3f(var_W,lookdir.x,lookdir.y,lookdir.z) );
}

// -----------------------------------------------------------------------------

void create_materials( RTcontext context, RTmaterial material[] )
{
    RT_CHECK_ERROR( rtMaterialCreate(context, material+0) );
    RT_CHECK_ERROR( rtMaterialCreate(context, material+1) );

    makeMaterialPrograms(context, material[0], "checkerboard.cu", "closest_hit_radiance", "any_hit_shadow");
    makeMaterialVariable3f(context, material[0], "tile_size",        1.0f, 1.0f, 1.0f);
    makeMaterialVariable3f(context, material[0], "tile_color_dark",  0.0f, 0.0f, 1.0f);
    makeMaterialVariable3f(context, material[0], "tile_color_light", 0.0f, 1.0f, 1.0f);
    makeMaterialVariable3f(context, material[0], "light_direction",  0.0f, 1.0f, 1.0f);

    makeMaterialPrograms(context, material[1], "checkerboard.cu", "closest_hit_radiance", "any_hit_shadow");
    makeMaterialVariable3f(context, material[1], "tile_size",        1.0f, 1.0f, 1.0f);
    makeMaterialVariable3f(context, material[1], "tile_color_dark",  1.0f, 0.0f, 0.0f);
    makeMaterialVariable3f(context, material[1], "tile_color_light", 1.0f, 1.0f, 0.0f);
    makeMaterialVariable3f(context, material[1], "light_direction",  0.0f, 1.0f, 1.0f);
}

// -----------------------------------------------------------------------------

void create_geometry( RTcontext context, RTmaterial material[] )
{
    /* Setup two geometry groups */

    // Geometry nodes (two spheres at same position, but with different radii)
    RTgeometry geometry[2];

    geometry[0] = makeGeometry(context, 1);
    makeGeometryPrograms(context, geometry[0], "sphere.cu", "intersect", "bounds");
    makeGeometryVariable4f(context, geometry[0], "sphere", 0.0f, 0.0f, 0.0f, 0.5f);

    geometry[1] = makeGeometry(context, 1);
    makeGeometryPrograms(context, geometry[1], "sphere.cu", "intersect", "bounds");
    makeGeometryVariable4f(context, geometry[1], "sphere", 0.0f, 0.0f, 0.0f, 1.0f);

    // Geometry instance nodes
    RTgeometryinstance instance[2];
    instance[0] = makeGeometryInstance( context, geometry[0], material[0] );
    instance[1] = makeGeometryInstance( context, geometry[1], material[1] );

    // Accelerations nodes
    RTacceleration acceleration[2];
    acceleration[0] = makeAcceleration( context, "NoAccel" );
    acceleration[1] = makeAcceleration( context, "NoAccel" );

    // Geometry group nodes
    RTgeometrygroup group[2];
    group[0] = makeGeometryGroup( context, instance[0], acceleration[0] );
    group[1] = makeGeometryGroup( context, instance[1], acceleration[1] );

    /* Setup selector as top objects */

    // Init selector node
    RTselector selector;
    RTprogram  stor_visit_program;
    RT_CHECK_ERROR( rtSelectorCreate(context,&selector) );
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "selector_example.cu" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "visit", &stor_visit_program) );
    RT_CHECK_ERROR( rtSelectorSetVisitProgram(selector,stor_visit_program) );
    RT_CHECK_ERROR( rtSelectorSetChildCount(selector,2) );
    RT_CHECK_ERROR( rtSelectorSetChild(selector, 0, group[0]) );
    RT_CHECK_ERROR( rtSelectorSetChild(selector, 1, group[1]) );

    // Attach selector to context as top object
    RTvariable var_group;
    RT_CHECK_ERROR( rtContextDeclareVariable(context,"top_object",&var_group) );
    RT_CHECK_ERROR( rtVariableSetObject(var_group, selector) );
}

// -----------------------------------------------------------------------------

void makeMaterialPrograms( RTcontext context, RTmaterial material, const char *filename, 
        const char *ch_program_name,
        const char *ah_program_name )
{
    RTprogram ch_program;
    RTprogram ah_program;

    const char *ptx = sutil::getPtxString( SAMPLE_NAME, filename );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, ch_program_name, &ch_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, ah_program_name, &ah_program ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram(material, 0, ch_program) );
    RT_CHECK_ERROR( rtMaterialSetAnyHitProgram(material, 1, ah_program) );
}

// -----------------------------------------------------------------------------

RTvariable makeMaterialVariable1i( RTcontext context, RTmaterial material, const char *name, int i1 )
{
    RTvariable variable;

    RT_CHECK_ERROR( rtMaterialDeclareVariable(material, name, &variable) );
    RT_CHECK_ERROR( rtVariableSet1i(variable, i1) );

    return variable;
}

RTvariable makeMaterialVariable1f( RTcontext context, RTmaterial material, const char *name, float f1 )
{
    RTvariable variable;

    RT_CHECK_ERROR( rtMaterialDeclareVariable(material, name, &variable) );
    RT_CHECK_ERROR( rtVariableSet1f(variable, f1) );

    return variable;
}

RTvariable makeMaterialVariable3f( RTcontext context, RTmaterial material, const char *name, float f1, float f2, float f3 )
{
    RTvariable variable;

    RT_CHECK_ERROR( rtMaterialDeclareVariable(material, name, &variable) );
    RT_CHECK_ERROR( rtVariableSet3f(variable, f1, f2, f3) );

    return variable;
}

// -----------------------------------------------------------------------------

RTgeometry makeGeometry( RTcontext context, unsigned int primitives)
{
    RTgeometry geometry;

    RT_CHECK_ERROR( rtGeometryCreate(context, &geometry) );
    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount(geometry, primitives) );

    return geometry;
}

// -----------------------------------------------------------------------------

void makeGeometryPrograms( RTcontext context, RTgeometry geometry, const char *filename, 
        const char *is_program_name,
        const char *bb_program_name )
{
    RTprogram is_program;
    RTprogram bb_program;

    const char *ptx = sutil::getPtxString( SAMPLE_NAME, filename );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, is_program_name, &is_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, bb_program_name, &bb_program ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram(geometry, is_program) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram(geometry, bb_program) );
}

// -----------------------------------------------------------------------------

RTvariable makeGeometryVariable4f( RTcontext context, RTgeometry geometry, const char *name, float f1, float f2, float f3, float f4 )
{
    RTvariable variable;

    RT_CHECK_ERROR( rtGeometryDeclareVariable(geometry, name, &variable) );
    RT_CHECK_ERROR( rtVariableSet4f(variable, f1, f2, f3, f4) );

    return variable;
}

// -----------------------------------------------------------------------------

RTgeometryinstance makeGeometryInstance( RTcontext context, RTgeometry geometry, RTmaterial material )
{
    RTgeometryinstance instance;

    RT_CHECK_ERROR( rtGeometryInstanceCreate(context, &instance) );
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry(instance, geometry) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount(instance, 1) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial(instance, 0, material) );

    return instance;
}

// -----------------------------------------------------------------------------

RTgeometrygroup makeGeometryGroup( RTcontext context, RTgeometryinstance instance, RTacceleration acceleration )
{
    RTgeometrygroup geometrygroup;

    RT_CHECK_ERROR( rtGeometryGroupCreate(context,& geometrygroup) );
    RT_CHECK_ERROR( rtGeometryGroupSetChildCount(geometrygroup, 1) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild(geometrygroup, 0, instance) );
    RT_CHECK_ERROR( rtGeometryGroupSetAcceleration(geometrygroup, acceleration) );

    return geometrygroup;
}

// -----------------------------------------------------------------------------

RTacceleration makeAcceleration( RTcontext context, const char *builder )
{
    RTacceleration acceleration;

    RT_CHECK_ERROR( rtAccelerationCreate(context, &acceleration) );
    RT_CHECK_ERROR( rtAccelerationSetBuilder(acceleration, builder) );

    return acceleration;
}

// -----------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "Usage  : " << argv0 << " <options>\n"
        << "Options: --help  | -h             Print this usage message\n"
        << "         --file  | -f <filename>  Specify file for image output\n"
        << "         --dim=<width>x<height>   Set image dimensions; defaults to 1024x768\n";
    exit(1); 
}
