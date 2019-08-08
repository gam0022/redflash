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
// optixPrimitiveIndexOffsets -- Demonstrates usage of Geometry primitive index
//                          offsets.
//
//-----------------------------------------------------------------------------


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <sutil.h>
#include "common.h"
#include <iostream>
#include <cstdlib>
#include <cstring>

using namespace optix;

const char* const SAMPLE_NAME = "optixPrimitiveIndexOffsets";

int width  = 720;
int height = 480;
const int NUM_TETS = 4u;

//-----------------------------------------------------------------------------
// 
// Helpers 
//
//-----------------------------------------------------------------------------

Context createContext() 
{
    // Set up context
    Context context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 1200 );
    context->setMaxTraceDepth( 2 );

    context["max_depth"]->setInt( 5 );
    context["scene_epsilon"]->setFloat( 1.e-4f );

    Variable output_buffer = context["output_buffer"];
    Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height );
    output_buffer->set(buffer);

    // Camera
    float3 cam_eye = { 0.0f, 1.5f, -6.5f };
    float3 lookat  = { 0.0f, 0.0f,  0.0f };
    float3 up      = { 0.0f, 1.0f, 0.0f };
    float  hfov    = 81.786782f;
    float  aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables( 
            cam_eye, lookat, up, hfov, aspect_ratio,
            camera_u, camera_v, camera_w );

    context["eye"]->setFloat( cam_eye );
    context["U"]->setFloat( camera_u );
    context["V"]->setFloat( camera_v );
    context["W"]->setFloat( camera_w );

    // Ray generation program
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "pinhole_camera.cu" );
    Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    context->setMissProgram( 0, context->createProgramFromPTXString( sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" ), "miss" ) );
    context["bg_color"]->setFloat( 
            make_float3(108.0f/255.0f, 166.0f/255.0f, 205.0f/255.0f) * 0.5f );

    return context;
}

void fillGeometryBuffers( float3* verts, int3* indices, unsigned* mats )
{
    //
    // Create tets 
    //
    for( int i = 0; i < NUM_TETS; ++i )
    {
        mats[i*4+0] = 0u;
        mats[i*4+1] = 0u;
        mats[i*4+2] = 0u;
        mats[i*4+3] = 0u;
        
        const float3 offset = make_float3( -3.0f + 2.0f * i, 0.0f, 0.0f );
        verts[ i*4+0 ] = make_float3(  1.0f,  0.0f, -1.0f/(sqrtf( 2.0f ) ) )+offset;
        verts[ i*4+1 ] = make_float3( -1.0f,  0.0f, -1.0f/(sqrtf( 2.0f ) ) )+offset;
        verts[ i*4+2 ] = make_float3(  0.0f,  1.0f,  1.0f/(sqrtf( 2.0f ) ) )+offset;
        verts[ i*4+3 ] = make_float3(  0.0f, -1.0f,  1.0f/(sqrtf( 2.0f ) ) )+offset;
  
        // Load our triangle data, starting at primitive index offset
        indices[ i*4+0 ] = make_int3( i*4+0, i*4+1, i*4+2 );
        indices[ i*4+1 ] = make_int3( i*4+1, i*4+3, i*4+2 );
        indices[ i*4+2 ] = make_int3( i*4+3, i*4+0, i*4+2 );
        indices[ i*4+3 ] = make_int3( i*4+1, i*4+0, i*4+3 );
    }
  
    //
    // Create floor mesh consisting of two triangles
    //
    mats[NUM_TETS*4+0] = 0u;
    mats[NUM_TETS*4+1] = 0u;
    
    verts[ NUM_TETS*4+0 ] = make_float3(  20.0f, -1.25, -20.0f );
    verts[ NUM_TETS*4+1 ] = make_float3( -20.0f, -1.25, -20.0f );
    verts[ NUM_TETS*4+2 ] = make_float3( -20.0f, -1.25,  20.0f );
    verts[ NUM_TETS*4+3 ] = make_float3(  20.0f, -1.25,  20.0f );
  
    // Load our triangle data, starting at primitive index offset
    indices[ NUM_TETS*4+0 ] = make_int3( NUM_TETS*4+0, NUM_TETS*4+1, NUM_TETS*4+2 );
    indices[ NUM_TETS*4+1 ] = make_int3( NUM_TETS*4+0, NUM_TETS*4+2, NUM_TETS*4+3 );

}


// Put all geometry in a single geometry group
Group createSingleGeometryGroup( Context context, Program mesh_intersect, Program mesh_bounds, Material matl, const std::string& builder )
{
    Group group = context->createGroup();
    group->setAcceleration( context->createAcceleration( builder.c_str() ) );

    //
    // Create geometry instances, 1 per tetrahedron
    //

    std::vector<GeometryInstance> gis;
    for( int i = 0; i < NUM_TETS; ++i )
    {
        Geometry mesh = context->createGeometry();
        mesh->setIntersectionProgram( mesh_intersect ); 
        mesh->setBoundingBoxProgram( mesh_bounds ); 
        mesh->setPrimitiveCount( 4u );  // 4 triangles per tet
        mesh->setPrimitiveIndexOffset( i*4 ); // Set our beginning prim_idx offset

        gis.push_back( context->createGeometryInstance( mesh, &matl, &matl+1 ) );
    }

    //
    // Create geometry instance for floor
    //

    Geometry floor = context->createGeometry();
    floor->setIntersectionProgram( mesh_intersect ); 
    floor->setBoundingBoxProgram( mesh_bounds ); 
    floor->setPrimitiveCount( 2u );
    floor->setPrimitiveIndexOffset( NUM_TETS*4 ); // Set our beginning prim_idx offset
   
    gis.push_back( context->createGeometryInstance( floor, &matl, &matl+1 ) );

    // Add all geometry instances to a geometry group
    GeometryGroup geometrygroup = context->createGeometryGroup();
    geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
    for( unsigned i = 0u; i < gis.size(); ++i ) {
        geometrygroup->setChild( i, gis[i] );
    }

    Acceleration accel = context->createAcceleration( builder.c_str() );
    geometrygroup->setAcceleration( accel );
    
    group->setChildCount( 1 );
    group->setChild( 0, geometrygroup );

    return group;
}

// Put each tetrahedron in its own geometry group
Group createMultipleGeometryGroups( Context context, Program mesh_intersect, Program mesh_bounds, Material matl, const std::string& builder )
{
    Group group = context->createGroup();
    group->setAcceleration( context->createAcceleration( builder.c_str() ) );

    //
    // Create geometry groups, 1 per tetrahedron
    //

    std::vector<GeometryGroup> ggs;
    for( int i = 0; i < NUM_TETS; ++i )
    {
        Geometry mesh = context->createGeometry();
        mesh->setIntersectionProgram( mesh_intersect ); 
        mesh->setBoundingBoxProgram( mesh_bounds ); 
        mesh->setPrimitiveCount( 4u );  // 4 triangles per tet
        mesh->setPrimitiveIndexOffset( i*4 ); // Set our beginning prim_idx offset

        GeometryInstance gi = context->createGeometryInstance( mesh, &matl, &matl+1 );

        GeometryGroup gg = context->createGeometryGroup();
        gg->setAcceleration( context->createAcceleration( builder.c_str() ) );
        gg->setChildCount( 1 );
        gg->setChild( 0, gi );
        ggs.push_back( gg );
    }

    //
    // Create geometry group for floor
    //

    Geometry floor = context->createGeometry();
    floor->setIntersectionProgram( mesh_intersect ); 
    floor->setBoundingBoxProgram( mesh_bounds ); 
    floor->setPrimitiveCount( 2u );
    floor->setPrimitiveIndexOffset( NUM_TETS*4 ); // Set our beginning prim_idx offset
   
    GeometryInstance gi = context->createGeometryInstance( floor, &matl, &matl+1 );
    GeometryGroup gg = context->createGeometryGroup();
    gg->setAcceleration( context->createAcceleration( builder.c_str() ) );
    gg->setChildCount( 1 );
    gg->setChild( 0, gi );
    ggs.push_back( gg );

    // Add all geometry groups to top group
    
    group->setChildCount( static_cast<unsigned int>(ggs.size()) );
    for( unsigned i = 0u; i < ggs.size(); ++i ) {
        group->setChild( i, ggs[i] );
    }

    return group;
}

void createScene( Context context, const std::string& builder, bool use_multiple_geometry_groups )
{
    //
    // Single material
    //
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "phong.cu" );
    Program phong_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program phong_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
    Material matl = context->createMaterial();
    matl->setClosestHitProgram( 0, phong_ch);
    matl->setAnyHitProgram( 1, phong_ah );
    matl["Kd"           ]->setFloat( 0.7f, 0.7f, 0.7f );
    matl["Ka"           ]->setFloat( 1.0f, 1.0f, 1.0f );
    matl["Kr"           ]->setFloat( 0.0f, 0.0f, 0.0f );
    matl["phong_exp"    ]->setFloat( 1.0f );

    //
    // These buffers will be shared across all Geometries
    //
    ptx = sutil::getPtxString( SAMPLE_NAME, "triangle_mesh.cu" );
    Program mesh_intersect = context->createProgramFromPTXString( ptx, "mesh_intersect" );
    Program mesh_bounds    = context->createProgramFromPTXString( ptx, "mesh_bounds" );
    Buffer verts_buffer    = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, NUM_TETS*4 + 4 );
    Buffer index_buffer    = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, NUM_TETS*4 + 2 );
    Buffer matl_buffer     = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, NUM_TETS*4 + 2);

    float3*   verts   = reinterpret_cast<float3*>( verts_buffer->map() );
    int3*     indices = reinterpret_cast<int3*>( index_buffer->map() );
    unsigned* mats    = static_cast<unsigned*>( matl_buffer->map() );

    // unused buffers
    Buffer nbuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0 );
    Buffer tbuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 0 );

    //
    // Attach all of the buffers at the Context level since they are shared
    //
    context[ "vertex_buffer"   ]->setBuffer( verts_buffer );
    context[ "normal_buffer"   ]->setBuffer( nbuffer );
    context[ "texcoord_buffer" ]->setBuffer( tbuffer );
    context[ "index_buffer"    ]->setBuffer( index_buffer );
    context[ "material_buffer" ]->setBuffer( matl_buffer );

    fillGeometryBuffers( verts, indices, mats );

    //
    // Add geometry to the node graph
    //

    Group group;
    if ( use_multiple_geometry_groups )
        group = createMultipleGeometryGroups( context, mesh_intersect, mesh_bounds, matl, builder );
    else
        group = createSingleGeometryGroup( context, mesh_intersect, mesh_bounds, matl, builder );

    context[ "top_object"   ]->set( group );
    context[ "top_shadower" ]->set( group );
    
    verts_buffer->unmap();
    index_buffer->unmap();
    matl_buffer->unmap();
    
    // Setup lights
    context["ambient_light_color"]->setFloat(0.1f,0.1f,0.1f);
    BasicLight lights[] = { 
        { { 0.0f, 8.0f, -5.0f }, { .4f, .4f, .4f }, 1 },
        { { 5.0f, 8.0f,  0.0f }, { .4f, .4f, .4f }, 1 }
    };

    Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(BasicLight));
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    context["lights"]->set(light_buffer);
}


void printUsageAndExit( const std::string& argv0 )
{
    std::cerr
        << "Usage  : " << argv0 << " [options]\n"
        << "Options:\n"
        << "  --help | -h             Print this usage message\n"
        << "  --build <name>          Acceleration structure builder\n"
        << "  --file | -f <filename>  Specify file for image output\n"
        << "  --multiple-groups       Put geometry in multiple geometry groups.  Default is single geometry group.\n"
        << "  --dim=<width>x<height>  Set image dimensions; defaults to " << width << "x" << height << "\n";

    exit(1);
}

int main( int argc, char** argv )
{

    std::string outfile;
    std::string builder( "NoAccel" );
    bool use_multiple_geometry_groups = false;

    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if( arg == "--build" )
        {
            if ( i == argc-1 )
                printUsageAndExit( argv[0] );
            builder = argv[++i];
        } else if ( arg.substr( 0, 6 ) == "--dim=" ) {
            std::string dims_arg = arg.substr(6);
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        } else if( arg == "--file" || arg == "-f" ) {
            if( i < argc-1 ) {
                outfile = argv[++i];
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( arg == "--multiple-groups" )
        {
            use_multiple_geometry_groups = true;
        } else if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    if( outfile.empty() ) {
        sutil::initGlut( &argc, argv );
    }

    try {

        Context context = createContext();
        createScene( context, builder, use_multiple_geometry_groups );

        context->validate();
        context->launch( 0, width, height );

        // DisplayImage
        if( outfile.empty() ) {
            sutil::displayBufferGlut( argv[0], context["output_buffer"]->getBuffer() );
        } else {
            sutil::displayBufferPPM( outfile.c_str(), context["output_buffer"]->getBuffer() );
        }

    } catch( Exception& e ){
        sutil::reportErrorMessage( e.getErrorString().c_str() );
        exit(1);
    }

    return 0;
}

