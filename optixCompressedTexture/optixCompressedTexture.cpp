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
 *  optixCompressedTexture.cpp -- Draws a compressed texture to the screen.
 */


#include <optix.h>
#include <sutil.h>

#include <cstdlib>
#include <cstring>

#include <fstream>


const char* const SAMPLE_NAME = "optixCompressedTexture";

void printUsageAndExit( const char* argv0 );

int main( int argc, char** argv )
{
    RTcontext context = 0;

    try
    {
        int trace_width  = 1920;
        int trace_height = 1080;

        std::string outfile;

        for( int i = 1; i < argc; ++i )
        {
            if( !std::strcmp( argv[i], "--help" ) || !std::strcmp( argv[i], "-h" ) )
            {
                printUsageAndExit( argv[0] );
            }
            else if( !std::strcmp( argv[i], "--file" ) || !std::strcmp( argv[i], "-f" ) )
            {
                if( i < argc - 1 )
                {
                    outfile = argv[++i];
                }
                else
                {
                    printUsageAndExit( argv[0] );
                }
            }
            else if( std::strncmp( argv[i], "--dim=", 6 ) == 0 )
            {
                const char* dims_arg = &argv[i][6];
                sutil::parseDimensions( dims_arg, trace_width, trace_height );
            }
            else
            {
                std::fprintf( stderr, "Unknown option '%s'\n", argv[i] );
                printUsageAndExit( argv[0] );
            }
        }


        /* Process command line args */
        sutil::initGlut( &argc, argv );

        /* Create the context */
        RT_CHECK_ERROR( rtContextCreate( &context ) );

        /* Create the buffer that represents the texture data */
        const unsigned int tex_width  = 4096 / 4;
        const unsigned int tex_height = 2048 / 4;
        const char*        path       = "/data/pano.bc6h";
        const RTformat     format     = RT_FORMAT_UNSIGNED_BC6H;

        void*    tex_data_ptr = 0;
        RTbuffer tex_buffer   = 0;

        RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &tex_buffer ) );
        RT_CHECK_ERROR( rtBufferSetFormat( tex_buffer, format ) );
        RT_CHECK_ERROR( rtBufferSetSize2D( tex_buffer, tex_width, tex_height ) );
        RT_CHECK_ERROR( rtBufferMap( tex_buffer, &tex_data_ptr ) );

        std::ifstream file( ( std::string( sutil::samplesDir() ) + path ).c_str(), std::ios::binary | std::ios::ate );
        if( !file.good() )
        {
            RT_CHECK_ERROR( RT_ERROR_INVALID_VALUE );
        }
        const std::size_t size = file.tellg();
        file.seekg( 0, std::ios::beg );
        file.read( (char*)tex_data_ptr, size );

        RT_CHECK_ERROR( rtBufferUnmap( tex_buffer ) );

        /* Create the texture sampler */
        RTtexturesampler tex_sampler = 0;
        RT_CHECK_ERROR( rtTextureSamplerCreate( context, &tex_sampler ) );
        RT_CHECK_ERROR( rtTextureSamplerSetWrapMode( tex_sampler, 0, RT_WRAP_CLAMP_TO_EDGE ) );
        RT_CHECK_ERROR( rtTextureSamplerSetWrapMode( tex_sampler, 1, RT_WRAP_CLAMP_TO_EDGE ) );
        RT_CHECK_ERROR( rtTextureSamplerSetFilteringModes( tex_sampler, RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE ) );
        RT_CHECK_ERROR( rtTextureSamplerSetIndexingMode( tex_sampler, RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ) );
        RT_CHECK_ERROR( rtTextureSamplerSetReadMode( tex_sampler, RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB ) );
        RT_CHECK_ERROR( rtTextureSamplerSetMaxAnisotropy( tex_sampler, 1.f ) );
        RT_CHECK_ERROR( rtTextureSamplerSetMipLevelCount( tex_sampler, 1 ) );
        RT_CHECK_ERROR( rtTextureSamplerSetArraySize( tex_sampler, 1 ) );
        RT_CHECK_ERROR( rtTextureSamplerSetBuffer( tex_sampler, 0, 0, tex_buffer ) );

        int        tex_id     = 0;
        RTvariable tex_id_var = 0;
        RT_CHECK_ERROR( rtTextureSamplerGetId( tex_sampler, &tex_id ) );
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "input_texture", &tex_id_var ) );
        RT_CHECK_ERROR( rtVariableSet1i( tex_id_var, tex_id ) );

        /* Create the output buffer */
        RTbuffer   result_buffer     = 0;
        RTvariable result_buffer_var = 0;
        RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_OUTPUT, &result_buffer ) );
        RT_CHECK_ERROR( rtBufferSetFormat( result_buffer, RT_FORMAT_FLOAT4 ) );
        RT_CHECK_ERROR( rtBufferSetSize2D( result_buffer, trace_width, trace_height ) );
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "result_buffer", &result_buffer_var ) );
        RT_CHECK_ERROR( rtVariableSetObject( result_buffer_var, result_buffer ) );

        /* Create ray generation program */
        RTprogram   draw_tex = 0;
        const char* ptx      = sutil::getPtxString( SAMPLE_NAME, "draw_texture.cu" );
        RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "draw_texture", &draw_tex ) );

        /* Create exception program */
        RTprogram exception = 0;
        RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "exception", &exception ) );

        /* Setup context programs */
        RT_CHECK_ERROR( rtContextSetEntryPointCount( context, 1 ) );
        RT_CHECK_ERROR( rtContextSetRayTypeCount( context, 1 ) );
        RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, 0, draw_tex ) );
        RT_CHECK_ERROR( rtContextSetExceptionProgram( context, 0, exception ) );

        /* Render */
        RT_CHECK_ERROR( rtContextValidate( context ) );
        RT_CHECK_ERROR( rtContextLaunch2D( context, 0, trace_width, trace_height ) );

        /* Display image */
        if( outfile.empty() )
        {
            sutil::displayBufferGlut( argv[0], result_buffer );
        }
        else
        {
            sutil::displayBufferPPM( outfile.c_str(), result_buffer, false );
        }

        /* Cleanup */
        RT_CHECK_ERROR( rtContextDestroy( context ) );

        return ( 0 );
    }
    SUTIL_CATCH( context )
}

void printUsageAndExit( const char* argv0 )
{
    std::fprintf( stderr, "Usage  : %s [options]\n", argv0 );
    std::fprintf( stderr, "Options: --help | -h             Print this usage message\n" );
    std::fprintf( stderr, "Options: --file | -f <filename>  Specify file for image output\n" );
    std::fprintf( stderr, "         --dim=<width>x<height>  Set image dimensions; defaults to 1920x1080\n" );
    std::exit( 1 );
}
