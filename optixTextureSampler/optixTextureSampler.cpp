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
 *  optixTextureSampler.cpp -- Draws a texture to the screen, where the texture is created
 *                 by the RTAPI.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <optix.h>
#include <sutil.h>

const char* const SAMPLE_NAME = "optixTextureSampler";

void printUsageAndExit( const char* argv0 );

int main(int argc, char** argv)
{
    RTcontext context = 0;

    try
    {
        RTbuffer  tex_buffer;
        RTbuffer  result_buffer;
        RTprogram draw_tex;
        RTprogram exception;
        RTtexturesampler tex_sampler;
        RTvariable tex_sampler_var;
        RTvariable result_buffer_var;
        void* tex_data_ptr;
        unsigned char* tex_data;
        int k;
        unsigned int i,j;

        unsigned int tex_width  = 64;
        unsigned int tex_height = 64;

        int trace_width  = 512;
        int trace_height = 384;

        char outfile[512];

        outfile[0] = '\0';

        for( k = 1; k < argc; ++k ) {
            if( strcmp( argv[k], "--help" ) == 0 || strcmp( argv[k], "-h" ) == 0 ) {
                printUsageAndExit( argv[0] );
            } else if( strcmp( argv[k], "--file" ) == 0 || strcmp( argv[k], "-f" ) == 0 ) {
                if( k < argc-1 ) {
                    strcpy( outfile, argv[++k] );
                } else {
                    printUsageAndExit( argv[0] );
                }
            } else if ( strncmp( argv[k], "--dim=", 6 ) == 0 ) {
                const char *dims_arg = &argv[k][6];
                sutil::parseDimensions( dims_arg, trace_width, trace_height );
            } else {
                fprintf( stderr, "Unknown option '%s'\n", argv[k] );
                printUsageAndExit( argv[0] );
            }
        }

        /* Process command line args */
        if( strlen( outfile ) == 0 ) {
            sutil::initGlut(&argc, argv);
        }

        /* Create the context */
        RT_CHECK_ERROR( rtContextCreate( &context ) );

        /* Create the buffer that represents the texture data */
        RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &tex_buffer ) );
        RT_CHECK_ERROR( rtBufferSetFormat( tex_buffer, RT_FORMAT_UNSIGNED_BYTE4 ) );
        RT_CHECK_ERROR( rtBufferSetSize2D( tex_buffer, tex_width, tex_height ) );
        RT_CHECK_ERROR( rtBufferMap( tex_buffer, &tex_data_ptr ) );
        tex_data = (unsigned char*)tex_data_ptr;
        for(j = 0; j < tex_height; ++j) {
            for(i = 0; i < tex_width; ++i) {
                *tex_data++ = (unsigned char)((i+j)/((float)(tex_width+tex_height))*255); /* R */
                *tex_data++ = (unsigned char)(i/((float)tex_width)*255);                  /* G */
                *tex_data++ = (unsigned char)(j/((float)tex_height)*255);                 /* B */
                *tex_data++ = 255;                                                        /* A */
            }
        }
        RT_CHECK_ERROR( rtBufferUnmap( tex_buffer ) );

        /* Create the texture sampler */
        RT_CHECK_ERROR( rtTextureSamplerCreate( context, &tex_sampler ) );
        RT_CHECK_ERROR( rtTextureSamplerSetWrapMode( tex_sampler, 0, RT_WRAP_CLAMP_TO_EDGE ) );
        RT_CHECK_ERROR( rtTextureSamplerSetWrapMode( tex_sampler, 1, RT_WRAP_CLAMP_TO_EDGE ) );
        RT_CHECK_ERROR( rtTextureSamplerSetFilteringModes( tex_sampler, RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE ) );
        RT_CHECK_ERROR( rtTextureSamplerSetIndexingMode( tex_sampler, RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ) );
        RT_CHECK_ERROR( rtTextureSamplerSetReadMode( tex_sampler, RT_TEXTURE_READ_NORMALIZED_FLOAT ) );
        RT_CHECK_ERROR( rtTextureSamplerSetMaxAnisotropy( tex_sampler, 1.f ) );
        RT_CHECK_ERROR( rtTextureSamplerSetMipLevelCount( tex_sampler, 1 ) );
        RT_CHECK_ERROR( rtTextureSamplerSetArraySize( tex_sampler, 1 ) );
        RT_CHECK_ERROR( rtTextureSamplerSetBuffer( tex_sampler, 0, 0, tex_buffer ) );
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "input_texture" , &tex_sampler_var) );
        RT_CHECK_ERROR( rtVariableSetObject( tex_sampler_var, tex_sampler ) );

        /* Create the output buffer */
        RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_OUTPUT, &result_buffer ) );
        RT_CHECK_ERROR( rtBufferSetFormat( result_buffer, RT_FORMAT_FLOAT4 ) );
        RT_CHECK_ERROR( rtBufferSetSize2D( result_buffer, trace_width, trace_height ) );
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "result_buffer" , &result_buffer_var) );
        RT_CHECK_ERROR( rtVariableSetObject( result_buffer_var, result_buffer ) );

        /* Create ray generation program */
        const char *ptx = sutil::getPtxString( SAMPLE_NAME, "draw_texture.cu" );
        RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "draw_texture", &draw_tex ) );

        /* Create exception program */
        RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "exception", &exception ) );

        /* Setup context programs */
        RT_CHECK_ERROR( rtContextSetEntryPointCount( context, 1 ) );
        RT_CHECK_ERROR( rtContextSetRayTypeCount( context, 1 ) );
        RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, 0, draw_tex ) );
        RT_CHECK_ERROR( rtContextSetExceptionProgram( context, 0, exception ) );

        /* Trace */
        RT_CHECK_ERROR( rtContextValidate( context ) );
        RT_CHECK_ERROR( rtContextLaunch2D( context, 0, trace_width, trace_height ) );

        /* Display image */
        if( strlen( outfile ) == 0 ) {
            sutil::displayBufferGlut( argv[0], result_buffer );
        } else {
            sutil::displayBufferPPM( outfile, result_buffer, false );
        }

        /* Cleanup */
        RT_CHECK_ERROR( rtContextDestroy( context ) );

        return( 0 );
    } SUTIL_CATCH( context )

}

void printUsageAndExit( const char* argv0 )
{
    fprintf( stderr, "Usage  : %s [options]\n", argv0 );
    fprintf( stderr, "Options: --help | -h             Print this usage message\n" );
    fprintf( stderr, "Options: --file | -f <filename>      Specify file for image output\n" );
    fprintf( stderr, "         --dim=<width>x<height>  Set image dimensions; defaults to 512x384\n" );
    exit(1);
}
