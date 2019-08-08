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
// optixSSIMPredictor: example of using ssim prediction post processing stage
//
//-----------------------------------------------------------------------------

#include <stdlib.h>
#include <fstream>
#include <optixu/optixu_math_stream_namespace.h>
#include <sutil.h>
#include <math.h>
#include <stdint.h>

using namespace optix;


//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

const char* const SAMPLE_NAME = "optixSSIMPredictor";

// Post-processing
CommandList         commandList;
PostprocessingStage ssimPredictorStage;


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------


// ceilDiv(x,y) returns the integer ceil of x/y
// https://stackoverflow.com/a/14878734/1424242
int ceilDiv( int x, int y )
{
    return x / y + ( x % y != 0 );
}


void ssimPredictorReportCallback( int lvl, const char* tag, const char* msg, void* cbdata )
{
    if( std::string( "DLSSIMPREDICTOR" ) == tag )
        std::cerr << "[" << tag << "] " << msg;
}


void initContext( Context context, Buffer inputBuffer, int64_t maxMemBytes )
{
    // Setup a pipeline with 2 steps:
    // Step 1 - run ssim prediction on the input buffer
    // Step 2 - copy the ssim output to an output buffer for file save

    context->setUsageReportCallback( ssimPredictorReportCallback, 2, NULL );

    const char* ptx = sutil::getPtxString( SAMPLE_NAME, "optixSSIMPredictor.cu" );
    commandList     = context->createCommandList();

    context->setRayTypeCount( 1 ) ;
    context->setEntryPointCount( 1 );

    RTsize in_width, in_height;
    inputBuffer->getSize( in_width, in_height );

    // Step 1 - run ssim prediction on the input buffer
    // Allocate some space for the SSIM prediction output,
    // which will be 1/16 scale from the input image, rounded up
    const int heatmap_shrink_factor = 16;

    int ssim_width  = ceilDiv( (int)in_width, heatmap_shrink_factor );
    int ssim_height = ceilDiv( (int)in_height, heatmap_shrink_factor );

    Buffer ssimPredictedBuffer = sutil::createInputOutputBuffer( context, RT_FORMAT_FLOAT, ssim_width, ssim_height, /* use_pbo = */ false );

    ssimPredictorStage = context->createBuiltinPostProcessingStage( "DLSSIMPredictor" );
    ssimPredictorStage->declareVariable( "input_buffer" )->set( inputBuffer );
    ssimPredictorStage->declareVariable( "output_buffer" )->set( ssimPredictedBuffer );
    ssimPredictorStage->declareVariable( "maxmem" )->setLongLong( maxMemBytes );

    commandList->appendPostprocessingStage( ssimPredictorStage, (int)in_width, (int)in_height );  // run ssim prediction

    // Step 2 - copy the ssim output to an output buffer
    // The SSIM output buffer is a single float channel, so can't be used to output a standard rgb image file
    // Let's copy the float value to a grayscale float4 buffer for output
    Buffer ssimDisplayBuffer = sutil::createOutputBuffer( context, RT_FORMAT_FLOAT4, ssim_width, ssim_height, /* use_pbo = */ false );

    context["ssim_input_buffer"]->set( ssimPredictedBuffer );
    context["ssim_output_buffer"]->set( ssimDisplayBuffer );

    context->setRayGenerationProgram( 0, context->createProgramFromPTXString( ptx, "ssim_copy_float_to_float4" ) );
    commandList->appendLaunch( 0, ssim_width, ssim_height );  // copy ssim results to output buffer for output

    // Lock the command list
    commandList->finalize();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr << "App Options:\n"
                 "  -h  | --help                 Print this usage message and exit.\n"
                 "  -i  | --input  <path.ppm>    Specify input .ppm file path.\n"
                 "  -f  | --file   <path.ppm>    Specify output .ppm file path.\n"
                 "  -m  | --maxmem <max mem MiB> Specify memory limit in MiB.\n"
              << std::endl;

    exit( 1 );
}


bool endsWith( std::string const& text, std::string const& token )
{
    const std::string::size_type token_len = token.length();
    const std::string::size_type text_len  = text.length();
    return ( text_len >= token_len ) && ( 0 == text.compare( text_len - token_len, token_len, token ) );
}

bool fileExists( std::string filename )
{
    return std::ifstream( filename.c_str() ).good();
}

int main( int argc, char** argv )
{
    std::string in_file;
    std::string out_file;
    float maxMemMiB = 0.f;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "-i" || arg == "--input" )
        {
            if( i == argc - 1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            in_file = argv[++i];
            if( !endsWith( in_file, ".ppm" ) )
            {
                std::cerr << "Input file '" << in_file << "' should be .ppm.\n";
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg == "-f" || arg == "--file" )
        {
            if( i == argc - 1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            out_file = argv[++i];
            if( !endsWith( out_file, ".ppm" ) )
            {
                std::cerr << "Output file '" << out_file << "' should be .ppm.\n";
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg == "-m" )
        {
            if( i == argc - 1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            const char* maxMemMiBStr = argv[++i];
            maxMemMiB                = float( atof( maxMemMiBStr ) );
            if( maxMemMiB < 0.f || maxMemMiB > 1000000000.f || maxMemMiB != maxMemMiB )
            {
                std::cerr << "I didn't understand memory limit '" << maxMemMiBStr << "'. Please use a positive number like '100'.\n";
                printUsageAndExit( argv[0] );
            }
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    // Make sure user specified input & output files
    if( in_file.empty() )
    {
        in_file = std::string( sutil::samplesDir() ) + "/data/optixSSIMPredictor_test.ppm";
        std::cerr << "No input file given; defaulting to " << in_file << ".\n";
    }

    if( out_file.empty() )
    {
        out_file = std::string( "out.ppm" );
        std::cerr << "No input file given; defaulting to " << out_file << ".\n";
    }

    // Make sure input file exists
    if( !fileExists( in_file ) )
    {
        std::cerr << "Unable to read input file " << in_file << "\n";
        printUsageAndExit( argv[0] );
    }

    Context context = 0;
    try
    {
        context = Context::create();
        optix::Buffer inputBuffer = sutil::loadPPMFloat4Buffer( context, in_file );
        if( !inputBuffer )
        {
            std::cerr << "Unable to read input file " << in_file << "\n";
            printUsageAndExit( argv[0] );
        }

        const float bytesPerMebibyte = float(1 << 20);
        const int64_t maxMemBytes = int64_t(maxMemMiB * bytesPerMebibyte);
        initContext( context, inputBuffer, maxMemBytes );
        context->validate();
        commandList->execute();
        sutil::displayBufferPPM( out_file.c_str(), context["ssim_output_buffer"]->getBuffer() );
        std::cout << "Wrote " << out_file << "\n";
        context->destroy();
        return 0;
    }
    SUTIL_CATCH( context->get() )
}
