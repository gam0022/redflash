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
//  mdl_helper.cpp - Helper class to compile MDL materials
//
//-----------------------------------------------------------------------------

#include <optixu/optixu_math_namespace.h>
#include "mdl_helper.h"
#include "neuray_loader.h"
#include <iostream>

// Uncomment to enable dumping all generated PTX code to disk.
// #define MDL_HELPER_DUMP_PTX

// Uncomment to enable verbose output during texture loading.
// #define MDL_HELPER_VERBOSE_TEXTURES

using namespace mi::neuraylib;

#define MIHandle mi::base::Handle


// Throws an exception with the given error message, when expr is false.
void check_success(bool expr, const char *errMsg)
{
    if ( expr ) return;

    throw optix::Exception(errMsg);
}

// Throws an exception with the given error message, when expr is false.
void check_success(bool expr, const std::string &errMsg)
{
    if ( expr ) return;

    throw optix::Exception(errMsg);
}

// Throws an exception with the given error message, when expr is negative.
void check_success(mi::Sint32 errVal, const char *errMsg)
{
    if ( errVal >= 0 ) return;

    throw optix::Exception(std::string(errMsg) + "(" + to_string(errVal) + ")");
}

// Throws an exception with the given error message, when expr is negative.
void check_success(mi::Sint32 errVal, const std::string &errMsg)
{
    if ( errVal >= 0 ) return;

    throw optix::Exception(errMsg + "(" + to_string(errVal) + ")");
}

// Returns a string-representation of the given message severity.
const char* message_severity_to_string(mi::base::Message_severity severity)
{
    switch (severity)
    {
        case mi::base::MESSAGE_SEVERITY_ERROR:
            return "error";
        case mi::base::MESSAGE_SEVERITY_WARNING:
            return "warning";
        case mi::base::MESSAGE_SEVERITY_INFO:
            return "info";
        case mi::base::MESSAGE_SEVERITY_VERBOSE:
            return "verbose";
        case mi::base::MESSAGE_SEVERITY_DEBUG:
            return "debug";
        default:
            break;
    }
    return "";
}

// Returns a string-representation of the given message category
const char* message_kind_to_string(mi::neuraylib::IMessage::Kind message_kind)
{
    switch (message_kind)
    {
        case mi::neuraylib::IMessage::MSG_INTEGRATION:
            return "MDL SDK";
        case mi::neuraylib::IMessage::MSG_IMP_EXP:
            return "Importer/Exporter";
        case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:
            return "Compiler Backend";
        case mi::neuraylib::IMessage::MSG_COMILER_CORE:
            return "Compiler Core";
        case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
            return "Compiler Archive Tool";
        case mi::neuraylib::IMessage::MSG_COMPILER_DAG:
            return "Compiler DAG generator";
        default:
            break;
    }
    return "";
}

// Prints the messages of the given context.
void check_success(mi::neuraylib::IMdl_execution_context* context)
{
    for (mi::Size i = 0; i < context->get_messages_count(); ++i)
    {
        mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));

        switch (message->get_severity())
        {
            case mi::base::MESSAGE_SEVERITY_ERROR:
            case mi::base::MESSAGE_SEVERITY_FATAL:
            {
                std::string severity = message_severity_to_string(message->get_severity());
                std::string body = message->get_string();
                std::string kind = message_kind_to_string(message->get_kind());
                throw optix::Exception(severity + ": " + body + "(" + kind + ")");
            }
            default:
                fprintf(stderr, "%s: %s (%s)\n",
                    message_severity_to_string(message->get_severity()),
                    message->get_string(),
                    message_kind_to_string(message->get_kind()));
                break;
        }
    }
}

// Constructs an Mdl_helper object with the given OptiX context,
// the optional given path to the PTX file of mdl_textures.cu and
// and optional path to search for MDL modules and resources.
Mdl_helper::Mdl_helper(
        optix::Context optix_ctx,
        const std::string &mdl_textures_ptx_path,
        const std::string &module_path,
        unsigned num_texture_spaces,
        unsigned num_texture_results)
    : m_mdl_textures_ptx_path(mdl_textures_ptx_path)
    , m_optix_ctx(optix_ctx)
    , m_next_name_id(0)
{
    m_neuray = load_and_get_ineuray();
    check_success(m_neuray.is_valid_interface(),
#ifdef MI_PLATFORM_WINDOWS
        "ERROR: Initialization of MDL SDK failed: libmdl_sdk.dll not found or wrong version."
#else
        "ERROR: Initialization of MDL SDK failed: libmdl_sdk.so not found or wrong version."
#endif
    );

    m_mdl_compiler = m_neuray->get_api_component<IMdl_compiler>();
    check_success(m_mdl_compiler, "ERROR: Initialization of MDL compiler failed!");

    // Set module path for MDL file and resources, if provided
    if ( !module_path.empty() )
        add_module_path(module_path);

    // Load required plugins for texture support
#ifdef MI_PLATFORM_WINDOWS
    check_success(m_mdl_compiler->load_plugin_library("nvmdl_freeimage.dll"),
        "ERROR: Couldn't load plugin nvmdl_freeimage.dll!");

    // Consider the dds plugin as optional
    m_mdl_compiler->load_plugin_library("dds.dll");
#else
    check_success(m_mdl_compiler->load_plugin_library("nvmdl_freeimage.so"),
        "ERROR: Couldn't load plugin nvmdl_freeimage.so!");

    // Consider the dds plugin as optional
    m_mdl_compiler->load_plugin_library("dds.so");
#endif

    check_success(m_neuray->start(), "ERROR: Starting MDL SDK failed!");

    m_database = m_neuray->get_api_component<IDatabase>();
    m_global_scope = m_database->get_global_scope();

    m_mdl_factory = m_neuray->get_api_component<IMdl_factory>();

    // Configure the execution context, that is used for various configurable operations and for 
    // querying warnings and error messages.
    // It is possible to have more than one in order to use different settings.
    m_execution_context = m_mdl_factory->create_execution_context();

    m_execution_context->set_option("internal_space", "coordinate_world");  // equals default
    m_execution_context->set_option("bundle_resources", false);             // equals default
    m_execution_context->set_option("mdl_meters_per_scene_unit", 1.0f);     // equals default
    m_execution_context->set_option("mdl_wavelength_min", 380.0f);          // equals default
    m_execution_context->set_option("mdl_wavelength_max", 780.0f);          // equals default
    m_execution_context->set_option("include_geometry_normal", true);       // equals default

    m_be_cuda_ptx = m_mdl_compiler->get_backend(IMdl_compiler::MB_CUDA_PTX);
    check_success(m_be_cuda_ptx->set_option(
        "num_texture_spaces", to_string(num_texture_spaces).c_str()) == 0,
        "ERROR: Setting PTX option num_texture_spaces failed");
    check_success(m_be_cuda_ptx->set_option(
        "num_texture_results", to_string(num_texture_results).c_str()) == 0,
        "ERROR: Setting PTX option num_texture_results failed");
    check_success(m_be_cuda_ptx->set_option("sm_version", "30") == 0,
        "ERROR: Setting PTX option sm_version failed");
    check_success(m_be_cuda_ptx->set_option("tex_lookup_call_mode", "optix_cp") == 0,
        "ERROR: Setting PTX option tex_lookup_call_mode failed");

    m_image_api = m_neuray->get_api_component<IImage_api>();
}

// Destructor shutting down Neuray.
Mdl_helper::~Mdl_helper()
{
    m_image_api.reset();
    m_be_cuda_ptx.reset();
    m_execution_context.reset();
    m_mdl_factory.reset();
    m_global_scope.reset();
    m_database.reset();
    m_mdl_compiler.reset();

    m_neuray->shutdown();
}

// Adds a path to search for MDL modules and resources.
void Mdl_helper::add_module_path(const std::string &module_path)
{
    // Set module path for MDL file and resources
    check_success(m_mdl_compiler->add_module_path(module_path.c_str()),
        "ERROR: Adding module path failed!");
}


//
// Define a type trait to get type specific maximum alpha value for
// conversions from an image without alpha channel
//

template<typename I> struct Alpha_type_trait {};

template<> struct Alpha_type_trait<mi::Uint8>
{
    static mi::Uint8 get_alpha_max() { return 255; }
};

template<> struct Alpha_type_trait<mi::Sint8>
{
    static mi::Sint8 get_alpha_max() { return 127; }
};

template<> struct Alpha_type_trait<mi::Uint16>
{
    static mi::Uint16 get_alpha_max() { return 0xffff; }
};

template<> struct Alpha_type_trait<mi::Sint32>
{
    static mi::Sint32 get_alpha_max() { return 0x7fffffff; }
};

template<> struct Alpha_type_trait<mi::Float32>
{
    static mi::Float32 get_alpha_max() { return 1.0f; }
};


// Creates an OptiX buffer and fills it with image data converting it on the fly.
template <typename data_type, RTformat format, Mdl_helper::Convertmode conv_mode>
optix::Buffer Mdl_helper::load_image_data(
    MIHandle<const ICanvas> canvas,
    ITarget_code::Texture_shape texture_shape)
{
    mi::Uint32 res_x = canvas->get_resolution_x();
    mi::Uint32 res_y = canvas->get_resolution_y();
    mi::Uint32 num_layers = canvas->get_layers_size();
    mi::Uint32 tiles_x = canvas->get_tiles_size_x();
    mi::Uint32 tiles_y = canvas->get_tiles_size_y();
    mi::Uint32 tile_res_x = canvas->get_tile_resolution_x();
    mi::Uint32 tile_res_y = canvas->get_tile_resolution_y();

    optix::Buffer buffer;

    if ( texture_shape == ITarget_code::Texture_shape_invalid )
        throw optix::Exception("ERROR: Invalid texture shape used");
    else if ( texture_shape == ITarget_code::Texture_shape_ptex )
        throw optix::Exception("ERROR: texture_ptex not supported yet");
    else if ( texture_shape == ITarget_code::Texture_shape_cube && num_layers != 6 )
        throw optix::Exception("ERROR: texture_cube must be used with a texture with 6 layers");

    // We need to differentiate between createBuffer with 4 and 5 parameters here,
    // because the 5 parameter version always uses rtSetBufferSize3D
    // which prevents rtTex2D* functions from working.
    if ( texture_shape == ITarget_code::Texture_shape_2d )
    {
        buffer = m_optix_ctx->createBuffer(RT_BUFFER_INPUT, format, res_x, res_y);

        // When using a texture as texture_2d, always enforce using only the first layer.
        num_layers = 1;
    }
    else if ( texture_shape == ITarget_code::Texture_shape_3d )
    {
        buffer = m_optix_ctx->createBuffer(RT_BUFFER_INPUT, format, res_x, res_y, num_layers);
    }
    else
    {
        buffer = m_optix_ctx->createBuffer(
            RT_BUFFER_INPUT |
            (texture_shape == ITarget_code::Texture_shape_cube ? RT_BUFFER_CUBEMAP : 0),
            format, res_x, res_y, num_layers);
    }

    data_type *buffer_data = static_cast<data_type *>(buffer->map());

    mi::Uint32 num_src_components =
          conv_mode == CONV_IMG_1_TO_1 ? 1
        : conv_mode == CONV_IMG_2_TO_2 ? 2
        : conv_mode == CONV_IMG_3_TO_4 ? 3
        : conv_mode == CONV_IMG_4_TO_4 ? 4
        : 0;

    for ( mi::Uint32 layer = 0; layer < num_layers; ++layer )
    {
        for ( mi::Uint32 tile_y = 0, tile_ypos = 0; tile_y < tiles_y;
            ++tile_y, tile_ypos += tile_res_y )
        {
            for ( mi::Uint32 tile_x = 0, tile_xpos = 0; tile_x < tiles_x;
                ++tile_x, tile_xpos += tile_res_x )
            {
                MIHandle<const ITile> tile(canvas->get_tile(tile_xpos, tile_ypos, layer));
                const data_type *tile_data =
                    static_cast<const data_type *>(tile->get_data());

                mi::Uint32 x_end = optix::min(tile_xpos + tile_res_x, res_x);
                mi::Uint32 x_pad = (tile_xpos + tile_res_x - x_end) * num_src_components;
                mi::Uint32 y_end = optix::min(tile_ypos + tile_res_y, res_y);
                for ( mi::Uint32 y = tile_y; y < y_end; ++y )
                {
                    for ( mi::Uint32 x = tile_x; x < x_end; ++x )
                    {
                        *buffer_data++ = *tile_data++;                                     // R
                        if ( conv_mode != CONV_IMG_1_TO_1 )
                        {
                            *buffer_data++ = *tile_data++;                                 // G
                            if ( conv_mode != CONV_IMG_2_TO_2 )
                            {
                                *buffer_data++ = *tile_data++;                             // B
                                data_type alpha;
                                if ( conv_mode == CONV_IMG_3_TO_4 )
                                    alpha = Alpha_type_trait<data_type>::get_alpha_max();
                                else  // conv_mode == CONV_IMG_4_TO_4
                                    alpha = *tile_data++;
                                *buffer_data++ = alpha;                                    // A
                            }
                        }
                    }
                    tile_data += x_pad;  // possible padding due to tile size
                }
            }
        }
    }

    buffer->unmap();

    return buffer;
}

// Loads the texture given by the database name into an OptiX buffer
// converting it as necessary.
optix::Buffer Mdl_helper::load_texture(
    MIHandle<ITransaction> transaction,
    const char *texture_name,
    ITarget_code::Texture_shape texture_shape)
{
    // First check the texture cache
    std::string entry_name = std::string(texture_name) + "_" +
        to_string(unsigned(texture_shape));
    TextureCache::iterator it = m_texture_cache.find(entry_name);
    if ( it != m_texture_cache.end() )
        return it->second;

    MIHandle<const ITexture> texture(transaction->access<ITexture>(texture_name));
    MIHandle<const IImage> image(transaction->access<IImage>(texture->get_image()));
    MIHandle<const ICanvas> canvas(image->get_canvas());

#ifdef MDL_HELPER_VERBOSE_TEXTURES
    std::cout << "   Image type: " << image->get_type()
        << "\n   Resolution: " << image->resolution_x() << " * " << image->resolution_y()
        << "\n   Layers: " << canvas->get_layers_size()
        << "\n   Canvas Resolution: "
        << canvas->get_resolution_x() << " * " << canvas->get_resolution_y()
        << "\n   Tile Resolution: "
        << canvas->get_tile_resolution_x() << " * " << canvas->get_tile_resolution_y()
        << "\n   Tile Size: "
        << canvas->get_tiles_size_x() << " * " << canvas->get_tiles_size_y()
        << "\n   Canvas Gamma: "
        << canvas->get_gamma()
        << "\n   Texture Gamma: " << texture->get_gamma() << " (effective: "
        << texture->get_effective_gamma() << ")"
        << std::endl;
#endif

    const char *image_type = image->get_type();

    // Convert to linear color space if necessary
    if ( texture->get_effective_gamma() != 1.0f )
    {
        // Copy canvas and adjust gamma from "effective gamma" to 1
        MIHandle<ICanvas> gamma_canvas(m_image_api->convert(canvas.get(), image->get_type()));
        gamma_canvas->set_gamma(texture->get_effective_gamma());
        m_image_api->adjust_gamma(gamma_canvas.get(), 1.0f);
        canvas = gamma_canvas;
    }

    // Handle the different image types (see \ref mi_neuray_types for the available pixel types)

    optix::Buffer buffer;
    if ( !strcmp(image_type, "Sint8") )
    {
        buffer = load_image_data<mi::Sint8, RT_FORMAT_BYTE, CONV_IMG_1_TO_1>(
            canvas, texture_shape);
    }
    else if ( !strcmp(image_type, "Sint32") )
    {
        buffer = load_image_data<mi::Sint32, RT_FORMAT_INT, CONV_IMG_1_TO_1>(
            canvas, texture_shape);
    }
    else if ( !strcmp(image_type, "Float32") )
    {
        buffer = load_image_data<mi::Float32, RT_FORMAT_FLOAT, CONV_IMG_1_TO_1>(
            canvas, texture_shape);
    }
    else if ( !strcmp(image_type, "Float32<2>") )
    {
        buffer = load_image_data<mi::Float32, RT_FORMAT_FLOAT2, CONV_IMG_2_TO_2>(
            canvas, texture_shape);
    }
    else if ( !strcmp(image_type, "Rgb") )
    {
        // Note: OptiX does not support RT_FORMAT_UNSIGNED_BYTE3
        buffer = load_image_data<mi::Uint8, RT_FORMAT_UNSIGNED_BYTE4, CONV_IMG_3_TO_4>(
            canvas, texture_shape);
    }
    else if ( !strcmp(image_type, "Rgba") )
    {
        buffer = load_image_data<mi::Uint8, RT_FORMAT_UNSIGNED_BYTE4, CONV_IMG_4_TO_4>(
            canvas, texture_shape);
    }
    else if ( !strcmp(image_type, "Rgbe") )
    {
        // Convert to Rgb_fp first
        canvas = m_image_api->convert(canvas.get(), "Rgb_fp");
        buffer = load_image_data<mi::Float32, RT_FORMAT_FLOAT4, CONV_IMG_3_TO_4>(
            canvas, texture_shape);
    }
    else if ( !strcmp(image_type, "Rgbea") )
    {
        // Convert to Color first
        canvas = m_image_api->convert(canvas.get(), "Color");
        buffer = load_image_data<mi::Float32, RT_FORMAT_FLOAT4, CONV_IMG_4_TO_4>(
            canvas, texture_shape);
    }
    else if ( !strcmp(image_type, "Rgb_16") )
    {
        // Note: OptiX does not support RT_FORMAT_UNSIGNED_SHORT3
        buffer = load_image_data<mi::Uint16, RT_FORMAT_UNSIGNED_SHORT4, CONV_IMG_3_TO_4>(
            canvas, texture_shape);
    }
    else if ( !strcmp(image_type, "Rgba_16") )
    {
        buffer = load_image_data<mi::Uint16, RT_FORMAT_UNSIGNED_SHORT4, CONV_IMG_4_TO_4>(
            canvas, texture_shape);
    }
    else if ( !strcmp(image_type, "Rgb_fp") || !strcmp(image_type, "Float32<3>") )
    {
        // Note: OptiX does not support RT_FORMAT_FLOAT3
        buffer = load_image_data<mi::Float32, RT_FORMAT_FLOAT4, CONV_IMG_3_TO_4>(
            canvas, texture_shape);
    }
    else if ( !strcmp(image_type, "Color") || !strcmp(image_type, "Float32<4>") )
    {
        buffer = load_image_data<mi::Float32, RT_FORMAT_FLOAT4, CONV_IMG_4_TO_4>(
            canvas, texture_shape);
    }
    else
        throw optix::Exception(
            std::string("ERROR: Image type \"") + image_type + "\" not supported, yet!");

    m_texture_cache[entry_name] = buffer;
    return buffer;
}

// Loads all textures into buffers and returns a buffer of corresponding texture sampler IDs.
optix::Buffer Mdl_helper::create_texture_samplers(
    mi::base::Handle<mi::neuraylib::ITransaction> transaction,
    mi::base::Handle<const mi::neuraylib::ITarget_code> target_code)
{
    //
    // Use bindless textures:
    // Load all textures (ignoring the invalid texture) into buffers and
    // associate them to a texture sampler.
    // Keep the texture sampler IDs in a buffer for use in mdl_textures.cu.
    //

    optix::Buffer tex_samplers = m_optix_ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT,
        target_code->get_texture_count() - 1);
    int *tex_sampler_ids = static_cast<int *>(tex_samplers->map());

    // i = 0 is the invalid texture, so we skip it
    for ( mi::Size i = 1; i < target_code->get_texture_count(); ++i )
    {
#ifdef MDL_HELPER_VERBOSE_TEXTURES
        std::cout << " - " << target_code->get_texture(i) << std::endl;
#endif

        ITarget_code::Texture_shape texture_shape = target_code->get_texture_shape(i);
        optix::Buffer buffer = load_texture(transaction, target_code->get_texture(i),
            target_code->get_texture_shape(i));

        // Create a texture sampler for the buffer and store the texture sampler id.

        optix::TextureSampler sampler = m_optix_ctx->createTextureSampler();

        // For cube maps use clamped address mode to avoid artifacts in the corners
        if ( texture_shape == ITarget_code::Texture_shape_cube )
        {
            sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
            sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
            sampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
        }

        sampler->setBuffer(buffer);
        tex_sampler_ids[i - 1] = sampler->getId();
    }

    tex_samplers->unmap();
    return tex_samplers;
}

// Sets all texture functions on the given program and associate the texture samplers.
void Mdl_helper::set_texture_functions(
    std::vector<optix::Program *> programs,
    optix::Buffer texture_samplers)
{
    // Set up the texture access functions for the MDL expression code.
    static const char *tex_prog_names[] = {
        "tex_lookup_float4_2d",
        "tex_lookup_float3_2d",
        "tex_lookup_float3_cube",
        "tex_texel_float4_2d",
        "tex_lookup_float4_3d",
        "tex_lookup_float3_3d",
        "tex_texel_float4_3d",
        "tex_resolution_2d"
    };

    for ( size_t i = 0; i < sizeof(tex_prog_names) / sizeof(*tex_prog_names); ++i )
    {
        optix::Program texprog;
        if ( !m_mdl_textures_ptx_string.empty() )
            texprog = m_optix_ctx->createProgramFromPTXString(
                m_mdl_textures_ptx_string, tex_prog_names[i]);
        else
            texprog = m_optix_ctx->createProgramFromPTXFile(
                m_mdl_textures_ptx_path, tex_prog_names[i]);
        check_success(
            texprog, std::string("ERROR: Compiling ") + tex_prog_names[i] + "failed!");
        texprog["texture_sampler_ids"]->set(texture_samplers);
        for ( size_t j = 0, n = programs.size(); j < n; ++j )
        {
            (*programs[j])[tex_prog_names[i]]->setProgramId(texprog);
        }
    }
}

// Compiles the given target code to an OptiX program and loads all referenced textures.
optix::Program Mdl_helper::create_program(
    MIHandle<ITransaction> transaction,
    MIHandle<const ITarget_code> target_code,
    const std::string &function_name)
{
    // Generate the program object
    std::string ptx_code = std::string(target_code->get_code(), target_code->get_code_size());

#ifdef MDL_HELPER_DUMP_PTX
    static int prog_id = 0;
    std::string fname = std::string("prog_") + to_string(prog_id++) + ".ptx";
    FILE *file = fopen(fname.c_str(), "wt");
    fwrite(target_code->get_code(), 1, target_code->get_code_size(), file);
    fclose(file);
#endif

    optix::Program mdl_expr_prog =
        m_optix_ctx->createProgramFromPTXString(ptx_code, function_name);

    // Prepare texture access, if the generated code calls texture functions
    if ( target_code->get_texture_count() > 0 )
    {
        optix::Buffer tex_samplers = create_texture_samplers(transaction, target_code);

        std::vector<optix::Program *> programs;
        programs.push_back(&mdl_expr_prog);
        set_texture_functions(programs, tex_samplers);
    }

    return mdl_expr_prog;
}

// Compiles the given target code to an OptiX program and loads all referenced textures.
void Mdl_helper::create_programs(
    MIHandle<ITransaction> transaction,
    MIHandle<const ITarget_code> target_code,
    std::vector<optix::Program *> &out_programs,
    std::vector<std::string> &function_names)
{
    // Generate the program object
    std::string ptx_code = std::string(target_code->get_code(), target_code->get_code_size());

#ifdef MDL_HELPER_DUMP_PTX
    static int prog_id = 0;
    std::string fname = std::string("prog_") + to_string(prog_id++) + ".ptx";
    FILE *file = fopen(fname.c_str(), "wt");
    fwrite(target_code->get_code(), 1, target_code->get_code_size(), file);
    fclose(file);
#endif

    if ( out_programs.size() != function_names.size() )
        throw optix::Exception(
            "ERROR: create_programs got inconsistent number of programs and names");

    for ( size_t i = 0, n = out_programs.size(); i < n; ++i )
    {
        *out_programs[i] = m_optix_ctx->createProgramFromPTXString(ptx_code, function_names[i]);
    }

    // Load the referenced textures if more than just the invalid texture used
    if ( target_code->get_texture_count() > 0 )
    {
        optix::Buffer tex_samplers = create_texture_samplers(transaction, target_code);

        set_texture_functions(out_programs, tex_samplers);
    }
}

// Loads the given module to allow accessing any members with the given transaction.
void Mdl_helper::load_module(
    const std::string &module_name,
    mi::base::Handle<mi::neuraylib::ITransaction> transaction)
{
    std::string mdl_module_name = "::" + module_name;
    check_success(m_mdl_compiler->load_module(transaction.get(), mdl_module_name.c_str()),
        "ERROR: Loading " + module_name + ".mdl failed!");
}

// Compiles the MDL expression given by the module and material name and the expression path
// to an OptiX program with a function with the given PTX name.
optix::Program Mdl_helper::compile_expression(
    const std::string &module_name,
    const std::string &material_name,
    const std::string &expression_path,
    MIHandle<const IExpression_list> arguments,
    const std::string &res_function_name)
{
    Mdl_compile_result compile_result(compile(module_name, material_name, arguments));

    optix::Program mdl_expr_prog(
        compile_result.compile_expression(expression_path, res_function_name));

    compile_result.commit_transaction();

    return mdl_expr_prog;
}

// Compiles the MDL expression given by the module and material name and the expression path
// to an OptiX material using the given function name for the PTX code.
optix::Material Mdl_helper::compile_material(
    const std::string &module_name,
    const std::string &material_name,
    const std::string &expression_path,
    MIHandle<const IExpression_list> arguments,
    const std::string &res_function_name)
{
    //
    // Compile the material to a program
    //

    Mdl_compile_result compile_result(compile(module_name, material_name, arguments));

    optix::Program mdl_expr_prog(
        compile_result.compile_expression(expression_path, res_function_name));

    // Create an OptiX material and associate the MDL expression program with the material.

    optix::Material material = m_optix_ctx->createMaterial();
    material["mdl_expr"]->setProgramId(mdl_expr_prog);
    material["mdl_shading_normal_expr"]->setProgramId(
        compile_result.compile_expression("geometry.normal", "normal"));

    compile_result.commit_transaction();

    return material;
}

// Compile the given MDL distribution function into a group of OptiX programs.
Mdl_BSDF_program_group Mdl_helper::compile_df(
    const std::string &module_name,
    const std::string &material_name,
    const std::string &expression_path,
    mi::base::Handle<const mi::neuraylib::IExpression_list> arguments,
    const std::string &res_function_name_base)
{
    Mdl_compile_result compile_result(compile(module_name, material_name, arguments));

    Mdl_BSDF_program_group progs = compile_result.compile_df(
        expression_path,
        res_function_name_base);

    compile_result.commit_transaction();

    return progs;
}

// Compiles the MDL expression given by the module and material name and the function name
// as an environment to an OptiX program using the given function name for the PTX code.
optix::Program Mdl_helper::compile_environment(
    const std::string &module_name,
    const std::string &function_name,
    mi::base::Handle<const mi::neuraylib::IExpression_list> arguments,
    const std::string &res_function_name)
{
    optix::Program mdl_expr_prog;

    std::string func_name;
    if ( res_function_name.empty() ) {
        func_name.assign(function_name);
        std::replace(func_name.begin(), func_name.end(), '.', '_');
        size_t paren_idx = func_name.find('(');
        if ( paren_idx != std::string::npos )
            func_name.erase(paren_idx);
    } else {
        func_name.assign(res_function_name);
    }

    //
    // Compile the environment function to a program
    //

    MIHandle<ITransaction> transaction(m_global_scope->create_transaction());

    {
        std::string mdl_module_name = "::" + module_name;
        check_success(m_mdl_compiler->load_module(transaction.get(), mdl_module_name.c_str()),
            "ERROR: Loading " + module_name + ".mdl failed!");

        std::string mdl_func_name = "mdl::" + module_name + "::" + function_name;
        MIHandle<const IFunction_definition> func_definition(
            transaction->access<IFunction_definition>(mdl_func_name.c_str()));
        check_success(func_definition, "ERROR: Accessing function failed!");

        // Create a function call with only default parameters
        mi::Sint32 result;
        MIHandle<IFunction_call> func_call(
            func_definition->create_function_call(arguments.get(), &result));
        check_success(result, "ERROR: Creating MDL function call failed!");

        MIHandle<const ITarget_code> code_cuda_ptx(
            m_be_cuda_ptx->translate_environment(
                transaction.get(), func_call.get(),
                func_name.c_str(), m_execution_context.get()));
        check_success(m_execution_context.get());

        mdl_expr_prog = create_program(transaction, code_cuda_ptx, func_name);
    }

    transaction->commit();

    return mdl_expr_prog;
}

// Instantiates the MDL material given by the module and material name with the provided
// optional arguments using the given optional transaction and compiles the material.
// The returned object can be used to compile expressions and examine the compiled material.
// If no arguments are provided, the default values are used according to the material
// definition.
Mdl_helper::Mdl_compile_result Mdl_helper::compile(
    const std::string &module_name,
    const std::string &material_name,
    MIHandle<const IExpression_list> arguments,
    MIHandle<ITransaction> transaction)
{
    // Create a transaction if none was provided
    if ( !transaction )
        transaction = m_global_scope->create_transaction();

    Mdl_helper::Mdl_compile_result compile_result(*this, transaction);

    //
    // Instantiate and compile the material
    //

    {
        std::string mdl_module_name =
            (module_name.find("::") == 0) ? module_name : "::" + module_name;
        check_success(m_mdl_compiler->load_module(transaction.get(), mdl_module_name.c_str()),
            "ERROR: Loading " + module_name + ".mdl failed!");

        std::string mdl_material_name = "mdl" + mdl_module_name + "::" + material_name;

        MIHandle<const IMaterial_definition> material_definition(
            transaction->access<IMaterial_definition>(mdl_material_name.c_str()));
        check_success(material_definition, "ERROR: Material " + mdl_material_name + " not found!");

        // Instantiate the material
        mi::Sint32 result;
        MIHandle<IMaterial_instance> material_instance(
            material_definition->create_material_instance(arguments.get(), &result));
        check_success(result, "ERROR: Creating material instance failed!");

        MIHandle<ICompiled_material> compiled_material(
            material_instance->create_compiled_material(
                IMaterial_instance::DEFAULT_OPTIONS,
                m_execution_context.get()));
        check_success(m_execution_context.get());

        compile_result.set_compiled_material(compiled_material);
    }

    return compile_result;
}

// Releases the compiled material and commits the transaction.
void Mdl_helper::Mdl_compile_result::commit_transaction()
{
    m_compiled_material.reset();
    m_transaction->commit();
}

// Compile the given MDL subexpression to an OptiX program with the given output function name.
optix::Program Mdl_helper::Mdl_compile_result::compile_expression(
    const std::string &expression_path, const std::string &res_function_name)
{
    std::string func_name;
    if ( res_function_name.empty() ) {
        func_name.assign(expression_path);
        std::replace(func_name.begin(), func_name.end(), '.', '_');
    } else {
        func_name.assign(res_function_name);
    }

    MIHandle<const ITarget_code> code_cuda_ptx(
    m_mdl_helper.m_be_cuda_ptx->translate_material_expression(
        m_transaction.get(), m_compiled_material.get(),
        expression_path.c_str(), func_name.c_str(), m_mdl_helper.m_execution_context.get()));
    check_success(m_mdl_helper.m_execution_context.get());

    return m_mdl_helper.create_program(m_transaction, code_cuda_ptx, func_name);
}

// Compile an MDL distribution function into a group of OptiX programs.
Mdl_BSDF_program_group Mdl_helper::Mdl_compile_result::compile_df(
    const std::string &expression_path, const std::string &res_function_name_base)
{
    std::string func_name;
    if ( res_function_name_base.empty() ) {
        func_name.assign(expression_path);
        std::replace(func_name.begin(), func_name.end(), '.', '_');
    } else {
        func_name.assign(res_function_name_base);
    }

    // Generate code for the MDL distribution function.
    // This will result in 4 functions in the target code: init, sample, evaluate and pdf.
    MIHandle<const ITarget_code> code_cuda_ptx(
        m_mdl_helper.m_be_cuda_ptx->translate_material_df(
            m_transaction.get(), m_compiled_material.get(),
            expression_path.c_str(), func_name.c_str(), m_mdl_helper.m_execution_context.get()));
    check_success(m_mdl_helper.m_execution_context.get());

    Mdl_BSDF_program_group programs;

    std::vector<optix::Program *> prog_list;
    std::vector<std::string> func_names;

    prog_list.push_back(&programs.init_prog);
    func_names.push_back(code_cuda_ptx->get_callable_function(0));

    prog_list.push_back(&programs.sample_prog);
    func_names.push_back(code_cuda_ptx->get_callable_function(1));

    prog_list.push_back(&programs.evaluate_prog);
    func_names.push_back(code_cuda_ptx->get_callable_function(2));

    prog_list.push_back(&programs.pdf_prog);
    func_names.push_back(code_cuda_ptx->get_callable_function(3));

    m_mdl_helper.create_programs(m_transaction, code_cuda_ptx, prog_list, func_names);

    return programs;
}
