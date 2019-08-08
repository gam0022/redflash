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

#include <optixu/optixpp_namespace.h>
#include "mdl_helper.h"

#ifndef MDLWRAPPERAPI
#  if mdl_wrapper_EXPORTS // Set by CMAKE
#    if defined( _WIN32 )
#      define MDLWRAPPERAPI extern "C" __declspec(dllexport)
#    elif defined( __linux__ ) || defined ( __CYGWIN__ )
#      define MDLWRAPPERAPI extern "C" __attribute__ ((visibility ("default")))
#    elif defined( __APPLE__ ) && defined( __MACH__ )
#      define MDLWRAPPERAPI extern "C" __attribute__ ((visibility ("default")))
#    else
#      error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#    endif
#  endif
#endif

typedef void *Mdl_wrapper_context;

/// A structure containing the C-typed sample, evaluate and pdf OptiX program of a compiled BSDF.
struct Mdl_BSDF_program_group_C
{
    RTprogram init_prog;
    RTprogram sample_prog;
    RTprogram evaluate_prog;
    RTprogram pdf_prog;
};

 // Constructor of an MDL wrapper context.
MDLWRAPPERAPI Mdl_wrapper_context mdl_wrapper_create(
    RTcontext optix_ctx,
    const char *mdl_textures_ptx_path,
    const char *module_path,
    unsigned num_texture_spaces,
    unsigned num_texture_results )
{
    optix::Context optix_ctx_obj = optix::Context::take( optix_ctx );

    // Artificially add a reference to avoid that the OptiX context is destroyed when the
    // Mdl_helper is destructed. The Mdl_wrapper class will ensure, that the OptiX context
    // references are counted correctly.
    optix_ctx_obj->addReference();

    return static_cast<Mdl_wrapper_context>(
        new Mdl_helper(
            optix_ctx_obj,
            mdl_textures_ptx_path,
            module_path,
            num_texture_spaces,
            num_texture_results ));
}


// Destructor shutting down Neuray.
MDLWRAPPERAPI void mdl_wrapper_destroy( Mdl_wrapper_context mdl_ctx )
{
    Mdl_helper *helper = static_cast<Mdl_helper *>( mdl_ctx );
    delete helper;
}


// Adds a path to search for MDL modules and resources.
MDLWRAPPERAPI void mdl_wrapper_add_module_path(
    Mdl_wrapper_context mdl_ctx,
    const char *module_path )
{
    Mdl_helper *helper = static_cast<Mdl_helper *>( mdl_ctx );
    helper->add_module_path( module_path );
}


// Sets the path to the PTX file of mdl_textures.cu.
MDLWRAPPERAPI void mdl_wrapper_set_mdl_textures_ptx_path(
    Mdl_wrapper_context mdl_ctx,
    const char *mdl_textures_ptx_path )
{
    Mdl_helper *helper = static_cast<Mdl_helper *>( mdl_ctx );
    helper->set_mdl_textures_ptx_path( mdl_textures_ptx_path );
}

// Sets the content of the PTX code of mdl_textures.cu.
MDLWRAPPERAPI void mdl_wrapper_set_mdl_textures_ptx_string(
    Mdl_wrapper_context mdl_ctx,
    const char *mdl_textures_ptx_string )
{
    Mdl_helper *helper = static_cast<Mdl_helper *>( mdl_ctx );
    helper->set_mdl_textures_ptx_string( mdl_textures_ptx_string );
}

// Compiles the MDL expression given by the module and material name and the expression path
// to an OptiX program.
MDLWRAPPERAPI RTprogram mdl_wrapper_compile_expression(
    Mdl_wrapper_context mdl_ctx,
    const char *module_name,
    const char *material_name,
    const char *expression_path,
    const char *res_function_name )
{
    Mdl_helper *helper = static_cast<Mdl_helper *>( mdl_ctx );
    optix::Program prog = helper->compile_expression(
        module_name,
        material_name,
        expression_path,
        res_function_name );

    prog->addReference();  // ensure it will not be destroyed here
    return prog->get();
}

// Compiles the MDL expression given by the module and material name and the function name
// as an environment to an OptiX program.
MDLWRAPPERAPI RTprogram mdl_wrapper_compile_environment(
    Mdl_wrapper_context mdl_ctx,
    const char *module_name,
    const char *function_name,
    const char *res_function_name )
{
    Mdl_helper *helper = static_cast<Mdl_helper *>( mdl_ctx );
    optix::Program prog = helper->compile_environment(
        module_name,
        function_name,
        res_function_name );

    prog->addReference();  // ensure it will not be destroyed here
    return prog->get();
}

// Compiles the given material into a group of OptiX programs.
MDLWRAPPERAPI Mdl_BSDF_program_group_C mdl_wrapper_compile_df(
    Mdl_wrapper_context mdl_ctx,
    const char *module_name,
    const char *material_name,
    const char *expression_path,
    const char *res_function_name_base )
{
    Mdl_BSDF_program_group_C res;

    Mdl_helper *helper = static_cast<Mdl_helper *>(mdl_ctx);
    Mdl_BSDF_program_group progs = helper->compile_df(
        module_name,
        material_name,
        expression_path,
        res_function_name_base );

    // ensure the program objects will not be destroyed
    progs.init_prog->addReference();
    progs.sample_prog->addReference();
    progs.evaluate_prog->addReference();
    progs.pdf_prog->addReference();

    res.init_prog     = progs.init_prog->get();
    res.sample_prog   = progs.sample_prog->get();
    res.evaluate_prog = progs.evaluate_prog->get();
    res.pdf_prog      = progs.pdf_prog->get();

    return res;
}

// Clears the texture cache which holds references to the used OptiX buffers.
MDLWRAPPERAPI void mdl_wrapper_clear_texture_cache( Mdl_wrapper_context mdl_ctx )
{
    Mdl_helper *helper = static_cast<Mdl_helper *>( mdl_ctx );
    helper->clear_texture_cache();
}
