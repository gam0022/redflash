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

#pragma once

#include <optixu/optixpp_namespace.h>
#include <mi/mdl_sdk.h>
#include <map>


/// A structure containing the init, sample, evaluate and pdf OptiX programs of a compiled
/// MDL distribution function.
struct Mdl_BSDF_program_group
{
    optix::Program init_prog;
    optix::Program sample_prog;
    optix::Program evaluate_prog;
    optix::Program pdf_prog;
};


template<typename T>
std::string to_string(T val)
{
    std::ostringstream stream;
    stream << val;
    return stream.str();
}



/// A helper class for handling MDL in OptiX.
class Mdl_helper
{
public:
    /// Constructs an Mdl_helper object.
    /// \param optix_ctx             the given OptiX context,
    /// \param mdl_textures_ptx_path the path to the PTX file of mdl_textures.cu.
    ///                              If you don't need texture support, this can be empty.
    ///                              It can also be set later or you can later set the PTX code
    ///                              directly (e.g. as obtained by the CUDA runtime compiler)
    /// \param module_path           an optional search path for MDL modules and resources.
    /// \param num_texture_spaces    the number of texture spaces provided in the MDL_SDK_State
    ///                              fields text_coords, tangent_t and tangent_v by the renderer
    ///                              If invalid texture spaces are requested in the MDL materials,
    ///                              null values will be returned.
    /// \param num_texture_results   the size of the text_results array in the MDL_SDK_State
    ///                              text_results field provided by the renderer.
    ///                              The text_results array will be filled in the init function
    ///                              created for a distribution function and used by the sample,
    ///                              evaluate and pdf functions, if the size is non-zero.
    Mdl_helper(
        optix::Context optix_ctx,
        const std::string &mdl_textures_ptx_path = std::string(),
        const std::string &module_path = std::string(),
        unsigned num_texture_spaces = 1,
        unsigned num_texture_results = 0 );

    /// Destructor shutting down Neuray.
    ~Mdl_helper();

    class Mdl_compile_result
    {
    public:
        /// Constructor an MDL compile result which can be used to generate OptiX programs.
        ///
        /// \param mdl_helper   the MDL helper to use to generate the programs
        /// \param transaction  the transaction to use for the MDL backend
        Mdl_compile_result(
                Mdl_helper &mdl_helper,
                mi::base::Handle<mi::neuraylib::ITransaction> transaction)
            : m_mdl_helper(mdl_helper)
            , m_transaction(transaction)
        {
        }

        /// Releases the compiled material and commits the transaction.
        void commit_transaction();

        /// Set the compiled material for this compile result.
        /// This is done separately from the constructor to avoid problems with transactions.
        void set_compiled_material(
            mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material)
        {
            m_compiled_material = compiled_material;
        }

        /// Returns the compiled material used by this compile result.
        mi::base::Handle<const mi::neuraylib::ICompiled_material> get_compiled_material() const
        {
            return m_compiled_material;
        }

        /// Returns the MDL helper.
        Mdl_helper *get_mdl_helper() const { return &m_mdl_helper; }

        /// Compile an MDL subexpression of the associated compiled material to an OptiX program.
        ///
        /// You can use the resulting function via a callable program declared like this:
        ///
        /// \code
        /// rtDeclareVariable(
        ///     rtCallableProgramId<
        ///         void(void*, MDL_SDK_State const*, MDL_SDK_Res_data_pair const*, void*)>,
        ///     YOUR_FUNCTION_NAME,,);
        /// \endcode
        ///
        /// The first parameter is the result buffer and must be big enough for the result.
        /// The second parameter is the renderer state.
        /// The third parameter is the resource data which may be used via its \c thread_data
        /// field to provide a context to the texture lookup functions.
        /// The forth parameter is currently unused and should be NULL.
        ///
        /// \param expression_path    the path of the MDL subexpression
        /// \param res_function_name  a name for the function which will be generated.
        ///                           If it is empty, a name based on the expression path will
        ///                           be generated.
        ///
        /// \returns the generated OptiX program
        optix::Program compile_expression(
            const std::string &expression_path,
            const std::string &res_function_name = std::string());

        /// Compile an MDL distribution function into a group of OptiX programs.
        Mdl_BSDF_program_group compile_df(
            const std::string &expression_path,
            const std::string &res_function_name_base = std::string());

    private:
        /// The MDL helper.
        Mdl_helper &m_mdl_helper;

        /// The database transaction used for the compiled material.
        mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;

        /// The compiled material.
        mi::base::Handle<const mi::neuraylib::ICompiled_material> m_compiled_material;

        /// The expression factory
        mi::base::Handle<const mi::neuraylib::IExpression_factory> m_expr_factory;
    };

    /// Adds a path to search for MDL modules and resources.
    void add_module_path(const std::string &module_path);

    /// Sets the path to the PTX file of mdl_textures.cu.
    void set_mdl_textures_ptx_path(const std::string &mdl_textures_ptx_path)
    {
        m_mdl_textures_ptx_path = mdl_textures_ptx_path;
    }

    /// Sets the content of the PTX code of mdl_textures.cu.
    void set_mdl_textures_ptx_string(const std::string &mdl_textures_ptx_string)
    {
        m_mdl_textures_ptx_string = mdl_textures_ptx_string;
    }

    /// Creates a new Neuray transaction.
    mi::base::Handle<mi::neuraylib::ITransaction> create_transaction()
    {
        return mi::base::Handle<mi::neuraylib::ITransaction>(m_global_scope->create_transaction());
    }

    /// Returns the MDL factory.
    mi::base::Handle<mi::neuraylib::IMdl_factory> get_mdl_factory() const
    {
        return m_mdl_factory;
    }

    /// Returns the next string which can be used as database names for a transaction.
    std::string get_next_unique_dbname()
    {
        return "_mdlhelper_" + to_string(m_next_name_id++);
    }

    /// Loads the given module to allow accessing any members with the given transaction.
    void load_module(
        const std::string &module_name,
        mi::base::Handle<mi::neuraylib::ITransaction> transaction);

    /// Compiles the MDL expression given by the module and material name and the expression path
    /// to an OptiX program with a function with the given PTX name.
    optix::Program compile_expression(
        const std::string &module_name,
        const std::string &material_name,
        const std::string &expression_path,
        mi::base::Handle<const mi::neuraylib::IExpression_list> arguments,
        const std::string &res_function_name = std::string());

    /// Compiles the MDL expression given by the module and material name and the expression path
    /// to an OptiX program with a function with the given PTX name.
    optix::Program compile_expression(
        const std::string &module_name,
        const std::string &material_name,
        const std::string &expression_path,
        const std::string &res_function_name = std::string())
    {
        mi::base::Handle<const mi::neuraylib::IExpression_list> args;
        return compile_expression(module_name, material_name, expression_path, args,
            res_function_name);
    }

    /// Compiles the MDL expression given by the module and material name and the expression path
    /// to an OptiX material using the given function name for the PTX code.
    optix::Material compile_material(
        const std::string &module_name,
        const std::string &material_name,
        const std::string &expression_path,
        mi::base::Handle<const mi::neuraylib::IExpression_list> arguments,
        const std::string &res_function_name);

    /// Compile the given MDL distribution function into a group of OptiX programs.
    Mdl_BSDF_program_group compile_df(
        const std::string &module_name,
        const std::string &material_name,
        const std::string &expression_path,
        mi::base::Handle<const mi::neuraylib::IExpression_list> arguments,
        const std::string &res_function_name_base = std::string());

    /// Compile the given MDL distribution function into a group of OptiX programs.
    Mdl_BSDF_program_group compile_df(
        const std::string &module_name,
        const std::string &material_name,
        const std::string &expression_path,
        const std::string &res_function_name_base = std::string())
    {
        mi::base::Handle<const mi::neuraylib::IExpression_list> args;
        return compile_df(
            module_name,
            material_name,
            expression_path,
            args,
            res_function_name_base);
    }

    /// Compiles the MDL expression given by the module and material name and the function name
    /// as an environment to an OptiX program using the given function name for the PTX code.
    optix::Program compile_environment(
        const std::string &module_name,
        const std::string &function_name,
        mi::base::Handle<const mi::neuraylib::IExpression_list> arguments,
        const std::string &res_function_name = std::string());

    /// Compiles the MDL expression given by the module and material name and the function name
    /// as an environment to an OptiX program using the given function name for the PTX code.
    optix::Program compile_environment(
        const std::string &module_name,
        const std::string &function_name,
        const std::string &res_function_name = std::string())
    {
        mi::base::Handle<const mi::neuraylib::IExpression_list> args;
        return compile_environment(module_name, function_name, args, res_function_name);
    }

    /// Instantiates the MDL material given by the module and material name with the provided
    /// optional arguments using the given optional transaction and compiles the material.
    /// The returned object can be used to compile expressions and examine the compiled material.
    /// If no arguments are provided, the default values are used according to the material
    /// definition.
    Mdl_compile_result compile(
        const std::string &module_name,
        const std::string &material_name,
        mi::base::Handle<const mi::neuraylib::IExpression_list> arguments =
            mi::base::Handle<const mi::neuraylib::IExpression_list>(),
        mi::base::Handle<mi::neuraylib::ITransaction> transaction =
            mi::base::Handle<mi::neuraylib::ITransaction>());

    /// Clears the texture cache which holds references to the used OptiX buffers.
    void clear_texture_cache()
    {
        m_texture_cache.clear();
    }

private:
    typedef enum
    {
        CONV_IMG_1_TO_1,  ///< source and destination both have one component
        CONV_IMG_2_TO_2,  ///< source and destination both have two components
        CONV_IMG_3_TO_4,  ///< source has three and destination has four components,
                          ///< alpha is set to maximum alpha value for the according type
        CONV_IMG_4_TO_4   ///< source and destination both have four components
    } Convertmode;

    /// Creates an OptiX buffer and fills it with image data converting it on the fly.
    template <typename data_type, RTformat format, Convertmode conv_mode>
    optix::Buffer load_image_data(
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas,
        mi::neuraylib::ITarget_code::Texture_shape texture_shape);

    /// Loads the texture given by the database name into an OptiX buffer
    /// converting it as necessary.
    optix::Buffer load_texture(
        mi::base::Handle<mi::neuraylib::ITransaction> transaction,
        const char *texture_name,
        mi::neuraylib::ITarget_code::Texture_shape texture_shape);

    /// Loads all textures into buffers and returns a buffer of corresponding texture sampler IDs.
    optix::Buffer create_texture_samplers(
        mi::base::Handle<mi::neuraylib::ITransaction> transaction,
        mi::base::Handle<const mi::neuraylib::ITarget_code> target_code);

    /// Sets all texture functions on the given programs and associate the texture samplers.
    void set_texture_functions(
        std::vector<optix::Program *> programs,
        optix::Buffer texture_samplers);

    /// Compiles the given target code to an OptiX program and loads all referenced textures.
    optix::Program create_program(
        mi::base::Handle<mi::neuraylib::ITransaction> transaction,
        mi::base::Handle<const mi::neuraylib::ITarget_code> target_code,
        const std::string &function_name);

    /// Compiles the given target code to multiple OptiX programs and loads all referenced
    /// textures.
    void create_programs(
        mi::base::Handle<mi::neuraylib::ITransaction> transaction,
        mi::base::Handle<const mi::neuraylib::ITarget_code> target_code,
        std::vector<optix::Program *> &out_programs,
        std::vector<std::string> &function_names);

    /// The path to the mdl_textures.ptx file.
    std::string m_mdl_textures_ptx_path;

    /// The content of the PTX code of mdl_textures.cu.
    std::string m_mdl_textures_ptx_string;

    /// The OptiX context.
    optix::Context m_optix_ctx;

    /// The Neuray interface of the MDL SDK.
    mi::base::Handle<mi::neuraylib::INeuray> m_neuray;

    /// The MDL compiler.
    mi::base::Handle<mi::neuraylib::IMdl_compiler> m_mdl_compiler;

    /// The Neuray database of the MDL SDK.
    mi::base::Handle<mi::neuraylib::IDatabase> m_database;

    /// The global scope of the data base used to create transactions.
    mi::base::Handle<mi::neuraylib::IScope> m_global_scope;

    /// The MDL factory.
    mi::base::Handle<mi::neuraylib::IMdl_factory> m_mdl_factory;

    /// Can be used to query status information like errors and warnings.
    /// The context is also used to set options for module loading, MDL export, 
    /// material compilation, and for the code generation.
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_execution_context;

    /// The CUDA PTX backend of the MDL compiler.
    mi::base::Handle<mi::neuraylib::IMdl_backend> m_be_cuda_ptx;

    /// The Image API for converting image to other formats.
    mi::base::Handle<mi::neuraylib::IImage_api> m_image_api;

    /// The next ID to use for unique database names.
    int m_next_name_id;

    typedef std::map<std::string, optix::Buffer> TextureCache;

    /// Maps a texture name and a texture shape to an OptiX buffer to avoid the texture
    /// being loaded and converted multiple times.
    TextureCache m_texture_cache;
};
