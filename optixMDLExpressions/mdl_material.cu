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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <mi/neuraylib/target_code_types.h>

#include "phong.h"

using namespace optix;

/// Signature of environment functions created via
/// #mi::neuraylib::IMdl_backend::translate_environment() and
/// #mi::neuraylib::ILink_unit::add_environment().
///
/// \param result           pointer to the result buffer which must be large enough for the result
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   unused, should be NULL
typedef void (Environment_function)  (void *result,
    const mi::neuraylib::Shading_state_environment *state,
    const mi::neuraylib::Resource_data *res_data,
    const void *exception_state,
    const char *arg_block_data);


rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_tangent,  attribute shading_tangent, );
rtDeclareVariable(float3, texcoord,         attribute texcoord, );

rtDeclareVariable(float, reflection_coefficient, , );

rtDeclareVariable(
    rtCallableProgramId<mi::neuraylib::Material_expr_function>, mdl_expr,,);

rtDeclareVariable(
    rtCallableProgramId<mi::neuraylib::Material_expr_function>, mdl_shading_normal_expr,,);

rtDeclareVariable(
    rtCallableProgramId<Environment_function>, mdl_env_expr,,);

__device__ const mi::neuraylib::Resource_data res_data = {
    NULL,
    NULL
};


// computes direct lighting using phong shading
RT_PROGRAM void closest_hit_radiance()
{
    float3 world_shading_normal =
        normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ));
    float3 world_geometric_normal =
        normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ));

    //
    // Initialize state for MDL
    //

    float world_to_object[16];
    float object_to_world[16];
    rtGetTransform(RT_WORLD_TO_OBJECT, world_to_object);
    rtGetTransform(RT_OBJECT_TO_WORLD, object_to_world);

    float3 hit_point = ray.origin + t_hit * ray.direction;

    float3 text_coords = texcoord;      // taking an address of an attribute is not supported
    float3 tangent_u = shading_tangent;
    float3 tangent_v = cross(world_shading_normal, tangent_u);

    mi::neuraylib::Shading_state_material state;
    state.normal = world_shading_normal;
    state.geom_normal = world_geometric_normal;
    state.position = hit_point;
    state.animation_time = 0;
    state.text_coords = &text_coords;
    state.tangent_u = &tangent_u;
    state.tangent_v = &tangent_v;
    state.text_results = NULL;
    state.ro_data_segment = NULL;
    state.world_to_object = (float4 *) world_to_object;
    state.object_to_world = (float4 *) object_to_world;
    state.object_id = 0;

    //
    // Calculate shading normal
    //

    mdl_shading_normal_expr(&world_shading_normal, &state, &res_data, NULL, NULL);
    state.normal = world_shading_normal;

    //
    // Calculate tint
    //

    float3 tint;
    mdl_expr(&tint, &state, &res_data, NULL, NULL);

    //
    // Shade with phong shading
    //

    float lum = luminanceCIE(tint);
    float specular_factor = 1.f - 0.4 * lum;
    phongShade(
        tint,                                 // diffuse
        make_float3(0.01f),                   // ambient
        make_float3(specular_factor),         // specular
        make_float3(reflection_coefficient),  // reflection
        80.f,                                 // phong exponent
        world_shading_normal);
}

RT_PROGRAM void any_hit_shadow()
{
    phongShadowed();
}


RT_PROGRAM void miss()
{
    //
    // Initialize state for MDL
    //

    mi::neuraylib::Shading_state_environment state = {
        optix::normalize(ray.direction)
    };

    //
    // Calculate environment color
    //

    mdl_env_expr(&prd.result, &state, &res_data, NULL, NULL);
}


