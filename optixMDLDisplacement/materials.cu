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

#include <optix_world.h>
#include <mi/neuraylib/target_code_types.h>

#include "common.h"
#include "helpers.h"

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


struct PerRayData_radiance
{
    float3 result;
    float importance;
    int depth;
};


rtBuffer<BasicLight> lights;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );


rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(
    rtCallableProgramId<mi::neuraylib::Material_expr_function>, mdl_expr, , );
rtDeclareVariable(
    rtCallableProgramId<Environment_function>, mdl_env_expr, , );

__device__ const mi::neuraylib::Resource_data res_data = {
    NULL,
    NULL
};

RT_PROGRAM void mdl_material_apply()
{
    float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

    //
    // Initialize state for MDL
    //

    float world_to_object[16];
    float object_to_world[16];
    rtGetTransform(RT_WORLD_TO_OBJECT, world_to_object);
    rtGetTransform(RT_OBJECT_TO_WORLD, object_to_world);

    float3 hit_point = ray.origin + t_hit * ray.direction;

    float3 text_coords = texcoord;      // taking an address of an attribute is not supported
    float3 tangent_u = make_float3(0);
    float3 tangent_v = make_float3(0);

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
    // Calculate tint
    //

    mdl_expr(&prd.result, &state, &res_data, NULL, NULL);

    //
    // Calculate attenuation
    //

    float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);
    float3 attenuation = make_float3(0);

    unsigned int num_lights = lights.size();
    for ( int i = 0; i < num_lights; ++i ) {
        BasicLight light = lights[i];
        float3 L = optix::normalize(light.pos - hit_point);
        attenuation += max(0.f, optix::dot(ffnormal, L)) * light.color;
    }

    prd.result *= attenuation;
}

RT_PROGRAM void mdl_environment_apply()
{
    mi::neuraylib::Shading_state_environment state = {
        optix::normalize(ray.direction)
    };

    mdl_env_expr(&prd.result, &state, &res_data, NULL, NULL);
}
