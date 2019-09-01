#pragma once

#include <optixu/optixu_math_namespace.h>

struct State
{
    optix::float3 hitpoint;
    optix::float3 normal;
    optix::float3 ffnormal;
};

enum BrdfType
{
    DISNEY, GLASS
};

#ifndef RT_FUNCTION
#define RT_FUNCTION __forceinline__ __device__
#endif

struct MaterialParameter
{
    RT_FUNCTION MaterialParameter()
    {
        albedo = optix::make_float3(1.0f, 1.0f, 1.0f);
        emission = optix::make_float3(0.0f);
        metallic = 0.0;
        subsurface = 0.0f;
        specular = 0.5f;
        roughness = 0.5f;
        specularTint = 0.0f;
        anisotropic = 0.0f;
        sheen = 0.0f;
        sheenTint = 0.5f;
        clearcoat = 0.0f;
        clearcoatGloss = 1.0f;
        brdf = DISNEY;
        albedoID = RT_TEXTURE_ID_NULL;
    }

    int albedoID;
    optix::float3 albedo;
    optix::float3 emission;
    float metallic;
    float subsurface;
    float specular;
    float roughness;
    float specularTint;
    float anisotropic;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;
    BrdfType brdf;
};

enum LightType
{
    SPHERE, QUAD
};

struct LightParameter
{
    optix::float3 position;
    optix::float3 normal;
    optix::float3 emission;
    optix::float3 u;
    optix::float3 v;
    float area;
    float radius;
    LightType lightType;
};

struct LightSample
{
    optix::float3 surfacePos;
    optix::float3 normal;
    optix::float3 emission;
    float pdf;
};