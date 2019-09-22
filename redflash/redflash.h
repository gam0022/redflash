#pragma once

#include <optixu/optixu_math_namespace.h>

using namespace optix;

#ifndef RT_FUNCTION
#define RT_FUNCTION __forceinline__ __device__
#endif

struct State
{
    float3 hitpoint;
    float3 normal;
    float3 ffnormal;
};

enum BSDFType
{
    DIFFUSE,
    DISNEY
};

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
        bsdf = DISNEY;
        albedoID = RT_TEXTURE_ID_NULL;
    }

    int albedoID;
    float3 albedo;
    float3 emission;
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
    BSDFType bsdf;
};

enum LightType
{
    SPHERE, QUAD
};

struct LightParameter
{
    float3 position;
    float3 normal;
    float3 emission;
    float3 u;
    float3 v;
    float area;
    float radius;
    LightType lightType;
};

struct LightSample
{
    float3 surfacePos;
    float3 normal;
    float3 emission;
    float pdf;
};

struct PerRayData_pathtrace
{
    float3 radiance;
    float3 attenuation;

    float3 albedo;
    float3 normal;

    float3 origin;
    float3 direction;

    float pdf;
    float3 wo;

    unsigned int seed;
    int depth;
    bool done;
    bool specularBounce;
};

struct PerRayData_pathtrace_shadow
{
    bool inShadow;
};