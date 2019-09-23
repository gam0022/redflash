#include <optixu/optixu_math_namespace.h>
#include "redflash.h"
#include "random.h"

using namespace optix;

RT_CALLABLE_PROGRAM void Pdf(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 n = state.ffnormal;
    float3 L = prd.direction;

    float pdfDiff = abs(dot(L, n))* (1.0f / M_PIf);

    prd.pdf = pdfDiff;
}

RT_CALLABLE_PROGRAM void Sample(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 N = state.ffnormal;

    float3 dir;

    float r1 = rnd(prd.seed);
    float r2 = rnd(prd.seed);

    optix::Onb onb(N);

    cosine_sample_hemisphere(r1, r2, dir);
    onb.inverse_transform(dir);

    prd.direction = dir;
}


RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 N = state.ffnormal;
    float3 V = prd.wo;
    float3 L = prd.direction;

    float NDotL = dot(N, L);
    float NDotV = dot(N, V);
    if (NDotL <= 0.0f || NDotV <= 0.0f) return make_float3(0.0f);

    float3 out = (1.0f / M_PIf) * mat.albedo;

    return out * clamp(dot(N, L), 0.0f, 1.0f);
}