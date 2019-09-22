#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <common.h>
#include "redflash.h"
#include "random.h"

using namespace optix;

static __host__ __device__ __inline__ float powerHeuristic(float a, float b)
{
    float t = a * a;
    return t / (b*b + t);
}

// Scene wide variables
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );

//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(Matrix3x3, normal_matrix, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(unsigned int, frame_number, , );
rtDeclareVariable(unsigned int, total_sample, , );
rtDeclareVariable(unsigned int, sample_per_launch, , );
rtDeclareVariable(unsigned int, rr_begin_depth, , );
rtDeclareVariable(unsigned int, max_depth, , );
rtDeclareVariable(unsigned int, use_post_tonemap, , );
rtDeclareVariable(float, tonemap_exposure, , );

rtBuffer<float4, 2> output_buffer;
rtBuffer<float4, 2> liner_buffer;
rtBuffer<float4, 2> input_albedo_buffer;
rtBuffer<float4, 2> input_normal_buffer;

__device__ inline float3 linear_to_sRGB(const float3& c)
{
    const float kInvGamma = 1.0f / 2.2f;
    return make_float3(powf(c.x, kInvGamma), powf(c.y, kInvGamma), powf(c.z, kInvGamma));
}

__device__ inline float3 tonemap_reinhard(const float3& c, float limit)
{
    float luminance = 0.3f * c.x + 0.6f * c.y + 0.1f * c.z;
    float3 col = c * 1.0f / (1.0f + luminance / limit);
    return make_float3(col.x, col.y, col.z);
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
__device__ inline float3 tonemap_acesFilm(const float3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

RT_PROGRAM void pathtrace_camera()
{
    size_t2 screen = output_buffer.size();
    float3 result = make_float3(0.0f);
    float3 albedo = make_float3(0.0f);
    float3 normal = make_float3(0.0f);
    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, total_sample);

    for (int i = 0; i < sample_per_launch; i++)
    {
        float2 subpixel_jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);
        float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
        float3 ray_origin = eye;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);

        // Initialze per-ray data
        PerRayData_pathtrace prd;
        prd.radiance = make_float3(0.0f);
        prd.attenuation = make_float3(1.0f);
        prd.done = false;
        prd.seed = seed;
        prd.depth = 0;

        // Each iteration is a segment of the ray path.  The closest hit will
        // return new segments to be traced here.
        for (;;)
        {
            Ray ray = make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
            prd.wo = -ray.direction;
            rtTrace(top_object, ray, prd);

            if (prd.done || prd.depth >= max_depth)
            {
                break;
            }

            // Russian roulette termination 
            /*if(prd.depth >= rr_begin_depth)
            {
                float pcont = fmaxf(prd.attenuation);
                if(rnd(prd.seed) >= pcont)
                    break;
                prd.attenuation /= pcont;
            }*/

            if (prd.depth == 0)
            {
                albedo += prd.albedo;
                normal += prd.normal;
            }

            // Update ray data for the next path segment
            ray_origin = prd.origin;
            ray_direction = prd.direction;

            prd.depth++;
        }

        result += prd.radiance;
    }

    //
    // Update the output buffer
    //
    float3 normal_eyespace = (length(normal) > 0.0f) ? normalize(normal_matrix * normal) : make_float3(0.0, 0.0, 1.0);

    float inv_sample_per_launch = 1.0f / static_cast<float>(sample_per_launch);
    float3 pixel_liner = result * inv_sample_per_launch;
    float3 pixel_albedo = albedo * inv_sample_per_launch;
    float3 pixel_normal = normal_eyespace;

    if (frame_number > 1)
    {
        float a = static_cast<float>(sample_per_launch) / static_cast<float>(total_sample + sample_per_launch);
        pixel_liner = lerp(make_float3(liner_buffer[launch_index]), pixel_liner, a);

        // NOTE: ノイズ用の情報は1フレーム目しか更新しない
        // pixel_albedo = lerp(make_float3(input_albedo_buffer[launch_index]), pixel_albedo, a);
        // pixel_normal = lerp(make_float3(input_normal_buffer[launch_index]), pixel_normal, a);
    }

    float3 pixel_output = use_post_tonemap ? pixel_liner : linear_to_sRGB(tonemap_acesFilm(pixel_liner * tonemap_exposure));

    // Save to buffer
    liner_buffer[launch_index] = make_float4(pixel_liner, 1.0);
    output_buffer[launch_index] = make_float4(pixel_output, 1.0);

    // NOTE: デノイズ用の情報は1フレーム目しか更新しない
    // NOTE: DOFとかモーションブラーなら毎フレーム更新した方がいいのかもしれない
    if (frame_number == 1)
    {
        input_albedo_buffer[launch_index] = make_float4(pixel_albedo, 1.0f);
        input_normal_buffer[launch_index] = make_float4(pixel_normal, 1.0f);
    }
}


//-----------------------------------------------------------------------------
//
//  closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtBuffer<MaterialParameter> sysMaterialParameters;
rtDeclareVariable(int, materialId, , );
rtDeclareVariable(int, bsdf_id, , );

rtDeclareVariable(int, sysNumberOfLights, , );
rtBuffer<LightParameter> sysLightParameters;
rtDeclareVariable(int, lightMaterialId, , );

rtBuffer< rtCallableProgramId<void(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)> > sysBRDFPdf;
rtBuffer< rtCallableProgramId<void(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)> > sysBRDFSample;
rtBuffer< rtCallableProgramId<float3(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)> > sysBRDFEval;

RT_PROGRAM void light_closest_hit()
{
    const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    const float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

    current_prd.albedo = make_float3(0.0f);
    current_prd.normal = ffnormal;

    LightParameter light = sysLightParameters[lightMaterialId];
    float cosTheta = dot(-ray.direction, light.normal);

    if ((light.lightType == QUAD && cosTheta > 0.0f) || light.lightType == SPHERE)
    {
        if (current_prd.depth == 0 || current_prd.specularBounce)
            current_prd.radiance += light.emission * current_prd.attenuation;
        else
        {
            float lightPdf = (t_hit * t_hit) / (light.area * clamp(cosTheta, 1.e-3f, 1.0f));
            current_prd.radiance += powerHeuristic(current_prd.pdf, lightPdf) * current_prd.attenuation * light.emission;
        }
    }

    current_prd.done = true;
}

RT_FUNCTION float3 UniformSampleSphere(float u1, float u2)
{
    float z = 1.f - 2.f * u1;
    float r = sqrtf(max(0.f, 1.f - z * z));
    float phi = 2.f * M_PIf * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);

    return make_float3(x, y, z);
}

RT_CALLABLE_PROGRAM void sphere_sample(LightParameter &light, PerRayData_pathtrace &prd, LightSample &sample)
{
    const float r1 = rnd(prd.seed);
    const float r2 = rnd(prd.seed);
    sample.surfacePos = light.position + UniformSampleSphere(r1, r2) * light.radius;
    sample.normal = normalize(sample.surfacePos - light.position);
    sample.emission = light.emission * sysNumberOfLights;
}

RT_FUNCTION float3 DirectLight(MaterialParameter &mat, State &state)
{
    //Pick a light to sample
    int index = optix::clamp(static_cast<int>(floorf(rnd(current_prd.seed) * sysNumberOfLights)), 0, sysNumberOfLights - 1);
    LightParameter light = sysLightParameters[index];
    LightSample lightSample;

    // float3 surfacePos = state.fhp;
    float3 surfacePos = state.hitpoint;
    float3 surfaceNormal = state.ffnormal;

    // sysLightSample[light.lightType](light, current_prd, lightSample);
    sphere_sample(light, current_prd, lightSample);

    float3 lightDir = lightSample.surfacePos - surfacePos;
    float lightDist = length(lightDir);
    float lightDistSq = lightDist * lightDist;
    lightDir /= sqrtf(lightDistSq);

    if (dot(lightDir, surfaceNormal) <= 0.0f || dot(lightDir, lightSample.normal) >= 0.0f)
        return make_float3(0.0f);

    PerRayData_pathtrace_shadow prd_shadow;
    prd_shadow.inShadow = false;
    optix::Ray shadowRay = optix::make_Ray(surfacePos, lightDir, 1, scene_epsilon, lightDist - scene_epsilon);
    rtTrace(top_object, shadowRay, prd_shadow);

    if (prd_shadow.inShadow)
        return make_float3(0.0f);

    float NdotL = dot(lightSample.normal, -lightDir);
    float lightPdf = lightDistSq / (light.area * NdotL);
    // if (lightPdf <= 0.0f)
    //    return make_float3(0.0f);

    current_prd.direction = lightDir;

    sysBRDFPdf[bsdf_id](mat, state, current_prd);
    float3 f = sysBRDFEval[bsdf_id](mat, state, current_prd);
    float3 result = powerHeuristic(lightPdf, current_prd.pdf) * current_prd.attenuation * f * lightSample.emission / max(0.001f, lightPdf);

    // FIXME: 根本の原因を解明したい
    if (isnan(result.x) || isnan(result.y) || isnan(result.z))
        return make_float3(0.0f);

    // NOTE: 負の輝度のレイを念の為チェック
    if (result.x < 0.0f || result.y < 0.0f || result.z < 0.0f)
        return make_float3(0.0f);

    return result;
}

RT_PROGRAM void closest_hit()
{
    float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

    float3 hitpoint = ray.origin + t_hit * ray.direction + ffnormal * scene_epsilon * 10.0;

    State state;
    state.hitpoint = hitpoint;
    state.normal = world_shading_normal;
    state.ffnormal = ffnormal;

    // FIXME: materialCustomProgramId みたいな名前で関数ポインタを渡して、パラメータをプロシージャルにセットしたい
    MaterialParameter mat = sysMaterialParameters[materialId];

    current_prd.radiance += mat.emission * current_prd.attenuation;
    current_prd.wo = -ray.direction;
    current_prd.albedo = mat.albedo;
    current_prd.normal = ffnormal;

    // FIXME: Sampleにもっていく
    current_prd.origin = hitpoint;

    // FIXME: bsdfId から判定
    current_prd.specularBounce = false;

    // Direct light Sampling
    if (!current_prd.specularBounce && current_prd.depth < max_depth)
    {
        current_prd.radiance += DirectLight(mat, state);
    }

    // BRDF Sampling
    sysBRDFSample[bsdf_id](mat, state, current_prd);
    sysBRDFPdf[bsdf_id](mat, state, current_prd);
    float3 f = sysBRDFEval[bsdf_id](mat, state, current_prd);

    if (current_prd.pdf > 0.0f)
    {
        current_prd.attenuation *= f / current_prd.pdf;
    }
    else
    {
        current_prd.done = true;
    }
}


//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow()
{
    current_prd_shadow.inShadow = true;
    rtTerminateRay();
}


//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_float4(bad_color, 1.0f);
}


//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void envmap_miss()
{
    float theta = atan2f(ray.direction.x, ray.direction.z);
    float phi = M_PIf * 0.5f - acosf(ray.direction.y);
    float u = (theta + M_PIf) * (0.5f * M_1_PIf);
    float v = 0.5f * (1.0f + sin(phi));
    current_prd.radiance += make_float3(tex2D(envmap, u, v)) * current_prd.attenuation;
    current_prd.albedo = make_float3(0.0f);
    current_prd.normal = -ray.direction;
    current_prd.done = true;
}