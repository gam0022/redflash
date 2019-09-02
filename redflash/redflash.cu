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

// Scene wide variables
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );



//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(Matrix3x3, normal_matrix, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  total_sample, , );
rtDeclareVariable(unsigned int,  sample_per_launch, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );
rtDeclareVariable(unsigned int, max_depth, , );

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

    for(int i = 0; i < sample_per_launch; i++)
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
        for(;;)
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
        float3 normal_eyespace = (length(prd.normal) > 0.f) ? normalize(normal_matrix * prd.normal) : make_float3(0., 0., 1.);
        normal = normal_eyespace;
    }

    //
    // Update the output buffer
    //
    float inv_sample_per_launch = 1.0f / static_cast<float>(sample_per_launch);
    float3 pixel_color = result * inv_sample_per_launch;
    float3 pixel_albedo = albedo * inv_sample_per_launch;
    float3 pixel_normal = normal * inv_sample_per_launch;

    // FIXME: pixel_liner にしたい
    float3 liner_val = pixel_color;

    if (frame_number > 1)
    {
        float a = static_cast<float>(sample_per_launch) / static_cast<float>(total_sample + sample_per_launch);
        liner_val = lerp(make_float3(liner_buffer[launch_index]), pixel_color, a);
        pixel_albedo = lerp(make_float3(input_albedo_buffer[launch_index]), pixel_albedo, a);
        pixel_normal = lerp(make_float3(input_normal_buffer[launch_index]), pixel_normal, a);
    }

    // float3 output_val = linear_to_sRGB(tonemap_acesFilm(liner_val));
    float3 output_val = liner_val;
    liner_buffer[launch_index] = make_float4(liner_val, 1.0);
    output_buffer[launch_index] = make_float4(output_val, 1.0);
    input_albedo_buffer[launch_index] = make_float4(pixel_albedo, 1.0f);
    input_normal_buffer[launch_index] = make_float4(pixel_normal, 1.0f);
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
rtDeclareVariable(int, programId, , );// unused

rtDeclareVariable(int, sysNumberOfLights, , );
rtBuffer<LightParameter> sysLightParameters;
rtDeclareVariable(int, lightMaterialId, , );

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


//-----------------------------------------------------------------------------
//
//  bsdf
//
//-----------------------------------------------------------------------------

RT_CALLABLE_PROGRAM void diffuse_Pdf(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 n = state.ffnormal;
    float3 L = prd.direction;

    float pdfDiff = abs(dot(L, n))* (1.0f / M_PIf);

    prd.pdf = pdfDiff;
}

RT_CALLABLE_PROGRAM void diffuse_Sample(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
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


RT_CALLABLE_PROGRAM float3 diffuse_Eval(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
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

RT_FUNCTION float sqr(float x) { return x * x; }

RT_FUNCTION float SchlickFresnel(float u)
{
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2*m; // pow(m,5)
}

RT_FUNCTION float GTR1(float NDotH, float a)
{
    if (a >= 1.0f) return (1.0f / M_PIf);
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f)*NDotH*NDotH;
    return (a2 - 1.0f) / (M_PIf*logf(a2)*t);
}

RT_FUNCTION float GTR2(float NDotH, float a)
{
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f)*NDotH*NDotH;
    return a2 / (M_PIf * t*t);
}

RT_FUNCTION float smithG_GGX(float NDotv, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NDotv * NDotv;
    return 1.0f / (NDotv + sqrtf(a + b - a * b));
}


/*
    http://simon-kallweit.me/rendercompo2015/
*/
RT_CALLABLE_PROGRAM void Pdf(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 n = state.ffnormal;
    float3 V = prd.wo;
    float3 L = prd.direction;

    float specularAlpha = max(0.001f, mat.roughness);
    float clearcoatAlpha = lerp(0.1f, 0.001f, mat.clearcoatGloss);

    float diffuseRatio = 0.5f * (1.f - mat.metallic);
    float specularRatio = 1.f - diffuseRatio;

    float3 half = normalize(L + V);

    float cosTheta = abs(dot(half, n));
    float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
    float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

    // calculate diffuse and specular pdfs and mix ratio
    float ratio = 1.0f / (1.0f + mat.clearcoat);
    float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * abs(dot(L, half)));
    float pdfDiff = abs(dot(L, n))* (1.0f / M_PIf);

    // weight pdfs according to ratios
    prd.pdf = diffuseRatio * pdfDiff + specularRatio * pdfSpec;
}

/*
    https://learnopengl.com/PBR/IBL/Specular-IBL
*/

RT_CALLABLE_PROGRAM void Sample(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 N = state.ffnormal;
    float3 V = prd.wo;

    float3 dir;

    float probability = rnd(prd.seed);
    float diffuseRatio = 0.5f * (1.0f - mat.metallic);

    float r1 = rnd(prd.seed);
    float r2 = rnd(prd.seed);

    optix::Onb onb(N); // basis

    if (probability < diffuseRatio) // sample diffuse
    {
        cosine_sample_hemisphere(r1, r2, dir);
        onb.inverse_transform(dir);
    }
    else
    {
        float a = max(0.001f, mat.roughness);

        float phi = r1 * 2.0f * M_PIf;

        float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a*a - 1.0f) *r2));
        float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
        float sinPhi = sinf(phi);
        float cosPhi = cosf(phi);

        float3 half = make_float3(sinTheta*cosPhi, sinTheta*sinPhi, cosTheta);
        onb.inverse_transform(half);

        dir = 2.0f*dot(V, half)*half - V; //reflection vector

    }
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

    float3 H = normalize(L + V);
    float NDotH = dot(N, H);
    float LDotH = dot(L, H);

    float3 Cdlin = mat.albedo;
    float Cdlum = 0.3f*Cdlin.x + 0.6f*Cdlin.y + 0.1f*Cdlin.z; // luminance approx.

    float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
    float3 Cspec0 = lerp(mat.specular*0.08f*lerp(make_float3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);
    float3 Csheen = lerp(make_float3(1.0f), Ctint, mat.sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
    float Fd90 = 0.5f + 2.0f * LDotH*LDotH * mat.roughness;
    float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotrokPic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LDotH * LDotH*mat.roughness;
    float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.0f / (NDotL + NDotV) - 0.5f) + 0.5f);

    // specular
    //float aspect = sqrt(1-mat.anisotrokPic*.9);
    //float ax = Max(.001f, sqr(mat.roughness)/aspect);
    //float ay = Max(.001f, sqr(mat.roughness)*aspect);
    //float Ds = GTR2_aniso(NDotH, Dot(H, X), Dot(H, Y), ax, ay);

    float a = max(0.001f, mat.roughness);
    float Ds = GTR2(NDotH, a);
    float FH = SchlickFresnel(LDotH);
    float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
    float roughg = sqr(mat.roughness*0.5f + 0.5f);
    float Gs = smithG_GGX(NDotL, roughg) * smithG_GGX(NDotV, roughg);

    // sheen
    float3 Fsheen = FH * mat.sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, mat.clearcoatGloss));
    float Fr = lerp(0.04f, 1.0f, FH);
    float Gr = smithG_GGX(NDotL, 0.25f) * smithG_GGX(NDotV, 0.25f);

    float3 out = ((1.0f / M_PIf) * lerp(Fd, ss, mat.subsurface)*Cdlin + Fsheen)
        * (1.0f - mat.metallic)
        + Gs * Fs*Ds + 0.25f*mat.clearcoat*Gr*Fr*Dr;

    return out * clamp(dot(N, L), 0.0f, 1.0f);
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

    // sysBRDFPdf[programId](mat, state, current_prd);
    Pdf(mat, state, current_prd);
    // float3 f = sysBRDFEval[programId](mat, state, current_prd);
    float3 f = Eval(mat, state, current_prd);
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
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

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
    Sample(mat, state, current_prd);
    Pdf(mat, state, current_prd);
    float3 f = Eval(mat, state, current_prd);

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