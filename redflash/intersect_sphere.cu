#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, );
rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, center, , );
rtDeclareVariable(float, radius, , );
rtDeclareVariable(float3, aabb_min, , );
rtDeclareVariable(float3, aabb_max, , );

template<bool use_robust_method>
static __device__
void intersect_sphere(void)
{
    float3 O = ray.origin - center;
    float3 D = ray.direction;

    float b = dot(O, D);
    float c = dot(O, O) - radius * radius;
    float disc = b * b - c;
    if (disc > 0.0f) {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);

        bool do_refine = false;

        float root11 = 0.0f;

        if (use_robust_method && fabsf(root1) > 10.f * radius) {
            do_refine = true;
        }

        if (do_refine) {
            // refine root1
            float3 O1 = O + root1 * ray.direction;
            b = dot(O1, D);
            c = dot(O1, O1) - radius * radius;
            disc = b * b - c;

            if (disc > 0.0f) {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }

        bool check_second = true;
        if (rtPotentialIntersection(root1 + root11)) {
            shading_normal = geometric_normal = (O + (root1 + root11)*D) / radius;
            float t = root1 + root11;
            front_hit_point = ray.origin + t * ray.direction;
            back_hit_point = ray.origin + t * ray.direction;
            if (rtReportIntersection(0))
                check_second = false;
        }
        if (check_second) {
            float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
            if (rtPotentialIntersection(root2)) {
                shading_normal = geometric_normal = (O + root2 * D) / radius;
                float t = root2;
                front_hit_point = ray.origin + t * ray.direction;
                back_hit_point = ray.origin + t * ray.direction;
                rtReportIntersection(0);
            }
        }
    }
}


RT_PROGRAM void sphere_intersect(int primIdx)
{
    intersect_sphere<false>();
}


RT_PROGRAM void sphere_intersect_robust(int primIdx)
{
    intersect_sphere<true>();
}


RT_PROGRAM void bounds(int, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = aabb_min;
    aabb->m_max = aabb_max;
}
