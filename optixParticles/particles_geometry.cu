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
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtBuffer<float3>    positions_buffer;
rtBuffer<float3>    velocities_buffer;
rtBuffer<float3>    colors_buffer;
rtBuffer<float>     radii_buffer;

rtDeclareVariable(float,         motion_blur, , );

// position and color of the particle that has been hit
rtDeclareVariable(float3,       particle_position,  attribute particle_position, );
rtDeclareVariable(float3,       particle_color,     attribute particle_color, );

rtDeclareVariable(optix::Ray,   ray,                rtCurrentRay, );
rtDeclareVariable(float,        cur_time,           rtCurrentTime, );


// returns the position of the given particle at the given time
__device__ __inline__ float3 get_particle_position( int primIdx, float t )
{
    float3 position = positions_buffer[ primIdx ];

    // offset the position by a fraction of the velocity, modulated by the motion blur factor
    const float3 velocity = velocities_buffer[ primIdx ];
    position += ( velocity * t * motion_blur );

    return position;
}

__device__ __inline__ void compute_particle_intersection( float3 position, int primIdx )
{
    const float radius = radii_buffer[ primIdx ];

    float3 O = ray.origin - position;
    float b = dot( O, ray.direction );
    float c = dot( O, O ) - radius * radius;
    float disc = b * b - c;

    if ( disc > 0.0f ) {
        float sdisc = sqrtf( disc );

        float root1 = ( -b - sdisc );
        bool check_second = true;

        if( rtPotentialIntersection( root1 ) ) {
            particle_position = position;
            particle_color = colors_buffer[ primIdx ];

            if( rtReportIntersection( 0 ) )
                check_second = false;
        }

        if( check_second ) {
            float root2 = ( -b + sdisc );

            if( rtPotentialIntersection( root2 ) ) {
                particle_position = position;
                particle_color = colors_buffer[ primIdx ];

                rtReportIntersection( 0 );
            }
        }
    }
}

// intersection with a particle, static case
RT_PROGRAM void particle_intersect( int primIdx )
{
    const float3 position = positions_buffer[ primIdx ];
    compute_particle_intersection( position, primIdx );
}

// intersection with a particle, motion case
RT_PROGRAM void particle_intersect_motion( int primIdx )
{
    const float3 position = get_particle_position( primIdx, cur_time );
    compute_particle_intersection( position, primIdx );
}

// bounding box of a spherical particle, static case
RT_PROGRAM void particle_bounds( int primIdx, float result[6] )
{
    const float3 position = positions_buffer[ primIdx ];
    const float  radius   = radii_buffer[ primIdx ];

    optix::Aabb *aabb = (optix::Aabb *) result;

    aabb->m_min.x = position.x - radius;
    aabb->m_min.y = position.y - radius;
    aabb->m_min.z = position.z - radius;

    aabb->m_max.x = position.x + radius;
    aabb->m_max.y = position.y + radius;
    aabb->m_max.z = position.z + radius;
}

// bounding box of a spherical particle, motion case
RT_PROGRAM void particle_bounds_motion( int primIdx, int motionIdx, float result[6] )
{
    float3 position = positions_buffer[ primIdx ]; // starting position

    if ( motionIdx > 0 ) {
        // offset the position with the velocity vector to get the ending position

        const float3 velocity = velocities_buffer[ primIdx ];
        position += ( velocity * motion_blur );
    }

    const float radius = radii_buffer[ primIdx ];

    optix::Aabb *aabb = (optix::Aabb *) result;

    aabb->m_min.x = position.x - radius;
    aabb->m_min.y = position.y - radius;
    aabb->m_min.z = position.z - radius;

    aabb->m_max.x = position.x + radius;
    aabb->m_max.y = position.y + radius;
    aabb->m_max.z = position.z + radius;
}
