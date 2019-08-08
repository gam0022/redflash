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
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <mi/neuraylib/target_code_types.h>

using namespace optix;

// For comparison, you can disable the use of MDL for the displacement and use callable programs
// with implementations defined in this file instead.
// As MDL requires the full state structure, there is some performance overhead.
#define USE_MDL_DISPLACEMENT

// Displacement function for MDL case.
rtDeclareVariable(
    rtCallableProgramId<mi::neuraylib::Material_expr_function>, mdl_displace_expr,,);

// Displacement function for non-MDL case.
rtDeclareVariable(
    rtCallableProgramId<float(float, float, float3 const &)>, non_mdl_displace,,);


#define DISPLACE_MAX_D    0.003f   // maximum displacement
#define DISPLACE_N        4.0f     // tesselation rate, each triangle is tesselated with the 2D triangle-grid DISPLACE_N x DISPLACE_N / 2
#define USE_BRUTE 0


// Use constants for some MDL state members to reduce memory writes. We know the materials don't use them.
__device__ const float identity[16] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f };
__device__ const float3 fixed_tangent_u = { 1.0f, 0.0f, 0.0f };
__device__ const float3 fixed_tangent_v = { 0.0f, 1.0f, 0.0f };
__device__ const mi::neuraylib::Resource_data res_data = {
    NULL,
    NULL
};


/*
  calculates the relative displacement across the normal
*/
__device__ float displace_fish(float x, float y, float3 const &P)
{
    return DISPLACE_MAX_D * sinf(x * M_PIf) * sinf(y * M_PIf);
}

__device__ float displace_fish_rings(float x, float y, float3 const &P)
{
    return DISPLACE_MAX_D * fmaxf(0, sinf(P.x * 80 * M_PIf));
}


#ifdef USE_MDL_DISPLACEMENT

/*
  calculates the absolute displacement by evaluating an MDL expression
*/
static __device__ __forceinline__ float3 mdl_displace(
    float x, float y, float3 const &P, float3 const &N,
    mi::neuraylib::Shading_state_material *state)
{
    float3 text_coords = make_float3(x, y, 0);

    state->normal = N;
    state->geom_normal = N;
    state->position = P;
    state->text_coords = &text_coords;

    float3 res;
    mdl_displace_expr(&res, state, &res_data, NULL, NULL);
    return res;
}

#endif

#define FACE_U(face)  face.x
#define FACE_V(face)  face.y
#define FACE_hitT(face)  face.z
#define FACE_Idx(face)  face.w

/*
  basic ray and tri data structures used here
*/
struct D_RAY
{
  float3 org, dir;
};

struct D_TRI
{
  float3 A, B, C;
  float3 Na, Nb, Nc;
  float2 texA, texB, texC;
};

static __device__ float3 bilerp(float3 A, float3 B, float3 C, float2 coord)
{
  float _r = 1.0f / DISPLACE_N;
  float u = coord.x * _r;
  float v = coord.y * _r;
  return (1.0f - u - v) * A + u * B + v * C;
}

static __device__ float2 bilerp(float2 A, float2 B, float2 C, float2 coord)
{
  float _r = 1.0f / DISPLACE_N;
  float u = coord.x * _r;
  float v = coord.y * _r;
  float x = (1.0f - u - v) * A.x + u * B.x + v * C.x;
  float y = (1.0f - u - v) * A.y + u * B.y + v * C.y;
  return make_float2(x, y);
}

static __device__ int on_right_side(D_RAY & ray, D_RAY & vec)
{
  // ray0 on the left side: -1, ray0 on right side: 1
  return dot(cross(vec.dir, ray.org - vec.org), ray.dir) >= 0.0f ? 1 : -1;
}

static __device__ bool between(D_RAY & ray, D_RAY & vec1, D_RAY & vec2)
{
  return on_right_side(ray, vec1) + on_right_side(ray, vec2) == 0;
}

static __device__ bool within_side(int2 cell, int side_idx, D_RAY & ray, D_TRI & tri)
{
  float2 p1 = make_float2(cell.x, cell.y), p2 = make_float2(cell.x, cell.y);

  if(side_idx == 0) {
    p2.x += 1.0f;
  } else if(side_idx == 1) {
    p1.x += 1.0f;
    p2.x += 1.0f;
    if(cell.x + cell.y < int(DISPLACE_N) - 1)
      p2.y += 1.0f;
  } else if(side_idx == 2) {
    p1.x += 1.0f;
    if(cell.x + cell.y < int(DISPLACE_N) - 1)
      p1.y += 1.0f;
    p2.y += 1.0f;
  } else if(side_idx == 3) {
    p1.y += 1.0f;
  }

  D_RAY vec1, vec2;
  vec1.org = bilerp(tri.A, tri.B, tri.C, p1), vec1.dir = normalize(bilerp(tri.Na, tri.Nb, tri.Nc, p1));
  vec2.org = bilerp(tri.A, tri.B, tri.C, p2), vec2.dir = normalize(bilerp(tri.Na, tri.Nb, tri.Nc, p2));

  return between(ray, vec1, vec2);
}

static __device__ float2 edge_point_to_coord(float point, float edgeIdx)
{
  float2 coord = make_float2(point, point);

  if(edgeIdx == 1.0f)      // AB edge
    coord.y = 0.0f;
  else if(edgeIdx == 2.0f)  // BC edge
    coord.x = DISPLACE_N - coord.x;
  else if(edgeIdx == 3.0f)  // AC edge
    coord.x = 0.0f;

  return coord;
}

static __device__ float patch_intersection_cell(float faceIdx, D_RAY & ray, D_TRI & tri)
{
  int a = 0, b = 4;
  while(b - a > 1) { // size = 1, 2, 4, 8
    int m = (a + b) / 2;

    float2 p1 = edge_point_to_coord(float(a), faceIdx);
    float2 p2 = edge_point_to_coord(float(m), faceIdx);

    D_RAY vec1, vec2;
    vec1.org = bilerp(tri.A, tri.B, tri.C, p1), vec1.dir = bilerp(tri.Na, tri.Nb, tri.Nc, p1);
    vec2.org = bilerp(tri.A, tri.B, tri.C, p2), vec2.dir = bilerp(tri.Na, tri.Nb, tri.Nc, p2);

    if(between(ray, vec1, vec2)) b = m;
    else a = m;
  }

  return float(a);
}

static __device__ int intersect_tri(float Tmin, float & Tmax, float & u, float & v, float3 & N,
                                    D_RAY & ray, float3 A, float3 B, float3 C)
{
  int res = 0;

  float3 ae = ray.org - A, ab = B - A, ac = C - A;
  float3 aedir = cross(ae, ray.dir);

  float3 n = cross(ab, ac);
  float i_ndir = 1.0f / dot(n, ray.dir);
  float t = -dot(n, ae) * i_ndir;
  u = -dot(ac, aedir) * i_ndir;
  v =  dot(ab, aedir) * i_ndir;

  if(Tmin < t && t < Tmax && 0.0f <= u && 0.0f <= v && u + v <= 1.0f)
    res = 1, Tmax = t, N = n;

  return res;
}

static __device__ int intersect_tripair(float Tmin, float & Tmax, float3 & N, D_RAY & ray,
                                        float3 A, float3 B, float3 C, float3 D)
{
  int res = 0;

  float3 ae = ray.org - A, ab = B - A, ac = C - A, ad = D - A;
  float3 aedir = cross(ae, ray.dir);
  float dot_abdir = dot(ab, aedir);

  float3 n = cross(ab, ac);
  float i_ndir = 1.0f / dot(n, ray.dir);
  float t = -dot(n, ae) * i_ndir;

  if(Tmin < t && t < Tmax) {
    float u = -dot(ac, aedir) * i_ndir;
    float v =  dot_abdir * i_ndir;
    if(0.0f <= u && 0.0f <= v && u + v <= 1.0f)
      res = 1, Tmax = t, N = n;
  }

  n = cross(ab, ad);
  i_ndir = 1.0f / dot(n, ray.dir);
  t = -dot(n, ae) * i_ndir;

  if(Tmin < t && t < Tmax) {
    float u = -dot(ad, aedir) * i_ndir;
    float v =  dot_abdir * i_ndir;
    if(0.0f <= u && 0.0f <= v && u + v <= 1.0f)
      res = 1, Tmax = t, N = n;
  }

  return res;
}

static __device__ int intersect_patch(float Tmin, float & Tmax,
                                      D_RAY & ray, float3 A, float3 B, float3 C, float3 D)
{
  float3 N;
  return intersect_tripair(Tmin, Tmax, N, ray, A, B, C, D) + intersect_tripair(Tmin, Tmax, N, ray, C, D, A, B);
}

static __device__ int intersect_displaced_surfel(int2 cell, float Tmin, float & Tmax, float3 & N,
                                                 D_RAY & ray, D_TRI & tri)
{
  float2 a = make_float2(cell.x + 1, cell.y);
  float2 b = make_float2(cell.x, cell.y + 1);
  float2 c = make_float2(cell.x, cell.y);
  float2 d = make_float2(cell.x + 1, (cell.x + cell.y < int(DISPLACE_N) - 1) ? cell.y + 1 : cell.y);

  float3 Pa = bilerp(tri.A, tri.B, tri.C, a);
  float3 Pb = bilerp(tri.A, tri.B, tri.C, b);
  float3 Pc = bilerp(tri.A, tri.B, tri.C, c);
  float3 Pd = bilerp(tri.A, tri.B, tri.C, d);

  float3 NPa = normalize(bilerp(tri.Na, tri.Nb, tri.Nc, a));
  float3 NPb = normalize(bilerp(tri.Na, tri.Nb, tri.Nc, b));
  float3 NPc = normalize(bilerp(tri.Na, tri.Nb, tri.Nc, c));
  float3 NPd = normalize(bilerp(tri.Na, tri.Nb, tri.Nc, d));

  float2 disp_a = bilerp(tri.texA, tri.texB, tri.texC, a);
  float2 disp_b = bilerp(tri.texA, tri.texB, tri.texC, b);
  float2 disp_c = bilerp(tri.texA, tri.texB, tri.texC, c);
  float2 disp_d = bilerp(tri.texA, tri.texB, tri.texC, d);

#ifdef USE_MDL_DISPLACEMENT
  // Use same state object for all 4 calls to mdl_displace
  mi::neuraylib::Shading_state_material state;
  state.animation_time = 0;
  state.tangent_u = &fixed_tangent_u;
  state.tangent_v = &fixed_tangent_v;
  state.text_results = NULL;
  state.ro_data_segment = NULL;
  state.world_to_object = (float4 *) &identity;
  state.object_to_world = (float4 *) &identity;
  state.object_id = 0;

  float3 _Pa = Pa + mdl_displace(disp_a.x, disp_a.y, Pa, NPa, &state);
  float3 _Pb = Pb + mdl_displace(disp_b.x, disp_b.y, Pb, NPb, &state);
  float3 _Pc = Pc + mdl_displace(disp_c.x, disp_c.y, Pc, NPc, &state);
  float3 _Pd = Pd + mdl_displace(disp_d.x, disp_d.y, Pd, NPd, &state);
#else
  float3 _Pa = Pa + NPa * non_mdl_displace(disp_a.x, disp_a.y, Pa);
  float3 _Pb = Pb + NPb * non_mdl_displace(disp_b.x, disp_b.y, Pb);
  float3 _Pc = Pc + NPc * non_mdl_displace(disp_c.x, disp_c.y, Pc);
  float3 _Pd = Pd + NPd * non_mdl_displace(disp_d.x, disp_d.y, Pd);
#endif

  return intersect_tripair(Tmin, Tmax, N, ray, _Pa, _Pb, _Pc, _Pd);
}

#if USE_BRUTE
static __device__ bool traverse_displaced_brute(float & hitT, float3 & N, D_RAY & ray, D_TRI & tri)
{
  for(int i = 0; i < int(DISPLACE_N); i++)
    for(int j = 0; j < int(DISPLACE_N); j++)
      if(0 <= i && 0 <= j && i + j <= int(DISPLACE_N) - 1)
      {
        int2 cell = make_int2(i,j);
        if(intersect_displaced_surfel(cell, 0.0f, hitT, N, ray, tri) > 0)
          return true;
      }

  return false;
}
#endif


static __device__ bool traverse_displaced(int2 cell, int2 cell_exit, int from_side, float & hitT, float3 & N, D_RAY & ray, D_TRI & tri)
{
  // (i,j) where we start in a grid, from_side - initial cell side
  // from_side = 0, 1, 2, 3 (the ids of displacement cell edges)
  int numSteps = int(DISPLACE_N) * 10;
  while(0 <= cell.x && 0 <= cell.y && cell.x + cell.y <= int(DISPLACE_N) - 1 && numSteps-- > 0)
  {
    if(intersect_displaced_surfel(cell, 0.0f, hitT, N, ray, tri) > 0)
      return true;

    int goto_side_idx = from_side;
    for(int s = 0; s <= 3; s++)
      if(s != from_side && within_side(cell, s, ray, tri)) {
        goto_side_idx = s;
        break;
      }

    // go to the other cells
    if(goto_side_idx == 0) cell.y -= 1, from_side = 2;
    else if(goto_side_idx == 1) cell.x += 1, from_side = 3;
    else if(goto_side_idx == 2) cell.y += 1, from_side = 0;
    else if(goto_side_idx == 3) cell.x -= 1, from_side = 1;
  }

  return false;
}

/*
  intersects the ray with the bounding volume of the base triangle and takes enter/exit faces of this volume
*/
static __device__ int enter_exit_faces(float4 * Faces, D_RAY & ray, D_TRI & tri)
{
  // bounding volume (for the beta implementation the volume is infi)
  float3 AT = tri.A + 1.5f * DISPLACE_MAX_D * tri.Na;  // 1.5f = conservative multiplier
  float3 BT = tri.B + 1.5f * DISPLACE_MAX_D * tri.Nb;
  float3 CT = tri.C + 1.5f * DISPLACE_MAX_D * tri.Nc;
  float3 AB = tri.A - 1.5f * DISPLACE_MAX_D * tri.Na;
  float3 BB = tri.B - 1.5f * DISPLACE_MAX_D * tri.Nb;
  float3 CB = tri.C - 1.5f * DISPLACE_MAX_D * tri.Nc;

  int numFaces = 0;

  // ray as infinite line
  float T = 1e+6f;
  if(intersect_patch(-1e+6f, T, ray, AB, BT, AT, BB) > 0 && numFaces < 2) {
    FACE_hitT(Faces[numFaces]) = T;
    FACE_Idx(Faces[numFaces]) = 1.0f;
    numFaces++;
  }

  T = 1e+6f;
  if(intersect_patch(-1e+6f, T, ray, BB, CT, BT, CB) > 0 && numFaces < 2) {
    FACE_hitT(Faces[numFaces]) = T;
    FACE_Idx(Faces[numFaces]) = 2.0f;
    numFaces++;
  }

  T = 1e+6f;
  if(intersect_patch(-1e+6f, T, ray, CB, AT, CT, AB) > 0 && numFaces < 2) {
    FACE_hitT(Faces[numFaces]) = T;
    FACE_Idx(Faces[numFaces]) = 3.0f;
    numFaces++;
  }

  T = 1e+6f;
  float u45, v45; float3 N45;
  if(intersect_tri(-1e+6f, T, u45, v45, N45, ray, AB, BB, CB) > 0 && numFaces < 2) {
    FACE_U(Faces[numFaces]) = u45;
    FACE_V(Faces[numFaces]) = v45;
    FACE_hitT(Faces[numFaces]) = T;
    FACE_Idx(Faces[numFaces]) = 4.0f;
    numFaces++;
  }

  T = 1e+6f;
  if(intersect_tri(-1e+6f, T, u45, v45, N45, ray, AT, BT, CT) > 0 && numFaces < 2) {
    FACE_U(Faces[numFaces]) = u45;
    FACE_V(Faces[numFaces]) = v45;
    FACE_hitT(Faces[numFaces]) = T;
    FACE_Idx(Faces[numFaces]) = 5.0f;
    numFaces++;
  }

  if(numFaces != 2)
    return numFaces;

  for(int f = 0; f < 2; f++)
  {
    if(FACE_Idx(Faces[f]) <= 3.0f)  // the ray hit the side wall of the volume
    {
      float patch_cell = patch_intersection_cell(FACE_Idx(Faces[f]), ray, tri);

      if(FACE_Idx(Faces[f]) == 1.0f) {
        FACE_U(Faces[f]) = patch_cell;
        FACE_V(Faces[f]) = 0.0f;
      } else if(FACE_Idx(Faces[f]) == 2.0f) {
        FACE_U(Faces[f]) = DISPLACE_N - patch_cell - 1.0f;
        FACE_V(Faces[f]) = patch_cell;
      } else if(FACE_Idx(Faces[f]) == 3.0f) {
        FACE_U(Faces[f]) = 0.0f;
        FACE_V(Faces[f]) = patch_cell;
      }
    }
    else // the ray hits the top or bottom sides of the volume
    {
      FACE_U(Faces[f]) = float(clamp(int(Faces[f].x * DISPLACE_N), 0, int(DISPLACE_N) - 1));
      FACE_V(Faces[f]) = float(clamp(int(Faces[f].y * DISPLACE_N), 0, int(DISPLACE_N) - 1));
    }
  }

  return numFaces;
}

static __device__ bool intersect_displaced_tri(float & hitT, float3 & N, D_RAY & ray, D_TRI & tri)
{
#if USE_BRUTE

  //brute-force intersection for testing the correctness
  return traverse_displaced_brute(hitT, N, ray, tri);
#endif

  // fix bad normals
  float3 n = normalize(cross(tri.B - tri.A, tri.C - tri.A));
  if(dot(n, tri.Na) < 0.0f) tri.Na = -tri.Na;
  if(dot(n, tri.Nb) < 0.0f) tri.Nb = -tri.Nb;
  if(dot(n, tri.Nc) < 0.0f) tri.Nc = -tri.Nc;

  if(dot(n, tri.Na) < 0.7f) tri.Na = n;
  if(dot(n, tri.Nb) < 0.7f) tri.Nb = n;
  if(dot(n, tri.Nc) < 0.7f) tri.Nc = n;

  float4 Faces[2];
  int numFaces = enter_exit_faces(Faces, ray, tri);

  // the ray should intersect 2 faces of the bounding volume of the base triangle
  if(numFaces != 2)
    return false;

  float4 enter, exit;
  if(FACE_hitT(Faces[0]) < FACE_hitT(Faces[1]))
    enter = Faces[0], exit = Faces[1];
  else enter = Faces[1], exit = Faces[0];

  int2 cell_enter = make_int2(FACE_U(enter), FACE_V(enter));
  int2 cell_exit = make_int2(FACE_U(exit), FACE_V(exit));

  // the ray enters one of the volume sides
  if(FACE_Idx(enter) <= 3.0f)
  {
    int from_side;
    if(FACE_Idx(enter) == 1.0f)
      from_side = 0;
    else if(FACE_Idx(enter) == 2.0f)
      from_side = 2;
    else from_side = 3;

    bool inter = false;
    for(int i = 0; i < int(DISPLACE_N); i++) {
      int2 icell_enter;
      if(FACE_Idx(enter) == 1.0f) {
        icell_enter.x = i;
        icell_enter.y = 0;
      } else if(FACE_Idx(enter) == 2.0f) {
        icell_enter.x = int(DISPLACE_N) - i - 1;
        icell_enter.y = i;
      } else if(FACE_Idx(enter) == 3.0f) {
        icell_enter.x = 0;
        icell_enter.y = i;
      }

      if(traverse_displaced(icell_enter, cell_exit, from_side, hitT, N, ray, tri))
        inter = true;
    }

    return inter;
  }
  else // the ray enters the top or bottom sides of the volume
  {
    // we enter top/bottom side of the volume at some surfel, check it for intersection
    if(intersect_displaced_surfel(cell_enter, 0.0f, hitT, N, ray, tri) > 0)
      return true;

    float2 vec = make_float2(cell_exit.x - cell_enter.x, cell_exit.y - cell_enter.y);

    if(vec.x == 0.0f && vec.y == 0.0f)
      return false;

    // check the neigbour surfels for intersection
    bool inter = false;
    for(int s = 0; s <= 3; s++)
    {
      if(vec.x > 0.0f && s == 3) continue;
      if(vec.x < 0.0f && s == 1) continue;
      if(vec.y > 0.0f && s == 0) continue;
      if(vec.y < 0.0f && s == 2) continue;

      if(within_side(cell_enter, s, ray, tri))
      {
        int2 cell_next = cell_enter;
        int from_side = s;

        // go to the other cells
        if(s == 0) cell_next.y -= 1, from_side = 2;
        else if(s == 1) cell_next.x += 1, from_side = 3;
        else if(s == 2) cell_next.y += 1, from_side = 0;
        else if(s == 3) cell_next.x -= 1, from_side = 1;

        if(traverse_displaced(cell_next, cell_exit, from_side, hitT, N, ray, tri))
          inter = true;
      }
    }

    return inter;
  }
}

// This is to be plugged into an RTgeometry object to represent
// a triangle mesh with a vertex buffer of triangle soup (triangle list)
// with an interleaved position, normal, texturecoordinate layout.

rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3>   index_buffer;
rtBuffer<int>    material_buffer; // per-face material index

rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void mesh_intersect( int primIdx )
{
  int3 v_idx = index_buffer[ primIdx ];
  float3 p0 = vertex_buffer[ v_idx.x ];
  float3 p1 = vertex_buffer[ v_idx.y ];
  float3 p2 = vertex_buffer[ v_idx.z ];

  float3 n0 = normalize(normal_buffer[ v_idx.x ]);
  float3 n1 = normalize(normal_buffer[ v_idx.y ]);
  float3 n2 = normalize(normal_buffer[ v_idx.z ]);

  float2 t0 = make_float2(0.0f, 0.0f);
  float2 t1 = make_float2(1.0f, 0.0f);
  float2 t2 = make_float2(0.0f, 1.0f);

  D_RAY tmp_ray;
  tmp_ray.org = ray.origin;
  tmp_ray.dir = ray.direction;

  D_TRI tmp_tri;
  tmp_tri.A = p0,     tmp_tri.B = p1,     tmp_tri.C = p2;
  tmp_tri.Na = n0,    tmp_tri.Nb = n1,    tmp_tri.Nc = n2;
  tmp_tri.texA = t0,  tmp_tri.texB = t1,  tmp_tri.texC = t2;

  float hitT = RT_DEFAULT_MAX;
  float3 N;

  if(intersect_displaced_tri(hitT, N, tmp_ray, tmp_tri))
  {
    if(hitT < ray.tmax && hitT > ray.tmin) {
      if( rtPotentialIntersection( hitT ) ) {
        shading_normal = N;
        geometric_normal = N;
        texcoord = make_float3( 0,0,0 );
        rtReportIntersection(material_buffer[primIdx]);
      }
    }
  }
}

RT_PROGRAM void mesh_bounds(int primIdx, float result[6])
{
  int3 v_idx = index_buffer[ primIdx ];
  float3 v0 = vertex_buffer[ v_idx.x ];
  float3 v1 = vertex_buffer[ v_idx.y ];
  float3 v2 = vertex_buffer[ v_idx.z ];

  float3 n0 = normalize(normal_buffer[ v_idx.x ]);
  float3 n1 = normalize(normal_buffer[ v_idx.y ]);
  float3 n2 = normalize(normal_buffer[ v_idx.z ]);

  // conservative bounds
  float3 _vb0 = v0 - 1.5f * DISPLACE_MAX_D * n0;
  float3 _vb1 = v1 - 1.5f * DISPLACE_MAX_D * n1;
  float3 _vb2 = v2 - 1.5f * DISPLACE_MAX_D * n2;
  float3 _vt0 = v0 + 1.5f * DISPLACE_MAX_D * n0;
  float3 _vt1 = v1 + 1.5f * DISPLACE_MAX_D * n1;
  float3 _vt2 = v2 + 1.5f * DISPLACE_MAX_D * n2;

  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = fminf(fminf(fminf(_vb0, _vb1), fminf(_vb2, _vt0)), fminf(_vt1, _vt2));
  aabb->m_max = fmaxf(fmaxf(fmaxf(_vb0, _vb1), fmaxf(_vb2, _vt0)), fmaxf(_vt1, _vt2));
}
