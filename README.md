# Redflash Renderer

Redflash is a physics-based renderer based on path tracing implemented on NVIDIA® OptiX 6.0, which can consistently draw scenes with mixed polygons and raymarching.

Redflash は NVIDIA® OptiX 6.0 上で実装したパストレーシングによる物理ベースレンダラーで、ポリゴンとレイマーチングが混在したシーンを一貫して描画できます。

This is implemented based on optixPathTracer of NVIDIA official OptiX-Samples.

これは、NVIDIA 公式の OptiX-Samples の optixPathTracer をベースにして実装されています。

## Features

- Unidirectional Path Tracing
  - Next Event Estimation (Direct Light Sampling)
  - Multiple Importance Sampling
- Disney BRDF
- Primitives
  - Sphere
  - Mesh
  - Distance Function ( **Raymarching** )
- ACES Filmic Tone Mapping
- Deep Learning Denoising
