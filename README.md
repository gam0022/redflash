# Redflash Renderer

![pr33_v6_t3000_s1030_1920x1080](https://user-images.githubusercontent.com/759115/64941257-1549c080-d8a1-11e9-9cc6-a145bdaed7d8.png)

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

## Development Environment

- Operation confirmed
  - Windows 10 + NVIDIA RTX 2070
  - Windows Server 2016 Base + NVIDIA Tesla V100 GPUs
- Dependences
  - CUDA 10.1
  - OptiX 6.0.0

## Gallery

### レイトレ合宿7 提出バージョン / RaytracingCamp7 submission version

Won 4th prize at [レイトレ合宿7](https://sites.google.com/site/raytracingcamp7/).

![pr33_v6_t3000_s1030_1920x1080](https://user-images.githubusercontent.com/759115/64941257-1549c080-d8a1-11e9-9cc6-a145bdaed7d8.png)

#### Camera Angle Variation 1
![cut_far_v1](https://user-images.githubusercontent.com/759115/64941285-272b6380-d8a1-11e9-943c-7bf38f5e9538.png)

#### Camera Angle Variation 2

![cut_far_v2](https://user-images.githubusercontent.com/759115/64941286-2a265400-d8a1-11e9-84a4-245cfe70fed1.png)

#### Camera Angle Variation 3

![cut_far_v3](https://user-images.githubusercontent.com/759115/64941288-2b578100-d8a1-11e9-9494-8395a5310c6f.png)

## Links

- [redflash renderer / Raytracing Camp 7 - Speaker Deck](https://speakerdeck.com/gam0022/raytracing-camp-7)
